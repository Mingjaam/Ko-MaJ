"""Watermark schemes: generation-side bias (per request, vLLM adapter) and detection.

HanMark : green list over Hangul syllables, context = previous syllable (h=1).
KGW     : green list over vocab ids, context = previous token id (LeftHash h=1).
Unigram : fixed green list over vocab ids.
"""
import math
from collections import OrderedDict

import numpy as np
import torch

from .core import (NSYL, SYL_ALL, SylStream, decode_partial, syllables, token_bytes_table,
                    u01_int, u01_np, u01_pairs)


def zscore(g, t, gamma):
    if t == 0:
        return 0.0
    return (g - gamma * t) / math.sqrt(t * gamma * (1 - gamma))


# ---------------------------------------------------------------- detection
def detect_hanmark(text, key, gamma, dedup=True, weights=None):
    s = syllables(text)
    pairs = list(zip(s[:-1], s[1:]))
    if dedup:
        pairs = list(dict.fromkeys(pairs))
    if not pairs:
        return 0.0, 0
    A, B = zip(*pairs)
    x = (u01_pairs(key, A, B) < gamma).astype(np.float64)
    if weights is None:
        return zscore(x.sum(), len(pairs), gamma), len(pairs)
    # weighted z (EWD-style) with public per-context weights
    w = np.asarray(weights, dtype=np.float64)[list(A)]
    if w.sum() == 0:
        return 0.0, len(pairs)
    return float((w @ x - gamma * w.sum()) / math.sqrt(gamma * (1 - gamma) * (w @ w))), len(pairs)


def detect_kgw(ids, key, gamma, dedup=True):
    pairs = list(zip(ids[:-1], ids[1:]))
    if dedup:
        pairs = list(dict.fromkeys(pairs))
    if not pairs:
        return 0.0, 0
    A, B = zip(*pairs)
    g = (u01_pairs(key, np.asarray(A) + 1, B) < gamma).sum()  # ctx = prev+1, as in KGWReq
    return zscore(g, len(pairs), gamma), len(pairs)


def detect_unigram(ids, key, gamma):
    u = list(dict.fromkeys(ids))
    g = (u01_pairs(key, np.zeros(len(u), dtype=np.int64), u) < gamma).sum() if u else 0
    return zscore(g, len(u), gamma), len(u)


# ---------------------------------------------------------------- generation
class HanMarkTables:
    """Per-tokenizer precomputation for vectorised HanMark bias (empty UTF-8 buffer case)."""

    def __init__(self, tok, vocab_size, key, gamma):
        self.key, self.gamma = key, gamma
        self.tbytes = token_bytes_table(tok, vocab_size)
        V = vocab_size
        first = np.full(V, -1, dtype=np.int64)
        nsyl = np.zeros(V, dtype=np.int64)
        inner = np.zeros(V, dtype=np.int64)  # green pairs inside the token
        for t, b in enumerate(self.tbytes):
            if not b:
                continue
            text, _ = decode_partial(b)
            s = syllables(text)
            if not s:
                continue
            first[t] = s[0]
            nsyl[t] = len(s)
            inner[t] = sum(u01_int(key, x, y) < gamma for x, y in zip(s[:-1], s[1:]))
        self.first_np, self.nsyl_np, self.inner_np = first, nsyl, inner
        self.dev = None

    def to(self, device):
        if self.dev != device:
            self.first = torch.from_numpy(np.where(self.first_np < 0, 0, self.first_np)).to(device)
            self.has = torch.from_numpy(self.first_np >= 0).to(device)
            self.nsyl = torch.from_numpy(self.nsyl_np).to(device).float()
            self.inner = torch.from_numpy(self.inner_np).to(device).float()
            self.rows = OrderedDict()
            self.dev = device
        return self

    def green_row(self, ctx):
        r = self.rows.get(ctx)
        if r is None:
            r = torch.from_numpy(u01_np(self.key, ctx, SYL_ALL) < self.gamma).to(self.dev).float()
            self.rows[ctx] = r
        return r


class HanMarkReq:
    """Per-request state: tracks UTF-8 buffer + last syllable, biases logits."""

    def __init__(self, tables: HanMarkTables, delta, ent_thresh=None):
        self.T, self.delta, self.ent_thresh = tables, delta, ent_thresh
        self.pos, self.buf, self.ctx, self.stream = 0, b"", None, SylStream()

    def _advance(self, out_ids):
        if len(out_ids) < self.pos:  # preempted / recomputed: rebuild
            self.pos, self.buf, self.ctx, self.stream = 0, b"", None, SylStream()
        while self.pos < len(out_ids):
            t = out_ids[self.pos]
            b = self.T.tbytes[t] if t < len(self.T.tbytes) else b""
            text, self.buf = decode_partial(self.buf + b)
            for _, v in self.stream.feed(text):  # same canonicalisation C(x) as the detector
                self.ctx = v
            self.pos += 1

    def __call__(self, out_ids, logits):
        self._advance(out_ids)
        T = self.T.to(logits.device)
        V = T.first.shape[0]
        if self.ent_thresh is not None:
            p = torch.softmax(logits.float(), -1)
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum().item()
            if ent < self.ent_thresh:
                return logits
        if self.buf:  # mid-character: slow path on top-k candidates
            k = min(64, logits.shape[-1])
            top = torch.topk(logits, k).indices.tolist()
            bias = []
            for t in top:
                b = T.tbytes[t] if t < len(T.tbytes) else b""
                text, _ = decode_partial(self.buf + b)
                s = syllables(text)
                ctx, g, n = self.ctx, 0, 0
                for x in s:
                    if ctx is not None:
                        g += u01_int(T.key, ctx, x) < T.gamma
                        n += 1
                    ctx = x
                bias.append(self.delta * g / n if n else 0.0)
            logits[top] += torch.tensor(bias, device=logits.device, dtype=logits.dtype)
            return logits
        if self.ctx is None:
            n = (T.nsyl - 1).clamp_min(0)
            g = T.inner
        else:
            n = T.nsyl
            g = T.inner + T.green_row(self.ctx)[T.first] * T.has
        frac = torch.where(n > 0, g / n.clamp_min(1), torch.zeros_like(g))
        logits[:V] += (self.delta * frac).to(logits.dtype)
        return logits


class VocabGreen:
    """Green rows over the vocab for KGW (keyed by prev id) / Unigram (fixed)."""

    def __init__(self, vocab_size, key, gamma, cache=2048):
        self.V, self.key, self.gamma, self.cache = vocab_size, key, gamma, cache
        self.ids = np.arange(vocab_size, dtype=np.uint64)
        self.rows = OrderedDict()

    def row(self, ctx, device):
        r = self.rows.get(ctx)
        if r is None:
            r = torch.from_numpy(u01_np(self.key, ctx, self.ids) < self.gamma).to(device)
            self.rows[ctx] = r
            if len(self.rows) > self.cache:
                self.rows.popitem(last=False)
        else:
            self.rows.move_to_end(ctx)
        return r


class KGWReq:
    def __init__(self, green: VocabGreen, delta):
        self.G, self.delta = green, delta

    def __call__(self, prompt_ids, out_ids, logits):
        prev = out_ids[-1] if out_ids else prompt_ids[-1]
        m = self.G.row(int(prev) + 1, logits.device)  # +1 keeps ctx space disjoint from Unigram's 0
        V = m.shape[0]
        logits[:V] += self.delta * m.to(logits.dtype)
        return logits


class UnigramReq:
    def __init__(self, green: VocabGreen, delta):
        self.G, self.delta = green, delta

    def __call__(self, out_ids, logits):
        m = self.G.row(0, logits.device)
        V = m.shape[0]
        logits[:V] += self.delta * m.to(logits.dtype)
        return logits


# ================================================================ Ko-MaJ h=2 (anti-repetition variants)
# Context = previous TWO syllables, encoded as NSYL + a*NSYL + b (disjoint from h=1 contexts 0..NSYL-1).
# skip=True additionally skips any position whose 2-syllable context already occurred earlier
# (generation: no bias at that step; detection: position not scored), mirroring SynthID's
# context-repetition masking.
def ctx2(a, b):
    return NSYL + a * NSYL + b


def detect_hanmark2(text, key, gamma, skip=False):
    s = syllables(text)
    ctxs, nxt, seen = [], [], set()
    for i in range(2, len(s)):
        c = (s[i - 2], s[i - 1])
        if skip:
            if c in seen:
                continue
            seen.add(c)
        ctxs.append(ctx2(*c))
        nxt.append(s[i])
    if not skip:  # dedup identical (context, next) triples
        uniq = list(dict.fromkeys(zip(ctxs, nxt)))
        ctxs, nxt = ([a for a, _ in uniq], [b for _, b in uniq]) if uniq else ([], [])
    if not ctxs:
        return 0.0, 0
    g = (u01_pairs(key, ctxs, nxt) < gamma).sum()
    return zscore(g, len(ctxs), gamma), len(ctxs)


class HanMark2Tables:
    def __init__(self, tok, vocab_size, key, gamma):
        self.key, self.gamma = key, gamma
        self.tbytes = token_bytes_table(tok, vocab_size)
        V = vocab_size
        first = np.full(V, -1, dtype=np.int64)
        second = np.full(V, -1, dtype=np.int64)
        nsyl = np.zeros(V, dtype=np.int64)
        inner = np.zeros(V, dtype=np.int64)  # green triples fully inside the token (3rd syllable on)
        for t, b in enumerate(self.tbytes):
            if not b:
                continue
            s = syllables(decode_partial(b)[0])
            if not s:
                continue
            first[t], nsyl[t] = s[0], len(s)
            if len(s) > 1:
                second[t] = s[1]
            inner[t] = sum(u01_int(key, ctx2(s[j - 2], s[j - 1]), s[j]) < gamma for j in range(2, len(s)))
        self.first_np, self.second_np, self.nsyl_np, self.inner_np = first, second, nsyl, inner
        self.multi_idx = np.nonzero(second >= 0)[0]
        # h2R: every in-token position j>=2 of every token, with its fixed context (s[j-2], s[j-1])
        ft, fa, fb, fg, fd = [], [], [], [], []
        for t, b in enumerate(self.tbytes):
            if nsyl[t] < 3:
                continue
            s = syllables(decode_partial(b)[0])
            prev = set()
            for j in range(2, len(s)):
                c = (s[j - 2], s[j - 1])
                ft.append(t); fa.append(c[0]); fb.append(c[1])
                fg.append(u01_int(key, ctx2(*c), s[j]) < gamma)
                fd.append(c in prev)  # same context earlier inside the token
                prev.add(c)
        self.fix_np = {"tok": np.array(ft, np.int64), "a": np.array(fa, np.int64), "b": np.array(fb, np.int64),
                       "green": np.array(fg, np.float32), "dup": np.array(fd, bool)}
        self.dev = None

    def to(self, device):
        if self.dev != device:
            self.first = torch.from_numpy(np.where(self.first_np < 0, 0, self.first_np)).to(device)
            self.has = torch.from_numpy(self.first_np >= 0).to(device).float()
            self.nsyl = torch.from_numpy(self.nsyl_np).to(device).float()
            self.inner = torch.from_numpy(self.inner_np).to(device).float()
            self.multi = torch.from_numpy(self.multi_idx).to(device)
            self.fix = {k: torch.from_numpy(v).to(device) for k, v in self.fix_np.items()}
            self.fix["ctx"] = self.fix["a"] * NSYL + self.fix["b"]
            self.fix["first"] = self.first[self.fix["tok"]]
            self.rows, self.sec = OrderedDict(), OrderedDict()
            self.dev = device
        return self

    def green_row(self, a, b):  # green over next syllables for context (a, b)
        k = (a, b)
        r = self.rows.get(k)
        if r is None:
            r = torch.from_numpy(u01_np(self.key, ctx2(a, b), SYL_ALL) < self.gamma).to(self.dev).float()
            self.rows[k] = r
            if len(self.rows) > 20000:
                self.rows.popitem(last=False)
        return r

    def second_green(self, p1):  # per token: is its 2nd syllable green given context (p1, first)?
        r = self.sec.get(p1)
        if r is None:
            idx = self.multi_idx
            g = u01_pairs(self.key, NSYL + p1 * NSYL + self.first_np[idx], self.second_np[idx]) < self.gamma
            r = torch.zeros(len(self.first_np), device=self.dev)
            r[self.multi] = torch.from_numpy(g.astype(np.float32)).to(self.dev)
            self.sec[p1] = r
            if len(self.sec) > 4000:
                self.sec.popitem(last=False)
        return r


class HanMark2Req:
    """h2 / h2R generation. With skip=True the bias follows the detector exactly, per syllable position:
    a position whose 2-syllable context already occurred earlier in the canonical stream (or earlier inside
    the candidate token) contributes nothing; the remaining positions give (green count) / (count)."""

    def __init__(self, tables: HanMark2Tables, delta, skip=False):
        self.T, self.delta, self.skip = tables, delta, skip
        self._reset()

    def _reset(self):
        self.pos, self.buf, self.p2, self.p1, self.stream = 0, b"", None, None, SylStream()
        self.seen, self.seen_t = set(), None  # contexts (a, b) of positions already emitted

    def _apply(self, op, v):
        if op == "replace":  # last syllable re-composed: its own context is unchanged
            self.p1 = v
            return
        if self.p2 is not None:
            self.seen.add((self.p2, self.p1))
            self.seen_t = None
        self.p2, self.p1 = self.p1, v

    def _advance(self, out_ids):
        if len(out_ids) < self.pos:
            self._reset()
        while self.pos < len(out_ids):
            t = out_ids[self.pos]
            b = self.T.tbytes[t] if t < len(self.T.tbytes) else b""
            text, self.buf = decode_partial(self.buf + b)
            for op, v in self.stream.feed(text):
                self._apply(op, v)
            self.pos += 1

    def _slow_bias(self, syls):
        p2, p1, g, n, seen = self.p2, self.p1, 0, 0, set(self.seen)
        for x in syls:
            if p2 is not None:
                c = (p2, p1)
                if not (self.skip and c in seen):
                    g += u01_int(self.T.key, ctx2(*c), x) < self.T.gamma
                    n += 1
                seen.add(c)
            p2, p1 = p1, x
        return self.delta * g / n if n else 0.0

    def _seen_ids(self, device):
        if self.seen_t is None:
            self.seen_t = torch.tensor([a * NSYL + b for a, b in self.seen] or [-1], device=device, dtype=torch.int64)
        return self.seen_t

    def __call__(self, out_ids, logits):
        self._advance(out_ids)
        T = self.T.to(logits.device)
        V = T.first.shape[0]
        if self.buf:
            k = min(64, logits.shape[-1])
            top = torch.topk(logits, k).indices.tolist()
            bias = [self._slow_bias(syllables(decode_partial(self.buf + (T.tbytes[t] if t < len(T.tbytes) else b""))[0]))
                    for t in top]
            logits[top] += torch.tensor(bias, device=logits.device, dtype=logits.dtype)
            return logits
        n, g = self.counts(T, logits.device)
        frac = torch.where(n > 0, g / n.clamp_min(1), torch.zeros_like(g))
        logits[:V] += (self.delta * frac).to(logits.dtype)
        return logits

    def counts(self, T, device):
        """Per-token (scored positions, green positions) given the current stream state."""
        p2, p1 = self.p2, self.p1
        if not self.skip:
            if p1 is None:
                return (T.nsyl - 2).clamp_min(0), T.inner
            if p2 is None:
                return (T.nsyl - 1).clamp_min(0), T.inner + T.second_green(p1)
            return T.nsyl, T.inner + T.second_green(p1) + T.green_row(p2, p1)[T.first] * T.has
        seen, F, zero = self._seen_ids(device), T.fix, torch.zeros_like(T.has)
        # position 0 of a token has context (p2, p1); position 1 has (p1, s0); positions j>=2 are fixed.
        # A context counts once: skip it if it is in `seen` or equals an earlier position's context.
        if p2 is not None and (p2, p1) not in self.seen:
            ok0 = T.has
            g0 = T.green_row(p2, p1)[T.first] * ok0
        else:
            ok0 = g0 = zero
        if p1 is not None:
            skip1 = torch.isin(p1 * NSYL + T.first, seen)
            if p2 is not None:
                skip1 = skip1 | ((T.first == p1) & (p1 == p2))
            ok1 = (T.nsyl >= 2).float() * (~skip1).float()
            g1 = T.second_green(p1) * ok1
        else:
            ok1 = g1 = zero
        skipf = torch.isin(F["ctx"], seen) | F["dup"]
        if p2 is not None:
            skipf = skipf | ((F["a"] == p2) & (F["b"] == p1))
        if p1 is not None:
            skipf = skipf | ((F["a"] == p1) & (F["b"] == F["first"]))
        okf = (~skipf).float()
        n = ok0 + ok1 + zero.clone().index_add_(0, F["tok"], okf)
        g = g0 + g1 + zero.clone().index_add_(0, F["tok"], F["green"] * okf)
        return n, g
