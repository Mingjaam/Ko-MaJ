"""Check that the vectorised generation bias equals a slow reference written from the detector's rule.

For random histories (real model outputs cut at random points) and EVERY token with Hangul, the reference
scores the positions of H+S that fall in the candidate token S using the detector's own loop
(detect_hanmark2 semantics), and compares (scored count, green count) with HanMark2Req.counts.
Also checks h1 / h2 streaming state against syllables(full text).
usage: python test_exact.py --tok Qwen/Qwen2.5-7B-Instruct --gen ../out/gen_main/qwen25_7b/test/none.jsonl
"""
import argparse
import json
import random
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

from common import decode_partial, syllables, u01_int
from schemes import HanMark2Req, HanMark2Tables, HanMarkReq, HanMarkTables, ctx2

ap = argparse.ArgumentParser()
ap.add_argument("--tok", required=True)
ap.add_argument("--gen", required=True)
ap.add_argument("--n", type=int, default=30)
ap.add_argument("--min_checked", type=int, default=10000)
a = ap.parse_args()
key, gamma = 15485863, 0.25
tok = AutoTokenizer.from_pretrained(a.tok, trust_remote_code=True)
V = len(tok)
T2 = HanMark2Tables(tok, V, key, gamma).to("cpu")
T1 = HanMarkTables(tok, V, key, gamma)
rows = [json.loads(l) for l in open(a.gen, encoding="utf-8")]
rng = random.Random(0)
cand = [t for t in range(V) if T2.nsyl_np[t] > 0]
tsyl = {t: syllables(decode_partial(T2.tbytes[t])[0]) for t in cand}


def ref_counts(H, S, skip):
    seq, seen, g, n = H + S, set(), 0, 0
    for i in range(2, len(seq)):
        c = (seq[i - 2], seq[i - 1])
        if skip:
            if c in seen:
                continue
            seen.add(c)
        if i >= len(H):
            g += u01_int(key, ctx2(*c), seq[i]) < gamma
            n += 1
    return n, g


bad = {True: 0, False: 0}
bad_h1 = 0
checked = 0
for k in range(a.n):
    r = rows[rng.randrange(len(rows))]
    ids = r["token_ids"]
    cut = rng.randrange(1, len(ids))
    pre = ids[:cut]
    buf = b"".join(T2.tbytes[t] for t in pre)
    text, rest = decode_partial(buf)
    if rest:
        continue  # buffer path is the top-64 slow path; tested separately by construction
    H = syllables(text)
    # streaming state == batch canonicalisation
    q1 = HanMarkReq(T1, 2.0); q1._advance(pre)
    assert q1.ctx == (H[-1] if H else None), "h1 stream state"
    for skip in (False, True):
        q = HanMark2Req(T2, 2.0, skip); q._advance(pre)
        assert (q.p2, q.p1) == ((H[-2] if len(H) > 1 else None), (H[-1] if H else None)), "h2 stream state"
        n, g = q.counts(T2, "cpu")
        n, g = n.numpy(), g.numpy()
        for t in cand:
            rn, rg = ref_counts(H, tsyl[t], skip)
            if (rn, rg) != (int(n[t]), int(g[t])):
                bad[skip] += 1
        checked += len(cand)
    # h1 fast path: bias == delta * (green pairs completed by the token) / (pairs completed)
    logits = torch.zeros(V)
    out = q1(pre, logits.clone())
    for t in cand:
        S = tsyl[t]
        prev, gg, nn = (H[-1] if H else None), 0, 0
        for x in S:
            if prev is not None:
                gg += u01_int(key, prev, x) < gamma
                nn += 1
            prev = x
        ref = 2.0 * gg / nn if nn else 0.0
        if abs(float(out[t]) - ref) > 1e-5:
            bad_h1 += 1
print(f"{a.tok}: {checked} (history, token) pairs, mismatches h1={bad_h1} h2={bad[False]} h2R={bad[True]}")
# Scope: candidates are whole precomposed-syllable tokens with an empty UTF-8 buffer and no pending jamo in the
# stream. Byte-fragment steps use the top-64 slow path and lone conjoining-jamo candidates are not covered.
if checked < a.min_checked or bad_h1 or bad[False] or bad[True]:
    sys.exit(1)
