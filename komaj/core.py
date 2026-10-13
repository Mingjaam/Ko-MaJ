"""Shared utilities: keyed hashing, Hangul syllable extraction, token->bytes mapping."""
import codecs
import unicodedata
import numpy as np

M64 = 0xFFFFFFFFFFFFFFFF
HANGUL0, NSYL = 0xAC00, 11172
SYL_ALL = np.arange(NSYL, dtype=np.uint64)


def _mix_int(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & M64
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & M64
    x ^= x >> 31
    return x


def _mix_np(x):
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(30)
        x = x * np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(27)
        x = x * np.uint64(0x94D049BB133111EB)
        x ^= x >> np.uint64(31)
    return x


def u01_int(key, a, b):
    """Keyed hash of the pair (a, b) -> [0, 1). Scalar version."""
    return (_mix_int(_mix_int((key ^ _mix_int(a + 1)) & M64) ^ b) >> 11) / float(1 << 53)


def u01_np(key, a, b_arr):
    """Same as u01_int with scalar a and array b."""
    h = np.uint64(_mix_int((key ^ _mix_int(a + 1)) & M64))
    return (_mix_np(h ^ b_arr.astype(np.uint64)) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def u01_pairs(key, a_arr, b_arr):
    """Vectorised u01_int over arrays of (a, b)."""
    a = np.asarray(a_arr, dtype=np.uint64)
    b = np.asarray(b_arr, dtype=np.uint64)
    with np.errstate(over="ignore"):
        h = _mix_np(np.uint64(key) ^ _mix_np(a + np.uint64(1)))
    return (_mix_np(h ^ b) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def is_syl(ch):
    return HANGUL0 <= ord(ch) < HANGUL0 + NSYL


def canon(text):
    """C(x): drop invisible format characters (Unicode Cf: ZWSP, ZWJ, soft hyphen, BOM ...), then NFC."""
    if not text.isascii():
        text = "".join(c for c in text if unicodedata.category(c) != "Cf")
        text = unicodedata.normalize("NFC", text)
    return text


def syllables(text):
    """N(x): C(x), then keep Hangul syllables only, as indices 0..11171."""
    return [ord(c) - HANGUL0 for c in canon(text) if HANGUL0 <= ord(c) < HANGUL0 + NSYL]


def n_syl(text):
    return sum(HANGUL0 <= ord(c) < HANGUL0 + NSYL for c in canon(text))


def cut_at_syllables(text, n):
    """Shortest prefix of the RAW text whose canonical form C(prefix) holds n Hangul syllables (None if fewer).

    The prefix is returned un-normalised, so token-based detectors see exactly what the attacker produced;
    Ko-MaJ canonicalises inside its detector. Length is still measured on the canonical content so every
    method is scored on the same amount of text.
    """
    if canon(text) == text:  # fast path (no Cf, already NFC): plain scan
        k = 0
        for i, c in enumerate(text):
            if HANGUL0 <= ord(c) < HANGUL0 + NSYL:
                k += 1
                if k == n:
                    return text[: i + 1]
        return None
    if n_syl(text) < n:
        return None
    lo, hi = n, len(text)  # n_syl(prefix) is non-decreasing in the prefix length
    while lo < hi:
        mid = (lo + hi) // 2
        if n_syl(text[:mid]) >= n:
            hi = mid
        else:
            lo = mid + 1
    # A decomposed syllable counts as soon as its vowel appears; keep the trailing final consonant (and any
    # format characters between) so the n-th canonical syllable is complete: syllables(cut) == syllables(text)[:n].
    while lo < len(text) and (0x1160 <= ord(text[lo]) <= 0x11FF or unicodedata.category(text[lo]) == "Cf"):
        lo += 1
    return text[:lo]


_LV = lambda c: HANGUL0 <= ord(c) < HANGUL0 + NSYL and (ord(c) - HANGUL0) % 28 == 0  # syllable w/o final
_JAMO = lambda c: 0x1100 <= ord(c) <= 0x11FF


class SylStream:
    """Streaming version of syllables(): feed decoded text piece by piece, get the same canonical syllables
    as syllables(full_text). Returns ops: ("push", s) for a new syllable, ("replace", s) when a trailing
    conjoining final consonant re-composes the last emitted open syllable (e.g. 가 + U+11AB -> 간)."""

    def __init__(self):
        self.tail, self.pending = "", ""  # tail: last emitted open syllable that may still take a final

    def feed(self, text):
        text = "".join(c for c in text if unicodedata.category(c) != "Cf")
        if not text:
            return []
        n = unicodedata.normalize("NFC", self.tail + self.pending + text)
        ops = []
        if self.tail:
            if n[0] != self.tail:
                ops.append(("replace", ord(n[0]) - HANGUL0))
            n = n[1:]
        k = len(n)
        while k > 0 and _JAMO(n[k - 1]):
            k -= 1
        head, self.pending = n[:k], n[k:]
        for c in head:
            if HANGUL0 <= ord(c) < HANGUL0 + NSYL:
                ops.append(("push", ord(c) - HANGUL0))
        if self.pending:
            self.tail = ""
        else:  # empty head here means the new text was absorbed into the old tail (now closed)
            self.tail = head[-1] if head and _LV(head[-1]) else ""
        return ops


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


_BYTE_DEC = {v: k for k, v in _bytes_to_unicode().items()}


def token_bytes_table(tok, vocab_size):
    """Raw bytes for every token id (special tokens -> b'')."""
    special = set(tok.all_special_ids)
    out = []
    for i in range(vocab_size):
        if i >= len(tok) or i in special:
            out.append(b"")
            continue
        s = tok.convert_ids_to_tokens(i)
        if s is None:
            out.append(b"")
        elif len(s) == 6 and s.startswith("<0x") and s.endswith(">"):  # sentencepiece byte fallback
            out.append(bytes([int(s[3:5], 16)]))  # (checked first: "<0xEA>" is also valid byte-BPE text)
        elif all(c in _BYTE_DEC for c in s):  # GPT-2 style byte-level BPE
            out.append(bytes(_BYTE_DEC[c] for c in s))
        else:
            out.append(s.replace("▁", " ").encode("utf-8"))
    return out


def decode_partial(buf: bytes):
    """Decode as much of buf as possible; return (completed_text, leftover_incomplete_bytes)."""
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    text = dec.decode(buf, final=False)
    rest = dec.getstate()[0]
    return text, rest
