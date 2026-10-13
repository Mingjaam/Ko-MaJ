"""Rule-based attacks (no model). LLM attacks live in llm_attack.py."""
import random
import unicodedata

from common import is_syl


def nospace(text, seed=0):
    return text.replace(" ", "")


def randspace(text, seed=0, p_del=0.3, p_ins=0.15):
    """Delete each space w.p. p_del; insert a space after a Hangul syllable w.p. p_ins."""
    rng = random.Random(seed)
    out = []
    for i, c in enumerate(text):
        if c == " ":
            if rng.random() >= p_del:
                out.append(c)
            continue
        out.append(c)
        if is_syl(c) and i + 1 < len(text) and is_syl(text[i + 1]) and rng.random() < p_ins:
            out.append(" ")
    return "".join(out)


def nfd(text, seed=0):
    """Decompose every syllable into conjoining jamo (renders identically in most fonts)."""
    return unicodedata.normalize("NFD", text)


def zwsp(text, seed=0, p=0.5):
    """Insert an invisible zero-width space after a Hangul syllable w.p. p."""
    rng = random.Random(seed)
    return "".join(c + ("​" if is_syl(c) and rng.random() < p else "") for c in text)


ATTACKS = {"clean": lambda t, seed=0: t, "nospace": nospace, "randspace": randspace, "nfd": nfd, "zwsp": zwsp}


# ---------------------------------------------------------------- revision attacks (2026-10-13)
_FREQ = None


def _syl_freq():
    """Syllable distribution of the human Korean corpus (replacement syllables look natural in frequency)."""
    global _FREQ
    if _FREQ is None:
        import collections
        import json
        import os
        c = collections.Counter()
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/human.jsonl")
        for l in open(p, encoding="utf-8"):
            c.update(ch for ch in json.loads(l)["text"] if is_syl(ch))
        top = c.most_common(2000)
        _FREQ = ([s for s, _ in top], [n for _, n in top])
    return _FREQ


def nfd_zw(text, seed=0, p=0.5):
    """Compound: NFD, then a zero-width space INSIDE a decomposed syllable (between its jamo) w.p. p."""
    rng = random.Random(seed)
    out = []
    for c in text:
        if is_syl(c):
            j = unicodedata.normalize("NFD", c)
            out.append(j[0] + ("\u200b" if rng.random() < p else "") + j[1:])
        else:
            out.append(c)
    return "".join(out)


_CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_JONG = ["", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ", "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ",
         "ㅅ", "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ"]


def compat_jamo(text, seed=0):
    """Spell every syllable with compatibility jamo (한 -> ㅎㅏㄴ). Visibly different and hard to read: a
    boundary case where the canonical syllable sequence is destroyed (no method is expected to survive)."""
    out = []
    for c in text:
        if is_syl(c):
            k = ord(c) - 0xAC00
            out.append(_CHO[k // 588] + _JUNG[(k % 588) // 28] + _JONG[k % 28])
        else:
            out.append(c)
    return "".join(out)


def syl_sub(text, seed=0, p=0.1):
    """Replace each Hangul syllable w.p. p with a frequency-weighted random syllable."""
    rng = random.Random(seed)
    pool, w = _syl_freq()
    return "".join(rng.choices(pool, w)[0] if is_syl(c) and rng.random() < p else c for c in text)


def syl_del(text, seed=0, p=0.1):
    """Delete each Hangul syllable w.p. p."""
    rng = random.Random(seed)
    return "".join(c for c in text if not (is_syl(c) and rng.random() < p))


def excerpt(text, seed=0, frac=1 / 3):
    """Drop the first third of the text (cut at a space), keep the rest: a quoted excerpt."""
    k = int(len(text) * frac)
    sp = text.find(" ", k)
    return text[sp + 1:] if sp >= 0 else text[k:]


def mix_human(text, seed=0, ref=None):
    """Interleave sentences of the watermarked text with sentences of a human text (about half each)."""
    import re
    if not ref:
        return text
    split = lambda t: [s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s]
    a, b = split(text), split(ref)
    out = []
    for i in range(max(len(a), len(b))):
        if i < len(a):
            out.append(a[i])
        if i < len(b):
            out.append(b[i])
    return " ".join(out)


ATTACKS.update({
    "nfdzw": nfd_zw, "jamo": compat_jamo,
    "sub10": lambda t, seed=0: syl_sub(t, seed, 0.1), "sub20": lambda t, seed=0: syl_sub(t, seed, 0.2),
    "del10": lambda t, seed=0: syl_del(t, seed, 0.1), "del20": lambda t, seed=0: syl_del(t, seed, 0.2),
    "excerpt": excerpt, "mix": mix_human,
})
NEEDS_REF = {"mix"}
