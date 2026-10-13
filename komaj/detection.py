"""Key-only detection (numpy only, no torch / model / tokenizer). Same rules as komaj.schemes."""
import math

import numpy as np

from .core import NSYL, syllables, u01_pairs


def zscore(g, t, gamma):
    if t == 0:
        return 0.0
    return (g - gamma * t) / math.sqrt(t * gamma * (1 - gamma))


def ctx2(a, b):
    return NSYL + a * NSYL + b


def detect_hanmark(text, key, gamma=0.25):
    """h1: green pair = H(K, previous syllable, syllable) < gamma; repeated pairs counted once."""
    s = syllables(text)
    pairs = list(dict.fromkeys(zip(s[:-1], s[1:])))
    if not pairs:
        return 0.0, 0
    A, B = zip(*pairs)
    g = (u01_pairs(key, A, B) < gamma).sum()
    return zscore(g, len(pairs), gamma), len(pairs)


def detect_hanmark2(text, key, gamma=0.25, skip=False):
    """h2: context = previous two syllables. skip=True (h2R): a context already seen is not scored again."""
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
    if not skip:
        uniq = list(dict.fromkeys(zip(ctxs, nxt)))
        ctxs, nxt = ([a for a, _ in uniq], [b for _, b in uniq]) if uniq else ([], [])
    if not ctxs:
        return 0.0, 0
    g = (u01_pairs(key, np.asarray(ctxs), nxt) < gamma).sum()
    return zscore(g, len(ctxs), gamma), len(ctxs)
