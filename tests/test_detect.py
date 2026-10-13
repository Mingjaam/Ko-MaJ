"""Invariance and sanity tests for key-only detection (numpy only)."""
import random
import unicodedata

import komaj
from komaj.core import SylStream, canon, cut_at_syllables, syllables

KEY = 15485863
TEXT = ("대한민국의 수도는 서울이다. 서울은 한강을 중심으로 발달한 도시로, 조선 시대부터 정치와 경제의 중심지였다. "
        "오늘날 서울에는 약 천만 명이 살고 있으며, 수도권 전체 인구는 이천육백만 명에 이른다.")


def attacks(t):
    rng = random.Random(0)
    yield t.replace(" ", "")                                                     # remove every space
    yield "".join(c + (" " if rng.random() < .2 else "") for c in t)            # random spaces
    yield "".join(c + "​" for c in t)                                       # zero-width spaces
    yield unicodedata.normalize("NFD", t)                                        # jamo decomposition
    yield "".join(unicodedata.normalize("NFD", c)[0] + "​" + unicodedata.normalize("NFD", c)[1:]
                  if "가" <= c <= "힣" else c for c in t)                        # ZWSP inside a decomposed syllable
    yield t.replace(".", "!").replace(",", "")                                   # punctuation


def test_score_invariant_under_canonical_preserving_edits():
    for variant in ("h1", "h2", "h2r"):
        base = komaj.detect(TEXT, KEY, variant=variant)
        for t in attacks(TEXT):
            assert syllables(t) == syllables(TEXT)
            r = komaj.detect(t, KEY, variant=variant)
            assert (r.z, r.n_pairs) == (base.z, base.n_pairs)


def test_wrong_key_is_not_special():
    zs = [komaj.detect(TEXT, k).z for k in range(1, 200)]
    assert abs(sum(zs) / len(zs)) < 1.0


def test_canon_and_cut():
    assert canon("한​국") == "한국"
    t = unicodedata.normalize("NFD", TEXT)
    c = cut_at_syllables(t, 10)
    assert len(syllables(c)) == 10 and t.startswith(c)


def test_stream_matches_batch():
    rng = random.Random(1)
    t = unicodedata.normalize("NFD", TEXT)
    st, out, i = SylStream(), [], 0
    while i < len(t):
        k = rng.randint(1, 4)
        for op, v in st.feed(t[i:i + k]):
            if op == "push":
                out.append(v)
            else:
                out[-1] = v
        i += k
    assert out == syllables(TEXT)
