"""Input defences a token-based detector can reasonably add (used for the 'defended baseline' comparison).

none  : raw text, exactly as the attacker left it
norm  : drop invisible format chars (Cf) + NFC                (undoes ZWSP / NFD tricks)
space : norm + Korean spacing restoration with Kiwi           (re-spaces text whose spaces were removed/shuffled)
Ko-MaJ always uses its own canonicalisation C(x) and never needs these.
"""
from common import canon

_KIWI = None


def _kiwi():
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI = Kiwi(num_workers=1)
    return _KIWI


def apply(text, mode):
    if mode == "none":
        return text
    text = canon(text)
    if mode == "space":
        text = _kiwi().space(text, reset_whitespace=True)
    return text
