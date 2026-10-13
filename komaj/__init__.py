"""Ko-MaJ: Korean Watermarking by Syllable Juxtaposition.

A generation-time watermark for Korean LLM output whose green list is defined over adjacent pairs of
normalized Hangul syllables. Detection needs only the secret key: no model, no tokenizer, no
morphological analyzer, and the score does not change under spacing edits, invisible characters,
jamo decomposition or re-tokenization.

    >>> import komaj
    >>> r = komaj.detect("검사할 한국어 글 ...", key=123456789)
    >>> r.z, r.n_pairs, r.is_watermarked
"""
from dataclasses import dataclass
import math

from .core import canon, cut_at_syllables, syllables
from .detection import detect_hanmark, detect_hanmark2

__version__ = "0.1.0"
__all__ = ["detect", "Detection", "canon", "syllables", "cut_at_syllables", "__version__"]


@dataclass
class Detection:
    z: float                # z-score of the green-pair count
    n_pairs: int            # scored syllable pairs (repeats counted once)
    p_value: float          # one-sided p-value under the standard normal null
    threshold: float        # z threshold used for the decision
    is_watermarked: bool

    def __str__(self):
        verdict = "watermarked" if self.is_watermarked else "not watermarked"
        return f"z = {self.z:.2f} over {self.n_pairs} syllable pairs (p = {self.p_value:.2e}) -> {verdict}"


def detect(text, key, gamma=0.25, variant="h1", threshold=4.0, max_syllables=None):
    """Detect a Ko-MaJ watermark in `text` with the secret `key`.

    variant   : "h1" (context = previous syllable, default), "h2" (previous two), "h2r" (h2 + repeated contexts skipped)
    threshold : z threshold; 4.0 means about 3e-5 false positives under the normal null. For a calibrated
                1% operating point, take the 99th percentile of z on human text of the same length.
    max_syllables : score only the first N canonical syllables (e.g. 200, as in the paper)
    """
    if max_syllables:
        cut = cut_at_syllables(text, max_syllables)
        text = cut if cut is not None else text
    if variant == "h1":
        z, n = detect_hanmark(text, key, gamma)
    elif variant in ("h2", "h2r"):
        z, n = detect_hanmark2(text, key, gamma, skip=variant == "h2r")
    else:
        raise ValueError("variant must be 'h1', 'h2' or 'h2r'")
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return Detection(z=float(z), n_pairs=int(n), p_value=p, threshold=threshold, is_watermarked=bool(n > 0 and z > threshold))
