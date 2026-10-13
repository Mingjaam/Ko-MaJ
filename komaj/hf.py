"""Ko-MaJ watermarking for HuggingFace `transformers` generation.

    from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
    from komaj.hf import KoMaJLogitsProcessor

    proc = KoMaJLogitsProcessor(tok, key=123456789, delta=2.5, prompt_len=inputs.input_ids.shape[1])
    out = model.generate(**inputs, logits_processor=LogitsProcessorList([proc]), do_sample=True)

The bias added to a candidate token is delta * (green syllable pairs it completes) / (pairs it completes);
tokens that complete no Hangul syllable (spaces, punctuation, Latin, byte fragments) are left alone.
"""
import torch
from transformers import LogitsProcessor

from .schemes import HanMark2Req, HanMark2Tables, HanMarkReq, HanMarkTables

_TABLES = {}


class KoMaJLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, key, delta=2.5, gamma=0.25, prompt_len=0, variant="h1", vocab_size=None):
        if variant not in ("h1", "h2", "h2r"):
            raise ValueError("variant must be 'h1', 'h2' or 'h2r'")
        V = vocab_size or len(tokenizer)
        ck = (id(tokenizer), V, key, gamma, variant != "h1")
        if ck not in _TABLES:  # per-tokenizer precomputation (a few seconds), reused across calls
            _TABLES[ck] = (HanMarkTables if variant == "h1" else HanMark2Tables)(tokenizer, V, key, gamma)
        self.tables, self.delta, self.variant, self.prompt_len = _TABLES[ck], delta, variant, prompt_len
        self.reqs = {}

    def _req(self, b):
        if b not in self.reqs:
            self.reqs[b] = HanMarkReq(self.tables, self.delta) if self.variant == "h1" else \
                HanMark2Req(self.tables, self.delta, skip=self.variant == "h2r")
        return self.reqs[b]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for b in range(input_ids.shape[0]):
            out_ids = input_ids[b, self.prompt_len:].tolist()
            scores[b] = self._req(b)(out_ids, scores[b])
        return scores
