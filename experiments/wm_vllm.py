"""vLLM v1 logits processor dispatching to watermark schemes via SamplingParams.extra_args['wm'].

extra_args = {"wm": {"scheme": "hanmark"|"kgw"|"unigram", "key": int, "gamma": float,
                     "delta": float, "ent": float|None}}
"""
from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from schemes import HanMark2Req, HanMark2Tables, HanMarkReq, HanMarkTables, KGWReq, UnigramReq, VocabGreen


class WatermarkLP(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        mc = vllm_config.model_config
        self.tok = AutoTokenizer.from_pretrained(mc.tokenizer, trust_remote_code=True)
        self.V = mc.get_vocab_size()
        self.cache = {}

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        cfg = (params.extra_args or {}).get("wm")
        if not cfg or cfg["scheme"] == "none":
            return None
        s, key, gamma, delta = cfg["scheme"], int(cfg["key"]), float(cfg["gamma"]), float(cfg["delta"])
        ck = (s, key, gamma)
        if ck not in self.cache:
            if s == "hanmark":
                self.cache[ck] = HanMarkTables(self.tok, self.V, key, gamma)
            elif s == "hanmark2":
                self.cache[ck] = HanMark2Tables(self.tok, self.V, key, gamma)
            else:
                self.cache[ck] = VocabGreen(self.V, key, gamma)
        obj = self.cache[ck]
        if s == "hanmark":
            return HanMarkReq(obj, delta, cfg.get("ent"))
        if s == "hanmark2":
            return HanMark2Req(obj, delta, bool(cfg.get("skip", False)))
        if s == "kgw":
            return KGWReq(obj, delta)
        if s == "unigram":
            return UnigramReq(obj, delta)
        raise ValueError(s)
