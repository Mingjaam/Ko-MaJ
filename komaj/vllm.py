"""Ko-MaJ watermarking for vLLM (v1 engine, tested with vLLM 0.29).

    from vllm import LLM, SamplingParams
    from komaj.vllm import KoMaJVLLM

    llm = LLM(model="kakaocorp/kanana-1.5-8b-instruct-2505", logits_processors=[KoMaJVLLM])
    sp = SamplingParams(temperature=1.0, max_tokens=400,
                        extra_args={"komaj": {"key": 123456789, "delta": 2.5, "variant": "h1"}})
    out = llm.generate(prompts, sp)

Requests without extra_args["komaj"] are generated without a watermark.
"""
from transformers import AutoTokenizer
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from .schemes import HanMark2Req, HanMark2Tables, HanMarkReq, HanMarkTables


class KoMaJVLLM(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        mc = vllm_config.model_config
        self.tok = AutoTokenizer.from_pretrained(mc.tokenizer, trust_remote_code=True)
        self.V = mc.get_vocab_size()
        self.cache = {}

    def is_argmax_invariant(self):
        return False

    def new_req_logits_processor(self, params):
        cfg = (params.extra_args or {}).get("komaj")
        if not cfg:
            return None
        key, gamma, delta = int(cfg["key"]), float(cfg.get("gamma", 0.25)), float(cfg.get("delta", 2.5))
        variant = cfg.get("variant", "h1")
        ck = (key, gamma, variant != "h1")
        if ck not in self.cache:
            self.cache[ck] = (HanMarkTables if variant == "h1" else HanMark2Tables)(self.tok, self.V, key, gamma)
        t = self.cache[ck]
        return HanMarkReq(t, delta) if variant == "h1" else HanMark2Req(t, delta, skip=variant == "h2r")
