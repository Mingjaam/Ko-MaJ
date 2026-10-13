"""Generate a watermarked Korean continuation with HuggingFace transformers, then detect it with the key only.

    pip install "komaj[hf]"
    python examples/generate_hf.py --model Qwen/Qwen2.5-0.5B-Instruct --key 123456789
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

import komaj
from komaj.hf import KoMaJLogitsProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--key", type=int, default=123456789)
ap.add_argument("--delta", type=float, default=3.0)
ap.add_argument("--max_new_tokens", type=int, default=200)
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
msgs = [{"role": "user", "content": "한국의 사계절에 대해 자연스러운 한국어로 한 문단 써 주세요."}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
plen = inputs["input_ids"].shape[1]

for name, procs in [("watermarked", [KoMaJLogitsProcessor(tok, key=a.key, delta=a.delta, prompt_len=plen)]), ("plain", [])]:
    torch.manual_seed(0)
    out = model.generate(**inputs, do_sample=True, temperature=1.0, top_p=1.0, max_new_tokens=a.max_new_tokens,
                         logits_processor=LogitsProcessorList(procs))
    text = tok.decode(out[0, plen:], skip_special_tokens=True)
    print(f"--- {name}\n{text[:160]}...\n{komaj.detect(text, a.key)}")
