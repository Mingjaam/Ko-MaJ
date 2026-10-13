"""Generate with the official LUNA / SynthID-Text / STELA code (HF transformers, one sample at a time).

Same prompts, chat template and sampling as gen.py (T=1.0, top_p=1.0, 200-400 new tokens, per-sample seed 1000+i).
usage: python baseline_hf.py --alg LUNA --model LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct --name exaone_8b --split test
Output: ../out/gen_main/{name}/{split}/{tag}.jsonl  (tag = luna | synthid | stela), resumable per sample.
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

BASE = os.environ.get("KOMAJ_BASELINES", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "baselines"))
REPO = {"LUNA": "luna_watermark", "SynthID": "luna_watermark", "STELA": "stela_watermark"}
CFG = {"LUNA": "config/LUNA_ko_k2.json", "SynthID": "config/LUNA_ko_k2.json", "STELA": "config/STELA_ko_lang_k4.json"}
INSTR = ("다음은 한국어 위키백과 문서의 앞부분입니다. 이어지는 내용을 자연스러운 한국어로 계속 작성하세요. "
         "제목, 머리말, 설명 없이 본문만 이어서 쓰세요.\n\n")


def load_wm(alg, model, tok, device="cuda"):
    repo = os.path.join(BASE, REPO[alg])
    sys.path.insert(0, repo)
    os.chdir(repo)  # configs reference relative data/ paths
    from utils.transformers_config import TransformersConfig
    from watermark.auto_watermark import AutoWatermark
    cfg = json.load(open(CFG[alg]))
    if alg == "SynthID":  # same tournament backbone, no linguistic schedule
        cfg = {k: cfg[k] for k in ["ngram_len", "keys", "sampling_table_size", "sampling_table_seed",
                                   "watermark_mode", "num_leaves", "context_history_size", "detector_type",
                                   "threshold", "top_k"]}
        cfg["algorithm_name"] = "SynthID"
    if "temperature" in cfg or alg == "SynthID":
        cfg["temperature"] = 1.0  # match T=1.0 used for every method
    f = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, dir=repo)
    json.dump(cfg, f)
    f.close()
    tc = TransformersConfig(model=model, tokenizer=tok, vocab_size=model.config.vocab_size if model is not None else len(tok),
                            device=device, do_sample=True)
    return AutoWatermark.load(alg, algorithm_config=f.name, transformers_config=tc)


def processor(wm):
    if hasattr(wm, "logits_processor_template"):
        return wm.logits_processor_template(wm.config)
    lp = wm.logits_processor
    if hasattr(lp, "reset_state"):
        lp.reset_state()
    return lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, choices=list(REPO))
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.environ.get("KOMAJ_OUT", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "out")), "gen_main"))
    ap.add_argument("--data", default=os.environ.get("KOMAJ_DATA", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data")))
    ap.add_argument("--max_new", type=int, default=400)
    ap.add_argument("--min_new", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{a.data}/{a.split}.jsonl", encoding="utf-8")]
    if a.n:
        rows = rows[: a.n]
    odir = f"{a.out}/{a.name}/{a.split}"
    os.makedirs(odir, exist_ok=True)
    final, part = f"{odir}/{a.alg.lower()}.jsonl", f"{odir}/{a.alg.lower()}.jsonl.part"
    if os.path.exists(final):
        print("done", final)
        return
    done = set()
    if os.path.exists(part):
        done = {json.loads(l)["id"] for l in open(part, encoding="utf-8")}

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, trust_remote_code=True).to(a.device).eval()
    wm = load_wm(a.alg, model, tok, a.device)
    gen_kwargs = dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new, min_new_tokens=a.min_new,
                      pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    t0, n = time.time(), 0
    with open(part, "a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            p = tok.apply_chat_template([{"role": "user", "content": INSTR + r["prompt"]}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
            enc = tok(p, return_tensors="pt", add_special_tokens=True).to(a.device)
            torch.manual_seed(1000 + i)
            with torch.inference_mode():
                out = model.generate(**enc, logits_processor=LogitsProcessorList([processor(wm)]), **gen_kwargs)
            new = out[0, enc["input_ids"].shape[1]:].tolist()
            text = tok.decode(new, skip_special_tokens=True)
            f.write(json.dumps({"id": r["id"], "text": text, "token_ids": new}, ensure_ascii=False) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                print(f"[{a.name}/{a.alg}] {len(done) + n}/{len(rows)}  {(time.time() - t0) / n:.1f}s/sample", flush=True)
    os.replace(part, final)
    print(f"[{a.name}/{a.alg}] finished {len(rows)} samples", flush=True)


if __name__ == "__main__":
    main()
