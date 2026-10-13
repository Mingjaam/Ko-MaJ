"""Oracle perplexity of generated continuations (conditioned on the Wikipedia prefix), Qwen3.8-27B via vLLM.

usage: python ppl.py --names qwen25_7b --split dev
Writes ../out/ppl/{name}_{split}_{tag}.json (resumable).
"""
import argparse
import glob
import json
import math
import os

from vllm import LLM, SamplingParams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default=os.environ.get("KOMAJ_BIG_MODEL", "Qwen3.8-27B-FP8"))
    ap.add_argument("--names", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--gen_sub", default="gen", help="subdir of out/ holding generations; results go to out/ppl_<suffix>")
    a = ap.parse_args()

    prefix = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(f"../data/{a.split}.jsonl", encoding="utf-8")}
    pdir = f"{a.out}/ppl" if a.gen_sub == "gen" else f"{a.out}/ppl_{a.gen_sub[4:] or a.gen_sub}"
    jobs = []
    for name in a.names.split(","):
        for p in sorted(glob.glob(f"{a.out}/{a.gen_sub}/{name}/{a.split}/*.jsonl")):
            tag = os.path.basename(p)[:-6]
            o = f"{pdir}/{name}_{a.split}_{tag}.json"
            if not os.path.exists(o):
                jobs.append((p, o))
    if not jobs:
        print("all done"); return
    os.makedirs(pdir, exist_ok=True)
    llm = LLM(model=a.oracle, max_model_len=4096, gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=256)
    tok = llm.get_tokenizer()
    sp = SamplingParams(max_tokens=1, prompt_logprobs=0)
    for p, o in jobs:
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        ins, spans = [], []
        for r in rows:
            ctx = tok(prefix[r["id"]] + "\n", add_special_tokens=False)["input_ids"]
            full = tok(prefix[r["id"]] + "\n" + r["text"], add_special_tokens=False)["input_ids"]
            ins.append({"prompt_token_ids": full})
            spans.append((len(ctx), len(full)))
        outs = llm.generate(ins, sp)
        res = []
        for r, out, (s, e) in zip(rows, outs, spans):
            lps = [list(d.values())[0].logprob if len(d) == 1 else d[out.prompt_token_ids[k]].logprob
                   for k, d in enumerate(out.prompt_logprobs) if d is not None and s <= k < e]
            res.append({"id": r["id"], "ppl": math.exp(-sum(lps) / len(lps)) if lps else float("nan"), "n": len(lps)})
        json.dump(res, open(o, "w"))
        v = sorted(x["ppl"] for x in res if x["n"] > 0)
        print(f"{os.path.basename(o)}: median ppl {v[len(v)//2]:.2f}", flush=True)


if __name__ == "__main__":
    main()
