"""Generate continuations with vLLM under several watermark configs (resumable: skips existing outputs).

usage: python gen.py --model PATH --name qwen25_7b --split dev --n 100 --configs configs/e0.json
"""
import argparse
import json
import os
import time

from vllm import LLM, SamplingParams

from wm_vllm import WatermarkLP

INSTR = ("다음은 한국어 위키백과 문서의 앞부분입니다. 이어지는 내용을 자연스러운 한국어로 계속 작성하세요. "
         "제목, 머리말, 설명 없이 본문만 이어서 쓰세요.\n\n")
INSTR_NEWS = ("다음은 한국어 뉴스 기사의 앞부분입니다. 이어지는 내용을 자연스러운 한국어로 계속 작성하세요. "
              "제목, 머리말, 설명 없이 본문만 이어서 쓰세요.\n\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--configs", required=True)
    ap.add_argument("--max_tokens", type=int, default=400)
    ap.add_argument("--min_tokens", type=int, default=200)
    ap.add_argument("--data", default="../data")
    ap.add_argument("--out", default="../out/gen")
    ap.add_argument("--gpu_util", type=float, default=0.85)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(f"{a.data}/{a.split}.jsonl", encoding="utf-8")]
    if a.n:
        rows = rows[: a.n]
    configs = json.load(open(a.configs))
    odir = f"{a.out}/{a.name}/{a.split}"
    os.makedirs(odir, exist_ok=True)
    def done(c):
        p = f"{odir}/{c['tag']}.jsonl"
        return os.path.exists(p) and sum(1 for _ in open(p, encoding="utf-8")) == len(rows)
    todo = [c for c in configs if not done(c)]
    if not todo:
        print("all done")
        return

    kw = dict(model=a.model, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=a.gpu_util,
              logits_processors=[WatermarkLP], seed=0)
    if "Qwen3.8" in a.model or "qwen3_5" in a.model.lower():
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        kw["max_num_seqs"] = 256
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    prompts = []
    for r in rows:
        msgs = [{"role": "user", "content": (INSTR_NEWS if a.split.startswith("news") else INSTR) + r["prompt"]}]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))

    for c in todo:
        wm = {k: c[k] for k in ("scheme", "key", "gamma", "delta", "ent", "skip") if k in c}
        sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, min_tokens=a.min_tokens, seed=1000 + i,
                              extra_args={"wm": wm}) for i in range(len(rows))]
        t0 = time.time()
        outs = llm.generate(prompts, sps)
        dt = time.time() - t0
        with open(f"{odir}/{c['tag']}.jsonl.tmp", "w", encoding="utf-8") as f:
            for r, o in zip(rows, outs):
                g = o.outputs[0]
                f.write(json.dumps({"id": r["id"], "text": g.text, "token_ids": list(g.token_ids),
                                    "finish": g.finish_reason}, ensure_ascii=False) + "\n")
        os.replace(f"{odir}/{c['tag']}.jsonl.tmp", f"{odir}/{c['tag']}.jsonl")
        ntok = sum(len(o.outputs[0].token_ids) for o in outs)
        print(f"[{a.name}] {c['tag']}: {len(rows)} samples, {ntok} tokens, {dt:.1f}s ({ntok/dt:.0f} tok/s)", flush=True)


if __name__ == "__main__":
    main()
