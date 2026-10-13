"""LLM-as-judge (Qwen3.8-27B, thinking off, greedy) for quality beyond PPL and for attack validity.

quality  : continuation given the prompt -> fluency / coherence / relevance, each 1-10
meaning  : original vs attacked text -> meaning preserved 1-5 (is the rewrite still the same content?)
Writes ../out/judge/{kind}/{name}/{split}/{tag}[__{attack}].json (complete files only; resumable).
usage: python judge.py --names qwen25_7b,exaone_8b --split test
"""
import argparse
import glob
import json
import os
import re

from vllm import LLM, SamplingParams

Q_PROMPT = ("다음은 한국어 글의 앞부분(맥락)과 그 뒤에 이어 쓴 글입니다. 이어 쓴 글만 평가하세요.\n"
            "- fluency: 문법과 표현이 자연스러운 한국어인가 (1~10)\n"
            "- coherence: 반복이나 모순 없이 논리적으로 이어지는가 (1~10)\n"
            "- relevance: 맥락의 주제와 내용에 맞게 이어지는가 (1~10)\n"
            '다른 말 없이 {{"fluency": 정수, "coherence": 정수, "relevance": 정수}} 형식의 JSON만 출력하세요.\n\n'
            "[맥락]\n{prompt}\n\n[이어 쓴 글]\n{text}")
M_PROMPT = ("두 한국어 글이 같은 내용을 전달하는지 평가하세요. 표현이나 문체가 달라도 사실과 주장이 같으면 높은 점수입니다.\n"
            "1 = 전혀 다른 내용, 3 = 일부 내용이 빠지거나 바뀜, 5 = 내용이 완전히 같음\n"
            '다른 말 없이 {{"meaning": 정수}} 형식의 JSON만 출력하세요.\n\n[원문]\n{a}\n\n[수정본]\n{b}')
Q_TAGS = None  # all generation files
M_TAGS = ["none", "han_d3", "han_d2.5", "kgw_d2", "unigram_d2", "synthid", "luna", "stela", "han2r_d3"]
M_ATTACKS = ["style", "paraphrase", "backtrans"]


def parse(s, keys):
    m = re.search(r"\{.*?\}", s, re.S)
    try:
        d = json.loads(m.group(0)) if m else {}
        return {k: int(d[k]) for k in keys}
    except Exception:
        return {k: None for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=os.environ.get("KOMAJ_BIG_MODEL", "Qwen3.8-27B-FP8"))
    ap.add_argument("--names", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--kinds", default="quality,meaning")
    ap.add_argument("--out", default="../out")
    ap.add_argument("--tags", default="", help="quality: only these generation tags")
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--mtags", default=",".join(M_TAGS), help="meaning: only these tags")
    a = ap.parse_args()
    prompts = {json.loads(l)["id"]: json.loads(l)["prompt"] for l in open(f"../data/{a.split}.jsonl", encoding="utf-8")}
    jobs = []
    for name in a.names.split(","):
        gdir = f"{a.out}/gen_main/{name}/{a.split}"
        if "quality" in a.kinds:
            for p in sorted(glob.glob(f"{gdir}/*.jsonl")):
                tag = os.path.basename(p)[:-6]
                if a.tags and tag not in a.tags.split(","):
                    continue
                dst = f"{a.out}/judge/quality/{name}/{a.split}/{tag}.json"
                if not os.path.exists(dst):
                    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
                    jobs.append(("quality", dst, [(r["id"], Q_PROMPT.format(prompt=prompts[r["id"]], text=r["text"])) for r in rows]))
        if "meaning" in a.kinds:
            for tag in a.mtags.split(","):
                src = f"{gdir}/{tag}.jsonl"
                if not os.path.exists(src):
                    continue
                orig = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(src, encoding="utf-8")}
                for atk in M_ATTACKS:
                    ap_ = f"{a.out}/attack/{atk}/{name}/{a.split}/{tag}.jsonl"
                    dst = f"{a.out}/judge/meaning/{name}/{a.split}/{tag}__{atk}.json"
                    if os.path.exists(ap_) and not os.path.exists(dst):
                        rows = [json.loads(l) for l in open(ap_, encoding="utf-8")]
                        jobs.append(("meaning", dst, [(r["id"], M_PROMPT.format(a=orig[r["id"]], b=r["text"])) for r in rows]))
    if not jobs:
        print("all done"); return
    llm = LLM(model=a.judge, trust_remote_code=True, max_model_len=6144, gpu_memory_utilization=a.gpu_util, seed=0,
              limit_mm_per_prompt={"image": 0, "video": 0}, max_num_seqs=a.max_num_seqs)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.0, max_tokens=40)
    for kind, dst, items in jobs:
        if os.path.exists(dst):  # another judge process finished it meanwhile
            continue
        ps = [tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False) for _, q in items]
        outs = llm.generate(ps, sp)
        keys = ["fluency", "coherence", "relevance"] if kind == "quality" else ["meaning"]
        res = [{"id": i, **parse(o.outputs[0].text, keys)} for (i, _), o in zip(items, outs)]
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        json.dump(res, open(dst + ".tmp", "w"))
        os.replace(dst + ".tmp", dst)
        ok = [r for r in res if r[keys[0]] is not None]
        print(f"{dst}: {len(ok)}/{len(res)} parsed, mean {sum(r[keys[0]] for r in ok) / max(1, len(ok)):.2f}", flush=True)


if __name__ == "__main__":
    main()
