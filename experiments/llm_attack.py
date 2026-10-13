"""LLM attacks with a fixed attacker (default Qwen3.8-27B, thinking off).

style      : switch sentence-final style (-다 <-> -습니다 / -습니다 -> -어요), keep wording
paraphrase : rewrite with different expressions, same meaning
backtrans  : ko -> en -> ko
usage: python llm_attack.py --names qwen25_7b --split dev --tags none,kgw_d2,han_d2 --attacks style,paraphrase,backtrans
Writes ../out/attack/{attack}/{name}/{split}/{tag}.jsonl (resumable).
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams

P = {
    "style": ("다음 글의 문장 끝 문체만 바꾸세요. '-다'로 끝나는 평서문은 '-습니다/-ㅂ니다'체로, "
              "'-습니다'체는 '-어요/-해요'체로 바꾸고, 단어와 내용은 최대한 그대로 두세요. 바꾼 글만 출력하세요.\n\n"),
    "paraphrase": "다음 글을 의미는 그대로 유지하면서 다른 단어와 문장 구조로 바꿔 쓰세요. 바꾼 글만 출력하세요.\n\n",
    "to_en": "Translate the following Korean text into English. Output only the translation.\n\n",
    "to_ko": "다음 영어 글을 자연스러운 한국어로 번역하세요. 번역문만 출력하세요.\n\n",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacker", default=os.environ.get("KOMAJ_BIG_MODEL", "Qwen3.8-27B-FP8"))
    ap.add_argument("--attacker_tag", default="", help="suffix for output dir, e.g. _exaone for cross-attack")
    ap.add_argument("--names", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--tags", required=True)
    ap.add_argument("--attacks", default="style,paraphrase,backtrans")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--out", default="../out")
    ap.add_argument("--gen_sub", default="gen_main")
    a = ap.parse_args()

    jobs = []
    for atk in a.attacks.split(","):
        for name in a.names.split(","):
            for tag in a.tags.split(","):
                src = f"{a.out}/{a.gen_sub}/{name}/{a.split}/{tag}.jsonl"
                dst = f"{a.out}/attack/{atk}{a.attacker_tag}/{name}/{a.split}/{tag}.jsonl"
                if os.path.exists(src) and not os.path.exists(dst):
                    jobs.append((atk, src, dst))
    if not jobs:
        print("all done"); return

    kw = dict(model=a.attacker, trust_remote_code=True, max_model_len=4096, gpu_memory_utilization=0.85, seed=0)
    if "Qwen3.8" in a.attacker:
        kw["limit_mm_per_prompt"] = {"image": 0, "video": 0}
        kw["max_num_seqs"] = 256
    llm = LLM(**kw)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=1200, seed=0)


    def run(instr, texts):
        ps = [tok.apply_chat_template([{"role": "user", "content": instr + t}], tokenize=False,
                                      add_generation_prompt=True, enable_thinking=False) for t in texts]
        return [o.outputs[0].text.strip() for o in llm.generate(ps, sp)]


    for atk, src, dst in jobs:
        rows = [json.loads(l) for l in open(src, encoding="utf-8")]
        if a.n:
            rows = rows[: a.n]
        texts = [r["text"] for r in rows]
        if atk == "backtrans":
            out = run(P["to_ko"], run(P["to_en"], texts))
        else:
            out = run(P[atk], texts)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            for r, t in zip(rows, out):
                f.write(json.dumps({"id": r["id"], "text": t}, ensure_ascii=False) + "\n")
        print(f"{dst}: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
