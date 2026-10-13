"""Revision detection pass for the official LUNA / SynthID / STELA detectors (GPU: their RNG is device-bound).

Same output layout as det_rev.py: out/det_rev/{name}/{split}/{attack}__{defense}__{alg}.json with z for the
watermarked texts, z for the unwatermarked model output under the same attack (none_z), and the human null.
usage: python detbl_rev.py --alg STELA --model Qwen/Qwen2.5-7B-Instruct --name qwen25_7b --attacks clean,nospace
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
from transformers import AutoConfig, AutoTokenizer

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import defense  # noqa: E402
from attacks import ATTACKS, NEEDS_REF  # noqa: E402
from common import cut_at_syllables  # noqa: E402
import baseline_hf  # noqa: E402

LLM_ATTACKS = {"style", "paraphrase", "backtrans"}
OUT = os.environ.get("KOMAJ_OUT", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "out"))
DATA = os.environ.get("KOMAJ_DATA", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--attacks", default="clean")
    ap.add_argument("--defenses", default="none")
    ap.add_argument("--lens", default="50,100,150,200,300")
    a = ap.parse_args()
    LENS = [int(x) for x in a.lens.split(",")]
    alg = a.alg.lower()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    V = AutoConfig.from_pretrained(a.model, trust_remote_code=True).vocab_size

    class _Stub:
        class config:
            vocab_size = V

    wm = baseline_hf.load_wm(a.alg, _Stub(), tok, device="cuda")
    hfile = f"{DATA}/human.jsonl"
    human = [json.loads(l)["text"] for l in open(hfile, encoding="utf-8")]
    hhash = hashlib.md5(open(hfile, "rb").read()).hexdigest()[:10]
    prompts = [json.loads(l) for l in open(f"{DATA}/{a.split}.jsonl", encoding="utf-8")]
    ids = [r["id"] for r in prompts]
    ref = {r["id"]: r.get("human", "") for r in prompts}

    def zmat(texts, dfn):
        Z = np.full((len(texts), len(LENS)), np.nan)
        for i, t in enumerate(texts):
            if t is None:
                continue
            for j, L in enumerate(LENS):
                c = cut_at_syllables(t, L)
                if c is not None:
                    Z[i, j] = float(wm.detect_watermark(defense.apply(c, dfn))["score"])
        return Z

    def human_z(dfn):
        fp = f"human-{hhash}_n{len(human)}_L{'-'.join(map(str, LENS))}_{alg}_{a.name}_{dfn}"
        p = f"{OUT}/det_rev/human/{fp}.npy"
        if os.path.exists(p):
            return np.load(p)
        Z = zmat(human, dfn)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p + ".tmp.npy", Z)
        os.replace(p + ".tmp.npy", p)
        return Z

    def load(tag, atk):
        src = (f"{OUT}/attack/{atk}/{a.name}/{a.split}/{tag}.jsonl" if atk in LLM_ATTACKS
               else f"{OUT}/gen_main/{a.name}/{a.split}/{tag}.jsonl")
        if not os.path.exists(src):
            return None
        d = {json.loads(l)["id"]: json.loads(l)["text"] for l in open(src, encoding="utf-8")}
        out = []
        for i, k in enumerate(ids):
            t = d.get(k)
            if t is not None and atk not in LLM_ATTACKS and atk != "clean":
                f = ATTACKS[atk]
                t = f(t, seed=i, ref=ref[k]) if atk in NEEDS_REF else f(t, seed=i)
            out.append(t)
        return out

    for atk in a.attacks.split(","):
        for dfn in a.defenses.split(","):
            dst = f"{OUT}/det_rev/{a.name}/{a.split}/{atk}__{dfn}__{alg}.json"
            if os.path.exists(dst):
                continue
            texts = load(alg, atk)
            if texts is None:
                continue
            t0 = time.time()
            res = {"model": a.name, "split": a.split, "attack": atk, "defense": dfn, "lens": LENS, "ids": ids,
                   "z": {alg: zmat(texts, dfn).tolist()}, "none_z": {}, "human_fp": {alg: human_z(dfn).tolist()}}
            nt = load("none", atk)
            if nt is not None:
                res["none_z"][alg] = zmat(nt, dfn).tolist()
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(res, open(dst + ".tmp", "w"))
            os.replace(dst + ".tmp", dst)
            print(f"{dst}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
