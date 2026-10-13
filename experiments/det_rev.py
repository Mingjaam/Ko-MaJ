"""Revision detection pass: z-scores only (thresholds, FPR and CIs are computed later in eval_rev.py).

For one model/split, every attack x defence x tag: z[i, j] for sample i (aligned to the prompt list) cut at
LENS[j] canonical syllables (NaN if the text is shorter). Also the human null z for every detector/key/
defence, cached under a fingerprint of (human file hash, n, lens, detector, key, tokenizer, defence).
usage: python det_rev.py --name qwen25_7b --tok Qwen/Qwen2.5-7B-Instruct --split test --attacks clean,nospace
"""
import argparse
import glob
import hashlib
import json
import os
import time

import numpy as np
from transformers import AutoTokenizer

import defense
from attacks import ATTACKS, NEEDS_REF
from common import cut_at_syllables
from schemes import detect_hanmark, detect_hanmark2, detect_kgw, detect_unigram

KEYS = {"": 15485863, "k2": 104729, "k3": 7919}
DET = {"han": "hanmark", "han2": "hanmark2", "han2r": "hanmark2r", "kgw": "kgw", "unigram": "unigram"}
LLM_ATTACKS = {"style", "paraphrase", "backtrans", "paraphrase_exaone"}
TOKEN_DETS = {"kgw", "unigram"}


def parse_tag(tag):
    """'han_d3' -> ('hanmark', key0); 'kgw_d2_k2' -> ('kgw', key2); 'none' -> (None, None)."""
    if tag == "none":
        return None, None
    parts = tag.split("_")
    k = parts[-1] if parts[-1] in KEYS else ""
    return DET[parts[0]], KEYS[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--human", default="", help="default: human.jsonl (wiki) / news_human.jsonl for news splits")
    ap.add_argument("--attacks", default="clean")
    ap.add_argument("--defenses", default="none")
    ap.add_argument("--tags", default="", help="comma list (default: every file with a known scheme)")
    ap.add_argument("--lens", default="50,100,150,200,300")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--out", default="../out")
    ap.add_argument("--label", default="", help="output stem instead of the attack name (e.g. cross_exaone)")
    a = ap.parse_args()
    LENS = [int(x) for x in a.lens.split(",")]
    tok = AutoTokenizer.from_pretrained(a.tok, trust_remote_code=True)
    hfile = a.human or ("../data/news_human.jsonl" if a.split.startswith("news") else "../data/human.jsonl")
    human = [json.loads(l)["text"] for l in open(hfile, encoding="utf-8")]
    hhash = hashlib.md5(open(hfile, "rb").read()).hexdigest()[:10]
    prompts = [json.loads(l) for l in open(f"../data/{a.split}.jsonl", encoding="utf-8")]
    ids = [r["id"] for r in prompts]
    ref = {r["id"]: r.get("human", "") for r in prompts}
    gdir = f"{a.out}/gen_main/{a.name}/{a.split}"
    tags = sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{gdir}/*.jsonl"))
    tags = [t for t in tags if t == "none" or t.split("_")[0] in DET]
    if a.tags:
        tags = [t for t in tags if t in a.tags.split(",")]

    def score(text, det, key, dfn):
        if det == "hanmark":
            return detect_hanmark(text, key, a.gamma)[0]
        if det in ("hanmark2", "hanmark2r"):
            return detect_hanmark2(text, key, a.gamma, skip=det == "hanmark2r")[0]
        ids_ = tok.encode(defense.apply(text, dfn), add_special_tokens=False)
        return (detect_kgw if det == "kgw" else detect_unigram)(ids_, key, a.gamma)[0]

    def zmat(texts, det, key, dfn):
        Z = np.full((len(texts), len(LENS)), np.nan)
        for i, t in enumerate(texts):
            if t is None:
                continue
            for j, L in enumerate(LENS):
                c = cut_at_syllables(t, L)
                if c is not None:
                    Z[i, j] = score(c, det, key, dfn)
        return Z

    def human_z(det, key, dfn):
        tk = "any" if det not in TOKEN_DETS else os.path.basename(a.tok.rstrip("/"))
        dfn_ = "none" if det not in TOKEN_DETS else dfn
        fp = f"{os.path.basename(hfile)[:-6]}-{hhash}_n{len(human)}_L{'-'.join(map(str, LENS))}_{det}_{key}_{tk}_{dfn_}"
        p = f"{a.out}/det_rev/human/{fp}.npy"
        if os.path.exists(p):
            return np.load(p)
        Z = zmat(human, det, key, dfn_)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        np.save(p + ".tmp.npy", Z)
        os.replace(p + ".tmp.npy", p)
        return Z

    def load(tag, atk):
        """Texts aligned to `ids` (None where missing)."""
        if atk in LLM_ATTACKS:
            src = f"{a.out}/attack/{atk}/{a.name}/{a.split}/{tag}.jsonl"
        else:
            src = f"{gdir}/{tag}.jsonl"
        if not os.path.exists(src):
            return None
        d = {}
        for l in open(src, encoding="utf-8"):
            r = json.loads(l)
            d[r["id"]] = r["text"]
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
            dst = f"{a.out}/det_rev/{a.name}/{a.split}/{a.label or atk}__{dfn}.json"
            t0 = time.time()
            res = {"model": a.name, "split": a.split, "attack": a.label or atk, "tok": a.tok, "defense": dfn, "lens": LENS, "ids": ids,
                   "z": {}, "none_z": {}, "human_fp": {}}
            if os.path.exists(dst):  # merge: keep tags scored by an earlier run
                old = json.load(open(dst))
                if old["lens"] == LENS and old["ids"] == ids:
                    for k in ("z", "none_z", "human_fp"):
                        res[k].update(old[k])
            dets_needed = set()
            for tag in tags:
                det, key = parse_tag(tag)
                if det is None or (dfn != "none" and det not in TOKEN_DETS):
                    continue
                texts = load(tag, atk)
                if texts is None:
                    continue
                res["z"][tag] = zmat(texts, det, key, dfn).tolist()
                dets_needed.add((det, key))
            if not res["z"]:
                continue
            none_t = load("none", atk) if "none" in tags else None
            for det, key in sorted(dets_needed):
                dk = f"{det}/{key}"
                res["human_fp"][dk] = human_z(det, key, dfn).tolist()
                if none_t is not None:
                    res["none_z"][dk] = zmat(none_t, det, key, dfn).tolist()
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            json.dump(res, open(dst + ".tmp", "w"))
            os.replace(dst + ".tmp", dst)
            print(f"{dst}: {len(res['z'])} tags, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
