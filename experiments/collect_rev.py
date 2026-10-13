"""Collect every revision number the paper / slides / page need into ../out/final2.json.

Source: out/rev.json (eval_rev.py) + judge + implementation stats + survival/tok from the old final.json.
Main metric: TPR at the calibrated 1% threshold on the first 200 canonical syllables, among outputs that
reach 200 syllables ("elig"); "all" divides by all 500 prompts. FPR is measured on held-out human text.
"""
import glob
import json
import os

import numpy as np

OUT = os.environ.get("KOMAJ_OUT", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "out"))
R = json.load(open(f"{OUT}/rev.json"))
OLD = json.load(open(f"{OUT}/final.json"))
MODELS = ["qwen25_7b", "exaone_8b", "kanana_8b", "qwen38_27b"]
BASE = ["synthid", "luna", "stela"]
TAGS = ["han_d3", "han_d2.5", "han_d2", "han2_d3", "han2r_d3", "kgw_d2", "kgw_d3", "unigram_d2"] + BASE
CHR = ["clean", "nospace", "randspace", "zwsp", "nfd", "nfdzw"]
EDIT = ["sub10", "sub20", "del10", "del20", "excerpt", "mix", "jamo"]
LLM = ["style", "paraphrase", "backtrans", "paraphrase_exaone"]
D = R["det"]


def cell(sp, m, atk, dfn, tag, L="200"):
    try:
        return D[sp][m][atk][dfn][tag][L]
    except KeyError:
        return None


INVALID = []  # cells where the detector also flags unwatermarked text under the same attack (FPR > 20%)


def val(sp, m, atk, dfn, tag, k="tpr_elig", L="200"):
    c = cell(sp, m, atk, dfn, tag, L)
    if c is None:
        return None
    if c.get("fpr_none") is not None and c["fpr_none"] > 0.2:
        if (sp, m, atk, dfn, tag) not in [tuple(x[:5]) for x in INVALID]:
            INVALID.append([sp, m, atk, dfn, tag, round(c[k], 3), round(c["fpr_none"], 3)])
        return None
    return None if c[k] is None else round(c[k], 4)


F = {"models": MODELS, "main": {}, "all": {}, "nelig": {}, "defended": {}, "length": {}, "fpr": {}, "quality": {},
     "judge": {}, "meaning": {}, "matched": R.get("matched", {}), "keys": {}, "news": {}, "cross": {}, "paired": {},
     "impl": json.load(open(f"{OUT}/stats_impl.json")), "tok": OLD.get("tok", {}), "survival": OLD.get("survival", {}),
     "cost": OLD.get("cost"), "speed": OLD.get("speed")}
for m in MODELS:
    F["main"][m], F["all"][m], F["nelig"][m], F["defended"][m], F["length"][m], F["fpr"][m] = {}, {}, {}, {}, {}, {}
    for t in TAGS:
        F["main"][m][t] = {a: val("test", m, a, "none", t) for a in CHR + EDIT + LLM}
        F["all"][m][t] = {a: val("test", m, a, "none", t, "tpr_all") for a in CHR + EDIT + LLM}
        F["nelig"][m][t] = {a: (cell("test", m, a, "none", t) or {}).get("n_elig") for a in CHR + EDIT + LLM}
        F["defended"][m][t] = {d: {a: val("test", m, a, d, t) for a in CHR} for d in ["none", "norm", "space"]}
        lens = ["50", "100", "150", "200", "300"]
        F["length"][m][t] = [val("test", m, "clean", "none", t, L=L) for L in lens]
        c = cell("test", m, "clean", "none", t)
        if c:
            F["fpr"][m][t] = {"human": c["fpr_human_test"], "human_ci": c["fpr_human_ci"], "n_human": c["n_human_test"],
                              "none": c.get("fpr_none"), "none_ci": c.get("fpr_none_ci"), "n_none": c.get("n_none"),
                              "thr": c["thr"]}
    F["quality"][m] = R["quality"].get("test", {}).get(m, {})
    # quality-detection points (no attack): every strength of each family
    import re
    F.setdefault("pareto", {})[m] = {}
    for fam in ["han", "kgw", "unigram"]:
        pts = []
        for t, q in F["quality"][m].items():
            if re.fullmatch(fam + r"_d[\d.]+", t) and q.get("dppl") is not None:
                v100, v200 = val("test", m, "clean", "none", t, L="100"), val("test", m, "clean", "none", t)
                if v200 is not None:
                    pts.append({"tag": t, "dppl": round(q["dppl"], 3), "tpr100": v100, "tpr200": v200})
        F["pareto"][m][fam] = sorted(pts, key=lambda d: d["dppl"])
    F["paired"][m] = R["paired"].get("test", {}).get(m, {})
    # cross-tokenizer: watermarked by m, detected with another model's tokenizer
    F["cross"][m] = {}
    for f in glob.glob(f"{OUT}/det_rev/{m}/test/cross_*__none.json"):
        d = os.path.basename(f)[len("cross_"):-len("__none.json")]
        F["cross"][m][d] = {t: val("test", m, f"cross_{d}", "none", t) for t in ["han_d3", "han_d2.5", "kgw_d2", "unigram_d2"]}
    # other keys
    F["keys"][m] = {}
    for k in ["", "_k2", "_k3"]:
        F["keys"][m][k or "_k1"] = {t: val("test", m, "clean", "none", t + k) for t in ["han_d3", "han_d2.5", "kgw_d2", "unigram_d2"]}
        F["keys"][m][(k or "_k1") + "_nospace"] = {t: val("test", m, "nospace", "none", t + k) for t in ["han_d3", "han_d2.5", "kgw_d2", "unigram_d2"]}
        F["keys"][m][(k or "_k1") + "_fpr"] = {t: (cell("test", m, "clean", "none", t + k) or {}).get("fpr_human_test")
                                              for t in ["han_d3", "han_d2.5", "kgw_d2", "unigram_d2"]}
    # news domain
    F["news"][m] = {t: {a: val("news_test", m, a, "none", t) for a in ["clean", "nospace", "randspace", "zwsp"]}
                    for t in ["han_d3", "han_d2.5", "han2r_d3", "kgw_d2", "unigram_d2"]}
    F["news"][m]["_fpr"] = {t: (cell("news_test", m, "clean", "none", t) or {}).get("fpr_human_test")
                            for t in ["han_d3", "kgw_d2", "unigram_d2"]}
    F["news"][m]["_quality"] = R["quality"].get("news_test", {}).get(m, {})


# LLM judge
def jmean(p, keys):
    if not os.path.exists(p):
        return None
    rows = json.load(open(p))
    return {k: round(float(np.mean([r[k] for r in rows if r[k] is not None])), 3) for k in keys} | \
           {"parsed": sum(r[keys[0]] is not None for r in rows), "n": len(rows)}


for sp in ["test", "news_test"]:
    for m in MODELS:
        for p in glob.glob(f"{OUT}/judge/quality/{m}/{sp}/*.json"):
            F["judge"].setdefault(sp, {}).setdefault(m, {})[os.path.basename(p)[:-5]] = jmean(p, ["fluency", "coherence", "relevance"])
for m in MODELS:
    for p in glob.glob(f"{OUT}/judge/meaning/{m}/test/*.json"):
        tag, atk = os.path.basename(p)[:-5].split("__")
        F["meaning"].setdefault(m, {}).setdefault(tag, {})[atk] = jmean(p, ["meaning"])

# null histogram of Ko-MaJ human z at 200 syllables (model independent) and KGW (Qwen2.5 tokenizer)
for f in glob.glob(f"{OUT}/det_rev/human/human-*_hanmark_15485863_any_none.npy"):
    z = np.load(f)[:, 3]
    z = z[~np.isnan(z)]
    F["null"] = {"n": int(len(z)), "mean": float(z.mean()), "std": float(z.std()), "max": float(z.max()),
                 "gt4": int((z > 4).sum()), "hist": np.histogram(z, bins=np.arange(-4, 4.01, 0.5))[0].tolist()}
F["invalid"] = INVALID
json.dump(F, open(f"{OUT}/final2.json", "w"), indent=1)
print("written final2.json")
