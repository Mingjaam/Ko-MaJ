"""Revision evaluation from det_rev z-scores (CPU only).

Protocol
- Human null split once (seed 0) into calibration (1500) and held-out test (1500) halves.
- Threshold = 99th percentile of the CALIBRATION human z at the same length, per detector/key/defence.
- Reported at that threshold: achieved FPR on held-out human (Clopper-Pearson 95% CI), FPR on the
  unwatermarked model output under the same attack ("none"), and for watermarked output both
  TPR_elig (among texts that reach L canonical syllables) and TPR_all (= detected / all 500 prompts;
  too-short or missing outputs count as misses), with n_all, n_elig, n_det. TPR_elig is null when no output
  reaches the length (e.g. compatibility-jamo text has no Hangul syllables: detection cannot run).
- Paired bootstrap (2000 resamples of prompts) of TPR_all differences between methods.
- Quality: oracle PPL (median over prompts, paired per-prompt differences vs none), trigram repetition,
  Hangul ratio; matched-quality: linear interpolation of the attacked TPR_elig of Ko-MaJ (delta 2..3) at the
  KGW delta=2 quality cost on the test dPPL axis (an estimate, not a newly generated operating point).
Writes ../out/rev.json
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.stats import beta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import syllables  # noqa: E402

OUT = os.environ.get("KOMAJ_OUT", os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."), "out"))
rng0 = np.random.default_rng(0)
PERM = {}


def split_idx(n):
    if n not in PERM:
        p = np.random.default_rng(0).permutation(n)
        PERM[n] = (p[: n // 2], p[n // 2:])
    return PERM[n]


def cp(k, n):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def boot_diff(a, b, B=2000):
    """a, b: per-prompt 0/1 detection (misses = 0). Returns diff and 95% CI of mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    idx = rng0.integers(0, n, size=(B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return [float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def det_key(tag, dk_list):
    """Which human/none key belongs to this tag."""
    if tag in ("synthid", "luna", "stela"):
        return tag
    from det_rev import parse_tag
    det, key = parse_tag(tag)
    return f"{det}/{key}"


R = {"protocol": __doc__, "det": {}, "hits": {}, "quality": {}, "paired": {}}
files = sorted(glob.glob(f"{OUT}/det_rev/*/*/*.json"))
for f in files:
    d = json.load(open(f))
    m, sp, atk, dfn = d["model"], d["split"], d["attack"], d["defense"]
    L = d["lens"]
    for tag, Z in d["z"].items():
        dk = det_key(tag, d["human_fp"])
        Zh = np.array(d["human_fp"][dk], float)
        cal, tst = split_idx(len(Zh))
        Zn = np.array(d["none_z"][dk], float) if dk in d["none_z"] else None
        Z = np.array(Z, float)
        out = {}
        hits = {}
        for j, LL in enumerate(L):
            thr = float(np.nanpercentile(Zh[cal, j], 99))
            ht = Zh[tst, j][~np.isnan(Zh[tst, j])]
            v = Z[:, j]
            elig = ~np.isnan(v)
            det = elig & (np.nan_to_num(v, nan=-1e9) > thr)
            r = {"thr": thr, "n_all": int(len(v)), "n_elig": int(elig.sum()), "n_det": int(det.sum()),
                 "tpr_elig": (float(det.sum() / elig.sum()) if elig.sum() else None), "tpr_all": float(det.mean()),
                 "fpr_human_test": float((ht > thr).mean()) if len(ht) else None, "n_human_test": int(len(ht)),
                 "fpr_human_ci": cp(int((ht > thr).sum()), len(ht)),
                 "human_z4": [int((ht > 4).sum()), int(len(ht))]}
            if Zn is not None:
                vn = Zn[:, j][~np.isnan(Zn[:, j])]
                r.update({"fpr_none": float((vn > thr).mean()) if len(vn) else None, "n_none": int(len(vn)),
                          "fpr_none_ci": cp(int((vn > thr).sum()), len(vn))})
            out[str(LL)] = r
            hits[str(LL)] = det.astype(int).tolist()
        R["det"].setdefault(sp, {}).setdefault(m, {}).setdefault(atk, {}).setdefault(dfn, {})[tag] = out
        R["hits"].setdefault(sp, {}).setdefault(m, {}).setdefault(atk, {}).setdefault(dfn, {})[tag] = hits

# paired comparisons at L=200: ours vs each baseline (undefended and each defence), same prompts
OURS = ["han_d3", "han_d2.5", "han2r_d3"]
OTHERS = ["kgw_d2", "kgw_d3", "unigram_d2", "synthid", "luna", "stela"]
for sp, MM in R["hits"].items():
    for m, AA in MM.items():
        for atk, DD in AA.items():
            if "none" not in DD:
                continue
            for o in OURS:
                if o not in DD["none"]:
                    continue
                a = DD["none"][o]["200"]
                for dfn, TT in DD.items():
                    for b in OTHERS:
                        if b in TT:
                            R["paired"].setdefault(sp, {}).setdefault(m, {}).setdefault(atk, {})[f"{o}|{b}|{dfn}"] = \
                                boot_diff(a, TT[b]["200"])


# ---- quality
def ppl_map(m, sp, tag):
    p = f"{OUT}/ppl_main/{m}_{sp}_{tag}.json"
    if not os.path.exists(p):
        return None
    return {r["id"]: r["ppl"] for r in json.load(open(p)) if r["n"] > 0 and not math.isnan(r["ppl"])}


def gen_stats(m, sp, tag):
    p = f"{OUT}/gen_main/{m}/{sp}/{tag}.jsonl"
    if not os.path.exists(p):
        return None
    rep, hr, ns = [], [], []
    for l in open(p, encoding="utf-8"):
        t = json.loads(l)["text"]
        s = syllables(t)
        tri = list(zip(s, s[1:], s[2:]))
        rep.append(1 - len(set(tri)) / max(1, len(tri)))
        hr.append(len(s) / max(1, len(t.replace(" ", "").replace("\n", ""))))
        ns.append(len(s))
    return {"rep": float(np.mean(rep)), "hangul": float(np.mean(hr)), "reach200": float(np.mean(np.array(ns) >= 200))}


for sp in ["test", "news_test"]:
    for m in ["qwen25_7b", "exaone_8b", "kanana_8b", "qwen38_27b"]:
        base = ppl_map(m, sp, "none")
        if base is None:
            continue
        tags = sorted(os.path.basename(p)[:-6] for p in glob.glob(f"{OUT}/gen_main/{m}/{sp}/*.jsonl"))
        Q = {}
        for t in tags:
            pm = ppl_map(m, sp, t)
            st = gen_stats(m, sp, t)
            q = dict(st or {})
            if pm:
                q["ppl"] = float(np.median(list(pm.values())))
                q["dppl"] = q["ppl"] - float(np.median(list(base.values())))
                common = [k for k in pm if k in base]
                diff = np.array([pm[k] - base[k] for k in common])
                idx = rng0.integers(0, len(diff), size=(2000, len(diff)))
                med = np.median(diff[idx], axis=1)
                q["paired_dppl"] = [float(np.median(diff)), float(np.percentile(med, 2.5)), float(np.percentile(med, 97.5))]
            Q[t] = q
        R["quality"].setdefault(sp, {})[m] = Q

# ---- matched quality: interpolate Ko-MaJ (h1) attacked TPR_elig at KGW d2's quality cost (dppl)
R["matched"] = {}
for m, Q in R["quality"].get("test", {}).items():
    if "kgw_d2" not in Q or "dppl" not in Q["kgw_d2"]:
        continue
    target = Q["kgw_d2"]["dppl"]
    pts = sorted((Q[t]["dppl"], t) for t in ["han_d1.5", "han_d2", "han_d2.5", "han_d3", "han_d4"] if t in Q and "dppl" in Q[t])
    res = {"target_dppl": target, "points": pts}
    for atk in ["clean", "nospace", "style", "paraphrase", "backtrans"]:
        D = R["det"]["test"][m].get(atk, {}).get("none", {})
        xy = [(x, D[t]["200"]["tpr_elig"]) for x, t in pts if t in D]
        if len(xy) < 2 or "kgw_d2" not in D:
            continue
        xs, ys = zip(*xy)
        inside = xs[0] <= target <= xs[-1]
        res[atk] = {"ours_interp": float(np.interp(target, xs, ys)) if inside else None,
                    "ours_points": xy, "kgw_d2": D["kgw_d2"]["200"]["tpr_elig"], "extrapolated": not inside}
    R["matched"][m] = res

json.dump(R, open(f"{OUT}/rev.json", "w"))
print("written", f"{OUT}/rev.json", len(files), "files")
