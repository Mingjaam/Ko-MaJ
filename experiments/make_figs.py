"""Paper figures from build/final.json -> build/fig_overview.png, build/fig_pareto.png (single-column width)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import u01_int  # noqa: E402

B = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "build")
F = json.load(open(os.path.join(B, "final2.json"), encoding="utf-8"))
plt.rcParams.update({"font.family": "Malgun Gothic", "font.size": 7.2, "axes.unicode_minus": False,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
NAVY, GREEN, RED, GREY, LIGHT = "#1f3b73", "#2f7d55", "#b4575a", "#8a949c", "#eef2f7"
NAME = {"qwen25_7b": "Qwen2.5-7B", "exaone_8b": "EXAONE-3.5-7.8B", "kanana_8b": "Kanana-1.5-8B", "qwen38_27b": "Qwen3.8-27B"}


def box(ax, x, y, w, h, title, sub, fc=LIGHT, ec=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.025", fc=fc, ec=ec, lw=0.8))
    ax.text(x + w / 2, y + h * 0.74, title, ha="center", va="center", fontsize=6.6, weight="bold", color=NAVY)
    ax.text(x + w / 2, y + h * 0.33, sub, ha="center", va="center", fontsize=5.3, color="#333", linespacing=1.15)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.8, color=NAVY))


def overview():
    fig, ax = plt.subplots(figsize=(2.9, 2.75), dpi=400)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.0, 0.975, "(a) Generation", fontsize=7.4, weight="bold", color=NAVY, va="top")
    w, h = 0.30, 0.22
    xs = [0.0, 0.35, 0.70]
    box(ax, xs[0], 0.68, w, h, "LLM logits", "next-token\ncandidates v")
    box(ax, xs[1], 0.68, w, h, "Syllables S(v)", "completed by v;\nbyte fragment: none")
    box(ax, xs[2], 0.68, w, h, "Bias", "δ·g(v)/|S(v)|,\ng = # green pairs")
    arrow(ax, xs[0] + w, 0.79, xs[1], 0.79); arrow(ax, xs[1] + w, 0.79, xs[2], 0.79)
    ax.text(0.0, 0.575, "(b) Detection (key only: no model, no tokenizer)", fontsize=7.4, weight="bold", color=NAVY, va="top")
    box(ax, xs[0], 0.27, w, h, "Normalize C(x)", "drop format chars, NFC;\nkeep Hangul syllables")
    box(ax, xs[1], 0.27, w, h, "Pairs", "green test (key K);\nrepeats counted once")
    box(ax, xs[2], 0.27, w, h, "z-score", "z > threshold\n→ watermarked")
    arrow(ax, xs[0] + w, 0.38, xs[1], 0.38); arrow(ax, xs[1] + w, 0.38, xs[2], 0.38)
    # colours computed with the real experiment key (common.u01_int), not hand-picked
    s = [ord(c) - 0xAC00 for c in "대한민국의수도는"]
    ex = [(chr(a + 0xAC00) + chr(b + 0xAC00), u01_int(15485863, a, b) < 0.25) for a, b in zip(s, s[1:])]
    x = 0.0
    for s, g in ex:
        ax.add_patch(FancyBboxPatch((x, 0.075), 0.128, 0.11, boxstyle="round,pad=0.004,rounding_size=0.02",
                                    fc="#d6eee0" if g else "#f4dede", ec="none"))
        ax.text(x + 0.064, 0.13, s, ha="center", va="center", fontsize=6.6, weight="bold", color=GREEN if g else RED)
        x += 0.1428
    ax.text(0.0, 0.015, "\"대한민국의 수도는\" → syllable pairs; spacing never enters the test",
            fontsize=5.4, color="#444")
    fig.savefig(os.path.join(B, "fig_overview.png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def pareto():
    fig, axs = plt.subplots(2, 2, figsize=(2.8, 2.55), dpi=400, sharey=True)
    for ax, m in zip(axs.flat, F["models"]):
        P = F["pareto"][m]
        for fam, col, lab, lw in [("unigram", GREY, "Unigram", 0.9), ("kgw", RED, "KGW", 0.9), ("han", GREEN, "Ko-MaJ", 1.4)]:
            pts = [(d["dppl"], d["tpr200"] * 100) for d in P.get(fam, []) if d["tpr200"] is not None]
            if pts:
                ax.plot(*zip(*pts), "-o", color=col, lw=lw, ms=2.2, label=lab)
        ax.set_title(NAME[m], fontsize=6.6, pad=2)
        ax.set_ylim(0, 105); ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=5.6, length=2, pad=1)
    for ax in axs[1]:
        ax.set_xlabel("ΔPPL (quality loss)", fontsize=6.0, labelpad=1)
    for ax in axs[:, 0]:
        ax.set_ylabel("TPR (%), 200 syl.", fontsize=6.0, labelpad=1)
    axs[0, 0].legend(fontsize=5.4, frameon=False, loc="lower right", handlelength=1.2)
    fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.3)
    fig.savefig(os.path.join(B, "fig_pareto.png"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


COLS = {"han_d3": (GREEN, "Ko-MaJ (proposed)"), "kgw_d2": (RED, "KGW"), "unigram_d2": (GREY, "Unigram"),
        "synthid": ("#eb6834", "SynthID"), "luna": ("#7a5ac8", "LUNA"), "stela": ("#3a8fb7", "STELA")}
SH = {"qwen25_7b": "Qwen2.5", "exaone_8b": "EXAONE", "kanana_8b": "Kanana", "qwen38_27b": "Qwen3.8"}


def tp(m, t, a):
    r = F["main"][m].get(t) or {}
    return None if r.get(a) is None else r[a] * 100


def save(fig, name):
    fig.savefig(os.path.join(B, name), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def grouped(ax, groups, series, ylab):
    import numpy as np
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for k, (lab, col, vals) in enumerate(series):
        vv = [0 if v is None else v for v in vals]
        bars = ax.bar(x - 0.4 + w * (k + 0.5), vv, w * 0.92, color=col, label=lab)
        for b, v in zip(bars, vals):
            if v is None:
                b.set_alpha(0)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 105)
    ax.set_ylabel(ylab, fontsize=6.2, labelpad=1)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.tick_params(labelsize=5.8, length=2, pad=1)


def nospace():
    fig, ax = plt.subplots(figsize=(2.9, 1.75), dpi=400)
    meths = ["han_d3", "kgw_d2", "unigram_d2", "synthid", "luna", "stela"]
    grouped(ax, [SH[m] for m in F["models"]], [(COLS[t][1], COLS[t][0], [tp(m, t, "nospace") for m in F["models"]]) for t in meths],
            "TPR after removing spaces (%)")
    ax.legend(fontsize=4.9, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.30), handlelength=1.0, columnspacing=0.8)
    save(fig, "fig_nospace.png")


def length():
    fig, axs = plt.subplots(1, 4, figsize=(2.9, 1.25), dpi=400, sharey=True)
    L = [50, 100, 150, 200, 300]
    for ax, m in zip(axs, F["models"]):
        for t in ["unigram_d2", "kgw_d2", "han_d3"]:
            ys = [None if q is None else q * 100 for q in F["length"][m][t]]
            ax.plot(L, ys, "-o", color=COLS[t][0], lw=1.3 if t == "han_d3" else 0.8, ms=1.8, label=COLS[t][1])
        ax.set_title(SH[m], fontsize=6.0, pad=2)
        ax.set_ylim(0, 105); ax.set_xticks([100, 300]); ax.grid(alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=5.4, length=2, pad=1)
    axs[0].set_ylabel("TPR (%)", fontsize=6.0, labelpad=1)
    fig.text(0.5, -0.04, "detected length (syllables)", ha="center", fontsize=6.0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h[::-1], l[::-1], fontsize=5.0, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10), handlelength=1.2)
    fig.tight_layout(pad=0.2, w_pad=0.3)
    save(fig, "fig_length.png")


def null():
    import numpy as np
    fig, ax = plt.subplots(figsize=(2.9, 1.45), dpi=400)
    edges = np.arange(-4, 4.01, 0.5)
    c = (edges[:-1] + edges[1:]) / 2
    nl = F["null"]
    ax.bar(c, nl["hist"], 0.42, color=GREEN, label=f"Ko-MaJ, {nl['n']:,} human texts")
    zz = np.linspace(-4, 4, 200)
    ax.plot(zz, nl["n"] * 0.5 * np.exp(-zz ** 2 / 2) / np.sqrt(2 * np.pi), color=NAVY, lw=0.9, label="N(0,1)")
    ax.set_xlabel("z-score (200 syllables)", fontsize=6.0, labelpad=1); ax.set_ylabel("# texts", fontsize=6.0, labelpad=1)
    ax.tick_params(labelsize=5.6, length=2, pad=1); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=5.0, frameon=False, loc="upper left")
    save(fig, "fig_null.png")


def cross():
    import numpy as np
    fig, ax = plt.subplots(figsize=(2.9, 1.5), dpi=400)
    src = list(F["models"])
    x = np.arange(len(src))
    for k, (t, col, lab) in enumerate([("han_d3", GREEN, "Ko-MaJ (proposed)"), ("kgw_d2", RED, "KGW"), ("unigram_d2", GREY, "Unigram")]):
        hi = []
        for m in src:
            v = [d[t] * 100 for d in F["cross"][m].values() if d.get(t) is not None]
            hi.append(max(v) if v else 0)
        ax.bar(x - 0.27 + 0.27 * k, hi, 0.25, color=col, label=lab)
    ax.set_xticks(x, [SH[m] for m in src]); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_ylabel("TPR (%), best of 3\nother tokenizers", fontsize=5.8, labelpad=1)
    ax.set_xlabel("generating model", fontsize=5.8, labelpad=1)
    ax.tick_params(labelsize=5.8, length=2, pad=1)
    ax.legend(fontsize=5.0, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=1.0)
    save(fig, "fig_cross.png")


def edit():
    fig, axs = plt.subplots(2, 2, figsize=(2.9, 2.4), dpi=400, sharey=True)
    atts = ["sub10", "sub20", "del10", "del20", "excerpt", "mix"]
    for ax, m in zip(axs.flat, F["models"]):
        grouped(ax, atts, [(COLS[t][1], COLS[t][0], [tp(m, t, a) for a in atts]) for t in ["han_d3", "kgw_d2", "unigram_d2"]], "")
        ax.set_title(NAME[m], fontsize=6.0, pad=2); ax.tick_params(axis="x", labelsize=4.6, rotation=30)
    for ax in axs[:, 0]:
        ax.set_ylabel("TPR (%)", fontsize=6.0, labelpad=1)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.0, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05), handlelength=1.0)
    fig.tight_layout(pad=0.2, h_pad=0.5, w_pad=0.3)
    save(fig, "fig_edit.png")


def llm():
    """Attacked TPR at equal quality cost: KGW d2 (bar) vs Ko-MaJ interpolated at the same dPPL (dot)."""
    import numpy as np
    fig, axs = plt.subplots(1, 4, figsize=(2.9, 1.35), dpi=400, sharey=True)
    atts = [("clean", "none"), ("style", "style"), ("paraphrase", "para."), ("backtrans", "back.")]
    for ax, m in zip(axs, F["models"]):
        M_ = F["matched"].get(m, {})
        x = np.arange(len(atts))
        kg = [None if a not in M_ else M_[a]["kgw_d2"] * 100 for a, _ in atts]
        ou = [None if a not in M_ or M_[a]["ours_interp"] is None else M_[a]["ours_interp"] * 100 for a, _ in atts]
        ax.bar(x, [v or 0 for v in kg], 0.6, color=RED, label="KGW δ=2")
        ax.plot(x, [np.nan if v is None else v for v in ou], "o", color=GREEN, ms=3.2, label="Ko-MaJ at the same ΔPPL")
        ax.set_xticks(x, [b for _, b in atts]); ax.tick_params(axis="x", labelsize=4.6, rotation=40)
        ax.set_title(SH[m], fontsize=6.0, pad=2); ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.25, lw=0.4)
        ax.tick_params(labelsize=5.2, length=2, pad=1)
    axs[0].set_ylabel("TPR (%)", fontsize=6.0, labelpad=1)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=5.0, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.12), handlelength=1.0)
    fig.tight_layout(pad=0.2, w_pad=0.3)
    save(fig, "fig_llm.png")


def side():
    fig, axs = plt.subplots(1, 2, figsize=(2.9, 1.45), dpi=400)
    Q = F["quality"]
    hv = lambda m, t: None if not Q[m].get(t) else Q[m][t]["hangul"] * 100
    grouped(axs[0], [SH[m] for m in F["models"]], [("no WM", "#c9cfd6", [hv(m, "none") for m in F["models"]]),
                                                   ("KGW", RED, [hv(m, "kgw_d2") for m in F["models"]]),
                                                   ("Ko-MaJ", GREEN, [hv(m, "han_d3") for m in F["models"]])], "Hangul ratio (%)")
    axs[0].tick_params(axis="x", labelsize=4.8)
    axs[0].legend(fontsize=4.6, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), handlelength=0.9, columnspacing=0.6)
    axs[0].set_ylim(0, 112)
    q = Q["qwen38_27b"]
    names = [("none", "no WM"), ("kgw_d2", "KGW"), ("han_d3", "Ko-MaJ"), ("han2_d3", "h2"), ("han2r_d3", "h2R")]
    vals = [None if not q.get(t) else q[t]["rep"] * 100 for t, _ in names]
    axs[1].bar(range(5), [0 if v is None else v for v in vals], color=["#c9cfd6", RED, GREEN, "#2d6a9f", "#7fb3d5"])
    axs[1].set_xticks(range(5), [n for _, n in names]); axs[1].tick_params(labelsize=4.8, length=2, pad=1)
    axs[1].set_ylabel("3-syllable repeat (%)", fontsize=5.8, labelpad=1); axs[1].grid(axis="y", alpha=0.25, lw=0.4)
    axs[1].set_title("Qwen3.8-27B, δ=3", fontsize=5.6, pad=2)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    save(fig, "fig_side.png")


if __name__ == "__main__":
    overview()
    pareto()
    nospace(); length(); null(); cross(); edit(); llm(); side()
    print("figures written")
