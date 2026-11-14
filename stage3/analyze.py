"""Summarise Stage 3 and apply the decision rule (PROTOCOL.md, incl. Amendment 1).

    python analyze.py        # reads results/results*.jsonl

Writes results/summary.md and figures/headroom.png.
"""
import glob, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["cora", "citeseer", "roman_empire", "amazon_ratings"]
DS_NAME = {"cora": "Cora", "citeseer": "Citeseer", "roman_empire": "Roman-empire", "amazon_ratings": "Amazon-ratings"}
RATIOS = [0.1, 0.05, 0.02]
LABEL_FREE = ["loukas", "gb_struct"]
ORACLES = ["oracle", "oracle_sep"]
METHODS = ["loukas", "gb_struct", "sgbgc_keep", "oracle", "oracle_sep"]
NAME = {"full": "Full graph", "loukas": "Loukas/SCAL", "gb_struct": "GB-struct", "sgbgc_keep": "SGBGC-keep (20/class)",
        "oracle": "Oracle O", "oracle_sep": "Oracle O-sep"}
SEEDS = list(range(10))

R = {}
for p in sorted(glob.glob(os.path.join(HERE, "results", "results*.jsonl"))):
    for line in open(p):
        r = json.loads(line)
        R[(r["dataset"], r["seed"], r["method"], None if r["method"] == "full" else r["ratio_target"])] = r


def accs(ds, m, ratio):
    key_ratio = None if m == "full" else ratio
    return np.array([100 * R[(ds, s, m, key_ratio)]["test"] for s in SEEDS if (ds, s, m, key_ratio) in R])


def complete():
    return all((ds, s, m, None if m == "full" else r) in R
               for ds in DATASETS for s in SEEDS for m in ["full"] + METHODS for r in RATIOS)


def paired(ds, a, b, ratio):
    d = accs(ds, a, ratio) - accs(ds, b, ratio)
    return d.mean(), 1.96 * d.std(ddof=1) / np.sqrt(len(d))


out = ["# Stage 3 results — headroom test", "",
       f"{len(R)} training runs. Test accuracy (%), mean over seeds; ± = 95% CI of paired per-seed differences.", ""]
done = complete()
if not done:
    out += ["**PROVISIONAL — incomplete run.** The decision rule is applied only to the complete run.", ""]

out += ["## Headroom per cell", "",
        "| Dataset | Keep | Best label-free | Best oracle | H = oracle − label-free | G = oracle − SGBGC-keep | Target cell? |",
        "|---|---|---|---|---|---|---|"]
cells = []
for ds in DATASETS:
    for r in RATIOS:
        if any(len(accs(ds, m, r)) == 0 for m in METHODS):
            continue
        lf = max(LABEL_FREE, key=lambda m: accs(ds, m, r).mean())
        orc = max(ORACLES, key=lambda m: accs(ds, m, r).mean())
        h, hc = paired(ds, orc, lf, r)
        g, gc = paired(ds, orc, "sgbgc_keep", r)
        target = h >= 3 and (h - hc) > 0 and g >= 2
        cells.append((ds, r, h, g, target))
        out.append(f"| {DS_NAME[ds]} | {int(r * 100)}% | {NAME[lf]} ({accs(ds, lf, r).mean():.1f}) | "
                   f"{NAME[orc]} ({accs(ds, orc, r).mean():.1f}) | {h:+.2f} ± {hc:.2f} | {g:+.2f} ± {gc:.2f} | "
                   f"{'**yes**' if target else 'no'} |")

if done:
    if any(c[4] for c in cells):
        dec = "**TARGET FOUND** in: " + ", ".join(f"{DS_NAME[c[0]]} at {int(c[1] * 100)}%" for c in cells if c[4])
    elif all(c[2] <= 1 for c in cells):
        dec = "**NO HEADROOM** — H ≤ 1 point in every cell."
    else:
        dec = "**INCONCLUSIVE** — no target cell, but H > 1 point in some cells."
    out += ["", f"- Decision: {dec}"]
out += [""]

out += ["## Accuracy by method", ""]
for ds in DATASETS:
    if len(accs(ds, "full", None)) == 0:
        continue
    f = accs(ds, "full", None)
    out += [f"### {DS_NAME[ds]} (full graph: {f.mean():.1f} ± {f.std():.1f})", "",
            "| Method | " + " | ".join(f"keep {int(r * 100)}%" for r in RATIOS) + " |", "|---|" + "---|" * len(RATIOS)]
    for m in METHODS:
        cells_txt = []
        for r in RATIOS:
            a = accs(ds, m, r)
            cells_txt.append(f"{a.mean():.1f} ± {a.std():.1f}" if len(a) else "—")
        out.append(f"| {NAME[m]} | " + " | ".join(cells_txt) + " |")
    out.append("")

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# figure: accuracy relative to the full graph, per dataset and ratio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dss = [d for d in DATASETS if len(accs(d, "full", None))]
if dss:
    color = {"loukas": "#7f7f7f", "gb_struct": "#bcbd22", "sgbgc_keep": "#1f77b4", "oracle": "#ff7f0e", "oracle_sep": "#d62728"}
    fig, axes = plt.subplots(1, len(dss), figsize=(4.2 * len(dss), 3.8), squeeze=False)
    w = 0.16
    for i, ds in enumerate(dss):
        ax = axes[0][i]
        full = accs(ds, "full", None).mean()
        for j, m in enumerate(METHODS):
            ys = [accs(ds, m, r).mean() - full if len(accs(ds, m, r)) else np.nan for r in RATIOS]
            ax.bar(np.arange(len(RATIOS)) + (j - 2) * w, ys, w, color=color[m], label=NAME[m])
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xticks(range(len(RATIOS)), [f"{int(r * 100)}%" for r in RATIOS])
        ax.set_title(f"{DS_NAME[ds]} (full graph {full:.1f}%)", fontsize=10)
        ax.set_xlabel("nodes kept"); ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Accuracy minus full graph (points)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(METHODS), fontsize=8, frameon=False)
    fig.suptitle("Stage 3: coarsened accuracy relative to the full graph (20 training labels per class)", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "headroom.png"), dpi=150)
    print("wrote results/summary.md and figures/headroom.png")
