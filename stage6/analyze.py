"""Summarise Stage 6 and apply the decision rule (PROTOCOL.md).

    python analyze.py        # reads results/results*.jsonl

Writes results/summary.md and figures/guided_kmeans.png.
"""
import glob, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["cora", "citeseer", "pubmed"]
DS_NAME = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "PubMed"}
RATIOS = [0.02, 0.01, 0.005]
VARIANTS = ["K0", "K1", "K2", "K3"]
NAME = {"full": "Full graph", "K0": "K0 k-means", "K1": "K1 + feature protection",
        "K2": "K2 label-propagation features", "K3": "K3 = K2 + protection"}
SEEDS = list(range(30, 40))

R = {}
for p in sorted(glob.glob(os.path.join(HERE, "results", "results*.jsonl"))):
    for line in open(p):
        r = json.loads(line)
        R[(r["dataset"], r["seed"], r["method"], None if r["method"] == "full" else r["ratio_target"])] = r


def get(ds, m, ratio, key="test"):
    rk = None if m == "full" else ratio
    return np.array([(100 if key == "test" else 1) * R[(ds, s, m, rk)][key] for s in SEEDS if (ds, s, m, rk) in R])


def ci(d):
    return 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")


complete = all((ds, s, m, None if m == "full" else r) in R
               for ds in DATASETS for s in SEEDS for m in ["full"] + VARIANTS for r in RATIOS)
out = ["# Stage 6 results — label-guided k-means coarsening", "",
       f"{len(R)} training runs. Test accuracy (%); ± = 95% CI over seeds of paired per-seed differences.", ""]
if not complete:
    out += ["**PROVISIONAL — incomplete run.** The decision rule is applied only to the complete run.", ""]

out += ["## Decision (pre-registered rule; primary method K2)", "",
        "| Dataset | Gain K2 − K0 (mean of 3 ratios) | Passes (≥ 1.5, CI > 0)? |", "|---|---|---|"]
gains, passes, harm = {}, [], []
for ds in DATASETS:
    if any(len(get(ds, m, r)) < len(SEEDS) for m in ("K0", "K2") for r in RATIOS):
        continue
    per = np.mean([get(ds, "K2", r) - get(ds, "K0", r) for r in RATIOS], axis=0)
    g, c = per.mean(), ci(per)
    gains[ds] = g
    ok = g >= 1.5 and g - c > 0
    passes.append(ok)
    if g + c < -1:
        harm.append(ds)
    out.append(f"| {DS_NAME[ds]} | {g:+.2f} ± {c:.2f} | {'**yes**' if ok else 'no'} |")
if complete:
    if sum(passes) >= 2 and not harm:
        dec = "**GO** — label-propagation features improve k-means by ≥ 1.5 points on at least 2 of 3 datasets."
    elif all(g <= 0.5 for g in gains.values()):
        dec = "**STOP** — gain ≤ 0.5 points on every dataset."
    else:
        dec = "**INCONCLUSIVE** — neither condition met" + (f" (clearly harmed: {', '.join(DS_NAME[d] for d in harm)})" if harm else "") + "."
    out += ["", f"- Decision: {dec}"]
out += [""]

out += ["## Per cell", "",
        "| Dataset | Keep | Full graph | K0 | K1 | K2 | K3 | K2 − K0 | K1 − K0 | K3 − K0 | Labels lost by K0 |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    f = get(ds, "full", None)
    for r in RATIOS:
        a = {m: get(ds, m, r) for m in VARIANTS}
        if any(len(v) == 0 for v in a.values()) or len(f) == 0:
            continue
        lost, ntr = get(ds, "K0", r, "labels_lost").mean(), get(ds, "K0", r, "n_train").mean()
        d = lambda m: f"{(a[m] - a['K0']).mean():+.2f} ± {ci(a[m] - a['K0']):.2f}"
        out.append(f"| {DS_NAME[ds]} | {r * 100:g}% | {f.mean():.1f} | " + " | ".join(f"{a[m].mean():.1f}" for m in VARIANTS)
                   + f" | {d('K2')} | {d('K1')} | {d('K3')} | {lost:.0f}/{ntr:.0f} |")
out += [""]

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dss = [d for d in DATASETS if len(get(d, "full", None))]
if dss:
    color = {"K0": "#9467bd", "K1": "#c5b0d5", "K2": "#1f77b4", "K3": "#aec7e8"}
    fig, axes = plt.subplots(1, len(dss), figsize=(4.4 * len(dss), 3.8), squeeze=False)
    for i, ds in enumerate(dss):
        ax = axes[0][i]
        full = get(ds, "full", None).mean()
        for j, m in enumerate(VARIANTS):
            ys = [get(ds, m, r).mean() if len(get(ds, m, r)) else np.nan for r in RATIOS]
            ax.plot(range(len(RATIOS)), ys, marker="o", color=color[m], label=NAME[m])
        ax.axhline(full, color="black", ls="--", lw=0.8, label="Full graph")
        ax.set_xticks(range(len(RATIOS)), [f"{r * 100:g}%" for r in RATIOS])
        ax.set_xlabel("nodes kept"); ax.set_title(DS_NAME[ds], fontsize=10)
        ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("Test accuracy (%)")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle("Stage 6: k-means coarsening with and without label guidance (20 labels per class)", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "guided_kmeans.png"), dpi=150)
    print("wrote results/summary.md and figures/guided_kmeans.png")
