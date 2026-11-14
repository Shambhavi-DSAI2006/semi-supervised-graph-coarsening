"""Summarise Stage 5 and apply the decision rule (PROTOCOL.md).

    python analyze.py        # reads results/results*.jsonl

Writes results/summary.md and figures/protection_gain.png.
"""
import glob, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["cora", "citeseer", "pubmed"]
DS_NAME = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "PubMed"}
BASES = ["loukas", "gb_struct", "kmeans"]
B_NAME = {"loukas": "Loukas/SCAL", "gb_struct": "GB-struct", "kmeans": "k-means"}
RATIOS = [0.05, 0.02, 0.01]
SEEDS = list(range(20, 30))
ALL = ["full"] + [m for b in BASES for m in (b, b + "_protect")] + ["loukas_sep"]

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


complete = all((ds, s, m, None if m == "full" else r) in R for ds in DATASETS for s in SEEDS for m in ALL for r in RATIOS)
out = ["# Stage 5 results — general label-protection wrapper", "",
       f"{len(R)} training runs. Test accuracy (%); ± = 95% CI over seeds of paired per-seed differences.", ""]
if not complete:
    out += ["**PROVISIONAL — incomplete run.** The decision rule is applied only to the complete run.", ""]

# ---------------------------------------------------------------- decision
out += ["## Decision (pre-registered rule)", "",
        "Gain = (base + protection) − base, averaged over keeping 5%, 2% and 1% of nodes.", "",
        "| Base method | " + " | ".join(DS_NAME[d] for d in DATASETS) + " | Datasets passing |",
        "|---|" + "---|" * len(DATASETS) + "---|"]
pair = {}
for b in BASES:
    cells, n_pass = [], 0
    for ds in DATASETS:
        if any(len(get(ds, m, r)) < len(SEEDS) for m in (b, b + "_protect") for r in RATIOS):
            cells.append("—"); continue
        per = np.mean([get(ds, b + "_protect", r) - get(ds, b, r) for r in RATIOS], axis=0)
        g, c = per.mean(), ci(per)
        ok = g >= 2 and g - c > 0
        n_pass += ok
        pair[(b, ds)] = (g, c)
        cells.append(f"{g:+.2f} ± {c:.2f}" + (" ✓" if ok else ""))
    out.append(f"| {B_NAME[b]} | " + " | ".join(cells) + f" | {n_pass} / 3 |")
if complete:
    bases_ok = sum(sum(pair[(b, d)][0] >= 2 and pair[(b, d)][0] - pair[(b, d)][1] > 0 for d in DATASETS) >= 2 for b in BASES)
    harm = [(b, d) for (b, d), (g, c) in pair.items() if g + c < -1]
    if bases_ok >= 2 and not harm:
        dec = f"**GO (general)** — passes on ≥ 2 of 3 datasets for {bases_ok} of 3 base methods, and no pair clearly harmed."
    elif all(g <= 1 for g, c in pair.values()):
        dec = "**STOP** — gain ≤ 1 point for all 9 pairs."
    else:
        dec = (f"**INCONCLUSIVE** — {bases_ok} of 3 base methods pass on ≥ 2 datasets"
               + (f"; clearly harmed: {', '.join(f'{B_NAME[b]} on {DS_NAME[d]}' for b, d in harm)}" if harm else "") + ".")
    out += ["", f"- Decision: {dec}"]
out += [""]

# ---------------------------------------------------------------- per cell
out += ["## Per cell", "",
        "| Dataset | Keep | Base | Base acc. | + protection | Gain | Labels lost by base |",
        "|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    for r in RATIOS:
        for b in BASES:
            a0, a1 = get(ds, b, r), get(ds, b + "_protect", r)
            if not (len(a0) and len(a1)):
                continue
            lost, ntr = get(ds, b, r, "labels_lost").mean(), get(ds, b, r, "n_train").mean()
            out.append(f"| {DS_NAME[ds]} | {r * 100:g}% | {B_NAME[b]} | {a0.mean():.1f} | {a1.mean():.1f} | "
                       f"{(a1 - a0).mean():+.2f} ± {ci(a1 - a0):.2f} | {lost:.0f}/{ntr:.0f} ({100 * lost / ntr:.0f}%) |")
out += [""]

# ---------------------------------------------------------------- references
out += ["## References (not part of the rule)", "",
        "| Dataset | Keep | Full graph | Loukas | Loukas + protection | Loukas + Stage 4 separation | protection − Stage 4 |",
        "|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    f = get(ds, "full", None)
    for r in RATIOS:
        a, p, s4 = get(ds, "loukas", r), get(ds, "loukas_protect", r), get(ds, "loukas_sep", r)
        if not (len(a) and len(p) and len(s4) and len(f)):
            continue
        out.append(f"| {DS_NAME[ds]} | {r * 100:g}% | {f.mean():.1f} | {a.mean():.1f} | {p.mean():.1f} | {s4.mean():.1f} | "
                   f"{(p - s4).mean():+.2f} ± {ci(p - s4):.2f} |")
out += [""]

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(DATASETS), figsize=(13, 3.8), sharey=True)
colors = {"loukas": "#7f7f7f", "gb_struct": "#bcbd22", "kmeans": "#9467bd"}
xs = np.arange(len(RATIOS))
for i, ds in enumerate(DATASETS):
    ax = axes[i]
    for k, b in enumerate(BASES):
        g, c = [], []
        for r in RATIOS:
            d = get(ds, b + "_protect", r) - get(ds, b, r)
            g.append(d.mean() if len(d) else np.nan); c.append(ci(d) if len(d) else np.nan)
        ax.errorbar(xs + (k - 1) * 0.13, g, yerr=c, fmt="o", capsize=3, color=colors[b], label=B_NAME[b])
    ax.axhline(0, color="black", lw=0.8); ax.axhline(2, color="gray", lw=0.8, ls="--")
    ax.set_xticks(xs, [f"{r * 100:g}%" for r in RATIOS]); ax.set_xlabel("nodes kept")
    ax.set_title(DS_NAME[ds], fontsize=10); ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Gain from protection (points)")
axes[-1].legend(fontsize=8, frameon=False)
fig.suptitle("Stage 5: accuracy gain from the label-protection wrapper, per base method (95% CI; dashed = 2-point bar)", fontsize=11)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "protection_gain.png"), dpi=150)
print("wrote results/summary.md and figures/protection_gain.png")
