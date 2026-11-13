"""Build the comparison tables and the noise figure from results/results.jsonl.

    python scripts/make_report.py

Writes results/tables.md and figures/noise_citeseer.png. Paper values are taken
from Xia et al. (AAAI-25), Tables 1-2, Fig. 3 and the text accompanying Fig. 4.
"""
import json, os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = [json.loads(l) for l in open(os.path.join(ROOT, "results", "results.jsonl"))]
NAME = {"cora": "Cora", "citeseer": "Citeseer", "cs": "Co-CS"}

PAPER_T1 = {("cora", 0.5): (87.85, 0.64), ("cora", 0.3): (86.29, 0.79), ("cora", 0.1): (82.99, 1.00),
            ("citeseer", 0.5): (75.25, 1.02), ("citeseer", 0.3): (75.73, 0.88), ("citeseer", 0.1): (73.69, 0.93),
            ("cs", 0.3): (93.09, 0.13), ("cs", 0.1): (92.22, 0.22), ("cs", 0.05): (91.21, 0.30)}
PAPER_T2 = {"cora": (0.45, 87.57, 88.38), "citeseer": (0.49, 75.38, 76.48), "cs": (0.41, 93.28, 93.78)}
PAPER_TIME = [("cora", 0.3, 4.92), ("citeseer", 0.3, 3.47), ("cs", 0.05, 20.18)]
PAPER_NOISE_5 = {"SGBGC": 75.20, "Full graph": 73.21}      # only rate given numerically in the paper
NOISE = ["0.05", "0.1", "0.15", "0.2"]


def get(exp, ds, setting, epochs):
    m = [r for r in R if r["experiment"] == exp and r["dataset"] == ds
         and r["setting"] == setting and r["epochs"] == epochs]
    if len(m) != 1:
        raise KeyError(f"expected one result for {exp}/{ds}/{setting}/{epochs}, found {len(m)}")
    return m[0]


def pm(r):
    return f"{r['mean']:.2f} ± {r['std']:.2f}"


out = []
out.append("## Table 1 — accuracy at fixed coarsening ratios\n")
out.append("| Dataset | Ratio | Paper | Reproduced, 60 epochs | Diff | Reproduced, 200 epochs | Diff |")
out.append("|---|---|---|---|---|---|---|")
for (d, r), (m, s) in PAPER_T1.items():
    a, b = get("table1", d, f"r={r}", 60), get("table1", d, f"r={r}", 200)
    out.append(f"| {NAME[d]} | {r} | {m:.2f} ± {s:.2f} | {pm(a)} | {a['mean'] - m:+.2f} | {pm(b)} | {b['mean'] - m:+.2f} |")

out.append("\n## Table 2 — adaptive ratio and full-graph GCN (60 epochs)\n")
out.append("| Dataset | Ratio, paper | Ratio, reproduced | SGBGC, paper | SGBGC, reproduced | FULL, paper | FULL, reproduced |")
out.append("|---|---|---|---|---|---|---|")
for d, (pr, ps, pf) in PAPER_T2.items():
    a, f = get("adaptive", d, "purity-only", 60), get("full", d, "uncoarsened", 60)
    out.append(f"| {NAME[d]} | {pr:.2f} | {a['ratio_achieved']:.3f} ({a['balls']:,} balls) | {ps:.2f} | {pm(a)} | {pf:.2f} | {pm(f)} |")

out.append("\n## Fig. 4 — label noise on Citeseer, ratio 0.3\n")
out.append("| Noise | SGBGC, 60 ep. | Full graph, 60 ep. | Gap | SGBGC, 200 ep. | Full graph, 200 ep. | Gap |")
out.append("|---|---|---|---|---|---|---|")
for n in NOISE:
    row = [f"{float(n) * 100:.0f}%"]
    for e in (60, 200):
        s, f = get("noise", "citeseer", f"noise={n} SGBGC r=0.3", e), get("noise", "citeseer", f"noise={n} FULL", e)
        row += [pm(s), pm(f), f"{s['mean'] - f['mean']:+.2f}"]
    out.append("| " + " | ".join(row) + " |")
out.append(f"\nPaper, 5% noise: SGBGC {PAPER_NOISE_5['SGBGC']:.2f}, full graph {PAPER_NOISE_5['Full graph']:.2f}.")

out.append("\n## Fig. 3 — coarsening time\n")
out.append("| Dataset | Ratio | Paper | Reproduced |")
out.append("|---|---|---|---|")
for d, r, t in PAPER_TIME:
    out.append(f"| {NAME[d]} | {r} | {t:.2f} s | {get('table1', d, f'r={r}', 60)['coarsen_seconds']:.1f} s |")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
with open(os.path.join(ROOT, "results", "tables.md"), "w") as fh:
    fh.write("\n".join(out) + "\n")
print("\n".join(out))

# Noise figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xs = [float(n) * 100 for n in NOISE]
fig, ax = plt.subplots(figsize=(7, 4.2))
for label, key, color in [("SGBGC (ratio 0.3)", "SGBGC r=0.3", "#2a6fdb"), ("Full graph", "FULL", "#9a9a9a")]:
    rs = [get("noise", "citeseer", f"noise={n} {key}", 60) for n in NOISE]
    ax.errorbar(xs, [r["mean"] for r in rs], yerr=[r["std"] for r in rs], marker="o", capsize=3,
                color=color, label=f"{label}, reproduced")
ax.scatter([5 - 0.6], [PAPER_NOISE_5["SGBGC"]], facecolors="none", edgecolors="#2a6fdb", s=60, label="SGBGC, paper (5%)")
ax.scatter([5 - 0.6], [PAPER_NOISE_5["Full graph"]], facecolors="none", edgecolors="#9a9a9a", s=60, label="Full graph, paper (5%)")
ax.set_xticks(xs, [f"{x:.0f}%" for x in xs])
ax.set_xlabel("Labels flipped on training nodes")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Citeseer: SGBGC vs full graph under label noise (60 epochs, 20 runs)")
ax.grid(axis="y", alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figures", "noise_citeseer.png"), dpi=160)
print("wrote results/tables.md and figures/noise_citeseer.png")
