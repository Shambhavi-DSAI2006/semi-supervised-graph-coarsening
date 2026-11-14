"""Summarise Stage 4 and apply the decision rule (PROTOCOL.md).

    python analyze.py        # reads results/results*.jsonl

Writes results/summary.md and figures/separation_gain.png.
"""
import glob, json, os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["cora", "citeseer", "pubmed"]
DS_NAME = {"cora": "Cora", "citeseer": "Citeseer", "pubmed": "PubMed"}
RATIOS = [0.05, 0.02, 0.01]
METHODS = ["loukas", "loukas_sep", "sgbgc_keep"]
NAME = {"full": "Full graph", "loukas": "Loukas/SCAL (M4)", "loukas_sep": "Loukas + label separation (M7)",
        "sgbgc_keep": "SGBGC-keep (M3)"}
SEEDS = list(range(10, 20))

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
               for ds in DATASETS for s in SEEDS for m in ["full"] + METHODS for r in RATIOS)
out = ["# Stage 4 results — label-separation test", "",
       f"{len(R)} training runs. Test accuracy (%); ± = 95% CI over seeds of paired per-seed differences.", ""]
if not complete:
    out += ["**PROVISIONAL — incomplete run.** The decision rule is applied only to the complete run.", ""]

# ---------------------------------------------------------------- decision
out += ["## Decision (pre-registered rule)", "",
        "| Dataset | Gain: M7 − M4 (mean of 3 ratios) | Passes (≥ 2, CI > 0)? |", "|---|---|---|"]
passes, gains = [], {}
for ds in DATASETS:
    if any(len(get(ds, m, r)) < len(SEEDS) for m in ("loukas", "loukas_sep") for r in RATIOS):
        continue
    per_seed = np.mean([get(ds, "loukas_sep", r) - get(ds, "loukas", r) for r in RATIOS], axis=0)
    g, c = per_seed.mean(), ci(per_seed)
    gains[ds] = g
    ok = g >= 2 and g - c > 0
    passes.append(ok)
    out.append(f"| {DS_NAME[ds]} | {g:+.2f} ± {c:.2f} | {'**yes**' if ok else 'no'} |")
if complete:
    if sum(passes) >= 2:
        dec = "**GO** — label separation gains ≥ 2 points on at least 2 of 3 datasets."
    elif all(g <= 1 for g in gains.values()):
        dec = "**STOP** — label separation gains ≤ 1 point on every dataset."
    else:
        dec = "**INCONCLUSIVE** — neither condition met."
    out += ["", f"- Decision: {dec}"]
out += [""]

# ---------------------------------------------------------------- per cell
out += ["## Per cell", "",
        "| Dataset | Keep | Full graph | M4 Loukas | M7 Loukas + sep | M3 SGBGC-keep | M7 − M4 | M3 − M4 | Labels lost by M4 |",
        "|---|---|---|---|---|---|---|---|---|"]
for ds in DATASETS:
    f = get(ds, "full", None)
    for r in RATIOS:
        a4, a7, a3 = get(ds, "loukas", r), get(ds, "loukas_sep", r), get(ds, "sgbgc_keep", r)
        if not (len(a4) and len(a7) and len(a3)) or len(f) == 0:
            continue
        lost = get(ds, "loukas", r, "labels_lost").mean()
        ntr = get(ds, "loukas", r, "n_train").mean()
        out.append(f"| {DS_NAME[ds]} | {r * 100:g}% | {f.mean():.1f} | {a4.mean():.1f} ± {a4.std():.1f} | "
                   f"{a7.mean():.1f} ± {a7.std():.1f} | {a3.mean():.1f} ± {a3.std():.1f} | "
                   f"{(a7 - a4).mean():+.2f} ± {ci(a7 - a4):.2f} | {(a3 - a4).mean():+.2f} ± {ci(a3 - a4):.2f} | "
                   f"{lost:.0f} / {ntr:.0f} ({100 * lost / ntr:.0f}%) |")
out += ["", "Super-nodes (mean): " + "; ".join(
    f"{DS_NAME[ds]} {r * 100:g}%: M4 {get(ds, 'loukas', r, 'super_nodes').mean():.0f}, M7 {get(ds, 'loukas_sep', r, 'super_nodes').mean():.0f}, M3 {get(ds, 'sgbgc_keep', r, 'super_nodes').mean():.0f}"
    for ds in DATASETS for r in RATIOS if len(get(ds, "loukas_sep", r))), ""]

os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
open(os.path.join(HERE, "results", "summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))

# ---------------------------------------------------------------- figure: gain vs labels lost
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
colors = {"cora": "#1f77b4", "citeseer": "#ff7f0e", "pubmed": "#2ca02c"}
xs = np.arange(len(RATIOS))
for k, ds in enumerate(DATASETS):
    g, c, lostpct = [], [], []
    for r in RATIOS:
        d = get(ds, "loukas_sep", r) - get(ds, "loukas", r)
        g.append(d.mean() if len(d) else np.nan); c.append(ci(d) if len(d) else np.nan)
        lo = get(ds, "loukas", r, "labels_lost"); nt = get(ds, "loukas", r, "n_train")
        lostpct.append(100 * lo.mean() / nt.mean() if len(lo) else np.nan)
    axes[0].errorbar(xs + (k - 1) * 0.12, g, yerr=c, fmt="o", capsize=3, color=colors[ds], label=DS_NAME[ds])
    axes[1].scatter(lostpct, g, color=colors[ds], s=40, label=DS_NAME[ds])
axes[0].axhline(0, color="black", lw=0.8); axes[0].axhline(2, color="gray", lw=0.8, ls="--")
axes[0].set_xticks(xs, [f"{r * 100:g}%" for r in RATIOS]); axes[0].set_xlabel("nodes kept")
axes[0].set_ylabel("Gain: Loukas + separation − Loukas (points)")
axes[0].set_title("Gain per cell (95% CI); dashed = 2-point bar", fontsize=10)
axes[1].axhline(0, color="black", lw=0.8)
axes[1].set_xlabel("Training labels lost by plain Loukas (%)"); axes[1].set_ylabel("Gain (points)")
axes[1].set_title("Does the gain follow the labels lost?", fontsize=10)
for ax in axes:
    ax.grid(alpha=0.3); ax.spines[["top", "right"]].set_visible(False); ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figures", "separation_gain.png"), dpi=150)
print("wrote results/summary.md and figures/separation_gain.png")
