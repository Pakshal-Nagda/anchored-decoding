import json
import os
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

METRICS = ["score_rouge_1", "score_rouge_l", "score_lcs", "score_minhash", "score_acs", "score_lcs_char"]
METRIC_LABELS = {
    "score_rouge_1":  "ROUGE-1",
    "score_rouge_l":  "ROUGE-L",
    "score_lcs":      "LCS (word-level)",
    "score_minhash":  "MinHash Similarity",
    "score_acs":      "ACS (word-level)",
    "score_lcs_char": "LCS (char-level)",
}

METHODS = {
    "anchored":  dict(lambdas=[0.1, 0.5, 1, 5], color="#1f77b4", marker="o",  label="Anchored Dec."),
    "clip":      dict(lambdas=[0.1, 0.5, 1, 5], color="#2ca02c", marker="s",  label="CLIP"),
    "renyi2":    dict(lambdas=[0.1, 0.5, 1, 5], color="#ff7f0e", marker="^",  label=r"Renyi ($\alpha$=2)"),
    "renyi5":    dict(lambdas=[0.1, 0.5, 1, 5], color="#e377c2", marker="P",  label=r"Renyi ($\alpha$=5)"),
    "renyi10":   dict(lambdas=[0.1, 0.5, 1, 5], color="#d62728", marker="D",  label=r"Renyi ($\alpha$=10)"),
    "reversekl": dict(lambdas=[0.1, 0.5, 1, 5], color="#9467bd", marker="v",  label="Reverse KL"),
}


def load_data(filename):
    with open(os.path.join(RESULTS_DIR, filename)) as f:
        return {inst["id"]: inst for inst in json.load(f)}


def scores_for(data_by_id, allowed_ids=None):
    rows = data_by_id.values() if allowed_ids is None else [data_by_id[i] for i in allowed_ids if i in data_by_id]
    rows = list(rows)
    out = {}
    for m in METRICS:
        vals = [inst[m] for inst in rows if m in inst]
        if not vals:
            out[m] = (np.nan, np.nan)
        elif len(vals) == 1:
            out[m] = (float(vals[0]), 0.0)
        else:
            out[m] = (np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals)))
    return out


def ncr(risky_mean, safe_mean, method_mean):
    denom = risky_mean - safe_mean
    return 0.0 if abs(denom) < 1e-12 else (risky_mean - method_mean) / denom * 100


risky_data = load_data("risky_literal_results.json")
safe_data  = load_data("safe_literal_results.json")

method_scores = {}
for method, info in METHODS.items():
    method_scores[method] = {}
    for lam in info["lambdas"]:
        fname = f"{method}_{lam}_literal_results.json"
        path  = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(path):
            method_scores[method][lam] = load_data(fname)

available_method_data = [
    data
    for method_lams in method_scores.values()
    for data in method_lams.values()
]
if not available_method_data:
    raise RuntimeError(f"No method result files found in {RESULTS_DIR!r}.")

# IDs common to every method file; used to anchor risky/safe baselines fairly.
common_ids = None
for data in available_method_data:
    common_ids = set(data.keys()) if common_ids is None else common_ids & set(data.keys())

if not common_ids:
    raise RuntimeError("No shared instance IDs across method result files.")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, metric in enumerate(METRICS):
    ax = axes[idx]

    all_xs = [0, 100]  # risky and safe anchor points

    for method, info in METHODS.items():
        pts = []
        for lam in info["lambdas"]:
            if lam not in method_scores[method]:
                continue
            method_data = method_scores[method][lam]
            method_ids  = set(method_data.keys())
            # Compute risky/safe means on exactly the same instances as this method
            r = scores_for(risky_data, method_ids)
            s = scores_for(safe_data,  method_ids)
            r_mean, r_se = r[metric]
            s_mean, s_se = s[metric]
            m_mean, m_se = scores_for(method_data)[metric]
            x = ncr(r_mean, s_mean, m_mean)
            pts.append((x, m_mean, m_se * 1.96))
            all_xs.append(x)
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        xs, ys, errs = zip(*pts)
        ax.errorbar(xs, ys, yerr=errs,
                    color=info["color"], marker=info["marker"],
                    label=info["label"], capsize=3,
                    linewidth=1.5, markersize=6, zorder=3)

    r = scores_for(risky_data, common_ids)
    s = scores_for(safe_data,  common_ids)
    r_mean, r_se = r[metric]
    s_mean, s_se = s[metric]
    ax.errorbar([0],   [r_mean], yerr=[r_se * 1.96], color="#8c564b",
                marker="s", markersize=8, capsize=3, linewidth=1.5,
                label="Risky LM $p_r$", zorder=4)
    ax.errorbar([100], [s_mean], yerr=[s_se * 1.96], color="#17becf",
                marker="s", markersize=8, capsize=3, linewidth=1.5,
                label="Safe LM $p_s$", zorder=4)

    ax.set_title(METRIC_LABELS[metric], fontsize=11, fontweight="bold")
    ax.set_xlabel("Normalized copying reduction (%)", fontsize=9)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=9)
    pad = (max(all_xs) - min(all_xs)) * 0.05
    ax.set_xlim(min(all_xs) - pad, max(all_xs) + pad)
    ax.tick_params(labelsize=8)

handles, labels = axes[0].get_legend_handles_labels()
# collect baselines from last subplot too (they aren't in axes[0])
h2, l2 = axes[-1].get_legend_handles_labels()
all_h = {l: h for h, l in zip(handles + h2, labels + l2)}
fig.legend(all_h.values(), all_h.keys(),
           loc="lower center", ncol=4,
           fontsize=9, frameon=True,
           bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Copyright Reduction vs. Literal Copying Metrics", fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.06, 1, 1])

os.makedirs("figures", exist_ok=True)
out_path = "figures/tradeoff_all_metrics.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out_path}")
