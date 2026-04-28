import json
import os
import re

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
FIGURES_DIR = "figures"

KS = [0.1, 0.5, 1, 5]
TECHNIQUES = {
    "anchored": "Anchored Dec.",
    "clip": "CLIP",
    "renyi2": r"Renyi ($\alpha$=2)",
    "renyi5": r"Renyi ($\alpha$=5)",
    "renyi10": r"Renyi ($\alpha$=10)",
    "reversekl": "Reverse KL",
}

METRICS = [
    "score_rouge_1",
    "score_rouge_l",
    "score_lcs",
    "score_minhash",
    "score_acs",
    "score_lcs_char",
]
METRIC_LABELS = {
    "score_rouge_1": "ROUGE-1",
    "score_rouge_l": "ROUGE-L",
    "score_lcs": "LCS (word)",
    "score_minhash": "MinHash",
    "score_acs": "ACS",
    "score_lcs_char": "LCS (char)",
}

METRIC_STYLES = {
    "score_rouge_1": dict(color="#1f77b4", marker="o"),
    "score_rouge_l": dict(color="#ff7f0e", marker="s"),
    "score_lcs": dict(color="#2ca02c", marker="^"),
    "score_minhash": dict(color="#d62728", marker="D"),
    "score_acs": dict(color="#9467bd", marker="v"),
    "score_lcs_char": dict(color="#8c564b", marker="P"),
}


def load_data(filename):
    with open(os.path.join(RESULTS_DIR, filename), encoding="utf-8") as f:
        return {inst["id"]: inst for inst in json.load(f)}


def mean_for(rows, field):
    vals = [row[field] for row in rows if field in row]
    return float(np.mean(vals)) if vals else np.nan


def rows_for(data_by_id, allowed_ids):
    return [data_by_id[i] for i in allowed_ids if i in data_by_id]


def normalized_copying_reduction(risky_mean, safe_mean, method_mean):
    denom = risky_mean - safe_mean
    if abs(denom) < 1e-12:
        return np.nan
    return (risky_mean - method_mean) / denom * 100


def discover_result_files():
    pattern = re.compile(r"^(?P<technique>[a-z0-9]+)_(?P<k>0\.1|0\.5|1|5)_literal_results\.json$")
    files = {}
    for filename in os.listdir(RESULTS_DIR):
        match = pattern.match(filename)
        if not match:
            continue
        technique = match.group("technique")
        if technique not in TECHNIQUES:
            continue
        k = float(match.group("k"))
        files.setdefault(technique, {})[k] = filename
    return files


def plot_technique(technique, files, risky_data, safe_data):
    fig, ax = plt.subplots(figsize=(8, 5))

    metric_points = {metric: [] for metric in METRICS}
    for k in KS:
        filename = files.get(k)
        if not filename:
            continue

        method_data = load_data(filename)
        method_ids = set(method_data)
        common_ids = method_ids & set(risky_data) & set(safe_data)
        if not common_ids:
            continue

        method_rows = rows_for(method_data, common_ids)
        risky_rows = rows_for(risky_data, common_ids)
        safe_rows = rows_for(safe_data, common_ids)

        budget_utilization = mean_for(method_rows, "budget_utilization_per_seq")
        x = k * budget_utilization / 100.0

        for metric in METRICS:
            risky_mean = mean_for(risky_rows, metric)
            safe_mean = mean_for(safe_rows, metric)
            method_mean = mean_for(method_rows, metric)
            y = normalized_copying_reduction(risky_mean, safe_mean, method_mean)
            if not np.isnan(x) and not np.isnan(y):
                metric_points[metric].append((x, y, k))

    for metric, points in metric_points.items():
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        xs, ys, ks = zip(*points)
        ax.plot(
            xs,
            ys,
            label=METRIC_LABELS[metric],
            linewidth=1.8,
            markersize=6,
            **METRIC_STYLES[metric],
        )
    ax.axhline(0, color="#444444", linewidth=0.8, alpha=0.5)
    ax.axhline(100, color="#444444", linewidth=0.8, alpha=0.5, linestyle="--")
    ax.set_title(TECHNIQUES[technique], fontsize=13, fontweight="bold")
    ax.set_xlabel(r"$k \times$ budget utilization / 100")
    ax.set_ylabel("Normalized copying reduction (%)")
    ax.legend(frameon=True, fontsize=8, ncol=2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, f"{technique}_budget_utilization_vs_ncr.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    mpl.rcParams.update({"font.family": "sans-serif"})

    risky_data = load_data("risky_literal_results.json")
    safe_data = load_data("safe_literal_results.json")
    files_by_technique = discover_result_files()

    out_paths = []
    for technique in TECHNIQUES:
        files = files_by_technique.get(technique, {})
        if not files:
            continue
        out_paths.append(plot_technique(technique, files, risky_data, safe_data))

    if not out_paths:
        raise RuntimeError(f"No technique result files found in {RESULTS_DIR!r}.")

    print("Saved:")
    for path in out_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
