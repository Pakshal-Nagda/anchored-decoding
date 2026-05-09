r"""Plot HP MAUVE against ROUGE-1.

Default run:
    python .\plot_hp_mauve_vs_rouge1.py

ROUGE-1 is computed from hp_results/ with duplicate IDs excluded by keeping the
first occurrence of each ID. MAUVE is averaged across the risky HP gold runs in
mauve_scores_risky_hp_gen_*_gold_gpt2.json.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


FIGURES_DIR = "figures"
RESULTS_DIR = "hp_results"

METHODS = {
    "anchored": dict(color="#1f77b4", marker="o", label="Anchored Dec."),
    "clip": dict(color="#2ca02c", marker="s", label="CLIP"),
    "renyi2": dict(color="#ff7f0e", marker="^", label=r"Renyi ($\alpha$=2)"),
    "renyi5": dict(color="#e377c2", marker="P", label=r"Renyi ($\alpha$=5)"),
    "renyi10": dict(color="#d62728", marker="D", label=r"Renyi ($\alpha$=10)"),
    "reversekl": dict(color="#9467bd", marker="v", label="Reverse KL"),
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def unique_rows_by_id(rows):
    seen = set()
    out = []
    for row in rows:
        row_id = row.get("id")
        if row_id in seen:
            continue
        seen.add(row_id)
        out.append(row)
    return out


def unique_mauve_rows(rows):
    seen = set()
    out = []
    for row in rows:
        key = (
            row.get("gold"),
            row.get("filename"),
            row.get("method"),
            row.get("k"),
            bool(row.get("baseline")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def mean_se(values):
    values = np.array(values, dtype=float)
    mean = float(np.mean(values))
    if values.size < 2:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / np.sqrt(values.size))


def result_filename(gen_filename):
    return gen_filename.replace("_hp_gen.json", "_hp_results.json")


def rouge1_stats(results_dir, filename):
    path = os.path.join(results_dir, result_filename(filename))
    rows = unique_rows_by_id(load_json(path))
    values = [row["score_rouge_1"] for row in rows if "score_rouge_1" in row]
    mean, se = mean_se(values)
    return {
        "mean": mean,
        "se": se,
        "n": len(values),
    }


def baseline_result_filename(gen_filename):
    return gen_filename.replace("_hp_gen_", "_hp_results_")


def baseline_rouge1_stats(results_dir, gen_filename):
    path = os.path.join(results_dir, baseline_result_filename(gen_filename))
    rows = unique_rows_by_id(load_json(path))
    values = [row["score_rouge_1"] for row in rows if "score_rouge_1" in row]
    mean, se = mean_se(values)
    return {
        "mean": mean,
        "se": se,
        "n": len(values),
    }


def transform(value, use_exp):
    return float(np.exp(value)) if use_exp else float(value)


def transform_err(center, err, use_exp):
    if not use_exp:
        return err
    low = np.exp(center) - np.exp(max(0.0, center - err))
    high = np.exp(center + err) - np.exp(center)
    return np.array([[max(0.0, low)], [max(0.0, high)]])


def main(args):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    mauve_paths = sorted(glob.glob(args.input_glob))
    if not mauve_paths:
        raise RuntimeError(f"No MAUVE score files matched {args.input_glob!r}.")

    rows = []
    for path in mauve_paths:
        rows.extend(load_json(path))
    rows = unique_mauve_rows(rows)

    grouped = defaultdict(list)
    baselines = defaultdict(list)
    n_examples = set()
    for row in rows:
        if "n_examples" in row:
            n_examples.add(row["n_examples"])

        if row.get("baseline"):
            label = "Gold LM" if row.get("mauve") == 1.0 else "Safe LM"
            baselines[label].append(row)
            continue

        method = row.get("method")
        k = row.get("k")
        mauve = row.get("mauve")
        if method not in METHODS or k is None or mauve is None:
            continue
        grouped[(method, float(k), row["filename"])].append(float(mauve))

    points_by_method = defaultdict(list)
    for (method, k, filename), mauves in grouped.items():
        rouge = rouge1_stats(args.results_dir, filename)
        mauve_mean, mauve_se = mean_se(mauves)
        points_by_method[method].append({
            "k": k,
            "rouge_mean": rouge["mean"],
            "rouge_se": rouge["se"],
            "rouge_n": rouge["n"],
            "mauve_mean": mauve_mean,
            "mauve_se": mauve_se,
            "mauve_runs": len(mauves),
        })

    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_x = []
    all_y = []

    for method, points in points_by_method.items():
        info = METHODS[method]
        points = sorted(points, key=lambda row: row["k"])
        raw_xs = np.array([row["rouge_mean"] for row in points], dtype=float)
        raw_ys = np.array([row["mauve_mean"] for row in points], dtype=float)
        xs = np.array([transform(x, args.exp_axes) for x in raw_xs], dtype=float)
        ys = np.array([transform(y, args.exp_axes) for y in raw_ys], dtype=float)
        xerrs = [
            transform_err(x, row["rouge_se"] * 1.96, args.exp_axes)
            for x, row in zip(raw_xs, points)
        ]
        yerrs = [
            transform_err(y, row["mauve_se"] * 1.96, args.exp_axes)
            for y, row in zip(raw_ys, points)
        ]
        all_x.extend(xs.tolist())
        all_y.extend(ys.tolist())
        ax.errorbar(
            xs,
            ys,
            xerr=np.hstack(xerrs) if args.exp_axes else np.array(xerrs, dtype=float),
            yerr=np.hstack(yerrs) if args.exp_axes else np.array(yerrs, dtype=float),
            color=info["color"],
            marker=info["marker"],
            linewidth=1.8,
            markersize=6,
            capsize=3,
            label=info["label"],
        )

    for label, color, marker in [
        ("Gold LM", "#8c564b", "s"),
        ("Safe LM", "#17becf", "s"),
    ]:
        if label not in baselines:
            continue
        mauve_values = [float(row["mauve"]) for row in baselines[label]]
        mauve_mean, mauve_se = mean_se(mauve_values)
        if label == "Gold LM":
            rouge_values = [
                baseline_rouge1_stats(args.results_dir, row["filename"])["mean"]
                for row in baselines[label]
            ]
            rouge_mean, rouge_se = mean_se(rouge_values)
        else:
            rouge_stats = baseline_rouge1_stats(args.results_dir, baselines[label][0]["filename"])
            rouge_mean = rouge_stats["mean"]
            rouge_se = rouge_stats["se"]
        x = transform(rouge_mean, args.exp_axes)
        y = transform(mauve_mean, args.exp_axes)
        all_x.append(x)
        all_y.append(y)
        ax.errorbar(
            [x],
            [y],
            xerr=transform_err(rouge_mean, rouge_se * 1.96, args.exp_axes),
            yerr=transform_err(mauve_mean, mauve_se * 1.96, args.exp_axes),
            color=color,
            marker=marker,
            markersize=7,
            linestyle="none",
            capsize=3,
            label=label,
            zorder=4,
        )

    if args.exp_axes:
        ax.set_xlabel(r"$e^{\mathrm{ROUGE\text{-}1}}$")
        ax.set_ylabel(r"$e^{\mathrm{MAUVE}}$")
        ax.set_title("HP: exp(MAUVE) vs. exp(ROUGE-1)", fontsize=13, fontweight="bold")
    else:
        ax.set_xlabel("ROUGE-1")
        ax.set_ylabel("MAUVE")
        ax.set_title("HP: MAUVE vs. ROUGE-1", fontsize=13, fontweight="bold")

    if all_x:
        xmin, xmax = min(all_x), max(all_x)
        xpad = (xmax - xmin) * 0.08 if xmax > xmin else 0.02
        ax.set_xlim(max(0.0, xmin - xpad), xmax + xpad)
    if all_y:
        ymin, ymax = min(all_y), max(all_y)
        ypad = (ymax - ymin) * 0.12 if ymax > ymin else 0.03
        ymax_limit = ymax + ypad if args.exp_axes else min(1.02, ymax + ypad)
        ax.set_ylim(max(0.0, ymin - ypad), ymax_limit)

    ax.legend(frameon=True, fontsize=9, ncol=2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Read {len(mauve_paths)} MAUVE score files.")
    print(f"Plotted {sum(len(v) for v in points_by_method.values())} method/k points.")
    print("ROUGE-1 uses first occurrence per duplicate id.")
    if n_examples:
        print(f"Input MAUVE score files report n_examples={sorted(n_examples)}.")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot HP MAUVE against ROUGE-1.")
    parser.add_argument(
        "--input-glob",
        default="mauve_scores_risky_hp_gen_*_gold_gpt2.json",
        help="Glob for HP MAUVE score JSON files.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Directory containing HP literal-copying result JSON files.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(FIGURES_DIR, "hp_mauve_vs_rouge1.png"),
        help="Output PNG path.",
    )
    parser.add_argument(
        "--exp-axes",
        action="store_true",
        help="Plot exp(ROUGE-1) against exp(MAUVE).",
    )
    main(parser.parse_args())
