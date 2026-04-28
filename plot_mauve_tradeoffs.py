r"""Plot MAUVE against aggregate normalized copying reduction.

Default run, using the GPT-2-large risky-gold MAUVE scores:
    python .\plot_mauve_tradeoffs.py

Other common runs:
    python .\plot_mauve_tradeoffs.py --mauve_scores mauve\mauve_scores_risky_gold_gpt2.json
    python .\plot_mauve_tradeoffs.py --loglog --output figures\mauve_tradeoff_loglog.png
    python .\plot_mauve_tradeoffs.py --exp-exp --output figures\mauve_tradeoff_gpt2_large_exp_exp.png

Useful arguments:
    --mauve_scores     Input JSON from compute_mauve_scores.py.
    --output           Output PNG path.
    --loglog           Use log-log axes; plots NCR + 1 so risky at NCR 0 is visible.
    --exp-exp          Plot exp(NCR / 100) vs exp(MAUVE).

The x-axis is NCR from a min-max normalized average of median copying metrics.
The y-axis is MAUVE, usually computed against risky model outputs as gold.
"""

import argparse
import json
import os

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt


RESULTS_DIR = "results"
FIGURES_DIR = "figures"

METRICS = [
    "score_rouge_1",
    "score_rouge_l",
    "score_lcs",
    "score_minhash",
    "score_acs",
    "score_lcs_char",
]

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


def load_result_data(filename):
    return {row["id"]: row for row in load_json(os.path.join(RESULTS_DIR, filename))}


def median_scores(data_by_id, allowed_ids):
    rows = [data_by_id[i] for i in allowed_ids if i in data_by_id]
    scores = {}
    for metric in METRICS:
        vals = [row[metric] for row in rows if metric in row]
        scores[metric] = float(np.median(vals)) if vals else np.nan
    return scores


def average_normalized_score(scores, metric_ranges):
    vals = []
    for metric in METRICS:
        score = scores[metric]
        lo, hi = metric_ranges[metric]
        if not np.isfinite(score) or abs(hi - lo) < 1e-12:
            continue
        vals.append((score - lo) / (hi - lo))
    return float(np.mean(vals)) if vals else np.nan


def ncr(risky_score, safe_score, method_score):
    denom = risky_score - safe_score
    if abs(denom) < 1e-12:
        return np.nan
    return (risky_score - method_score) / denom * 100.0


def plot_x(x, args):
    if args.exp_exp:
        return float(np.exp(x / 100.0))
    return x + 1.0 if args.loglog else x


def plot_y(y, args):
    return float(np.exp(y)) if args.exp_exp else y


def main(args):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })

    mauve_rows = load_json(args.mauve_scores)
    by_filename = {row["filename"]: row for row in mauve_rows}

    risky_data = load_result_data("risky_literal_results.json")
    safe_data = load_result_data("safe_literal_results.json")

    method_files = [
        row["filename"]
        for row in mauve_rows
        if not row.get("baseline") and row["method"] in METHODS
    ]
    common_ids = None
    for filename in method_files:
        ids = set(load_result_data(filename))
        common_ids = ids if common_ids is None else common_ids & ids
    if not common_ids:
        raise RuntimeError("No shared method IDs available for NCR calculation.")

    score_points = []
    for row in mauve_rows:
        filename = row["filename"]
        if not os.path.exists(os.path.join(RESULTS_DIR, filename)):
            continue
        data = load_result_data(filename)
        scores = median_scores(data, common_ids)
        score_points.append({**row, "scores": scores})

    metric_ranges = {}
    for metric in METRICS:
        vals = [point["scores"][metric] for point in score_points if np.isfinite(point["scores"][metric])]
        if not vals:
            raise RuntimeError(f"No finite values for {metric}.")
        metric_ranges[metric] = (min(vals), max(vals))

    risky_scores = next(point["scores"] for point in score_points if point["filename"] == "risky_literal_results.json")
    safe_scores = next(point["scores"] for point in score_points if point["filename"] == "safe_literal_results.json")
    risky_norm = average_normalized_score(risky_scores, metric_ranges)
    safe_norm = average_normalized_score(safe_scores, metric_ranges)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    all_xs = [plot_x(0, args), plot_x(100, args)]
    all_ys = []

    risky_mauve = by_filename["risky_literal_results.json"]["mauve"]
    safe_mauve = by_filename["safe_literal_results.json"]["mauve"]
    risky_y = plot_y(risky_mauve, args)
    safe_y = plot_y(safe_mauve, args)
    all_ys.extend([risky_y, safe_y])
    ax.scatter([plot_x(0, args)], [risky_y], color="#8c564b", marker="s", s=70, label="Risky LM", zorder=4)
    ax.scatter([plot_x(100, args)], [safe_y], color="#17becf", marker="s", s=70, label="Safe LM", zorder=4)

    for method, info in METHODS.items():
        points = []
        for point in score_points:
            if point.get("baseline") or point["method"] != method:
                continue
            method_norm = average_normalized_score(point["scores"], metric_ranges)
            x = ncr(risky_norm, safe_norm, method_norm)
            y = point["mauve"]
            if np.isfinite(x) and np.isfinite(y):
                x_to_plot = plot_x(x, args)
                y_to_plot = plot_y(y, args)
                if args.loglog and (x_to_plot <= 0 or y_to_plot <= 0):
                    continue
                points.append((point["k"], x_to_plot, y_to_plot))
                all_xs.append(x_to_plot)
                all_ys.append(y_to_plot)
        if not points:
            continue
        points.sort(key=lambda item: item[0])
        _, xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            color=info["color"],
            marker=info["marker"],
            label=info["label"],
            linewidth=1.8,
            markersize=6,
        )

    ax.set_title("MAUVE vs. Min-Max Avg. Median-Metric NCR", fontsize=13, fontweight="bold")
    if args.exp_exp:
        ax.set_xlabel(r"$e^{\mathrm{NCR}/100}$")
        ax.set_ylabel(r"$e^{\mathrm{MAUVE}}$")
    elif args.loglog:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("NCR of min-max avg. median copying metric + 1")
        ax.set_ylabel("MAUVE")
    else:
        ax.set_xlabel("NCR of min-max avg. median copying metric (%)")
        ax.set_ylabel("MAUVE")

    if args.loglog:
        ax.set_xlim(min(all_xs) * 0.9, max(all_xs) * 1.1)
        ax.set_ylim(min(all_ys) * 0.9, min(1.0, max(all_ys) * 1.1))
    elif args.exp_exp:
        xmin, xmax = min(all_xs), max(all_xs)
        xpad = (xmax - xmin) * 0.05 if xmax > xmin else 0.1
        ax.set_xlim(xmin - xpad, xmax + xpad)

        ymin, ymax = min(all_ys), max(all_ys)
        ypad = (ymax - ymin) * 0.08 if ymax > ymin else 0.1
        ax.set_ylim(ymin - ypad, ymax + ypad)
    else:
        xmin, xmax = min(all_xs), max(all_xs)
        xpad = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
        ax.set_xlim(xmin - xpad, xmax + xpad)

        ymin, ymax = min(all_ys), max(all_ys)
        ypad = (ymax - ymin) * 0.08 if ymax > ymin else 0.05
        ax.set_ylim(max(0, ymin - ypad), min(1, ymax + ypad))

    ax.legend(frameon=True, fontsize=9, ncol=2)
    fig.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.exp_exp:
        default_name = "mauve_tradeoff_exp_exp.png"
    elif args.loglog:
        default_name = "mauve_tradeoff_loglog.png"
    else:
        default_name = "mauve_tradeoff.png"
    out_path = args.output or os.path.join(args.output_dir, default_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MAUVE against aggregate normalized copying reduction.")
    parser.add_argument(
        "--mauve_scores",
        default="mauve/mauve_scores_risky_gold_gpt2_large.json",
        help="Input MAUVE score JSON. Defaults to the GPT-2-large risky-gold score file.",
    )
    parser.add_argument(
        "--output_dir",
        default=FIGURES_DIR,
        help="Directory where the plot is written when --output is not provided.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output PNG path.",
    )
    parser.add_argument(
        "--loglog",
        action="store_true",
        help="Use log-log axes; x-axis plots NCR + 1 to include the risky baseline.",
    )
    parser.add_argument(
        "--exp-exp",
        dest="exp_exp",
        action="store_true",
        help="Plot exp(NCR / 100) against exp(MAUVE).",
    )
    main(parser.parse_args())
