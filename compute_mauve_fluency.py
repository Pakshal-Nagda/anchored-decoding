r"""Compute MAUVE scores for all literal generation result files.

Default run, using GPT-2 features and GPU 0:
    python .\compute_mauve_scores.py

GPT-2-large is available through the wrapper:
    python .\compute_mauve_scores_gpt2_large.py

Useful arguments:
    --model            MAUVE feature model, e.g. gpt2 or gpt2-large.
    --device_id        GPU id to use; set -1 for CPU.
    --batch_size       Lower this if CUDA runs out of memory.
    --max_examples     Debug mode; score only the first N shared examples.
    --overwrite        Recompute even if the output JSON already exists.

This script treats risky_literal_results.json outputs as the gold distribution.
That means risky-vs-risky is written as MAUVE = 1.0, and every other file is
compared against the risky model outputs on the same shared IDs.
"""

import argparse
import json
import os
import re


RESULTS_DIR = "results"
MAUVE_DIR = "."


METHOD_LABELS = {
    "anchored": "Anchored Dec.",
    "clip": "CLIP",
    "renyi2": r"Renyi ($\alpha$=2)",
    "renyi5": r"Renyi ($\alpha$=5)",
    "renyi10": r"Renyi ($\alpha$=10)",
    "reversekl": "Reverse KL",
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def discover_result_files(results_dir):
    method_pattern = re.compile(
        r"^(?P<method>[a-z0-9]+)_(?P<k>0\.1|0\.5|1|5|10|20)_literal_results\.json$"
    )
    files = []
    for filename in sorted(os.listdir(results_dir)):
        path = os.path.join(results_dir, filename)
        if not os.path.isfile(path):
            continue

        if filename in {"risky_literal_results.json", "safe_literal_results.json"}:
            files.append({
                "filename": filename,
                "path": path,
                "method": filename.removesuffix("_literal_results.json"),
                "k": None,
                "label": "Risky LM" if filename.startswith("risky") else "Safe LM",
                "baseline": True,
            })
            continue

        match = method_pattern.match(filename)
        if not match:
            continue
        method = match.group("method")
        if method not in METHOD_LABELS:
            continue
        files.append({
            "filename": filename,
            "path": path,
            "method": method,
            "k": float(match.group("k")),
            "label": METHOD_LABELS[method],
            "baseline": False,
        })
    return files


def common_method_ids(file_infos):
    ids = None
    for info in file_infos:
        if info["baseline"]:
            continue
        data = load_json(info["path"])
        file_ids = {row["id"] for row in data}
        ids = file_ids if ids is None else ids & file_ids
    if not ids:
        raise RuntimeError("No shared IDs found across method result files.")
    return ids


def rows_for_file(path, allowed_ids, max_examples=None):
    rows = [row for row in load_json(path) if row.get("id") in allowed_ids]
    rows.sort(key=lambda row: row["id"])
    if max_examples is not None:
        rows = rows[:max_examples]
    return rows


def texts_for_file(path, allowed_ids, max_examples=None):
    rows = rows_for_file(path, allowed_ids, max_examples)
    references = [row["reference"] for row in rows]
    outputs = [row["output"] for row in rows]
    return references, outputs, [row["id"] for row in rows]


import mauve
import numpy as np

def get_fluency_at_diversity(mauve_out, target_diversity=0.01):
    """
    Computes MAUVE and extracts Precision (Fluency) at a specific Recall (Diversity).
    
    target_diversity: A float between 0 and 1 (e.g., 0.01 for 1%)
    """
    # 2. Extract the divergence frontier
    # mauve_out.divergence_curve is a numpy array of shape (2, num_points)
    # Index 0 contains Precision (p) values, Index 1 contains Recall (q) values
    precision = mauve_out.divergence_curve[0, :]
    recall = mauve_out.divergence_curve[1, :]
    
    # 3. Sort the arrays by recall to prepare for interpolation
    # np.interp requires the x-axis (recall) to be monotonically increasing
    sort_indices = np.argsort(recall)
    sorted_recall = recall[sort_indices]
    sorted_precision = precision[sort_indices]
    
    # 4. Interpolate the exact precision at the target recall
    precision_at_target = np.interp(target_diversity, sorted_recall, sorted_precision)
    return precision_at_target

def main(args):
    try:
        import mauve
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: mauve-text. Install it with `pip install mauve-text`."
        ) from exc

    file_infos = discover_result_files(args.results_dir)
    if not file_infos:
        raise RuntimeError(f"No literal result files found in {args.results_dir!r}.")

    allowed_ids = common_method_ids(file_infos)
    print(f"[INFO] Using {len(allowed_ids)} shared method IDs.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = args.output or os.path.join(
        args.output_dir, f"mauve_scores_risky_gold_{args.model.replace('/', '_')}.json"
    )

    risky_path = os.path.join(args.results_dir, "risky_literal_results.json")
    risky_rows = rows_for_file(risky_path, allowed_ids, args.max_examples)
    risky_outputs_by_id = {row["id"]: row["output"] for row in risky_rows}

    existing = []
    done = set()
    if os.path.exists(output_path) and not args.overwrite:
        existing = load_json(output_path)
        done = {row["filename"] for row in existing}
        print(f"[INFO] Resuming from {output_path}; {len(done)} files already scored.")

    scores = list(existing)
    for info in file_infos:
        if info["filename"] in done:
            continue

        _, outputs, used_ids = texts_for_file(info["path"], allowed_ids, args.max_examples)
        gold_outputs = [risky_outputs_by_id[row_id] for row_id in used_ids]
        print(f"[INFO] Computing MAUVE for {info['filename']} on {len(used_ids)} examples.")

        if info["filename"] == "risky_literal_results.json":
            mauve_score = 1.0
        else:
            result = mauve.compute_mauve(
                p_text=gold_outputs,
                q_text=outputs,
                featurize_model_name=args.model,
                device_id=args.device_id,
                max_text_length=args.max_text_length,
                batch_size=args.batch_size,
                verbose=True,
            )
            mauve_score = get_fluency_at_diversity(result)

        scores.append({
            "filename": info["filename"],
            "method": info["method"],
            "label": info["label"],
            "k": info["k"],
            "baseline": info["baseline"],
            "mauve": mauve_score,
            "gold": "risky_literal_results.json",
            "model": args.model,
            "max_text_length": args.max_text_length,
            "n_examples": len(used_ids),
        })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2)

    print(f"[INFO] Saved {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute risky-gold MAUVE scores for literal generation result files."
    )
    parser.add_argument(
        "--results_dir",
        default=RESULTS_DIR,
        help="Directory containing *_literal_results.json files.",
    )
    parser.add_argument(
        "--output_dir",
        default=MAUVE_DIR,
        help="Directory where the MAUVE score JSON will be written.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional explicit output JSON path. Defaults to mauve/mauve_scores_risky_gold_<model>.json.",
    )
    parser.add_argument(
        "--model",
        default="gpt2",
        help="MAUVE featurizer model, e.g. gpt2 or gpt2-large.",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="GPU id to use. Set to -1 to force CPU.",
    )
    parser.add_argument(
        "--max_text_length",
        type=int,
        default=256,
        help="Maximum text length passed to MAUVE featurization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="MAUVE featurization batch size. Lower this if you hit CUDA OOM.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional debug limit on the number of shared examples scored per file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore an existing output JSON and recompute all files from scratch.",
    )
    main(parser.parse_args())
