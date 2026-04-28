import sys

from compute_mauve_scores import main, MAUVE_DIR


if __name__ == "__main__":
    default_args = [
        "--model",
        "gpt2-large",
        "--output",
        f"{MAUVE_DIR}/mauve_scores_risky_gold_gpt2_large.json",
    ]
    main_args = default_args + sys.argv[1:]

    import argparse
    from compute_mauve_scores import RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Compute risky-gold MAUVE scores with gpt2-large features."
    )
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--output_dir", default=MAUVE_DIR)
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="gpt2-large", help="MAUVE featurizer model.")
    parser.add_argument("--device_id", type=int, default=0, help="GPU id, or -1 for CPU.")
    parser.add_argument("--max_text_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args(main_args))
