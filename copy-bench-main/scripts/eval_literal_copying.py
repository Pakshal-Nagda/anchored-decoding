import json
from tqdm import tqdm
import os

from rouge_score import rouge_scorer
from datasketch import MinHash


def _minhash_similarity(tokens1, tokens2, num_perm=128, ngram=3):
    """MinHash Jaccard similarity over word n-grams."""
    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return [' '.join(tokens)] if tokens else ['']
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

    m1 = MinHash(num_perm=num_perm)
    m2 = MinHash(num_perm=num_perm)
    for gram in get_ngrams(tokens1, ngram):
        m1.update(gram.encode('utf8'))
    for gram in get_ngrams(tokens2, ngram):
        m2.update(gram.encode('utf8'))
    return m1.jaccard(m2)


def _word_acs(words1, words2):
    """Word-level Accumulated Common Substrings.

    Greedily sums lengths of non-overlapping maximal common word substrings,
    longest first.
    """
    m, n = len(words1), len(words2)
    if m == 0 or n == 0:
        return 0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    matches = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            if dp[i][j] > 0:
                matches.append((dp[i][j], i - 1, j - 1))

    matches.sort(key=lambda x: -x[0])
    used1, used2 = set(), set()
    total = 0
    for length, end1, end2 in matches:
        start1, start2 = end1 - length + 1, end2 - length + 1
        range1 = set(range(start1, end1 + 1))
        range2 = set(range(start2, end2 + 1))
        if not range1 & used1 and not range2 & used2:
            total += length
            used1 |= range1
            used2 |= range2
    return total


def _char_lcs(s1, s2):
    """Character-level Longest Common Subsequence length (space-optimized DP)."""
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def main(args):
    input_path = args.input
    output_path = args.output

    with open(input_path, "r") as f:
        results_list = json.load(f)

    # Load already-scored instances keyed by id
    scored = {}
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing = json.load(f)
        scored = {inst["id"]: inst for inst in existing if "score_rouge_1" in inst}

    pending = [inst for inst in results_list if inst["id"] not in scored]
    if not pending:
        print(f"All {len(results_list)} instances already scored, skipping.")
        return

    print(f"Scoring {len(pending)} new instances ({len(scored)} already done).")
    rouge_model = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    for inst in tqdm(pending):
        label_text, output_text = inst["reference"], inst["output"]
        try:
            label_tokens = rouge_model._tokenizer.tokenize(label_text)[:args.words]
            output_tokens = rouge_model._tokenizer.tokenize(output_text)[:args.words]

            rouge_l_dict = rouge_scorer._score_lcs(label_tokens, output_tokens)
            lcs_table = rouge_scorer._lcs_table(label_tokens, output_tokens)
            rouge_l = rouge_l_dict.fmeasure
            lcs = lcs_table[-1][-1]

            rouge_dict = rouge_model.score(output_text, label_text)
            rouge_1 = rouge_dict["rouge1"].fmeasure

            minhash = _minhash_similarity(label_tokens, output_tokens)
            acs = _word_acs(label_tokens, output_tokens)

            label_char = ' '.join(label_tokens)
            output_char = ' '.join(output_tokens)
            lcs_char = _char_lcs(label_char, output_char)

        except Exception as e:
            rouge_1 = 0.0
            rouge_l = 0.0
            lcs = 0
            minhash = 0.0
            acs = 0
            lcs_char = 0

        inst["score_rouge_1"] = rouge_1
        inst["score_rouge_l"] = rouge_l
        inst["score_lcs"] = lcs
        inst["score_minhash"] = minhash
        inst["score_acs"] = acs
        inst["score_lcs_char"] = lcs_char

    # Merge newly scored instances with previously scored ones
    scored.update({inst["id"]: inst for inst in pending})
    merged = [scored.get(inst["id"], inst) for inst in results_list]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(merged, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--words", type=int, default=50)
    args = parser.parse_args()
    main(args)
