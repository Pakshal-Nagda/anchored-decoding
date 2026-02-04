"""
Rolling byte log probabilities via batched tree inference.

This module provides efficient computation of byte-level log probabilities
across all positions in a text sequence using a single batched model call
instead of O(n) sequential calls.

This enables fast prefix debt computation with:
- O(2) forward passes instead of O(2N) for N-byte prompts
- KV cache reuse for generation (prefill reuse)
- Logit-gather optimization for memory efficiency
"""

import torch
from .byte_conditioning import ByteConditioning
from .radix_cache import RadixCacheManager
from .streaming_bpe import StreamingBPE
from .utils import scatter_logsumexp


def rolling_byte_logprobs(
    bc: ByteConditioning,
    text: bytes | str,
    sampler=None,
    batch_idx: int = 0,
    return_processed_text: bool = False,
):
    """
    Compute byte-level log probabilities for each position in the text.

    This function builds a tree structure encoding all byte positions,
    then uses RadixCacheManager to compute all log probabilities in a
    single batched inference call.

    Args:
        bc: ByteConditioning instance
        text: Input text (bytes or str)
        sampler: Optional BytewiseSampler instance. If provided, uses its RCM
                 and updates its internal state for prefill reuse.
        batch_idx: Batch index when using sampler (default 0)
        return_processed_text: If True, also return the processed bytes

    Returns:
        Tensor of shape [len(text), 257] containing log probabilities
        for each byte (0-255) plus a stop token probability at index 256
        at each position in the text.

        If return_processed_text=True, returns (logprobs, processed_bytes).
    """
    # Apply same normalization as add_context does
    if isinstance(text, str):
        if bc.btok.normalizer is not None:
            text = bc.btok.normalizer.normalize_str(text)
        text = text.encode()

    # Cache STOP_TOKENS on the ByteConditioning object to avoid rebuilding
    if not hasattr(bc, "_stop_tokens_cache"):
        bc._stop_tokens_cache = torch.tensor(
            [
                tid
                for tid, at in bc.tokenizer.added_tokens_decoder.items()
                if at.special
            ],
            device=bc.device,
        )
    STOP_TOKENS = bc._stop_tokens_cache

    S = bc.get_streaming_byte_tree()
    trunk, roots, trees = [bc.bos], [], []
    pointer = {}
    full_tree = {bc.bos: pointer}

    for i, b in enumerate(text):
        tree = S.eval_tree(inclusive=True, filter_tensors=False)
        roots.append(len(trunk))
        trees.append(tree)
        StreamingBPE.tree_update(pointer, tree, copy=True)

        new_tokens = S.push(b)
        for tid in new_tokens:
            pointer = pointer.setdefault(tid, {})
        trunk.extend(new_tokens)

    # Use sampler's RCM if provided (for prefill reuse), otherwise create temporary one
    if sampler is not None:
        rcm = sampler.rcm
        # Update sampler's internal state so it can skip add_context
        sampler.trunks[batch_idx] = trunk.copy()
        sampler.sbps[batch_idx] = S  # Transfer the StreamingBPE state
        sampler.lens[batch_idx] = len(text)
        # Compute trunk_lens: sum of byte lengths of tokens AFTER the BOS (trunk[0])
        # Use bc.vrev (not vrev_all) to match add_context behavior - special tokens return b""
        trunk_lens = sum(len(bc.vrev.get(tid, b"")) for tid in trunk[1:])
        sampler.trunk_lens[batch_idx] = trunk_lens
    else:
        rcm = RadixCacheManager(bc.model, bc.tokenizer)

    result = rcm.query([full_tree])[0]
    target_idx = 0

    def get_dists(eval_tree, lp_tree, past_bytes=0):
        byte_logprobs, stop_logprobs = [], []

        # walk the tree
        def extract_bytes(eval_tree, lp_tree, past_bytes=0):
            for tid, eval_subtree in eval_tree.items():
                lp_subtree = lp_tree[tid]
                if tid is None:
                    subset = eval_subtree
                    prompt_offset = target_idx - past_bytes

                    if prompt_offset == 0:
                        stop_logprobs.append(
                            torch.logsumexp(lp_subtree[STOP_TOKENS], 0)
                        )

                    selectors = bc.token_index_cache.get(prompt_offset)[subset]
                    lp_subset = lp_subtree[subset]

                    byte_logprobs.append(
                        scatter_logsumexp(lp_subset, selectors, dim_size=257)
                    )

                else:
                    extract_bytes(
                        eval_subtree,
                        lp_subtree,
                        past_bytes + len(bc.vrev_all[tid]),
                    )

        extract_bytes(eval_tree, lp_tree, past_bytes)
        stop_logprob = torch.logsumexp(torch.tensor(stop_logprobs, device=bc.device), 0)
        return torch.hstack(
            [
                torch.logsumexp(torch.vstack(byte_logprobs)[:, :-1], 0),
                stop_logprob,
            ]
        )

    lp_tree = result
    last_lp_root = 0
    past_bytes = -len(bc.vrev_all[bc.bos])
    dists = []
    for target_idx, (lp_root, eval_tree) in enumerate(zip(roots, trees)):
        for tid in trunk[last_lp_root:lp_root]:
            lp_tree = lp_tree[tid]
            past_bytes += len(bc.vrev_all[tid])
        dists.append(get_dists(eval_tree, lp_tree, past_bytes))
        last_lp_root = lp_root

    logprobs = torch.log_softmax(torch.vstack(dists), -1)
    if return_processed_text:
        return logprobs, text
    return logprobs


def rolling_byte_logprobs_batched(
    bc: ByteConditioning,
    texts: list[bytes | str],
    sampler=None,
):
    """
    Batched version of rolling_byte_logprobs for multiple texts.

    Processes all texts together in a single RCM query, enabling
    prefill reuse with batch_size > 1.

    Args:
        bc: ByteConditioning instance
        texts: List of input texts (bytes or str)
        sampler: Optional BytewiseSampler instance. If provided, uses its RCM
                 and updates its internal state for prefill reuse.
                 Must have batch_size == len(texts).

    Returns:
        List of tensors, each of shape [len(text_i), 257] containing log probabilities
        for each byte (0-255) plus a stop token probability at index 256.
        Also returns list of processed texts (bytes).
    """
    batch_size = len(texts)

    # Apply normalization to all texts
    processed_texts = []
    for text in texts:
        if isinstance(text, str):
            if bc.btok.normalizer is not None:
                text = bc.btok.normalizer.normalize_str(text)
            text = text.encode()
        processed_texts.append(text)

    # Cache STOP_TOKENS
    if not hasattr(bc, "_stop_tokens_cache"):
        bc._stop_tokens_cache = torch.tensor(
            [
                tid
                for tid, at in bc.tokenizer.added_tokens_decoder.items()
                if at.special
            ],
            device=bc.device,
        )
    STOP_TOKENS = bc._stop_tokens_cache

    # Build trees for all texts
    all_full_trees = []
    all_roots = []
    all_trees = []
    all_trunks = []
    all_sbps = []

    for text in processed_texts:
        S = bc.get_streaming_byte_tree()
        trunk, roots, trees = [bc.bos], [], []
        pointer = {}
        full_tree = {bc.bos: pointer}

        for b in text:
            tree = S.eval_tree(inclusive=True, filter_tensors=False)
            roots.append(len(trunk))
            trees.append(tree)
            StreamingBPE.tree_update(pointer, tree, copy=True)

            new_tokens = S.push(b)
            for tid in new_tokens:
                pointer = pointer.setdefault(tid, {})
            trunk.extend(new_tokens)

        all_full_trees.append(full_tree)
        all_roots.append(roots)
        all_trees.append(trees)
        all_trunks.append(trunk)
        all_sbps.append(S)

    # Use sampler's RCM if provided, otherwise create temporary one
    if sampler is not None:
        assert (
            sampler.batch_size == batch_size
        ), f"Sampler batch_size {sampler.batch_size} != len(texts) {batch_size}"
        rcm = sampler.rcm
        # Update sampler's internal state for all batch elements
        for i in range(batch_size):
            sampler.trunks[i] = all_trunks[i].copy()
            sampler.sbps[i] = all_sbps[i]
            sampler.lens[i] = len(processed_texts[i])
            trunk_lens = sum(len(bc.vrev.get(tid, b"")) for tid in all_trunks[i][1:])
            sampler.trunk_lens[i] = trunk_lens
    else:
        rcm = RadixCacheManager(bc.model, bc.tokenizer)

    # Single batched query for all trees
    results = rcm.query(all_full_trees)

    # Process results for each text
    all_logprobs = []
    for batch_idx in range(batch_size):
        text = processed_texts[batch_idx]
        roots = all_roots[batch_idx]
        trees = all_trees[batch_idx]
        trunk = all_trunks[batch_idx]
        result = results[batch_idx]

        if len(text) == 0:
            # Empty text - return empty tensor
            all_logprobs.append(torch.zeros(0, 257, device=bc.device))
            continue

        def get_dists(eval_tree, lp_tree, past_bytes, target_idx):
            byte_logprobs, stop_logprobs = [], []

            def extract_bytes(eval_tree, lp_tree, past_bytes=0):
                for tid, eval_subtree in eval_tree.items():
                    lp_subtree = lp_tree[tid]
                    if tid is None:
                        subset = eval_subtree
                        prompt_offset = target_idx - past_bytes

                        if prompt_offset == 0:
                            stop_logprobs.append(
                                torch.logsumexp(lp_subtree[STOP_TOKENS], 0)
                            )

                        selectors = bc.token_index_cache.get(prompt_offset)[subset]
                        lp_subset = lp_subtree[subset]

                        byte_logprobs.append(
                            scatter_logsumexp(lp_subset, selectors, dim_size=257)
                        )
                    else:
                        extract_bytes(
                            eval_subtree,
                            lp_subtree,
                            past_bytes + len(bc.vrev_all[tid]),
                        )

            extract_bytes(eval_tree, lp_tree, past_bytes)
            stop_logprob = torch.logsumexp(
                torch.tensor(stop_logprobs, device=bc.device), 0
            )
            return torch.hstack(
                [
                    torch.logsumexp(torch.vstack(byte_logprobs)[:, :-1], 0),
                    stop_logprob,
                ]
            )

        lp_tree = result
        last_lp_root = 0
        past_bytes = -len(bc.vrev_all[bc.bos])
        dists = []
        for target_idx, (lp_root, eval_tree) in enumerate(zip(roots, trees)):
            for tid in trunk[last_lp_root:lp_root]:
                lp_tree = lp_tree[tid]
                past_bytes += len(bc.vrev_all[tid])
            dists.append(get_dists(eval_tree, lp_tree, past_bytes, target_idx))
            last_lp_root = lp_root

        logprobs = torch.log_softmax(torch.vstack(dists), -1)
        all_logprobs.append(logprobs)

    return all_logprobs, processed_texts


__all__ = ["rolling_byte_logprobs", "rolling_byte_logprobs_batched"]
