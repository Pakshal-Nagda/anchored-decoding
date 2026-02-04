from typing import List, Optional, Union
from dataclasses import dataclass
import random
import warnings
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from .byte_sampling.byte_conditioning import ByteConditioning
from .byte_sampling.distillation import (
    rolling_byte_logprobs,
    rolling_byte_logprobs_batched,
)
from .byte_utils import safe_kl_pd_pc, solve_optimization_newton
from .byte_sampling.utils import sample_from_logits, sample_from_prob_tree

TOKEN_TO_BYTE = 4


def get_prefix_debt_from_llrs(
    llrs: list[dict],
    prompts: list[Union[str, bytes]],
    n: int = 7,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Compute prefix debt from precomputed per-byte LLRs.

    Args:
        llrs: List of dicts with 'llr_per_byte' keys (as saved by precompute_llrs.py).
              Must be in the same order as prompts (positional matching).
        prompts: List of prompts (strings or bytes) for each batch element
        n: Number of top LLR values to average
        device: Device to put the result tensor on

    Returns:
        tensor of shape [batch_size] with prefix debt for each sequence
    """
    bsize = len(prompts)
    assert (
        len(llrs) == bsize
    ), f"LLR list length {len(llrs)} != prompt batch size {bsize}"

    prefix_debts = []

    for i, (llr_item, prompt) in enumerate(zip(llrs, prompts)):
        llr_per_byte = llr_item["llr_per_byte"]

        if len(llr_per_byte) == 0:
            prefix_debts.append(0.0)
            continue

        # Convert to tensor for top-k operation
        llr_tensor = torch.tensor(llr_per_byte, dtype=torch.float32)

        # Take top-n LLRs
        n_eff = min(n, len(llr_per_byte))
        topv, _ = llr_tensor.topk(n_eff, largest=True)

        # Clamp to 0 so "good" tokens (negative LLR) don't reduce the debt
        debt = topv.clamp(min=0.0).mean().item()
        prefix_debts.append(debt)

    return torch.tensor(prefix_debts, dtype=torch.float32, device=device)


@dataclass
class BytewiseGenerateOutput:
    """Output container for bytewise generation, mimics transformers GenerateDecoderOnlyOutput."""

    sequences: torch.Tensor
    text: List[str]


class BytewiseAnchoredDecodingFactory:
    def __init__(
        self,
        tcs_safe: ByteConditioning,
        tcs_risky: ByteConditioning,
        k_radius: float = 1.0,
        **kwargs,
    ):
        self.tcs_safe = tcs_safe
        self.tcs_risky = tcs_risky
        self.k_radius = k_radius
        self.kwargs = kwargs
        self._last_sampler = None

    @classmethod
    def from_pretrained(
        cls,
        safe_model_path: Optional[str] = None,
        risky_model_path: Optional[str] = None,
        safe_model: Optional[torch.nn.Module] = None,
        risky_model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        k_radius: float = 1.0,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Create a BytewiseAnchoredDecodingFactory from model paths or pre-loaded models.

        Args:
            safe_model_path: Path to safe model (used if safe_model not provided)
            risky_model_path: Path to risky model (used if risky_model not provided)
            safe_model: Pre-loaded safe model
            risky_model: Pre-loaded risky model
            tokenizer: Tokenizer (if None, loaded from safe_model_path)
            k_radius: KL budget radius
            device: Device to use
            torch_dtype: Torch dtype for models
            device_map: Device map for model loading
            trust_remote_code: Whether to trust remote code
            load_in_4bit: Load in 4-bit quantization
            load_in_8bit: Load in 8-bit quantization
            verbose: Print loading info
            **kwargs: Additional kwargs passed to ByteConditioning
        """
        load_kwargs = dict(
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

        # Create ByteConditioning for safe model
        if safe_model is not None:
            # Model object provided - need tokenizer
            safe_tokenizer = (
                tokenizer
                if tokenizer is not None
                else AutoTokenizer.from_pretrained(
                    safe_model_path or safe_model.config._name_or_path
                )
            )
            tcs_safe = ByteConditioning(safe_model, tokenizer=safe_tokenizer)
            if verbose:
                print(f"[INFO] Created ByteConditioning for safe model (from object)")
        elif safe_model_path is not None:
            # Load from path
            tcs_safe = ByteConditioning(safe_model_path, load_kwargs=load_kwargs)
            if verbose:
                print(
                    f"[INFO] Created ByteConditioning for safe model from: {safe_model_path}"
                )
        else:
            raise ValueError("Either safe_model or safe_model_path must be provided")

        # Create ByteConditioning for risky model
        if risky_model is not None:
            # Model object provided - need its own tokenizer
            risky_tokenizer = AutoTokenizer.from_pretrained(
                risky_model_path or risky_model.config._name_or_path
            )
            tcs_risky = ByteConditioning(risky_model, tokenizer=risky_tokenizer)
            if verbose:
                print(f"[INFO] Created ByteConditioning for risky model (from object)")
        elif risky_model_path is not None:
            # Load from path
            tcs_risky = ByteConditioning(risky_model_path, load_kwargs=load_kwargs)
            if verbose:
                print(
                    f"[INFO] Created ByteConditioning for risky model from: {risky_model_path}"
                )
        else:
            raise ValueError("Either risky_model or risky_model_path must be provided")

        if verbose:
            print(
                f"[INFO] BytewiseAnchoredDecodingFactory initialized with k_radius={k_radius}"
            )

        return cls(tcs_safe=tcs_safe, tcs_risky=tcs_risky, k_radius=k_radius, **kwargs)

    def get_bytewise_sampler(self, batch_size):
        sampler = BytewiseAnchoredDecoding(
            batch_size,
            tcs_safe=self.tcs_safe,
            tcs_risky=self.tcs_risky,
            k_radius=self.k_radius,
            **self.kwargs,
        )
        self._last_sampler = sampler
        return sampler

    def generate(
        self,
        text: Optional[List[str]] = None,
        input_ids: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: int = 100,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        seed: Optional[int] = None,
        k_radius: Optional[float] = None,
        stop_strings: tuple = (),
        use_prefix_debt: bool = True,
        prefix_debt_n: int = 5,
        log_kl_stats: bool = False,
        **kwargs,
    ) -> BytewiseGenerateOutput:
        """
        Generate text using bytewise anchored decoding.

        Args:
            text: List of prompt strings
            input_ids: Alternative to text - token IDs (will be decoded to text)
            generation_config: HuggingFace GenerationConfig (extracts max_new_tokens, temperature, etc.)
            max_new_tokens: Maximum new tokens to generate
            do_sample: Whether to sample (vs greedy)
            temperature: Sampling temperature (applied before fusion)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty (>1.0 penalizes repeats, applied before fusion)
            seed: Random seed
            k_radius: Override factory k_radius for this generation
            stop_strings: Strings to stop generation at
            use_prefix_debt: Whether to use prefix debt
            prefix_debt_n: Number of top LLRs to average for prefix debt
            log_kl_stats: Whether to log KL statistics
            **kwargs: Additional kwargs passed to generate_byte

        Returns:
            BytewiseGenerateOutput with .sequences (token tensors) and .text (decoded strings)
        """
        # Extract settings from generation_config if provided
        if generation_config is not None:
            max_new_tokens = (
                getattr(generation_config, "max_new_tokens", max_new_tokens)
                or max_new_tokens
            )
            do_sample = getattr(generation_config, "do_sample", do_sample)
            temperature = (
                getattr(generation_config, "temperature", temperature) or temperature
            )
            repetition_penalty = (
                getattr(generation_config, "repetition_penalty", repetition_penalty)
                or repetition_penalty
            )
            top_k = getattr(generation_config, "top_k", top_k)
            top_p = getattr(generation_config, "top_p", top_p)

        # Handle input_ids -> text conversion
        if text is None:
            if input_ids is None:
                raise ValueError("Either text or input_ids must be provided")
            # Decode input_ids to text using safe model's tokenizer
            text = self.tcs_safe.tokenizer.batch_decode(
                input_ids, skip_special_tokens=True
            )

        # Use factory k_radius if not overridden
        if k_radius is None:
            k_radius = self.k_radius

        # Temporarily update factory k_radius for this generation
        original_k_radius = self.k_radius
        self.k_radius = k_radius

        try:
            # Call generate_byte
            outputs = generate_byte(
                sampler_factory=self,
                prompts=text,
                max_new_bytes=max_new_tokens
                * TOKEN_TO_BYTE,  # Approximate bytes from tokens
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=seed,
                stop_strings=stop_strings,
                use_prefix_debt=use_prefix_debt,
                prefix_debt_n=prefix_debt_n,
                log_kl_stats=log_kl_stats,
                **kwargs,
            )
        finally:
            self.k_radius = original_k_radius

        # Combine prompts with generated text
        full_texts = [prompt + output for prompt, output in zip(text, outputs)]

        # Tokenize full outputs for .sequences compatibility
        tokenized = self.tcs_safe.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Handle both dict and BatchEncoding return types
        if isinstance(tokenized, dict):
            input_ids = tokenized["input_ids"]
        else:
            input_ids = tokenized.input_ids

        return BytewiseGenerateOutput(
            sequences=input_ids,
            text=full_texts,
        )


class BytewiseAnchoredDecoding:
    def __init__(
        self,
        batch_size,
        tcs_safe,
        tcs_risky,
        k_radius=1.0,
        log_kl_stats=False,
        **kwargs,
    ):
        self.batch_size = batch_size
        self.tcs_safe = tcs_safe
        self.tcs_risky = tcs_risky
        self.k_radius = k_radius
        self.kwargs = kwargs
        self.bs_safe = tcs_safe.get_bytewise_sampler(batch_size=batch_size)
        self.bs_risky = tcs_risky.get_bytewise_sampler(batch_size=batch_size)

        self.bss = [self.bs_safe, self.bs_risky]
        self.kwargs = kwargs
        self.eps_kl = 1e-3

        # KL statistics logging
        self.log_kl_stats = log_kl_stats
        self.kl_stats_history = []

    def get_k_radius(self):
        return self.k_radius

    # apply log transforms, then solve for fused distribution
    def get_dists(
        self,
        k_radius=None,
        return_components: bool = False,
        save_kl_to_safe: bool = False,
        mask_special_tokens: bool = False,
        **kwargs,
    ):
        if k_radius is None:
            k_radius = self.k_radius  # this is the local setting

        if isinstance(k_radius, (int, float)) and not save_kl_to_safe:
            if k_radius == 0.0:  # return safe
                safe_logits = self.bs_safe.get_dists(**kwargs)
                safe_logp = torch.log_softmax(safe_logits, -1)
                if return_components:
                    # Return 5 values for consistent interface: (fused, safe, risky, bc, bd)
                    # For k=0, fused=safe and weights are (1, 0)
                    risky_logits = self.bs_risky.get_dists(**kwargs)
                    risky_logp = torch.log_softmax(risky_logits, -1)
                    bc = torch.ones(safe_logp.shape[0], 1, device=safe_logp.device)
                    bd = torch.zeros(safe_logp.shape[0], 1, device=safe_logp.device)
                    return safe_logp, safe_logp, risky_logp, bc, bd
                return safe_logp
            elif k_radius == -1.0:  # return risky
                risky_logits = self.bs_risky.get_dists(**kwargs)
                risky_logp = torch.log_softmax(risky_logits, -1)
                if return_components:
                    safe_logits = self.bs_safe.get_dists(**kwargs)
                    safe_logp = torch.log_softmax(safe_logits, -1)
                    # Return 5 values for consistent interface: (fused, safe, risky, bc, bd)
                    # For k=-1, fused=risky and weights are (0, 1)
                    bc = torch.zeros(risky_logp.shape[0], 1, device=risky_logp.device)
                    bd = torch.ones(risky_logp.shape[0], 1, device=risky_logp.device)
                    return risky_logp, safe_logp, risky_logp, bc, bd
                else:
                    return risky_logp

        safe_logits = self.bs_safe.get_dists(**kwargs).float()
        risky_logits = self.bs_risky.get_dists(**kwargs).float()

        if save_kl_to_safe:  # always decode w/ risky model only
            log_pd = torch.log_softmax(risky_logits[:, :256], dim=-1)  # [B,256]
            log_pc = torch.log_softmax(safe_logits[:, :256], dim=-1)  # [B,256]
            kl_to_safe = safe_kl_pd_pc(log_pd, log_pc)

            return risky_logits, kl_to_safe

        if safe_logits.dim() == 3:
            safe_logits = safe_logits[:, -1, :]  # [B, V]
            risky_logits = risky_logits[:, -1, :]  # [B, V]
        elif safe_logits.dim() != 2:
            raise ValueError(f"Unsupported logits shape: {safe_logits.shape}")

        # Define a fused device; move computation there
        fuse_device = risky_logits.device
        safe_logits = safe_logits.to(fuse_device)
        risky_logits = risky_logits.to(fuse_device)

        if mask_special_tokens:
            safe_logits[:, 256:] = float("-inf")
            risky_logits[:, 256:] = float("-inf")

        valid = torch.isfinite(safe_logits) & torch.isfinite(risky_logits)

        # If your solve_optimization expects logits, keep them masked
        safe_m = safe_logits.masked_fill(~valid, float("-inf"))
        risky_m = risky_logits.masked_fill(~valid, float("-inf"))

        bc, bd = solve_optimization_newton(safe_m, risky_m, k_radius)

        # Ensure fusion weights bc/bd live with your logits
        bc = torch.as_tensor(bc, device=fuse_device, dtype=risky_m.dtype)
        bd = torch.as_tensor(bd, device=fuse_device, dtype=risky_m.dtype)

        # Make the assertion device/dtype-safe
        assert torch.allclose(bc + bd, torch.ones_like(bc), atol=1e-5, rtol=1e-5)

        bc = bc.view(-1, 1) if bc.dim() == 1 else bc
        bd = bd.view(-1, 1) if bd.dim() == 1 else bd

        # Geometric fusion: logits = w_c * logits_c + w_d * logits_d
        # Handle -inf in logits by temporarily replacing with 0, then re-masking
        safe_safe = torch.nan_to_num(safe_m, neginf=0.0)
        risky_safe = torch.nan_to_num(risky_m, neginf=0.0)
        fused_logits = bc * safe_safe + bd * risky_safe
        fused_logits = fused_logits.masked_fill(~valid, float("-inf"))

        fused_log_probs = torch.log_softmax(fused_logits, dim=-1)
        if return_components:
            safe_log_probs = torch.log_softmax(safe_m, -1)
            risky_log_probs = torch.log_softmax(risky_m, -1)
            return fused_log_probs, safe_log_probs, risky_log_probs, bc, bd
        return fused_log_probs

    def add_context(self, prompts: list[Union[str, bytes]]):
        for bs in self.bss:
            bs.add_context(prompts)


@torch.inference_mode()
def generate_byte(
    sampler_factory,
    prompts: list[str],
    min_new_bytes: int = 0,
    max_new_bytes: int = 100,
    do_sample: bool = True,
    temperature: float = 1.0,
    top_k: float | None = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    display: bool = False,
    stop_strings: tuple[str] = (),
    include_stop_str_in_output: bool = False,
    save_kl_to_safe: bool = False,
    use_prefix_debt: bool = True,
    use_precomputed_llrs: list[dict] = None,
    prefix_debt_n: int = None,
    scale_prefix_debt: bool = True,
    allow_special: bool = True,
    logprob_transforms: dict | None = None,
    log_kl_stats: bool = False,
):
    """
    Generate text using bytewise anchored decoding.

    Args:
        sampler_factory: BytewiseAnchoredDecodingFactory or sampler
        prompts: List of prompt strings
        min_new_bytes: Minimum bytes to generate before allowing stop
        max_new_bytes: Maximum bytes to generate
        do_sample: Whether to sample (vs greedy)
        temperature: Sampling temperature (applied before fusion for correct KL computation)
        top_k: Top-k sampling parameter (applied during sampling)
        top_p: Top-p (nucleus) sampling parameter (applied during sampling)
        repetition_penalty: Repetition penalty (applied before fusion, >1.0 penalizes repeats)
        seed: Random seed
        generator: PyTorch random generator
        display: Whether to print output as generated
        stop_strings: Strings to stop generation at
        include_stop_str_in_output: Include stop string in output
        save_kl_to_safe: Whether to save KL to safe model
        use_prefix_debt: Whether to use prefix debt
        use_precomputed_llrs: Precomputed LLRs for prefix debt (skips computation)
        prefix_debt_n: Number of top LLRs to average for prefix debt
        scale_prefix_debt: Whether to scale prefix debt by TOKEN_TO_BYTE
        allow_special: Allow special tokens (>= 256)
        logprob_transforms: Additional log-prob transformations (overrides temperature/repetition_penalty if set)
        log_kl_stats: Whether to log KL statistics

    Returns:
        List of generated strings (or tuple with KL history if save_kl_to_safe=True)
    """
    assert not isinstance(
        stop_strings, str
    ), "stop_strings should be a sequence of strings"
    stop_strings = tuple(sorted(stop_strings, key=len, reverse=True))
    assert not isinstance(prompts, str)
    assert seed is None or generator is None, "can pass only one of seed/generator"

    # Build logprob_transforms from temperature and repetition_penalty if not provided
    # These transforms are applied BEFORE fusion for correct KL constraint computation
    if logprob_transforms is None:
        logprob_transforms = {}
        if temperature != 1.0:
            logprob_transforms["temperature"] = temperature
        if repetition_penalty != 1.0:
            logprob_transforms["repetition_penalty"] = repetition_penalty
    # If logprob_transforms was explicitly provided, use it as-is (user override)

    bsize = len(prompts)
    assert not (display and bsize > 1)

    try:
        bs = sampler_factory.get_bytewise_sampler(batch_size=bsize)
    except AttributeError:
        bs = sampler_factory

    if log_kl_stats:
        bs.log_kl_stats = True
        bs.kl_stats_history = []

    device = bs.tcs_safe.device

    unfinished_sequences = torch.ones(bsize, device=device, dtype=torch.long)

    outputs = [[] for _ in range(bsize)]
    decode_bufs = [b"" for _ in range(bsize)]
    stop_found = [False for _ in range(bsize)]

    if display:
        print(prompts[0], end="", flush=True)

    if save_kl_to_safe:
        kl_to_safe_history = []

    ## k
    k_radius = bs.get_k_radius()
    if not save_kl_to_safe:
        assert (
            k_radius != -1.0 and k_radius != 0.0
        ), "Use generate_batched instead for your k-radius"
    cum_kl_spent = torch.zeros(bsize, device=device, dtype=torch.float32)

    # Compute prefix debt BEFORE add_context to enable KV cache reuse.
    # The fast method also adds the prompt to the main sampler's context via prefill reuse.
    context_already_added = False
    if use_prefix_debt and prefix_debt_n is not None and prefix_debt_n > 0:
        if use_precomputed_llrs is not None:
            prefix_debt = get_prefix_debt_from_llrs(
                use_precomputed_llrs, prompts, n=prefix_debt_n
            )
            # Precomputed LLRs don't add context, so we still need to call add_context
        else:
            prefix_debt = get_prefix_debt_bytewise(bs, prompts, n=prefix_debt_n)
            # Check if prefill reuse was enabled
            if getattr(bs, "_context_already_added", False):
                context_already_added = True
        if scale_prefix_debt:
            prefix_debt *= TOKEN_TO_BYTE

    else:
        prefix_debt = torch.zeros(bsize, device=device, dtype=torch.float32)

    # Only add context if not already added by prefix debt computation
    if not context_already_added:
        bs.add_context([prompt.encode() for prompt in prompts])

    for t_gen in range(max_new_bytes):

        try:
            if save_kl_to_safe:
                assert k_radius == -1, "kl_to_safe only works with k=-1"
                dists, kl_to_safe_list = bs.get_dists(
                    logprob_transforms=logprob_transforms, save_kl_to_safe=True
                )
                kl_to_safe_history.append(kl_to_safe_list)
            else:
                # Compute k_t:
                budget_so_far = (float(t_gen + 1) * float(k_radius)) - prefix_debt
                remaining = (budget_so_far - cum_kl_spent).clamp(
                    min=0.0
                )  # should be fp32
                k_t = remaining * unfinished_sequences.float()
                should_mask = not allow_special or t_gen < min_new_bytes
                dists, safe_log_probs, risky_log_probs, bc, bd = bs.get_dists(
                    logprob_transforms=logprob_transforms,
                    k_radius=k_t,
                    return_components=True,
                    mask_special_tokens=should_mask,
                )
                # Compute KL immediately on the distributions returned by get_dists
                # (before any further modifications)
                kl_step = safe_kl_pd_pc(dists, safe_log_probs).float()

                # Log KL statistics if enabled
                if getattr(bs, "log_kl_stats", False):
                    kl_to_risky = safe_kl_pd_pc(dists, risky_log_probs).float()
                    kl_pd_pc = safe_kl_pd_pc(risky_log_probs, safe_log_probs).float()

                    bs.kl_stats_history.append(
                        {
                            "step": t_gen,
                            "kl_to_safe": kl_step.detach().cpu().tolist(),
                            "kl_to_risky": kl_to_risky.detach().cpu().tolist(),
                            "kl_pd_pc": kl_pd_pc.detach().cpu().tolist(),
                            "bc": (
                                bc.squeeze(-1).detach().cpu().tolist()
                                if bc.dim() > 1
                                else bc.detach().cpu().tolist()
                            ),
                            "bd": (
                                bd.squeeze(-1).detach().cpu().tolist()
                                if bd.dim() > 1
                                else bd.detach().cpu().tolist()
                            ),
                            "k_t": k_t.detach().cpu().tolist(),
                            "cum_kl_spent": (
                                cum_kl_spent + kl_step * unfinished_sequences.float()
                            )
                            .detach()
                            .cpu()
                            .tolist(),
                            "budget_so_far": (
                                budget_so_far.detach().cpu().tolist()
                                if isinstance(budget_so_far, torch.Tensor)
                                else [budget_so_far] * bsize
                            ),
                        }
                    )
        except RecursionError:
            # Tree got too deep, use uniform distribution over bytes and continue
            warnings.warn(
                f"RecursionError at step {t_gen}, using fallback distribution",
                RuntimeWarning,
            )
            dists = torch.zeros(bsize, 257, device=device)
            dists[:, :256] = 0.0  # Uniform over bytes (log-prob 0 = equal probability)
            dists[:, 256] = -torch.inf  # Mask stop token
            if save_kl_to_safe:
                kl_to_safe_list = torch.zeros(bsize, device=device)
                kl_to_safe_history.append(kl_to_safe_list)
            else:
                # For non-save_kl mode, set safe_log_probs and kl_step to safe defaults
                safe_log_probs = dists.clone()
                kl_step = torch.zeros(bsize, device=device, dtype=torch.float32)

        # Apply transformations (e.g., zeroing special tokens) for sampling
        # Note: When mask_special_tokens=True was passed to get_dists, the logits were
        # already masked before log_softmax, so dists is already correct for byte-only.
        # However, we still need to ensure dists has -inf for special tokens for sampling.
        if not allow_special or t_gen < min_new_bytes:
            dists[:, 256:] = -torch.inf

        # init the generator late so we know which device to put it on
        if generator is None and seed is not None:
            generator = torch.Generator(device=dists.device).manual_seed(seed)

        new_bytes = sample_from_logits(
            dists,
            do_sample=do_sample,
            temperature=1.0,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        ).tolist()

        for i, new_byte in enumerate(new_bytes):
            if new_byte >= 256 and t_gen >= min_new_bytes:
                stop_found[i] = True
                unfinished_sequences[i] = 0

        new_bytes = [
            bytes([b]) if not sf else bytes() for b, sf in zip(new_bytes, stop_found)
        ]

        bs.add_context(new_bytes)

        for i, new_byte in enumerate(new_bytes):
            if stop_found[i]:
                continue
            try:
                decode_bufs[i] += new_byte
                char = decode_bufs[i].decode()
                outputs[i].append(char)
                if display:
                    print(char, end="", flush=True)
                decode_bufs[i] = b""
            except UnicodeDecodeError:
                pass

        if stop_strings:
            for i, output in enumerate(outputs):
                if stop_found[i]:
                    continue

                suffix = "".join(output[-max(map(len, stop_strings)) :])
                if suffix.endswith(stop_strings):
                    if t_gen < min_new_bytes:
                        continue
                    if not include_stop_str_in_output:
                        for stop in stop_strings:
                            if suffix.endswith(stop):
                                outputs[i] = output[: -len(stop)]
                                break

                    stop_found[i] = True
                    unfinished_sequences[i] = 0

        if all(stop_found):
            break

        # Update cum_kl_spent and BANK the KL spend (only when not in save_kl_to_safe mode)
        # NOTE: kl_step was computed BEFORE modifying dists (zeroing special tokens)
        if not save_kl_to_safe:
            mask = unfinished_sequences.bool()
            assert torch.all(
                kl_step[mask] <= k_t[mask] + bs.eps_kl
            ), f"Local KL violated: kl_step[mask]={kl_step[mask].tolist()}, k_t[mask]={k_t[mask].tolist()}, diff={(kl_step[mask] - k_t[mask]).tolist()}, eps={bs.eps_kl}"
            cum_kl_spent = cum_kl_spent + kl_step * unfinished_sequences.float()

    output = ["".join(output) for output in outputs]
    if save_kl_to_safe:
        return output, kl_to_safe_history
    else:
        return output


def get_prefix_debt_bytewise(
    bs, prompts: list[Union[str, bytes]], n: int = 5, reuse_prefill: bool = True
):
    """
    Optimized prefix debt computation using rolling_byte_logprobs.

    This version uses batched tree inference (single model call per model)
    instead of O(max_len) sequential calls, providing significant speedup
    for long prompts.

    When reuse_prefill=True (default), the KV cache from computing prefix debt
    is reused for generation, avoiding a redundant prefill pass. This sets
    bs._context_already_added = True to signal that add_context should be skipped.

    Complexity:
        - Original get_prefix_debt_bytewise: O(max_len × 2) model forward passes
        - This version: O(2) model forward passes (one per model)
        - With reuse_prefill=True: saves 2 additional prefill passes for generation

    Args:
        bs: BytewiseAnchoredDecoding sampler
        prompts: List of prompts (strings or bytes) for each batch element
        n: number of top LLR values to average
        reuse_prefill: If True, reuse the KV cache for generation (default True)

    Returns:
        tensor of shape [batch_size] with prefix debt for each sequence
    """
    device = bs.tcs_risky.device
    batch_size = len(prompts)

    if reuse_prefill:
        # Use batched version for prefill reuse - single RCM query for all prompts
        all_safe_logprobs, _ = rolling_byte_logprobs_batched(
            bs.tcs_safe, prompts, sampler=bs.bs_safe
        )
        all_risky_logprobs, all_texts = rolling_byte_logprobs_batched(
            bs.tcs_risky, prompts, sampler=bs.bs_risky
        )
    else:
        # Process individually without prefill reuse
        all_safe_logprobs = []
        all_risky_logprobs = []
        all_texts = []
        for prompt in prompts:
            safe_lp = rolling_byte_logprobs(bs.tcs_safe, prompt)
            risky_lp, text = rolling_byte_logprobs(
                bs.tcs_risky, prompt, return_processed_text=True
            )
            all_safe_logprobs.append(safe_lp)
            all_risky_logprobs.append(risky_lp)
            all_texts.append(text)

    prefix_debts = []

    for i in range(batch_size):
        safe_logprobs_257 = all_safe_logprobs[i]
        risky_logprobs_257 = all_risky_logprobs[i]
        text = all_texts[i]

        if len(text) == 0:
            prefix_debts.append(0.0)
            continue

        # Move to same device if needed
        safe_logprobs_257 = safe_logprobs_257.to(device)
        risky_logprobs_257 = risky_logprobs_257.to(device)

        # Re-normalize over just 256 bytes to match the original implementation.
        # Original does: log_softmax(dists[:, :256], dim=-1)
        # rolling_byte_logprobs returns log_softmax over 257 dims.
        # To convert: log(q_i) = log(p_i) - log(1 - p_256) where p_256 is stop token prob
        safe_log_stop = safe_logprobs_257[:, 256:257]
        risky_log_stop = risky_logprobs_257[:, 256:257]

        # log(1 - exp(log_stop)) = log(sum of byte probs) using numerically stable log1p
        safe_log_byte_sum = torch.log1p(-torch.exp(safe_log_stop))
        risky_log_byte_sum = torch.log1p(-torch.exp(risky_log_stop))

        # Re-normalize: subtract log of the sum of byte probs
        safe_logprobs = safe_logprobs_257[:, :256] - safe_log_byte_sum
        risky_logprobs = risky_logprobs_257[:, :256] - risky_log_byte_sum

        # Gather LLRs for actual bytes at each position
        byte_indices = torch.tensor([b for b in text], device=device, dtype=torch.long)
        positions = torch.arange(len(text), device=device)

        safe_lp = safe_logprobs[positions, byte_indices]
        risky_lp = risky_logprobs[positions, byte_indices]

        # Compute LLRs: log p_risky(byte) - log p_safe(byte)
        llrs = risky_lp - safe_lp

        # Exclude first byte (position 0) per the original algorithm
        llrs = llrs[1:]

        if len(llrs) == 0:
            prefix_debts.append(0.0)
            continue

        # Handle NaN and inf
        llrs = torch.nan_to_num(llrs, nan=0.0, posinf=0.0, neginf=-1e9)

        # Take top-n LLRs
        n_eff = min(n, len(llrs))
        topv, _ = llrs.topk(n_eff)

        # Clamp to 0 so "good" tokens (negative LLR) don't reduce the debt
        debt = topv.clamp(min=0.0).mean().item()
        prefix_debts.append(debt)

    # Signal that context was already added via prefill reuse
    if reuse_prefill:
        bs._context_already_added = True

    return torch.tensor(prefix_debts, dtype=torch.float32, device=device)
