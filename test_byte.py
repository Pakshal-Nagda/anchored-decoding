#!/usr/bin/env python3
"""Interactive Byte-level Anchored Decoding demo."""

import typer
import torch
from anchoreddecode import BytewiseAnchoredDecodingFactory
from transformers import GenerationConfig, AutoModelForCausalLM, logging as hf_logging

hf_logging.set_verbosity_error()
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.status import Status

app = typer.Typer(help="Interactive Byte-level Anchored Decoding demo")
console = Console()


@app.command()
def main(
    safe_model: str = typer.Option(
        "common-pile/comma-v0.1-2t",
        "--safe",
        "-s",
        help="Safe model path or HF identifier",
    ),
    risky_model: str = typer.Option(
        "meta-llama/Llama-3.1-70B",
        "--risky",
        "-r",
        help="Risky model path or HF identifier",
    ),
    k_radius: float = typer.Option(
        0.5,
        "--k-radius",
        "-k",
        help="KL budget per step (byte-level, typically smaller)",
    ),
    max_new_tokens: int = typer.Option(
        128, "--max-tokens", "-n", help="Maximum new tokens to generate"
    ),
    temperature: float = typer.Option(
        0.7, "--temperature", "-t", help="Sampling temperature"
    ),
    compare: bool = typer.Option(
        True,
        "--compare/--no-compare",
        help="Also generate with vanilla base models for comparison",
    ),
):
    """Interactive byte-level anchored decoding. Type prompts and get responses."""
    # Check GPU availability
    num_gpus = torch.cuda.device_count()
    if num_gpus < 2:
        console.print(
            f"[bold red]Error: This demo requires 2 GPUs, but only {num_gpus} found."
        )
        console.print(
            "For optimal performance, each model should be on a separate GPU."
        )
        raise typer.Exit(1)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    console.print(f"[bold green]Found {num_gpus} GPUs: {', '.join(gpu_names)}")

    console.print(f"[bold blue]Loading models...")
    console.print(f"  [green]Safe model: {safe_model}[/green]")
    console.print(f"  [red]Risky model: {risky_model}[/red]")
    console.print(f"  [cyan]k_radius: {k_radius}[/cyan]")

    common_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    safe_model_obj = AutoModelForCausalLM.from_pretrained(safe_model, **common_kwargs)
    risky_model_obj = AutoModelForCausalLM.from_pretrained(risky_model, **common_kwargs)

    console.print(f"[bold blue]Initializing BytewiseAnchoredDecodingFactory...")
    factory = BytewiseAnchoredDecodingFactory.from_pretrained(
        safe_model=safe_model_obj,
        risky_model=risky_model_obj,
        safe_model_path=safe_model,
        risky_model_path=risky_model,
        k_radius=k_radius,
    )

    # Ensure tokenizers have pad tokens and use left-padding
    for tcs in [factory.tcs_safe, factory.tcs_risky]:
        if tcs.tokenizer.pad_token is None:
            tcs.tokenizer.pad_token = tcs.tokenizer.eos_token
        tcs.tokenizer.padding_side = "left"

    console.print()
    console.print(
        "[bold green]Ready! Enter prompts below. Type 'quit' or Ctrl+C to exit."
    )
    console.print()

    max_length = 2048  # Model context limit

    # Interactive loop
    while True:
        try:
            prompt = Prompt.ask(
                "[bold cyan]Prompt[/bold cyan] [dim](Ctrl+C to exit)[/dim]"
            )

            if prompt.lower() in ("quit", "exit", "q"):
                console.print("[bold yellow]Goodbye!")
                break

            if not prompt.strip():
                continue

            prompts = [prompt]

            # Use safe tokenizer to estimate input length for capping
            safe_inputs = factory.tcs_safe.tokenizer(
                prompts, return_tensors="pt", padding=True
            )
            input_len = safe_inputs.input_ids.shape[1]

            # Cap max_new_tokens to not exceed max_length
            effective_max_new_tokens = min(max_new_tokens, max_length - input_len)
            if effective_max_new_tokens <= 0:
                console.print(
                    f"[bold red]Prompt too long ({input_len} tokens). Max context is {max_length}.[/bold red]"
                )
                continue

            config = GenerationConfig(
                max_new_tokens=effective_max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )

            # Generate baseline outputs if comparing
            if compare:
                safe_config = GenerationConfig(
                    max_new_tokens=effective_max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=factory.tcs_safe.tokenizer.pad_token_id,
                    eos_token_id=factory.tcs_safe.tokenizer.eos_token_id,
                )
                with console.status(
                    "[bold yellow]Generating with Safe LM...", spinner="aesthetic"
                ):
                    output_safe = safe_model_obj.generate(
                        input_ids=safe_inputs.input_ids.to(safe_model_obj.device),
                        attention_mask=safe_inputs.attention_mask.to(
                            safe_model_obj.device
                        ),
                        generation_config=safe_config,
                    )
                risky_inputs = factory.tcs_risky.tokenizer(
                    prompts, return_tensors="pt", padding=True
                )
                risky_config = GenerationConfig(
                    max_new_tokens=effective_max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=factory.tcs_risky.tokenizer.pad_token_id,
                    eos_token_id=factory.tcs_risky.tokenizer.eos_token_id,
                )
                with console.status(
                    "[bold red]Generating with Risky LM...", spinner="aesthetic"
                ):
                    output_risky = risky_model_obj.generate(
                        input_ids=risky_inputs.input_ids.to(risky_model_obj.device),
                        attention_mask=risky_inputs.attention_mask.to(
                            risky_model_obj.device
                        ),
                        generation_config=risky_config,
                    )

            anchored_config = GenerationConfig(
                max_new_tokens=effective_max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )
            with console.status(
                "[bold green]Generating with Byte-level Anchored Decoding...",
                spinner="aesthetic",
            ):
                output_anchored = factory.generate(
                    text=prompts, generation_config=anchored_config
                )

            # Display results
            console.print()
            if compare:
                safe_text = factory.tcs_safe.tokenizer.decode(
                    output_safe[0], skip_special_tokens=True
                )[len(prompt) :]
                risky_text = factory.tcs_risky.tokenizer.decode(
                    output_risky[0], skip_special_tokens=True
                )[len(prompt) :]
                console.print(
                    Panel(
                        Text(safe_text, style="bold yellow"),
                        title=f"Safe Model ({safe_model})",
                        border_style="yellow",
                    )
                )
                console.print(
                    Panel(
                        Text(risky_text, style="bold red"),
                        title=f"Risky Model ({risky_model})",
                        border_style="red",
                    )
                )

            anchored_text = output_anchored.text[0][len(prompt) :]
            console.print(
                Panel(
                    Text(anchored_text, style="bold green"),
                    title="Byte-level Anchored Decoding",
                    border_style="green",
                )
            )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Goodbye!")
            break


if __name__ == "__main__":
    app()
