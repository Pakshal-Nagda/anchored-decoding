#!/usr/bin/env python3
"""Interactive Token-level Anchored Decoding demo."""

import typer
import torch
from anchoreddecode import AnchoredDecodingFactory
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    logging as hf_logging,
)

hf_logging.set_verbosity_error()
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt
from rich.status import Status

app = typer.Typer(help="Interactive Token-level Anchored Decoding demo")
console = Console()


@app.command()
def main(
    safe_model: str = typer.Option(
        "jacquelinehe/tinycomma-1.8b-llama3-tokenizer",
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
    k_radius: float = typer.Option(2.5, "--k-radius", "-k", help="KL budget per step"),
    max_new_tokens: int = typer.Option(
        100, "--max-tokens", "-n", help="Maximum new tokens to generate"
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
    """Interactive token-level anchored decoding. Type prompts and get responses."""
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

    tokenizer = AutoTokenizer.from_pretrained(safe_model, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    common_kwargs = dict(device_map="auto", torch_dtype=torch.bfloat16)
    safe_model_obj = AutoModelForCausalLM.from_pretrained(safe_model, **common_kwargs)
    risky_model_obj = AutoModelForCausalLM.from_pretrained(risky_model, **common_kwargs)

    console.print(
        f"[bold blue]Initializing AnchoredDecodingFactory (k_radius={k_radius})..."
    )
    factory = AnchoredDecodingFactory.from_pretrained(
        safe_model=safe_model_obj,
        risky_model=risky_model_obj,
        tokenizer=tokenizer,
        k_radius=k_radius,
    )

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
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(
                safe_model_obj.device
            )
            input_len = inputs.input_ids.shape[1]

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
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Generate baseline outputs if comparing
            if compare:
                with console.status(
                    "[bold yellow]Generating with Safe LM...", spinner="aesthetic"
                ):
                    output_safe = safe_model_obj.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=config,
                    )
                with console.status(
                    "[bold red]Generating with Risky LM...", spinner="aesthetic"
                ):
                    output_risky = risky_model_obj.generate(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        generation_config=config,
                    )

            with console.status(
                "[bold green]Generating with Anchored Decoding...", spinner="aesthetic"
            ):
                output_anchored = factory.generate(
                    text=prompts, generation_config=config
                )

            # Display results
            def get_clean_output(sequences, idx=0):
                decoded = tokenizer.decode(sequences[idx], skip_special_tokens=True)
                return decoded[len(prompt) :] if decoded.startswith(prompt) else decoded

            console.print()
            if compare:
                console.print(
                    Panel(
                        Text(get_clean_output(output_safe), style="bold yellow"),
                        title=f"Safe Model ({safe_model})",
                        border_style="yellow",
                    )
                )
                console.print(
                    Panel(
                        Text(get_clean_output(output_risky), style="bold red"),
                        title=f"Risky Model ({risky_model})",
                        border_style="red",
                    )
                )

            console.print(
                Panel(
                    Text(
                        get_clean_output(output_anchored.sequences), style="bold green"
                    ),
                    title="Anchored Decoding",
                    border_style="green",
                )
            )
            console.print()

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Goodbye!")
            break


if __name__ == "__main__":
    app()
