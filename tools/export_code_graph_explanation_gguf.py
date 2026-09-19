#!/usr/bin/env python3
"""Merge the final Code Graph Explanation LoRA and export it as GGUF.

Run from the project root on the training server::

    python tools/export_code_graph_explanation_gguf.py

The saved LoRA's ``adapter_config.json`` identifies the original base model.
Unsloth loads that base model, applies the cumulative adapter weights, merges
them during export, and writes an Ollama-compatible quantized GGUF file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = (
    ROOT_DIR
    / "lora_model_code_graph_explanation_lora_model_hierarchical_from_"
    "lora_model_code_repair_unsloth_qwen3_4b_instruct_2507_unsloth_bnb_4bit"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "export" / "code_graph_explanation_q4_k_m"
REQUIRED_ADAPTER_FILES = ("adapter_config.json",)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge a trained Code Graph Explanation LoRA with its Qwen base "
            "model and export an Ollama-compatible GGUF."
        )
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Final LoRA adapter directory produced by train_code_graph_explanation.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="New or empty directory in which Unsloth writes the GGUF.",
    )
    parser.add_argument(
        "--quantization",
        default="q4_k_m",
        choices=("q4_k_m", "q5_k_m", "q6_k", "q8_0", "f16", "bf16"),
    )
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load the base model in 4-bit before merging to reduce GPU memory use.",
    )
    parser.add_argument(
        "--minimum-free-gb",
        type=float,
        default=15.0,
        help="Abort if the output filesystem has less free space than this value.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and print the export plan without loading the model.",
    )
    args = parser.parse_args(argv)
    if args.max_seq_length < 1024:
        parser.error("--max-seq-length must be at least 1024")
    if args.minimum_free_gb < 0:
        parser.error("--minimum-free-gb cannot be negative")
    return args


def adapter_weight_files(model_dir: Path) -> list[Path]:
    patterns = ("adapter_model*.safetensors", "adapter_model*.bin")
    return sorted({path for pattern in patterns for path in model_dir.glob(pattern)})


def read_adapter_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "adapter_config.json"
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid adapter configuration {config_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{config_path} must contain a JSON object.")
    return payload


def validate_paths(model_dir: Path, output_dir: Path, minimum_free_gb: float) -> dict[str, Any]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"LoRA model directory does not exist: {model_dir}")
    for filename in REQUIRED_ADAPTER_FILES:
        path = model_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required adapter file does not exist: {path}")
    weights = adapter_weight_files(model_dir)
    if not weights:
        raise FileNotFoundError(
            f"No adapter_model*.safetensors or adapter_model*.bin found in {model_dir}"
        )

    model_resolved = model_dir.resolve()
    output_resolved = output_dir.resolve()
    if output_resolved == model_resolved or model_resolved in output_resolved.parents:
        raise ValueError("--output-dir must not be the adapter directory or inside it.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Choose a new directory "
            "so an earlier export cannot be mistaken for the new model."
        )

    output_parent = output_dir.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(output_parent).free
    free_gb = free_bytes / (1024**3)
    if free_gb < minimum_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.2f} GiB is free under {output_parent}; "
            f"at least {minimum_free_gb:.2f} GiB is required."
        )

    config = read_adapter_config(model_dir)
    return {
        "model_dir": str(model_resolved),
        "output_dir": str(output_resolved),
        "adapter_weights": [str(path.resolve()) for path in weights],
        "base_model": config.get("base_model_name_or_path", "unknown"),
        "free_gb": free_gb,
    }


def print_plan(plan: dict[str, Any], args: argparse.Namespace) -> None:
    print("GGUF export plan")
    print(f"  LoRA adapter:  {plan['model_dir']}")
    print(f"  Base model:    {plan['base_model']}")
    print(f"  Adapter files: {len(plan['adapter_weights'])}")
    print(f"  Output:        {plan['output_dir']}")
    print(f"  Quantization:  {args.quantization}")
    print(f"  Context:       {args.max_seq_length}")
    print(f"  Free space:    {plan['free_gb']:.2f} GiB")


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    plan = validate_paths(model_dir, output_dir, args.minimum_free_gb)
    print_plan(plan, args)
    if args.dry_run:
        print("Dry run complete; no model was loaded and no GGUF was created.")
        return

    # Import after validation so --dry-run works even outside the CUDA training
    # environment and catches path/disk problems before allocating GPU memory.
    from unsloth import FastLanguageModel

    print("Loading the base model and cumulative LoRA adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_dir),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    print("Merging adapter weights and exporting GGUF...")
    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=args.quantization,
    )

    gguf_files = sorted(output_dir.rglob("*.gguf"))
    if not gguf_files:
        raise RuntimeError(
            f"Unsloth returned without creating a GGUF under {output_dir}."
        )

    print("Export complete.")
    for path in gguf_files:
        size_gb = path.stat().st_size / (1024**3)
        print(f"  {path} ({size_gb:.2f} GiB)")
    modelfiles = sorted(output_dir.rglob("Modelfile"))
    for path in modelfiles:
        print(f"  Ollama Modelfile: {path}")


if __name__ == "__main__":
    main()
