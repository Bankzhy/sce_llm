#!/usr/bin/env python3
"""Fine-tune a model to identify Extract Method opportunities in Java methods."""

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT_DIR / "dataset" / "em_train.json"

EM_INSTRUCTION = """You are a Code refactoring expert.
Analyze the given method and identify all code regions that are suitable for Extract Method refactoring.
For each opportunity, locate the code region, propose a meaningful extracted method signature, and explain why the extraction improves the code.
Use the input method's original line numbering when locating a region.
Return only the identified Extract Method opportunities. If none exist, state that no suitable opportunity was found."""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def model_suffix_from_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a lightweight LLM for Extract Method recommendation."
    )
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--model-name", default="unsloth/Llama-3.2-3B-Instruct-bnb-4bit")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load, validate, and preview samples without loading a model.",
    )
    parser.add_argument("--preview-samples", type=int, default=2)
    args = parser.parse_args()

    suffix = model_suffix_from_name(args.model_name)
    args.output_dir = args.output_dir or str(ROOT_DIR / "outputs" / f"em_{suffix}")
    args.save_dir = args.save_dir or str(ROOT_DIR / f"lora_model_em_{suffix}")
    return args


def validate_opportunity(opportunity: Any, sample_id: str) -> None:
    if not isinstance(opportunity, dict):
        raise ValueError(f"Sample {sample_id}: each opportunity must be an object.")
    required = {"callee", "merged_method_lines", "explanation"}
    missing = required - opportunity.keys()
    if missing:
        raise ValueError(f"Sample {sample_id}: opportunity missing fields {sorted(missing)}.")
    lines = opportunity["merged_method_lines"]
    if (
        not isinstance(lines, list)
        or len(lines) != 2
        or not all(isinstance(line, int) for line in lines)
    ):
        raise ValueError(
            f"Sample {sample_id}: merged_method_lines must be [start_line, end_line]."
        )


def format_opportunities(opportunities: list[dict[str, Any]]) -> str:
    if not opportunities:
        return "No suitable Extract Method opportunity was found."

    formatted = []
    for index, opportunity in enumerate(opportunities, start=1):
        start_line, end_line = opportunity["merged_method_lines"]
        formatted.append(
            "\n".join(
                [
                    f"Opportunity {index}",
                    f"Extracted method: {opportunity['callee']}",
                    f"Lines: {start_line}-{end_line}",
                    f"Reason: {opportunity['explanation']}",
                ]
            )
        )
    return "\n\n".join(formatted)


def load_em_examples(json_file: str, max_samples: int | None = None) -> list[dict[str, str]]:
    path = Path(json_file)
    with path.open(encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array.")

    examples: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object.")
        sample_id = str(record.get("id", index))
        merged_method = record.get("merged_method")
        opportunities = record.get("opportunities")
        if not isinstance(merged_method, str) or not merged_method.strip():
            raise ValueError(f"Sample {sample_id}: merged_method must be a non-empty string.")
        if not isinstance(opportunities, list):
            raise ValueError(f"Sample {sample_id}: opportunities must be an array.")
        for opportunity in opportunities:
            validate_opportunity(opportunity, sample_id)

        examples.append(
            {
                "instruction": EM_INSTRUCTION,
                "input": merged_method.strip(),
                "output": format_opportunities(opportunities),
            }
        )
        if max_samples is not None and len(examples) >= max_samples:
            break

    if not examples:
        raise ValueError(f"No training examples were loaded from {path}.")
    return examples


def format_training_text(example: dict[str, str], eos_token: str = "") -> str:
    return ALPACA_PROMPT.format(
        example["instruction"], example["input"], example["output"]
    ) + eos_token


def build_formatter(tokenizer):
    eos_token = tokenizer.eos_token or ""

    def formatting_prompts_func(batch):
        return {
            "text": [
                ALPACA_PROMPT.format(instruction, input_code, output) + eos_token
                for instruction, input_code, output in zip(
                    batch["instruction"], batch["input"], batch["output"]
                )
            ]
        }

    return formatting_prompts_func


def preview_dataset(args: argparse.Namespace) -> None:
    train_examples = load_em_examples(args.train_file, args.max_train_samples)
    print(f"train_file: {args.train_file}")
    print(f"train_examples: {len(train_examples)}")
    print(f"model_name: {args.model_name}")
    print(f"output_dir: {args.output_dir}")
    print(f"save_dir: {args.save_dir}")

    for index, example in enumerate(train_examples[: args.preview_samples]):
        text = format_training_text(example)
        print(f"\n===== sample {index} =====")
        print(f"input_chars: {len(example['input'])}")
        print(f"output_chars: {len(example['output'])}")
        print(text[:3000])


def main() -> None:
    args = parse_args()
    if args.dry_run:
        preview_dataset(args)
        return

    import torch
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    try:
        from trl import SFTConfig
    except ImportError:
        SFTConfig = None

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    train_dataset = Dataset.from_list(
        load_em_examples(args.train_file, args.max_train_samples)
    ).map(build_formatter(tokenizer), batched=True)

    training_kwargs = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "learning_rate": args.learning_rate,
        "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(),
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "output_dir": args.output_dir,
        "seed": args.seed,
        "report_to": "none",
    }

    if SFTConfig is None:
        trainer_args = TrainingArguments(**training_kwargs)
    else:
        params = inspect.signature(SFTConfig.__init__).parameters
        compatible_kwargs = {
            key: value for key, value in training_kwargs.items() if key in params
        }
        if "dataset_text_field" in params:
            compatible_kwargs["dataset_text_field"] = "text"
        if "max_length" in params:
            compatible_kwargs["max_length"] = args.max_seq_length
        elif "max_seq_length" in params:
            compatible_kwargs["max_seq_length"] = args.max_seq_length
        if "packing" in params:
            compatible_kwargs["packing"] = args.packing
        trainer_args = SFTConfig(**compatible_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_dataset,
        "args": trainer_args,
    }
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_kwargs["max_seq_length"] = args.max_seq_length
    if "packing" in trainer_params:
        trainer_kwargs["packing"] = args.packing

    trainer = SFTTrainer(**trainer_kwargs)
    trainer.train()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
