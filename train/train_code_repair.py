#!/usr/bin/env python3
"""QLoRA fine-tuning for the CodeSearchNet synthetic code-repair dataset.

The overall training structure follows ``train_hierarchical.py`` while using
Qwen's native chat template and masking prompt tokens from the training loss.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILES = (
    ROOT_DIR / "dataset" / "code_repair" / "java" / "train.jsonl",
    ROOT_DIR / "dataset" / "code_repair" / "python" / "train.jsonl",
)
DEFAULT_MODEL = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
EXPECTED_ROLES = ("system", "user", "assistant")

SYSTEM_PROMPT = """You only repair {language} syntax and incomplete control structures. Unresolved symbols are allowed. Never invent methods, classes, imports, or business logic. Return only the repaired source."""

USER_PROMPT = """Repair the incomplete or invalid {language} source below so it can be parsed.

Make the smallest possible changes. Preserve existing lines, indentation, names, conditions, and statements whenever possible. Add only what is needed to complete the syntax and control structures.

Unresolved method calls, variables, and types are valid for this task. Never add definitions for referenced symbols. Do not add imports, comments, helper methods, classes, or new business statements.

Prefer appending missing braces or tokens without moving existing lines. Do not redesign or explain the code.

Parser error: {static_error}
File: {file_name}

Return the entire repaired file inside one `{language}` code block and nothing else.
```{language}
{source_code}
```"""


def model_suffix_from_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_name).strip("_").lower()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-4B for Java/Python code repair."
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        default=[str(path) for path in DEFAULT_TRAIN_FILES],
        help="One or more code-repair JSONL files.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument(
        "--load-in-4bit", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument(
        "--packing", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask system/user tokens and train only on the repaired function.",
    )
    parser.add_argument(
        "--drop-overlength",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop samples whose complete prompt and answer exceed max sequence length.",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview records without loading a model.",
    )
    parser.add_argument("--preview-samples", type=int, default=2)
    args = parser.parse_args(argv)

    suffix = model_suffix_from_name(args.model_name)
    args.output_dir = args.output_dir or str(
        ROOT_DIR / "outputs" / f"code_repair_{suffix}"
    )
    args.save_dir = args.save_dir or str(
        ROOT_DIR / f"lora_model_code_repair_{suffix}"
    )

    if args.max_seq_length < 256:
        parser.error("--max-seq-length must be at least 256")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if not 0 <= args.eval_ratio < 1:
        parser.error("--eval-ratio must be in [0, 1)")
    if args.per_device_train_batch_size < 1:
        parser.error("--per-device-train-batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    return args


def validate_message(message: Any, expected_role: str, sample_id: str) -> dict[str, str]:
    if not isinstance(message, dict):
        raise ValueError(f"Sample {sample_id}: every message must be an object.")
    role = message.get("role")
    content = message.get("content")
    if role != expected_role:
        raise ValueError(
            f"Sample {sample_id}: expected role {expected_role!r}, got {role!r}."
        )
    if not isinstance(content, str) or not content.strip():
        raise ValueError(
            f"Sample {sample_id}: {expected_role} content must be non-empty."
        )
    return {"role": role, "content": content}


def validate_record(record: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object.")
    sample_id = str(record.get("id", f"{path.name}:{line_number}"))
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != len(EXPECTED_ROLES):
        raise ValueError(
            f"Sample {sample_id}: messages must contain system, user, and assistant."
        )
    validated_messages = [
        validate_message(message, role, sample_id)
        for message, role in zip(messages, EXPECTED_ROLES)
    ]

    buggy_code = record.get("buggy_code")
    fixed_code = record.get("fixed_code")
    language = record.get("language")
    if language not in {"java", "python"}:
        raise ValueError(f"Sample {sample_id}: unsupported language {language!r}.")
    if not isinstance(buggy_code, str) or not buggy_code.strip():
        raise ValueError(f"Sample {sample_id}: buggy_code must be non-empty.")
    if not isinstance(fixed_code, str) or not fixed_code.strip():
        raise ValueError(f"Sample {sample_id}: fixed_code must be non-empty.")
    if buggy_code == fixed_code:
        raise ValueError(f"Sample {sample_id}: buggy and fixed code are identical.")
    if validated_messages[-1]["content"].strip() != fixed_code.strip():
        raise ValueError(
            f"Sample {sample_id}: assistant answer does not match fixed_code."
        )

    mutation = record.get("mutation")
    if not isinstance(mutation, dict):
        raise ValueError(f"Sample {sample_id}: mutation must be an object.")
    failure_class = mutation.get("failure_class")
    if not isinstance(failure_class, str) or not failure_class:
        raise ValueError(f"Sample {sample_id}: mutation.failure_class is required.")

    source = record.get("source")
    if source is None:
        source = {}
    if not isinstance(source, dict):
        raise ValueError(f"Sample {sample_id}: source must be an object.")

    file_name = infer_file_name(source, language, sample_id)
    static_error = build_static_error(record, mutation)
    training_messages = build_training_messages(
        language=language,
        static_error=static_error,
        file_name=file_name,
        buggy_code=buggy_code,
        fixed_code=fixed_code,
    )

    return {
        "id": sample_id,
        "language": language,
        "messages": training_messages,
        "mutation_type": str(mutation.get("mutation_type", "")),
        "failure_class": failure_class,
        "file_name": file_name,
        "static_error": static_error,
    }


def infer_file_name(source: dict[str, Any], language: str, sample_id: str) -> str:
    source_path = source.get("path")
    if isinstance(source_path, str) and source_path.strip():
        return Path(source_path).name

    function_name = source.get("func_name")
    extension = ".java" if language == "java" else ".py"
    if isinstance(function_name, str) and function_name.strip():
        clean_name = function_name.strip()
        if language == "java" and "." in clean_name:
            clean_name = clean_name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
        else:
            clean_name = clean_name.rsplit(".", 1)[-1]
        clean_name = re.sub(r"[^A-Za-z0-9_$-]+", "_", clean_name).strip("_")
        if clean_name:
            return clean_name + extension
    return sample_id + extension


def build_static_error(record: dict[str, Any], mutation: dict[str, Any]) -> str:
    supplied_error = record.get("static_error")
    if isinstance(supplied_error, str) and supplied_error.strip():
        return supplied_error.strip()
    line = mutation.get("line")
    column = mutation.get("column")
    if isinstance(line, int) and isinstance(column, int):
        return f"Tree-sitter parse error near line {line}, column {column}."
    return "Tree-sitter reported an invalid or incomplete syntax structure."


def build_training_messages(
    *,
    language: str,
    static_error: str,
    file_name: str,
    buggy_code: str,
    fixed_code: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT.format(language=language),
        },
        {
            "role": "user",
            "content": USER_PROMPT.format(
                language=language,
                static_error=static_error,
                file_name=file_name,
                source_code=buggy_code,
            ),
        },
        {
            "role": "assistant",
            "content": f"```{language}\n{fixed_code}\n```",
        },
    ]


def iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


def load_repair_examples(
    train_files: list[str],
    *,
    max_samples: int | None = None,
    seed: int = 3407,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for filename in train_files:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Training file does not exist: {path}")
        for line_number, record in iter_jsonl(path):
            example = validate_record(record, path, line_number)
            # The task prompt explicitly permits unresolved symbols. Training on
            # name/type resolution mutations would contradict that instruction.
            if example["failure_class"] != "syntax_error":
                continue
            if example["id"] in seen_ids:
                raise ValueError(f"Duplicate sample id: {example['id']}")
            seen_ids.add(example["id"])
            examples.append(example)

    if not examples:
        raise ValueError("No valid code-repair examples were loaded.")
    random.Random(seed).shuffle(examples)
    if max_samples is not None:
        examples = examples[:max_samples]
    return examples


def format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def build_formatter(tokenizer: Any):
    def formatting_prompts_func(batch: dict[str, list[Any]]) -> dict[str, list[str]]:
        return {
            "text": [format_messages(tokenizer, messages) for messages in batch["messages"]]
        }

    return formatting_prompts_func


def token_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def preview_dataset(args: argparse.Namespace) -> None:
    examples = load_repair_examples(
        args.train_files, max_samples=args.max_samples, seed=args.seed
    )
    language_counts: dict[str, int] = {}
    mutation_counts: dict[str, int] = {}
    for example in examples:
        language_counts[example["language"]] = language_counts.get(example["language"], 0) + 1
        mutation = example["mutation_type"]
        mutation_counts[mutation] = mutation_counts.get(mutation, 0) + 1

    print(f"train_files: {args.train_files}")
    print(f"model_name: {args.model_name}")
    print(f"output_dir: {args.output_dir}")
    print(f"save_dir: {args.save_dir}")
    print(f"num_examples: {len(examples)}")
    print(f"language_counts: {dict(sorted(language_counts.items()))}")
    print(f"mutation_counts: {dict(sorted(mutation_counts.items()))}")
    for index, example in enumerate(examples[: args.preview_samples]):
        print(f"\n===== sample {index} ({example['id']}) =====")
        print(json.dumps(example["messages"], ensure_ascii=False, indent=2)[:5000])


def compatible_config(config_class: Any, values: dict[str, Any]) -> Any:
    parameters = inspect.signature(config_class.__init__).parameters
    compatible = {key: value for key, value in values.items() if key in parameters}
    return config_class(**compatible)


def main() -> None:
    args = parse_args()
    if args.dry_run:
        preview_dataset(args)
        return

    import torch
    from datasets import Dataset
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer

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
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    examples = load_repair_examples(
        args.train_files, max_samples=args.max_samples, seed=args.seed
    )
    formatted_examples = []
    for example in examples:
        formatted_example = dict(example)
        formatted_example["text"] = format_messages(tokenizer, example["messages"])
        formatted_example["token_length"] = token_length(
            tokenizer, formatted_example["text"]
        )
        formatted_examples.append(formatted_example)

    original_count = len(formatted_examples)
    if args.drop_overlength:
        formatted_examples = [
            example
            for example in formatted_examples
            if example["token_length"] <= args.max_seq_length
        ]
    dropped_count = original_count - len(formatted_examples)
    if not formatted_examples:
        raise ValueError("No training samples remain after length filtering.")

    dataset = Dataset.from_list(formatted_examples)

    eval_dataset = None
    if args.eval_ratio > 0 and len(dataset) > 1:
        split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset

    print(f"loaded_examples: {original_count}")
    print(f"overlength_examples_dropped: {dropped_count}")
    print(f"train_examples: {len(train_dataset)}")
    print(f"eval_examples: {len(eval_dataset) if eval_dataset is not None else 0}")

    training_values = {
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
        "save_total_limit": args.save_total_limit,
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "output_dir": args.output_dir,
        "seed": args.seed,
        "report_to": "none",
        "eval_strategy": "steps" if eval_dataset is not None else "no",
        "evaluation_strategy": "steps" if eval_dataset is not None else "no",
        "eval_steps": args.save_steps,
    }

    if SFTConfig is None:
        trainer_args = compatible_config(TrainingArguments, training_values)
    else:
        parameters = inspect.signature(SFTConfig.__init__).parameters
        sft_values = dict(training_values)
        sft_values["dataset_text_field"] = "text"
        if "max_length" in parameters:
            sft_values["max_length"] = args.max_seq_length
        else:
            sft_values["max_seq_length"] = args.max_seq_length
        sft_values["packing"] = args.packing
        trainer_args = compatible_config(SFTConfig, sft_values)

    trainer_values: dict[str, Any] = {
        "model": model,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "args": trainer_args,
    }
    trainer_parameters = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in trainer_parameters:
        trainer_values["processing_class"] = tokenizer
    elif "tokenizer" in trainer_parameters:
        trainer_values["tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_parameters:
        trainer_values["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_parameters:
        trainer_values["max_seq_length"] = args.max_seq_length
    if "packing" in trainer_parameters:
        trainer_values["packing"] = args.packing

    trainer = SFTTrainer(**trainer_values)
    if args.assistant_only_loss:
        try:
            from unsloth.chat_templates import train_on_responses_only
        except ImportError as error:
            raise RuntimeError(
                "This Unsloth version does not provide train_on_responses_only; "
                "upgrade Unsloth or run with --no-assistant-only-loss."
            ) from error
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
