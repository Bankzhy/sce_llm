#!/usr/bin/env python3
"""Generate Extract Method predictions for the validation and test datasets."""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VALID_FILE = ROOT_DIR / "dataset" / "em_valid.json"
DEFAULT_TEST_FILE = ROOT_DIR / "dataset" / "em_test.json"
DEFAULT_MODEL_DIR = ROOT_DIR / "lora_model_em_unsloth_qwen2_5_coder_7b_instruct_bnb_4bit"

EM_INSTRUCTION = """Analyze the following method and identify all code regions suitable for Extract Method refactoring.

For every opportunity:
1. Propose a meaningful extracted method signature.
2. Give the exact inclusive line range using the input method's line numbers.
3. Explain why extracting the selected region improves the code from three aspects: method length, method complexity, and method functionality.

Count lines from 1. Annotations and the method declaration count as lines.
Return only the opportunities in exactly this format:

Opportunity 1
Extracted method: <method signature>
Lines: <start line>-<end line>
Reason: <reason for extraction>

Input method:
{}"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def safe_name(value: str, fallback: str = "sample") -> str:
    """Convert a model name or sample ID into a filesystem-safe name."""
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return name or fallback


def model_result_name(model_dir: str) -> str:
    """Return the model portion used by the default result directory."""
    if "/" in model_dir and not Path(model_dir).is_absolute():
        raw_name = model_dir
    else:
        raw_name = Path(model_dir.rstrip("/")).name or model_dir
    for prefix in ("lora_model_em_", "em_"):
        if raw_name.startswith(prefix):
            raw_name = raw_name[len(prefix) :]
            break
    return safe_name(raw_name, "model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a fine-tuned Extract Method model on em_valid.json and "
            "em_test.json, saving one prediction per text file."
        )
    )
    parser.add_argument("--valid-file", default=str(DEFAULT_VALID_FILE))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_MODEL_DIR),
        help="Fine-tuned LoRA model directory produced by train/train_em.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: eval/em_<model-name>_results",
    )
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist. By default they are skipped.",
    )
    return parser.parse_args()


def load_examples(json_file: str, max_samples: int | None = None) -> list[dict[str, str]]:
    path = Path(json_file)
    with path.open(encoding="utf-8") as file:
        records: Any = json.load(file)

    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array.")

    examples: list[dict[str, str]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} must be an object.")

        sample_id = str(record.get("id", index))
        merged_method = record.get("merged_method")
        if not isinstance(merged_method, str) or not merged_method.strip():
            raise ValueError(
                f"{path}: sample {sample_id} has no non-empty merged_method."
            )

        examples.append({"id": sample_id, "input": merged_method.strip()})
        if max_samples is not None and len(examples) >= max_samples:
            break

    if not examples:
        raise ValueError(f"No evaluation examples were loaded from {path}.")
    return examples


def build_prompt(code: str) -> str:
    return ALPACA_PROMPT.format(EM_INSTRUCTION, code, "")


def generate_prediction(model, tokenizer, code: str, max_new_tokens: int) -> str:
    inputs = tokenizer(build_prompt(code), return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=(
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        ),
    )
    generated_tokens = outputs[0, inputs.input_ids.shape[1] :]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def prediction_path(
    output_dir: Path,
    split: str,
    index: int,
    sample_id: str,
) -> Path:
    safe_id = safe_name(sample_id)
    return output_dir / split / f"{index:06d}_{safe_id}.txt"


def evaluate_split(
    *,
    split: str,
    json_file: str,
    max_samples: int | None,
    model,
    tokenizer,
    output_dir: Path,
    max_new_tokens: int,
    overwrite: bool,
) -> tuple[int, int]:
    examples = load_examples(json_file, max_samples)
    (output_dir / split).mkdir(parents=True, exist_ok=True)
    generated = 0
    skipped = 0
    start_time = time.time()

    print(f"\n{split}: {json_file} ({len(examples)} samples)", flush=True)
    for index, example in enumerate(examples, start=1):
        result_file = prediction_path(output_dir, split, index, example["id"])
        if result_file.exists() and not overwrite:
            skipped += 1
            print(
                f"[{split} {index}/{len(examples)}] skip existing: {result_file.name}",
                flush=True,
            )
            continue

        prediction = generate_prediction(
            model,
            tokenizer,
            example["input"],
            max_new_tokens,
        )
        result_file.write_text(prediction + "\n", encoding="utf-8")
        generated += 1

        elapsed = time.time() - start_time
        average = elapsed / max(generated, 1)
        remaining = average * (len(examples) - index)
        print(
            f"[{split} {index}/{len(examples)}] saved {result_file.name} "
            f"(elapsed={elapsed / 60:.1f}m, eta={remaining / 60:.1f}m)",
            flush=True,
        )

    return generated, skipped


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).resolve().parent
        / f"em_{model_result_name(args.model_dir)}_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Import Unsloth before Transformers/TRL so its runtime patches are applied.
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    print(f"model_dir: {args.model_dir}")
    print(f"output_dir: {output_dir}")

    total_generated = 0
    total_skipped = 0
    for split, json_file, max_samples in (
        ("valid", args.valid_file, args.max_valid_samples),
        ("test", args.test_file, args.max_test_samples),
    ):
        generated, skipped = evaluate_split(
            split=split,
            json_file=json_file,
            max_samples=max_samples,
            model=model,
            tokenizer=tokenizer,
            output_dir=output_dir,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
        total_generated += generated
        total_skipped += skipped

    print(
        f"\nDone. generated={total_generated}, skipped={total_skipped}, "
        f"output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
