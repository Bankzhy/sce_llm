#!/usr/bin/env python3
"""Evaluate a base Extract Method model through a local Ollama server."""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_VALID_FILE = ROOT_DIR / "dataset" / "em_valid.json"
DEFAULT_TEST_FILE = ROOT_DIR / "dataset" / "em_test.json"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "codellama:7b"

SYSTEM_PROMPT = "You are a code refactoring expert."

EM_PROMPT = """Analyze the following method and identify all code regions suitable for Extract Method refactoring.

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

Repeat the same four fields for every additional opportunity.
Do not omit the Lines or Reason field.

Input method:
{}"""


def safe_name(value: str, fallback: str = "sample") -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return name or fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a local Ollama base model on em_valid.json and "
            "em_test.json, saving one prediction per text file."
        )
    )
    parser.add_argument("--valid-file", default=str(DEFAULT_VALID_FILE))
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-valid-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing prediction files instead of skipping them.",
    )
    return parser.parse_args()


def default_output_dir(model: str) -> Path:
    model_name = safe_name(model)
    return Path(__file__).resolve().parent / f"em_{model_name}_base_results"


def load_examples(json_file: str, max_samples: int | None) -> list[dict[str, str]]:
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
        code = record.get("merged_method")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(
                f"{path}: sample {sample_id} has no non-empty merged_method."
            )

        examples.append({"id": sample_id, "code": code.strip()})
        if max_samples is not None and len(examples) >= max_samples:
            break

    if not examples:
        raise ValueError(f"No evaluation examples were loaded from {path}.")
    return examples


def build_user_prompt(code: str) -> str:
    return EM_PROMPT.format(code)


def predict_with_ollama(
    *,
    url: str,
    model: str,
    code: str,
    max_new_tokens: int,
    temperature: float,
    timeout: float,
) -> str:
    response = requests.post(
        url,
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(code)},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_new_tokens,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()

    try:
        prediction = payload["message"]["content"].strip()
    except (KeyError, TypeError, AttributeError) as exc:
        raise RuntimeError(f"Unexpected Ollama response: {payload}") from exc

    if not prediction:
        raise RuntimeError(
            f"Ollama returned an empty prediction for model {model}. "
            f"Response metadata: {payload}"
        )
    return prediction


def result_path(output_dir: Path, split: str, index: int, sample_id: str) -> Path:
    return output_dir / split / f"{index:06d}_{safe_name(sample_id)}.txt"


def evaluate_split(
    *,
    split: str,
    json_file: str,
    max_samples: int | None,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[int, int]:
    examples = load_examples(json_file, max_samples)
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    skipped = 0
    start_time = time.time()

    print(f"\n{split}: {json_file} ({len(examples)} samples)", flush=True)
    for index, example in enumerate(examples, start=1):
        output_file = result_path(output_dir, split, index, example["id"])
        if output_file.exists() and not args.overwrite:
            skipped += 1
            print(
                f"[{split} {index}/{len(examples)}] skip existing: "
                f"{output_file.name}",
                flush=True,
            )
            continue

        prediction = predict_with_ollama(
            url=args.url,
            model=args.model,
            code=example["code"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )
        output_file.write_text(prediction + "\n", encoding="utf-8")
        generated += 1

        elapsed = time.time() - start_time
        average = elapsed / max(generated, 1)
        remaining = average * (len(examples) - index)
        print(
            f"[{split} {index}/{len(examples)}] saved {output_file.name} "
            f"(elapsed={elapsed / 60:.1f}m, eta={remaining / 60:.1f}m)",
            flush=True,
        )

    return generated, skipped


def main() -> int:
    args = parse_args()
    output_dir = (
        Path(args.output_dir) if args.output_dir else default_output_dir(args.model)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"ollama_url: {args.url}")
    print(f"model: {args.model}")
    print(f"output_dir: {output_dir}")

    try:
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
                output_dir=output_dir,
                args=args,
            )
            total_generated += generated
            total_skipped += skipped
    except requests.RequestException as exc:
        raise SystemExit(
            f"Ollama request failed: {exc}. Ensure Ollama is running and "
            f"model '{args.model}' is installed."
        ) from exc

    print(
        f"\nDone. generated={total_generated}, skipped={total_skipped}, "
        f"output_dir={output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
