import argparse
import csv
import re
import sys
import time
from pathlib import Path

from unsloth import FastLanguageModel


csv.field_size_limit(200_000_000)

ROOT_DIR = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from graph_eval_common import (
    clean_prediction,
    extract_graph,
    graph_metrics,
    new_summary,
    normalize_graph_for_metrics,
    print_summary,
    save_metrics_summary,
    to_bool,
    update_summary,
)
from train.train_single_pdg import ALPACA_PROMPT, PDG_INSTRUCTION, model_suffix_from_name


DEFAULT_TEST_FILE = ROOT_DIR / "dataset" / "codesearchnet_filtered_test.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "lora_model_single_pdg_unsloth_codellama_7b_bnb_4bit"
GRAPH_TYPES = ["PDG"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a single-PDG fine-tuned model.")
    parser.add_argument("--test-file", default=str(DEFAULT_TEST_FILE))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--metrics-file", default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--small-test",
        action="store_true",
        help="Quick smoke test: evaluate only the first 10 samples after language filtering.",
    )
    parser.add_argument("--preview-samples", type=int, default=3)
    parser.add_argument(
        "--language-group",
        choices=["all", "java_python", "javascript"],
        default="all",
        help="Preset language split for experiments.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Explicit language filter, e.g. --languages java python. Overrides --language-group.",
    )
    return parser.parse_args()


def language_filter_from_args(args: argparse.Namespace) -> set[str] | None:
    if args.languages:
        return {language.lower() for language in args.languages}
    if args.language_group == "java_python":
        return {"java", "python"}
    if args.language_group == "javascript":
        return {"javascript"}
    return None


def language_suffix(languages: set[str] | None) -> str:
    if languages is None:
        return "all"
    return "_".join(sorted(languages))


def default_output_file(model_dir: str, languages: set[str] | None) -> str:
    suffix = model_suffix_from_name(Path(model_dir).name)
    return str(ROOT_DIR / "eval" / f"predict_single_pdg_{suffix}_{language_suffix(languages)}.csv")


def default_metrics_file(output_file: str) -> str:
    path = Path(output_file)
    return str(path.with_name(f"{path.stem}_metrics.csv"))


def build_prompt(code: str) -> str:
    return ALPACA_PROMPT.format(PDG_INSTRUCTION, code, "")


def load_examples(
    csv_file: str,
    max_samples: int | None = None,
    languages: set[str] | None = None,
) -> list[dict[str, str]]:
    examples = []
    with open(csv_file, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"code", "PDG", "language", "is_error"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing_columns)}")

        for row in reader:
            if languages is not None and row.get("language", "").lower() not in languages:
                continue
            examples.append(row)
            if max_samples is not None and len(examples) >= max_samples:
                break
    return examples


def generate_pdg(model, tokenizer, code: str, max_new_tokens: int) -> str:
    inputs = tokenizer(build_prompt(code), return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
    return clean_prediction(tokenizer.decode(generated_tokens, skip_special_tokens=True))


def evaluate() -> None:
    args = parse_args()
    if args.small_test and args.max_samples is None:
        args.max_samples = 10

    languages = language_filter_from_args(args)
    output_file = args.output_file or default_output_file(args.model_dir, languages)
    metrics_file = args.metrics_file or default_metrics_file(output_file)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_dir,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)

    examples = load_examples(args.test_file, args.max_samples, languages)
    if not examples:
        raise ValueError(
            f"No examples loaded from {args.test_file} with language filter {languages}."
        )

    print(f"test_file: {args.test_file}")
    print(f"model_dir: {args.model_dir}")
    print(f"language_filter: {sorted(languages) if languages else 'all'}")
    print(f"num_examples: {len(examples)}")
    print(f"output_file: {output_file}")
    print(f"metrics_file: {metrics_file}")

    rows = []
    overall_summary = new_summary()
    clean_summary = new_summary()
    error_summary = new_summary()
    start_time = time.time()

    for index, example in enumerate(examples, start=1):
        elapsed = time.time() - start_time
        avg_seconds = elapsed / max(index - 1, 1)
        remaining = avg_seconds * (len(examples) - index + 1)
        percent = index / len(examples) * 100
        print(
            f"[{index}/{len(examples)} | {percent:6.2f}%] "
            f"language={example['language']} is_error={example['is_error']} "
            f"elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
            flush=True,
        )

        prediction = generate_pdg(model, tokenizer, example["code"], args.max_new_tokens)
        pred_pdg = normalize_graph_for_metrics(extract_graph(prediction, "PDG"), "PDG")
        gt_pdg = normalize_graph_for_metrics(example["PDG"], "PDG")
        metrics = graph_metrics(pred_pdg, gt_pdg, "PDG")

        update_summary(overall_summary, "PDG", metrics)
        target_summary = error_summary if to_bool(example["is_error"]) else clean_summary
        update_summary(target_summary, "PDG", metrics)

        rows.append(
            {
                "code": example["code"],
                "language": example["language"],
                "is_error": example["is_error"],
                "PDG": gt_pdg,
                "predict": prediction,
                "predict_PDG": pred_pdg,
                "PDG_Node-F1": metrics["node_f1"],
                "PDG_Edge-F1": metrics["edge_f1"],
                "PDG_Node-EM": metrics["node_exact"],
                "PDG_Node-ZM": metrics["node_zero"],
            }
        )

        if index <= args.preview_samples:
            print(prediction[:1200])
            print()

    fieldnames = list(rows[0].keys()) if rows else []
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved predictions to {output_file}")
    save_metrics_summary(
        metrics_file,
        [
            ("overall", overall_summary),
            ("is_error_false", clean_summary),
            ("is_error_true", error_summary),
        ],
        GRAPH_TYPES,
    )
    print(f"Saved metric summary to {metrics_file}")
    print_summary("overall", overall_summary, GRAPH_TYPES)
    print_summary("is_error_false", clean_summary, GRAPH_TYPES)
    print_summary("is_error_true", error_summary, GRAPH_TYPES)


if __name__ == "__main__":
    evaluate()
