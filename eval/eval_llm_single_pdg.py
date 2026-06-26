import argparse
import csv
import re
import sys
import time
from pathlib import Path

from unsloth import FastLanguageModel


csv.field_size_limit(200_000_000)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


DEFAULT_TEST_FILE = ROOT_DIR / "dataset" / "codesearchnet_filtered_test.csv"
DEFAULT_MODEL_DIR = ROOT_DIR / "lora_model_single_pdg_unsloth_codellama_7b_bnb_4bit"
PDG_NODE_SIM_THRESHOLD = 0.90

PDG_INSTRUCTION = """You are a program dependence graph generator.

Task:
Generate the Program Dependence Graph (PDG) for the given code.
Construct PDG in following format:
digraph PDG_<MethodName> {
    <NodeID> [type="<NodeType>", offset="lines:<StartLine>-<EndLine>"];
    <SourceNodeID> -> <TargetNodeID> [type="<EdgeType>"];
}
NodeType=[process_statement, conditional_statement, loop_statement, return_statement]
EdgeType=[control_dependency, data_dependency]
Output Requirements:
1. Output ONLY the PDG graph.
2. Do NOT output explanations.
3. Do NOT output markdown.
4. Follow the exact DOT digraph format below.
"""

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
        description="Evaluate a single-PDG fine-tuned model."
    )
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


def clean_prediction(text: str) -> str:
    text = str(text).strip()
    text = text.replace("Ċ", "\n")
    text = text.replace("Ġ", " ")
    return text


def restore_graph_newlines(text: str) -> str:
    return str(text).replace("\\n", "\n").strip()


def extract_pdg(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = restore_graph_newlines(text)
    pattern = re.compile(
        r"digraph\s+PDG_[A-Za-z0-9_]*\s*\{.*?\n\s*\}",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(0).strip() if match else ""


def normalize_dot_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def parse_offset(offset: str) -> tuple[int, int] | None:
    line_match = re.search(r"lines:(\d+)-(\d+)", offset or "")
    return tuple(map(int, line_match.groups())) if line_match else None


def range_iou(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float:
    if left is None or right is None:
        return 0.0

    left_start, left_end = left
    right_start, right_end = right
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    union = max(left_end, right_end) - min(left_start, right_start) + 1
    return intersection / union if union > 0 else 0.0


def parse_nodes(graph_text: str) -> dict[str, dict[str, str]]:
    if not isinstance(graph_text, str):
        return {}
    graph_text = restore_graph_newlines(graph_text)
    pattern = re.compile(
        r'^\s*(\d+)\s+\[type="([^"]+)",\s*offset="([^"]+)"\];',
        re.MULTILINE,
    )
    return {
        node_id: {
            "type": normalize_dot_value(node_type),
            "offset": normalize_dot_value(offset),
        }
        for node_id, node_type, offset in pattern.findall(graph_text)
    }


def parse_edges(graph_text: str) -> set[tuple[str, str, str]]:
    if not isinstance(graph_text, str):
        return set()
    graph_text = restore_graph_newlines(graph_text)

    typed_pattern = re.compile(
        r'^\s*(\d+)\s*->\s*(\d+)\s+\[type="([^"]+)"\];',
        re.MULTILINE,
    )
    plain_pattern = re.compile(
        r"^\s*(\d+)\s*->\s*(\d+);",
        re.MULTILINE,
    )

    edges = {
        (source, target, normalize_dot_value(edge_type))
        for source, target, edge_type in typed_pattern.findall(graph_text)
    }
    typed_spans = [match.span() for match in typed_pattern.finditer(graph_text)]

    for match in plain_pattern.finditer(graph_text):
        if any(start <= match.start() < end for start, end in typed_spans):
            continue
        source, target = match.groups()
        edges.add((source, target, ""))
    return edges


def prf(tp: int, pred_total: int, gt_total: int) -> tuple[float, float, float]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def node_match_score(pred_node: dict[str, str], gt_node: dict[str, str]) -> float:
    if pred_node["type"] != gt_node["type"]:
        return 0.0
    return range_iou(parse_offset(pred_node["offset"]), parse_offset(gt_node["offset"]))


def match_nodes(
    pred_nodes: dict[str, dict[str, str]],
    gt_nodes: dict[str, dict[str, str]],
) -> list[tuple[str, str, float]]:
    matched_pairs = []
    used_gt = set()

    for pred_id, pred_node in pred_nodes.items():
        best_gt_id = None
        best_score = 0.0

        for gt_id, gt_node in gt_nodes.items():
            if gt_id in used_gt:
                continue

            score = node_match_score(pred_node, gt_node)
            if score > best_score:
                best_score = score
                best_gt_id = gt_id

        if best_gt_id is not None and best_score >= PDG_NODE_SIM_THRESHOLD:
            matched_pairs.append((pred_id, best_gt_id, best_score))
            used_gt.add(best_gt_id)

    return matched_pairs


def mapped_pred_edges(
    pred_edges: set[tuple[str, str, str]],
    matched_pairs: list[tuple[str, str, float]],
) -> set[tuple[str, str, str]]:
    pred_to_gt = {pred_id: gt_id for pred_id, gt_id, _ in matched_pairs}
    result = set()
    for source, target, edge_type in pred_edges:
        if source in pred_to_gt and target in pred_to_gt:
            result.add((pred_to_gt[source], pred_to_gt[target], edge_type))
    return result


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def graph_metrics(pred_pdg: str, gt_pdg: str) -> dict[str, float | int]:
    pred_nodes = parse_nodes(pred_pdg)
    gt_nodes = parse_nodes(gt_pdg)
    pred_edges = parse_edges(pred_pdg)
    gt_edges = parse_edges(gt_pdg)

    matched_pairs = match_nodes(pred_nodes, gt_nodes)
    node_tp = len(matched_pairs)
    pred_node_total = len(pred_nodes)
    gt_node_total = len(gt_nodes)
    node_p, node_r, node_f1 = prf(node_tp, pred_node_total, gt_node_total)

    pred_edges_mapped = mapped_pred_edges(pred_edges, matched_pairs)
    edge_tp = len(pred_edges_mapped & gt_edges)
    pred_edge_total = len(pred_edges_mapped)
    gt_edge_total = len(gt_edges)
    edge_p, edge_r, edge_f1 = prf(edge_tp, pred_edge_total, gt_edge_total)

    return {
        "node_precision": node_p,
        "node_recall": node_r,
        "node_f1": node_f1,
        "edge_precision": edge_p,
        "edge_recall": edge_r,
        "edge_f1": edge_f1,
        "node_exact": int(node_f1 == 1.0 and pred_node_total == gt_node_total),
        "edge_exact": int(edge_f1 == 1.0 and pred_edge_total == gt_edge_total),
        "graph_exact": int(
            node_f1 == 1.0
            and edge_f1 == 1.0
            and pred_node_total == gt_node_total
            and pred_edge_total == gt_edge_total
        ),
        "node_tp": node_tp,
        "pred_node_total": pred_node_total,
        "gt_node_total": gt_node_total,
        "edge_tp": edge_tp,
        "pred_edge_total": len(pred_edges),
        "mapped_pred_edge_total": pred_edge_total,
        "gt_edge_total": gt_edge_total,
        "avg_node_match_score": mean([score for _, _, score in matched_pairs]),
    }


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def new_summary() -> dict:
    return {
        "count": 0,
        "node_precision": [],
        "node_recall": [],
        "node_f1": [],
        "edge_precision": [],
        "edge_recall": [],
        "edge_f1": [],
        "node_exact": 0,
        "edge_exact": 0,
        "graph_exact": 0,
    }


def update_summary(summary: dict, metrics: dict[str, float | int]) -> None:
    summary["count"] += 1
    for key in [
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_precision",
        "edge_recall",
        "edge_f1",
    ]:
        summary[key].append(float(metrics[key]))
    for key in ["node_exact", "edge_exact", "graph_exact"]:
        summary[key] += int(metrics[key])


def print_summary(title: str, summary: dict) -> None:
    count = summary["count"]
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"samples={count}")
    print(f"node_precision={mean(summary['node_precision']):.4f}")
    print(f"node_recall={mean(summary['node_recall']):.4f}")
    print(f"node_f1={mean(summary['node_f1']):.4f}")
    print(f"edge_precision={mean(summary['edge_precision']):.4f}")
    print(f"edge_recall={mean(summary['edge_recall']):.4f}")
    print(f"edge_f1={mean(summary['edge_f1']):.4f}")
    print(f"node_exact={summary['node_exact'] / count if count else 0:.4f}")
    print(f"edge_exact={summary['edge_exact'] / count if count else 0:.4f}")
    print(f"graph_exact={summary['graph_exact'] / count if count else 0:.4f}")
    print()


def summary_row(split_name: str, summary: dict) -> dict[str, float | int | str]:
    count = summary["count"]
    return {
        "split": split_name,
        "graph_type": "PDG",
        "samples": count,
        "node_precision": mean(summary["node_precision"]),
        "node_recall": mean(summary["node_recall"]),
        "node_f1": mean(summary["node_f1"]),
        "edge_precision": mean(summary["edge_precision"]),
        "edge_recall": mean(summary["edge_recall"]),
        "edge_f1": mean(summary["edge_f1"]),
        "node_exact": summary["node_exact"] / count if count else 0.0,
        "edge_exact": summary["edge_exact"] / count if count else 0.0,
        "graph_exact": summary["graph_exact"] / count if count else 0.0,
    }


def save_metrics_summary(metrics_file: str, summaries: list[tuple[str, dict]]) -> None:
    rows = [summary_row(split_name, summary) for split_name, summary in summaries]
    fieldnames = [
        "split",
        "graph_type",
        "samples",
        "node_precision",
        "node_recall",
        "node_f1",
        "edge_precision",
        "edge_recall",
        "edge_f1",
        "node_exact",
        "edge_exact",
        "graph_exact",
    ]
    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        pred_pdg = extract_pdg(prediction)
        gt_pdg = restore_graph_newlines(example["PDG"])
        metrics = graph_metrics(pred_pdg, gt_pdg)

        update_summary(overall_summary, metrics)
        target_summary = error_summary if to_bool(example["is_error"]) else clean_summary
        update_summary(target_summary, metrics)

        row = {
            "code": example["code"],
            "language": example["language"],
            "is_error": example["is_error"],
            "PDG": gt_pdg,
            "predict": prediction,
            "predict_PDG": pred_pdg,
        }
        for metric_name, metric_value in metrics.items():
            row[f"PDG_{metric_name}"] = metric_value
        rows.append(row)

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
    )
    print(f"Saved metric summary to {metrics_file}")
    print_summary("Overall PDG", overall_summary)
    print_summary("is_error=False PDG", clean_summary)
    print_summary("is_error=True PDG", error_summary)


if __name__ == "__main__":
    evaluate()
