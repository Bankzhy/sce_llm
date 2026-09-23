#!/usr/bin/env python3
"""QLoRA fine-tuning for HCG's on-demand AST, CFG, and PDG generation.

Each CSV source row becomes three independent chat examples. The prompts are
rebuilt from the current HCG client contract, so historical prompt text is not
used for training.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import re
from collections import Counter
from pathlib import Path
from typing import Any


csv.field_size_limit(200_000_000)
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT_DIR / "dataset" / "codesearchnet_filtered_train.csv"
DEFAULT_CODE_REPAIR_MODEL = ROOT_DIR / "lora_model_code_repair_unsloth_qwen3_4b_instruct_2507_unsloth_bnb_4bit"
GRAPH_TYPES = ("AST", "CFG", "PDG")
GRAPH_INSTRUCTIONS = {
    kind: f"Generate only the {kind} from source code as one DOT digraph named {kind}_graph."
    for kind in GRAPH_TYPES
}
GRAPH_RULES = {
    "AST": """Generate a compact abstract syntax tree, not a concrete syntax tree.
Allowed node types: root, class_declaration, method_declaration, process_statement, conditional_statement, loop_statement, return_statement, type_identifier, var_identifier, method_identifier, literal_value.
Create exactly one root node spanning the analyzed source. The root has no parent. Every other node has exactly one incoming edge. The graph must be connected and acyclic, so the edge count is node count minus one.
Create structural nodes for declarations and executable statements according to lexical nesting. Do not create nodes for braces, punctuation, comments, or blank lines.
Only a process_statement may have semantic leaf children. Add at most one type_identifier, one var_identifier or method_identifier, and one literal_value child when present. Every identifier or literal node must have a short label attribute. Semantic leaf nodes must not have children.
If the node budget is insufficient, preserve root and structural nodes first, then omit optional semantic leaf nodes. AST edges have no type.""",
    "CFG": "Node types: process_statement, conditional_statement, loop_statement, return_statement. Include executable control flow only; exclude method, function, class, parameter, type, import, and comment nodes.",
    "PDG": "Node types: process_statement, conditional_statement, loop_statement, return_statement. Edge types: control_dependency, data_dependency. Include executable dependencies only; exclude method, function, class, parameter, type, import, and comment nodes.",
}
AST_LEAVES = {"type_identifier", "var_identifier", "method_identifier", "literal_value"}
ALLOWED_NODES = {
    "AST": {"root", "class_declaration", "method_declaration", "process_statement", "conditional_statement", "loop_statement", "return_statement", *AST_LEAVES},
    "CFG": {"process_statement", "conditional_statement", "loop_statement", "return_statement"},
    "PDG": {"process_statement", "conditional_statement", "loop_statement", "return_statement"},
}


def model_suffix_from_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", (Path(model_name).name or model_name)).strip("_").lower()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune separate AST, CFG, and PDG generation.")
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--model-name", default=str(DEFAULT_CODE_REPAIR_MODEL))
    parser.add_argument("--output-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int, help="Maximum source rows; each row produces three examples.")
    parser.add_argument("--include-error-samples", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--assistant-only-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--drop-overlength", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--preview-samples", type=int, default=3)
    args = parser.parse_args(argv)
    suffix = model_suffix_from_name(args.model_name)
    args.output_dir = args.output_dir or str(ROOT_DIR / "outputs" / f"graph_generation_{suffix}")
    args.save_dir = args.save_dir or str(ROOT_DIR / f"lora_model_graph_generation_{suffix}")
    if args.max_seq_length < 1024:
        parser.error("--max-seq-length must be at least 1024")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    return args


def has_lora_adapter(model: Any) -> bool:
    return bool(getattr(model, "peft_config", None))


def make_existing_adapter_trainable(model: Any) -> int:
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()
    model.train()
    for name, parameter in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            parameter.requires_grad_(True)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count == 0:
        raise RuntimeError("A LoRA adapter was detected, but it has no trainable parameters.")
    return count


def prepare_lora_model(model: Any, fast_language_model: Any, args: argparse.Namespace):
    if has_lora_adapter(model):
        count = make_existing_adapter_trainable(model)
        print(f"lora_initialization: continuing existing adapter ({count:,} trainable parameters)")
        return model
    print("lora_initialization: creating a new adapter on the base model")
    return fast_language_model.get_peft_model(
        model, r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha, lora_dropout=getattr(args, "lora_dropout", 0.0), bias="none",
        use_gradient_checkpointing=True, random_state=args.seed,
    )


def find_column(fieldnames: list[str] | None, column: str) -> str:
    matches = {name.lower(): name for name in (fieldnames or [])}
    if column.lower() not in matches:
        raise ValueError(f"The training CSV must contain a {column} column.")
    return matches[column.lower()]


def _is_meaningful_line(line: str) -> bool:
    value = line.strip()
    return bool(value) and not value.startswith(("//", "#", "/*", "*", "*/", '"""', "'''"))


def graph_budget(graph_type: str, source: str) -> tuple[int, int]:
    meaningful = sum(_is_meaningful_line(line) for line in source.split("\n"))
    if graph_type == "AST":
        nodes = min(max(1 + meaningful * 3, 6), 48)
        return nodes, min(max(nodes - 1, 3), 47)
    nodes = min(max(meaningful + 2, 4), 32)
    return nodes, min(max(nodes * 2, 4), 64)


def numbered_source(source: str) -> str:
    return "\n".join(f"{i}: {line}" for i, line in enumerate(source.split("\n"), 1))


def build_graph_user_prompt(graph_type: str, language: str, source: str) -> str:
    graph_type = graph_type.upper()
    nodes, edges = graph_budget(graph_type, source)
    line_count = len(source.split("\n"))
    return f"""{GRAPH_INSTRUCTIONS[graph_type]}

Use 1-based source lines. The source has exactly {line_count} lines. Every offset must satisfy 1 <= START <= END <= {line_count}. Ignore comments and blank lines. Declare every node as: id [type="TYPE", offset="lines:START-END"]; and use only declared node IDs in edges.

Use at most {nodes} nodes and {edges} edges. Never repeat a node or edge. Finish the graph with a closing brace before the output limit.

{GRAPH_RULES[graph_type]}

Return only the DOT digraph. Do not return explanations, markdown, JSON, thinking, or tool calls. Use only the supplied source.

Language: {language}
Source (the numeric prefixes are line numbers, not source text):
{numbered_source(source)}

Start the response exactly with:
digraph {graph_type}_graph {{"""


def _attribute(attributes: str, name: str) -> str | None:
    match = re.search(rf'\b{re.escape(name)}\s*=\s*"([^"]*)"', attributes)
    return match.group(1) if match else None


def normalize_graph(graph: str, graph_type: str, source: str) -> str:
    """Convert historical targets to the current bounded DOT contract."""
    graph_type = graph_type.upper()
    source_lines = source.split("\n")
    line_count = max(len(source_lines), 1)
    max_nodes, max_edges = graph_budget(graph_type, source)
    node_re = re.compile(r'^\s*("[^"\n]+"|[^\s\[\]-]+)\s*\[([^;\n]+)\]?\s*;?\s*$', re.MULTILINE)
    edge_re = re.compile(r'^\s*("[^"\n]+"|[^\s\[\]-]+)\s*->\s*("[^"\n]+"|[^\s\[\];]+)(?:\s*\[([^;\n]+)\]?)?\s*;?\s*$', re.MULTILINE)
    nodes: list[dict[str, Any]] = []
    for match in node_re.finditer(graph):
        node_id, attrs = match.group(1).strip('"'), match.group(2)
        node_type = _attribute(attrs, "type") or "process_statement"
        offset = re.search(r'(?:lines:)?(\d+)\s*-\s*(\d+)', attrs)
        start = min(max(int(offset.group(1)) if offset else 1, 1), line_count)
        end = min(max(int(offset.group(2)) if offset else start, start), line_count)
        if node_type not in ALLOWED_NODES[graph_type]:
            continue
        if not any(_is_meaningful_line(source_lines[i - 1]) for i in range(start, end + 1)):
            continue
        nodes.append({"id": node_id, "type": node_type, "start": start, "end": end, "label": _attribute(attrs, "label")})
    raw_edges = [
        (match.group(1).strip('"'), match.group(2).strip('"'), _attribute(match.group(3) or "", "type"))
        for match in edge_re.finditer(graph)
    ]
    if graph_type == "AST":
        by_id = {node["id"]: node for node in nodes}
        structural = [n for n in nodes if n["type"] not in AST_LEAVES and n["type"] != "root"]
        # The client contract permits semantic leaves only below process nodes,
        # with at most one type, one identifier, and one literal per process.
        leaf_slots: set[tuple[str, str]] = set()
        leaves = []
        for source_id, target_id, _ in raw_edges:
            parent, child = by_id.get(source_id), by_id.get(target_id)
            if not parent or not child or parent["type"] != "process_statement" or child["type"] not in AST_LEAVES:
                continue
            category = "identifier" if child["type"] in {"var_identifier", "method_identifier"} else child["type"]
            slot = (source_id, category)
            if slot not in leaf_slots:
                leaves.append(child)
                leaf_slots.add(slot)
        synthetic = [
            {"id": "hcg_root", "type": "root", "start": 1, "end": line_count, "label": None},
            {"id": "hcg_method", "type": "method_declaration", "start": 1, "end": line_count, "label": None},
        ]
        nodes = (synthetic + structural + leaves)[:max_nodes]
    else:
        nodes = nodes[:max_nodes]
    if not nodes:
        raise ValueError(f"No usable {graph_type} nodes remain after normalization.")
    node_ids = {node["id"] for node in nodes}
    edges: list[tuple[str, str, str | None]] = []
    node_types = {node["id"]: node["type"] for node in nodes}
    for source_id, target_id, edge_type in raw_edges:
        if source_id in node_ids and target_id in node_ids:
            if graph_type == "AST" and (
                node_types[source_id] in AST_LEAVES
                or (node_types[target_id] in AST_LEAVES and node_types[source_id] != "process_statement")
            ):
                continue
            edges.append((source_id, target_id, edge_type))
    if graph_type == "AST":
        parented: set[str] = set()
        tree: list[tuple[str, str, str | None]] = [("hcg_root", "hcg_method", None)]
        parented.add("hcg_method")
        for source_id, target_id, _ in edges:
            if target_id != "hcg_root" and target_id not in parented and source_id != target_id:
                tree.append((source_id, target_id, None))
                parented.add(target_id)
        for node in nodes[1:]:
            if node["id"] not in parented:
                # Top-level executable structures belong to the method body.
                tree.append(("hcg_method", node["id"], None))
        edges = tree[: min(max_edges, len(nodes) - 1)]
    else:
        edges = edges[:max_edges]
    output = [f"digraph {graph_type}_graph {{"]
    for node in nodes:
        attrs = [f'type="{node["type"]}"', f'offset="lines:{node["start"]}-{node["end"]}"']
        if node["type"] in AST_LEAVES and node["label"]:
            label = str(node["label"]).replace("\\", "\\\\").replace('"', '\\"')
            attrs.append(f'label="{label}"')
        output.append(f'    "{node["id"]}" [{", ".join(attrs)}];')
    for source_id, target_id, edge_type in edges:
        if graph_type == "PDG":
            kind = "data_dependency" if edge_type and "data" in edge_type else "control_dependency"
            output.append(f'    "{source_id}" -> "{target_id}" [type="{kind}"];')
        else:
            output.append(f'    "{source_id}" -> "{target_id}";')
    return "\n".join([*output, "}"])


def build_graph_messages(graph_type: str, language: str, source: str, graph: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": build_graph_user_prompt(graph_type, language, source)},
        {"role": "assistant", "content": normalize_graph(graph, graph_type, source)},
    ]


def decode_escaped_newlines(value: str) -> str:
    """Decode the literal line separators used by the historical CSV export."""
    if "\n" not in value and r"\n" in value:
        return value.replace(r"\r\n", "\n").replace(r"\n", "\n")
    return value


def load_hierarchical_examples(csv_file: str, max_samples: int | None = None, include_error_samples: bool = False) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    retained = 0
    with open(csv_file, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        code_col, language_col = find_column(fields, "code"), find_column(fields, "language")
        graph_cols = {kind: find_column(fields, kind) for kind in GRAPH_TYPES}
        for row_number, row in enumerate(reader, 1):
            if not include_error_samples and str(row.get("is_error", "")).lower() == "true":
                continue
            source = decode_escaped_newlines((row.get(code_col) or "").strip())
            language = (row.get(language_col) or "").strip().lower()
            targets = {
                kind: decode_escaped_newlines((row.get(column) or "").strip())
                for kind, column in graph_cols.items()
            }
            if not source or language not in {"java", "python", "javascript"} or not all(targets.values()):
                continue
            try:
                batch = [{
                    "id": f'{row.get("id") or row_number}:{kind.lower()}',
                    "language": language, "graph_type": kind, "source": source,
                    "messages": build_graph_messages(kind, language, source, targets[kind]),
                } for kind in GRAPH_TYPES]
            except ValueError:
                continue
            examples.extend(batch)
            retained += 1
            if max_samples is not None and retained >= max_samples:
                break
    if not examples:
        raise ValueError(f"No valid graph-generation examples were loaded from {csv_file}.")
    return examples


def format_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def compatible_config(config_class: Any, values: dict[str, Any]) -> Any:
    params = inspect.signature(config_class.__init__).parameters
    return config_class(**{key: value for key, value in values.items() if key in params})


def preview_dataset(args: argparse.Namespace) -> None:
    examples = load_hierarchical_examples(args.train_file, args.max_samples, args.include_error_samples)
    print(f"train_file: {args.train_file}")
    print(f"model_name: {args.model_name}")
    print(f"num_examples: {len(examples)}")
    print(f"graph_type_counts: {dict(Counter(x['graph_type'] for x in examples))}")
    for index, example in enumerate(examples[: args.preview_samples]):
        print(f"\n===== sample {index} ({example['id']}) =====")
        print(example["messages"][0]["content"][:4000])
        print("\n--- assistant ---")
        print(example["messages"][1]["content"][:4000])


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
        model_name=args.model_name, max_seq_length=args.max_seq_length, dtype=None, load_in_4bit=args.load_in_4bit
    )
    model = prepare_lora_model(model, FastLanguageModel, args)
    examples = load_hierarchical_examples(args.train_file, args.max_samples, args.include_error_samples)
    formatted = []
    for example in examples:
        text = format_messages(tokenizer, example["messages"])
        length = len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])
        if not args.drop_overlength or length <= args.max_seq_length:
            formatted.append({**example, "text": text, "token_length": length})
    if not formatted:
        raise ValueError("No samples remain after sequence-length filtering.")
    dataset = Dataset.from_list(formatted)
    values = {
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs, "warmup_ratio": args.warmup_ratio,
        "learning_rate": args.learning_rate, "fp16": not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_bf16_supported(), "logging_steps": args.logging_steps,
        "save_steps": args.save_steps, "save_total_limit": args.save_total_limit,
        "optim": "adamw_8bit", "weight_decay": 0.01, "lr_scheduler_type": "cosine",
        "output_dir": args.output_dir, "seed": args.seed, "report_to": "none",
    }
    if SFTConfig is None:
        trainer_args = compatible_config(TrainingArguments, values)
    else:
        config = {**values, "dataset_text_field": "text", "packing": args.packing}
        config["max_length" if "max_length" in inspect.signature(SFTConfig.__init__).parameters else "max_seq_length"] = args.max_seq_length
        trainer_args = compatible_config(SFTConfig, config)
    trainer_params = inspect.signature(SFTTrainer.__init__).parameters
    trainer_values: dict[str, Any] = {"model": model, "train_dataset": dataset, "args": trainer_args}
    trainer_values["processing_class" if "processing_class" in trainer_params else "tokenizer"] = tokenizer
    if "dataset_text_field" in trainer_params:
        trainer_values["dataset_text_field"] = "text"
    if "max_seq_length" in trainer_params:
        trainer_values["max_seq_length"] = args.max_seq_length
    if "packing" in trainer_params:
        trainer_values["packing"] = args.packing
    trainer = SFTTrainer(**trainer_values)
    if args.assistant_only_loss:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
