#!/usr/bin/env python3
"""QLoRA fine-tuning for graph-grounded code explanation and static-analysis QA.

The training flow follows ``train_hierarchical.py``.  Training messages are
rebuilt from each record using the HCG Code explanation client contract so old
JSONL files automatically use the current inference prompt without rewriting
or regenerating the dataset.
"""

from __future__ import annotations

import argparse
import inspect
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILES = (
    ROOT_DIR / "dataset" / "static_analysis_qa_10000" / "java" / "train.jsonl",
    ROOT_DIR / "dataset" / "static_analysis_qa_10000" / "python" / "train.jsonl",
)
DEFAULT_HIERARCHICAL_MODEL = (
    ROOT_DIR
    / "lora_model_hierarchical_from_lora_model_code_repair_unsloth_qwen3_4b_instruct_2507_unsloth_bnb_4bit"
)
EXPECTED_ROLES = ("system", "user", "assistant")
EXPLANATION_SYSTEM_PROMPT = (
    "You are a code static-analysis assistant. Answer using only the supplied "
    "source code and graphs.\n"
    "Return one JSON object only. Do not use markdown, tool calls, reasoning "
    "tags, or text outside the JSON.\n"
    'The JSON schema is: {"answer":"string","relevant_lines":[]}.\n'
    'Populate "relevant_lines" with the source line numbers required by the question.\n'
    'Use the same natural language as the question for "answer".\n'
    'Explain the code behavior in "answer"; do not answer with only line numbers.\n'
    '"relevant_lines" must contain only directly supporting 1-based source line '
    "numbers, never graph node IDs."
)
TASK_TYPES = {
    "DATA_FLOW",
    "VARIABLE_DEF",
    "VARIABLE_USE",
    "FORWARD_SLICE",
    "BACKWARD_SLICE",
    "DATA_DEPENDENCY",
    "CONTROL_DEPENDENCY",
    "CONTROL_FLOW",
    "BRANCH_PATH",
    "REACHABILITY",
    "AST_SUBTREE",
    "AST_RELATION",
}


def model_suffix_from_name(model_name: str) -> str:
    compact_name = Path(model_name).name or model_name
    return re.sub(r"[^a-zA-Z0-9]+", "_", compact_name).strip("_").lower()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a lightweight LLM to answer graph-grounded code "
            "static-analysis questions."
        )
    )
    parser.add_argument(
        "--train-files",
        nargs="+",
        default=[str(path) for path in DEFAULT_TRAIN_FILES],
        help="Java and Python static-analysis QA JSONL files.",
    )
    parser.add_argument(
        "--model-name",
        default=str(DEFAULT_HIERARCHICAL_MODEL),
        help=(
            "Saved hierarchical-graph LoRA directory, merged model directory, "
            "or base Hugging Face model id. Existing LoRA adapters are continued."
        ),
    )
    parser.add_argument("--output-dir")
    parser.add_argument("--save-dir")
    parser.add_argument("--max-seq-length", type=int, default=32768)
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
    parser.add_argument(
        "--packing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Packing is disabled by default because examples contain long graphs.",
    )
    parser.add_argument(
        "--assistant-only-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask system/user tokens and train only on assistant JSON answers.",
    )
    parser.add_argument(
        "--drop-overlength",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop complete conversations exceeding --max-seq-length.",
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and preview JSONL records without loading a model.",
    )
    parser.add_argument("--preview-samples", type=int, default=2)
    args = parser.parse_args(argv)

    suffix = model_suffix_from_name(args.model_name)
    args.output_dir = args.output_dir or str(
        ROOT_DIR / "outputs" / f"code_graph_explanation_{suffix}"
    )
    args.save_dir = args.save_dir or str(
        ROOT_DIR / f"lora_model_code_graph_explanation_{suffix}"
    )

    if args.max_seq_length < 1024:
        parser.error("--max-seq-length must be at least 1024")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1")
    if args.per_device_train_batch_size < 1:
        parser.error("--per-device-train-batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        parser.error("--gradient-accumulation-steps must be at least 1")
    if args.num_train_epochs <= 0:
        parser.error("--num-train-epochs must be positive")
    if args.preview_samples < 0:
        parser.error("--preview-samples cannot be negative")
    return args


def iter_jsonl(path: Path) -> Iterable[tuple[int, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error


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


def validate_assistant_answer(
    content: str,
    *,
    record: dict[str, Any],
    sample_id: str,
    code_line_count: int,
) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Sample {sample_id}: assistant content must be valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict) or set(payload) != {"answer", "relevant_lines"}:
        raise ValueError(
            f"Sample {sample_id}: assistant JSON must contain only answer and relevant_lines."
        )
    answer = payload.get("answer")
    relevant_lines = payload.get("relevant_lines")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"Sample {sample_id}: assistant answer must be non-empty.")
    if (
        not isinstance(relevant_lines, list)
        or not relevant_lines
        or any(
            type(line) is not int or line < 1 or line > code_line_count
            for line in relevant_lines
        )
    ):
        raise ValueError(f"Sample {sample_id}: invalid relevant_lines.")
    if relevant_lines != sorted(set(relevant_lines)):
        raise ValueError(
            f"Sample {sample_id}: relevant_lines must be sorted and unique."
        )
    if answer.strip() != str(record.get("answer", "")).strip():
        raise ValueError(
            f"Sample {sample_id}: assistant answer differs from the top-level answer."
        )
    if relevant_lines != record.get("relevant_lines"):
        raise ValueError(
            f"Sample {sample_id}: assistant relevant_lines differ from the top-level field."
        )
    return payload


def build_client_user_prompt(
    *,
    language: str,
    code: str,
    ast_graph: str,
    dependency_enriched_cfg: str,
    question: str,
) -> str:
    """Mirror HCG's LlmGraphExplanationService._prompt exactly."""
    return f"""Task: Answer the static-analysis question.
Language: {language}

Question:
<question>
{question}
</question>

Numbered source code:
<source>
{number_source(code)}
</source>

AST in DOT format:
{compact_dot_graph(ast_graph, name="AST", code=code, max_nodes=64)}

Dependency-enriched CFG in DOT format:
{compact_dot_graph(dependency_enriched_cfg, name="CFG", code=code, max_nodes=96)}

Return exactly:
{{"answer":"your answer","relevant_lines":[]}}

Rules:
- Base the answer only on the supplied source and graphs.
- Use the source line numbers printed before "|".
- relevant_lines may be non-contiguous.
- For a control-flow question, include the controlling condition and the directly affected branch, call, or return statements.
- When the question compares branches or outcomes, describe every relevant outcome in the answer.
- For a data-flow question, include the required definition and use statements.
- For a "where/called/used/referenced" question, include only the exact source lines containing the requested occurrence; do not include surrounding control flow.
- Do not put AST or CFG node IDs in relevant_lines.
- Do not add unrelated lines merely to make a continuous range."""


def number_source(code: str) -> str:
    lines = code.split("\n")
    width = len(str(len(lines)))
    return "\n".join(
        f"{str(index).rjust(width)} | {line}"
        for index, line in enumerate(lines, start=1)
    )


def _dot_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "")
    )


def _attribute(attributes: str, name: str) -> str | None:
    match = re.search(rf'\b{re.escape(name)}\s*=\s*"([^"]*)"', attributes)
    return match.group(1) if match else None


def compact_dot_graph(graph: str, *, name: str, code: str, max_nodes: int) -> str:
    """Mirror the compact graph representation sent by the HCG client."""
    node_pattern = re.compile(
        r'^\s*("[^"\n]+"|[^\s\[\]-]+)\s*\[([^;\n]+)\]?\s*;?\s*$',
        re.MULTILINE,
    )
    edge_pattern = re.compile(
        r'^\s*("[^"\n]+"|[^\s\[\]-]+)\s*->\s*'
        r'("[^"\n]+"|[^\s\[\];]+)(?:\s*\[([^;\n]+)\]?)?\s*;?\s*$',
        re.MULTILINE,
    )
    source_lines = code.split("\n")
    nodes: list[dict[str, Any]] = []
    for match in node_pattern.finditer(graph):
        node_id = match.group(1).strip('"')
        attributes = match.group(2)
        node_type = _attribute(attributes, "type") or "process_statement"
        offset = re.search(r'(?:lines:)?(\d+)\s*[-:]\s*(\d+)', attributes)
        line = int(offset.group(1)) if offset else 1
        line = min(max(line, 1), max(len(source_lines), 1))
        label = _attribute(attributes, "label")
        if label is None and source_lines:
            label = source_lines[line - 1].strip()
        label = label or node_type
        if label.lstrip().startswith(("#", "//", "/*", "*", '"""', "'''")):
            continue
        nodes.append({"id": node_id, "type": node_type, "line": line, "label": label})

    leaf_types = {
        "identifier",
        "type-node",
        "type",
        "value",
        "type_identifier",
        "var_identifier",
        "method_identifier",
        "literal_value",
    }
    structural = [node for node in nodes if node["type"] not in leaf_types]
    leaves = [node for node in nodes if node["type"] in leaf_types]
    selected = (structural + leaves)[:max_nodes]
    prefix = "a" if name.upper().startswith("AST") else "c"
    short_ids = {node["id"]: f"{prefix}{index}" for index, node in enumerate(selected)}
    output = [f"digraph {name}{{"]
    for node in selected:
        label = node["label"]
        if len(label) > 80:
            label = label[:77] + "..."
        output.append(
            f'{short_ids[node["id"]]}[t="{_dot_escape(node["type"])}",'
            f'l={node["line"]},x="{_dot_escape(label)}"];'
        )
    for match in edge_pattern.finditer(graph):
        source = match.group(1).strip('"')
        target = match.group(2).strip('"')
        if source not in short_ids or target not in short_ids:
            continue
        if name.upper().startswith("AST"):
            output.append(f"{short_ids[source]}->{short_ids[target]};")
        else:
            edge_type = _attribute(match.group(3) or "", "type") or "control_flow"
            if edge_type == "data" or edge_type.startswith("data_dependency"):
                edge_type = "data_dependency"
            elif edge_type.startswith("control-") or edge_type.startswith(
                "control_dependency"
            ):
                edge_type = "control_dependency"
            else:
                edge_type = "control_flow"
            output.append(
                f'{short_ids[source]}->{short_ids[target]}[t="{edge_type}"];'
            )
    omitted = len(nodes) - len(selected)
    if omitted > 0:
        output.append(f"// {omitted} low-priority nodes omitted")
    output.append("}")
    return "\n".join(output)


def location_evidence(question: str, code: str) -> list[int] | None:
    """Mirror the client's deterministic correction for location questions."""
    if not re.search(
        r"where|location|called|invoked|referenced|used|哪里|在哪|何处|调用|使用|引用|出现",
        question,
        re.IGNORECASE,
    ):
        return None
    chinese = re.search(
        r"(?:变量|函数|方法)?\s*`?([A-Za-z_]\w*)`?\s*(?:在哪里|在哪|何处)",
        question,
    )
    english = re.search(
        r"(?:where\s+(?:is|are)\s+|locations?\s+of\s+)([A-Za-z_]\w*)",
        question,
        re.IGNORECASE,
    )
    symbol = (chinese or english).group(1) if chinese or english else None
    if not symbol:
        return None
    source_lines = code.split("\n")
    if re.search(r"called|invoked|调用", question, re.IGNORECASE):
        call = re.compile(rf"\b{re.escape(symbol)}\s*\(")
        calls = [
            index
            for index, line in enumerate(source_lines, 1)
            if call.search(line) and not re.search(r"^\s*(?:def|class)\s+", line)
        ]
        if calls:
            return calls
    identifier = re.compile(rf"\b{re.escape(symbol)}\b")
    return [index for index, line in enumerate(source_lines, 1) if identifier.search(line)]


def validate_record(record: Any, path: Path, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError(f"{path}:{line_number}: record must be an object.")
    sample_id = str(record.get("id", f"{path.name}:{line_number}"))
    language = record.get("language")
    if language not in {"java", "python"}:
        raise ValueError(f"Sample {sample_id}: unsupported language {language!r}.")
    task_type = record.get("task_type")
    if task_type not in TASK_TYPES:
        raise ValueError(f"Sample {sample_id}: unsupported task type {task_type!r}.")

    code = record.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"Sample {sample_id}: code must be non-empty.")
    graphs = record.get("graphs")
    if not isinstance(graphs, dict):
        raise ValueError(f"Sample {sample_id}: graphs must be an object.")
    if not isinstance(graphs.get("ast"), str) or not graphs["ast"].strip():
        raise ValueError(f"Sample {sample_id}: graphs.ast must be non-empty.")
    if not isinstance(graphs.get("cfg"), str) or not graphs["cfg"].strip():
        raise ValueError(f"Sample {sample_id}: graphs.cfg must be non-empty.")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != len(EXPECTED_ROLES):
        raise ValueError(
            f"Sample {sample_id}: messages must contain system, user, and assistant."
        )
    stored_messages = [
        validate_message(message, role, sample_id)
        for message, role in zip(messages, EXPECTED_ROLES)
    ]
    assistant_payload = validate_assistant_answer(
        stored_messages[-1]["content"],
        record=record,
        sample_id=sample_id,
        code_line_count=len(code.splitlines()),
    )
    question = record.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError(f"Sample {sample_id}: question must be non-empty.")
    exact_location_lines = location_evidence(question, code)
    if exact_location_lines:
        assistant_payload["relevant_lines"] = exact_location_lines
    assistant_message = {
        "role": "assistant",
        "content": json.dumps(
            assistant_payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ),
    }

    # Do not train on the historical prompt embedded in existing records.  The
    # client contract is canonical and is reconstructed from lossless fields.
    training_messages = [
        {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_client_user_prompt(
                language=language,
                code=code,
                ast_graph=graphs["ast"],
                dependency_enriched_cfg=graphs["cfg"],
                question=question,
            ),
        },
        assistant_message,
    ]

    return {
        "id": sample_id,
        "language": language,
        "task_type": task_type,
        "code": code,
        "question": question,
        "messages": training_messages,
    }


def normalized_code(code: str) -> str:
    return code.replace("\r\n", "\n").replace("\r", "\n").strip()


def load_explanation_examples(
    train_files: list[str],
    *,
    max_samples: int | None = None,
    seed: int = 3407,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_codes: set[str] = set()
    for filename in train_files:
        path = Path(filename)
        if not path.is_file():
            raise FileNotFoundError(f"Training file does not exist: {path}")
        for line_number, record in iter_jsonl(path):
            example = validate_record(record, path, line_number)
            if example["id"] in seen_ids:
                raise ValueError(f"Duplicate sample id: {example['id']}")
            code_key = normalized_code(example["code"])
            if code_key in seen_codes:
                raise ValueError(f"Duplicate source code: {example['id']}")
            seen_ids.add(example["id"])
            seen_codes.add(code_key)
            examples.append(example)

    if not examples:
        raise ValueError("No valid code-graph explanation examples were loaded.")
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


def token_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"])


def has_lora_adapter(model: Any) -> bool:
    return bool(getattr(model, "peft_config", None))


def make_existing_adapter_trainable(model: Any) -> int:
    if hasattr(model, "enable_adapter_layers"):
        model.enable_adapter_layers()
    model.train()
    for name, parameter in model.named_parameters():
        if "lora_" in name or "modules_to_save" in name:
            parameter.requires_grad_(True)
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable == 0:
        raise RuntimeError(
            "A LoRA adapter was detected but has no trainable parameters."
        )
    return trainable


def prepare_lora_model(model: Any, fast_language_model: Any, args: argparse.Namespace):
    if has_lora_adapter(model):
        trainable = make_existing_adapter_trainable(model)
        print(
            "lora_initialization: continuing existing adapter "
            f"({trainable:,} trainable parameters)"
        )
        return model
    print("lora_initialization: creating a new adapter on the base model")
    return fast_language_model.get_peft_model(
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


def compatible_config(config_class: Any, values: dict[str, Any]) -> Any:
    parameters = inspect.signature(config_class.__init__).parameters
    return config_class(**{key: value for key, value in values.items() if key in parameters})


def preview_dataset(args: argparse.Namespace) -> None:
    examples = load_explanation_examples(
        args.train_files,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    language_counts = Counter(example["language"] for example in examples)
    task_counts = Counter(example["task_type"] for example in examples)
    print(f"train_files: {args.train_files}")
    print(f"model_name: {args.model_name}")
    print(f"output_dir: {args.output_dir}")
    print(f"save_dir: {args.save_dir}")
    print(f"max_seq_length: {args.max_seq_length}")
    print(f"num_examples: {len(examples)}")
    print(f"language_counts: {dict(sorted(language_counts.items()))}")
    print(f"task_counts: {dict(sorted(task_counts.items()))}")
    for index, example in enumerate(examples[: args.preview_samples]):
        print(f"\n===== sample {index} ({example['id']}) =====")
        print(f"language: {example['language']}")
        print(f"task_type: {example['task_type']}")
        print(json.dumps(example["messages"], ensure_ascii=False, indent=2)[:6000])


def main() -> None:
    args = parse_args()
    if args.dry_run:
        preview_dataset(args)
        return

    from unsloth import FastLanguageModel

    import torch
    from datasets import Dataset
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
    model = prepare_lora_model(model, FastLanguageModel, args)

    examples = load_explanation_examples(
        args.train_files,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    formatted_examples: list[dict[str, Any]] = []
    for example in examples:
        formatted = dict(example)
        formatted["text"] = format_messages(tokenizer, example["messages"])
        formatted["token_length"] = token_length(tokenizer, formatted["text"])
        formatted_examples.append(formatted)

    original_count = len(formatted_examples)
    if args.drop_overlength:
        formatted_examples = [
            example
            for example in formatted_examples
            if example["token_length"] <= args.max_seq_length
        ]
    dropped_count = original_count - len(formatted_examples)
    if not formatted_examples:
        raise ValueError("No samples remain after sequence-length filtering.")
    dataset = Dataset.from_list(formatted_examples)

    lengths = [example["token_length"] for example in formatted_examples]
    print(f"loaded_examples: {original_count}")
    print(f"overlength_examples_dropped: {dropped_count}")
    print(f"train_examples: {len(dataset)}")
    print(f"maximum_retained_tokens: {max(lengths)}")

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
        "eval_strategy": "no",
        "evaluation_strategy": "no",
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
        "train_dataset": dataset,
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
                "upgrade Unsloth or use --no-assistant-only-loss."
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
