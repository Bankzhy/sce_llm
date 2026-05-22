import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel


OVER_SIZE_LIMIT = 200_000_000
csv.field_size_limit(OVER_SIZE_LIMIT)

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"


TASKS = {
    "ast": {
        "graph_column": "AST",
        "default_train_file": DATASET_DIR / "ast_codesearchnet_train.csv",
        "default_output_dir": Path("outputs") / "ast",
        "default_save_dir": Path("lora_model_ast"),
        "instruction": """You are an abstract syntax tree graph generator.

Task:
Generate the Abstract Syntax Tree (AST) for the given Java or Python method/function.

Output Requirements:
1. Output ONLY the AST graph.
2. Do NOT output explanations.
3. Do NOT output markdown.
4. Follow the exact DOT digraph format below.

AST Format:
digraph AST_<MethodName> {
    <NodeID> [type="<NodeType>", offset="lines:<StartLine>-<EndLine>;bytes:<StartByte>-<EndByte>"];
    <SourceNodeID> -> <TargetNodeID>;
}

AST Rules:
1. Node ids are plain integers starting from 1.
2. Use tree-sitter AST node types as node type values.
3. Use parent-child AST relations as edges.
4. Edges do not have attributes.
5. Treat the first line of the given method/function as line 1 in offset values.
6. Preserve byte offsets from the given method/function code.
7. The graph name must use the form AST_<MethodName>.
""",
    },
    "cfg": {
        "graph_column": "CFG",
        "default_train_file": DATASET_DIR / "cfg_codesearchnet_train.csv",
        "default_output_dir": Path("outputs") / "cfg",
        "default_save_dir": Path("lora_model_cfg"),
        "instruction": """You are a control flow graph generator.

Task:
Generate the Control Flow Graph (CFG) for the given Java or Python method/function.

Output Requirements:
1. Output ONLY the CFG graph.
2. Do NOT output explanations.
3. Do NOT output markdown.
4. Follow the exact DOT digraph format below.

CFG Format:
digraph CFG_<MethodName> {
    <NodeID> [type="<StatementType>", offset="lines:<StartLine>-<EndLine>"];
    <SourceNodeID> -> <TargetNodeID>;
}

CFG Rules:
1. Each executable statement is one node.
2. Node ids are plain integers.
3. Node type is the statement type, such as local_variable_declaration, if_statement, for_statement, expression_statement, or return_statement.
4. Ignore comments, block-only nodes, punctuation, labels, and shape attributes.
5. Edges represent executable control flow and do not have attributes.
6. Treat the first line of the given method/function as line 1 in offset values.
7. The graph name must use the form CFG_<MethodName>.
""",
    },
    "pdg": {
        "graph_column": "PDG",
        "default_train_file": DATASET_DIR / "pdg_codesearchnet_train.csv",
        "default_output_dir": Path("outputs") / "pdg",
        "default_save_dir": Path("lora_model_pdg"),
        "instruction": """You are a program dependence graph generator.

Task:
Generate the Program Dependence Graph (PDG) for the given Java or Python method/function.

Output Requirements:
1. Output ONLY the PDG graph.
2. Do NOT output explanations.
3. Do NOT output markdown.
4. Follow the exact DOT digraph format below.

PDG Format:
digraph PDG_<MethodName> {
    <NodeID> [type="<StatementType>", offset="lines:<StartLine>-<EndLine>"];
    <SourceNodeID> -> <TargetNodeID> [type="<EdgeType>"];
}

PDG Rules:
1. Each executable statement is one node.
2. Node ids are plain integers.
3. Node type is the statement type, such as local_variable_declaration, if_statement, for_statement, expression_statement, or return_statement.
4. Ignore comments, block-only nodes, punctuation, labels, and shape attributes.
5. Edges must use only data_dependence or control_dependence.
6. Do not generate control_flow edges.
7. Treat the first line of the given method/function as line 1 in offset values.
8. The graph name must use the form PDG_<MethodName>.
""",
    },
}


ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def parse_args(graph_type: str):
    task = TASKS[graph_type]
    parser = argparse.ArgumentParser(description=f"Fine-tune an LLM to generate {graph_type.upper()} graphs.")
    parser.add_argument("--train-file", default=str(task["default_train_file"]))
    parser.add_argument("--model-name", default="unsloth/codellama-7b-bnb-4bit")
    parser.add_argument("--output-dir", default=str(task["default_output_dir"]))
    parser.add_argument("--save-dir", default=str(task["default_save_dir"]))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_graph_examples(csv_file: str, graph_type: str, max_samples: int | None = None):
    task = TASKS[graph_type]
    graph_column = task["graph_column"]
    examples = []

    with open(csv_file, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required_columns = {"code", graph_column}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing_columns)}")

        for row in reader:
            graph = row[graph_column].strip()
            code = row["code"].strip()
            if not code or not graph:
                continue
            examples.append(
                {
                    "instruction": task["instruction"],
                    "input": code,
                    "output": graph,
                }
            )
            if max_samples is not None and len(examples) >= max_samples:
                break

    return examples


def build_formatter(tokenizer):
    eos_token = tokenizer.eos_token

    def formatting_prompts_func(examples):
        texts = []
        for instruction, input_code, output_graph in zip(
            examples["instruction"], examples["input"], examples["output"]
        ):
            texts.append(ALPACA_PROMPT.format(instruction, input_code, output_graph) + eos_token)
        return {"text": texts}

    return formatting_prompts_func


def train_graph_model(graph_type: str):
    args = parse_args(graph_type)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    examples = load_graph_examples(args.train_file, graph_type, args.max_samples)
    dataset = Dataset.from_pandas(pd.DataFrame(data=examples))
    dataset = dataset.map(build_formatter(tokenizer), batched=True)

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
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        packing=args.packing,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            num_train_epochs=args.num_train_epochs,
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            output_dir=args.output_dir,
            seed=args.seed,
        ),
    )

    trainer.train()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


__all__ = ["train_graph_model"]
