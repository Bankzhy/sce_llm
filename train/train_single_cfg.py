import argparse
import csv
from pathlib import Path

from datasets import Dataset


csv.field_size_limit(200_000_000)

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_FILE = ROOT_DIR / "dataset" / "all_train_60k.csv"

CFG_INSTRUCTION = """You are a control flow graph generator.

Task:
Generate the Control Flow Graph (CFG) for the given Java, Python, or JavaScript method/function.

Output Requirements:
1. Output ONLY the CFG graph.
2. Do NOT output explanations.
3. Do NOT output markdown.
4. Follow the exact DOT digraph format below.

CFG Format:
digraph CFG_<MethodName> {
    <NodeID> [type="<NodeType>", offset="lines:<StartLine>-<EndLine>"];
    <SourceNodeID> -> <TargetNodeID>;
}

CFG Rules:
1. Each executable statement is one node.
2. Node ids are plain integers.
3. Use only the following CFG node types:
   process_statement, conditional_statement, loop_statement, return_statement.
4. Ignore comments, block-only nodes, punctuation, labels, and shape attributes.
5. Edges represent executable control flow and do not have attributes.
6. Treat the first line of the given method/function as line 1 in offset values.
7. The graph name must use the form CFG_<MethodName>.
"""

ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a lightweight LLM to generate CFG graphs only."
    )
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--model-name", default="unsloth/codellama-7b-bnb-4bit")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "outputs" / "single_cfg"))
    parser.add_argument("--save-dir", default=str(ROOT_DIR / "lora_model_single_cfg"))
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--include-error-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include rows whose is_error column is True. Use --no-include-error-samples for clean rows only.",
    )
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only load and format CFG samples; do not load a model or start training.",
    )
    parser.add_argument("--preview-samples", type=int, default=2)
    return parser.parse_args()


def pick_cfg_column(fieldnames: list[str] | None) -> str:
    if not fieldnames:
        raise ValueError("The training CSV has no header row.")
    if "CFG" in fieldnames:
        return "CFG"
    if "cfg" in fieldnames:
        return "cfg"
    raise ValueError("The training CSV must contain a CFG or cfg column.")


def load_cfg_examples(
    csv_file: str,
    max_samples: int | None = None,
    include_error_samples: bool = True,
) -> list[dict[str, str]]:
    examples = []

    with open(csv_file, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cfg_column = pick_cfg_column(reader.fieldnames)
        if "code" not in (reader.fieldnames or []):
            raise ValueError("The training CSV must contain a code column.")

        for row in reader:
            if not include_error_samples and row.get("is_error", "").lower() == "true":
                continue

            code = (row.get("code") or "").strip()
            cfg = (row.get(cfg_column) or "").strip()
            if not code or not cfg:
                continue

            examples.append(
                {
                    "instruction": CFG_INSTRUCTION,
                    "input": code,
                    "output": cfg,
                }
            )
            if max_samples is not None and len(examples) >= max_samples:
                break

    if not examples:
        raise ValueError(f"No valid CFG examples were loaded from {csv_file}.")
    return examples


def build_formatter(tokenizer):
    eos_token = tokenizer.eos_token or ""

    def formatting_prompts_func(batch):
        texts = []
        for instruction, input_code, output_graph in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            texts.append(ALPACA_PROMPT.format(instruction, input_code, output_graph) + eos_token)
        return {"text": texts}

    return formatting_prompts_func


def format_training_text(example: dict[str, str], eos_token: str = "") -> str:
    return ALPACA_PROMPT.format(
        example["instruction"],
        example["input"],
        example["output"],
    ) + eos_token


def preview_dataset(args: argparse.Namespace) -> None:
    examples = load_cfg_examples(
        args.train_file,
        max_samples=args.max_samples,
        include_error_samples=args.include_error_samples,
    )
    dataset = Dataset.from_list(examples)
    formatted_dataset = dataset.map(
        lambda batch: {
            "text": [
                format_training_text(
                    {
                        "instruction": instruction,
                        "input": input_code,
                        "output": output_graph,
                    }
                )
                for instruction, input_code, output_graph in zip(
                    batch["instruction"], batch["input"], batch["output"]
                )
            ]
        },
        batched=True,
    )

    print(f"train_file: {args.train_file}")
    print(f"num_examples: {len(examples)}")
    print(f"dataset_columns: {formatted_dataset.column_names}")
    print()

    for index in range(min(args.preview_samples, len(formatted_dataset))):
        row = formatted_dataset[index]
        print(f"===== sample {index} =====")
        print(f"input_chars: {len(row['input'])}")
        print(f"output_chars: {len(row['output'])}")
        print(f"text_chars: {len(row['text'])}")
        print("input_preview:")
        print(row["input"][:500])
        print("output_preview:")
        print(row["output"][:800])
        print("text_preview:")
        print(row["text"][:1000])
        print()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        preview_dataset(args)
        return

    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )

    examples = load_cfg_examples(
        args.train_file,
        max_samples=args.max_samples,
        include_error_samples=args.include_error_samples,
    )
    dataset = Dataset.from_list(examples)
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
            save_steps=args.save_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            output_dir=args.output_dir,
            seed=args.seed,
            report_to="none",
        ),
    )

    trainer.train()
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
