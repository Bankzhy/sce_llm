import csv
import json
import os
from tqdm import tqdm
import pandas as pd
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset
import sys
OVER_SIZE_LIMIT = 200_000_000

csv.field_size_limit(OVER_SIZE_LIMIT)
# 加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/codellama-7b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "YOUR_HF_TOKEN", # HF Token for gated models
)

def load_big_code_ast():
    csv_file = os.path.join("../dataset", "cfg.csv")
    examples = []

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0:
                continue

            if index == 1000:
                break

            sample = convert_sample(row[0], row[1])
            examples.append(sample)


    return examples

def convert_sample(code, cfg):
    prompt = f"""
        You are a control flow graph generator.

        Task:
        Generate the Control Flow Graph (CFG) for the given method.

        Output Requirements:
        1. Output ONLY the CFG.
        2. Do NOT output explanations.
        3. Do NOT output markdown.
        4. Follow the exact digraph format.

        CFG Rules:
        1. Create one node for each executable statement.
        2. Preserve the original execution order.
        3. Use the original statement order as node ids (starting from 1).
        4. Add a BEGIN node (id=0).
        5. Add an END node (last id).

        Control Flow Rules:
        - Sequential statements connect to the next statement.
        - If statement:
          - condition node -> true branch
          - condition node -> false branch (next statement after if block)
        - If-else statement:
          - condition node -> if branch
          - condition node -> else branch
          - both branches merge to next statement
        - Loop statement (for, while):
          - loop condition is one node
          - true branch -> loop body
          - loop body -> loop condition
          - false branch -> next statement

        Node Shapes:
        - process statement -> rectangle
        - condition statement -> diamond
        - loop statement -> hexagon
        - return/output statement -> parallelogram

        Output Format:
        digraph <method_name> {{
            0 [label="BEGIN", shape="oval"]
            <id> [label="<statement>", shape="<shape>"]
            ...
            <last_id> [label="END", shape="oval"]

            <from> -> <to>
            ...
        }}

        Example:

        Input:
        def twoSum(nums, target):
            map = {{}}
            for i, num in enumerate(nums):
                complement = target - num
                if complement in map:
                    return [map[complement], i]
                map[num] = i
            return []

        Output:
        diGraph twoSum {{
            1[label=map = {{ }}, shape=rectangle]
            2[label=for i , num in enumerate ( nums ) :, shape=hexagon]
            3[label=return [ ], shape=parallelogram]
            4[label=complement = target - num, shape=rectangle]
            5[label=if complement in map :, shape=diamond]
            6[label=map [ num ] = i, shape=rectangle]
            7[label=return [ map [ complement ] , i ], shape=parallelogram]
            1->2
            2->3
            4->5
            5->6
            6->2
            5->7
            2->4
        }}
    """
    return {
        "instruction": prompt,
        "input": f"{code}",
        "output": f"{cfg}"
    }

def formatting_prompts_func(examples):
    # 准备训练数据
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {}
    ### Input:
    {}
    ### Response:
    {}"""

    EOS_TOKEN = tokenizer.eos_token  # 必须添加 EOS_TOKEN
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # 必须添加EOS_TOKEN，否则无限生成
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }
pass

if __name__ == '__main__':
    # codes_1, codes_2, labels = load_big_code_clone()
    # dataset = load_big_code_clone()
    dataset = Dataset.from_pandas(pd.DataFrame(data=load_big_code_ast()))
    dataset = dataset.map(formatting_prompts_func, batched=True, )


    # 设置训练参数
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,
        loftq_config=None,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            max_steps=60,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir="outputs",
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
        ),
    )
    # 开始训练
    trainer.train()

    # 保存微调模型
    model.save_pretrained("lora_model_1")

    # 合并模型，保存为16位hf
    # model.save_pretrained_merged("outputs", tokenizer, save_method="merged_16bit", )

