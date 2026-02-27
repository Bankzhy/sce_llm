import json
import os
from tqdm import tqdm
import pandas as pd
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset, Dataset


# 加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="codellama/CodeLlama-7b-hf",
    max_seq_length = None,
    dtype = dtype,
    load_in_4bit = True,   # strongly recommended
)

def load_big_code_clone():
    split = "train"
    json_file = os.path.join("../bc_data", "data.json")
    file = os.path.join("../bc_data", (split + ".txt"))

    codes_1 = []
    codes_2 = []
    labels = []
    json_data = {}
    examples = []

    with open(json_file, encoding='ISO-8859-1') as jf:
        lines = jf.readlines()
        print("loading dataset:")
        for line in tqdm(lines):
            data = json.loads(line.strip())
            source = data['code']
            json_data[data["idx"]] = {
                "code": source,

            }

    with open(file, encoding='ISO-8859-1') as f:
        lines = f.readlines()
        for index,line in tqdm(enumerate(lines)):

            if index > 350000:
                break

            try:
                ll = line.split("\t")
                if ll[0] not in json_data.keys() or ll[1] not in json_data.keys():
                    continue

                codes_1.append(json_data[ll[0]]["code"])
                codes_2.append(json_data[ll[0]]["code"])
                label = ll[2].replace("\n", "")
                labels.append(int(label))
                sample = convert_sample(json_data[ll[0]]["code"], json_data[ll[1]]["code"], label)
                examples.append(sample)
            except Exception as e:
                # print(e)
                continue
    # return codes_1, codes_2, labels
    return examples

def convert_sample(code1, code2, label):
    return {
        "instruction": "Please identify the following two codes is code clone or not, answer me just 'true' or 'false', no more other words.",
        "input": f"Code A:\n{code1}\n\nCode B:\n{code2}",
        "output": (
            "TRUE"
            if label == 1 else
            "FALSE"
        )
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
    dataset = Dataset.from_pandas(pd.DataFrame(data=load_big_code_clone()))
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

