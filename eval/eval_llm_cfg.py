import csv
import os
import re
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer


########################################
# 1. 加载模型
########################################

max_seq_length = 4096

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="lora_model_1",   # 你的训练结果目录
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)


########################################
# 2. 构造推理prompt
########################################

def build_prompt(code):
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
    return f"""
        {prompt}
        
        Code:
        {code}
        
        CFG:
    """


########################################
# 3. 生成CFG
########################################

def generate_cfg(code):
    prompt = build_prompt(code)

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.1,
        do_sample=False,
    )

    result = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    cfg = result.split("CFG:")[-1].strip()
    print(cfg)
    return cfg

def load_test_data():
    csv_file = os.path.join("../dataset", "cfg_train.csv")
    examples = []

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            examples.append(row)


    return examples


def evaluate():
    examples = load_test_data()
    for example in examples:
        generated_code = generate_cfg(example[0])
        break

if __name__ == '__main__':
    evaluate()