import csv
import json
import os
from zss import Node, simple_distance
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import load, generate

# 加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True


device = "mps"

model_name = "mlx-community/IQuest-Coder-V1-7B-Thinking-mlx_8bit"
token = "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"
# tokenizer = AutoTokenizer.from_pretrained(model_name, token=token, t)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16,
#     tokenizer=tokenizer,
#     token=token,
# ).to(device)
model, tokenizer = load(model_name)

def build_prompt(code):

    prompt = f"""
        You are a program analysis tool.
        
        Task:
        Generate a compact Abstract Syntax Tree (AST) for the Java code.
        
        Constraints:
        - Only keep structural nodes
        - Ignore identifiers and literals
        - Ignore punctuation tokens
        - Use JSON format
        - Output ONLY the AST
        
        Allowed nodes:
        MethodDeclaration
        IfStatement
        ForStatement
        WhileStatement
        ReturnStatement
        VariableDeclaration
        Assignment
        MethodInvocation
        
        Code:
        {code}
        
        AST:
        """
    return prompt

def load_big_code_ast():
    csv_file = os.path.join("../dataset", "ast_test_java.csv")
    examples = []

    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for index, row in enumerate(reader):
            if index == 0:
                continue

            if index == 1000:
                break

            sample = {
                "code": row[0],
                "ast": row[1],
            }
            examples.append(sample)


    return examples


def generate_ast(prompt):
    # inputs = tokenizer(
    #     prompt,
    #     return_tensors="pt",
    # ).to(device)
    #
    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=2048,
    #     temperature=0.1,
    # )
    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #
    # ast_text = result.split("AST:")[-1].strip()
    # print(ast_text)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=2048)
    print(response)
    try:
        ast = json.loads(response)
        return ast, True
    except:
        return None, False

############################################
# 5. AST → Tree (for TED)
############################################

def json_to_tree(ast):

    if isinstance(ast, dict):

        node = Node(ast.get("type", "unknown"))

        for k, v in ast.items():

            if isinstance(v, dict):
                node.addkid(json_to_tree(v))

            elif isinstance(v, list):

                for child in v:
                    node.addkid(json_to_tree(child))

        return node

    else:
        return Node(str(ast))

############################################
# 6. Node Extraction
############################################

def extract_nodes(ast):

    nodes = []

    def dfs(obj):

        if isinstance(obj, dict):

            if "type" in obj:
                nodes.append(obj["type"])

            for v in obj.values():
                dfs(v)

        elif isinstance(obj, list):

            for item in obj:
                dfs(item)

    dfs(ast)

    return nodes

############################################
# 7. Node-level F1
############################################

def node_f1(pred_ast, gt_ast):

    pred_nodes = extract_nodes(pred_ast)
    gt_nodes = extract_nodes(gt_ast)

    pred_set = set(pred_nodes)
    gt_set = set(gt_nodes)

    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return precision, recall, f1


############################################
# 8. Tree Edit Distance
############################################

def tree_edit_distance(pred_ast, gt_ast):

    try:

        pred_tree = json_to_tree(pred_ast)
        gt_tree = json_to_tree(gt_ast)

        dist = simple_distance(pred_tree, gt_tree)

        size = max(len(extract_nodes(pred_ast)), len(extract_nodes(gt_ast)))

        return dist / size

    except:
        return 1.0

if __name__ == '__main__':
    valid_count = 0
    test_data = load_big_code_ast()

    precisions = []
    recalls = []
    f1s = []
    teds = []

    for sample in test_data:
        code = sample["code"]
        ast = sample["ast"]
        prompt = build_prompt(code)

        pred_ast, valid = generate_ast(prompt)
        if not valid:
            continue
        valid_count += 1
        p, r, f1 = node_f1(pred_ast, ast)
        ted = tree_edit_distance(pred_ast, ast)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        teds.append(ted)
    valid_rate = valid_count / len(test_data)

    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_f1 = sum(f1s) / len(f1s)
    avg_ted = sum(teds) / len(teds)

    print("===== AST Generation Evaluation =====")

    print("AST Valid Rate:", valid_rate)

    print("Node Precision:", avg_precision)
    print("Node Recall:", avg_recall)
    print("Node F1:", avg_f1)

    print("Normalized TED:", avg_ted)
