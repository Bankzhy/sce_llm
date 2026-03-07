import csv
import json
import requests
import os
from zss import Node, simple_distance
# from unsloth import FastLanguageModel
# 加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/codellama-7b-bnb-4bit",
#     # max_seq_length=max_seq_length,
#     dtype=dtype,
#     load_in_4bit=load_in_4bit,
#     # token = "YOUR_HF_TOKEN", # HF Token for gated models
# )
# FastLanguageModel.for_inference(model)

def generate_ollama_local(content, history=[]):
    msg = history
    msg.append(
        {
            "role": "user",
            "content": content
        }
    )

    data = {
        "model": "llama3:latest",
        # "temperature": 0,
        "messages": msg,
        "stream": False,
        "options": {
        }
    }

    # data =  {
    #         "role": "user",
    #         "content": content
    # }

    response = requests.post("http://localhost:11434/api/chat", json=data)

    # api = APIMaster.objects.get(api_name="llama3")
    # api.api_current_count += len(ques)
    # api.save()

    result = response.json()
    result = result["message"]["content"]
    # print(result)
    return result

def build_prompt(code):

    prompt = f"""
        You are a program analysis tool.
        
        Task:
        Generate a Abstract Syntax Tree (AST) for the Java code.
        
        Constraints:
        - Only keep structural nodes
        - Ignore identifiers and literals
        - Ignore punctuation tokens
        - Use JSON format
        - Output ONLY the AST
        - The AST must be complete
        
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
    # ).to("cuda")
    #
    # outputs = model.generate(
    #     **inputs,
    #     # max_new_tokens=2048,
    #     temperature=0.1,
    # )
    # result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    result = generate_ollama_local(prompt)
    ast_text = result.split("AST:")[-1].strip()
    print(ast_text)
    try:
        ast = json.loads(ast_text)
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

    for index, sample in enumerate(test_data):
        code = sample["code"]
        ast = sample["ast"]
        prompt = build_prompt(code)

        pred_ast, valid = generate_ast(prompt)
        # with open(str(index)+".txt", "w", encoding="utf-8") as f:
        #     f.write(pred_ast)
        #     f.close()
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
