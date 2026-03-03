import csv
import json
import random
from typing import List

from datasets import load_dataset
from tree_sitter_languages import get_parser

def build_java_parse_code(code):
    code_content = "public class Test {\n"
    code_content += code
    code_content += "}"
    return code_content

def node_to_dict(node):
    return {
        "type": node.type,
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "children": [node_to_dict(child) for child in node.children],
        "text": node.text.decode(),
    }

def generate_java_ast(code):
    parse = get_parser("java")
    tree = parse.parse(code.encode())
    # Extract method_declaration node
    method_node = None
    class_node = tree.root_node.named_children[0]

    for mchild in class_node.named_children:
        if mchild.type == "class_body":
            for child in mchild.named_children:
                if child.type == "method_declaration":
                    method_node = child
                    break
    if method_node is None:
        return None
    method_ast_json = node_to_dict(method_node)
    return method_ast_json


def generate_java_error_code(code):
    # Java separators and operators
    JAVA_SYMBOLS = [
        # Separators
        "(", ")", "{", "}", "[", "]", ";", ",", ".", "@", "::",

        # Assignment
        "=", "+=", "-=", "*=", "/=", "&=", "|=", "^=", "%=",
        "<<=", ">>=", ">>>=",

        # Arithmetic
        "+", "-", "*", "/", "%", "++", "--",

        # Comparison
        "==", "!=", ">", "<", ">=", "<=",

        # Logical
        "&&", "||", "!",

        # Bitwise
        "&", "|", "^", "~", "<<", ">>", ">>>",

        # Ternary
        "?", ":",

        # Lambda
        "->",

        # Varargs
        "..."
    ]

    # Sort by length (important!)
    JAVA_SYMBOLS.sort(key=len, reverse=True)


    occurrences = []

    for symbol in JAVA_SYMBOLS:
        start = 0
        while True:
            idx = code.find(symbol, start)
            if idx == -1:
                break
            occurrences.append((idx, idx + len(symbol)))
            start = idx + 1

    if not occurrences:
        return code

    start, end = random.choice(occurrences)

    return code[:start] + code[end:]

def gen():
    rows = []
    dataset = load_dataset('code-search-net/code_search_net', 'java', split='test')
    for index in range(0, len(dataset)):
        data = dataset[index]
        code = build_java_parse_code(data['func_code_string'])

        ast = generate_java_ast(code)
        if ast is None:
            continue
        ast_json = json.dumps(ast)

        error_code_list = generate_java_error_code_list(data['func_code_string'], 3)
        rows.append(
            {
                "code": data['func_code_string'],
                "ast_json": ast_json,
                "label": 0
            }
        )

        for error_code in error_code_list:
            rows.append(
                {
                    "code": error_code,
                    "ast_json": ast_json,
                    "label": 1
                }
            )


        # print(json.dumps(ast, indent=2))
    # ---- Write CSV ----
    with open("test_java.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "ast_json", "label"])
        writer.writeheader()
        writer.writerows(rows)

        print(f"Saved to output.csv")
def generate_java_error_code_list(method_code, num) -> List[str]:
    """
    Generate multi unique syntax-error variants
    by removing random Java symbols.
    """
    variants = set()

    while len(variants) < num:
        mutated = generate_java_error_code(method_code)
        if mutated != method_code:
            variants.add(mutated)

    return list(variants)

if __name__ == '__main__':
    gen()