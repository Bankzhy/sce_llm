import csv
import re
import random
from datasets import load_dataset
from typing import List
from dataset.cfg_generator import CFGGenerator
from sitter.kast2core import KASTParse


def extract_code(text):
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def gen_py_cfg(code):
    ast = KASTParse("", "python")
    ast.setup()

    code_content = "class Test:\n   "

    code_content += code

    sr_project = ast.do_parse_content(code_content)
    sr_method = None
    for program in sr_project.program_list:
        for cls in program.class_list:
            if len(cls.method_list) > 0:
                sr_method = cls.method_list[0]
    if sr_method is not None:
        cfg_gen = CFGGenerator(
            sr_method=sr_method
        )

        res = cfg_gen.create_graph()
        diGraph = cfg_gen.to_diGraph()
        return diGraph



def gen_java_cfg(code):
    ast = KASTParse("", "java")
    ast.setup()
    code_content = "public class Test {\n"
    code_content += code
    code_content += "}"
    sr_project = ast.do_parse_content(code_content)
    sr_method = None
    for program in sr_project.program_list:
        for cls in program.class_list:
            if len(cls.method_list) > 0:
                sr_method = cls.method_list[0]
    if sr_method is not None:
        cfg_gen = CFGGenerator(
            sr_method=sr_method
        )

        res = cfg_gen.create_graph()
        diGraph = cfg_gen.to_diGraph()
        return diGraph


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
def generate_python_error_code_list(code: str, num: int) -> List[str]:
    """
    Generate multiple unique syntax-error variants
    by removing random Python symbols.
    """
    variants = set()

    while len(variants) < num:
        mutated = generate_python_error_code(code)
        if mutated != code:
            variants.add(mutated)

    return list(variants)
def generate_python_error_code(code: str) -> str:
    """
    Randomly remove one Python syntax/operator symbol
    to introduce syntax errors.
    """

    PYTHON_SYMBOLS = [
        # Brackets / separators
        "(", ")", "{", "}", "[", "]", ":", ",", ".", ";",

        # Assignment operators
        "=", "+=", "-=", "*=", "/=", "//=", "%=", "**=",
        "&=", "|=", "^=", ">>=", "<<=",

        # Arithmetic
        "+", "-", "*", "/", "//", "%", "**",

        # Comparison
        "==", "!=", ">", "<", ">=", "<=",

        # Logical
        "and", "or", "not",

        # Bitwise
        "&", "|", "^", "~", "<<", ">>",

        # Lambda
        "->",   # (not real Python operator but included for robustness if parsing transformed code)

    ]

    # Sort by length (important: multi-char first)
    PYTHON_SYMBOLS.sort(key=len, reverse=True)

    occurrences = []

    for symbol in PYTHON_SYMBOLS:
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
    dataset = load_dataset("greengerong/leetcode", split="train")
    error_num = 0
    for index in range(0, len(dataset)):
        try:
            java_code = dataset[index]["java"]
            python_code = dataset[index]["python"]
            java_code = extract_code(java_code)
            python_code = extract_code(python_code)
            java_cfg = gen_java_cfg(java_code)
            python_cfg = gen_py_cfg(python_code)

            if java_cfg is not None and python_cfg is not None:
                error_java_code_list = generate_java_error_code_list(java_code, 4)
                error_python_code_list = generate_python_error_code_list(python_code, 4)
                rows.append(
                    {
                        "code": java_code,
                        "cfg": java_cfg,
                        "lang": "java",
                        "is_error": False
                    }
                )
                for error_code in error_java_code_list:
                    rows.append(
                        {
                            "code": error_code,
                            "cfg": java_cfg,
                            "lang": "java",
                            "is_error": True
                        }
                    )

                rows.append(
                    {
                        "code": python_code,
                        "cfg": python_cfg,
                        "lang": "python",
                        "is_error": False
                    }
                )
                for error_code in error_python_code_list:
                    rows.append(
                        {
                            "code": error_code,
                            "cfg": python_cfg,
                            "lang": "python",
                            "is_error": True
                        }
                    )
        except Exception as e:
            print(e)
            error_num += 1
            continue
    # ---- Write CSV ----
    with open("cfg.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["code", "cfg", "lang", "is_error"])
        writer.writeheader()
        writer.writerows(rows)

        print(f"Error: {error_num}")
        print(f"Saved to cfg.csv")


if __name__ == '__main__':
    gen()