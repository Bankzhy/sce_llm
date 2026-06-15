import re

from datasets import load_dataset

from dataset.cfg_generator import CFGGenerator
from dataset.gen_ast import java_wrapper_insert_offset
from sitter.kast2core import KASTParse


def extract_code(text):
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def build_java_parse_code(code: str) -> str:
    insert_at = java_wrapper_insert_offset(code)
    wrapper_start = "public class Test {\n"
    wrapper_end = "\n}"
    return code[:insert_at] + wrapper_start + code[insert_at:] + wrapper_end


def build_python_parse_code(code: str) -> str:
    spaces = 4
    prefix = " " * spaces
    indent_python_code = "\n".join(prefix + line if line.strip() else line for line in code.splitlines())
    return "class Test:\n" + indent_python_code

def build_js_parse_code(code: str) -> str:
    code_content = "class Test {\n"
    code_content += code
    code_content += "}"
    return code_content

def select_target_method(sr_project):
    methods = []
    for program in sr_project.program_list:
        for cls in program.class_list:
            methods.extend(cls.method_list)

    if not methods:
        return None

    return max(methods, key=lambda method: len(method.statement_list))

def gen_code_graph(code, lang):
    if lang == "java":
        parse_code = build_java_parse_code(code)
    elif lang == "python":
        parse_code = build_python_parse_code(code)
    elif lang == "javascript":
        parse_code = build_js_parse_code(code)
    else:
        raise ValueError(f"Unsupported language: {lang}")

    parser = KASTParse("", lang)
    parser.setup()

    sr_project = parser.do_parse_content(parse_code)
    sr_method = select_target_method(sr_project)
    if sr_method is None:
        return None

    cfg_gen = CFGGenerator(sr_method=sr_method)
    if not cfg_gen.create_graph():
        return None



def gen():
    dataset = load_dataset("greengerong/leetcode", split="train")
    for index in range(0, len(dataset)):
        java_code = dataset[index]["java"]
        python_code = dataset[index]["python"]
        js_code = dataset[index]["javascript"]

        java_code = extract_code(java_code)
        python_code = extract_code(python_code)
        js_code = extract_code(js_code)

        java_ast, java_cfg, java_pdg = gen_code_graph(java_code)
        py_ast, py_cfg, py_pdg = gen_code_graph(python_code)
        js_ast, js_cfg, js_pdg = gen_code_graph(js_code)




if __name__ == '__main__':
    gen()