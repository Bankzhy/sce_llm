import re
from datasets import load_dataset
from tree_sitter_languages import get_parser


# ==============================
# 配置：保留的关键节点类型
# ==============================

IMPORTANT_NODE_TYPES = {
    # method / class
    "program",
    "class_declaration",
    "method_declaration",
    "constructor_declaration",

    # statements
    "block",
    "local_variable_declaration",
    "variable_declarator",
    "expression_statement",
    "assignment_expression",
    "return_statement",

    # control flow
    "if_statement",
    "for_statement",
    "enhanced_for_statement",
    "while_statement",
    "do_statement",
    "switch_expression",
    "switch_block",
    "switch_rule",
    "break_statement",
    "continue_statement",

    # try-catch
    "try_statement",
    "catch_clause",
    "finally_clause",
    "throw_statement",

    # expressions
    "binary_expression",
    "unary_expression",
    "method_invocation",
    "object_creation_expression",
    "field_access",
    "array_access",

    # identifiers / literals / types
    "identifier",
    "type_identifier",
    "integral_type",
    "floating_point_type",
    "boolean_type",
    "void_type",
    "decimal_integer_literal",
    "string_literal",
    "true",
    "false",
    "null_literal",
}


# ==============================
# 不保留的无意义节点
# ==============================

IGNORED_NODE_TYPES = {
    "{", "}", "(", ")", "[", "]",
    ";", ",", ".", "=", "+", "-", "*", "/", "%",
    "<", ">", "<=", ">=", "==", "!=",
    "&&", "||", "!",
    "public", "private", "protected",
    "static", "final",
}


def extract_code(text):
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def gen_java_ast(code):
    parse = get_parser("java")
    tree = parse.parse(code.encode())
    print(tree)


def gen():
    dataset = load_dataset("greengerong/leetcode", split="train")
    for index in range(0, len(dataset)):
        java_code = dataset[index]["java"]
        python_code = dataset[index]["python"]
        java_code = extract_code(java_code)
        python_code = extract_code(python_code)
        java_ast = gen_java_ast(java_code)



if __name__ == '__main__':
    gen()