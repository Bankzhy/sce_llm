import argparse
import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from tree_sitter_languages import get_parser


# ==============================
# Config: important AST node types
# ==============================

IMPORTANT_NODE_TYPES = {
    # Java method / class
    "method_declaration",
    "constructor_declaration",
    "formal_parameters",
    "formal_parameter",

    # Python method / function
    "function_definition",
    "parameters",
    "typed_parameter",
    "default_parameter",

    # statements
    "block",
    "local_variable_declaration",
    "variable_declarator",
    "expression_statement",
    "assignment_expression",
    "return_statement",
    "assignment",
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
    "try_statement",
    "catch_clause",
    "finally_clause",
    "throw_statement",

    # Python control flow
    "elif_clause",
    "else_clause",
    "for_in_clause",
    "with_statement",
    "raise_statement",

    # expressions
    "binary_expression",
    "unary_expression",
    "method_invocation",
    "object_creation_expression",
    "field_access",
    "array_access",
    "call",
    "attribute",
    "subscript",
    "comparison_operator",
    "boolean_operator",
    "binary_operator",
    "unary_operator",

    # identifiers / literals / types
    "identifier",
    "type_identifier",
    "integral_type",
    "floating_point_type",
    "boolean_type",
    "void_type",
    "decimal_integer_literal",
    "string_literal",
    "integer",
    "float",
    "string",
    "true",
    "false",
    "null_literal",
    "none",
}


# Tree-sitter punctuation and modifiers that do not add useful AST supervision.
IGNORED_NODE_TYPES = {
    "{",
    "}",
    "(",
    ")",
    "[",
    "]",
    ";",
    ",",
    ".",
    "=",
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
    "!",
    "public",
    "private",
    "protected",
    "static",
    "final",
}


@dataclass
class AstNodeRecord:
    node_id: str
    node_type: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int


def extract_code(text: str) -> str:
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def sanitize_graph_name(name: str) -> str:
    name = re.sub(r"\W+", "_", name.strip())
    name = name.strip("_")
    return name or "Anonymous"


def build_parse_code(code: str, lang: str) -> Tuple[str, int, int]:
    """Return parsable code plus wrapper offsets to map nodes back to source code."""
    if lang == "java":
        insert_at = java_wrapper_insert_offset(code)
        wrapper_start = "public class Test {\n"
        wrapper_end = "\n}"
        parse_code = code[:insert_at] + wrapper_start + code[insert_at:] + wrapper_end
        return parse_code, 1, len(wrapper_start.encode("utf-8"))
    if lang == "python":
        return code, 0, 0
    raise ValueError(f"Unsupported language: {lang}")


def java_wrapper_insert_offset(code: str) -> int:
    """Insert the synthetic class after package/import lines, keeping imports legal."""
    offset = 0
    saw_header = False
    for line in code.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("package ") or stripped.startswith("import "):
            offset += len(line)
            saw_header = True
            continue
        if saw_header and stripped == "":
            offset += len(line)
            continue
        break
    return offset


def find_first_node(root, node_types: Iterable[str]):
    wanted = set(node_types)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in wanted:
            return node
        stack.extend(reversed(node.named_children))
    return None


def iter_nodes(root, node_types: Iterable[str]):
    wanted = set(node_types)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.type in wanted:
            yield node
        stack.extend(reversed(node.named_children))


def find_target_method(root, lang: str):
    if lang == "java":
        methods = list(
            iter_nodes(root, ("method_declaration", "constructor_declaration"))
        )
        non_constructor_methods = [
            node for node in methods if node.type == "method_declaration"
        ]
        if non_constructor_methods:
            return max(non_constructor_methods, key=lambda node: node.end_byte - node.start_byte)
        return methods[0] if methods else None

    functions = list(iter_nodes(root, ("function_definition",)))
    if functions:
        return max(functions, key=lambda node: node.end_byte - node.start_byte)
    return None


def get_method_name(method_node, lang: str) -> str:
    if lang == "java":
        for child in method_node.children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
    if lang == "python":
        child = method_node.child_by_field_name("name")
        if child is not None:
            return child.text.decode("utf-8", errors="replace")
    return "Anonymous"


def should_keep_node(node, important_only: bool) -> bool:
    if node.type in IGNORED_NODE_TYPES:
        return False
    if not node.is_named:
        return False
    if important_only and node.type not in IMPORTANT_NODE_TYPES:
        return False
    return True


def adjusted_offset(node, base_line: int, base_byte: int) -> Tuple[int, int, int, int]:
    start_line = max(1, node.start_point[0] - base_line + 1)
    end_line = max(start_line, node.end_point[0] - base_line + 1)
    start_byte = max(0, node.start_byte - base_byte)
    end_byte = max(start_byte, node.end_byte - base_byte)
    return start_line, end_line, start_byte, end_byte


def collect_ast(method_node, important_only: bool):
    nodes: List[AstNodeRecord] = []
    edges: List[Tuple[str, str]] = []
    next_id = 1
    base_line = method_node.start_point[0]
    base_byte = method_node.start_byte

    def visit(node, parent_id: Optional[str] = None) -> Optional[str]:
        nonlocal next_id

        current_id = None
        if should_keep_node(node, important_only):
            current_id = str(next_id)
            next_id += 1
            start_line, end_line, start_byte, end_byte = adjusted_offset(node, base_line, base_byte)
            nodes.append(
                AstNodeRecord(
                    node_id=current_id,
                    node_type=node.type,
                    start_line=start_line,
                    end_line=end_line,
                    start_byte=start_byte,
                    end_byte=end_byte,
                )
            )
            if parent_id is not None:
                edges.append((parent_id, current_id))

        next_parent_id = current_id if current_id is not None else parent_id
        for child in node.children:
            visit(child, next_parent_id)

        return current_id

    visit(method_node)
    return nodes, edges


def ast_to_dot(graph_name: str, nodes: List[AstNodeRecord], edges: List[Tuple[str, str]]) -> str:
    lines = [f"digraph AST_{sanitize_graph_name(graph_name)} {{"]

    for node in nodes:
        offset = (
            f"lines:{node.start_line}-{node.end_line};"
            f"bytes:{node.start_byte}-{node.end_byte}"
        )
        lines.append(
            f'    {node.node_id} [type="{node.node_type}", offset="{offset}"];'
        )

    for source, target in edges:
        lines.append(f"    {source} -> {target};")

    lines.append("}")
    return "\n".join(lines)


def gen_ast(code: str, lang: str = "java", important_only: bool = False) -> Optional[str]:
    parse_code, line_offset, byte_offset = build_parse_code(code, lang)
    parser = get_parser(lang)
    tree = parser.parse(parse_code.encode("utf-8"))

    if tree.root_node.has_error:
        return None

    method_node = find_target_method(tree.root_node, lang)

    if method_node is None:
        return None

    method_name = get_method_name(method_node, lang)
    nodes, edges = collect_ast(method_node, important_only)
    if not nodes:
        return None

    return ast_to_dot(method_name, nodes, edges)


def generate_error_code_list(code: str, lang: str, num: int) -> List[str]:
    variants = set()
    max_attempts = num * 20
    attempts = 0

    while len(variants) < num and attempts < max_attempts:
        attempts += 1
        mutated = generate_error_code(code, lang)
        if mutated != code:
            variants.add(mutated)

    return list(variants)


def generate_error_code(code: str, lang: str) -> str:
    java_symbols = [
        ">>>=",
        "<<=",
        ">>=",
        "...",
        "++",
        "--",
        "+=",
        "-=",
        "*=",
        "/=",
        "&=",
        "|=",
        "^=",
        "%=",
        "==",
        "!=",
        ">=",
        "<=",
        "&&",
        "||",
        "::",
        ">>>",
        "<<",
        ">>",
        "->",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        ";",
        ",",
        ".",
        "@",
        "=",
        "+",
        "-",
        "*",
        "/",
        "%",
        ">",
        "<",
        "!",
        "&",
        "|",
        "^",
        "~",
        "?",
        ":",
    ]
    python_symbols = [
        "**=",
        "//=",
        "+=",
        "-=",
        "*=",
        "/=",
        "%=",
        "&=",
        "|=",
        "^=",
        ">>=",
        "<<=",
        "==",
        "!=",
        ">=",
        "<=",
        "**",
        "//",
        "<<",
        ">>",
        "->",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        ":",
        ",",
        ".",
        ";",
        "=",
        "+",
        "-",
        "*",
        "/",
        "%",
        ">",
        "<",
        "&",
        "|",
        "^",
        "~",
    ]
    symbols = java_symbols if lang == "java" else python_symbols

    occurrences = []
    for symbol in symbols:
        start = 0
        while True:
            index = code.find(symbol, start)
            if index == -1:
                break
            occurrences.append((index, index + len(symbol)))
            start = index + 1

    if not occurrences:
        return code

    start, end = random.choice(occurrences)
    return code[:start] + code[end:]


def row_to_code(row, lang: str) -> Optional[str]:
    if lang in row and row[lang]:
        return extract_code(row[lang])
    if "func_code_string" in row and row["func_code_string"]:
        return row["func_code_string"].strip()
    if "code" in row and row["code"]:
        return row["code"].strip()
    return None


def write_rows(path: str, rows: List[dict]) -> None:
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "code", "AST", "is_error", "language"])
        writer.writeheader()
        writer.writerows(rows)


def load_code_dataset(load_dataset, dataset_name: str, split: str, lang: str, config: Optional[str]):
    if dataset_name == "code-search-net/code_search_net":
        return load_dataset(dataset_name, config or lang, split=split)
    return load_dataset(dataset_name, config, split=split)


def build_rows_for_split(
    dataset,
    lang: str,
    split_name: str,
    important_only: bool,
    id_start: int = 0,
    max_source_samples: Optional[int] = None,
    max_rows: Optional[int] = None,
) -> Tuple[List[dict], int]:
    rows = []
    error_count = 0
    sample_count = 0

    total = len(dataset) if max_source_samples is None else min(max_source_samples, len(dataset))
    for index in range(total):
        if max_rows is not None and len(rows) + 2 > max_rows:
            break

        row = dataset[index]
        try:
            code = row_to_code(row, lang)
            if not code:
                continue

            ast = gen_ast(code, lang=lang, important_only=important_only)
            if ast is None:
                continue

            sample_id = f"{split_name}_{id_start + sample_count}"
            rows.append(
                {
                    "id": sample_id,
                    "code": code,
                    "AST": ast,
                    "is_error": False,
                    "language": lang,
                }
            )

            error_code = generate_error_code(code, lang)
            if error_code != code:
                rows.append(
                    {
                        "id": f"{sample_id}_error",
                        "code": error_code,
                        "AST": ast,
                        "is_error": True,
                        "language": lang,
                    }
                )
            sample_count += 1
        except Exception as exc:
            print(f"[skip] split={split_name} index={index} lang={lang}: {exc}")
            error_count += 1

    if max_rows is not None:
        rows = rows[:max_rows]

    return rows, error_count


def gen(args) -> None:
    from datasets import load_dataset

    train_rows = []
    test_rows = []
    error_count = 0
    train_id_start = 0
    test_id_start = 0
    train_rows_per_lang = split_total(args.train_total, len(args.lang))
    test_rows_per_lang = split_total(args.test_total, len(args.lang))

    for lang_index, lang in enumerate(args.lang):
        if not args.skip_train:
            train_dataset = load_code_dataset(load_dataset, args.dataset, args.train_split, lang, args.config)
            lang_train_rows, lang_train_errors = build_rows_for_split(
                train_dataset,
                lang=lang,
                split_name="train",
                important_only=args.important_only,
                id_start=train_id_start,
                max_source_samples=args.train_limit,
                max_rows=train_rows_per_lang[lang_index],
            )
            train_rows.extend(lang_train_rows)
            train_id_start += count_normal_rows(lang_train_rows)
            error_count += lang_train_errors

        if not args.skip_test:
            test_dataset = load_code_dataset(load_dataset, args.dataset, args.test_split, lang, args.config)
            lang_test_rows, lang_test_errors = build_rows_for_split(
                test_dataset,
                lang=lang,
                split_name="test",
                important_only=args.important_only,
                id_start=test_id_start,
                max_source_samples=args.test_source_limit,
                max_rows=test_rows_per_lang[lang_index],
            )
            test_rows.extend(lang_test_rows)
            test_id_start += count_normal_rows(lang_test_rows)
            error_count += lang_test_errors

    if not args.skip_train:
        write_rows(args.train_output, train_rows)
        print(f"Saved train rows: {len(train_rows)} -> {args.train_output}")
    if not args.skip_test:
        write_rows(args.test_output, test_rows)
        print(f"Saved test rows: {len(test_rows)} -> {args.test_output}")
    print(f"Skipped samples: {error_count}")


def split_total(total: Optional[int], parts: int) -> List[Optional[int]]:
    if total is None:
        return [None] * parts

    base = total // parts
    remainder = total % parts
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def count_normal_rows(rows: List[dict]) -> int:
    return sum(1 for row in rows if row["is_error"] is False)


def parse_args():
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate AST datasets in the unified DOT-like graph format."
    )
    parser.add_argument("--dataset", default="code-search-net/code_search_net")
    parser.add_argument("--config", default=None)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--lang", nargs="+", default=["java", "python"], choices=["java", "python"])
    parser.add_argument("--train-total", type=int, default=10000)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-source-limit", type=int, default=None)
    parser.add_argument("--test-total", type=int, default=1000)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--important-only", action="store_true")
    parser.add_argument("--train-output", default=str(output_dir / "ast_codesearchnet_train.csv"))
    parser.add_argument("--test-output", default=str(output_dir / "ast_codesearchnet_test.csv"))
    return parser.parse_args()


if __name__ == "__main__":
    gen(parse_args())
