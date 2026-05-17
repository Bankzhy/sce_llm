import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cfg_generator import CFGGenerator
from dataset.gen_ast import generate_error_code, java_wrapper_insert_offset, sanitize_graph_name
from sitter.kast2core import KASTParse


def indent_python_code(code: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in code.splitlines())


def build_java_parse_code(code: str) -> str:
    insert_at = java_wrapper_insert_offset(code)
    wrapper_start = "public class Test {\n"
    wrapper_end = "\n}"
    return code[:insert_at] + wrapper_start + code[insert_at:] + wrapper_end


def build_python_parse_code(code: str) -> str:
    return "class Test:\n" + indent_python_code(code)


def select_target_method(sr_project):
    methods = []
    for program in sr_project.program_list:
        for cls in program.class_list:
            methods.extend(cls.method_list)

    if not methods:
        return None

    return max(methods, key=lambda method: len(method.statement_list))


def build_cfg_graph(code: str, lang: str):
    parser = KASTParse("", lang)
    parser.setup()

    if lang == "java":
        parse_code = build_java_parse_code(code)
    elif lang == "python":
        parse_code = build_python_parse_code(code)
    else:
        raise ValueError(f"Unsupported language: {lang}")

    sr_project = parser.do_parse_content(parse_code)
    sr_method = select_target_method(sr_project)
    if sr_method is None:
        return None

    cfg_gen = CFGGenerator(sr_method=sr_method)
    if not cfg_gen.create_graph():
        return None

    return sr_method, cfg_gen


def escape_dot_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\n")
        .replace("\n", "\\n")
    )


def normalize_statement_type(statement_type: Optional[str]) -> str:
    return statement_type or "statement"


IGNORED_CFG_NODE_TYPES = {
    "comment",
    "line_comment",
    "block_comment",
    "block",
    "{",
    "}",
    "(",
    ")",
    ":",
    ";",
    ",",
    "else",
    "catch",
    "catch_formal_parameter",
}


def should_keep_cfg_node(statement) -> bool:
    statement_type = normalize_statement_type(getattr(statement, "type", None))
    if statement_type == "Fake":
        return False
    if statement_type in IGNORED_CFG_NODE_TYPES:
        return False
    return True


def normalize_statement_offset(statement, method_start_line: int) -> str:
    start_line = getattr(statement, "start_line", None)
    end_line = getattr(statement, "end_line", None)
    if start_line is None or end_line is None:
        return "lines:1-1"

    relative_start = max(1, start_line - method_start_line + 1)
    relative_end = max(relative_start, end_line - method_start_line + 1)
    return f"lines:{relative_start}-{relative_end}"


def cfg_to_dot(sr_method, cfg_gen: CFGGenerator) -> Optional[str]:
    method_name = sanitize_graph_name(sr_method.method_name)
    output = [f"digraph CFG_{method_name} {{"]

    node_id_map = {}
    next_id = 1
    method_start_line = getattr(sr_method, "start_line", 1)

    for node in cfg_gen.node_list:
        statement = node.sr_statement
        if not should_keep_cfg_node(statement):
            continue

        node_id_map[node.id] = str(next_id)
        statement_type = normalize_statement_type(getattr(statement, "type", None))
        offset = normalize_statement_offset(statement, method_start_line)
        output.append(f'    {next_id} [type="{statement_type}", offset="{offset}"];')
        next_id += 1

    if not node_id_map:
        return None

    adjacency = {}
    for edge in cfg_gen.flow_edge_list:
        adjacency.setdefault(edge.source, []).append(edge.target)

    normalized_edges = set()
    for source in node_id_map:
        stack = list(adjacency.get(source, []))
        visited = set()
        while stack:
            target = stack.pop()
            if target in visited:
                continue
            visited.add(target)

            if target in node_id_map:
                normalized_edges.add((node_id_map[source], node_id_map[target]))
            else:
                stack.extend(adjacency.get(target, []))

    for source, target in sorted(normalized_edges, key=lambda edge: (int(edge[0]), int(edge[1]))):
        output.append(f"    {source} -> {target};")

    output.append("}")
    return "\n".join(output)


def count_cfg_nodes(cfg: str) -> int:
    return sum(
        1
        for line in cfg.splitlines()
        if re.match(r"^\s*\d+\s*\[", line)
    )


def gen_cfg(code: str, lang: str) -> Optional[str]:
    cfg_graph = build_cfg_graph(code, lang)
    if cfg_graph is None:
        return None
    sr_method, cfg_gen = cfg_graph
    return cfg_to_dot(sr_method, cfg_gen)


def row_to_code(row) -> Optional[str]:
    if "func_code_string" in row and row["func_code_string"]:
        return row["func_code_string"].strip()
    if "code" in row and row["code"]:
        return row["code"].strip()
    return None


def load_code_dataset(load_dataset, dataset_name: str, split: str, lang: str, config: Optional[str]):
    if dataset_name == "code-search-net/code_search_net":
        return load_dataset(dataset_name, config or lang, split=split)
    return load_dataset(dataset_name, config, split=split)


def split_total(total: Optional[int], parts: int) -> List[Optional[int]]:
    if total is None:
        return [None] * parts

    base = total // parts
    remainder = total % parts
    return [base + (1 if index < remainder else 0) for index in range(parts)]


def count_normal_rows(rows: List[dict]) -> int:
    return sum(1 for row in rows if row["is_error"] is False)


def build_rows_for_split(
    dataset,
    lang: str,
    split_name: str,
    min_nodes: int,
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
            code = row_to_code(row)
            if not code:
                continue

            cfg = gen_cfg(code, lang)
            if cfg is None:
                continue
            if count_cfg_nodes(cfg) < min_nodes:
                continue

            sample_id = f"{split_name}_{id_start + sample_count}"
            rows.append(
                {
                    "id": sample_id,
                    "code": code,
                    "CFG": cfg,
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
                        "CFG": cfg,
                        "is_error": True,
                        "language": lang,
                    }
                )
            sample_count += 1
        except Exception as exc:
            print(f"[skip] split={split_name} index={index} lang={lang}: {exc}")
            error_count += 1

    return rows, error_count


def write_rows(path: str, rows: List[dict]) -> None:
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "code", "CFG", "is_error", "language"])
        writer.writeheader()
        writer.writerows(rows)


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
                min_nodes=args.min_nodes,
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
                min_nodes=args.min_nodes,
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


def write_preview(args) -> None:
    from datasets import load_dataset

    random.seed(args.preview_seed)
    preview_rows = []
    preview_per_lang = split_total(args.preview_total, len(args.lang))
    for lang_index, lang in enumerate(args.lang):
        dataset = load_code_dataset(load_dataset, args.dataset, args.train_split, lang, args.config)
        total = min(args.preview_source_limit, len(dataset))
        indices = list(range(total))
        if args.preview_random:
            indices = random.sample(indices, min(total, max(preview_per_lang[lang_index] * 10, preview_per_lang[lang_index])))
        lang_preview_count = 0
        for index in indices:
            if lang_preview_count >= preview_per_lang[lang_index]:
                break

            code = row_to_code(dataset[index])
            if not code:
                continue

            cfg = gen_cfg(code, lang)
            if cfg is None:
                continue
            if count_cfg_nodes(cfg) < args.min_nodes:
                continue

            preview_rows.append(
                {
                    "id": f"preview_{len(preview_rows)}",
                    "code": code,
                    "CFG": cfg,
                    "is_error": False,
                    "language": lang,
                }
            )
            lang_preview_count += 1

    with open(args.preview_output, mode="w", encoding="utf-8") as f:
        for row in preview_rows[: args.preview_total]:
            f.write("=" * 120 + "\n")
            f.write(f"ID: {row['id']}\n")
            f.write(f"LANGUAGE: {row['language']}\n")
            f.write(f"IS_ERROR: {row['is_error']}\n")
            f.write("INPUT:\n")
            f.write(row["code"] + "\n")
            f.write("OUTPUT:\n")
            f.write(row["CFG"] + "\n")

    print(f"Saved preview rows: {min(len(preview_rows), args.preview_total)} -> {args.preview_output}")


def parse_args():
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Generate CFG datasets in normalized DOT format."
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
    parser.add_argument("--min-nodes", type=int, default=5)
    parser.add_argument("--train-output", default=str(output_dir / "cfg_codesearchnet_train.csv"))
    parser.add_argument("--test-output", default=str(output_dir / "cfg_codesearchnet_test.csv"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-total", type=int, default=5)
    parser.add_argument("--preview-source-limit", type=int, default=20)
    parser.add_argument("--preview-random", action="store_true")
    parser.add_argument("--preview-seed", type=int, default=3407)
    parser.add_argument("--preview-output", default=str(output_dir / "cfg_first5_preview.txt"))
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.preview:
        write_preview(parsed_args)
    else:
        gen(parsed_args)
