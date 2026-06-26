import argparse
import csv
import random
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tree_sitter_languages import get_parser

from dataset.cfg_generator import CFGGenerator
from dataset.gen_ast import java_wrapper_insert_offset, sanitize_graph_name
from dataset.gen_pdg import PDGBuilder
from sitter.kast2core import KASTParse


AST_NODE_TYPES = {
    "process_statement",
    "conditional_statement",
    "loop_statement",
    "return_statement",
    "type_identifier",
    "var_identifier",
    "method_identifier",
}
CFG_NODE_TYPES = {
    "process_statement",
    "conditional_statement",
    "loop_statement",
    "return_statement",
}
CSV_FIELDNAMES = ["id", "code", "AST", "CFG", "PDG", "is_error", "language"]
DEFAULT_LANGUAGES = ("java", "python", "javascript")
DEFAULT_CODESEARCHNET_DATASET = "code-search-net/code_search_net"
LEETCODE_DATASET = "greengerong/leetcode"
DEFAULT_LEETCODE_TRAIN_ROWS = 2000
DEFAULT_MIN_CODE_LINES = 5
DEFAULT_MAX_CODE_LINES = 30
JAVA_ERROR_SYMBOLS = [
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
PYTHON_ERROR_SYMBOLS = [
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


def error_symbols_for_lang(lang: str) -> List[str]:
    if lang == "python":
        return PYTHON_ERROR_SYMBOLS
    if lang in {"java", "javascript"}:
        return JAVA_ERROR_SYMBOLS
    raise ValueError(f"Unsupported language: {lang}")


def generate_error_code(code: str, lang: str) -> str:
    symbols = sorted(error_symbols_for_lang(lang), key=len, reverse=True)
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


def generate_error_code_list(
    code: str,
    lang: str,
    num: int,
    min_code_lines: int = DEFAULT_MIN_CODE_LINES,
    max_code_lines: int = DEFAULT_MAX_CODE_LINES,
) -> List[str]:
    variants = set()
    max_attempts = max(1, num * 80)
    attempts = 0

    while len(variants) < num and attempts < max_attempts:
        attempts += 1
        mutated = generate_error_code(code, lang)
        if (
            mutated != code
            and code_line_count_in_range(mutated, min_code_lines, max_code_lines)
        ):
            variants.add(mutated)

    return list(variants)


def extract_code(text: str) -> str:
    pattern = r"```(?:\w+)?\s*\n(.*?)```"
    match = re.search(pattern, text or "", re.DOTALL)
    return (match.group(1) if match else text or "").strip()


def row_to_code(row, lang: Optional[str] = None) -> Optional[str]:
    if lang and hasattr(row, "get") and row.get(lang):
        return extract_code(row.get(lang))
    for field in ("func_code_string", "whole_func_string", "code"):
        value = row.get(field) if hasattr(row, "get") else None
        if value:
            return extract_code(value)
    return None


def code_line_count(code: str) -> int:
    return len(code.replace("\\n", "\n").strip().splitlines())


def code_line_count_in_range(code: str, min_lines: int, max_lines: int) -> bool:
    line_count = code_line_count(code)
    return min_lines <= line_count <= max_lines


def make_csv_row(row: Dict[str, object]) -> Dict[str, object]:
    return {
        key: (
            value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
            if isinstance(value, str)
            else value
        )
        for key, value in row.items()
    }


def load_language_dataset(load_dataset, dataset_name: str, lang: str, split: str):
    if dataset_name == LEETCODE_DATASET:
        return load_dataset(dataset_name, split="train")
    return load_dataset(dataset_name, lang, split=split)


def source_index_bounds(
    dataset_len: int,
    dataset_name: str,
    split: str,
    leetcode_train_rows: int,
    max_source_samples: Optional[int],
) -> Tuple[int, int]:
    if dataset_name == LEETCODE_DATASET:
        split_name = split.lower()
        train_end = min(leetcode_train_rows, dataset_len)
        if split_name == "train":
            start, end = 0, train_end
        elif split_name in {"test", "validation", "valid", "eval"}:
            start, end = train_end, dataset_len
        else:
            raise ValueError(
                f"Unsupported split for {LEETCODE_DATASET}: {split}. Use train or test."
            )
    else:
        start, end = 0, dataset_len

    if max_source_samples is not None:
        end = min(end, start + max_source_samples)
    return start, end


def indent_code(code: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in code.splitlines())


def java_method_is_already_in_class(code: str) -> bool:
    tree = get_parser("java").parse(code.encode("utf-8"))
    methods = [
        node
        for node in iter_tree_nodes(tree.root_node)
        if node.type in {"method_declaration", "constructor_declaration"}
    ]
    if not methods:
        return False

    methods_in_class = 0
    for method in methods:
        parent = method.parent
        while parent is not None and parent != tree.root_node:
            if parent.type in {
                "class_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            }:
                methods_in_class += 1
                break
            parent = parent.parent
    return methods_in_class == len(methods)


def build_sr_parse_code(code: str, lang: str) -> str:
    if lang == "java":
        if java_method_is_already_in_class(code):
            return code
        insert_at = java_wrapper_insert_offset(code)
        return code[:insert_at] + "public class Test {\n" + code[insert_at:] + "\n}"
    if lang == "python":
        return "class Test:\n" + indent_code(code)
    if lang == "javascript":
        return "class Test {\n" + code + "\n}"
    raise ValueError(f"Unsupported language: {lang}")


def build_ast_parse_code(code: str, lang: str) -> str:
    if lang == "java":
        return build_sr_parse_code(code, lang)
    if lang in {"python", "javascript"}:
        return code
    raise ValueError(f"Unsupported language: {lang}")


def iter_tree_nodes(root) -> Iterable:
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(reversed(node.named_children))


def find_target_method(root, lang: str):
    method_types = {
        "java": {"method_declaration", "constructor_declaration"},
        "python": {"function_definition"},
        "javascript": {
            "function",
            "method_definition",
            "function_declaration",
            "function_expression",
            "arrow_function",
        },
    }[lang]
    methods = [node for node in iter_tree_nodes(root) if node.type in method_types]
    return max(methods, key=lambda node: node.end_byte - node.start_byte) if methods else None


def get_method_name(method_node, lang: str) -> str:
    name = method_node.child_by_field_name("name")
    if name is not None:
        return name.text.decode("utf-8", errors="replace")
    if lang == "java":
        for child in method_node.named_children:
            if child.type == "identifier":
                return child.text.decode("utf-8", errors="replace")
    return "Anonymous"


def is_method_identifier(node) -> bool:
    parent = node.parent
    if parent is None:
        return False

    name_node = parent.child_by_field_name("name")
    function_node = parent.child_by_field_name("function")
    if node == name_node and parent.type in {
        "method_declaration",
        "constructor_declaration",
        "function_definition",
        "method_definition",
        "function_declaration",
        "method_invocation",
        "call",
        "call_expression",
    }:
        return True
    if node == function_node:
        return True
    return parent.type in {"method_invocation", "call_expression"} and node.type in {
        "identifier",
        "property_identifier",
    }


def normalize_ast_node_type(node) -> Optional[str]:
    node_type = node.type.lower()

    # A method/function declaration is only a container in the source AST.
    # Its name is emitted separately as method_identifier; mapping the whole
    # declaration to process_statement creates a false root statement.
    if node_type in {
        "method_declaration",
        "constructor_declaration",
        "function_definition",
        "method_definition",
        "function_declaration",
        "function_expression",
        "function",
        "arrow_function",
    }:
        return None

    if node_type in {
        "type_identifier",
        "integral_type",
        "floating_point_type",
        "boolean_type",
        "void_type",
        "primitive_type",
        "predefined_type",
    } or node_type.endswith("_type"):
        return "type_identifier"

    if node_type in {"identifier", "property_identifier", "field_identifier"}:
        return "method_identifier" if is_method_identifier(node) else "var_identifier"

    if "return" in node_type:
        return "return_statement"
    if any(word in node_type for word in ("if_", "switch", "conditional", "match_")):
        return "conditional_statement"
    if any(word in node_type for word in ("for_", "while", "do_statement", "loop")):
        return "loop_statement"
    if (
        node_type.endswith(("statement", "declaration", "assignment"))
        or node_type
        in {
            "assignment",
            "assignment_expression",
            "variable_declarator",
            "expression_list",
        }
    ):
        return "process_statement"
    return None


def normalize_statement_type(statement_type: Optional[str]) -> str:
    value = (statement_type or "").lower()
    if "return" in value:
        return "return_statement"
    if any(word in value for word in ("if", "switch", "conditional", "case")):
        return "conditional_statement"
    if any(word in value for word in ("for", "while", "loop", "do_statement")):
        return "loop_statement"
    return "process_statement"


def line_byte_ranges(method_bytes: bytes) -> List[Tuple[int, int]]:
    lines = method_bytes.splitlines(keepends=True) or [b""]
    ranges = []
    cursor = 0
    for line in lines:
        content_end = cursor + len(line.rstrip(b"\r\n"))
        ranges.append((cursor, content_end))
        cursor += len(line)
    return ranges


def offset_from_lines(
    start_line: int,
    end_line: int,
    byte_ranges: List[Tuple[int, int]],
) -> str:
    start_line = max(1, min(start_line, len(byte_ranges)))
    end_line = max(start_line, min(end_line, len(byte_ranges)))
    return f"lines:{start_line}-{end_line}"


def escape_dot_value(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\r", "\\r")
        .replace("\n", "\\n")
    )


def ast_to_dot(method_node, method_name: str, test_mode: bool = False) -> str:
    method_start_line = method_node.start_point[0]
    method_start_byte = method_node.start_byte
    nodes: List[Tuple[str, str, str, str]] = []
    edges: List[Tuple[str, str]] = []
    node_by_key: Dict[Tuple[str, int, int], str] = {}
    next_id = 1

    def visit(node, parent_id: Optional[str] = None) -> Optional[str]:
        nonlocal next_id

        # The declaration node is a structural container, while the schema has
        # no dedicated method-declaration type. Use its method_identifier as
        # the AST root and attach parameters/body nodes to that identifier.
        if node == method_node:
            name_node = node.child_by_field_name("name")
            if name_node is None:
                name_node = next(
                    (
                        child
                        for child in node.named_children
                        if normalize_ast_node_type(child) == "method_identifier"
                    ),
                    None,
                )

            method_root_id = visit(name_node, None) if name_node is not None else None
            for child in node.named_children:
                if child != name_node:
                    visit(child, method_root_id)
            return method_root_id

        normalized_type = normalize_ast_node_type(node)
        current_id = None

        if normalized_type in AST_NODE_TYPES:
            start_line = node.start_point[0] - method_start_line + 1
            end_line = node.end_point[0] - method_start_line + 1
            start_byte = node.start_byte - method_start_byte
            end_byte = node.end_byte - method_start_byte
            node_key = (normalized_type, start_byte, end_byte)
            current_id = node_by_key.get(node_key)

            if current_id is None:
                current_id = str(next_id)
                next_id += 1
                node_by_key[node_key] = current_id
                offset = f"lines:{start_line}-{end_line}"
                label = escape_dot_value(
                    node.text.decode("utf-8", errors="replace")
                )
                nodes.append((current_id, normalized_type, offset, label))

            if parent_id is not None and parent_id != current_id:
                edges.append((parent_id, current_id))

        effective_parent = current_id if current_id is not None else parent_id
        for child in node.named_children:
            visit(child, effective_parent)
        return effective_parent

    visit(method_node)
    output = [f"digraph AST_{sanitize_graph_name(method_name)} {{"]
    if test_mode:
        output.extend(
            (
                f'    {node_id} [type="{node_type}", offset="{offset}", '
                f'label="{label}"];'
            )
            for node_id, node_type, offset, label in nodes
        )
    else:
        for node_id, node_type, offset, label in nodes:
            if node_type in {"type_identifier", "var_identifier", "method_identifier"}:
                output.append(
                    f'    {node_id} [type="{node_type}", offset="{offset}", '
                    f'label="{label}"];'
                )
            else:
                output.append(f'    {node_id} [type="{node_type}", offset="{offset}"];')
    output.extend(
        f"    {source} -> {target};"
        for source, target in dict.fromkeys(edges)
    )
    output.append("}")
    return "\n".join(output)


def unescape_dot_value(value: str) -> str:
    return (
        value.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace('\\"', '"')
        .replace("\\\\", "\\")
    )


def render_ast_tree(ast_dot: str, output_path: str) -> None:
    """Render an AST DOT string as a top-down PNG tree without Graphviz."""
    from PIL import Image, ImageDraw, ImageFont

    node_with_label_pattern = re.compile(
        r'^\s*(\d+)\s+\[type="([^"]+)", offset="([^"]+)", '
        r'label="((?:\\.|[^"])*)"\];$'
    )
    node_without_label_pattern = re.compile(
        r'^\s*(\d+)\s+\[type="([^"]+)", offset="([^"]+)"\];$'
    )
    edge_pattern = re.compile(r"^\s*(\d+)\s*->\s*(\d+);$")
    nodes = {}
    children: Dict[str, List[str]] = {}
    parents = set()

    for line in ast_dot.splitlines():
        node_match = node_with_label_pattern.match(line)
        if node_match:
            node_id, node_type, offset, label = node_match.groups()
            nodes[node_id] = {
                "type": node_type,
                "offset": offset,
                "label": unescape_dot_value(label),
            }
            continue
        node_match = node_without_label_pattern.match(line)
        if node_match:
            node_id, node_type, offset = node_match.groups()
            nodes[node_id] = {
                "type": node_type,
                "offset": offset,
                "label": "",
            }
            continue

        edge_match = edge_pattern.match(line)
        if edge_match:
            source, target = edge_match.groups()
            children.setdefault(source, []).append(target)
            parents.add(target)

    if not nodes:
        raise ValueError("AST DOT contains no renderable nodes")

    roots = [node_id for node_id in nodes if node_id not in parents]
    if not roots:
        roots = [next(iter(nodes))]

    positions = {}
    next_leaf_x = 0.0

    def layout(node_id: str, depth: int, visiting: set) -> float:
        nonlocal next_leaf_x
        if node_id in positions:
            return positions[node_id][0]
        if node_id in visiting:
            x = next_leaf_x
            next_leaf_x += 1.0
            positions[node_id] = (x, -depth)
            return x

        visiting.add(node_id)
        child_ids = children.get(node_id, [])
        child_x = [layout(child, depth + 1, visiting) for child in child_ids]
        visiting.remove(node_id)

        if child_x:
            x = sum(child_x) / len(child_x)
        else:
            x = next_leaf_x
            next_leaf_x += 1.0
        positions[node_id] = (x, -depth)
        return x

    for root_index, root in enumerate(roots):
        if root_index:
            next_leaf_x += 1.0
        layout(root, 0, set())

    color_map = {
        "method_identifier": (217, 234, 253),
        "var_identifier": (237, 244, 255),
        "type_identifier": (232, 222, 248),
        "process_statement": (228, 244, 223),
        "conditional_statement": (255, 240, 191),
        "loop_statement": (255, 216, 181),
        "return_statement": (255, 214, 214),
    }
    max_depth = max(-y for _, y in positions.values())
    horizontal_spacing = 360
    vertical_spacing = 155
    box_width = 320
    box_height = 86
    margin_x = 190
    margin_y = 100
    image_width = max(900, int(max(next_leaf_x, 1.0) * horizontal_spacing + margin_x * 2))
    image_height = max(500, int((max_depth + 1) * vertical_spacing + margin_y * 2))
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        title_font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    pixel_positions = {
        node_id: (
            int(margin_x + x * horizontal_spacing),
            int(margin_y + (-y) * vertical_spacing),
        )
        for node_id, (x, y) in positions.items()
    }

    for source, targets in children.items():
        if source not in pixel_positions:
            continue
        source_x, source_y = pixel_positions[source]
        for target in targets:
            if target not in pixel_positions:
                continue
            target_x, target_y = pixel_positions[target]
            start = (source_x, source_y + box_height // 2)
            end = (target_x, target_y - box_height // 2)
            draw.line((start, end), fill=(100, 116, 139), width=2)
            draw.polygon(
                (
                    end,
                    (end[0] - 6, end[1] - 10),
                    (end[0] + 6, end[1] - 10),
                ),
                fill=(100, 116, 139),
            )

    for node_id, (x, y) in pixel_positions.items():
        node = nodes[node_id]
        label = node["label"].replace("\n", " / ")
        if len(label) > 46:
            label = label[:43] + "..."
        text_lines = [
            f'{node_id}  {node["type"]}',
            label,
            node["offset"],
        ]
        left = x - box_width // 2
        top = y - box_height // 2
        draw.rounded_rectangle(
            (left, top, left + box_width, top + box_height),
            radius=12,
            fill=color_map.get(node["type"], (243, 244, 246)),
            outline=(71, 85, 105),
            width=2,
        )
        for line_index, text in enumerate(text_lines):
            text_box = draw.textbbox((0, 0), text, font=font)
            text_width = text_box[2] - text_box[0]
            draw.text(
                (x - text_width / 2, top + 9 + line_index * 24),
                text,
                fill=(30, 41, 59),
                font=font,
            )

    title = ast_dot.splitlines()[0].replace("digraph ", "").replace(" {", "")
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((image_width - (title_box[2] - title_box[0])) / 2, 20),
        title,
        fill=(15, 23, 42),
        font=title_font,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")


def render_directed_graph(graph_dot: str, output_path: str) -> None:
    """Render a possibly cyclic CFG/PDG DOT graph as a PNG."""
    import networkx as nx
    from PIL import Image, ImageDraw, ImageFont

    node_with_label_pattern = re.compile(
        r'^\s*(\d+)\s+\[type="([^"]+)", offset="([^"]+)", '
        r'label="((?:\\.|[^"])*)"\];$'
    )
    node_without_label_pattern = re.compile(
        r'^\s*(\d+)\s+\[type="([^"]+)", offset="([^"]+)"\];$'
    )
    typed_edge_pattern = re.compile(
        r'^\s*(\d+)\s*->\s*(\d+)\s+\[type="([^"]+)"\];$'
    )
    plain_edge_pattern = re.compile(r"^\s*(\d+)\s*->\s*(\d+);$")
    nodes = {}
    edges = []

    for line in graph_dot.splitlines():
        match = node_with_label_pattern.match(line)
        if match:
            node_id, node_type, offset, label = match.groups()
            nodes[node_id] = {
                "type": node_type,
                "offset": offset,
                "label": unescape_dot_value(label),
            }
            continue
        match = node_without_label_pattern.match(line)
        if match:
            node_id, node_type, offset = match.groups()
            nodes[node_id] = {
                "type": node_type,
                "offset": offset,
                "label": "",
            }
            continue
        match = typed_edge_pattern.match(line)
        if match:
            edges.append(match.groups())
            continue
        match = plain_edge_pattern.match(line)
        if match:
            source, target = match.groups()
            edges.append((source, target, ""))

    if not nodes:
        raise ValueError("DOT graph contains no renderable nodes")

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from((source, target) for source, target, _ in edges)
    raw_positions = nx.circular_layout(graph)

    image_width = max(1400, len(nodes) * 210)
    image_height = max(900, len(nodes) * 125)
    margin_x = 230
    margin_y = 150
    box_width = 330
    box_height = 92
    image = Image.new("RGB", (image_width, image_height), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", 14)
        edge_font = ImageFont.truetype("arial.ttf", 12)
        title_font = ImageFont.truetype("arialbd.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
        edge_font = ImageFont.load_default()
        title_font = ImageFont.load_default()

    xs = [position[0] for position in raw_positions.values()]
    ys = [position[1] for position in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def scale(value, low, high, target_low, target_high):
        if high == low:
            return (target_low + target_high) / 2
        return target_low + (value - low) * (target_high - target_low) / (high - low)

    positions = {
        node_id: (
            int(scale(position[0], min_x, max_x, margin_x, image_width - margin_x)),
            int(scale(position[1], min_y, max_y, margin_y, image_height - margin_y)),
        )
        for node_id, position in raw_positions.items()
    }
    color_map = {
        "process_statement": (228, 244, 223),
        "conditional_statement": (255, 240, 191),
        "loop_statement": (255, 216, 181),
        "return_statement": (255, 214, 214),
    }
    edge_colors = {
        "control_dependency": (220, 38, 38),
        "data_dependency": (37, 99, 235),
        "": (71, 85, 105),
    }
    parallel_counts = {}

    for source, target, edge_type in edges:
        if source not in positions or target not in positions:
            continue
        key = (source, target)
        parallel_index = parallel_counts.get(key, 0)
        parallel_counts[key] = parallel_index + 1
        source_x, source_y = positions[source]
        target_x, target_y = positions[target]
        dx = target_x - source_x
        dy = target_y - source_y
        length = max((dx * dx + dy * dy) ** 0.5, 1)
        offset = (parallel_index - 0.5) * 22 if parallel_counts[key] > 1 else parallel_index * 22
        perpendicular_x = -dy / length * offset
        perpendicular_y = dx / length * offset
        start = (
            source_x + perpendicular_x,
            source_y + perpendicular_y,
        )
        end = (
            target_x + perpendicular_x,
            target_y + perpendicular_y,
        )
        color = edge_colors.get(edge_type, (71, 85, 105))
        draw.line((start, end), fill=color, width=3)
        unit_x, unit_y = dx / length, dy / length
        arrow_tip = (
            end[0] - unit_x * box_width * 0.48,
            end[1] - unit_y * box_height * 0.48,
        )
        draw.polygon(
            (
                arrow_tip,
                (
                    arrow_tip[0] - unit_x * 14 - unit_y * 7,
                    arrow_tip[1] - unit_y * 14 + unit_x * 7,
                ),
                (
                    arrow_tip[0] - unit_x * 14 + unit_y * 7,
                    arrow_tip[1] - unit_y * 14 - unit_x * 7,
                ),
            ),
            fill=color,
        )
    for node_id, (x, y) in positions.items():
        node = nodes[node_id]
        label = node["label"].replace("\n", " / ")
        if len(label) > 48:
            label = label[:45] + "..."
        text_lines = [
            f'{node_id}  {node["type"]}',
            label,
            node["offset"],
        ]
        left = x - box_width // 2
        top = y - box_height // 2
        draw.rounded_rectangle(
            (left, top, left + box_width, top + box_height),
            radius=12,
            fill=color_map.get(node["type"], (243, 244, 246)),
            outline=(71, 85, 105),
            width=2,
        )
        for line_index, text in enumerate(text_lines):
            text_box = draw.textbbox((0, 0), text, font=font)
            text_width = text_box[2] - text_box[0]
            draw.text(
                (x - text_width / 2, top + 10 + line_index * 25),
                text,
                fill=(30, 41, 59),
                font=font,
            )

    title = graph_dot.splitlines()[0].replace("digraph ", "").replace(" {", "")
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(
        ((image_width - (title_box[2] - title_box[0])) / 2, 25),
        title,
        fill=(15, 23, 42),
        font=title_font,
    )
    edge_types = {edge_type for _, _, edge_type in edges if edge_type}
    if edge_types:
        legend_x = 35
        legend_y = 30
        for edge_type in ("control_dependency", "data_dependency"):
            if edge_type not in edge_types:
                continue
            color = edge_colors[edge_type]
            draw.line(
                (legend_x, legend_y + 8, legend_x + 42, legend_y + 8),
                fill=color,
                width=4,
            )
            draw.text(
                (legend_x + 52, legend_y),
                edge_type,
                fill=color,
                font=edge_font,
            )
            legend_y += 24
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, "PNG")


def select_target_method(sr_project):
    methods = [
        method
        for program in sr_project.program_list
        for cls in program.class_list
        for method in cls.method_list
    ]
    return max(methods, key=lambda method: len(method.statement_list)) if methods else None


def build_sr_method(code: str, lang: str):
    parser = KASTParse("", lang)
    parser.setup()
    sr_project = parser.do_parse_content(build_sr_parse_code(code, lang))
    return select_target_method(sr_project)


def kept_graph_nodes(nodes) -> List:
    ignored_tokens = {"", ";", "{", "}", "(", ")", ":", ","}
    result = []
    for node in nodes:
        statement = node.sr_statement
        if getattr(statement, "type", None) == "Fake":
            continue
        words = [str(word).strip() for word in getattr(statement, "word_list", [])]
        if words and all(word in ignored_tokens for word in words):
            continue
        result.append(node)
    return result


def graph_node_data(sr_method, nodes, byte_ranges):
    method_start_line = getattr(sr_method, "start_line", 1)
    node_id_map: Dict[object, str] = {}
    output = []

    for index, node in enumerate(kept_graph_nodes(nodes), start=1):
        statement = node.sr_statement
        node_id_map[node.id] = str(index)
        start_line = max(
            1,
            getattr(statement, "start_line", method_start_line) - method_start_line + 1,
        )
        end_line = max(
            start_line,
            getattr(statement, "end_line", method_start_line) - method_start_line + 1,
        )
        node_type = normalize_statement_type(getattr(statement, "type", None))
        assert node_type in CFG_NODE_TYPES
        offset = offset_from_lines(start_line, end_line, byte_ranges)
        label = escape_dot_value(" ".join(getattr(statement, "word_list", [])))
        output.append((str(index), node_type, offset, label))

    return node_id_map, output


def normalize_edges_through_removed_nodes(edges, node_id_map):
    adjacency = {}
    for edge in edges:
        adjacency.setdefault(edge.source, []).append(edge.target)

    result = set()
    for source in node_id_map:
        stack = list(adjacency.get(source, []))
        visited = set()
        while stack:
            target = stack.pop()
            if target in visited:
                continue
            visited.add(target)
            if target in node_id_map:
                result.add((node_id_map[source], node_id_map[target]))
            else:
                stack.extend(adjacency.get(target, []))
    return result


def cfg_to_dot(
    sr_method,
    cfg_gen: CFGGenerator,
    method_bytes: bytes,
    test_mode: bool = False,
) -> str:
    method_name = sanitize_graph_name(sr_method.method_name)
    byte_ranges = line_byte_ranges(method_bytes)
    node_id_map, nodes = graph_node_data(sr_method, cfg_gen.node_list, byte_ranges)
    edges = normalize_edges_through_removed_nodes(cfg_gen.flow_edge_list, node_id_map)

    output = [f"digraph CFG_{method_name} {{"]
    if test_mode:
        output.extend(
            (
                f'    {node_id} [type="{node_type}", offset="{offset}", '
                f'label="{label}"];'
            )
            for node_id, node_type, offset, label in nodes
        )
    else:
        output.extend(
            f'    {node_id} [type="{node_type}", offset="{offset}"];'
            for node_id, node_type, offset, _ in nodes
        )
    output.extend(
        f"    {source} -> {target};"
        for source, target in sorted(edges, key=lambda item: (int(item[0]), int(item[1])))
    )
    output.append("}")
    return "\n".join(output)


def pdg_to_dot(
    sr_method,
    pdg_builder: PDGBuilder,
    method_bytes: bytes,
    test_mode: bool = False,
) -> str:
    method_name = sanitize_graph_name(sr_method.method_name)
    byte_ranges = line_byte_ranges(method_bytes)
    node_id_map, nodes = graph_node_data(sr_method, pdg_builder.node_list, byte_ranges)
    edge_type_map = {
        "control_dependence": "control_dependency",
        "control_dependency": "control_dependency",
        "data_dependence": "data_dependency",
        "data_dependency": "data_dependency",
    }
    edges = set()

    for edge in pdg_builder.cd_edge_list + pdg_builder.dd_edge_list:
        if edge.source not in node_id_map or edge.target not in node_id_map:
            continue
        source = node_id_map[edge.source]
        target = node_id_map[edge.target]
        if source != target:
            edges.add((source, target, edge_type_map[edge.type]))

    output = [f"digraph PDG_{method_name} {{"]
    if test_mode:
        output.extend(
            (
                f'    {node_id} [type="{node_type}", offset="{offset}", '
                f'label="{label}"];'
            )
            for node_id, node_type, offset, label in nodes
        )
    else:
        output.extend(
            f'    {node_id} [type="{node_type}", offset="{offset}"];'
            for node_id, node_type, offset, _ in nodes
        )
    output.extend(
        f'    {source} -> {target} [type="{edge_type}"];'
        for source, target, edge_type in sorted(
            edges, key=lambda item: (int(item[0]), int(item[1]), item[2])
        )
    )
    output.append("}")
    return "\n".join(output)


def graph_has_nodes(graph_dot: str) -> bool:
    return any(re.match(r"^\s*\d+\s+\[", line) for line in graph_dot.splitlines())


def graph_has_branch_or_loop(graph_dot: str) -> bool:
    return (
        'type="conditional_statement"' in graph_dot
        or 'type="loop_statement"' in graph_dot
    )


def gen_code_graph(
    code: str,
    lang: str,
    test_mode: bool = False,
) -> Optional[Tuple[str, str, str]]:
    ast_parser = get_parser(lang)
    ast_tree = ast_parser.parse(build_ast_parse_code(code, lang).encode("utf-8"))
    method_node = find_target_method(ast_tree.root_node, lang)
    if method_node is None:
        return None

    sr_method = build_sr_method(code, lang)
    if sr_method is None:
        return None

    method_name = get_method_name(method_node, lang)
    method_bytes = method_node.text
    ast_dot = ast_to_dot(method_node, method_name, test_mode=test_mode)
    if not graph_has_nodes(ast_dot):
        return None

    cfg_gen = CFGGenerator(sr_method=sr_method)
    if not cfg_gen.create_graph():
        return None
    cfg_dot = cfg_to_dot(sr_method, cfg_gen, method_bytes, test_mode=test_mode)
    if not graph_has_nodes(cfg_dot):
        return None

    pdg_builder = PDGBuilder(sr_method)
    if not pdg_builder.create_graph():
        return None
    pdg_dot = pdg_to_dot(sr_method, pdg_builder, method_bytes, test_mode=test_mode)
    if not graph_has_nodes(pdg_dot):
        return None
    return ast_dot, cfg_dot, pdg_dot


def gen(
    output_path: str,
    max_success_per_lang: int = 10000,
    render_ast_dir: Optional[str] = None,
    test_mode: bool = False,
    error_samples_per_code: int = 4,
    dataset_name: str = DEFAULT_CODESEARCHNET_DATASET,
    split: str = "train",
    languages: Tuple[str, ...] = DEFAULT_LANGUAGES,
    max_source_samples: Optional[int] = None,
    min_code_lines: int = DEFAULT_MIN_CODE_LINES,
    max_code_lines: int = DEFAULT_MAX_CODE_LINES,
    leetcode_train_rows: int = DEFAULT_LEETCODE_TRAIN_ROWS,
    require_branch_or_loop: bool = False,
) -> None:
    from datasets import load_dataset

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if test_mode and render_ast_dir is None:
        render_ast_dir = str(output_file.parent / "graph_test_images")

    with output_file.open("w", newline="", encoding="utf-8") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for lang in languages:
            try:
                dataset = load_language_dataset(load_dataset, dataset_name, lang, split)
            except Exception as exc:
                print(f"[load-failed] lang={lang}: {exc}")
                continue

            success_count = 0
            skipped_count = 0
            skipped_line_count = 0
            scanned_count = 0
            start_index, end_index = source_index_bounds(
                len(dataset),
                dataset_name,
                split,
                leetcode_train_rows,
                max_source_samples,
            )

            for source_index in range(start_index, end_index):
                if success_count >= max_success_per_lang:
                    break

                scanned_count += 1
                code = row_to_code(dataset[source_index], lang)
                if not code:
                    skipped_count += 1
                    continue

                if not code_line_count_in_range(code, min_code_lines, max_code_lines):
                    skipped_count += 1
                    skipped_line_count += 1
                    continue

                try:
                    graphs = gen_code_graph(code, lang, test_mode=test_mode)
                except Exception as exc:
                    skipped_count += 1
                    print(f"[skip] lang={lang} index={source_index}: {exc}")
                    continue

                if graphs is None:
                    skipped_count += 1
                    continue

                ast, cfg, pdg = graphs
                if require_branch_or_loop and not graph_has_branch_or_loop(cfg):
                    skipped_count += 1
                    continue

                error_codes = generate_error_code_list(
                    code,
                    lang,
                    error_samples_per_code,
                    min_code_lines,
                    max_code_lines,
                )
                if len(error_codes) < error_samples_per_code:
                    skipped_count += 1
                    skipped_line_count += 1
                    continue

                sample_id = f"{split}_{lang}_{success_count}"
                writer.writerow(
                    make_csv_row(
                        {
                            "id": sample_id,
                            "code": code,
                            "AST": ast,
                            "CFG": cfg,
                            "PDG": pdg,
                            "is_error": False,
                            "language": lang,
                        }
                    )
                )

                for error_index, error_code in enumerate(error_codes):
                    writer.writerow(
                        make_csv_row(
                            {
                                "id": f"{sample_id}_error_{error_index}",
                                "code": error_code,
                                "AST": ast,
                                "CFG": cfg,
                                "PDG": pdg,
                                "is_error": True,
                                "language": lang,
                            }
                        )
                    )

                if render_ast_dir:
                    image_dir = Path(render_ast_dir)
                    image_prefix = f"{lang}_{success_count}"
                    render_ast_tree(ast, str(image_dir / f"{image_prefix}_ast.png"))
                    render_directed_graph(cfg, str(image_dir / f"{image_prefix}_cfg.png"))
                    render_directed_graph(pdg, str(image_dir / f"{image_prefix}_pdg.png"))

                success_count += 1

            print(
                f"[done] lang={lang} success={success_count} "
                f"skipped={skipped_count} line_skipped={skipped_line_count} "
                f"scanned={scanned_count}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dataset/all_train.csv")
    parser.add_argument("--dataset", default=DEFAULT_CODESEARCHNET_DATASET)
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--lang",
        nargs="+",
        default=list(DEFAULT_LANGUAGES),
        choices=list(DEFAULT_LANGUAGES),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum successful original code samples to keep per language.",
    )
    parser.add_argument(
        "--max-source-samples",
        type=int,
        default=None,
        help="Optional maximum source rows to scan per language.",
    )
    parser.add_argument(
        "--render-graph-dir",
        "--render-ast-dir",
        dest="render_ast_dir",
        default=None,
        help="Optional directory for rendered AST, CFG, and PDG PNG images.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Include source labels and render AST, CFG, and PDG PNG images.",
    )
    parser.add_argument(
        "--error-samples",
        type=int,
        default=4,
        help="Number of random syntax-error code variants to attach per valid code sample.",
    )
    parser.add_argument(
        "--min-code-lines",
        type=int,
        default=DEFAULT_MIN_CODE_LINES,
        help="Minimum method/function line count to keep.",
    )
    parser.add_argument(
        "--max-code-lines",
        type=int,
        default=DEFAULT_MAX_CODE_LINES,
        help="Maximum method/function line count to keep.",
    )
    parser.add_argument(
        "--leetcode-train-rows",
        type=int,
        default=DEFAULT_LEETCODE_TRAIN_ROWS,
        help="Number of greengerong/leetcode source rows reserved for train; remaining rows are used for test.",
    )
    parser.add_argument(
        "--require-branch-or-loop",
        action="store_true",
        help="Keep only samples whose CFG contains a conditional_statement or loop_statement.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen(
        args.output,
        args.limit,
        args.render_ast_dir,
        args.test,
        args.error_samples,
        args.dataset,
        args.split,
        tuple(args.lang),
        args.max_source_samples,
        args.min_code_lines,
        args.max_code_lines,
        args.leetcode_train_rows,
        args.require_branch_or_loop,
    )
