from networkx.classes import edges
from py2cfg import CFGBuilder
from datasets import load_dataset
import random
from typing import List
import re

import re
from collections import OrderedDict
from collections import defaultdict

import re
from collections import defaultdict


def is_for_init(label):
    return "=" in label and "int " in label


def is_for_condition(label):
    ops = ["<", ">", "<=", ">=", "!=", "=="]
    return any(op in label for op in ops)


def is_for_update(label):
    return "++" in label or "--" in label or "+=" in label or "-=" in label


def merge_for_nodes(nodes, edges):
    """
    nodes: {id: {'label':..., 'shape':...}}
    edges: [(u,v)]

    return merged nodes + merged edges
    """

    succ = defaultdict(set)
    pred = defaultdict(set)

    for u, v in edges:
        succ[u].add(v)
        pred[v].add(u)

    removed = set()

    for cond in nodes:
        cond_label = nodes[cond].get("label", "")
        cond_shape = nodes[cond].get("shape", "")

        if cond_shape != "diamond":
            continue

        if not is_for_condition(cond_label):
            continue

        init_candidates = pred[cond]
        update_candidates = pred[cond]

        init_node = None
        update_node = None

        for n in init_candidates:
            label = nodes[n].get("label", "")

            if is_for_init(label):
                init_node = n

            if is_for_update(label):
                update_node = n

        if init_node and update_node:
            init_label = nodes[init_node]["label"]
            update_label = nodes[update_node]["label"]

            merged_label = f'for ({init_label}; {cond_label}; {update_label})'

            nodes[cond]["label"] = merged_label
            nodes[cond]["shape"] = "hexagon"

            removed.add(init_node)
            removed.add(update_node)

    # 重连边
    new_edges = []

    for u, v in edges:
        if u in removed or v in removed:
            continue
        new_edges.append((u, v))

    return nodes, new_edges, removed

def extract_attrs(attr_str):
    pattern = re.compile(r'(\w+)=(".*?"|[^,\s]+)')
    return dict(pattern.findall(attr_str))


def should_remove_node(label):
    remove_prefixes = [
        "BEGIN",
        "EXIT",
        "BLOCK_BEGIN",
        "BLOCK_END",
        "CONVERGE"
    ]
    return any(label.strip().startswith(p) for p in remove_prefixes)

def normalize_py_cfg(cfg):
    method_name = "null"
    nodes = []
    edges = []
    result = ""
    node_prefixs = []
    for index, line in enumerate(cfg.body):
        line = line.strip()
        # ---- NODE ----
        if re.match(r'^\d+\s*\[', line):

            m = re.match(r'^(\s*\d+\s*)\[(.*)\]\s*$', line)

            prefix = m.group(1)
            attrs_str = m.group(2)

            # extract attributes
            attrs = dict(re.findall(r'(\w+)=(".*?"|[^,\s]+)', attrs_str))

            label = attrs.get("label", None)
            label = label.replace("\\l", "")
            shape = attrs.get("shape", None)

            if index == 0:
                method_name = label
                continue
            if shape == "tab":
                continue

            new_attrs = []

            if label is not None:
                new_attrs.append(f'label={label}')
            if shape is not None:
                new_attrs.append(f'shape={shape}')

            # rebuild line
            if new_attrs:
                nodes.append(f"{prefix}[{', '.join(new_attrs)}]")
            else:
                nodes.append(f"{prefix}[]")
            node_prefixs.append(prefix.strip())

            continue

        # ---- EDGE (must check first? optional) ----
        if re.match(r'^\d+\s*->\s*\d+', line):
            m = re.match(r'^(\s*\d+\s*->\s*\d+\s*)(\[(.*)\])?\s*$', line)

            prefix = m.group(1)
            attr_block = m.group(3)

            if not attr_block:
                edges.append(prefix)
                continue

            attrs = dict(re.findall(r'(\w+)=(".*?"|[^,\s]+)', attr_block))

            label = attrs.get("label", None)
            label = label.replace("\\l", "")

            if label is not None:
                # edges.append(f'{prefix}[label={label}]')
                edges.append(f'{prefix}[label={label}]')
            else:
                edges.append(prefix)

            continue

    result = "diGraph "+method_name+" {"+"\n"
    for node in nodes:
        result += node
        result += "\n"
    for edge in edges:
        line = edge.rstrip()

        # edge: 10 -> 4 [...]
        m = re.match(r'^\s*(\d+)\s*->\s*(\d+)', line)
        if m:
            src = m.group(1)
            dst = m.group(2)

            # 如果 edge 引用了被删除的 node，就跳过
            if src not in node_prefixs or dst not in node_prefixs:
                continue
            result += edge
            result += "\n"
    result += "}"
    return result

def normalize_java_cfg(cfg):
    lines = cfg.splitlines()

    nodes = {}
    edges = []

    pred = defaultdict(set)
    succ = defaultdict(set)

    other_lines = []

    # --------------------------
    # parse nodes
    # --------------------------
    for line in lines:
        raw = line.rstrip()

        if re.match(r'^\s*\d+\s*\[', raw):
            m = re.match(r'^\s*(\d+)\s*\[(.*)\];?$', raw)
            if m:
                nid = m.group(1)
                attrs = extract_attrs(m.group(2))
                nodes[nid] = attrs
        else:
            other_lines.append(raw)

    # --------------------------
    # parse edges
    # --------------------------
    for line in lines:
        raw = line.rstrip()

        m = re.match(r'^\s*(\d+)\s*->\s*(\d+)', raw)
        if m:
            src = m.group(1)
            dst = m.group(2)

            edges.append((src, dst))

            succ[src].add(dst)
            pred[dst].add(src)

    nodes, edges, removed = merge_for_nodes(nodes, edges)

    # --------------------------
    # decide remove nodes
    # --------------------------
    removed = set()
    kept = set()

    for nid, attrs in nodes.items():
        label = attrs.get("label", "").strip('"')

        if should_remove_node(label):
            removed.add(nid)
        else:
            kept.add(nid)

    # --------------------------
    # reconnect edges
    # --------------------------
    new_edges = set(edges)

    for r in removed:
        for p in pred[r]:
            for s in succ[r]:
                if p != s:
                    new_edges.add((p, s))

        # remove old connected edges
        new_edges = {
            (u, v)
            for (u, v) in new_edges
            if u != r and v != r
        }


    # --------------------------
    # rebuild dot
    # --------------------------
    output = []

    for line in other_lines:
        if not re.match(r'^\s*\d+\s*\[', line) and "->" not in line:
            output.append(line)

    # rebuild nodes
    for nid, attrs in nodes.items():
        if nid in removed:
            continue

        label = attrs.get("label")
        shape = attrs.get("shape")

        kept_attrs = []

        if shape:
            kept_attrs.append(f"shape={shape}")
        if label:
            kept_attrs.append(f"label={label}")

        output.append(f'{nid} [{", ".join(kept_attrs)}];')

    # rebuild edges
    for src, dst in sorted(new_edges, key=lambda x: (int(x[0]), int(x[1]))):
        if src in kept and dst in kept:
            output.append(f'{src} -> {dst};')

    return "\n".join(output)


def gen_py_cfg(code):

    # 从代码字符串构建 CFG
    cfg = CFGBuilder().build_from_src(
        "example_cfg",
        code
    )

    # 导出 DOT 文本（不渲染图片）
    dot_content = cfg._build_visual()
    dot_content=normalize_py_cfg(dot_content)
    return dot_content
def gen_java_cfg(java_code):
    import subprocess
    result = subprocess.run(
        ["java", "-jar", "spoon.jar"],
        input=java_code,
        text=True,
        capture_output=True
    )

    if result.returncode == 0:
        cfg = result.stdout
        cfg = normalize_java_cfg(cfg)
        return cfg
    else:
        return None

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
def extract_code(text):
    pattern = r"```(?:\w+)?\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""

def gen():
    rows = []
    dataset = load_dataset("greengerong/leetcode",split="train")
    for index in range(0, len(dataset)):
        java_code = dataset[index]["java"]
        python_code = dataset[index]["python"]
        java_code = extract_code(java_code)
        python_code = extract_code(python_code)
        java_cfg = gen_java_cfg(java_code)
        python_cfg = gen_py_cfg(python_code)

        if java_cfg is not None and python_cfg is not None:
            print(python_cfg)
            print(java_cfg)
            print(java_code)




if __name__ == '__main__':
    gen()