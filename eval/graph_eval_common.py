import csv
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path


csv.field_size_limit(200_000_000)

AST_NODE_SIM_THRESHOLD = 0.80
CFG_PDG_NODE_SIM_THRESHOLD = 0.90
KEY_METRIC_FIELDS = [
    "split",
    "graph_type",
    "samples",
    "Node-F1",
    "Node-F1-RZ",
    "Edge-F1",
    "Edge-F1-RZ",
    "Node-EMR",
    "Node-ZMR",
]


def restore_graph_newlines(text: str) -> str:
    return str(text).replace("\\n", "\n").strip()


def clean_prediction(text: str) -> str:
    text = str(text).strip()
    text = text.replace("Ċ", "\n")
    text = text.replace("Ġ", " ")
    return text


def strip_byte_offsets(graph_text: str) -> str:
    return re.sub(r";bytes:\d+-\d+", "", restore_graph_newlines(graph_text))


def normalize_graph_for_metrics(graph_text: str, graph_type: str) -> str:
    if graph_type == "AST":
        return strip_byte_offsets(graph_text)
    return restore_graph_newlines(graph_text)


def extract_graph(text: str, graph_type: str) -> str:
    if not isinstance(text, str):
        return ""
    text = restore_graph_newlines(text)
    pattern = re.compile(
        rf"digraph\s+{graph_type}_[A-Za-z0-9_]*\s*\{{.*?\n\s*\}}",
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(0).strip() if match else ""


def normalize_dot_value(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip())


def ast_identifier_type(node_type: str) -> bool:
    return node_type in {"type_identifier", "var_identifier", "method_identifier"}


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(
        None,
        normalize_dot_value(left),
        normalize_dot_value(right),
    ).ratio()


def parse_offset(offset: str) -> dict[str, tuple[int, int] | None]:
    line_match = re.search(r"lines:(\d+)-(\d+)", offset or "")
    byte_match = re.search(r"bytes:(\d+)-(\d+)", offset or "")
    return {
        "lines": tuple(map(int, line_match.groups())) if line_match else None,
        "bytes": tuple(map(int, byte_match.groups())) if byte_match else None,
    }


def range_iou(left: tuple[int, int] | None, right: tuple[int, int] | None) -> float:
    if left is None or right is None:
        return 0.0

    left_start, left_end = left
    right_start, right_end = right
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    union = max(left_end, right_end) - min(left_start, right_start) + 1
    return intersection / union if union > 0 else 0.0


def parse_nodes(graph_text: str) -> dict[str, dict[str, str]]:
    if not isinstance(graph_text, str):
        return {}
    graph_text = restore_graph_newlines(graph_text)
    pattern = re.compile(
        r'^\s*(\d+)\s+\[type="([^"]+)",\s*offset="([^"]+)"(?:,\s*label="((?:\\.|[^"])*)")?\];',
        re.MULTILINE,
    )
    return {
        node_id: {
            "type": normalize_dot_value(node_type),
            "offset": normalize_dot_value(offset),
            "label": normalize_dot_value(label or ""),
        }
        for node_id, node_type, offset, label in pattern.findall(graph_text)
    }


def parse_edges(graph_text: str) -> set[tuple[str, str, str]]:
    if not isinstance(graph_text, str):
        return set()
    graph_text = restore_graph_newlines(graph_text)

    typed_pattern = re.compile(
        r'^\s*(\d+)\s*->\s*(\d+)\s+\[type="([^"]+)"\];',
        re.MULTILINE,
    )
    plain_pattern = re.compile(
        r"^\s*(\d+)\s*->\s*(\d+);",
        re.MULTILINE,
    )

    edges = {
        (source, target, normalize_dot_value(edge_type))
        for source, target, edge_type in typed_pattern.findall(graph_text)
    }
    typed_spans = [match.span() for match in typed_pattern.finditer(graph_text)]

    for match in plain_pattern.finditer(graph_text):
        if any(start <= match.start() < end for start, end in typed_spans):
            continue
        source, target = match.groups()
        edges.add((source, target, ""))
    return edges


def prf(tp: int, pred_total: int, gt_total: int) -> tuple[float, float, float]:
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def node_match_score(
    pred_node: dict[str, str],
    gt_node: dict[str, str],
    graph_type: str,
) -> float:
    if pred_node["type"] != gt_node["type"]:
        return 0.0

    pred_offset = parse_offset(pred_node["offset"])
    gt_offset = parse_offset(gt_node["offset"])
    line_score = range_iou(pred_offset["lines"], gt_offset["lines"])

    if graph_type == "AST" and ast_identifier_type(gt_node["type"]):
        label_score = text_similarity(pred_node.get("label", ""), gt_node.get("label", ""))
        return 0.5 * line_score + 0.5 * label_score

    return line_score


def match_nodes(
    pred_nodes: dict[str, dict[str, str]],
    gt_nodes: dict[str, dict[str, str]],
    graph_type: str,
) -> list[tuple[str, str, float]]:
    matched_pairs = []
    used_gt = set()

    for pred_id, pred_node in pred_nodes.items():
        best_gt_id = None
        best_score = 0.0

        for gt_id, gt_node in gt_nodes.items():
            if gt_id in used_gt:
                continue

            score = node_match_score(pred_node, gt_node, graph_type)
            if score > best_score:
                best_score = score
                best_gt_id = gt_id

        threshold = (
            AST_NODE_SIM_THRESHOLD
            if graph_type == "AST"
            else CFG_PDG_NODE_SIM_THRESHOLD
        )
        if best_gt_id is not None and best_score >= threshold:
            matched_pairs.append((pred_id, best_gt_id, best_score))
            used_gt.add(best_gt_id)

    return matched_pairs


def mapped_pred_edges(
    pred_edges: set[tuple[str, str, str]],
    matched_pairs: list[tuple[str, str, float]],
) -> set[tuple[str, str, str]]:
    pred_to_gt = {pred_id: gt_id for pred_id, gt_id, _ in matched_pairs}
    result = set()
    for source, target, edge_type in pred_edges:
        if source in pred_to_gt and target in pred_to_gt:
            result.add((pred_to_gt[source], pred_to_gt[target], edge_type))
    return result


def graph_metrics(pred_graph: str, gt_graph: str, graph_type: str) -> dict[str, float | int]:
    pred_nodes = parse_nodes(pred_graph)
    gt_nodes = parse_nodes(gt_graph)
    pred_edges = parse_edges(pred_graph)
    gt_edges = parse_edges(gt_graph)

    matched_pairs = match_nodes(pred_nodes, gt_nodes, graph_type)
    node_tp = len(matched_pairs)
    pred_node_total = len(pred_nodes)
    gt_node_total = len(gt_nodes)
    _, _, node_f1 = prf(node_tp, pred_node_total, gt_node_total)

    pred_edges_mapped = mapped_pred_edges(pred_edges, matched_pairs)
    edge_tp = len(pred_edges_mapped & gt_edges)
    pred_edge_total = len(pred_edges_mapped)
    gt_edge_total = len(gt_edges)
    _, _, edge_f1 = prf(edge_tp, pred_edge_total, gt_edge_total)

    return {
        "node_f1": node_f1,
        "edge_f1": edge_f1,
        "node_exact": int(node_f1 == 1.0 and pred_node_total == gt_node_total),
        "node_zero": int(node_tp == 0),
        "node_tp": node_tp,
        "pred_node_total": pred_node_total,
        "gt_node_total": gt_node_total,
        "edge_tp": edge_tp,
        "pred_edge_total": len(pred_edges),
        "mapped_pred_edge_total": pred_edge_total,
        "gt_edge_total": gt_edge_total,
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def new_summary() -> dict:
    return defaultdict(
        lambda: {
            "count": 0,
            "node_f1": [],
            "edge_f1": [],
            "node_f1_rz": [],
            "edge_f1_rz": [],
            "node_exact": 0,
            "node_zero": 0,
        }
    )


def update_summary(
    summary: dict,
    graph_type: str,
    metrics: dict[str, float | int],
) -> None:
    bucket = summary[graph_type]
    bucket["count"] += 1
    bucket["node_f1"].append(float(metrics["node_f1"]))
    bucket["edge_f1"].append(float(metrics["edge_f1"]))
    bucket["node_exact"] += int(metrics["node_exact"])
    bucket["node_zero"] += int(metrics["node_zero"])

    if not metrics["node_zero"]:
        bucket["node_f1_rz"].append(float(metrics["node_f1"]))
        bucket["edge_f1_rz"].append(float(metrics["edge_f1"]))


def summary_row(split_name: str, graph_type: str, bucket: dict) -> dict[str, float | int | str]:
    count = bucket["count"]
    return {
        "split": split_name,
        "graph_type": graph_type,
        "samples": count,
        "Node-F1": mean(bucket["node_f1"]),
        "Node-F1-RZ": mean(bucket["node_f1_rz"]),
        "Edge-F1": mean(bucket["edge_f1"]),
        "Edge-F1-RZ": mean(bucket["edge_f1_rz"]),
        "Node-EMR": bucket["node_exact"] / count if count else 0.0,
        "Node-ZMR": bucket["node_zero"] / count if count else 0.0,
    }


def summary_rows(
    split_name: str,
    summary: dict,
    graph_types: list[str],
) -> list[dict[str, float | int | str]]:
    return [
        summary_row(split_name, graph_type, summary[graph_type])
        for graph_type in graph_types
    ]


def save_metrics_summary(
    metrics_file: str,
    summaries: list[tuple[str, dict]],
    graph_types: list[str],
) -> None:
    rows = []
    for split_name, summary in summaries:
        rows.extend(summary_rows(split_name, summary, graph_types))

    Path(metrics_file).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEY_METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(title: str, summary: dict, graph_types: list[str]) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    for graph_type in graph_types:
        row = summary_row(title, graph_type, summary[graph_type])
        print(f"[{graph_type}] samples={row['samples']}")
        print(f"  Node-F1={row['Node-F1']:.4f}")
        print(f"  Node-F1-RZ={row['Node-F1-RZ']:.4f}")
        print(f"  Edge-F1={row['Edge-F1']:.4f}")
        print(f"  Edge-F1-RZ={row['Edge-F1-RZ']:.4f}")
        print(f"  Node-EMR={row['Node-EMR']:.4f}")
        print(f"  Node-ZMR={row['Node-ZMR']:.4f}")
        print()


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)
