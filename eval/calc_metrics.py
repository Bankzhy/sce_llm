import re
import pandas as pd
from difflib import SequenceMatcher


NODE_SIM_THRESHOLD = 0.95


def normalize_text(text):
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def text_similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def parse_nodes(cfg_text):
    if not isinstance(cfg_text, str):
        return {}

    pattern = r'(\d+)\s*\[\s*label=(.*?)\s*,\s*shape='
    matches = re.findall(pattern, cfg_text, re.DOTALL)

    nodes = {}
    for node_id, label in matches:
        label = label.strip()
        if label.startswith('"') and label.endswith('"'):
            label = label[1:-1]
        nodes[node_id] = normalize_text(label)

    return nodes


def parse_edges(cfg_text):
    if not isinstance(cfg_text, str):
        return set()

    pattern = r'(\d+)\s*->\s*(\d+)'
    matches = re.findall(pattern, cfg_text)
    return set(matches)


def match_nodes(pred_nodes, gt_nodes):
    matched_pairs = []
    used_gt = set()

    for pred_id, pred_label in pred_nodes.items():
        best_match = None
        best_score = 0

        for gt_id, gt_label in gt_nodes.items():
            if gt_id in used_gt:
                continue

            score = text_similarity(pred_label, gt_label)

            if score > best_score:
                best_score = score
                best_match = gt_id

        if best_match is not None and best_score >= NODE_SIM_THRESHOLD:
            matched_pairs.append((pred_id, best_match))
            used_gt.add(best_match)

    return matched_pairs


def compute_node_metrics(pred_cfg, gt_cfg):
    pred_nodes = parse_nodes(pred_cfg)
    gt_nodes = parse_nodes(gt_cfg)

    matched_pairs = match_nodes(pred_nodes, gt_nodes)

    matched_count = len(matched_pairs)
    pred_total = len(pred_nodes)
    gt_total = len(gt_nodes)

    precision = matched_count / pred_total if pred_total > 0 else 0
    recall = matched_count / gt_total if gt_total > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_pairs": matched_pairs
    }


def compute_edge_metrics(pred_cfg, gt_cfg, matched_pairs):
    pred_edges = parse_edges(pred_cfg)
    gt_edges = parse_edges(gt_cfg)

    pred_to_gt = {
        pred_id: gt_id
        for pred_id, gt_id in matched_pairs
    }

    mapped_pred_edges = set()

    for src, dst in pred_edges:
        if src in pred_to_gt and dst in pred_to_gt:
            mapped_pred_edges.add(
                (
                    pred_to_gt[src],
                    pred_to_gt[dst]
                )
            )

    tp = len(mapped_pred_edges & gt_edges)

    precision = (
        tp / len(mapped_pred_edges)
        if len(mapped_pred_edges) > 0 else 0
    )

    recall = (
        tp / len(gt_edges)
        if len(gt_edges) > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def mean(values):
    return sum(values) / len(values) if len(values) > 0 else 0


def print_metrics(title, data):
    print("=" * 60)
    print(title)
    print("=" * 60)

    print("Samples:", data["count"])

    print("Node Precision:", mean(data["node_p"]))
    print("Node Recall:", mean(data["node_r"]))
    print("Node F1:", mean(data["node_f1"]))

    print("Edge Precision:", mean(data["edge_p"]))
    print("Edge Recall:", mean(data["edge_r"]))
    print("Edge F1:", mean(data["edge_f1"]))

    print("Node Exact Match Ratio:",
          data["node_exact"] / data["count"]
          if data["count"] > 0 else 0)

    print("Node Zero Match Ratio:",
          data["node_zero"] / data["count"]
          if data["count"] > 0 else 0)

    print("Edge Exact Match Ratio:",
          data["edge_exact"] / data["count"]
          if data["count"] > 0 else 0)

    print("Edge Zero Match Ratio:",
          data["edge_zero"] / data["count"]
          if data["count"] > 0 else 0)

    print()
    print("Remove Node Zero-Match Samples:")

    print("Node Precision:", mean(data["valid_node_p"]))
    print("Node Recall:", mean(data["valid_node_r"]))
    print("Node F1:", mean(data["valid_node_f1"]))

    print("Edge Precision:", mean(data["valid_edge_p"]))
    print("Edge Recall:", mean(data["valid_edge_r"]))
    print("Edge F1:", mean(data["valid_edge_f1"]))

    print()


def empty_data():
    return {
        "count": 0,

        "node_p": [],
        "node_r": [],
        "node_f1": [],

        "edge_p": [],
        "edge_r": [],
        "edge_f1": [],

        "valid_node_p": [],
        "valid_node_r": [],
        "valid_node_f1": [],

        "valid_edge_p": [],
        "valid_edge_r": [],
        "valid_edge_f1": [],

        "node_exact": 0,
        "node_zero": 0,

        "edge_exact": 0,
        "edge_zero": 0,
    }


def update_data(data, node_metrics, edge_metrics):
    node_p = node_metrics["precision"]
    node_r = node_metrics["recall"]
    node_f1 = node_metrics["f1"]

    edge_p = edge_metrics["precision"]
    edge_r = edge_metrics["recall"]
    edge_f1 = edge_metrics["f1"]

    data["count"] += 1

    data["node_p"].append(node_p)
    data["node_r"].append(node_r)
    data["node_f1"].append(node_f1)

    data["edge_p"].append(edge_p)
    data["edge_r"].append(edge_r)
    data["edge_f1"].append(edge_f1)

    if node_f1 == 1.0:
        data["node_exact"] += 1

    if node_f1 == 0.0:
        data["node_zero"] += 1
    else:
        data["valid_node_p"].append(node_p)
        data["valid_node_r"].append(node_r)
        data["valid_node_f1"].append(node_f1)

        data["valid_edge_p"].append(edge_p)
        data["valid_edge_r"].append(edge_r)
        data["valid_edge_f1"].append(edge_f1)

    if edge_f1 == 1.0:
        data["edge_exact"] += 1

    if edge_f1 == 0.0:
        data["edge_zero"] += 1


def to_bool(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() == "true"

    return bool(value)


def clean_prediction(text):
    text = str(text).strip()
    text = text.replace("Ċ", "\n")
    text = text.replace("Ġ", " ")
    return text


def evaluate(csv_path):
    df = pd.read_csv(csv_path)

    overall_data = empty_data()
    normal_data = empty_data()
    error_data = empty_data()

    for _, row in df.iterrows():
        gt_cfg = str(row["cfg"]).strip()
        pred_cfg = clean_prediction(row["predict"])
        is_error = to_bool(row["is_error"])

        node_metrics = compute_node_metrics(pred_cfg, gt_cfg)

        edge_metrics = compute_edge_metrics(
            pred_cfg,
            gt_cfg,
            node_metrics["matched_pairs"]
        )

        update_data(overall_data, node_metrics, edge_metrics)

        if is_error is False:
            update_data(normal_data, node_metrics, edge_metrics)
        else:
            update_data(error_data, node_metrics, edge_metrics)

    print_metrics("Overall", overall_data)
    print_metrics("is_error=False", normal_data)
    print_metrics("is_error=True", error_data)


if __name__ == "__main__":
    evaluate("predict-phi-4.csv")