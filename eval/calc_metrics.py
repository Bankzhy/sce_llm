import re
import ast
import pandas as pd
from difflib import SequenceMatcher


########################################
# 配置
########################################

NODE_SIM_THRESHOLD = 0.75


########################################
# 文本相似度
########################################

def text_similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


def normalize_text(text):
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    return text


########################################
# 解析CFG节点
########################################

def parse_nodes(cfg_text):
    """
    解析:
    1[label="xxx", shape="rectangle"]
    """

    if not isinstance(cfg_text, str):
        return {}

    pattern = r'(\d+)\s*\[label="(.*?)"'

    matches = re.findall(
        pattern,
        cfg_text,
        re.DOTALL
    )

    nodes = {}

    for node_id, label in matches:
        nodes[node_id] = normalize_text(label)

    return nodes


########################################
# 解析CFG边
########################################

def parse_edges(cfg_text):
    """
    解析:
    1->2
    """

    if not isinstance(cfg_text, str):
        return set()

    pattern = r'(\d+)\s*->\s*(\d+)'

    matches = re.findall(pattern, cfg_text)

    return set(matches)


########################################
# 节点匹配（模糊匹配）
########################################

def match_nodes(pred_nodes, gt_nodes):
    """
    基于label相似度匹配节点
    返回:
    matched_pairs = [(pred_id, gt_id)]
    """

    matched_pairs = []
    used_gt = set()

    for pred_id, pred_label in pred_nodes.items():
        best_match = None
        best_score = 0

        for gt_id, gt_label in gt_nodes.items():
            if gt_id in used_gt:
                continue

            score = text_similarity(
                pred_label,
                gt_label
            )

            if score > best_score:
                best_score = score
                best_match = gt_id

        if (
            best_match is not None
            and best_score >= NODE_SIM_THRESHOLD
        ):
            matched_pairs.append(
                (pred_id, best_match)
            )
            used_gt.add(best_match)

    return matched_pairs


########################################
# 节点指标
########################################

def compute_node_metrics(pred_cfg, gt_cfg):
    pred_nodes = parse_nodes(pred_cfg)
    gt_nodes = parse_nodes(gt_cfg)

    matched_pairs = match_nodes(
        pred_nodes,
        gt_nodes
    )

    matched_pred = len(matched_pairs)

    pred_total = len(pred_nodes)
    gt_total = len(gt_nodes)

    coverage = (
        matched_pred / gt_total
        if gt_total > 0 else 0
    )

    precision = (
        matched_pred / pred_total
        if pred_total > 0 else 0
    )

    recall = (
        matched_pred / gt_total
        if gt_total > 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "coverage": coverage,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched_pairs": matched_pairs
    }


########################################
# 边指标
########################################

def compute_edge_metrics(
    pred_cfg,
    gt_cfg,
    matched_pairs
):
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

    tp = len(
        mapped_pred_edges & gt_edges
    )

    precision = (
        tp / len(mapped_pred_edges)
        if len(mapped_pred_edges) > 0 else 0
    )

    recall = (
        tp / len(gt_edges)
        if len(gt_edges) > 0 else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


########################################
# 枚举路径
########################################

def build_graph(edges):
    graph = {}

    for src, dst in edges:
        graph.setdefault(src, []).append(dst)

    return graph


def dfs_paths(
    graph,
    node,
    path,
    paths,
    visited,
    max_depth=30
):
    if len(path) > max_depth:
        return

    if node not in graph:
        paths.add(tuple(path))
        return

    for nxt in graph[node]:
        if (node, nxt) in visited:
            continue

        dfs_paths(
            graph,
            nxt,
            path + [nxt],
            paths,
            visited | {(node, nxt)},
            max_depth
        )


def extract_paths(cfg_text):
    edges = parse_edges(cfg_text)

    if not edges:
        return set()

    graph = build_graph(edges)

    all_nodes = set()
    child_nodes = set()

    for src, dst in edges:
        all_nodes.add(src)
        all_nodes.add(dst)
        child_nodes.add(dst)

    roots = all_nodes - child_nodes

    paths = set()

    for root in roots:
        dfs_paths(
            graph,
            root,
            [root],
            paths,
            set()
        )

    return paths


########################################
# Path Coverage Similarity
########################################

def compute_path_similarity(
    pred_cfg,
    gt_cfg
):
    pred_paths = extract_paths(pred_cfg)
    gt_paths = extract_paths(gt_cfg)

    union = pred_paths | gt_paths

    if len(union) == 0:
        return 0

    intersection = pred_paths & gt_paths

    return len(intersection) / len(union)


########################################
# 总评估
########################################

def evaluate(csv_path):
    df = pd.read_csv(csv_path)

    node_coverages = []
    node_precisions = []
    node_recalls = []
    node_f1s = []

    edge_precisions = []
    edge_recalls = []
    edge_f1s = []

    path_similarities = []

    valid_count = 0

    for _, row in df.iterrows():
        if row["is_error"]:
            continue

        gt_cfg = row["cfg"]
        pred_cfg = row["predict"]

        node_metrics = compute_node_metrics(
            pred_cfg,
            gt_cfg
        )

        edge_metrics = compute_edge_metrics(
            pred_cfg,
            gt_cfg,
            node_metrics["matched_pairs"]
        )

        path_similarity = compute_path_similarity(
            pred_cfg,
            gt_cfg
        )

        node_coverages.append(
            node_metrics["coverage"]
        )

        node_precisions.append(
            node_metrics["precision"]
        )

        node_recalls.append(
            node_metrics["recall"]
        )

        node_f1s.append(
            node_metrics["f1"]
        )

        edge_precisions.append(
            edge_metrics["precision"]
        )

        edge_recalls.append(
            edge_metrics["recall"]
        )

        edge_f1s.append(
            edge_metrics["f1"]
        )

        path_similarities.append(
            path_similarity
        )

        valid_count += 1

    print("=" * 50)
    print("Samples:", valid_count)
    print("=" * 50)

    print("Node Coverage:",
          sum(node_coverages) / valid_count)

    print("Node Precision:",
          sum(node_precisions) / valid_count)

    print("Node Recall:",
          sum(node_recalls) / valid_count)

    print("Node F1:",
          sum(node_f1s) / valid_count)

    print()

    print("Edge Precision:",
          sum(edge_precisions) / valid_count)

    print("Edge Recall:",
          sum(edge_recalls) / valid_count)

    print("Edge F1:",
          sum(edge_f1s) / valid_count)

    print()

    print("Path Coverage Similarity:",
          sum(path_similarities) / valid_count)


########################################
# 运行
########################################

if __name__ == "__main__":
    evaluate("predict.csv")