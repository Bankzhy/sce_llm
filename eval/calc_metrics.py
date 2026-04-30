import re
import ast
import pandas as pd
from difflib import SequenceMatcher


########################################
# 配置
########################################

NODE_SIM_THRESHOLD = 0.95


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
    兼容：
    1[label="xxx", shape=rectangle]
    1[label=xxx, shape=rectangle]
    """

    if not isinstance(cfg_text, str):
        return {}

    pattern = r'(\d+)\s*\[\s*label=(.*?)\s*,\s*shape='

    matches = re.findall(
        pattern,
        cfg_text,
        re.DOTALL
    )

    nodes = {}

    for node_id, label in matches:
        label = label.strip()

        if label.startswith('"') and label.endswith('"'):
            label = label[1:-1]

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

    ########################################
    # Overall
    ########################################
    node_precisions = []
    node_recalls = []
    node_f1s = []

    exact_match_count = 0
    zero_match_count = 0
    total_count = 0

    # overall remove zero-match
    overall_valid_precisions = []
    overall_valid_recalls = []
    overall_valid_f1s = []

    ########################################
    # is_error=False
    ########################################
    normal_precisions = []
    normal_recalls = []
    normal_f1s = []

    normal_exact_count = 0
    normal_zero_count = 0
    normal_count = 0

    # normal remove zero-match
    normal_valid_precisions = []
    normal_valid_recalls = []
    normal_valid_f1s = []

    ########################################
    # is_error=True
    ########################################
    error_precisions = []
    error_recalls = []
    error_f1s = []

    error_exact_count = 0
    error_zero_count = 0
    error_count = 0

    # error remove zero-match
    error_valid_precisions = []
    error_valid_recalls = []
    error_valid_f1s = []

    ########################################
    # Iterate samples
    ########################################
    for _, row in df.iterrows():
        gt_cfg = str(row["cfg"]).strip()
        pred_cfg = str(row["predict"]).strip()

        is_error = row["is_error"]

        node_metrics = compute_node_metrics(
            pred_cfg,
            gt_cfg
        )

        node_precision = node_metrics["precision"]
        node_recall = node_metrics["recall"]
        node_f1 = node_metrics["f1"]

        ########################################
        # Overall
        ########################################
        node_precisions.append(node_precision)
        node_recalls.append(node_recall)
        node_f1s.append(node_f1)

        if node_f1 == 1.0:
            exact_match_count += 1

        if node_f1 == 0.0:
            zero_match_count += 1
        else:
            overall_valid_precisions.append(
                node_precision
            )
            overall_valid_recalls.append(
                node_recall
            )
            overall_valid_f1s.append(
                node_f1
            )

        total_count += 1

        ########################################
        # is_error=False
        ########################################
        if is_error == False:
            normal_precisions.append(
                node_precision
            )
            normal_recalls.append(
                node_recall
            )
            normal_f1s.append(
                node_f1
            )

            if node_f1 == 1.0:
                normal_exact_count += 1

            if node_f1 == 0.0:
                normal_zero_count += 1
            else:
                normal_valid_precisions.append(
                    node_precision
                )
                normal_valid_recalls.append(
                    node_recall
                )
                normal_valid_f1s.append(
                    node_f1
                )

            normal_count += 1

        ########################################
        # is_error=True
        ########################################
        else:
            error_precisions.append(
                node_precision
            )
            error_recalls.append(
                node_recall
            )
            error_f1s.append(
                node_f1
            )

            if node_f1 == 1.0:
                error_exact_count += 1

            if node_f1 == 0.0:
                error_zero_count += 1
            else:
                error_valid_precisions.append(
                    node_precision
                )
                error_valid_recalls.append(
                    node_recall
                )
                error_valid_f1s.append(
                    node_f1
                )

            error_count += 1

    ########################################
    # Overall Output
    ########################################
    print("=" * 60)
    print("Overall")
    print("=" * 60)

    print("Samples:",
          total_count)

    print("Node Precision:",
          sum(node_precisions) / total_count)

    print("Node Recall:",
          sum(node_recalls) / total_count)

    print("Node F1:",
          sum(node_f1s) / total_count)

    print("Exact Match Ratio:",
          exact_match_count / total_count)

    print("Zero Match Ratio:",
          zero_match_count / total_count)

    print()

    print("Remove Zero-Match Samples:")

    if len(overall_valid_f1s) > 0:
        print("Precision:",
              sum(overall_valid_precisions)
              / len(overall_valid_precisions))

        print("Recall:",
              sum(overall_valid_recalls)
              / len(overall_valid_recalls))

        print("F1:",
              sum(overall_valid_f1s)
              / len(overall_valid_f1s))

    print()

    ########################################
    # is_error=False Output
    ########################################
    print("=" * 60)
    print("is_error=False")
    print("=" * 60)

    if normal_count > 0:
        print("Samples:",
              normal_count)

        print("Node Precision:",
              sum(normal_precisions)
              / normal_count)

        print("Node Recall:",
              sum(normal_recalls)
              / normal_count)

        print("Node F1:",
              sum(normal_f1s)
              / normal_count)

        print("Exact Match Ratio:",
              normal_exact_count
              / normal_count)

        print("Zero Match Ratio:",
              normal_zero_count
              / normal_count)

        print()

        print("Remove Zero-Match Samples:")

        if len(normal_valid_f1s) > 0:
            print("Precision:",
                  sum(normal_valid_precisions)
                  / len(normal_valid_precisions))

            print("Recall:",
                  sum(normal_valid_recalls)
                  / len(normal_valid_recalls))

            print("F1:",
                  sum(normal_valid_f1s)
                  / len(normal_valid_f1s))

    print()

    ########################################
    # is_error=True Output
    ########################################
    print("=" * 60)
    print("is_error=True")
    print("=" * 60)

    if error_count > 0:
        print("Samples:",
              error_count)

        print("Node Precision:",
              sum(error_precisions)
              / error_count)

        print("Node Recall:",
              sum(error_recalls)
              / error_count)

        print("Node F1:",
              sum(error_f1s)
              / error_count)

        print("Exact Match Ratio:",
              error_exact_count
              / error_count)

        print("Zero Match Ratio:",
              error_zero_count
              / error_count)

        print()

        print("Remove Zero-Match Samples:")

        if len(error_valid_f1s) > 0:
            print("Precision:",
                  sum(error_valid_precisions)
                  / len(error_valid_precisions))

            print("Recall:",
                  sum(error_valid_recalls)
                  / len(error_valid_recalls))

            print("F1:",
                  sum(error_valid_f1s)
                  / len(error_valid_f1s))


########################################
# 运行
########################################

if __name__ == "__main__":
    evaluate("predict_qwen_coder.csv")