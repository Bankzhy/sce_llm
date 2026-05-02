import re
import pandas as pd
from difflib import SequenceMatcher


########################################
# 配置
########################################

NODE_SIM_THRESHOLD = 0.90


########################################
# DeepSeek 输出清洗
########################################

def restore_tokenizer_artifacts(text):
    """
    处理 DeepSeek / Code LLM 中常见的 tokenizer artifact:
    Ċ -> 换行
    Ġ -> 空格
    """
    if not isinstance(text, str):
        return ""

    replacements = {
        "Ċ": "\n",
        "ĉ": "\n",
        "Ġ": " ",
        "▁": " ",
        "</s>": "",
        "<s>": "",
        "<pad>": "",
        "<|endoftext|>": "",
        "</code>": "",
        "<code>": "",
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def extract_digraph(cfg_text):
    """
    从模型输出中只提取第一个 digraph {...}。
    可处理：
    diGraphfirst_palindrome{...}Addendum...
    digraph xxx {...}
    """
    if not isinstance(cfg_text, str):
        return ""

    text = restore_tokenizer_artifacts(cfg_text)

    # 删除 markdown fence
    text = re.sub(r"```[a-zA-Z]*", "", text)
    text = text.replace("```", "")

    # 找 digraph / diGraph / DIGRAPH
    match = re.search(r"digraph", text, flags=re.IGNORECASE)

    if not match:
        return text.strip()

    start = match.start()

    brace_start = text.find("{", start)

    if brace_start == -1:
        return text[start:].strip()

    depth = 0
    end = None

    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1

            if depth == 0:
                end = i + 1
                break

    if end is not None:
        return text[start:end].strip()

    return text[start:].strip()


def normalize_cfg_text(cfg_text):
    """
    统一 CFG 文本格式。
    """
    text = extract_digraph(cfg_text)

    # diGraphfirst_palindrome{ -> digraph first_palindrome {
    text = re.sub(
        r"(?i)digraph\s*([A-Za-z_][A-Za-z0-9_]*)?\s*\{",
        lambda m: f"digraph {m.group(1) or 'G'} {{",
        text
    )

    return text.strip()


########################################
# 文本相似度
########################################

def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = restore_tokenizer_artifacts(text)

    text = text.strip()
    text = text.strip('"').strip("'")

    # 压缩空白
    text = re.sub(r"\s+", " ", text)

    # 去掉符号周围多余空格
    text = re.sub(
        r"\s*([=+\-*/%<>!&|:,.;()\[\]{}])\s*",
        r"\1",
        text
    )

    # DeepSeek 常见粘连修复
    # forwordinwords: -> for word in words:
    text = re.sub(
        r"\bfor([A-Za-z_][A-Za-z0-9_]*)in([A-Za-z_][A-Za-z0-9_]*)",
        r"for \1 in \2",
        text
    )

    # ifword==word[::-1]: -> if word==word[::-1]:
    text = re.sub(
        r"\bif([A-Za-z_][A-Za-z0-9_]*)",
        r"if \1",
        text
    )

    # returnword -> return word
    text = re.sub(
        r"\breturn([A-Za-z_][A-Za-z0-9_]*)",
        r"return \1",
        text
    )

    # return"""" -> return""
    text = text.replace('return""""', 'return""')

    # 再压缩一次空白
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def text_similarity(a, b):
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


########################################
# 解析 CFG 节点
########################################

def parse_nodes(cfg_text):
    """
    兼容：
    1[label="xxx", shape=rectangle]
    1[label=xxx,shape=rectangle]
    1 [label = xxx, shape = rectangle]
    DeepSeek 无换行输出
    """

    if not isinstance(cfg_text, str):
        return {}

    cfg_text = normalize_cfg_text(cfg_text)

    nodes = {}

    # 匹配所有 node attribute 块：
    # 1[label=xxx,shape=xxx]
    # 1 [label="xxx", shape=xxx]
    node_pattern = r"(\d+)\s*\[(.*?)\]"

    matches = re.findall(
        node_pattern,
        cfg_text,
        flags=re.DOTALL
    )

    for node_id, attr_text in matches:
        node_id = node_id.strip()

        label = ""

        # 优先匹配 label=... 到 ,shape= 或 ] 前
        label_match = re.search(
            r'label\s*=\s*(".*?"|\'.*?\'|.*?)(?=\s*,\s*shape\s*=|\s*\])',
            attr_text,
            flags=re.DOTALL | re.IGNORECASE
        )

        if label_match:
            label = label_match.group(1).strip()

            if (
                len(label) >= 2
                and (
                    label[0] == label[-1] == '"'
                    or label[0] == label[-1] == "'"
                )
            ):
                label = label[1:-1]

        nodes[node_id] = normalize_text(label)

    return nodes


########################################
# 解析 CFG 边
########################################

def parse_edges(cfg_text):
    """
    解析:
    1->2
    1 -> 2
    """

    if not isinstance(cfg_text, str):
        return set()

    cfg_text = normalize_cfg_text(cfg_text)

    pattern = r"(\d+)\s*->\s*(\d+)"

    matches = re.findall(pattern, cfg_text)

    return set(
        (src.strip(), dst.strip())
        for src, dst in matches
    )


########################################
# 节点匹配：模糊匹配
########################################

def match_nodes(pred_nodes, gt_nodes):
    """
    基于 label 相似度匹配节点。
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
        "matched_pairs": matched_pairs,
        "pred_nodes": pred_nodes,
        "gt_nodes": gt_nodes
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
# 安全平均值
########################################

def safe_avg(values):
    return sum(values) / len(values) if len(values) > 0 else 0


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

    edge_precisions = []
    edge_recalls = []
    edge_f1s = []

    exact_match_count = 0
    zero_match_count = 0
    total_count = 0

    overall_valid_precisions = []
    overall_valid_recalls = []
    overall_valid_f1s = []

    ########################################
    # is_error=False
    ########################################
    normal_precisions = []
    normal_recalls = []
    normal_f1s = []

    normal_edge_precisions = []
    normal_edge_recalls = []
    normal_edge_f1s = []

    normal_exact_count = 0
    normal_zero_count = 0
    normal_count = 0

    normal_valid_precisions = []
    normal_valid_recalls = []
    normal_valid_f1s = []

    ########################################
    # is_error=True
    ########################################
    error_precisions = []
    error_recalls = []
    error_f1s = []

    error_edge_precisions = []
    error_edge_recalls = []
    error_edge_f1s = []

    error_exact_count = 0
    error_zero_count = 0
    error_count = 0

    error_valid_precisions = []
    error_valid_recalls = []
    error_valid_f1s = []

    ########################################
    # Iterate samples
    ########################################
    for _, row in df.iterrows():
        gt_cfg = str(row["cfg"]).strip()
        pred_cfg = str(row["predict"]).strip()

        # 核心：先清洗 DeepSeek 输出
        gt_cfg = normalize_cfg_text(gt_cfg)
        pred_cfg = normalize_cfg_text(pred_cfg)

        is_error = row["is_error"]

        node_metrics = compute_node_metrics(
            pred_cfg,
            gt_cfg
        )

        edge_metrics = compute_edge_metrics(
            pred_cfg,
            gt_cfg,
            node_metrics["matched_pairs"]
        )

        node_precision = node_metrics["precision"]
        node_recall = node_metrics["recall"]
        node_f1 = node_metrics["f1"]

        edge_precision = edge_metrics["precision"]
        edge_recall = edge_metrics["recall"]
        edge_f1 = edge_metrics["f1"]

        ########################################
        # Overall
        ########################################
        node_precisions.append(node_precision)
        node_recalls.append(node_recall)
        node_f1s.append(node_f1)

        edge_precisions.append(edge_precision)
        edge_recalls.append(edge_recall)
        edge_f1s.append(edge_f1)

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
        if is_error == False or str(is_error).lower() == "false":
            normal_precisions.append(
                node_precision
            )
            normal_recalls.append(
                node_recall
            )
            normal_f1s.append(
                node_f1
            )

            normal_edge_precisions.append(edge_precision)
            normal_edge_recalls.append(edge_recall)
            normal_edge_f1s.append(edge_f1)

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

            error_edge_precisions.append(edge_precision)
            error_edge_recalls.append(edge_recall)
            error_edge_f1s.append(edge_f1)

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
          safe_avg(node_precisions))

    print("Node Recall:",
          safe_avg(node_recalls))

    print("Node F1:",
          safe_avg(node_f1s))

    print("Edge Precision:",
          safe_avg(edge_precisions))

    print("Edge Recall:",
          safe_avg(edge_recalls))

    print("Edge F1:",
          safe_avg(edge_f1s))

    print("Exact Match Ratio:",
          exact_match_count / total_count if total_count > 0 else 0)

    print("Zero Match Ratio:",
          zero_match_count / total_count if total_count > 0 else 0)

    print()

    print("Remove Zero-Match Samples:")

    print("Precision:",
          safe_avg(overall_valid_precisions))

    print("Recall:",
          safe_avg(overall_valid_recalls))

    print("F1:",
          safe_avg(overall_valid_f1s))

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
              safe_avg(normal_precisions))

        print("Node Recall:",
              safe_avg(normal_recalls))

        print("Node F1:",
              safe_avg(normal_f1s))

        print("Edge Precision:",
              safe_avg(normal_edge_precisions))

        print("Edge Recall:",
              safe_avg(normal_edge_recalls))

        print("Edge F1:",
              safe_avg(normal_edge_f1s))

        print("Exact Match Ratio:",
              normal_exact_count / normal_count)

        print("Zero Match Ratio:",
              normal_zero_count / normal_count)

        print()

        print("Remove Zero-Match Samples:")

        print("Precision:",
              safe_avg(normal_valid_precisions))

        print("Recall:",
              safe_avg(normal_valid_recalls))

        print("F1:",
              safe_avg(normal_valid_f1s))

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
              safe_avg(error_precisions))

        print("Node Recall:",
              safe_avg(error_recalls))

        print("Node F1:",
              safe_avg(error_f1s))

        print("Edge Precision:",
              safe_avg(error_edge_precisions))

        print("Edge Recall:",
              safe_avg(error_edge_recalls))

        print("Edge F1:",
              safe_avg(error_edge_f1s))

        print("Exact Match Ratio:",
              error_exact_count / error_count)

        print("Zero Match Ratio:",
              error_zero_count / error_count)

        print()

        print("Remove Zero-Match Samples:")

        print("Precision:",
              safe_avg(error_valid_precisions))

        print("Recall:",
              safe_avg(error_valid_recalls))

        print("F1:",
              safe_avg(error_valid_f1s))


########################################
# 运行
########################################

if __name__ == "__main__":
    evaluate("predict_deepseek_coder.csv")