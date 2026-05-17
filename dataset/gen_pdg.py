import argparse
import csv
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.cfg_generator import CFGEdge, CFGGenerator, CFGNode
from dataset.gen_ast import generate_error_code, sanitize_graph_name
from dataset.gen_cfg_new import (
    build_java_parse_code,
    build_python_parse_code,
    count_normal_rows,
    load_code_dataset,
    normalize_statement_offset,
    normalize_statement_type,
    row_to_code,
    should_keep_cfg_node,
    split_total,
)
from sitter.kast2core import KASTParse
from reflect.sr_statement import SRFORStatement, SRIFStatement, SRStatement, SRWhileStatement


class PDGEdge:
    def __init__(self, source, target, edge_type):
        self.source = source
        self.target = target
        self.type = edge_type


class GraphLite:
    def __init__(self, node_list, edge_list):
        self.node_list = list(node_list)
        self.edge_list = list(edge_list)
        self.node_map = {node.id: node for node in self.node_list}

    def get_node(self, node_id):
        return self.node_map.get(node_id)

    def get_edge(self, source, target):
        source_id = getattr(source, "id", source)
        target_id = getattr(target, "id", target)
        for edge in self.edge_list:
            if edge.source == source_id and edge.target == target_id:
                return edge
        return None

    def incoming_edges_of(self, node):
        node_id = getattr(node, "id", node)
        return [edge for edge in self.edge_list if edge.target == node_id]

    def get_edge_source(self, edge):
        return self.get_node(edge.source)

    def get_edge_target(self, edge):
        return self.get_node(edge.target)

    def get_dfs_node_list(self):
        return self.node_list


class PDGBuilder:
    def __init__(self, sr_method):
        self.sr_method = sr_method
        self.node_list = []
        self.flow_edge_list = []
        self.dd_edge_list = []
        self.cd_edge_list = []
        self.var_change_list = {}

    def create_graph(self) -> bool:
        cfg_gen = CFGGenerator(sr_method=self.sr_method)
        if not cfg_gen.create_graph():
            return False

        self.node_list = cfg_gen.node_list
        self.flow_edge_list = cfg_gen.flow_edge_list
        self._create_data_dependency()
        self._create_control_dependency(cfg_gen)
        return True

    def _create_data_dependency(self):
        self._get_var_change_from_statement()
        self._get_var_use_dependency()

    def _get_var_change_from_statement(self):
        for node in self.node_list:
            statement = node.sr_statement
            if isinstance(statement, (SRIFStatement, SRFORStatement, SRWhileStatement)):
                continue
            if "=" not in statement.word_list:
                continue

            for index, word in enumerate(statement.word_list):
                if index == 0 or index + 1 >= len(statement.word_list):
                    continue
                if word == "=":
                    var_change = statement.word_list[index - 1]
                    self.var_change_list.setdefault(var_change, []).append(node)

    def _get_var_use_dependency(self):
        for node in self.node_list:
            word_list = node.sr_statement.to_node_word_list()
            equal_index = word_list.index("=") if "=" in word_list else 0
            for word in word_list[equal_index + 1 :]:
                if word not in self.var_change_list:
                    continue
                var_change_node = self._get_nearest_dependency(
                    node, sorted(self.var_change_list[word], key=lambda item: item.index)
                )
                if var_change_node is not None and var_change_node.id != node.id:
                    self.dd_edge_list.append(PDGEdge(node.id, var_change_node.id, "data_dependence"))

    def _get_nearest_dependency(self, var_use_node, var_change_list):
        result = None
        for var_change_node in var_change_list:
            if var_use_node.index > var_change_node.index:
                result = var_change_node
        return result

    def _create_control_dependency(self, cfg_gen: CFGGenerator):
        reverse_graph = self._reverse_cfg(cfg_gen)
        start = None
        graph = nx.DiGraph()

        for node in reverse_graph.node_list:
            graph.add_node(node.id)
            if node.category == cfg_gen.END_ST_C:
                start = node.id

        for edge in reverse_graph.edge_list:
            graph.add_edge(edge.source, edge.target)

        if start is None or start not in graph:
            return

        try:
            dominators = nx.immediate_dominators(graph, start)
        except Exception:
            return

        for node in reverse_graph.node_list:
            node.i_dominator = dominators.get(node.id)

        dominator_tree = self._compute_dominator_tree(reverse_graph)
        cfg_graph = GraphLite(self.node_list, self.flow_edge_list)
        self._compute_control_dependency(cfg_graph, dominator_tree)

    def _reverse_cfg(self, cfg_gen: CFGGenerator):
        node_list = []
        edge_list = []

        for node in cfg_gen.node_list:
            if node.category == cfg_gen.init_index:
                statement = SRStatement(id=node.sr_statement.id, word_list=["E"], type="Fake")
                category = cfg_gen.END_ST_C
            elif node.category == cfg_gen.final_index:
                statement = SRStatement(id=node.sr_statement.id, word_list=["S"], type="Fake")
                category = cfg_gen.START_ST_C
            else:
                statement = node.sr_statement
                category = node.category

            node_list.append(CFGNode(id=node.id, index=node.index, category=category, sr_statement=statement))

        for edge in cfg_gen.flow_edge_list:
            edge_list.append(CFGEdge(id=edge.id, source=edge.target, target=edge.source, name=edge.name))

        return GraphLite(node_list, edge_list)

    def _compute_dominator_tree(self, graph: GraphLite):
        edges = []
        for node in graph.node_list:
            if node.i_dominator is None or node.i_dominator == node.id:
                continue
            edges.append(CFGEdge(id=random.randint(0, 100000000), source=node.i_dominator, target=node.id, name=""))
        return GraphLite(graph.node_list, edges)

    def _ancestors_of(self, tree: GraphLite, node):
        ancestors = []
        current = node
        while current is not None:
            incoming_edges = tree.incoming_edges_of(current)
            if not incoming_edges:
                break
            source = tree.get_edge_source(incoming_edges[0])
            if source is None:
                break
            ancestors.append(source)
            current = source
        return ancestors

    def _least_common_ancestor(self, tree: GraphLite, left, right):
        if tree.get_edge(left, right) is not None:
            return left
        if tree.get_edge(right, left) is not None:
            return right

        left_ancestors = [left] + self._ancestors_of(tree, left)
        right_ancestors = [right] + self._ancestors_of(tree, right)
        for ancestor in left_ancestors:
            if ancestor in right_ancestors:
                return ancestor
        return None

    def _edges_not_ancestral_in_tree(self, graph: GraphLite, tree: GraphLite):
        result = []
        for edge in graph.edge_list:
            source = graph.get_edge_source(edge)
            if source is None:
                continue
            ancestor_ids = {node.id for node in self._ancestors_of(tree, source)}
            if edge.target not in ancestor_ids:
                result.append(edge)
        return result

    def _compute_control_dependency(self, graph: GraphLite, dominator_tree: GraphLite):
        for edge in self._edges_not_ancestral_in_tree(graph, dominator_tree):
            source_node = graph.get_edge_source(edge)
            target_node = graph.get_edge_target(edge)
            if source_node is None or target_node is None:
                continue

            lca = self._least_common_ancestor(dominator_tree, source_node, target_node)
            if lca is None:
                continue

            self.cd_edge_list.append(PDGEdge(source_node.id, target_node.id, "control_dependence"))
            current = target_node
            while current is not None:
                incoming_edges = dominator_tree.incoming_edges_of(current)
                if not incoming_edges:
                    break
                parent = dominator_tree.get_edge_source(incoming_edges[0])
                if parent is None or parent.id == lca.id:
                    break
                self.cd_edge_list.append(PDGEdge(source_node.id, parent.id, "control_dependence"))
                current = parent


def select_target_class_and_method(sr_project):
    candidates = []
    for program in sr_project.program_list:
        for sr_class in program.class_list:
            for method in sr_class.method_list:
                candidates.append((sr_class, method))

    if not candidates:
        return None, None

    return max(candidates, key=lambda item: len(item[1].statement_list))


def build_pdg_graph(code: str, lang: str):
    parser = KASTParse("", lang)
    parser.setup()

    if lang == "java":
        parse_code = build_java_parse_code(code)
    elif lang == "python":
        parse_code = build_python_parse_code(code)
    else:
        raise ValueError(f"Unsupported language: {lang}")

    sr_project = parser.do_parse_content(parse_code)
    _, sr_method = select_target_class_and_method(sr_project)
    if sr_method is None:
        return None

    pdg_builder = PDGBuilder(sr_method)
    if not pdg_builder.create_graph():
        return None

    return sr_method, pdg_builder


def normalize_edges_through_ignored(edge_list, node_id_map):
    adjacency = {}
    for edge in edge_list:
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
    return normalized_edges


def pdg_to_dot(sr_method, pdg_builder: PDGBuilder) -> Optional[str]:
    method_name = sanitize_graph_name(sr_method.method_name)
    output = [f"digraph PDG_{method_name} {{"]
    method_start_line = getattr(sr_method, "start_line", 1)

    node_id_map = {}
    next_id = 1
    for node in pdg_builder.node_list:
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

    edges = set()

    for edge in pdg_builder.dd_edge_list + pdg_builder.cd_edge_list:
        if edge.source in node_id_map and edge.target in node_id_map:
            source = node_id_map[edge.source]
            target = node_id_map[edge.target]
            if source != target:
                edges.add((source, target, edge.type))

    for source, target, edge_type in sorted(edges, key=lambda item: (int(item[0]), int(item[1]), item[2])):
        output.append(f'    {source} -> {target} [type="{edge_type}"];')

    output.append("}")
    return "\n".join(output)


def count_pdg_nodes(pdg: str) -> int:
    return sum(1 for line in pdg.splitlines() if re.match(r"^\s*\d+\s*\[", line))


def gen_pdg(code: str, lang: str) -> Optional[str]:
    graph = build_pdg_graph(code, lang)
    if graph is None:
        return None
    sr_method, pdg_builder = graph
    return pdg_to_dot(sr_method, pdg_builder)


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

            pdg = gen_pdg(code, lang)
            if pdg is None:
                continue
            if count_pdg_nodes(pdg) < min_nodes:
                continue

            sample_id = f"{split_name}_{id_start + sample_count}"
            rows.append({"id": sample_id, "code": code, "PDG": pdg, "is_error": False, "language": lang})

            error_code = generate_error_code(code, lang)
            if error_code != code:
                rows.append(
                    {
                        "id": f"{sample_id}_error",
                        "code": error_code,
                        "PDG": pdg,
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
        writer = csv.DictWriter(f, fieldnames=["id", "code", "PDG", "is_error", "language"])
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

            pdg = gen_pdg(code, lang)
            if pdg is None or count_pdg_nodes(pdg) < args.min_nodes:
                continue

            preview_rows.append(
                {
                    "id": f"preview_{len(preview_rows)}",
                    "code": code,
                    "PDG": pdg,
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
            f.write(row["PDG"] + "\n")

    print(f"Saved preview rows: {min(len(preview_rows), args.preview_total)} -> {args.preview_output}")


def parse_args():
    output_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Generate PDG datasets in normalized DOT format.")
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
    parser.add_argument("--train-output", default=str(output_dir / "pdg_codesearchnet_train.csv"))
    parser.add_argument("--test-output", default=str(output_dir / "pdg_codesearchnet_test.csv"))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-total", type=int, default=5)
    parser.add_argument("--preview-source-limit", type=int, default=20)
    parser.add_argument("--preview-random", action="store_true")
    parser.add_argument("--preview-seed", type=int, default=3407)
    parser.add_argument("--preview-output", default=str(output_dir / "pdg_first5_preview.txt"))
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.preview:
        write_preview(parsed_args)
    else:
        gen(parsed_args)
