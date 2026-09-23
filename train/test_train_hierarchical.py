import ast
import csv
import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import train.train_hierarchical as train_hierarchical
from train.train_hierarchical import (
    build_graph_user_prompt,
    has_lora_adapter,
    load_hierarchical_examples,
    make_existing_adapter_trainable,
    normalize_graph,
    parse_args,
    prepare_lora_model,
    strip_source_comments,
)


class FakeParameter:
    def __init__(self, size, requires_grad=False):
        self._size = size
        self.requires_grad = requires_grad

    def requires_grad_(self, value):
        self.requires_grad = value
        return self

    def numel(self):
        return self._size


class FakeAdapterModel:
    peft_config = {"default": object()}

    def __init__(self):
        self.enabled = False
        self.training = False
        self.params = {
            "base.weight": FakeParameter(100, False),
            "layer.lora_A.default.weight": FakeParameter(12, False),
            "layer.lora_B.default.weight": FakeParameter(12, False),
        }

    def enable_adapter_layers(self):
        self.enabled = True

    def train(self):
        self.training = True

    def named_parameters(self):
        return self.params.items()

    def parameters(self):
        return self.params.values()


class FakeFastLanguageModel:
    calls = 0

    @classmethod
    def get_peft_model(cls, model, **kwargs):
        cls.calls += 1
        return (model, kwargs)


class TrainHierarchicalContinuationTests(unittest.TestCase):
    def test_unsloth_is_imported_before_transformers_and_trl(self):
        tree = ast.parse(inspect.getsource(train_hierarchical.main))
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imports.setdefault(node.module, node.lineno)

        self.assertLess(imports["unsloth"], imports["transformers"])
        self.assertLess(imports["unsloth"], imports["trl"])

    def test_defaults_use_separate_graph_training(self):
        args = parse_args([])
        self.assertFalse(args.packing)
        self.assertTrue(args.assistant_only_loss)
        self.assertFalse(args.include_error_samples)

    def test_existing_adapter_is_detected_and_enabled(self):
        model = FakeAdapterModel()
        self.assertTrue(has_lora_adapter(model))
        self.assertEqual(make_existing_adapter_trainable(model), 24)
        self.assertTrue(model.enabled)
        self.assertTrue(model.training)
        self.assertFalse(model.params["base.weight"].requires_grad)

    def test_existing_adapter_is_not_reinitialized(self):
        FakeFastLanguageModel.calls = 0
        model = FakeAdapterModel()
        result = prepare_lora_model(
            model,
            FakeFastLanguageModel,
            SimpleNamespace(lora_r=32, lora_alpha=64, seed=3407),
        )
        self.assertIs(result, model)
        self.assertEqual(FakeFastLanguageModel.calls, 0)

    def test_plain_base_model_gets_a_new_adapter(self):
        FakeFastLanguageModel.calls = 0
        model = object()
        result = prepare_lora_model(
            model,
            FakeFastLanguageModel,
            SimpleNamespace(lora_r=32, lora_alpha=64, seed=3407),
        )
        self.assertIs(result[0], model)
        self.assertEqual(FakeFastLanguageModel.calls, 1)

    def test_client_prompt_requests_exactly_one_graph(self):
        prompt = build_graph_user_prompt("CFG", "java", "void f() {\n  run();\n}")
        self.assertTrue(prompt.startswith("Generate only the CFG"))
        self.assertIn("The source has exactly 3 lines", prompt)
        self.assertIn("2:   run();", prompt)
        self.assertIn("Start the response exactly with:\ndigraph CFG_graph {", prompt)
        self.assertNotIn("Generate only the AST", prompt)
        self.assertNotIn("### Instruction", prompt)

    def test_one_csv_row_becomes_three_chat_examples(self):
        code = "void f() {\n  run();\n}"
        ast = '''digraph AST_f {
1 [type="method_identifier", offset="lines:1-1", label="f"];
2 [type="process_statement", offset="lines:2-2"];
1 -> 2;
}'''
        cfg = '''digraph CFG_f {
1 [type="process_statement", offset="lines:2-2"];
}'''
        pdg = '''digraph PDG_f {
1 [type="process_statement", offset="lines:2-2"];
}'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graphs.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "code", "AST", "CFG", "PDG", "is_error", "language"])
                writer.writeheader()
                writer.writerow({"id": "one", "code": code, "AST": ast, "CFG": cfg, "PDG": pdg, "is_error": "False", "language": "java"})
            examples = load_hierarchical_examples(str(path), max_samples=1)
        self.assertEqual([row["graph_type"] for row in examples], ["AST", "CFG", "PDG"])
        for example in examples:
            self.assertEqual([message["role"] for message in example["messages"]], ["user", "assistant"])
            self.assertTrue(example["messages"][1]["content"].startswith(f'digraph {example["graph_type"]}_graph {{'))

    def test_ast_target_gets_one_root_and_drops_comment_nodes(self):
        source = "void f() {\n// note\nrun();\n}"
        graph = '''digraph AST_f {
1 [type="method_identifier", offset="lines:1-1", label="f"];
2 [type="process_statement", offset="lines:2-2"];
3 [type="process_statement", offset="lines:3-3"];
1 -> 2;
1 -> 3;
}'''
        normalized = normalize_graph(graph, "AST", source)
        self.assertEqual(normalized.count('type="root"'), 1)
        self.assertNotIn('offset="lines:2-2"', normalized)
        self.assertTrue(normalized.startswith("digraph AST_graph {"))

    def test_current_ast_contract_is_not_given_duplicate_root_or_method(self):
        source = "void f() {\nrun();\n}"
        graph = '''digraph AST_graph {
n0 [type="root", offset="lines:1-3"];
n1 [type="method_declaration", offset="lines:1-3"];
n2 [type="process_statement", offset="lines:2-2"];
n0 -> n1;
n1 -> n2;
}'''
        normalized = normalize_graph(graph, "AST", source)
        self.assertEqual(normalized.count('type="root"'), 1)
        self.assertEqual(normalized.count('type="method_declaration"'), 1)
        self.assertIn('"n0" -> "n1"', normalized)

    def test_comment_filter_matches_client_shape(self):
        source = 'run("// text"); // comment\n/* block\ncomment */ next();'
        stripped = strip_source_comments(source, "java")
        self.assertEqual(len(stripped), len(source))
        self.assertEqual(stripped.count("\n"), source.count("\n"))
        self.assertIn('"// text"', stripped)
        self.assertNotIn("comment", stripped)

        python = 'value = "# text"  # comment\n"""doc\ntext"""\nrun()'
        filtered = strip_source_comments(python, "python")
        self.assertEqual(len(filtered), len(python))
        self.assertIn('"# text"', filtered)
        self.assertNotIn("comment", filtered)
        self.assertNotIn("doc", filtered)


if __name__ == "__main__":
    unittest.main()
