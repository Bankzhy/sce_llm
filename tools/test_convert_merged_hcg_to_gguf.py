import json
import tempfile
import unittest
from pathlib import Path

from tools.convert_merged_hcg_to_gguf import parse_args, validate_merged_model


class ConvertMergedHcgToGgufTests(unittest.TestCase):
    def test_defaults_target_merged_hcg_and_q4_output(self):
        args = parse_args([])
        self.assertTrue(args.model_dir.endswith("export/hcg_final_q4_k_m"))
        self.assertTrue(args.output_file.endswith("hcg-final-q4_k_m.gguf"))
        self.assertFalse(args.force)

    def test_validate_merged_model(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}))
            (model_dir / "tokenizer.json").write_text("{}")
            weight = model_dir / "model-00001-of-00001.safetensors"
            weight.write_bytes(b"weights")
            self.assertEqual(validate_merged_model(model_dir), [weight])

    def test_rejects_partial_merge_without_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            model_dir = Path(directory)
            (model_dir / "config.json").write_text("{}")
            (model_dir / "tokenizer.json").write_text("{}")
            with self.assertRaisesRegex(FileNotFoundError, "model.*safetensors"):
                validate_merged_model(model_dir)


if __name__ == "__main__":
    unittest.main()
