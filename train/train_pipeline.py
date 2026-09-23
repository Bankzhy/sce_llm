#!/usr/bin/env python3
"""Run HCG fine-tuning stages in dependency order with restart support.

The default assumes Code Repair is already trained at its standard save path,
then runs Graph Generation followed by Code Graph Explanation. Re-running the
script skips stages that this pipeline previously completed successfully.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .train_code_repair import DEFAULT_MODEL as DEFAULT_BASE_MODEL
    from .train_code_repair import model_suffix_from_name as repair_model_suffix
    from .train_hierarchical import model_suffix_from_name as graph_model_suffix
    from .train_code_graph_explanation import (
        model_suffix_from_name as explanation_model_suffix,
    )
except ImportError:  # Direct execution: python3 train/train_pipeline.py
    from train_code_repair import DEFAULT_MODEL as DEFAULT_BASE_MODEL
    from train_code_repair import model_suffix_from_name as repair_model_suffix
    from train_hierarchical import model_suffix_from_name as graph_model_suffix
    from train_code_graph_explanation import (
        model_suffix_from_name as explanation_model_suffix,
    )


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT_DIR / "train"
DEFAULT_REPAIR_MODEL = ROOT_DIR / f"lora_model_code_repair_{repair_model_suffix(DEFAULT_BASE_MODEL)}"
DEFAULT_GRAPH_DATA = ROOT_DIR / "dataset" / "codesearchnet_graph_train_clean.csv"
DEFAULT_EXPLANATION_DATA = (
    ROOT_DIR / "dataset" / "static_analysis_qa_10000"
)
DEFAULT_STATE_FILE = ROOT_DIR / "outputs" / "training_pipeline_state.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue from a trained Code Repair adapter, then train Graph "
            "Generation and Code Graph Explanation in order."
        )
    )
    parser.add_argument("--repair-model", default=str(DEFAULT_REPAIR_MODEL))
    parser.add_argument("--graph-train-file", default=str(DEFAULT_GRAPH_DATA))
    parser.add_argument(
        "--explanation-train-files",
        nargs="+",
        default=[
            str(DEFAULT_EXPLANATION_DATA / "java" / "train.jsonl"),
            str(DEFAULT_EXPLANATION_DATA / "python" / "train.jsonl"),
        ],
    )
    parser.add_argument("--graph-save-dir")
    parser.add_argument("--graph-output-dir")
    parser.add_argument("--explanation-save-dir")
    parser.add_argument("--explanation-output-dir")
    parser.add_argument("--state-file", default=str(DEFAULT_STATE_FILE))
    parser.add_argument("--graph-learning-rate", type=float, default=2e-4)
    parser.add_argument("--graph-epochs", type=float, default=3.0)
    parser.add_argument("--explanation-learning-rate", type=float, default=2e-4)
    parser.add_argument("--explanation-epochs", type=float, default=3.0)
    parser.add_argument("--graph-max-seq-length", type=int, default=8192)
    parser.add_argument("--explanation-max-seq-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--graph-resume-from-checkpoint")
    parser.add_argument("--explanation-resume-from-checkpoint")
    parser.add_argument("--max-graph-samples", type=int)
    parser.add_argument("--max-explanation-samples", type=int)
    parser.add_argument(
        "--force-graph",
        action="store_true",
        help="Retrain Graph Generation even if pipeline state marks it complete.",
    )
    parser.add_argument(
        "--force-explanation",
        action="store_true",
        help="Retrain Explanation even if pipeline state marks it complete.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate datasets and print both stages without loading a model.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the automatic dataset dry-run before GPU training.",
    )
    args = parser.parse_args(argv)

    repair_model = Path(args.repair_model).expanduser().resolve()
    graph_suffix = graph_model_suffix(str(repair_model))
    args.graph_save_dir = args.graph_save_dir or str(
        ROOT_DIR / f"lora_model_graph_generation_{graph_suffix}"
    )
    args.graph_output_dir = args.graph_output_dir or str(
        ROOT_DIR / "outputs" / f"graph_generation_{graph_suffix}"
    )
    explanation_suffix = explanation_model_suffix(Path(args.graph_save_dir).name)
    args.explanation_save_dir = args.explanation_save_dir or str(
        ROOT_DIR / f"lora_model_code_graph_explanation_{explanation_suffix}"
    )
    args.explanation_output_dir = args.explanation_output_dir or str(
        ROOT_DIR / "outputs" / f"code_graph_explanation_{explanation_suffix}"
    )
    return args


def _require_file(path: str, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_file():
        raise FileNotFoundError(f"{label} does not exist: {value}")
    return value


def _ensure_dataset(path: str, label: str) -> Path:
    """Use a dataset file, extracting its checked-in gzip when necessary."""
    value = Path(path).expanduser().resolve()
    if value.is_file():
        return value
    archive = Path(f"{value}.gz")
    if archive.is_file():
        value.parent.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {label}: {archive}")
        with gzip.open(archive, "rb") as source, value.open("wb") as target:
            shutil.copyfileobj(source, target)
        return value
    raise FileNotFoundError(
        f"{label} does not exist. Upload either of these "
        f"files to the training server:\n  {value}\n  {archive}"
    )


def _ensure_graph_dataset(path: str) -> Path:
    return _ensure_dataset(path, "Clean graph training dataset")


def _require_model(path: str, label: str) -> Path:
    value = Path(path).expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"{label} directory does not exist: {value}")
    markers = ("adapter_config.json", "config.json")
    if not any((value / marker).is_file() for marker in markers):
        raise FileNotFoundError(
            f"{label} does not contain adapter_config.json or config.json: {value}"
        )
    return value


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "stages": {}}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "stages": {}}
    return value if isinstance(value, dict) else {"version": 1, "stages": {}}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _completed(state: dict[str, Any], stage: str, save_dir: Path) -> bool:
    stage_state = state.get("stages", {}).get(stage, {})
    return (
        stage_state.get("status") == "complete"
        and Path(stage_state.get("save_dir", "")).resolve() == save_dir.resolve()
        and save_dir.is_dir()
        and any((save_dir / marker).is_file() for marker in ("adapter_config.json", "config.json"))
    )


def _run(command: list[str], *, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + " ".join(command), flush=True)
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT_DIR,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def graph_command(args: argparse.Namespace, *, dry_run: bool = False) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_DIR / "train_hierarchical.py"),
        "--train-file", str(Path(args.graph_train_file).expanduser().resolve()),
        "--model-name", str(Path(args.repair_model).expanduser().resolve()),
        "--output-dir", str(Path(args.graph_output_dir).expanduser().resolve()),
        "--save-dir", str(Path(args.graph_save_dir).expanduser().resolve()),
        "--max-seq-length", str(args.graph_max_seq_length),
        "--per-device-train-batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--num-train-epochs", str(args.graph_epochs),
        "--learning-rate", str(args.graph_learning_rate),
        "--seed", str(args.seed),
    ]
    if args.max_graph_samples is not None:
        command += ["--max-samples", str(args.max_graph_samples)]
    if args.graph_resume_from_checkpoint:
        command += ["--resume-from-checkpoint", args.graph_resume_from_checkpoint]
    if dry_run:
        command += ["--dry-run", "--preview-samples", "0"]
    return command


def explanation_command(
    args: argparse.Namespace,
    *,
    dry_run: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        str(TRAIN_DIR / "train_code_graph_explanation.py"),
        "--train-files",
        *[str(Path(path).expanduser().resolve()) for path in args.explanation_train_files],
        "--model-name", str(Path(args.graph_save_dir).expanduser().resolve()),
        "--output-dir", str(Path(args.explanation_output_dir).expanduser().resolve()),
        "--save-dir", str(Path(args.explanation_save_dir).expanduser().resolve()),
        "--max-seq-length", str(args.explanation_max_seq_length),
        "--per-device-train-batch-size", str(args.batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--num-train-epochs", str(args.explanation_epochs),
        "--learning-rate", str(args.explanation_learning_rate),
        "--seed", str(args.seed),
    ]
    if args.max_explanation_samples is not None:
        command += ["--max-samples", str(args.max_explanation_samples)]
    if args.explanation_resume_from_checkpoint:
        command += [
            "--resume-from-checkpoint",
            args.explanation_resume_from_checkpoint,
        ]
    if dry_run:
        command += ["--dry-run", "--preview-samples", "0"]
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    graph_data = _ensure_graph_dataset(args.graph_train_file)
    for path in args.explanation_train_files:
        _ensure_dataset(path, "Explanation training dataset")
    repair_model = Path(args.repair_model).expanduser().resolve()
    if not args.dry_run:
        _require_model(str(repair_model), "Code Repair model")

    graph_save = Path(args.graph_save_dir).expanduser().resolve()
    explanation_save = Path(args.explanation_save_dir).expanduser().resolve()
    state_path = Path(args.state_file).expanduser().resolve()
    state = _load_state(state_path)
    logs = ROOT_DIR / "outputs" / "pipeline_logs"

    print("HCG training pipeline")
    print(f"  repair model:       {repair_model}")
    print(f"  graph data:         {graph_data}")
    print(f"  graph model:        {graph_save}")
    print(f"  explanation model:  {explanation_save}")
    print(f"  state:              {state_path}")

    if args.dry_run:
        _run(graph_command(args, dry_run=True), log_path=logs / "preflight_graph.log")
        _run(
            explanation_command(args, dry_run=True),
            log_path=logs / "preflight_explanation.log",
        )
        print("\nDry-run passed. No model was trained.")
        return 0

    if not args.skip_preflight:
        _run(graph_command(args, dry_run=True), log_path=logs / "preflight_graph.log")
        _run(
            explanation_command(args, dry_run=True),
            log_path=logs / "preflight_explanation.log",
        )

    graph_trained_now = False
    if args.force_graph or not _completed(state, "graph_generation", graph_save):
        _run(graph_command(args), log_path=logs / "graph_generation.log")
        _require_model(str(graph_save), "Graph Generation model")
        graph_trained_now = True
        state.setdefault("stages", {})["graph_generation"] = {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "save_dir": str(graph_save),
        }
        _save_state(state_path, state)
    else:
        print("\nGraph Generation already completed; skipping.")

    _require_model(str(graph_save), "Graph Generation model")
    if graph_trained_now or args.force_explanation or not _completed(
        state,
        "code_graph_explanation",
        explanation_save,
    ):
        _run(
            explanation_command(args),
            log_path=logs / "code_graph_explanation.log",
        )
        _require_model(str(explanation_save), "Code Graph Explanation model")
        state.setdefault("stages", {})["code_graph_explanation"] = {
            "status": "complete",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "save_dir": str(explanation_save),
        }
        state["final_model"] = str(explanation_save)
        _save_state(state_path, state)
    else:
        print("\nCode Graph Explanation already completed; skipping.")

    print("\nTraining pipeline completed successfully.")
    print(f"Final client model: {explanation_save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
