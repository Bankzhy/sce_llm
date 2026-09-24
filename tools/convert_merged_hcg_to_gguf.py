#!/usr/bin/env python3
"""Convert the already-merged HCG model to Q4_K_M GGUF without a source build.

The script creates an isolated virtual environment and installs only the
pre-built ``llama-cpp-pydist`` wheel.  ``--only-binary=:all:`` deliberately
prevents pip from falling back to compiling llama.cpp or its dependencies.

Run from anywhere on the training server::

    python -u tools/convert_merged_hcg_to_gguf.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT_DIR / "export" / "hcg_final_q4_k_m"
DEFAULT_OUTPUT_FILE = ROOT_DIR / "export" / "gguf" / "hcg-final-q4_k_m.gguf"
DEFAULT_INTERMEDIATE_FILE = ROOT_DIR / "export" / "gguf" / "hcg-final-f16.gguf"
DEFAULT_VENV_DIR = ROOT_DIR / ".venv-gguf"
DEFAULT_BINARY_DIR = ROOT_DIR / ".cache" / "llama.cpp-b8329"
PYDIST_REQUIREMENT = "llama-cpp-pydist==0.40.0"
BOOTSTRAP_ENV = "HCG_GGUF_BINARY_ENV"
LLAMA_RELEASE = "b8329"
LLAMA_BINARY_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{LLAMA_RELEASE}/llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an already-merged Hugging Face HCG model to Q4_K_M GGUF "
            "using a pre-built wheel; llama.cpp is never compiled from source."
        )
    )
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-file", default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--intermediate-file", default=str(DEFAULT_INTERMEDIATE_FILE))
    parser.add_argument("--venv-dir", default=str(DEFAULT_VENV_DIR))
    parser.add_argument("--binary-dir", default=str(DEFAULT_BINARY_DIR))
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact output GGUF if it already exists.",
    )
    return parser.parse_args(argv)


def validate_merged_model(model_dir: Path) -> list[Path]:
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Merged model directory does not exist: {model_dir}")
    config = model_dir / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"Merged model config does not exist: {config}")
    weights = sorted(model_dir.glob("model*.safetensors"))
    if not weights:
        raise FileNotFoundError(
            f"No model*.safetensors files found in merged model: {model_dir}"
        )
    tokenizer_markers = (
        model_dir / "tokenizer.json",
        model_dir / "tokenizer.model",
        model_dir / "tokenizer_config.json",
    )
    if not any(path.is_file() for path in tokenizer_markers):
        raise FileNotFoundError(f"No tokenizer files found in merged model: {model_dir}")
    return weights


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def bootstrap_binary_environment(venv_dir: Path) -> Path:
    python = venv_python(venv_dir)
    if not python.is_file():
        print(f"Creating conversion environment: {venv_dir}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "venv", "--system-site-packages", str(venv_dir)],
            check=True,
        )

    print(f"Installing pre-built wheel only: {PYDIST_REQUIREMENT}", flush=True)
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--only-binary=:all:",
            PYDIST_REQUIREMENT,
        ],
        check=True,
    )
    return python


def reexecute_in_binary_environment(args: argparse.Namespace) -> int:
    python = bootstrap_binary_environment(Path(args.venv_dir).expanduser().resolve())
    environment = dict(os.environ)
    environment[BOOTSTRAP_ENV] = "1"
    command = [str(python), str(Path(__file__).resolve()), *sys.argv[1:]]
    return subprocess.run(command, env=environment, check=False).returncode


def ensure_prebuilt_quantizer(binary_dir: Path) -> Path:
    quantizer = binary_dir / f"llama-{LLAMA_RELEASE}" / "llama-quantize"
    if quantizer.is_file():
        return quantizer

    binary_dir.mkdir(parents=True, exist_ok=True)
    archive = binary_dir / f"llama-{LLAMA_RELEASE}-bin-ubuntu-x64.tar.gz"
    partial = archive.with_suffix(archive.suffix + ".part")
    print(f"Downloading official pre-built llama.cpp {LLAMA_RELEASE} binary...", flush=True)
    print(f"  {LLAMA_BINARY_URL}", flush=True)
    try:
        with urllib.request.urlopen(LLAMA_BINARY_URL, timeout=120) as response:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            with partial.open("wb") as target:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    target.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(
                            f"  downloaded: {downloaded / 1024**2:.1f} / "
                            f"{total / 1024**2:.1f} MiB",
                            flush=True,
                        )
        partial.replace(archive)
    except Exception:
        partial.unlink(missing_ok=True)
        raise

    print(f"Extracting pre-built binary: {archive}", flush=True)
    with tarfile.open(archive, "r:gz") as bundle:
        bundle.extractall(binary_dir, filter="data")
    if not quantizer.is_file():
        raise RuntimeError(
            f"Official binary archive did not contain llama-quantize: {quantizer}"
        )
    quantizer.chmod(quantizer.stat().st_mode | 0o111)
    return quantizer


def convert(args: argparse.Namespace) -> Path:
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_file = Path(args.output_file).expanduser().resolve()
    intermediate_file = Path(args.intermediate_file).expanduser().resolve()
    binary_dir = Path(args.binary_dir).expanduser().resolve()
    weights = validate_merged_model(model_dir)

    if output_file.exists():
        if not args.force:
            raise FileExistsError(
                f"Output already exists: {output_file}\n"
                "Use --force only if you intentionally want to replace it."
            )
        output_file.unlink()
    if args.force:
        intermediate_file.unlink(missing_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    intermediate_file.parent.mkdir(parents=True, exist_ok=True)

    print("HCG GGUF conversion", flush=True)
    print(f"  merged model: {model_dir}", flush=True)
    print(f"  weight shards: {len(weights)}", flush=True)
    print(f"  output:        {output_file}", flush=True)
    print(f"  intermediate:  {intermediate_file}", flush=True)
    print("  quantization:  Q4_K_M", flush=True)
    print("  install mode:  pre-built wheel and official binary only", flush=True)

    try:
        from llama_cpp import convert_hf_to_gguf
    except ImportError as error:
        raise RuntimeError(
            "The pre-built llama-cpp-pydist wheel was not installed correctly."
        ) from error

    if intermediate_file.is_file() and intermediate_file.stat().st_size > 0:
        print(f"Reusing existing F16 GGUF: {intermediate_file}", flush=True)
    else:
        print("Converting merged model to F16 GGUF...", flush=True)
        success, result = convert_hf_to_gguf(
            model_path_or_name=str(model_dir),
            output_dir=str(intermediate_file.parent),
            output_filename=intermediate_file.name,
            outtype="f16",
        )
        if not success:
            raise RuntimeError(f"F16 GGUF conversion failed: {result}")
        if not intermediate_file.is_file() or intermediate_file.stat().st_size == 0:
            raise RuntimeError(
                "Converter reported success but did not create a valid F16 GGUF: "
                f"{intermediate_file}"
            )

    quantizer = ensure_prebuilt_quantizer(binary_dir)
    runtime_dir = quantizer.parent
    environment = dict(os.environ)
    previous_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = (
        str(runtime_dir)
        if not previous_library_path
        else f"{runtime_dir}:{previous_library_path}"
    )
    print("Quantizing F16 GGUF to Q4_K_M...", flush=True)
    subprocess.run(
        [str(quantizer), str(intermediate_file), str(output_file), "Q4_K_M"],
        env=environment,
        check=True,
    )
    if not output_file.is_file() or output_file.stat().st_size == 0:
        raise RuntimeError(f"Quantizer did not create a valid file: {output_file}")

    size_gib = output_file.stat().st_size / (1024**3)
    print("GGUF conversion complete.", flush=True)
    print(f"  file: {output_file}", flush=True)
    print(f"  size: {size_gib:.2f} GiB", flush=True)
    return output_file


def main() -> int:
    args = parse_args()
    # Validate before installing anything, so a missing/partial merge fails fast.
    validate_merged_model(Path(args.model_dir).expanduser().resolve())
    if os.environ.get(BOOTSTRAP_ENV) != "1":
        return reexecute_in_binary_environment(args)
    convert(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
