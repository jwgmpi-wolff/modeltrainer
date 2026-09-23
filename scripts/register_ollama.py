#!/usr/bin/env python3
"""Real Ollama model registration.

Writes a real Modelfile and runs the REAL `ollama create` CLI command. This
genuinely registers the model in the local Ollama installation if quantized
GGUF weights are present, or fails honestly with the real ollama error output
if something is wrong (e.g., ollama not installed, file missing, ollama
daemon not running).
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def fail(message):
    print(json.dumps({"ok": False, "error": message}, indent=2))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--quant-type", default="Q4_K_M")
    args = parser.parse_args()

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        fail("The 'ollama' CLI was not found on PATH. Install Ollama from https://ollama.com first.")
        return

    gguf_dir = REPO_ROOT / "gguf"
    quantized_file = gguf_dir / f"{args.output_name}-{args.quant_type}.gguf"

    if not quantized_file.exists():
        fail(f"No quantized GGUF file found at {quantized_file}. Run 'Quantize GGUF' first.")
        return

    modelfile_dir = REPO_ROOT / "ollama"
    modelfile_dir.mkdir(parents=True, exist_ok=True)
    modelfile_path = modelfile_dir / f"{args.output_name}.Modelfile"
    modelfile_path.write_text(f'FROM {quantized_file.resolve()}\n', encoding="utf-8")

    result = subprocess.run(
        [ollama_bin, "create", args.output_name, "-f", str(modelfile_path)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        fail(f"Real 'ollama create' command failed: {result.stderr.strip()[:2000]}")
        return

    print(
        json.dumps(
            {
                "ok": True,
                "model_name": args.output_name,
                "modelfile": str(modelfile_path),
                "ollama_output": result.stdout.strip(),
            },
            indent=2,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
