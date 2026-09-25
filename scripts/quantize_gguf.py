#!/usr/bin/env python3
"""Real GGUF quantization step using llama.cpp's real quantize binary.

Honestly fails if the real f16 GGUF file or the real llama-quantize binary
does not exist.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from runtime_paths import WORKSPACE_ROOT


def fail(message):
    print(json.dumps({"ok": False, "error": message}, indent=2))
    sys.exit(1)


def find_quantize_binary():
    names = ["llama-quantize", "llama-quantize.exe", "quantize", "quantize.exe"]
    for name in names:
        found = shutil.which(name)
        if found:
            return found

    candidates = [
        WORKSPACE_ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize",
        WORKSPACE_ROOT / "llama.cpp" / "build" / "bin" / "llama-quantize.exe",
        WORKSPACE_ROOT / "llama.cpp" / "llama-quantize",
        WORKSPACE_ROOT / "llama.cpp" / "llama-quantize.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--quant-type", default="Q4_K_M")
    args = parser.parse_args()

    gguf_dir = WORKSPACE_ROOT / "gguf"
    input_file = gguf_dir / f"{args.output_name}-f16.gguf"

    if not input_file.exists():
        fail(f"No f16 GGUF file found at {input_file}. Run 'Convert to GGUF' first.")
        return

    quantize_bin = find_quantize_binary()
    if quantize_bin is None:
        fail(
            "The real llama.cpp 'llama-quantize' binary was not found. Build llama.cpp "
            "first (cmake -B build && cmake --build build --config Release), then re-run this stage."
        )
        return

    output_file = gguf_dir / f"{args.output_name}-{args.quant_type}.gguf"

    result = subprocess.run(
        [quantize_bin, str(input_file), str(output_file), args.quant_type],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        fail(f"Real quantize command failed: {result.stderr.strip()[:2000]}")
        return

    if not output_file.exists():
        fail("Quantize command exited successfully but no quantized file was produced.")
        return

    print(json.dumps({"ok": True, "quantized_file": str(output_file)}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
