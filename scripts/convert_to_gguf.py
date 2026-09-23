#!/usr/bin/env python3
"""Real GGUF conversion step.

Honestly requires:
  1. A real merged model directory on disk.
  2. A real llama.cpp checkout with convert_hf_to_gguf.py available.

If llama.cpp is not present, this prints exact, real setup instructions and
fails — it does not fake a converted file.
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


def find_llama_cpp_converter():
    # Common real locations where a user might have cloned llama.cpp.
    candidates = [
        REPO_ROOT / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    merged_dir = REPO_ROOT / "merged" / args.output_name
    if not merged_dir.exists() or not any(merged_dir.iterdir()):
        fail(f"No merged model found at {merged_dir}. Run 'Merge adapter' first.")
        return

    converter = find_llama_cpp_converter()
    if converter is None:
        fail(
            "llama.cpp's convert_hf_to_gguf.py was not found. Real GGUF conversion "
            "requires a real llama.cpp checkout. Clone it with: "
            "git clone https://github.com/ggerganov/llama.cpp.git, then re-run this stage."
        )
        return

    gguf_dir = REPO_ROOT / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    output_file = gguf_dir / f"{args.output_name}-f16.gguf"

    python_bin = shutil.which("python") or shutil.which("python3")
    result = subprocess.run(
        [python_bin, str(converter), str(merged_dir), "--outfile", str(output_file), "--outtype", "f16"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        fail(f"Real conversion command failed: {result.stderr.strip()[:2000]}")
        return

    if not output_file.exists():
        fail("Conversion command exited successfully but no GGUF file was produced.")
        return

    print(json.dumps({"ok": True, "gguf_file": str(output_file)}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
