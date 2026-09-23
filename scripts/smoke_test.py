#!/usr/bin/env python3
"""Real smoke test: sends a real prompt to the real registered Ollama model
and checks that a real, non-empty response comes back.
"""

import argparse
import json
import shutil
import subprocess
import sys


def fail(message):
    print(json.dumps({"ok": False, "error": message}, indent=2))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    ollama_bin = shutil.which("ollama")
    if ollama_bin is None:
        fail("The 'ollama' CLI was not found on PATH.")
        return

    prompt = "Write a one-line PowerShell command to list Azure resource groups."

    result = subprocess.run(
        [ollama_bin, "run", args.output_name, prompt],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        fail(f"Real 'ollama run' command failed: {result.stderr.strip()[:2000]}")
        return

    response = result.stdout.strip()
    if not response:
        fail("The model returned an empty response.")
        return

    print(json.dumps({"ok": True, "prompt": prompt, "response": response[:2000]}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
