#!/usr/bin/env python3
"""Real preflight-gated adapter evaluation.

Honestly reports failure if no real adapter checkpoint exists on disk.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    adapter_dir = REPO_ROOT / "output" / args.output_name
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"No adapter checkpoint found at {adapter_dir}. "
                        "Run the 'QLoRA training' stage first and confirm it completed."
                    ),
                },
                indent=2,
            )
        )
        sys.exit(1)

    print(json.dumps({"ok": True, "adapter_dir": str(adapter_dir), "note": "Adapter files found on disk."}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
