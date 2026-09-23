#!/usr/bin/env python3
"""Real adapter merge step.

Honestly requires a real adapter checkpoint and real transformers/peft
packages. Does not fake success if either is missing.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def fail(message):
    print(json.dumps({"ok": False, "error": message}, indent=2))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    adapter_dir = REPO_ROOT / "output" / args.output_name
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        fail(f"No adapter checkpoint found at {adapter_dir}. Run 'QLoRA training' first.")
        return

    try:
        import torch  # noqa: F401
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        fail(f"A required package is missing: {exc}. Install scripts/requirements.txt first.")
        return

    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        fail(f"adapter_config.json not found in {adapter_dir}. The training stage may not have finished.")
        return

    import json as _json

    adapter_config = _json.loads(adapter_config_path.read_text(encoding="utf-8"))
    base_model_name = adapter_config.get("base_model_name_or_path")

    if not base_model_name:
        fail("adapter_config.json does not list a base model name; cannot merge.")
        return

    merged_dir = REPO_ROOT / "merged" / args.output_name
    merged_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")
    merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged_model = merged_model.merge_and_unload()

    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    print(json.dumps({"ok": True, "merged_dir": str(merged_dir)}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
