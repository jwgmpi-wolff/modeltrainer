#!/usr/bin/env python3
"""Real adapter merge step.

Honestly requires a real adapter checkpoint and real transformers/peft
packages. Does not fake success if either is missing.
"""

import argparse
import json
import sys
from pathlib import Path

from runtime_paths import WORKSPACE_ROOT


def fail(message):
    print(json.dumps({"ok": False, "error": message}, indent=2))
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    adapter_dir = WORKSPACE_ROOT / "output" / args.output_name
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        fail(f"No adapter checkpoint found at {adapter_dir}. Run 'QLoRA training' first.")
        return

    try:
        import torch  # noqa: F401
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        fail(f"A required package is missing: {exc}. Install requirements-training.txt first.")
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

    # training_manifest.json is written by train_qlora.py (modality "text")
    # and train_video_vl.py (modality "video"). Older adapters trained before
    # this file existed are honestly treated as "text", which was the only
    # modality this app supported at the time.
    training_manifest_path = adapter_dir / "training_manifest.json"
    modality = "text"
    if training_manifest_path.exists():
        modality = _json.loads(training_manifest_path.read_text(encoding="utf-8")).get("modality", "text")

    merged_dir = WORKSPACE_ROOT / "merged" / args.output_name
    merged_dir.mkdir(parents=True, exist_ok=True)

    if modality == "video":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_video_vl import resolve_model_class  # noqa: E402
        from transformers import AutoProcessor

        model_cls, _ = resolve_model_class(base_model_name)
        processor = AutoProcessor.from_pretrained(base_model_name)
        base_model = model_cls.from_pretrained(base_model_name, torch_dtype="auto")
        merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = merged_model.merge_and_unload()

        merged_model.save_pretrained(str(merged_dir))
        processor.save_pretrained(str(merged_dir))
        (merged_dir / "training_manifest.json").write_text(
            _json.dumps({"modality": "video", "base_model": base_model_name}, indent=2), encoding="utf-8"
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name, torch_dtype="auto")
        merged_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        merged_model = merged_model.merge_and_unload()

        merged_model.save_pretrained(str(merged_dir))
        tokenizer.save_pretrained(str(merged_dir))

    print(json.dumps({"ok": True, "merged_dir": str(merged_dir), "modality": modality}, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
