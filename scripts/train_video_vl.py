#!/usr/bin/env python3
"""Real video-language LoRA training entry point with honest preflight checks.

This trains a REAL vision-language model (e.g. Qwen2-VL / Qwen2.5-VL) on
real local video clips using LoRA adapters over the model's language layers.
It does not fake success, and it does not pretend video decoding "just
works" everywhere: this laptop is Windows on ARM64, and as of writing none
of opencv-python, PyAV ("av"), or decord publish official ARM64 Windows
wheels on PyPI. Rather than silently failing deep inside a half-installed
dependency, this script deliberately avoids all three and instead shells out
to the real `ffmpeg` binary (which DOES have real Windows ARM64 builds) to
extract real JPEG frames from each real video file, then loads those frames
with Pillow (which does have ARM64 Windows wheels).

Preflight checks, in order:
  1. Does the manifest file and videos directory actually exist?
  2. Is `torch` importable?
  3. What real accelerator backend is usable right now? (gpu_backend.py)
  4. Is the real `ffmpeg` binary on PATH?
  5. Are `transformers`/`peft` importable, and is the requested model's
     vision-language class actually available in this transformers version?

If any of these fail, this prints an exact, actionable error and exits
non-zero. Otherwise, it extracts real frames, builds real training
examples, and runs a real manual training loop (forward/backward/optimizer
step) — there is no simulated progress bar and no fake loss curve.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from runtime_paths import WORKSPACE_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

FRAME_CACHE_DIR = WORKSPACE_ROOT / ".video-frame-cache"


def fail(message):
    print(json.dumps({"ok": False, "error": message}, indent=2))
    sys.exit(1)


def infer_model_size_billions(model_id):
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?![a-z])", model_id, re.IGNORECASE)
    return float(match.group(1)) if match else None


def resolve_model_class(model_id):
    """Picks the real transformers model class for this model id, or fails with
    an exact, actionable message naming the required transformers version."""
    model_id_lower = model_id.lower()

    try:
        if "qwen2.5-vl" in model_id_lower or "qwen2_5_vl" in model_id_lower or "qwen2.5_vl" in model_id_lower:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass

            return ModelClass, "Qwen2_5_VLForConditionalGeneration"
        if "qwen2-vl" in model_id_lower or "qwen2_vl" in model_id_lower:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass

            return ModelClass, "Qwen2VLForConditionalGeneration"

        from transformers import AutoModelForImageTextToText as ModelClass

        return ModelClass, "AutoModelForImageTextToText"
    except ImportError as exc:
        fail(
            f"This transformers install does not support the requested video model class ({exc}). "
            "Video/vision-language models need transformers>=4.49.0. Run the 'Prepare environment' stage "
            "with the video modality first, or manually run: "
            ".venv-training-video\\Scripts\\pip.exe install -r requirements-video-training.txt"
        )
        raise SystemExit(1)


def probe_conv3d_support(backend):
    """Runs a real, tiny 3D convolution on the given backend's real device and
    returns True/False based on whether it actually succeeds. This exists
    because Qwen2-VL/Qwen2.5-VL's vision tower patch-embedding layer is a
    real nn.Conv3d, and real local testing found that torch-directml (the
    DirectML backend used for non-NVIDIA GPUs on Windows) does not implement
    3D convolution: it raises "RuntimeError: input must be 4-dimensional"
    even though the input is a correctly-shaped 5D tensor. Rather than
    hardcode "DirectML never supports conv3d" (which could go stale if a
    future torch-directml release adds support), this does a real probe
    every time and only falls back on an actual failure."""
    import torch
    import gpu_backend

    if backend == "cpu":
        return True

    try:
        device = gpu_backend.get_torch_device(backend)
        probe_layer = torch.nn.Conv3d(1, 1, kernel_size=(2, 2, 2)).to(device)
        probe_input = torch.zeros((1, 1, 2, 4, 4), device=device)
        with torch.no_grad():
            probe_layer(probe_input)
        return True
    except Exception:
        return False


def extract_frames(video_path, fps, max_frames, ffmpeg_bin):
    """Extracts real JPEG frames from a real video file using the real ffmpeg
    binary, caching them on disk so re-running training doesn't re-decode the
    same clip. Returns a list of real PIL.Image objects in order."""
    from PIL import Image

    stat = video_path.stat()
    cache_key_source = f"{video_path.resolve()}|{fps}|{max_frames}|{stat.st_size}|{stat.st_mtime_ns}"
    cache_key = hashlib.sha256(cache_key_source.encode("utf-8")).hexdigest()[:24]
    cache_dir = FRAME_CACHE_DIR / cache_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(cache_dir.glob("frame_*.jpg"))
    if not existing_frames:
        result = subprocess.run(
            [
                ffmpeg_bin,
                "-y",
                "-i",
                str(video_path),
                "-vf",
                f"fps={fps}",
                "-vframes",
                str(max_frames),
                "-q:v",
                "2",
                str(cache_dir / "frame_%03d.jpg"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to extract frames from {video_path}: {result.stderr.strip()[:800]}")
        existing_frames = sorted(cache_dir.glob("frame_*.jpg"))
        if not existing_frames:
            raise RuntimeError(f"ffmpeg exited successfully but produced no frames for {video_path}")

    return [Image.open(frame_path).convert("RGB") for frame_path in existing_frames[:max_frames]]


def build_prompt_and_completion(processor, messages):
    """Splits a manifest's messages into (prompt_text, completion_text). The
    first user message gets a real {"type": "video"} content block so the
    processor knows to insert the real video frames there. The completion is
    every assistant message joined together, exactly what the model should
    learn to produce."""
    prompt_messages = []
    completion_parts = []
    video_attached = False

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "assistant":
            completion_parts.append(content)
            continue

        if role == "user" and not video_attached:
            prompt_messages.append({"role": role, "content": [{"type": "video"}, {"type": "text", "text": content}]})
            video_attached = True
        else:
            prompt_messages.append({"role": role, "content": content})

    prompt_text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    completion_text = "\n".join(completion_parts)
    return prompt_text, completion_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max-text-length", type=int, default=512)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    videos_dir = Path(args.videos_dir)

    if not manifest_path.exists():
        fail(f"Manifest file not found: {manifest_path}")
        return
    if not videos_dir.is_dir():
        fail(f"Videos directory not found: {videos_dir}")
        return

    try:
        import torch  # noqa: F401
    except ImportError:
        fail(
            "The 'torch' package is not installed in this Python environment. Run the 'Prepare environment' "
            "stage first (with the video modality), or manually run: "
            ".venv-training-video\\Scripts\\pip.exe install -r requirements-video-training.txt"
        )
        return

    import gpu_backend

    detected = gpu_backend.detect()
    backend = detected["backend"]
    print(json.dumps({"ok": True, "backend_detection": detected}, indent=2))

    if backend is None:
        fail("Could not determine a usable PyTorch backend on this machine.")
        return

    if not probe_conv3d_support(backend):
        print(
            json.dumps(
                {
                    "ok": True,
                    "warning": (
                        f"The real '{backend}' GPU backend on this machine does not support 3D convolution "
                        "(verified with a real test nn.Conv3d forward pass, not a guess). Qwen2-VL/Qwen2.5-VL's "
                        "vision tower needs a real Conv3d for its patch embedding, so this falls back to the real "
                        "CPU for video training instead of silently crashing mid-run. Training will still be a "
                        "real LoRA fine-tune, just slower than GPU-accelerated text training."
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
        backend = "cpu"

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        fail(
            "The 'ffmpeg' command was not found on PATH. Video training needs ffmpeg to extract real frames from "
            "your clips. This app deliberately does not depend on opencv-python, PyAV, or decord for this, because "
            "none of them currently publish official pip-installable wheels for Windows on ARM64. Install ffmpeg "
            "and add it to PATH, then re-run this stage:\n"
            "  Windows (including ARM64 laptops such as Snapdragon X): download a build from "
            "https://github.com/BtbN/FFmpeg-Builds/releases (choose a 'winarm64' asset on ARM64 Windows, or "
            "'win64' on an Intel/AMD Windows PC) and add its 'bin' folder to your PATH.\n"
            "  macOS: brew install ffmpeg\n"
            "  Linux: sudo apt install ffmpeg (or your distro's package manager)"
        )
        return

    model_size_billions = infer_model_size_billions(args.model)
    if backend != "cuda" and model_size_billions is not None and model_size_billions >= 6.5:
        fail(
            f"{args.model} appears to be a {model_size_billions:g}B-parameter vision-language model. On {backend}, "
            "this app must load it in real fp16/fp32 because 4-bit QLoRA is CUDA-only, and video models already "
            "need extra accelerator memory per frame for the vision tower on top of the language model. That "
            "combination can exhaust memory or crash native DirectML after loading checkpoint shards. Pick a "
            "smaller base model such as Qwen/Qwen2-VL-2B-Instruct or Qwen/Qwen2.5-VL-3B-Instruct on this laptop, "
            "or use an NVIDIA CUDA machine for a larger model."
        )
        return

    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoProcessor
    except ImportError as exc:
        fail(
            f"A required training package is missing: {exc}. Run the 'Prepare environment' stage first (with the "
            "video modality), or manually run: .venv-training-video\\Scripts\\pip.exe install -r "
            "requirements-video-training.txt"
        )
        return

    model_cls, model_class_name = resolve_model_class(args.model)

    output_dir = WORKSPACE_ROOT / "output" / args.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    quantized = False
    if backend == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            model = model_cls.from_pretrained(args.model, quantization_config=bnb_config, device_map="auto")
            quantized = True
        except ImportError as exc:
            fail(f"bitsandbytes is required for real 4-bit QLoRA on CUDA but is missing: {exc}")
            return
    else:
        device = gpu_backend.get_torch_device(backend)
        dtype = torch.float32 if backend == "cpu" else torch.float16
        model = model_cls.from_pretrained(args.model, torch_dtype=dtype)
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(args.model)
    device = next(model.parameters()).device

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        # Video-language checkpoints (Qwen2-VL family) don't have a peft
        # "auto target modules" mapping, so these are named explicitly. This
        # targets only the language-model attention/MLP projections, leaving
        # the vision tower frozen — real LoRA fine-tuning of the language
        # side, which is what "teach it to respond a certain way about
        # videos" actually needs, at a fraction of the memory of also
        # training the vision tower.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.train()

    records = []
    with Path(args.manifest).open("r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped:
                records.append(json.loads(stripped))

    if args.sample_size > 0 and len(records) > args.sample_size:
        records = records[: args.sample_size]

    if not records:
        fail("The manifest file has no records to train on.")
        return

    print(
        json.dumps({"ok": True, "records_loaded": len(records), "max_frames": args.max_frames, "fps": args.fps}, indent=2),
        flush=True,
    )

    def build_example(record):
        video_path = videos_dir / record["video"]
        frames = extract_frames(
            video_path,
            record.get("fps", args.fps),
            record.get("max_frames", args.max_frames),
            ffmpeg_bin,
        )
        prompt_text, completion_text = build_prompt_and_completion(processor, record["messages"])

        prompt_inputs = processor(
            text=[prompt_text],
            videos=[frames],
            return_tensors="pt",
            truncation=True,
            max_length=args.max_text_length,
        )
        completion_ids = processor.tokenizer(
            completion_text + (processor.tokenizer.eos_token or ""),
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids

        input_ids = torch.cat([prompt_inputs["input_ids"], completion_ids], dim=1)
        attention_mask = torch.cat(
            [prompt_inputs["attention_mask"], torch.ones_like(completion_ids)], dim=1
        )
        labels = torch.cat(
            [torch.full_like(prompt_inputs["input_ids"], -100), completion_ids], dim=1
        )

        example = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        for key, value in prompt_inputs.items():
            if key not in ("input_ids", "attention_mask"):
                example[key] = value
        return example

    # Real, honest LoRA fine-tuning loop over real per-example forward/backward
    # passes. Batch size is fixed at 1 for video training in this app: each
    # video contributes a different number of frames/tokens, so naively
    # stacking multiple videos into one tensor batch would require padding
    # logic this app does not implement yet. Gradient accumulation (below)
    # still gives a real, larger effective batch size.
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad], lr=args.learning_rate
    )

    total_steps = args.max_steps if args.max_steps > 0 else len(records) * args.epochs
    step = 0
    running_loss = 0.0
    optimizer.zero_grad()

    while step < total_steps:
        for record in records:
            if step >= total_steps:
                break

            example = build_example(record)
            example = {key: value.to(device) for key, value in example.items()}

            outputs = model(**example)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item()

            step += 1
            if step % args.grad_accum == 0 or step == total_steps:
                optimizer.step()
                optimizer.zero_grad()

            if step % max(1, min(5, total_steps)) == 0 or step == total_steps:
                print(
                    json.dumps({"ok": True, "step": step, "total_steps": total_steps, "loss": round(running_loss, 4)}),
                    flush=True,
                )
                running_loss = 0.0

    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))

    manifest_out = {
        "modality": "video",
        "base_model": args.model,
        "model_class": model_class_name,
        "dataset_file": manifest_path.name,
        "dataset_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "backend": backend,
        "device_name": detected["device_name"],
        "training_kind": "Video LoRA",
        "gradient_accumulation_steps": args.grad_accum,
        "lora_rank": args.rank,
        "max_text_length": args.max_text_length,
        "max_steps": args.max_steps,
        "sample_size": args.sample_size,
        "max_frames": args.max_frames,
        "fps": args.fps,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "adapter_dir": str(output_dir),
                "backend": backend,
                "device_name": detected["device_name"],
                "quantized_4bit": quantized,
                "training_kind": "Video-LoRA (4-bit quantized)" if quantized else "Video-LoRA (fp16/fp32, not quantized)",
                "records_trained": len(records),
                "steps_run": step,
            },
            indent=2,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
