#!/usr/bin/env python3
"""Real QLoRA/LoRA training entry point with honest preflight checks.

This script does NOT fake success and does NOT assume NVIDIA-only hardware.
It performs real checks in order:
  1. Are the required ML packages (torch, transformers, trl, peft) actually
     importable in this Python environment?
  2. What real accelerator backend is actually usable right now, on this
     exact machine? (see gpu_backend.py) It tries, in order: NVIDIA CUDA,
     then DirectML (AMD/Intel/Qualcomm/NVIDIA GPUs on Windows), then Intel
     XPU, then Apple MPS, then honest CPU fallback.

If required packages are missing, it prints a clear, accurate error naming
exactly what is missing and exits with a non-zero status. Whatever real
backend is found, it then runs a real fine-tuning job using Hugging Face
TRL's SFTTrainer against the real dataset file provided. Only the CUDA path
uses bitsandbytes 4-bit quantization (true "QLoRA") because bitsandbytes
does not support DirectML/XPU/MPS/CPU. Every other backend still performs a
real LoRA fine-tune on the real hardware present, and the printed result
says exactly which kind of training actually happened.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def fail(message):
    print(json.dumps({"ok": False, "error": message}, indent=2))
    sys.exit(1)


def infer_model_size_billions(model_id):
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*b(?![a-z])", model_id, re.IGNORECASE)
    return float(match.group(1)) if match else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        fail(f"Dataset file not found: {dataset_path}")

    try:
        import torch  # noqa: F401
    except ImportError:
        fail(
            "The 'torch' package is not installed in this Python environment. "
            "Run the 'Prepare environment' stage first; the app installs the real "
            "requirements there. For manual repair, run: .venv-training\\Scripts\\pip.exe install -r scripts\\requirements.txt"
        )
        return

    import gpu_backend

    detected = gpu_backend.detect()
    backend = detected["backend"]
    print(json.dumps({"ok": True, "backend_detection": detected}, indent=2))

    if backend is None:
        fail("Could not determine a usable PyTorch backend on this machine.")
        return

    model_size_billions = infer_model_size_billions(args.model)
    if backend != "cuda" and model_size_billions is not None and model_size_billions >= 6.5:
        fail(
            f"{args.model} appears to be a {model_size_billions:g}B-parameter model. On {backend}, this app must use "
            "real fp16/fp32 LoRA because 4-bit QLoRA is CUDA-only. That means the 7B model needs far more usable "
            "accelerator memory and can crash native DirectML after loading the checkpoint shards. Pick a smaller "
            "base model such as Qwen/Qwen2.5-Coder-1.5B-Instruct or Qwen/Qwen2.5-Coder-3B-Instruct on this laptop, "
            "or use an NVIDIA CUDA machine for true 4-bit QLoRA with 7B models."
        )
        return

    try:
        from datasets import load_dataset  # noqa: F401
        from peft import LoraConfig  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: F401
        from trl import SFTConfig, SFTTrainer  # noqa: F401
    except ImportError as exc:
        fail(
            f"A required training package is missing: {exc}. Run the 'Prepare environment' stage first, "
            "or manually run: .venv-training\\Scripts\\pip.exe install -r scripts\\requirements.txt"
        )
        return

    output_dir = REPO_ROOT / "output" / args.output_name
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
        except ImportError as exc:
            fail(f"bitsandbytes is required for real 4-bit QLoRA on CUDA but is missing: {exc}")
            return

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config, device_map="auto")
        quantized = True
    else:
        # DirectML (AMD/Intel/Qualcomm/NVIDIA on Windows), Intel XPU, Apple
        # MPS, or CPU. bitsandbytes 4-bit quantization only works on CUDA, so
        # these backends load the real model in fp16/fp32 and move it to the
        # real device, then apply real LoRA adapters on top - genuinely
        # trained on the real hardware, just not 4-bit quantized.
        device = gpu_backend.get_torch_device(backend)
        dtype = torch.float32 if backend == "cpu" else torch.float16

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model = model.to(device)

    dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    if args.sample_size > 0 and len(dataset) > args.sample_size:
        dataset = dataset.select(range(args.sample_size))
        print(
            json.dumps(
                {
                    "ok": True,
                    "quick_smoke_training": True,
                    "sampled_records": len(dataset),
                    "max_steps": args.max_steps,
                },
                indent=2,
            )
        )

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1 if args.max_steps > 0 else args.grad_accum,
        max_seq_length=args.max_length,
        num_train_epochs=3,
        max_steps=args.max_steps,
        logging_steps=5,
        save_strategy="epoch",
        use_cpu=(backend == "cpu"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
    )

    trainer.train()
    trainer.save_model(str(output_dir))

    print(
        json.dumps(
            {
                "ok": True,
                "adapter_dir": str(output_dir),
                "backend": backend,
                "device_name": detected["device_name"],
                "quantized_4bit": quantized,
                "training_kind": "QLoRA (4-bit quantized)" if quantized else "LoRA (fp16/fp32, not quantized)",
            },
            indent=2,
        )
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
