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
import hashlib
import json
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path

from runtime_paths import WORKSPACE_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

current_percent = 0.0


def report_progress(stage, percent, message, current_step=None, total_steps=None):
    global current_percent
    current_percent = percent
    event = {
        "stage": stage,
        "percent": round(percent, 1),
        "message": message,
    }
    if current_step is not None:
        event["currentStep"] = current_step
    if total_steps is not None:
        event["totalSteps"] = total_steps
    print(
        f"MODEL_TRAINER_EVENT|{json.dumps(event, separators=(',', ':'))}",
        flush=True,
    )


def fail(message):
    raise RuntimeError(message)


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
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.0002)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--resume-checkpoint")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        fail(f"Dataset file not found: {dataset_path}")

    output_dir = (WORKSPACE_ROOT / "output" / args.output_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "training_run.json"
    run_config = {
        "model": args.model,
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": 1 if args.max_steps > 0 else args.grad_accum,
        "lora_rank": args.rank,
        "max_length": args.max_length,
        "max_steps": args.max_steps,
        "sample_size": args.sample_size,
    }
    resume_checkpoint = None
    if args.resume_checkpoint:
        resume_checkpoint = Path(args.resume_checkpoint).resolve()
        try:
            resume_checkpoint.relative_to(output_dir)
        except ValueError as exc:
            fail("The resume checkpoint must be inside the selected output directory.")
            raise exc
        if resume_checkpoint.parent != output_dir:
            fail("The resume checkpoint must be directly inside the selected output directory.")
        manifest_path = resume_checkpoint / "training_run.json"
        if not manifest_path.is_file():
            fail("The selected checkpoint has no compatibility manifest and cannot be resumed safely.")
        saved_config = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved_config != run_config:
            fail(
                "The selected checkpoint was created with a different base model, "
                "dataset SHA-256, epochs, batch size, learning rate, or LoRA settings. "
                "Restore the original settings or start new training."
            )
    run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    report_progress("environment", 1, "Checking training dependencies and environment")
    try:
        import torch  # noqa: F401
    except ImportError:
        fail(
            "The 'torch' package is not installed in this Python environment. "
            "Run the 'Prepare environment' stage first; the app installs the real "
            "requirements there. For manual repair, run: .venv-training\\Scripts\\pip.exe install -r requirements-training.txt"
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
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer  # noqa: F401
    except ImportError as exc:
        fail(
            f"A required training package is missing: {exc}. Run the 'Prepare environment' stage first, "
            "or manually run: .venv-training\\Scripts\\pip.exe install -r requirements-training.txt"
        )
        return

    report_progress("dataset", 4, "Loading and formatting the training dataset")
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

    quantized = False
    report_progress("tokenization", 10, f"Loading tokenizer for {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    report_progress("model", 16, f"Loading base model on {backend}")
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

        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
        model = model.to(device)

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
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=1,
        save_strategy="steps",
        save_steps=10,
        save_total_limit=10,
        report_to=[],
        use_cpu=(backend == "cpu"),
    )

    runtime_device = device if backend != "cuda" else torch.device("cuda")

    def copy_to_cpu(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu()
        if isinstance(value, Mapping):
            return type(value)(
                {key: copy_to_cpu(item) for key, item in value.items()}
            )
        if isinstance(value, (tuple, list)):
            return type(value)(copy_to_cpu(item) for item in value)
        return value

    class DirectMLSFTTrainer(SFTTrainer):
        def create_accelerator_and_postprocess(self):
            super().create_accelerator_and_postprocess()
            self.accelerator.device_placement = False

        def _move_model_to_device(self, model, target_device):
            return

        def _prepare_input(self, data):
            if isinstance(data, Mapping):
                return type(data)(
                    {key: self._prepare_input(value) for key, value in data.items()}
                )
            if isinstance(data, (tuple, list)):
                return type(data)(self._prepare_input(value) for value in data)
            if isinstance(data, torch.Tensor):
                return data.to(runtime_device)
            return data

        def _save(self, output_dir=None, state_dict=None):
            cpu_state = copy_to_cpu(
                self.model.state_dict() if state_dict is None else state_dict
            )
            super()._save(output_dir, state_dict=cpu_state)

        def _save_optimizer_and_scheduler(self, checkpoint_dir):
            if self.args.should_save:
                torch.save(
                    copy_to_cpu(self.optimizer.state_dict()),
                    Path(checkpoint_dir) / "optimizer.pt",
                )
                torch.save(
                    copy_to_cpu(self.lr_scheduler.state_dict()),
                    Path(checkpoint_dir) / "scheduler.pt",
                )

    class ProgressCallback(TrainerCallback):
        def on_train_begin(self, training_args, state, control, **kwargs):
            if backend == "directml":
                model_device = next(kwargs["model"].parameters()).device
                if model_device.type != "privateuseone":
                    raise RuntimeError(
                        "DirectML model placement was lost; refusing silent CPU fallback."
                    )
            total = max(state.max_steps, 1)
            percent = 20 + (75 * state.global_step / total)
            report_progress(
                "training",
                percent,
                f"Training step {state.global_step}/{state.max_steps}",
                state.global_step,
                state.max_steps,
            )

        def on_step_end(self, training_args, state, control, **kwargs):
            total = max(state.max_steps, 1)
            percent = 20 + (75 * state.global_step / total)
            report_progress(
                "training",
                percent,
                f"Training step {state.global_step}/{state.max_steps}",
                state.global_step,
                state.max_steps,
            )

        def on_save(self, training_args, state, control, **kwargs):
            checkpoint_manifest = (
                output_dir / f"checkpoint-{state.global_step}" / "training_run.json"
            )
            shutil.copyfile(run_config_path, checkpoint_manifest)
            report_progress(
                "checkpoint",
                20 + (75 * state.global_step / max(state.max_steps, 1)),
                f"Saved complete checkpoint at step {state.global_step}",
                state.global_step,
                state.max_steps,
            )

        def on_train_end(self, training_args, state, control, **kwargs):
            report_progress("training", 95, "Optimizer steps complete")

    if backend == "directml":
        sft_config.__dict__["__cached__setup_devices"] = runtime_device

    trainer_class = DirectMLSFTTrainer if backend == "directml" else SFTTrainer
    report_progress("tokenization", 12, "Tokenizing training records")
    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=lora_config,
        args=sft_config,
        callbacks=[ProgressCallback()],
    )

    if resume_checkpoint:
        report_progress("resume", 18, f"Restoring checkpoint {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)
    report_progress("saving", 97, "Saving final LoRA adapter")
    trainer.save_model(str(output_dir))

    # Recorded so 'Merge adapter' (and later stages) can tell this apart from
    # a video/vision-language adapter without guessing from the base model
    # name.
    manifest_out = {"modality": "text", "base_model": args.model}
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest_out, indent=2), encoding="utf-8")

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
    report_progress("complete", 100, "Training complete")
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        report_progress("failed", current_percent, f"Training failed: {exc}")
        raise
