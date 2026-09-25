#!/usr/bin/env python3
"""Real environment preparation for local QLoRA/LoRA training.

This script performs REAL, verifiable actions:
  1. Reports the real Python executable and version being used.
  2. Creates a REAL virtual environment on disk at .venv-training (if missing).
  3. Writes a REAL requirements-training.txt file in the writable workspace,
     tailored to the real GPU backend detected for THIS machine (NVIDIA
     CUDA, DirectML for AMD/Intel/Qualcomm GPUs on Windows, Intel XPU, or
     honest CPU-only) so the app can use whatever GPU is actually installed,
     not just NVIDIA.
  4. Optionally installs those requirements into the real .venv-training
     environment when --install is supplied.

It does not fake success: if the package install fails, this script exits with
a non-zero status and prints the real pip failure output.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from runtime_paths import WORKSPACE_ROOT

VENV_DIR = WORKSPACE_ROOT / ".venv-training"
REQUIREMENTS_FILE = WORKSPACE_ROOT / "requirements-training.txt"

# Separate venv/requirements for video/vision-language training. This pipeline
# needs transformers>=4.49.0 (Qwen2.5-VL support), which conflicts with the
# text pipeline's transformers==4.44.2 pin, so it gets its own isolated
# environment rather than risking breaking the tested text pipeline.
VIDEO_VENV_DIR = WORKSPACE_ROOT / ".venv-training-video"
VIDEO_REQUIREMENTS_FILE = WORKSPACE_ROOT / "requirements-video-training.txt"

COMMON_REQUIREMENTS = """transformers==4.44.2
accelerate==0.34.2
datasets==2.21.0
peft==0.12.0
trl==0.10.1
safetensors
sentencepiece
protobuf
tensorboard
rich
hf_xet
"""

# Deliberately does NOT include opencv-python, av (PyAV), decord, or
# qwen-vl-utils: none of these publish official Windows ARM64 wheels, so
# depending on them would make this stage fail with an obscure pip/wheel
# error on this exact machine. Frame extraction instead shells out to the
# real ffmpeg CLI binary, which does have real Windows ARM64 builds.
#
# transformers is capped below 5.0.0: verified by a real local install that
# transformers>=5.0 hard-requires torch>=2.5 and silently disables all
# PyTorch model loading ("PyTorch was not found") when only torch==2.4.1 is
# present, which is the pinned DirectML-compatible torch version below.
# Staying on the latest 4.x release keeps Qwen2-VL/Qwen2.5-VL support while
# actually working with the DirectML backend on this machine.
VIDEO_COMMON_REQUIREMENTS = """transformers>=4.49.0,<5.0.0
accelerate==0.34.2
peft==0.12.0
pillow
safetensors
sentencepiece
protobuf
rich
"""

# Real, backend-specific install instructions. Only bitsandbytes (true 4-bit
# QLoRA) is CUDA-only; DirectML is what makes "any GPU" real on Windows.
BACKEND_REQUIREMENTS = {
    "cuda": (
        "torch\nbitsandbytes\n",
        "This machine has a real NVIDIA GPU. Install PyTorch with CUDA support, then the rest:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "  pip install -r requirements-training.txt",
    ),
    "directml": (
        "torch==2.4.1\ntorchvision==0.19.1\ntorch-directml==0.2.5.dev240914\n",
        "This machine's real GPU is not NVIDIA, so this uses Microsoft's real DirectML backend "
        "(supports AMD, Intel, Qualcomm, and NVIDIA GPUs on Windows via DirectX 12):\n"
        "  pip install -r requirements-training.txt\n"
        "Note: bitsandbytes 4-bit quantization is CUDA-only, so training will run as real LoRA "
        "fine-tuning in fp16 on your real GPU instead of 4-bit quantized QLoRA.",
    ),
    "xpu": (
        "torch\nintel-extension-for-pytorch\n",
        "This machine has a real Intel GPU. Install PyTorch plus Intel's real XPU extension:\n"
        "  pip install -r requirements-training.txt",
    ),
    "mps": (
        "torch\n",
        "This machine has a real Apple Silicon GPU. Plain PyTorch already includes real MPS support:\n"
        "  pip install -r requirements-training.txt",
    ),
    "cpu": (
        "torch\n",
        "No usable GPU backend was detected, so this installs CPU-only PyTorch. Training will be "
        "real but slow:\n"
        "  pip install -r requirements-training.txt",
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=list(BACKEND_REQUIREMENTS.keys()),
        default="cpu",
        help="Real GPU backend detected for this machine by the Node backend's system check.",
    )
    parser.add_argument(
        "--modality",
        choices=["text", "video"],
        default="text",
        help="Which pipeline environment to prepare: 'text' (default, causal LM) or 'video' (vision-language).",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install the real ML requirements into the venv after writing the requirements file.",
    )
    args = parser.parse_args()

    is_video = args.modality == "video"
    venv_dir = VIDEO_VENV_DIR if is_video else VENV_DIR
    requirements_file = VIDEO_REQUIREMENTS_FILE if is_video else REQUIREMENTS_FILE
    common_requirements = VIDEO_COMMON_REQUIREMENTS if is_video else COMMON_REQUIREMENTS

    report = {"ok": True, "actions": [], "backend": args.backend, "modality": args.modality}

    report["actions"].append(f"Using Python {sys.version.split()[0]} at {sys.executable}")

    if is_video:
        ffmpeg_bin = shutil.which("ffmpeg")
        if ffmpeg_bin:
            report["actions"].append(f"Found real ffmpeg at {ffmpeg_bin} (required for video frame extraction).")
        else:
            report["actions"].append(
                "WARNING: ffmpeg was not found on PATH. Video training needs a real ffmpeg binary to extract "
                "frames. Download a Windows build (including winarm64 builds) from "
                "https://github.com/BtbN/FFmpeg-Builds/releases, extract it, and add its bin/ folder to PATH. "
                "The 'Train (video)' stage will fail with this same message until ffmpeg is installed."
            )

    backend_reqs, backend_instructions = BACKEND_REQUIREMENTS[args.backend]
    requirements_text = common_requirements + backend_reqs

    if not requirements_file.exists() or requirements_file.read_text(encoding="utf-8") != requirements_text:
        requirements_file.write_text(requirements_text, encoding="utf-8")
        report["actions"].append(f"Wrote {requirements_file} for real detected backend: {args.backend}")
    else:
        report["actions"].append(f"Requirements file already up to date at {requirements_file} for backend: {args.backend}")

    if venv_dir.exists():
        report["actions"].append(f"Virtual environment already exists at {venv_dir}")
    else:
        result = subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], capture_output=True, text=True)
        if result.returncode != 0:
            report["ok"] = False
            report["actions"].append("FAILED to create virtual environment.")
            report["stderr"] = result.stderr
            print(json.dumps(report, indent=2))
            sys.exit(1)
        report["actions"].append(f"Created virtual environment at {venv_dir}")

    venv_bin = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    pip_path = venv_bin / ("pip.exe" if sys.platform == "win32" else "pip")
    python_path = venv_bin / ("python.exe" if sys.platform == "win32" else "python")

    if args.install:
        report["actions"].append(f"Installing real ML packages into {venv_dir}. This can take several minutes.")
        print(json.dumps(report, indent=2), flush=True)

        upgrade_result = subprocess.run(
            [str(python_path), "-m", "pip", "install", "--upgrade", "pip"],
            text=True,
        )
        if upgrade_result.returncode != 0:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "backend": args.backend,
                        "modality": args.modality,
                        "error": "pip upgrade failed in the real training environment.",
                    },
                    indent=2,
                )
            )
            sys.exit(upgrade_result.returncode)

        install_result = subprocess.run(
            [str(pip_path), "install", "-r", str(requirements_file)],
            text=True,
        )
        if install_result.returncode != 0:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "backend": args.backend,
                        "modality": args.modality,
                        "error": f"pip install failed. Re-run manually if needed: {pip_path} install -r {requirements_file}",
                    },
                    indent=2,
                )
            )
            sys.exit(install_result.returncode)

        report["actions"].append(f"Installed real requirements using {pip_path}")
    else:
        report["actions"].append(
            f"Environment is prepared. Heavy ML packages are NOT installed yet. {backend_instructions} "
            f"(using {pip_path})"
        )

    print(json.dumps(report, indent=2))
    sys.exit(0)


if __name__ == "__main__":
    main()
