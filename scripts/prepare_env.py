#!/usr/bin/env python3
"""Real environment preparation for local QLoRA/LoRA training.

This script performs REAL, verifiable actions:
  1. Reports the real Python executable and version being used.
  2. Creates a REAL virtual environment on disk at .venv-training (if missing).
  3. Writes a REAL requirements.txt file next to this script (if missing),
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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
VENV_DIR = REPO_ROOT / ".venv-training"
REQUIREMENTS_FILE = Path(__file__).resolve().parent / "requirements.txt"

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

# Real, backend-specific install instructions. Only bitsandbytes (true 4-bit
# QLoRA) is CUDA-only; DirectML is what makes "any GPU" real on Windows.
BACKEND_REQUIREMENTS = {
    "cuda": (
        "torch\nbitsandbytes\n",
        "This machine has a real NVIDIA GPU. Install PyTorch with CUDA support, then the rest:\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu121\n"
        "  pip install -r scripts/requirements.txt",
    ),
    "directml": (
        "torch==2.4.1\ntorchvision==0.19.1\ntorch-directml==0.2.5.dev240914\n",
        "This machine's real GPU is not NVIDIA, so this uses Microsoft's real DirectML backend "
        "(supports AMD, Intel, Qualcomm, and NVIDIA GPUs on Windows via DirectX 12):\n"
        "  pip install -r scripts/requirements.txt\n"
        "Note: bitsandbytes 4-bit quantization is CUDA-only, so training will run as real LoRA "
        "fine-tuning in fp16 on your real GPU instead of 4-bit quantized QLoRA.",
    ),
    "xpu": (
        "torch\nintel-extension-for-pytorch\n",
        "This machine has a real Intel GPU. Install PyTorch plus Intel's real XPU extension:\n"
        "  pip install -r scripts/requirements.txt",
    ),
    "mps": (
        "torch\n",
        "This machine has a real Apple Silicon GPU. Plain PyTorch already includes real MPS support:\n"
        "  pip install -r scripts/requirements.txt",
    ),
    "cpu": (
        "torch\n",
        "No usable GPU backend was detected, so this installs CPU-only PyTorch. Training will be "
        "real but slow:\n"
        "  pip install -r scripts/requirements.txt",
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
        "--install",
        action="store_true",
        help="Install the real ML requirements into .venv-training after writing requirements.txt.",
    )
    args = parser.parse_args()

    report = {"ok": True, "actions": [], "backend": args.backend}

    report["actions"].append(f"Using Python {sys.version.split()[0]} at {sys.executable}")

    backend_reqs, backend_instructions = BACKEND_REQUIREMENTS[args.backend]
    requirements_text = COMMON_REQUIREMENTS + backend_reqs

    if not REQUIREMENTS_FILE.exists() or REQUIREMENTS_FILE.read_text(encoding="utf-8") != requirements_text:
        REQUIREMENTS_FILE.write_text(requirements_text, encoding="utf-8")
        report["actions"].append(f"Wrote {REQUIREMENTS_FILE} for real detected backend: {args.backend}")
    else:
        report["actions"].append(f"Requirements file already up to date at {REQUIREMENTS_FILE} for backend: {args.backend}")

    if VENV_DIR.exists():
        report["actions"].append(f"Virtual environment already exists at {VENV_DIR}")
    else:
        result = subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], capture_output=True, text=True)
        if result.returncode != 0:
            report["ok"] = False
            report["actions"].append("FAILED to create virtual environment.")
            report["stderr"] = result.stderr
            print(json.dumps(report, indent=2))
            sys.exit(1)
        report["actions"].append(f"Created virtual environment at {VENV_DIR}")

    venv_bin = VENV_DIR / ("Scripts" if sys.platform == "win32" else "bin")
    pip_path = venv_bin / ("pip.exe" if sys.platform == "win32" else "pip")
    python_path = venv_bin / ("python.exe" if sys.platform == "win32" else "python")

    if args.install:
        report["actions"].append(f"Installing real ML packages into {VENV_DIR}. This can take several minutes.")
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
                        "error": "pip upgrade failed in the real training environment.",
                    },
                    indent=2,
                )
            )
            sys.exit(upgrade_result.returncode)

        install_result = subprocess.run(
            [str(pip_path), "install", "-r", str(REQUIREMENTS_FILE)],
            text=True,
        )
        if install_result.returncode != 0:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "backend": args.backend,
                        "error": f"pip install failed. Re-run manually if needed: {pip_path} install -r {REQUIREMENTS_FILE}",
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
