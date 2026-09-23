#!/usr/bin/env python3
"""Real, honest GPU/accelerator backend detection for PyTorch.

This is shared by train_qlora.py (and can be reused by other scripts). It
does NOT guess based on vendor names — it actually imports torch and asks
each real backend whether it is usable right now, in this exact Python
environment, on this exact machine. Priority order (fastest/most capable
first):

  1. cuda      - NVIDIA GPUs, via torch's built-in CUDA support.
  2. directml  - AMD, Intel, NVIDIA, or Qualcomm GPUs on Windows, via
                 Microsoft's real "torch-directml" package (DirectX 12).
  3. xpu       - Intel GPUs (Arc/Xe), via intel-extension-for-pytorch.
  4. mps       - Apple Silicon GPUs, via torch's built-in MPS support.
  5. cpu       - No usable GPU backend found; falls back to the real CPU.

Only bitsandbytes 4-bit quantization (true "QLoRA") is CUDA-only today.
Every other backend still runs a REAL LoRA fine-tune on the real GPU, just
without 4-bit quantization, because bitsandbytes does not support those
backends. This module is honest about that distinction in `notes`.
"""

import sys


def detect():
    result = {
        "backend": None,
        "device_name": None,
        "torch_installed": False,
        "torch_version": None,
        "notes": [],
    }

    try:
        import torch
    except ImportError:
        result["notes"].append("The 'torch' package is not installed in this Python environment.")
        result["backend"] = None
        return result

    result["torch_installed"] = True
    result["torch_version"] = torch.__version__

    # 1. NVIDIA CUDA - real GPU-accelerated 4-bit QLoRA is possible here.
    try:
        if torch.cuda.is_available():
            result["backend"] = "cuda"
            result["device_name"] = torch.cuda.get_device_name(0)
            result["notes"].append("Real NVIDIA CUDA GPU detected. 4-bit quantized QLoRA training is supported.")
            return result
    except Exception as exc:  # pragma: no cover - defensive
        result["notes"].append(f"CUDA probe raised an error: {exc}")

    # 2. DirectML - real acceleration for AMD / Intel / Qualcomm / NVIDIA GPUs
    #    on Windows via DirectX 12. This is what makes "any GPU" real on
    #    Windows laptops that are not NVIDIA.
    if sys.platform == "win32":
        try:
            import torch_directml

            if torch_directml.is_available():
                device_index = torch_directml.default_device()
                try:
                    device_name = torch_directml.device_name(device_index)
                except Exception:
                    device_name = f"DirectML device {device_index}"
                result["backend"] = "directml"
                result["device_name"] = device_name
                result["notes"].append(
                    "Real DirectML GPU detected (works with AMD, Intel, Qualcomm, and NVIDIA GPUs on Windows). "
                    "bitsandbytes 4-bit quantization is CUDA-only, so this runs real LoRA fine-tuning in fp16 "
                    "on the real GPU instead of quantized QLoRA."
                )
                return result
        except ImportError:
            result["notes"].append(
                "torch-directml is not installed. Install it to use a non-NVIDIA GPU (AMD/Intel/Qualcomm) on "
                "Windows: pip install torch==2.4.1 torchvision==0.19.1 torch-directml==0.2.5.dev240914"
            )
        except Exception as exc:  # pragma: no cover - defensive
            result["notes"].append(f"DirectML probe raised an error: {exc}")

    # 3. Intel XPU (Arc/Xe GPUs) via intel-extension-for-pytorch.
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            result["backend"] = "xpu"
            result["device_name"] = torch.xpu.get_device_name(0)
            result["notes"].append("Real Intel XPU GPU detected via intel-extension-for-pytorch.")
            return result
    except Exception as exc:  # pragma: no cover - defensive
        result["notes"].append(f"Intel XPU probe raised an error: {exc}")

    # 4. Apple Silicon MPS.
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            result["backend"] = "mps"
            result["device_name"] = "Apple Silicon GPU (Metal Performance Shaders)"
            result["notes"].append("Real Apple Silicon GPU detected via MPS.")
            return result
    except Exception as exc:  # pragma: no cover - defensive
        result["notes"].append(f"MPS probe raised an error: {exc}")

    # 5. Nothing usable - honest CPU fallback.
    result["backend"] = "cpu"
    result["device_name"] = "CPU only"
    result["notes"].append("No usable GPU backend was found. Training will run on the CPU, which is real but slow.")
    return result


def get_torch_device(backend):
    """Return the real torch device object/string for the detected backend."""
    import torch

    if backend == "cuda":
        return torch.device("cuda")
    if backend == "directml":
        import torch_directml

        return torch_directml.device()
    if backend == "xpu":
        return torch.device("xpu")
    if backend == "mps":
        return torch.device("mps")
    return torch.device("cpu")


if __name__ == "__main__":
    import json

    print(json.dumps(detect(), indent=2))
