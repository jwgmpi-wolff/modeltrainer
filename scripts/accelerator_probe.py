#!/usr/bin/env python3
import json
import tempfile
from pathlib import Path


def test_gpu():
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        backend = "CUDA"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        device_name = torch.xpu.get_device_name(0)
        backend = "Intel XPU"
    else:
        try:
            import torch_directml

            if not torch_directml.is_available():
                return False, "DirectML unavailable"
            index = torch_directml.default_device()
            device = torch_directml.device(index)
            device_name = torch_directml.device_name(index).replace("\x00", "")
            backend = "DirectML"
        except ImportError:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = torch.device("mps")
                device_name = "Apple Silicon GPU"
                backend = "MPS"
            else:
                return False, "No CUDA, Intel XPU, DirectML, or MPS training device is available"
    base_weight = torch.randn((32, 32), device=device)
    lora_a = torch.nn.Parameter(torch.randn((32, 4), device=device) * 0.01)
    lora_b = torch.nn.Parameter(torch.zeros((4, 32), device=device))
    parameters = (lora_a, lora_b)
    optimizer_state = {
        parameter: {
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
        for parameter in parameters
    }
    beta1, beta2 = (0.9, 0.999)
    learning_rate = 1e-2
    epsilon = 1e-8
    inputs = torch.randn((8, 32), device=device)
    targets = torch.randn((8, 32), device=device)
    for step in range(1, 11):
        for parameter in parameters:
            parameter.grad = None
        loss = ((inputs @ (base_weight + lora_a @ lora_b) - targets) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            for parameter in parameters:
                gradient = parameter.grad
                state = optimizer_state[parameter]
                state["exp_avg"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )
                denominator = (
                    state["exp_avg_sq"]
                    .sqrt()
                    .div_((1 - beta2**step) ** 0.5)
                    .add_(epsilon)
                )
                parameter.addcdiv_(
                    state["exp_avg"],
                    denominator,
                    value=-(learning_rate / (1 - beta1**step)),
                )
    train_loss = float(loss.detach().cpu())
    if not train_loss > 0:
        raise RuntimeError(f"DirectML LoRA training produced invalid loss {train_loss}.")
    return True, f"{device_name} via {backend}; ten-step AdamW LoRA train_loss={train_loss:.6f}"


def test_qualcomm_npu():
    import numpy as np
    import onnx
    import onnxruntime as ort
    import onnxruntime_qnn
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["z"])],
        "model-trainer-npu-probe",
        [x, y],
        [z],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])
    model.ir_version = 9
    model_path = Path(tempfile.gettempdir()) / "model-trainer-npu-probe.onnx"
    onnx.save(model, model_path)

    ort.register_execution_provider_library(
        "QNNExecutionProvider", onnxruntime_qnn.get_library_path()
    )
    npu_devices = [
        device
        for device in ort.get_ep_devices()
        if device.ep_name == "QNNExecutionProvider"
        and "NPU" in str(device.device.type)
    ]
    if not npu_devices:
        return False, "QNN NPU device unavailable"

    options = ort.SessionOptions()
    options.add_provider_for_devices(npu_devices, {"backend_type": "htp"})
    options.enable_profiling = True
    session = ort.InferenceSession(str(model_path), sess_options=options)
    output = session.run(
        None,
        {
            "x": np.ones((1, 4), dtype=np.float32),
            "y": np.ones((1, 4), dtype=np.float32),
        },
    )[0]
    profile_path = Path(session.end_profiling())
    events = json.loads(profile_path.read_text(encoding="utf-8"))
    profile_path.unlink(missing_ok=True)
    model_path.unlink(missing_ok=True)
    npu_nodes = sum(
        1
        for event in events
        if event.get("cat") == "Node"
        and event.get("args", {}).get("provider") == "QNNExecutionProvider"
    )
    verified = bool(np.allclose(output, np.full((1, 4), 2.0))) and npu_nodes > 0
    return verified, f"QNNExecutionProvider ({npu_nodes} node on Hexagon NPU)"


def test_intel_npu():
    import numpy as np
    import openvino as ov

    core = ov.Core()
    npu_device = next((device for device in core.available_devices if device.upper().startswith("NPU")), None)
    if not npu_device:
        return False, "OpenVINO Intel NPU device unavailable"

    parameter = ov.opset13.parameter([1, 4], np.float32, name="x")
    constant = ov.opset13.constant(np.ones((1, 4), dtype=np.float32))
    model = ov.Model([ov.opset13.add(parameter, constant)], [parameter], "model-trainer-intel-npu-probe")
    compiled = core.compile_model(model, npu_device)
    output = compiled([np.ones((1, 4), dtype=np.float32)])[0]
    verified = bool(np.allclose(output, np.full((1, 4), 2.0)))
    return verified, f"OpenVINO {npu_device}"


def test_npu():
    failures = []
    for test in (test_qualcomm_npu, test_intel_npu):
        try:
            verified, provider = test()
            if verified:
                return verified, provider
            failures.append(provider)
        except Exception as exc:
            failures.append(str(exc))
    return False, "; ".join(failures)


def main():
    result = {
        "gpuVerified": False,
        "npuVerified": False,
        "gpuDevice": "Unavailable",
        "npuProvider": "Unavailable",
        "details": [],
    }
    try:
        result["gpuVerified"], result["gpuDevice"] = test_gpu()
        result["details"].append("Ten real AdamW LoRA updates completed on the selected GPU backend.")
    except Exception as exc:
        result["details"].append(f"GPU test failed: {exc}")
    try:
        result["npuVerified"], result["npuProvider"] = test_npu()
        result["details"].append(
            f"An inference graph executed through {result['npuProvider']} without CPU fallback."
        )
    except Exception as exc:
        result["details"].append(f"NPU test failed: {exc}")
    result["details"] = " ".join(result["details"])
    print(json.dumps(result))
    raise SystemExit(0 if result["gpuVerified"] or result["npuVerified"] else 1)


if __name__ == "__main__":
    main()
