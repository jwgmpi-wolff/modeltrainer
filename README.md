# Local Model Lab

Local Model Lab is a real local fine-tuning pipeline manager for Hugging Face causal language models. It has a React/Vite frontend and an Express backend that runs real Python training, merge, GGUF conversion, quantization, Ollama registration, and smoke-test commands.

The project standard is explicit: **do not build mockups for pipeline behavior**. Every run button must execute real local work, stream real process output, and fail honestly when a dependency, model, GPU backend, or generated artifact is missing.

## What this app does

- Checks this laptop's real CPU, RAM, Python, Ollama, WSL, and GPU inventory.
- Chooses a real training backend:
  - NVIDIA CUDA when available for true 4-bit QLoRA.
  - DirectML on Windows for AMD, Intel, Qualcomm, and NVIDIA GPUs when CUDA is not available.
  - Intel XPU, Apple MPS, or CPU fallback where applicable.
- Uploads or selects real `.jsonl` training data.
- Validates the dataset line by line with a real Python validator.
- Prepares a real `.venv-training` virtual environment and installs real ML dependencies.
- Runs real LoRA/QLoRA fine-tuning.
- Saves complete Hugging Face Trainer checkpoints every 10 optimizer steps,
  retains the 10 newest checkpoints, and safely resumes adapter, optimizer,
  scheduler, RNG, and global-step state after compatibility validation.
- Streams structured, determinate training-stage and optimizer-step progress and
  persists the complete process output to `output/<model-name>/training.log`.
- Separately discovers and activates a completed adapter or checkpoint adapter
  for post-training agent/inference use; this selection is independent of the
  base model being fine-tuned.
- Recursively crawls a selected local folder for model artifacts and can move
  every discovered model into one destination without overwriting existing
  files. Each model receives its own subfolder under that destination.
  Multi-file Hugging Face, adapter, ONNX, TensorFlow SavedModel, OpenVINO, and
  compiled Core ML bundles remain intact; standalone GGUF, ONNX, ORT, Core ML,
  PyTorch, SafeTensors, and TensorFlow Lite files are supported.
- Runs a real accelerator self-test: a DirectML LoRA update on the Windows GPU
  plus a profiled ONNX graph on the Qualcomm QNN/Hexagon NPU.
- Evaluates whether adapter artifacts exist.
- Merges the adapter into a real Hugging Face model directory.
- Converts to GGUF with a real `llama.cpp` checkout.
- Quantizes with a real `llama.cpp` quantizer.
- Registers the model in real local Ollama.
- Smoke-tests the registered Ollama model with a real prompt.

## Repository layout

```text
modeltrainer/
  src/                 React/Vite frontend
  server/              Express API and Server-Sent Events pipeline runner
  scripts/             Python validation/training/merge/GGUF/Ollama stages
  samples/             Small checked-in JSONL examples
  uploads/             Local uploaded datasets; ignored by git
  output/              Local LoRA adapter output; ignored by git
  merged/              Local merged HF model output; ignored by git
  gguf/                Local GGUF files; ignored by git
  ollama/              Local Modelfiles; ignored by git
  .venv-training/      Local ML Python environment; ignored by git
```

## Prerequisites

Install these before recreating the app:

1. **Node.js 20+** and npm.
2. **Git** for source-based installation.
3. **Ollama** if you want the final register/smoke-test stages.
4. **CMake and a C++ compiler** if you want to build `llama.cpp` locally for GGUF conversion and quantization.
5. A Hugging Face account/token if the selected base model requires authentication.

Python does not have to be installed before the desktop app. During environment
preparation, Model Trainer selects a compatible x64 Python 3.10-3.12 or installs
a private, signature-verified x64 Python 3.11.9. On Windows ARM, x64 Python runs
under emulation and is required because PyTorch and DirectML do not publish
Windows ARM64 training wheels.

Windows GPU notes:

- CUDA/NVIDIA can use true 4-bit QLoRA.
- AMD, Intel, Qualcomm, and non-CUDA NVIDIA GPUs use DirectML. This is still real GPU-backed LoRA fine-tuning, but not bitsandbytes 4-bit QLoRA because bitsandbytes is CUDA-only.
- Large 7B-class models can crash or take impractically long on DirectML fp16/fp32. The UI defaults to Qwen 2.5 Coder 1.5B Instruct for a safer starting point.

## Install

Clone the standalone repository, then install its dependencies:

```powershell
git clone https://github.com/jwgmpi-wolff/modeltrainer.git
cd modeltrainer
npm run install:all
```

This installs:

- Frontend dependencies from `package.json`.
- Backend dependencies from `server/package.json`.

The ML Python environment is intentionally created by the real **Prepare environment** pipeline stage, because the required packages depend on the hardware backend detected on the machine running the app.

## Launch the clickable Windows app

Build the Windows desktop app and create shortcuts:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-to-local-machine.ps1
```

This creates `release\win-*-unpacked\Local Model Lab.exe`, adds **Local Model Lab** shortcuts on the desktop and Start menu, and starts the app. Launch the trainer later by double-clicking either shortcut. The app opens in its own desktop window and owns its local training service; it does not open a browser or require terminal windows to remain open.

The desktop app writes generated datasets, environments, adapters, merged models, and GGUF files under the current Windows user's Local Model Lab application-data workspace so packaged program files remain read-only.

### Install on another Windows computer from this repository

The standalone repository is private. On the remote computer, install GitHub
CLI and Node.js 20+, authenticate to GitHub, then clone and deploy:

```powershell
winget install --id GitHub.cli
winget install --id OpenJS.NodeJS.LTS
gh auth login
Set-Location $HOME
gh repo clone jwgmpi-wolff/modeltrainer
Set-Location .\modeltrainer
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\deploy-to-local-machine.ps1
```

If those prerequisites are already installed, start at `gh auth login`. The
deployment command works regardless of the clone's parent folder. It builds the
app, creates Desktop and Start Menu shortcuts, launches Local Model Lab, and
fails explicitly if the launched process exits immediately.

### Build registered Windows installers

Create a Windows ARM64 setup executable:

```powershell
npm run install:all
npm run desktop:installer
```

Create an Intel/AMD x64 setup executable with:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\build-windows-installer.ps1 -Architecture x64
```

The generated `release\Local Model Lab-Setup-1.0.1-<architecture>.exe`
installers include the complete packaged desktop application and backend. Each
installer:

- registers Local Model Lab in Windows **Installed apps** with an uninstaller;
- applies the Local Model Lab icon to the executable and shortcuts;
- creates Desktop and Start Menu shortcuts;
- launches the application after installation.

Use the `x64` installer for Intel/AMD Windows computers and the `arm64`
installer for Snapdragon/Windows ARM computers. When training is first
prepared, the app selects a compatible x64 Python 3.10-3.12 interpreter. If
none exists, it downloads the official Python Software Foundation-signed x64
Python 3.11.9 installer and creates a private interpreter under the app
workspace. This is required because Windows PyTorch and DirectML publish x64
wheels, while Windows ARM runs those tools successfully under x64 emulation.

## Developer mode

The browser-based Vite workflow remains available only for development. Open two terminals.

Terminal 1 - backend:

```powershell
cd C:\Users\jerrywolff\Documents\.git\modeltrainer
npm run dev:server
```

Backend URL:

```text
http://127.0.0.1:8787
```

Terminal 2 - frontend:

```powershell
cd C:\Users\jerrywolff\Documents\.git\modeltrainer
npm run dev
```

Frontend URL:

```text
http://127.0.0.1:3000
```

## Use the app

1. Confirm the header says `Backend: online`.
2. Confirm the `Real system check` card reports your actual hardware.
3. Pick a model. Start with `Qwen 2.5 Coder 1.5B Instruct` for DirectML systems.
4. Pick a `.jsonl` data file with **Browse files**, or choose a real sample from `samples/`.
5. Keep **Quick smoke training** enabled for the first proof run.
6. Choose **Start new training**, or click **Refresh** and select a compatible
   checkpoint to resume.
7. Press **Run now**.
8. Watch the determinate progress meter and **Real command output**. Progress
   comes from structured trainer callbacks rather than elapsed-time estimates.
9. After training, use **Post-training Agent / Inference model** to activate the
   adapter an agent should use. The **Base model** field remains a training-only
   choice.
10. To gather models spread across the machine, enter a source and destination
    under **Find and consolidate local models**, scan to preview every match,
    then choose **Move all found models**. Symbolic links are skipped, the
    destination subtree is excluded from the crawl, and every model remains
    distinct in its own subfolder. Duplicate names receive numbered suffixes
    instead of replacing existing files.

## Dataset format

Training data must be JSON Lines (`.jsonl`), one JSON object per line. The validator expects chat-style records that can be serialized into text by the training script. Keep the first run small while validating hardware.

Use the sample files in `samples/` as seed examples:

- `azure-powershell-basics.jsonl`
- `azure-powershell-commands.jsonl`
- `azure-powershell-validation.jsonl`

## Video / vision-language fine-tuning (experimental)

The app can also fine-tune a real vision-language model (Qwen2-VL / Qwen2.5-VL) on video clips, in addition to the
text-only pipeline above. This is a genuinely separate pipeline with its own model classes, venv, and requirements
file — it is not a simulation or a "describe the video as text" shortcut.

**Requirements specific to this pipeline:**

- **ffmpeg must be installed and on PATH.** Frame extraction shells out to the real `ffmpeg` CLI binary rather than
  a pip-installed video-decoding library, because `opencv-python`, `PyAV` (`av`), and `decord` do not currently
  publish official Windows **ARM64** wheels (only experimental/community ones). `ffmpeg` itself does have real,
  actively maintained Windows ARM64 builds:
  - Windows (including Snapdragon X / ARM64 laptops): download a build from
    [BtbN/FFmpeg-Builds releases](https://github.com/BtbN/FFmpeg-Builds/releases) — pick a `winarm64` asset on ARM64,
    or `win64` on an Intel/AMD PC — then add its `bin` folder to your PATH.
  - macOS: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg` (or your distro's package manager)
  - The "Prepare environment" and "Train (video)" stages both check for `ffmpeg` and fail with this exact
    installation guidance if it is missing — nothing pretends to succeed without it.
- A **separate Python virtual environment** (`.venv-training-video`, with `requirements-video-training.txt`) is used
  instead of the text pipeline's `.venv-training`, because Qwen2.5-VL needs `transformers>=4.49.0` while the text
  pipeline pins `transformers==4.44.2`. The video pipeline pins `transformers>=4.49.0,<5.0.0` — real testing found
  that `transformers>=5.0.0` requires `torch>=2.5`, but `torch-directml` (the DirectML backend) only supports
  `torch==2.4.1`, so unpinned installs silently disabled all PyTorch model loading in `transformers`. Run "Prepare
  environment" once with video mode selected to create it.

**Manifest format:** a `.jsonl` file, one JSON object per line, referencing real video clip files by name:

```json
{"video": "clip1.mp4", "messages": [{"role": "user", "content": "Describe what happens in this clip."}, {"role": "assistant", "content": "A person walks across a room and picks up a red mug."}], "fps": 1.0, "max_frames": 8}
```

Supported video extensions: `.mp4 .mov .mkv .webm .avi .m4v .mpeg .mpg`. In the app, switch **Training data type** to
**Video (vision-language)**, browse for the manifest and every real video file it references, click **Upload
dataset**, then optionally **Check this dataset** (this also probes each file with `ffprobe`, if installed, for real
duration/resolution).

**Known limitations of this pipeline (by design, not oversights):**

- **Batch size is fixed at 1.** Different videos produce different numbers of frames/tokens, so naive batch
  stacking isn't implemented; use **Grad. steps** (gradient accumulation) to reach a larger effective batch size.
- **GGUF conversion, quantization, Ollama registration, and the smoke test are not available for video models.**
  `llama.cpp` does not yet have an export path for these vision-language architectures' vision towers. The
  "Convert to GGUF" stage honestly refuses with this explanation instead of producing a broken file, and those four
  stages are grayed out automatically in the UI when video mode is selected. The merged model directory (under
  `merged/<name>`) is still directly usable with the `transformers` library (`AutoProcessor` + the model class
  recorded in that folder's `training_manifest.json`).
- **Only Qwen2-VL / Qwen2.5-VL 2B/3B are realistic on this hardware.** The same ≥6.5B-parameter guard used by the
  text pipeline applies here too, with extra vision-tower memory overhead factored in — 7B video models need a real
  NVIDIA CUDA GPU.
- **DirectML GPUs fall back to CPU for video training.** Qwen2-VL/Qwen2.5-VL's vision tower patch-embedding layer
  uses a real 3D convolution (`nn.Conv3d`), and real testing found that `torch-directml` does not support it —
  it raises `RuntimeError: input must be 4-dimensional` even on a correctly-shaped 5D tensor. Rather than crash
  mid-training or silently pretend GPU acceleration is happening, "Train (video)" runs a real preflight probe (an
  actual tiny `Conv3d` forward pass on the detected device) before training starts. If that probe fails, it prints
  an honest warning and falls back to the real CPU for the whole run instead. Training still completes correctly
  (a real adapter is produced), just much more slowly than a GPU-accelerated text run on the same hardware. This
  only affects DirectML; NVIDIA CUDA GPUs support Conv3d and are unaffected.

## Real smoke result observed on this machine

On a Snapdragon X / Qualcomm Adreno DirectML laptop, the real quick-smoke run with Qwen 2.5 Coder 1.5B:

- Validated 1231 records.
- Installed and used real DirectML dependencies.
- Completed one LoRA training step.
- Took about **63 minutes** for that single step.
- Produced a real adapter under `output/azure-powershell-assistant`.
- Merged successfully under `merged/azure-powershell-assistant`.
- Failed honestly at GGUF conversion because `llama.cpp/convert_hf_to_gguf.py` was not present.

This proves the pipeline is real, but it also shows that DirectML plus a 1.5B model can be too slow for practical iteration on this hardware. Use smaller models/settings for fast tests.

On the same Snapdragon X / Qualcomm Adreno DirectML laptop, the real video pipeline smoke test with
`Qwen/Qwen2-VL-2B-Instruct` on two synthetic test clips:

- Validated the manifest and both video files with real `ffprobe` calls.
- Installed a real, separate video-training venv (`transformers` pinned to `>=4.49.0,<5.0.0` — see the note below).
- Extracted real frames from each clip with `ffmpeg` and built a real prompt via `AutoProcessor.apply_chat_template`.
- Ran the real preflight Conv3d probe, which failed on the DirectML device (`RuntimeError: input must be
  4-dimensional`), so the run honestly fell back to CPU as described above.
- Completed one real LoRA training step on CPU (loss `4.1969`).
- Produced a real adapter, tokenizer files, and `training_manifest.json` (`modality: "video"`) under
  `output/video-smoke-test`.

This confirms the video pipeline's manifest validation, frame extraction, processor/tokenization, and training loop
are all genuinely correct end-to-end — the only real issue found was DirectML's lack of Conv3d support, which is now
handled with an honest CPU fallback instead of a crash.

## GPU training and NPU inference acceleration

Model Trainer follows the hardware pattern verified by Tax Intelligence Studio:

- NVIDIA CUDA performs real 4-bit QLoRA.
- Windows AMD, Intel, Qualcomm, and non-CUDA NVIDIA GPUs perform real LoRA
  training through DirectML.
- Intel XPU and Apple MPS are used where their PyTorch runtimes are available.
- Qualcomm Hexagon NPUs are verified with the ONNX Runtime QNN provider.
- Intel NPUs are verified with OpenVINO's `NPU` device.
- NPUs accelerate inference and validation, not optimizer-based weight
  training. Training remains on a GPU or CPU.

The accelerator self-test performs ten real AdamW LoRA updates and refuses
silent CPU fallback. It separately executes a graph on a detected Qualcomm or
Intel NPU provider.

## Qualcomm / Snapdragon NPU deployment and INT8 quantization (`scripts/deploy_to_npu.py`)

This is a **standalone script**, not a pipeline stage in the app UI. Run it manually after a merge:

```powershell
python scripts\deploy_to_npu.py --model-dir merged\azure-powershell-assistant --output-name azure-powershell-assistant
```

### How it taps into the Snapdragon NPU / GPU

- **NPU**: the script exports the merged Hugging Face model to real ONNX (via the Optimum CLI), then loads it with
  `onnxruntime` and registers the real **QNN Execution Provider** (`onnxruntime-qnn`'s bundled Qualcomm QNN/Hexagon
  backend libraries). `ort.get_ep_devices()` is used to find the actual on-device NPU entry
  (`QNNExecutionProvider` + a device whose type reports `NPU`) — this is the real, supported way to target the
  Snapdragon Hexagon NPU from onnxruntime on Windows; there is no separate "Snapdragon SDK" call needed once
  `onnxruntime-qnn` is installed.
- **GPU (DirectML)**: for LoRA training and inference elsewhere in this app, `torch-directml` is used instead — this
  is the real, generic way to reach any Windows GPU (AMD, Intel, Qualcomm Adreno, or non-CUDA NVIDIA) through
  DirectX 12, and is what backs the "Windows GPU notes" section above.
- Both paths are real hardware execution providers, not simulated/mocked backends — if the hardware or driver isn't
  present, the script fails honestly instead of pretending to accelerate.

### Real INT8 quantization stage added this round

Before verifying NPU placement, the script now runs a **real static QDQ INT8 quantization** pass
(`quantize_to_int8()`), tailored for the QNN/Hexagon compiler:

1. `qnn_preprocess_model()` fuses/prepares the ONNX graph for QNN.
2. A real `CalibrationDataReader` (`_build_calibration_reader()`) runs real forward passes through the model with
   varying sequence-length and KV-cache-length sample inputs, so calibration statistics come from genuine
   activations, not fabricated ranges.
3. `get_qnn_qdq_config()` + `quantize()` produce a real QDQ INT8 model using `QuantType.QUInt8` for both weights and
   activations — this is the standard, most portable QNN/HTP quantization configuration.
4. `_describe_kv_cache_inputs()` reads the model's own ONNX input signature (layer count, KV head count, head dim)
   directly from the exported graph instead of hardcoding it, so this works across different model sizes/architectures.

`verify_on_npu()` was also upgraded to run a **real `sess.run()` forward pass** (with correctly-shaped dummy
KV-cache inputs) before reading onnxruntime's structured profiling JSON, because session construction alone only
emits `"Session"`-category profiling events — only an actual executed inference emits the `"cat": "Node"` events
needed to see genuine per-node NPU-vs-CPU placement.

### Real result observed on this machine

On the same Snapdragon X / Qualcomm Adreno laptop, running the full pipeline against the `azure-powershell-assistant`
merged model produced this **honest** real finding, reproduced identically at three different precision levels:

| Quantization level | Result on real Hexagon NPU |
|---|---|
| fp32 (no quantization) | `QNN_GRAPH_ERROR_INVALID_TENSOR` (code `6004`), 0 of ~5,700+ nodes placed on NPU |
| Dynamic weight-only INT8 | Same error, same 0-node NPU placement |
| Static QDQ INT8 (new this round) | Same error, same 0-node NPU placement |

In every case onnxruntime **silently falls back to full CPU execution** and inference still completes correctly
(e.g. `logits shape: (1, 4, 151936)`), because onnxruntime treats EP placement failures as non-fatal by design. The
`int8/` quantized model itself was produced successfully (a real ~4 MB `model.onnx` + ~1.8 GB external-data file for
this 1.5B-parameter model) — quantization is not the blocker. The blocker is that this specific model's ONNX graph
(most likely something in how the KV-cache / dynamic-shape inputs are represented) is not currently compatible with
this machine's QNN/HTP compiler, independent of numeric precision. This is reported honestly in the script's JSON
output rather than being reported as a success.

**No screenshots are included for this stage** because it is a CLI script with JSON output, not a GUI feature of the
app — the table above and the JSON `note` field the script prints are the real, honest proof of what actually
happened on this hardware.

## GGUF / Ollama setup

The conversion, quantization, Ollama registration, and smoke-test stages require real local tooling.

Clone and build `llama.cpp` inside this project:

```powershell
cd C:\Users\jerrywolff\Documents\.git\modeltrainer
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
cmake -B build
cmake --build build --config Release
```

Then re-run failed stages in the UI:

1. Convert to GGUF
2. Quantize GGUF
3. Register in Ollama
4. Smoke test model

The app looks for:

- `modeltrainer\llama.cpp\convert_hf_to_gguf.py`
- `modeltrainer\llama.cpp\build\bin\llama-quantize.exe`
- `ollama` on PATH

## Build and deploy

The app **does not deploy itself from inside the UI**. Build the desktop launcher from this repository, then launch it by clicking the generated executable.

Deployment starts from this repository on the local machine.

[![Deploy to Local Machine](https://img.shields.io/badge/Deploy_to_Local_Machine-PowerShell-blue?logo=powershell)](scripts/deploy-to-local-machine.ps1)

The button above opens the local deployment script in the repo. Run it from PowerShell to install dependencies, build the desktop application, and open the generated launcher.

### One-command local machine deployment

From a standalone clone:

```powershell
git clone https://github.com/jwgmpi-wolff/modeltrainer.git
cd modeltrainer
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-to-local-machine.ps1
```

From a new monorepo clone, initialize its Model Trainer submodule:

```powershell
git clone --recurse-submodules https://github.com/jwgmpi-wolff/AppsandScripts.git
cd AppsandScripts\modeltrainer
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-to-local-machine.ps1
```

The installer works from any location. It first uses the checkout containing
the script, then searches the current folder and its parents. If no checkout is
available (including when the script body is run directly), it clones the
standalone repository into
`%LOCALAPPDATA%\Local Model Lab\source`. The old
`jerrywolff_microsoft/modeltrainer` URL is obsolete.

You can also run the npm wrapper after dependencies are available:

```powershell
npm run deploy:local
```

What the local deployment automation really does:

1. Verifies `node` and `npm` are on PATH.
2. Runs `npm run install:all`.
3. Closes only Local Model Lab processes running from this checkout's
   `release` folder so Windows does not lock the previous package.
4. Builds the React frontend.
5. Packages the frontend and backend together as a Windows desktop application.
6. Creates `release\win-*-unpacked\Local Model Lab.exe`.
7. Creates desktop and Start menu shortcuts to that executable.
8. Opens the desktop app.

The training pipeline still creates and installs `.venv-training` during the app's real **Prepare environment** stage because the ML requirements depend on the detected hardware backend.

To build the launcher without opening it:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-to-local-machine.ps1 -BuildOnly
```

To run the desktop shell directly from a development checkout:

```powershell
npm run desktop
```

### Local production preview

```powershell
npm run build
npm run preview
```

The static frontend alone is not the complete app. The packaged desktop application starts and owns the backend automatically.

### Static frontend deployment

Run:

```powershell
npm run deploy:static
```

Deploy the generated `dist/` folder to any static host. This is only useful when the user also has the backend running locally or you update `API_BASE` for a hosted backend.

### GitHub Pages automatic action

GitHub Pages can host only the static frontend. It cannot provide the local GPU/Python/Ollama backend. To use GitHub Pages, add a workflow in `.github/workflows/deploy-pages.yml`, commit it, push it, and enable GitHub Pages using GitHub Actions.

### Azure Static Web Apps automatic files

Azure Static Web Apps can host only the static frontend unless you separately design a hosted API. This app's real training backend is intended for a local workstation because it needs local GPU/Python/Ollama access.

If you still want a static Azure frontend, create/review `azure.yaml`, `infra/main.bicep`, and `public/staticwebapp.config.json`, then run:

```powershell
azd up
```

### Backend deployment

The backend launches local child processes and accesses local GPUs, Python, Ollama, and files. For that reason, the intended deployment is local workstation execution, not serverless hosting.

If hosting the backend on a VM or workstation:

1. Install Node.js, Python 3.11, Ollama, GPU drivers, and optional `llama.cpp`.
2. Run:

   ```powershell
   cd C:\path\to\modeltrainer
   npm run install:all
   npm run dev:server
   ```

3. Serve the frontend with `npm run preview` or a static host configured to call that backend.

## Standards and requirements

- No simulated pipeline success.
- No timer-only fake progress.
- Surface real stdout/stderr in the UI.
- Every failed prerequisite must fail the stage with a clear remediation message.
- Do not commit local datasets, virtual environments, adapters, merged models, GGUF files, or Modelfiles.
- Keep file-picking behavior as real file manager upload/selection behavior.
- Keep sample selection available from repository data.
- Keep "Other model" support so any Hugging Face model ID can be tried.
- Default to safe settings for the detected hardware.
- Use quick smoke mode before full training.
- Do not add an in-app self-deploy button. First deployment belongs in this README and repo automation because a non-installed app cannot deploy itself.

## Full VS Code / Copilot prompt to recreate this app

Use this prompt in VS Code with Copilot to recreate the project from scratch:

```text
Build a real local model training lab, not a mockup. Use React + Vite for the frontend and Node/Express for a local backend. The app must manage a real local LoRA/QLoRA fine-tuning pipeline for Hugging Face causal language models and must stream real command output to the UI.

Hard requirements:
- Do not fake training, validation, conversion, quantization, Ollama registration, progress, or success.
- Every button must call a real local backend endpoint or run a real local command.
- If a dependency is missing, fail honestly with a clear remediation message.
- Use a real file picker for dataset upload. The selected file must be written to disk on the backend.
- Provide a sample dataset picker that reads real .jsonl sample files from the repo.
- Provide base model choices for Qwen 2.5 Coder 7B, 3B, 1.5B, Llama 3.1 8B, Mistral 7B, Phi 3.5 Mini, and an "Other Hugging Face model ID" input.
- Default to Qwen 2.5 Coder 1.5B for broad hardware compatibility.
- Add quick smoke training, enabled by default, that runs real training with max_steps=1, sample_size=8, max_length capped at 512, and grad_accum=1.
- Show real system checks for CPU, RAM, GPUs, Python, Ollama, and WSL.
- Detect GPU backends in this order: CUDA, DirectML, Intel XPU, Apple MPS, CPU.
- CUDA should use bitsandbytes 4-bit QLoRA when available.
- Non-CUDA GPU backends must still run real LoRA fine-tuning and must clearly say they are not using 4-bit QLoRA because bitsandbytes is CUDA-only.
- DirectML on Windows must support AMD, Intel, Qualcomm, and NVIDIA GPUs through torch-directml.
- Guard against 7B-class models on non-CUDA backends with a clear error instead of letting native DirectML crash.
- Validate JSONL training data with Python and handle UTF-8 BOM files correctly.
- Use Server-Sent Events to stream stage output and status.
- Pipeline stages: validate dataset, prepare environment, train adapter, evaluate adapter, merge adapter, convert to GGUF, quantize GGUF, register in Ollama, smoke test Ollama model.
- Conversion must require a real llama.cpp checkout with convert_hf_to_gguf.py.
- Quantization must require a real llama.cpp quantize binary.
- Ollama registration must write a real Modelfile and run ollama create.
- Smoke test must run ollama run and verify a non-empty response.
- Add Windows-safe upload behavior: if re-uploading the same filename, delete/copy fallback around fs.renameSync EPERM issues.
- Add a README with install, run, deploy, llama.cpp setup, project standards, and this exact recreation prompt.
- Add a README "Deploy to Local Machine" button/link and a real `scripts/deploy-to-local-machine.ps1` automation script. Do not add an in-app self-deploy button, because the app cannot deploy itself before it exists on the desktop.
- Add a .gitignore that excludes node_modules, .venv-training, uploads, output, merged, gguf, ollama, llama.cpp, dist, logs, and .env files.

Implementation details:
- Frontend: React, Vite, lucide-react, framer-motion, dark UI.
- Backend: Express, cors, multer, Server-Sent Events.
- Python scripts:
  - validate_dataset.py
  - prepare_env.py
  - gpu_backend.py
  - train_qlora.py
  - evaluate_adapter.py
  - merge_adapter.py
  - convert_to_gguf.py
  - quantize_gguf.py
  - register_ollama.py
  - smoke_test.py
- Use npm scripts:
  - install:all
  - dev
  - dev:frontend
  - dev:server
  - build
  - preview
  - deploy:static
  - check

Validation:
- npm run build must pass.
- Node backend syntax must be valid.
- Python scripts must compile.
- A quick smoke run must show real stdout/stderr and write real artifacts or fail with a real actionable error.
```
