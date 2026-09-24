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
2. **Python 3.11** available as `python` on PATH.
3. **Git**.
4. **Ollama** if you want the final register/smoke-test stages.
5. **CMake and a C++ compiler** if you want to build `llama.cpp` locally for GGUF conversion and quantization.
6. A Hugging Face account/token if the selected base model requires authentication.

Windows GPU notes:

- CUDA/NVIDIA can use true 4-bit QLoRA.
- AMD, Intel, Qualcomm, and non-CUDA NVIDIA GPUs use DirectML. This is still real GPU-backed LoRA fine-tuning, but not bitsandbytes 4-bit QLoRA because bitsandbytes is CUDA-only.
- Large 7B-class models can crash or take impractically long on DirectML fp16/fp32. The UI defaults to Qwen 2.5 Coder 1.5B Instruct for a safer starting point.

## Install

From this folder:

```powershell
cd C:\Users\jerrywolff\Documents\.git\modeltrainer
npm run install:all
```

This installs:

- Frontend dependencies from `package.json`.
- Backend dependencies from `server/package.json`.

The ML Python environment is intentionally created by the real **Prepare environment** pipeline stage, because the required packages depend on the hardware backend detected on the machine running the app.

## Run locally

Open two terminals.

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
6. Press **Run now**.
7. Watch **Real command output**. The UI status is driven by real child-process exit codes.

## Dataset format

Training data must be JSON Lines (`.jsonl`), one JSON object per line. The validator expects chat-style records that can be serialized into text by the training script. Keep the first run small while validating hardware.

Use the sample files in `samples/` as seed examples:

- `azure-powershell-basics.jsonl`
- `azure-powershell-commands.jsonl`
- `azure-powershell-validation.jsonl`

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

The app **does not deploy itself from inside the UI**. If it is not already installed and running on a desktop, there is no app process that can deploy itself.

Deployment starts from this repository on the local machine.

[![Deploy to Local Machine](https://img.shields.io/badge/Deploy_to_Local_Machine-PowerShell-blue?logo=powershell)](scripts/deploy-to-local-machine.ps1)

The button above opens the local deployment script in the repo. Run it from PowerShell to install dependencies, build the frontend, start the backend, start the frontend, and open the app.

### One-command local machine deployment

From a standalone clone:

```powershell
git clone https://github.com/jerrywolff_microsoft/modeltrainer.git
cd modeltrainer
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-to-local-machine.ps1
```

From the monorepo:

```powershell
git clone https://github.com/jwgmpi-wolff/AppsandScripts.git
cd AppsandScripts\modeltrainer
powershell -ExecutionPolicy Bypass -File .\scripts\deploy-to-local-machine.ps1
```

You can also run the npm wrapper after dependencies are available:

```powershell
npm run deploy:local
```

What the local deployment automation really does:

1. Verifies `node`, `npm`, and `python` are on PATH.
2. Runs `npm run install:all`.
3. Runs `npm run build`.
4. Starts the backend at `http://127.0.0.1:8787` in a new PowerShell window.
5. Starts the frontend at `http://127.0.0.1:3000` in a new PowerShell window.
6. Opens the browser to the local app.

The training pipeline still creates and installs `.venv-training` during the app's real **Prepare environment** stage because the ML requirements depend on the detected hardware backend.

### Local production preview

```powershell
cd C:\Users\jerrywolff\Documents\.git\modeltrainer
npm run build
npm run preview
```

The static frontend alone is not the complete app; it still needs the local backend at `http://127.0.0.1:8787` for hardware checks, uploads, and pipeline execution.

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
