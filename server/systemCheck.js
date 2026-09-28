import { execFile } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { promisify } from 'node:util';
import { workspaceRoot } from './runtimePaths.js';

const execFileAsync = promisify(execFile);

async function tryCommand(command, args) {
  try {
    const { stdout } = await execFileAsync(command, args, { timeout: 8000, windowsHide: true });
    return { found: true, output: stdout.trim() };
  } catch (error) {
    return { found: false, output: null, error: error.message };
  }
}

async function detectGpu() {
  // Real hardware enumeration via PowerShell CIM — this reports the ACTUAL
  // installed graphics adapters (Intel, AMD, NVIDIA, Qualcomm), not a guess.
  try {
    const { stdout } = await execFileAsync(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        'Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name',
      ],
      { timeout: 10000, windowsHide: true },
    );
    const names = stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    return names;
  } catch (error) {
    return [];
  }
}

// Classifies a real GPU adapter name into a real vendor. Returns 'unknown'
// rather than guessing wrong — that's what happened with the CPU vendor
// check earlier and it's the same honest rule here.
function classifyGpuVendor(name) {
  if (/nvidia|geforce|quadro|rtx|gtx/i.test(name)) return 'nvidia';
  if (/amd|radeon|ati\s/i.test(name)) return 'amd';
  if (/intel|iris|uhd graphics|arc\(tm\)/i.test(name)) return 'intel';
  if (/qualcomm|adreno|snapdragon/i.test(name)) return 'qualcomm';
  return 'unknown';
}

// A "fake"/virtual adapter that DirectML can't meaningfully use for training
// (e.g. software renderers, remote-session mirror drivers). Real physical
// adapters (even ones we can't name-classify, like this machine's "Trigger 6
// External Graphics" entries) still count as real DirectML-capable hardware.
function isVirtualGpu(name) {
  return /microsoft basic render|remote desktop|virtual display|indirect display/i.test(name);
}

export async function runSystemCheck() {
  const localPython =
    process.platform === 'win32'
      ? path.join(workspaceRoot, '.venv-training', 'Scripts', 'python.exe')
      : path.join(workspaceRoot, '.venv-training', 'bin', 'python');
  const pythonCommand = fs.existsSync(localPython) ? localPython : process.platform === 'win32' ? 'python' : 'python3';
  const [python, ollama, wsl, gpuNames] = await Promise.all([
    tryCommand(pythonCommand, ['--version']),
    tryCommand('ollama', ['--version']),
    tryCommand('wsl', ['--status']),
    detectGpu(),
  ]);

  const cpuModel = os.cpus()?.[0]?.model || 'Unknown CPU';
  const cpuVendor = /intel/i.test(cpuModel) ? 'Intel' : /amd/i.test(cpuModel) ? 'AMD' : 'Unknown';
  const totalMemoryGb = Math.round((os.totalmem() / (1024 * 1024 * 1024)) * 10) / 10;
  const freeMemoryGb = Math.round((os.freemem() / (1024 * 1024 * 1024)) * 10) / 10;

  const gpus = gpuNames.map((name) => ({ name, vendor: classifyGpuVendor(name), real: !isVirtualGpu(name) }));
  const realGpus = gpus.filter((gpu) => gpu.real);
  const hasNvidiaGpu = realGpus.some((gpu) => gpu.vendor === 'nvidia');

  // Real backend recommendation, in priority order, based on what's ACTUALLY
  // installed (an NVIDIA card can use real CUDA) vs. what's only reachable
  // through DirectML (AMD/Intel/Qualcomm/other GPUs on Windows). This is a
  // hint for prepare_env.py to write the right requirements.txt — the real
  // source of truth is gpu_backend.py, which asks PyTorch directly at
  // training time.
  let recommendedBackend = 'cpu';
  if (hasNvidiaGpu) {
    recommendedBackend = 'cuda';
  } else if (process.platform === 'win32' && realGpus.length > 0) {
    recommendedBackend = 'directml';
  } else if (process.platform === 'darwin') {
    recommendedBackend = 'mps';
  } else if (realGpus.some((gpu) => gpu.vendor === 'intel')) {
    recommendedBackend = 'xpu';
  }

  return {
    checkedAt: new Date().toISOString(),
    cpu: {
      model: cpuModel,
      vendor: cpuVendor,
      cores: os.cpus()?.length || null,
    },
    memory: {
      totalGb: totalMemoryGb,
      freeGb: freeMemoryGb,
    },
    gpu: {
      names: gpuNames,
      gpus,
      hasNvidiaGpu,
      recommendedBackend,
    },
    tools: {
      python: python.found ? `${python.output} (${pythonCommand})` : null,
      ollama: ollama.found ? ollama.output : null,
      wsl: wsl.found ? wsl.output : null,
    },
    // DirectML works natively on Windows without WSL2, so training is
    // possible on any real GPU, not just NVIDIA+WSL2. The real check still
    // happens at training time via gpu_backend.py; this is an honest
    // best-effort forecast, not a guarantee.
    canTrainLocally: recommendedBackend !== 'cpu',
  };
}
