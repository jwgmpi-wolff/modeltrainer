import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { scriptsRoot, workspaceRoot } from './runtimePaths.js';

function pythonCandidates() {
  const executable = process.platform === 'win32' ? 'python.exe' : 'python';
  const binDirectory = process.platform === 'win32' ? 'Scripts' : 'bin';
  return [
    path.join(workspaceRoot, '.venv-training', binDirectory, executable),
    path.join(path.dirname(workspaceRoot), 'modeltrainer', '.venv-training', binDirectory, executable),
    process.platform === 'win32' ? 'python' : 'python3',
  ];
}

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: workspaceRoot, windowsHide: true });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString('utf8');
    });
    child.once('error', reject);
    child.once('close', (exitCode) => resolve({ exitCode, stdout, stderr }));
  });
}

export async function resolveAcceleratorPython() {
  const failures = [];
  for (const candidate of [...new Set(pythonCandidates())]) {
    if (path.isAbsolute(candidate) && !fs.existsSync(candidate)) continue;
    try {
      const probe = await runProcess(candidate, [
        '-c',
        'import torch,torch_directml,onnx,onnxruntime,onnxruntime_qnn; assert torch_directml.is_available()',
      ]);
      if (probe.exitCode === 0) return candidate;
      failures.push(`${candidate}: ${probe.stderr.trim()}`);
    } catch (error) {
      failures.push(`${candidate}: ${error.message}`);
    }
  }
  throw new Error(`No compatible DirectML/QNN Python environment was found. ${failures.join(' | ')}`);
}

export async function runAcceleratorSelfTest() {
  const python = await resolveAcceleratorPython();
  const result = await runProcess(python, [path.join(scriptsRoot, 'accelerator_probe.py')]);
  const line = result.stdout
    .split(/\r?\n/)
    .map((item) => item.replaceAll('\0', '').trim())
    .findLast((item) => item.startsWith('{'));
  if (!line) {
    throw new Error(`Accelerator probe returned no JSON result. ${result.stderr || result.stdout}`);
  }
  const parsed = JSON.parse(line);
  if (result.exitCode !== 0) {
    throw new Error(parsed.details || result.stderr || `Accelerator probe exited with ${result.exitCode}.`);
  }
  return { ...parsed, python };
}
