import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { workspaceRoot } from './runtimePaths.js';

const TRAINING_PYTHON_VERSION = '3.11.9';
const privatePythonRoot = path.join(workspaceRoot, 'python311-x64');
const privatePython = path.join(privatePythonRoot, 'python.exe');

function runProcess(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { windowsHide: true });
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

function pythonCandidates() {
  if (process.platform !== 'win32') {
    return [process.env.MODEL_TRAINER_PYTHON, 'python3', 'python'].filter(Boolean);
  }

  const localPrograms = process.env.LOCALAPPDATA || '';
  const programFiles = process.env.ProgramFiles || '';
  return [
    process.env.MODEL_TRAINER_PYTHON,
    privatePython,
    path.join(localPrograms, 'Programs', 'Python', 'Python312', 'python.exe'),
    path.join(localPrograms, 'Programs', 'Python', 'Python311', 'python.exe'),
    path.join(localPrograms, 'Programs', 'Python', 'Python310', 'python.exe'),
    path.join(programFiles, 'Python312', 'python.exe'),
    path.join(programFiles, 'Python311', 'python.exe'),
    path.join(programFiles, 'Python310', 'python.exe'),
    'python',
  ].filter(Boolean);
}

export async function probePython(command) {
  if (path.isAbsolute(command) && !fs.existsSync(command)) return null;
  try {
    const result = await runProcess(command, [
      '-c',
      'import json,platform,sys; print(json.dumps({"architecture":platform.machine(),"version":[sys.version_info.major,sys.version_info.minor,sys.version_info.micro],"executable":sys.executable}))',
    ]);
    if (result.exitCode !== 0) return null;
    const details = JSON.parse(result.stdout.trim().split(/\r?\n/).at(-1));
    const [major, minor] = details.version;
    const supportedVersion = major === 3 && minor >= 10 && minor < 13;
    const supportedArchitecture =
      process.platform !== 'win32' || /^(amd64|x86_64)$/i.test(details.architecture);
    return {
      ...details,
      command,
      compatible: supportedVersion && supportedArchitecture,
      reason: !supportedVersion
        ? `Python ${details.version.join('.')} is outside the supported 3.10-3.12 range.`
        : !supportedArchitecture
          ? `Python is ${details.architecture}; Windows PyTorch and DirectML training require x64 Python.`
          : null,
    };
  } catch {
    return null;
  }
}

export async function findCompatibleTrainingPython() {
  const attempted = [];
  for (const candidate of [...new Set(pythonCandidates())]) {
    const details = await probePython(candidate);
    if (!details) continue;
    attempted.push(details);
    if (details.compatible) return { ...details, attempted };
  }
  return { command: null, compatible: false, attempted };
}

export async function ensureTrainingPython(onProgress = () => {}) {
  const existing = await findCompatibleTrainingPython();
  if (existing.compatible) return existing;
  if (process.platform !== 'win32') {
    throw new Error('No compatible Python 3.10-3.12 interpreter was found.');
  }

  onProgress(
    'No compatible x64 Python was found. Installing private Python 3.11.9 for PyTorch/DirectML training.',
  );
  const installerPath = path.join(os.tmpdir(), `python-${TRAINING_PYTHON_VERSION}-amd64.exe`);
  const installerUrl = `https://www.python.org/ftp/python/${TRAINING_PYTHON_VERSION}/python-${TRAINING_PYTHON_VERSION}-amd64.exe`;
  const response = await fetch(installerUrl);
  if (!response.ok) {
    throw new Error(`Could not download x64 Python from python.org: HTTP ${response.status}.`);
  }
  const installerBytes = Buffer.from(await response.arrayBuffer());
  if (installerBytes.length < 20 * 1024 * 1024) {
    throw new Error(`The downloaded Python installer is unexpectedly small (${installerBytes.length} bytes).`);
  }
  fs.writeFileSync(installerPath, installerBytes);

  try {
    const signature = await runProcess('powershell.exe', [
      '-NoProfile',
      '-NonInteractive',
      '-Command',
      `(Get-AuthenticodeSignature -LiteralPath '${installerPath.replaceAll("'", "''")}') | Select-Object Status,@{Name='Signer';Expression={$_.SignerCertificate.Subject}} | ConvertTo-Json -Compress`,
    ]);
    const signatureDetails = signature.exitCode === 0 ? JSON.parse(signature.stdout.trim()) : null;
    if (
      signatureDetails?.Status !== 'Valid' ||
      !/Python Software Foundation/i.test(signatureDetails?.Signer || '')
    ) {
      throw new Error('The downloaded Python installer does not have a valid Python Software Foundation signature.');
    }

    fs.mkdirSync(privatePythonRoot, { recursive: true });
    const install = await runProcess(installerPath, [
      '/quiet',
      'InstallAllUsers=0',
      `TargetDir=${privatePythonRoot}`,
      'Include_pip=1',
      'Include_launcher=0',
      'Include_test=0',
      'Include_doc=0',
      'PrependPath=0',
      'Shortcuts=0',
    ]);
    if (install.exitCode !== 0) {
      throw new Error(`The private x64 Python installer exited with code ${install.exitCode}. ${install.stderr}`);
    }
  } finally {
    fs.rmSync(installerPath, { force: true });
  }

  const installed = await probePython(privatePython);
  if (!installed?.compatible) {
    throw new Error(`Private x64 Python installation did not produce a compatible interpreter at ${privatePython}.`);
  }
  return { ...installed, attempted: [...existing.attempted, installed] };
}
