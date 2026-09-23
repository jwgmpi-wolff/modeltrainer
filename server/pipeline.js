import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { runSystemCheck } from './systemCheck.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, '..');
const SCRIPTS_DIR = path.join(REPO_ROOT, 'scripts');
const VENV_DIR = path.join(REPO_ROOT, '.venv-training');

export const runs = new Map();

function makeRun(runId) {
  const emitter = new EventEmitter();
  const run = { id: runId, emitter, stageResults: {}, done: false };
  runs.set(runId, run);
  return run;
}

function emitEvent(run, event) {
  run.emitter.emit('event', event);
  if (event.type === 'run-complete') {
    run.done = true;
  }
}

/**
 * Spawns a real child process for a stage and streams real stdout/stderr lines.
 * The stage only completes if the real process exits with code 0. There is no
 * simulated success path.
 */
function runStageProcess(run, stage, command, args, options = {}) {
  return new Promise((resolve) => {
    emitEvent(run, { type: 'stage-start', stageId: stage.id, name: stage.name });

    let child;
    try {
      child = spawn(command, args, { cwd: REPO_ROOT, windowsHide: true, ...options });
    } catch (error) {
      emitEvent(run, {
        type: 'stage-log',
        stageId: stage.id,
        line: `ERROR: could not start "${command}": ${error.message}`,
      });
      emitEvent(run, { type: 'stage-end', stageId: stage.id, status: 'failed', exitCode: null });
      resolve({ status: 'failed', exitCode: null });
      return;
    }

    child.stdout?.on('data', (chunk) => {
      chunk
        .toString('utf8')
        .split(/\r?\n/)
        .filter(Boolean)
        .forEach((line) => emitEvent(run, { type: 'stage-log', stageId: stage.id, line }));
    });

    child.stderr?.on('data', (chunk) => {
      chunk
        .toString('utf8')
        .split(/\r?\n/)
        .filter(Boolean)
        .forEach((line) => emitEvent(run, { type: 'stage-log', stageId: stage.id, line: `stderr: ${line}` }));
    });

    child.on('error', (error) => {
      emitEvent(run, { type: 'stage-log', stageId: stage.id, line: `ERROR: ${error.message}` });
    });

    child.on('close', (exitCode) => {
      const status = exitCode === 0 ? 'complete' : 'failed';
      if (process.platform === 'win32' && exitCode === 3221225477) {
        emitEvent(run, {
          type: 'stage-log',
          stageId: stage.id,
          line:
            'ERROR: Windows stopped this stage with native access-violation code 0xC0000005. ' +
            'This usually means a native ML backend crashed (often model too large or unsupported GPU backend operation), ' +
            'not that the web page merely changed status.',
        });
      }
      emitEvent(run, { type: 'stage-end', stageId: stage.id, status, exitCode });
      resolve({ status, exitCode });
    });
  });
}

const pythonBin = process.platform === 'win32' ? 'python' : 'python3';
const venvPythonBin =
  process.platform === 'win32'
    ? path.join(VENV_DIR, 'Scripts', 'python.exe')
    : path.join(VENV_DIR, 'bin', 'python');

// Each stage maps to a REAL command. Stages that have real hard prerequisites
// (GPU, WSL2, llama.cpp) run a real preflight check first and honestly fail,
// with a real explanatory message, if the prerequisite is not present. Nothing
// here fakes success.
function buildStageCommand(stage, config) {
  switch (stage.id) {
    case 'validate':
      return { command: pythonBin, args: [path.join(SCRIPTS_DIR, 'validate_dataset.py'), config.datasetPath] };
    case 'prepare':
      return {
        command: pythonBin,
        args: [path.join(SCRIPTS_DIR, 'prepare_env.py'), '--backend', config.recommendedBackend || 'cpu', '--install'],
      };
    case 'train':
      return {
        command: venvPythonBin,
        args: [
          path.join(SCRIPTS_DIR, 'train_qlora.py'),
          '--dataset',
          config.datasetPath,
          '--model',
          config.model,
          '--max-length',
          String(config.quickTraining ? Math.min(Number(config.maxLength) || 2048, 512) : config.maxLength),
          '--batch-size',
          String(config.batchSize),
          '--grad-accum',
          String(config.quickTraining ? 1 : config.gradAccum),
          '--rank',
          String(config.rank),
          '--output-name',
          config.name,
          ...(config.quickTraining ? ['--max-steps', '1', '--sample-size', '8'] : []),
        ],
      };
    case 'evaluate':
      return { command: venvPythonBin, args: [path.join(SCRIPTS_DIR, 'evaluate_adapter.py'), '--output-name', config.name] };
    case 'merge':
      return { command: venvPythonBin, args: [path.join(SCRIPTS_DIR, 'merge_adapter.py'), '--output-name', config.name] };
    case 'convert':
      return { command: venvPythonBin, args: [path.join(SCRIPTS_DIR, 'convert_to_gguf.py'), '--output-name', config.name] };
    case 'quantize':
      return { command: venvPythonBin, args: [path.join(SCRIPTS_DIR, 'quantize_gguf.py'), '--output-name', config.name] };
    case 'ollama':
      return { command: venvPythonBin, args: [path.join(SCRIPTS_DIR, 'register_ollama.py'), '--output-name', config.name] };
    case 'smoke':
      return { command: venvPythonBin, args: [path.join(SCRIPTS_DIR, 'smoke_test.py'), '--output-name', config.name] };
    default:
      return null;
  }
}

export async function startPipelineRun(runId, config) {
  const run = makeRun(runId);

  (async () => {
    // Real, fresh hardware check right before running — this is what decides
    // which real GPU backend (CUDA / DirectML / XPU / MPS / CPU) the
    // "Prepare environment" stage tells pip to install for. Not cached from
    // an earlier page load — queried again right now.
    let recommendedBackend = 'cpu';
    try {
      const check = await runSystemCheck();
      recommendedBackend = check.gpu.recommendedBackend;
      emitEvent(run, {
        type: 'stage-log',
        stageId: 'prepare',
        line: `Real hardware check: recommended training backend is "${recommendedBackend}" (GPUs: ${
          check.gpu.names.join(', ') || 'none detected'
        }).`,
      });
    } catch (error) {
      emitEvent(run, {
        type: 'stage-log',
        stageId: 'prepare',
        line: `Could not re-check hardware, defaulting to CPU backend: ${error.message}`,
      });
    }

    const runConfig = { ...config, recommendedBackend };

    for (const stage of runConfig.stages) {
      const built = buildStageCommand(stage, runConfig);

      if (!built) {
        emitEvent(run, { type: 'stage-log', stageId: stage.id, line: `ERROR: unknown stage "${stage.id}".` });
        emitEvent(run, { type: 'stage-end', stageId: stage.id, status: 'failed', exitCode: null });
        run.stageResults[stage.id] = 'failed';
        break;
      }

      const result = await runStageProcess(run, stage, built.command, built.args);
      run.stageResults[stage.id] = result.status;

      if (result.status === 'failed') {
        break;
      }
    }

    emitEvent(run, { type: 'run-complete', stageResults: run.stageResults });
  })();

  return run;
}
