import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import fs from 'node:fs';
import path from 'node:path';
import readline from 'node:readline';
import { runSystemCheck } from './systemCheck.js';
import { scriptsRoot, workspaceRoot } from './runtimePaths.js';
import { getOutputDirectory } from './modelArtifacts.js';
import { ensureTrainingPython } from './pythonRuntime.js';

const VENV_DIR = path.join(workspaceRoot, '.venv-training');
const VIDEO_VENV_DIR = path.join(workspaceRoot, '.venv-training-video');

export const runs = new Map();

function makeRun(runId) {
  const emitter = new EventEmitter();
  const run = { id: runId, emitter, events: [], stageResults: {}, done: false };
  runs.set(runId, run);
  return run;
}

function emitEvent(run, event) {
  run.events.push(event);
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
      child = spawn(command, args, { cwd: workspaceRoot, windowsHide: true, ...options });
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

    if (options.logPath) {
      fs.mkdirSync(path.dirname(options.logPath), { recursive: true });
    }
    const logStream = options.logPath
      ? fs.createWriteStream(options.logPath, { flags: options.appendLog ? 'a' : 'w' })
      : null;
    const handleLine = (line, isError = false) => {
      if (!line) return;
      logStream?.write(`${isError ? 'stderr: ' : ''}${line}\n`);
      const eventPrefix = 'MODEL_TRAINER_EVENT|';
      if (!isError && line.startsWith(eventPrefix)) {
        try {
          const event = JSON.parse(line.slice(eventPrefix.length));
          emitEvent(run, { type: 'training-progress', stageId: stage.id, ...event });
          emitEvent(run, { type: 'stage-log', stageId: stage.id, line: event.message });
          return;
        } catch {
          // Preserve malformed event output in the log as an ordinary line.
        }
      }
      emitEvent(run, {
        type: 'stage-log',
        stageId: stage.id,
        line: isError ? `stderr: ${line}` : line,
      });
    };
    if (child.stdout) {
      readline.createInterface({ input: child.stdout }).on('line', (line) => handleLine(line));
    }
    if (child.stderr) {
      readline.createInterface({ input: child.stderr }).on('line', (line) => handleLine(line, true));
    }

    child.on('error', (error) => {
      emitEvent(run, { type: 'stage-log', stageId: stage.id, line: `ERROR: ${error.message}` });
    });

    child.on('close', (exitCode) => {
      logStream?.end();
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

const venvPythonBin =
  process.platform === 'win32'
    ? path.join(VENV_DIR, 'Scripts', 'python.exe')
    : path.join(VENV_DIR, 'bin', 'python');
const videoVenvPythonBin =
  process.platform === 'win32'
    ? path.join(VIDEO_VENV_DIR, 'Scripts', 'python.exe')
    : path.join(VIDEO_VENV_DIR, 'bin', 'python');

// The video pipeline uses its own venv (see prepare_env.py's --modality flag)
// because it needs a newer, incompatible transformers version than the text
// pipeline. Every stage after "prepare" must run inside the right venv for
// whichever modality this run's dataset actually is.
function pythonBinFor(config) {
  return config.datasetKind === 'video' ? videoVenvPythonBin : venvPythonBin;
}

// Each stage maps to a REAL command. Stages that have real hard prerequisites
// (GPU, WSL2, llama.cpp) run a real preflight check first and honestly fail,
// with a real explanatory message, if the prerequisite is not present. Nothing
// here fakes success.
function buildStageCommand(stage, config) {
  const isVideo = config.datasetKind === 'video';
  const stagePythonBin = pythonBinFor(config);
  const systemPythonBin = config.trainingPython;

  switch (stage.id) {
    case 'validate':
      if (isVideo) {
        return {
          command: systemPythonBin,
          args: [
            path.join(scriptsRoot, 'validate_video_dataset.py'),
            config.datasetPath,
            '--videos-dir',
            config.videosDir,
          ],
        };
      }
      return { command: systemPythonBin, args: [path.join(scriptsRoot, 'validate_dataset.py'), config.datasetPath] };
    case 'prepare':
      return {
        command: systemPythonBin,
        args: [
          path.join(scriptsRoot, 'prepare_env.py'),
          '--backend',
          config.recommendedBackend || 'cpu',
          '--modality',
          isVideo ? 'video' : 'text',
          '--install',
        ],
      };
    case 'train':
      if (isVideo) {
        return {
          command: stagePythonBin,
          args: [
            path.join(scriptsRoot, 'train_video_vl.py'),
            '--manifest',
            config.datasetPath,
            '--videos-dir',
            config.videosDir,
            '--model',
            config.model,
            '--max-frames',
            String(config.quickTraining ? Math.min(Number(config.maxFrames) || 8, 4) : config.maxFrames || 8),
            '--fps',
            String(config.fps || 1),
            '--grad-accum',
            String(config.quickTraining ? 1 : config.gradAccum),
            '--rank',
            String(config.rank),
            '--output-name',
            config.name,
            ...(config.quickTraining ? ['--max-steps', '1', '--sample-size', '4'] : []),
          ],
        };
      }
      return {
        command: stagePythonBin,
        args: [
          path.join(scriptsRoot, 'train_qlora.py'),
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
          '--epochs',
          String(config.epochs),
          '--learning-rate',
          String(config.learningRate),
          ...(config.resumeCheckpoint ? ['--resume-checkpoint', config.resumeCheckpoint] : []),
          ...(config.quickTraining ? ['--max-steps', '1', '--sample-size', '8'] : []),
        ],
      };
    case 'evaluate':
      return { command: stagePythonBin, args: [path.join(scriptsRoot, 'evaluate_adapter.py'), '--output-name', config.name] };
    case 'merge':
      return { command: stagePythonBin, args: [path.join(scriptsRoot, 'merge_adapter.py'), '--output-name', config.name] };
    case 'convert':
      return { command: stagePythonBin, args: [path.join(scriptsRoot, 'convert_to_gguf.py'), '--output-name', config.name] };
    case 'quantize':
      return { command: stagePythonBin, args: [path.join(scriptsRoot, 'quantize_gguf.py'), '--output-name', config.name] };
    case 'ollama':
      return { command: stagePythonBin, args: [path.join(scriptsRoot, 'register_ollama.py'), '--output-name', config.name] };
    case 'smoke':
      return { command: stagePythonBin, args: [path.join(scriptsRoot, 'smoke_test.py'), '--output-name', config.name] };
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
    let trainingPython;
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

    try {
      const resolvedPython = await ensureTrainingPython((line) =>
        emitEvent(run, { type: 'stage-log', stageId: 'prepare', line }),
      );
      trainingPython = resolvedPython.command;
      emitEvent(run, {
        type: 'stage-log',
        stageId: 'prepare',
        line: `Training interpreter: ${resolvedPython.executable} (${resolvedPython.architecture}, Python ${resolvedPython.version.join('.')}).`,
      });
    } catch (error) {
      emitEvent(run, { type: 'stage-log', stageId: 'prepare', line: `ERROR: ${error.message}` });
      emitEvent(run, { type: 'stage-end', stageId: 'prepare', status: 'failed', exitCode: null });
      run.stageResults.prepare = 'failed';
      emitEvent(run, { type: 'run-complete', stageResults: run.stageResults });
      return;
    }

    const runConfig = { ...config, recommendedBackend, trainingPython };

    for (const stage of runConfig.stages) {
      const built = buildStageCommand(stage, runConfig);

      if (!built) {
        emitEvent(run, { type: 'stage-log', stageId: stage.id, line: `ERROR: unknown stage "${stage.id}".` });
        emitEvent(run, { type: 'stage-end', stageId: stage.id, status: 'failed', exitCode: null });
        run.stageResults[stage.id] = 'failed';
        break;
      }

      const result = await runStageProcess(run, stage, built.command, built.args, stage.id === 'train'
        ? {
            logPath: path.join(getOutputDirectory(runConfig.name), 'training.log'),
            appendLog: Boolean(runConfig.resumeCheckpoint),
          }
        : {});
      run.stageResults[stage.id] = result.status;

      if (result.status === 'failed') {
        break;
      }
    }

    emitEvent(run, { type: 'run-complete', stageResults: run.stageResults });
  })();

  return run;
}
