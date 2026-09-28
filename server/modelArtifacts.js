import fs from 'node:fs';
import path from 'node:path';
import { workspaceRoot } from './runtimePaths.js';

const outputRoot = path.resolve(workspaceRoot, 'output');
const activeModelPath = path.join(workspaceRoot, 'active-agent-model.json');

export function requireSafeOutputName(outputName) {
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(outputName || '')) {
    throw new Error('Model name may contain only letters, numbers, dots, underscores, and hyphens.');
  }
  return outputName;
}

export function getOutputDirectory(outputName) {
  return path.join(outputRoot, requireSafeOutputName(outputName));
}

function isInside(parent, candidate) {
  const relative = path.relative(parent, candidate);
  return relative !== '' && !relative.startsWith(`..${path.sep}`) && relative !== '..' && !path.isAbsolute(relative);
}

export function validateResumeCheckpoint(outputName, checkpointPath) {
  if (!checkpointPath) return null;

  const outputDirectory = getOutputDirectory(outputName);
  if (!fs.existsSync(outputDirectory) || !fs.existsSync(checkpointPath)) {
    throw new Error('The selected checkpoint does not exist.');
  }

  const resolvedOutput = fs.realpathSync(outputDirectory);
  const resolvedCheckpoint = fs.realpathSync(checkpointPath);
  if (!isInside(resolvedOutput, resolvedCheckpoint) || path.dirname(resolvedCheckpoint) !== resolvedOutput) {
    throw new Error('The resume checkpoint must be directly inside the selected model output directory.');
  }
  if (!/^checkpoint-\d+$/.test(path.basename(resolvedCheckpoint))) {
    throw new Error('The selected directory is not a Hugging Face Trainer checkpoint.');
  }
  for (const requiredFile of ['trainer_state.json', 'training_run.json']) {
    if (!fs.existsSync(path.join(resolvedCheckpoint, requiredFile))) {
      throw new Error(`The selected checkpoint is incomplete: missing ${requiredFile}.`);
    }
  }
  return resolvedCheckpoint;
}

export function listCheckpoints(outputName) {
  const outputDirectory = getOutputDirectory(outputName);
  const checkpoints = [{ path: '', step: null, savedAt: null, displayName: 'Start new training' }];
  if (!fs.existsSync(outputDirectory)) return checkpoints;

  const discovered = fs
    .readdirSync(outputDirectory, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^checkpoint-\d+$/.test(entry.name))
    .map((entry) => {
      const checkpointPath = path.join(outputDirectory, entry.name);
      const statePath = path.join(checkpointPath, 'trainer_state.json');
      const manifestPath = path.join(checkpointPath, 'training_run.json');
      if (!fs.existsSync(statePath) || !fs.existsSync(manifestPath)) return null;
      const savedAt = fs.statSync(statePath).mtime;
      const step = Number(entry.name.slice('checkpoint-'.length));
      return {
        path: checkpointPath,
        step,
        savedAt: savedAt.toISOString(),
        displayName: `Step ${step.toLocaleString()} - ${savedAt.toLocaleString()}`,
      };
    })
    .filter(Boolean)
    .sort((left, right) => new Date(right.savedAt) - new Date(left.savedAt));

  return [...checkpoints, ...discovered];
}

function readCandidate(directory, artifactType, trainingStep = null) {
  const adapterConfigPath = path.join(directory, 'adapter_config.json');
  if (!fs.existsSync(adapterConfigPath)) return null;
  const hasWeights = ['adapter_model.safetensors', 'adapter_model.bin'].some((name) =>
    fs.existsSync(path.join(directory, name)),
  );
  if (!hasWeights) return null;

  const adapterConfig = JSON.parse(fs.readFileSync(adapterConfigPath, 'utf8'));
  const stats = fs.statSync(adapterConfigPath);
  const outputName =
    artifactType === 'Completed adapter'
      ? path.basename(directory)
      : path.basename(path.dirname(directory));
  return {
    id: directory,
    displayName: `${outputName} - ${artifactType}`,
    baseModel: adapterConfig.base_model_name_or_path || 'Unknown base model',
    adapterPath: directory,
    artifactType,
    trainingStep,
    savedAt: stats.mtime.toISOString(),
  };
}

export function discoverAgentModels() {
  if (!fs.existsSync(outputRoot)) return [];
  const candidates = [];

  for (const outputEntry of fs.readdirSync(outputRoot, { withFileTypes: true })) {
    if (!outputEntry.isDirectory()) continue;
    const modelDirectory = path.join(outputRoot, outputEntry.name);
    const completed = readCandidate(modelDirectory, 'Completed adapter');
    if (completed) candidates.push(completed);

    for (const entry of fs.readdirSync(modelDirectory, { withFileTypes: true })) {
      const match = entry.isDirectory() && /^checkpoint-(\d+)$/.exec(entry.name);
      if (!match) continue;
      const checkpoint = readCandidate(
        path.join(modelDirectory, entry.name),
        'Training checkpoint',
        Number(match[1]),
      );
      if (checkpoint) candidates.push(checkpoint);
    }
  }

  return candidates.sort((left, right) => new Date(right.savedAt) - new Date(left.savedAt));
}

export function getActiveAgentModel() {
  if (!fs.existsSync(activeModelPath)) return null;
  return JSON.parse(fs.readFileSync(activeModelPath, 'utf8'));
}

export function activateAgentModel(adapterPath) {
  if (!adapterPath || !fs.existsSync(adapterPath)) {
    throw new Error('The selected adapter does not exist.');
  }
  const resolvedRoot = fs.existsSync(outputRoot) ? fs.realpathSync(outputRoot) : outputRoot;
  const resolvedAdapter = fs.realpathSync(adapterPath);
  if (!isInside(resolvedRoot, resolvedAdapter)) {
    throw new Error('The selected agent model must be inside the Model Trainer output directory.');
  }

  const candidate = discoverAgentModels().find((item) => item.adapterPath === resolvedAdapter);
  if (!candidate) {
    throw new Error('The selected adapter is incomplete or is not a discovered model artifact.');
  }
  const active = { ...candidate, activatedAt: new Date().toISOString() };
  fs.writeFileSync(activeModelPath, JSON.stringify(active, null, 2), 'utf8');
  return active;
}
