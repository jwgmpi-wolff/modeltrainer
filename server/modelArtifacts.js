import fs from 'node:fs';
import path from 'node:path';
import { workspaceRoot } from './runtimePaths.js';

const outputRoot = path.resolve(workspaceRoot, 'output');
const activeModelPath = path.join(workspaceRoot, 'active-agent-model.json');
const standaloneModelExtensions = new Map([
  ['.ckpt', 'PyTorch checkpoint'],
  ['.gguf', 'GGUF'],
  ['.mlmodel', 'Core ML'],
  ['.onnx', 'ONNX'],
  ['.ort', 'ONNX Runtime'],
  ['.pt', 'PyTorch'],
  ['.pth', 'PyTorch'],
  ['.safetensors', 'SafeTensors'],
  ['.tflite', 'TensorFlow Lite'],
]);

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

function isSameOrInside(parent, candidate) {
  return path.resolve(parent) === path.resolve(candidate) || isInside(path.resolve(parent), path.resolve(candidate));
}

function getModelDirectoryFormat(directory, entries) {
  const fileNames = new Set(entries.filter((entry) => entry.isFile()).map((entry) => entry.name.toLowerCase()));
  const hasFileMatching = (pattern) => [...fileNames].some((name) => pattern.test(name));

  if (
    fileNames.has('adapter_config.json') &&
    (fileNames.has('adapter_model.safetensors') || fileNames.has('adapter_model.bin'))
  ) {
    return 'Hugging Face adapter';
  }
  if (
    fileNames.has('config.json') &&
    hasFileMatching(/^(?:model|pytorch_model)(?:-\d+-of-\d+)?\.(?:safetensors|bin)$/)
  ) {
    return 'Hugging Face model';
  }
  if (fileNames.has('config.json') && hasFileMatching(/\.onnx$/)) return 'ONNX model bundle';
  if (fileNames.has('saved_model.pb')) return 'TensorFlow SavedModel';
  if (hasFileMatching(/\.xml$/) && hasFileMatching(/\.bin$/)) return 'OpenVINO model';
  return null;
}

function describeStandaloneModel(filePath, siblingEntries) {
  const extension = path.extname(filePath).toLowerCase();
  const format = standaloneModelExtensions.get(extension);
  if (!format) return null;
  const fileName = path.basename(filePath);
  const companionPaths =
    extension === '.onnx'
      ? siblingEntries
          .filter(
            (entry) =>
              entry.isFile() &&
              (entry.name.startsWith(`${fileName}.`) || entry.name.startsWith(`${fileName}_`)),
          )
          .map((entry) => path.join(path.dirname(filePath), entry.name))
      : [];
  return {
    sourcePath: filePath,
    name: fileName,
    kind: 'file',
    format,
    companionPaths,
  };
}

export function crawlForModels(sourceRoot, destinationRoot = '') {
  if (!sourceRoot || typeof sourceRoot !== 'string') {
    throw new Error('A source folder is required.');
  }

  const resolvedSource = path.resolve(sourceRoot);
  if (!fs.existsSync(resolvedSource) || !fs.statSync(resolvedSource).isDirectory()) {
    throw new Error(`Source folder does not exist or is not a directory: ${resolvedSource}`);
  }

  const resolvedDestination = destinationRoot ? path.resolve(destinationRoot) : null;
  if (resolvedDestination && resolvedDestination === resolvedSource) {
    throw new Error('The source and destination folders must be different.');
  }
  const destinationIsInsideSource =
    resolvedDestination && isInside(resolvedSource, resolvedDestination);

  const models = [];
  let skippedSymlinks = 0;

  const visit = (directory, isRoot = false) => {
    if (
      destinationIsInsideSource &&
      isSameOrInside(resolvedDestination, directory)
    ) {
      return;
    }

    const entries = fs.readdirSync(directory, { withFileTypes: true });
    const directoryFormat = !isRoot ? getModelDirectoryFormat(directory, entries) : null;
    if (directoryFormat) {
      models.push({
        sourcePath: directory,
        name: path.basename(directory),
        kind: 'directory',
        format: directoryFormat,
      });
      return;
    }

    for (const entry of entries) {
      const entryPath = path.join(directory, entry.name);
      if (entry.isSymbolicLink()) {
        skippedSymlinks += 1;
      } else if (entry.isDirectory()) {
        if (entry.name.toLowerCase().endsWith('.mlmodelc')) {
          models.push({
            sourcePath: entryPath,
            name: entry.name,
            kind: 'directory',
            format: 'Core ML compiled model',
          });
        } else {
          visit(entryPath);
        }
      } else if (entry.isFile()) {
        const model = describeStandaloneModel(entryPath, entries);
        if (model) models.push(model);
      }
    }
  };

  visit(resolvedSource, true);
  return {
    sourceRoot: resolvedSource,
    destinationRoot: resolvedDestination,
    models: models.sort((left, right) => left.sourcePath.localeCompare(right.sourcePath)),
    skippedSymlinks,
  };
}

function availableDestinationDirectory(destinationRoot, model, reservedPaths) {
  const parsedName = path.parse(model.name);
  const stem = model.kind === 'file' ? parsedName.name : model.name;
  let suffix = 0;
  let candidate;
  do {
    const candidateName = suffix === 0 ? stem : `${stem} (${suffix})`;
    candidate = path.join(destinationRoot, candidateName);
    suffix += 1;
  } while (fs.existsSync(candidate) || reservedPaths.has(candidate.toLowerCase()));
  reservedPaths.add(candidate.toLowerCase());
  return candidate;
}

function moveAcrossFilesystems(sourcePath, destinationPath, kind) {
  try {
    fs.renameSync(sourcePath, destinationPath);
  } catch (error) {
    if (error.code !== 'EXDEV') throw error;
    try {
      if (kind === 'directory') {
        fs.cpSync(sourcePath, destinationPath, { recursive: true, errorOnExist: true });
        fs.rmSync(sourcePath, { recursive: true });
      } else {
        fs.copyFileSync(sourcePath, destinationPath, fs.constants.COPYFILE_EXCL);
        fs.unlinkSync(sourcePath);
      }
    } catch (copyError) {
      if (fs.existsSync(destinationPath)) {
        fs.rmSync(destinationPath, { recursive: kind === 'directory', force: true });
      }
      throw copyError;
    }
  }
}

export function moveDiscoveredModels(sourceRoot, destinationRoot) {
  if (!destinationRoot || typeof destinationRoot !== 'string') {
    throw new Error('A destination folder is required.');
  }

  const scan = crawlForModels(sourceRoot, destinationRoot);
  const resolvedDestination = scan.destinationRoot;
  fs.mkdirSync(resolvedDestination, { recursive: true });

  const reservedPaths = new Set();
  const moves = scan.models.map((model) => {
    const destinationPath = availableDestinationDirectory(resolvedDestination, model, reservedPaths);
    return {
      ...model,
      destinationPath,
      destinationModelPath:
        model.kind === 'directory' ? destinationPath : path.join(destinationPath, model.name),
    };
  });

  const moved = [];
  for (const model of moves) {
    try {
      if (model.kind === 'directory') {
        moveAcrossFilesystems(model.sourcePath, model.destinationModelPath, model.kind);
      } else {
        fs.mkdirSync(model.destinationPath);
        const filesToMove = [model.sourcePath, ...model.companionPaths];
        const completedFileMoves = [];
        try {
          for (const sourcePath of filesToMove) {
            const destinationPath = path.join(model.destinationPath, path.basename(sourcePath));
            moveAcrossFilesystems(sourcePath, destinationPath, 'file');
            completedFileMoves.push({ sourcePath, destinationPath });
          }
        } catch (error) {
          for (const completed of completedFileMoves.reverse()) {
            moveAcrossFilesystems(completed.destinationPath, completed.sourcePath, 'file');
          }
          throw error;
        }
      }
      moved.push(model);
    } catch (error) {
      if (model.kind === 'file' && fs.existsSync(model.destinationPath)) {
        fs.rmSync(model.destinationPath, { recursive: true, force: true });
      }
      const failure = new Error(
        `Moved ${moved.length} of ${moves.length} models, then failed to move "${model.sourcePath}": ${error.message}`,
      );
      failure.moved = moved;
      throw failure;
    }
  }

  return {
    sourceRoot: scan.sourceRoot,
    destinationRoot: resolvedDestination,
    foundCount: scan.models.length,
    moved,
    skippedSymlinks: scan.skippedSymlinks,
  };
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
