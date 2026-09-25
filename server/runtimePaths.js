import path from 'node:path';
import { fileURLToPath } from 'node:url';

const serverDirectory = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(serverDirectory, '..');

export const workspaceRoot = process.env.MODEL_TRAINER_WORKSPACE || packageRoot;
export const scriptsRoot = process.env.MODEL_TRAINER_SCRIPTS_ROOT || path.join(packageRoot, 'scripts');
export const samplesRoot = process.env.MODEL_TRAINER_SAMPLES_ROOT || path.join(packageRoot, 'samples');
