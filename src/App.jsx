import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  Circle,
  Cpu,
  FileCheck2,
  FolderSearch,
  MemoryStick,
  Move,
  Play,
  RefreshCw,
  Square,
  Terminal,
  Upload,
} from 'lucide-react';
import { motion } from 'framer-motion';

const apiBaseQuery = new URLSearchParams(window.location.search).get('apiBase');
const API_BASE =
  window.location.protocol === 'file:' && /^http:\/\/127\.0\.0\.1:\d+$/.test(apiBaseQuery || '')
    ? apiBaseQuery
    : 'http://127.0.0.1:8787';
const MODEL_LIBRARY_STORAGE_KEY = 'local-model-lab.model-library';

const initialStages = [
  { id: 'validate', name: 'Validate training data', status: 'ready', selected: true },
  { id: 'prepare', name: 'Prepare environment', status: 'ready', selected: true },
  { id: 'train', name: 'QLoRA training', status: 'ready', selected: true },
  { id: 'evaluate', name: 'Evaluate adapter', status: 'ready', selected: true },
  { id: 'merge', name: 'Merge adapter', status: 'ready', selected: true },
  { id: 'convert', name: 'Convert to GGUF', status: 'ready', selected: true },
  { id: 'quantize', name: 'Quantize GGUF', status: 'ready', selected: true },
  { id: 'ollama', name: 'Register in Ollama', status: 'ready', selected: true },
  { id: 'smoke', name: 'Smoke test model', status: 'ready', selected: true },
];

const statusIcon = (status) => {
  if (status === 'complete') return <CheckCircle2 className="h-4 w-4 text-emerald-500" />;
  if (status === 'failed') return <AlertTriangle className="h-4 w-4 text-red-500" />;
  if (status === 'running') return <RefreshCw className="h-4 w-4 animate-spin text-blue-500" />;
  return <Circle className="h-4 w-4 text-slate-400" />;
};

// Plain-language explanation of what each real GPU backend actually means,
// so users understand what will really happen on THEIR laptop's real
// hardware, whatever brand of GPU it has (NVIDIA, AMD, Intel, or Qualcomm).
const backendDescription = (backend) => {
  switch (backend) {
    case 'cuda':
      return 'This laptop has a real NVIDIA GPU. Full 4-bit quantized QLoRA training can really run here.';
    case 'directml':
      return (
        'This laptop\'s real GPU is not NVIDIA (could be AMD, Intel, or Qualcomm). Training will really use ' +
        'that GPU through Microsoft\'s DirectML, as real LoRA fine-tuning (not 4-bit quantized, since the ' +
        '4-bit library only supports NVIDIA). Still real training on the real GPU, not a simulation.'
      );
    case 'xpu':
      return 'This laptop has a real Intel GPU. Training will really use it through Intel\'s PyTorch extension.';
    case 'mps':
      return 'This laptop has a real Apple Silicon GPU. Training will really use it through Apple\'s MPS backend.';
    default:
      return (
        'No usable GPU was found for training on this laptop, so training would really run on the CPU only ' +
        '(very slow). Steps like "Validate training data" and "Register in Ollama" can still run for real.'
      );
  }
};

const Card = ({ className = '', children }) => (
  <div className={`rounded-2xl border border-slate-800 bg-slate-900 text-slate-100 ${className}`}>
    {children}
  </div>
);

const CardHeader = ({ children, className = '' }) => (
  <div className={`border-b border-slate-800 px-5 py-4 ${className}`}>{children}</div>
);

const CardTitle = ({ children, className = '' }) => (
  <h2 className={`text-lg font-semibold ${className}`}>{children}</h2>
);

const CardContent = ({ children, className = '' }) => (
  <div className={`p-5 ${className}`}>{children}</div>
);

const Label = ({ children, className = '' }) => (
  <label className={`block text-sm font-medium text-slate-200 ${className}`}>{children}</label>
);

const Input = ({ className = '', ...props }) => (
  <input
    {...props}
    className={`w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-blue-500 ${className}`}
  />
);

const Button = ({ children, className = '', variant = 'default', ...props }) => {
  const base = 'inline-flex items-center justify-center rounded-lg px-3 py-2 text-sm font-medium transition disabled:cursor-not-allowed disabled:opacity-50';
  const variants = {
    default: 'bg-blue-500 text-white hover:bg-blue-400',
    secondary: 'bg-slate-700 text-slate-100 hover:bg-slate-600',
    ghost: 'bg-transparent text-slate-300 hover:bg-slate-800',
    danger: 'bg-red-500/20 text-red-300 hover:bg-red-500/30',
  };

  return (
    <button {...props} className={`${base} ${variants[variant] || variants.default} ${className}`}>
      {children}
    </button>
  );
};

const Badge = ({ children, className = '', variant = 'default' }) => {
  const variants = {
    default: 'border border-slate-700 bg-slate-800 text-slate-200',
    outline: 'border border-slate-700 bg-transparent text-slate-300',
  };

  return (
    <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-medium ${variants[variant] || variants.default} ${className}`}>
      {children}
    </span>
  );
};

const Checkbox = ({ checked, onChange, className = '' }) => (
  <button
    type="button"
    aria-checked={checked}
    className={`flex h-4 w-4 items-center justify-center rounded border transition ${checked ? 'border-blue-500 bg-blue-500' : 'border-slate-600 bg-slate-950'} ${className}`}
    onClick={onChange}
  >
    {checked && <CheckCircle2 className="h-3 w-3 text-white" />}
  </button>
);

const Select = ({ value, onValueChange, children, className = '', 'aria-label': ariaLabel }) => (
  <select
    aria-label={ariaLabel}
    value={value}
    onChange={(e) => onValueChange(e.target.value)}
    className={`w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-blue-500 ${className}`}
  >
    {children}
  </select>
);

const SelectItem = ({ value, children, ...props }) => (
  <option value={value} {...props}>
    {children}
  </option>
);

const baseModelOptions = [
  { value: 'Qwen/Qwen2.5-Coder-7B-Instruct', label: 'Qwen 2.5 Coder 7B Instruct' },
  { value: 'Qwen/Qwen2.5-Coder-3B-Instruct', label: 'Qwen 2.5 Coder 3B Instruct' },
  { value: 'Qwen/Qwen2.5-Coder-1.5B-Instruct', label: 'Qwen 2.5 Coder 1.5B Instruct' },
  { value: 'meta-llama/Llama-3.1-8B-Instruct', label: 'Llama 3.1 8B Instruct' },
  { value: 'mistralai/Mistral-7B-Instruct-v0.3', label: 'Mistral 7B Instruct v0.3' },
  { value: 'microsoft/Phi-3.5-mini-instruct', label: 'Phi 3.5 Mini Instruct' },
  { value: 'custom', label: 'Other (type a Hugging Face model ID)' },
];

// Video/vision-language models this app can actually load with
// resolve_model_class() in scripts/train_video_vl.py. Kept intentionally
// small — Qwen2-VL/2.5-VL 2B/3B are the only sizes realistic on a laptop GPU
// without CUDA, per the model-size guard in that script.
const videoModelOptions = [
  { value: 'Qwen/Qwen2-VL-2B-Instruct', label: 'Qwen 2 VL 2B Instruct' },
  { value: 'Qwen/Qwen2.5-VL-3B-Instruct', label: 'Qwen 2.5 VL 3B Instruct' },
  { value: 'Qwen/Qwen2-VL-7B-Instruct', label: 'Qwen 2 VL 7B Instruct (needs a real NVIDIA GPU)' },
  { value: 'Qwen/Qwen2.5-VL-7B-Instruct', label: 'Qwen 2.5 VL 7B Instruct (needs a real NVIDIA GPU)' },
  { value: 'custom', label: 'Other (type a Hugging Face model ID)' },
];

// Stages that convert_to_gguf.py honestly refuses to run for video adapters
// (no GGUF export path exists yet for these vision-language architectures).
// Grayed out here too so the reason is visible before the run even starts.
const VIDEO_UNSUPPORTED_STAGE_IDS = new Set(['convert', 'quantize', 'ollama', 'smoke']);

const Progress = ({ value }) => (
  <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-800">
    <div className="h-full rounded-full bg-gradient-to-r from-blue-500 to-cyan-400 transition-all" style={{ width: `${value}%` }} />
  </div>
);

export default function App() {
  const [name, setName] = useState('azure-powershell-assistant');
  const [dataset, setDataset] = useState(null); // { path, name, sizeBytes }
  const [model, setModel] = useState('Qwen/Qwen2.5-Coder-1.5B-Instruct');
  const [customModel, setCustomModel] = useState('');
  const [maxLength, setMaxLength] = useState('2048');
  const [batchSize, setBatchSize] = useState('1');
  const [gradAccum, setGradAccum] = useState('16');
  const [rank, setRank] = useState('32');
  const [epochs, setEpochs] = useState('3');
  const [learningRate, setLearningRate] = useState('0.0002');
  const [quickTraining, setQuickTraining] = useState(true);
  const [stages, setStages] = useState(initialStages);
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState(['Pipeline ready. Connecting to the local backend...']);
  const [progress, setProgress] = useState(0);
  const [progressText, setProgressText] = useState('Not started');
  const [systemCheck, setSystemCheck] = useState(null);
  const [backendOnline, setBackendOnline] = useState(null); // null = checking, true/false = known
  const [samples, setSamples] = useState([]);
  const [validation, setValidation] = useState(null);
  const [validating, setValidating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [datasetKind, setDatasetKind] = useState('text'); // 'text' | 'video'
  const [videoModel, setVideoModel] = useState('Qwen/Qwen2-VL-2B-Instruct');
  const [customVideoModel, setCustomVideoModel] = useState('');
  const [maxFrames, setMaxFrames] = useState('8');
  const [fps, setFps] = useState('1');
  const [videoManifestFile, setVideoManifestFile] = useState(null); // File object, not yet uploaded
  const [videoClipFiles, setVideoClipFiles] = useState([]); // FileList/array, not yet uploaded
  const [checkpoints, setCheckpoints] = useState([
    { path: '', displayName: 'Start new training', step: null, savedAt: null },
  ]);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [refreshingCheckpoints, setRefreshingCheckpoints] = useState(false);
  const [agentModels, setAgentModels] = useState([]);
  const [selectedAgentModel, setSelectedAgentModel] = useState('');
  const [activeAgentModel, setActiveAgentModel] = useState(null);
  const [agentModelRoot, setAgentModelRoot] = useState('');
  const [refreshingAgentModels, setRefreshingAgentModels] = useState(false);
  const [selfTesting, setSelfTesting] = useState(false);
  const [modelSearchRoot, setModelSearchRoot] = useState('');
  const [modelDestinationRoot, setModelDestinationRoot] = useState(
    () => window.localStorage.getItem(MODEL_LIBRARY_STORAGE_KEY) || '',
  );
  const [modelFileScan, setModelFileScan] = useState(null);
  const [centralModelInventory, setCentralModelInventory] = useState([]);
  const [inventoryingCentralModels, setInventoryingCentralModels] = useState(false);
  const [scanningModelFiles, setScanningModelFiles] = useState(false);
  const [movingModelFiles, setMovingModelFiles] = useState(false);
  const datasetFileInput = useRef(null);
  const videoManifestInput = useRef(null);
  const videoClipsInput = useRef(null);
  const eventSourceRef = useRef(null);

  const selectedCount = useMemo(() => stages.filter((s) => s.selected).length, [stages]);
  const selectedAgentCandidate = useMemo(
    () => agentModels.find((candidate) => candidate.adapterPath === selectedAgentModel) || null,
    [agentModels, selectedAgentModel],
  );
  const trainableCentralModels = useMemo(
    () => centralModelInventory.filter((candidate) => candidate.format === 'Hugging Face model'),
    [centralModelInventory],
  );
  const nonTrainableCentralModels = useMemo(
    () => centralModelInventory.filter((candidate) => candidate.format !== 'Hugging Face model'),
    [centralModelInventory],
  );
  const effectiveModel = datasetKind === 'video'
    ? (videoModel === 'custom' ? customVideoModel.trim() : videoModel)
    : (model === 'custom' ? customModel.trim() : model);

  const appendLog = (line) => setLogs((old) => [...old.slice(-1999), line]);

  const runSystemCheck = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/system-check`);
      if (!response.ok) throw new Error(`Backend returned HTTP ${response.status}`);
      const result = await response.json();
      setSystemCheck(result);
      setBackendOnline(true);
      appendLog(
        `SYSTEM CHECK: CPU "${result.cpu.model}" (${result.cpu.vendor}), ${result.memory.totalGb} GB RAM, ` +
          `GPU: ${result.gpu.names.join(', ') || 'none detected'}. Python: ${result.tools.python || 'not found'}. ` +
          `NPU: ${result.npu.names.join(', ') || 'none detected'}. ` +
          `Ollama: ${result.tools.ollama || 'not found'}. WSL: ${result.tools.wsl ? 'installed' : 'not installed'}.`,
      );
    } catch (error) {
      setBackendOnline(false);
      appendLog(
        `ERROR: Could not reach the local training service at ${API_BASE}. ` +
          `Restart Local Model Lab, or start the backend when using developer mode. (${error.message})`,
      );
    }
  };

  const loadSamples = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/samples`);
      const result = await response.json();
      if (result.ok) {
        setSamples(result.samples);
      }
    } catch (error) {
      appendLog(`ERROR: Could not load sample files from the backend. (${error.message})`);
    }
  };

  const refreshCheckpoints = async () => {
    if (!name.trim()) return;
    setRefreshingCheckpoints(true);
    try {
      const response = await fetch(
        `${API_BASE}/api/checkpoints?outputName=${encodeURIComponent(name.trim())}`,
      );
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setCheckpoints(result.checkpoints);
      setSelectedCheckpoint((current) =>
        result.checkpoints.some((checkpoint) => checkpoint.path === current) ? current : '',
      );
      appendLog(`CHECKPOINTS: Found ${Math.max(0, result.checkpoints.length - 1)} resumable checkpoint(s).`);
    } catch (error) {
      appendLog(`ERROR: Could not refresh checkpoints. (${error.message})`);
    } finally {
      setRefreshingCheckpoints(false);
    }
  };

  const refreshAgentModels = async (root = agentModelRoot) => {
    setRefreshingAgentModels(true);
    try {
      const query = root.trim() ? `?root=${encodeURIComponent(root.trim())}` : '';
      const response = await fetch(`${API_BASE}/api/agent-models${query}`);
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setAgentModels(result.models);
      setAgentModelRoot(result.searchRoot);
      setActiveAgentModel(result.activeModel);
      setSelectedAgentModel((current) =>
        result.models.some((candidate) => candidate.adapterPath === current)
          ? current
          : result.activeModel?.adapterPath || result.models[0]?.adapterPath || '',
      );
      if (result.skippedDirectories?.length) {
        appendLog(
          `MODEL DISCOVERY: Found ${result.models.length} adapter(s); skipped ` +
            `${result.skippedDirectories.length} protected folder(s) that Windows did not permit access to.`,
        );
      }
      result.warnings?.forEach((warning) => appendLog(`MODEL DISCOVERY WARNING: ${warning}`));
    } catch (error) {
      appendLog(`ERROR: Could not discover post-training models. (${error.message})`);
    } finally {
      setRefreshingAgentModels(false);
    }
  };

  const activateAgentModel = async () => {
    if (!selectedAgentModel) {
      appendLog('ERROR: Select a trained adapter for agent/inference use first.');
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/api/agent-models/activate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ adapterPath: selectedAgentModel, searchRoot: agentModelRoot }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setActiveAgentModel(result.activeModel);
      appendLog(
        `AGENT MODEL: Activated ${result.activeModel.baseModel} with adapter ${result.activeModel.adapterPath}.`,
      );
    } catch (error) {
      appendLog(`ERROR: Could not activate agent model. (${error.message})`);
    }
  };

  const chooseFolder = async (initialPath, onSelected) => {
    try {
      const response = await fetch(`${API_BASE}/api/select-folder`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ initialPath }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      if (result.path) onSelected(result.path);
    } catch (error) {
      appendLog(`ERROR: Could not select a folder. (${error.message})`);
    }
  };

  const scanModelFiles = async () => {
    if (!modelSearchRoot.trim()) {
      appendLog('ERROR: Enter a source folder to search for model files.');
      return;
    }
    setScanningModelFiles(true);
    try {
      const response = await fetch(`${API_BASE}/api/model-files/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sourceRoot: modelSearchRoot.trim(),
          destinationRoot: modelDestinationRoot.trim(),
        }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setModelFileScan(result);
      setAgentModelRoot(result.sourceRoot);
      await refreshAgentModels(result.sourceRoot);
      appendLog(
        `MODEL CRAWL: Found ${result.models.length} model(s) under ${result.sourceRoot}. ` +
          `Skipped ${result.skippedSymlinks} symbolic link(s) and ` +
          `${result.skippedDirectories.length} protected folder(s).`,
      );
    } catch (error) {
      setModelFileScan(null);
      appendLog(`ERROR: Could not crawl for models. (${error.message})`);
    } finally {
      setScanningModelFiles(false);
    }
  };

  const refreshCentralModelInventory = async (root = modelDestinationRoot) => {
    const selectedRoot = root.trim();
    if (!selectedRoot) {
      setCentralModelInventory([]);
      return;
    }
    setInventoryingCentralModels(true);
    try {
      const response = await fetch(`${API_BASE}/api/model-files/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sourceRoot: selectedRoot }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setCentralModelInventory(result.models);
      appendLog(
        `MODEL LIBRARY: Inventoried ${result.models.length} model(s); ` +
          `${result.models.filter((candidate) => candidate.format === 'Hugging Face model').length} ` +
          'can be selected as a training base.',
      );
    } catch (error) {
      setCentralModelInventory([]);
      appendLog(`ERROR: Could not inventory the central model library. (${error.message})`);
    } finally {
      setInventoryingCentralModels(false);
    }
  };

  const moveModelFiles = async () => {
    if (!modelSearchRoot.trim() || !modelDestinationRoot.trim()) {
      appendLog('ERROR: Enter both source and destination folders before moving models.');
      return;
    }
    const modelCount = modelFileScan?.models?.length || 0;
    if (
      !window.confirm(
        `Move ${modelCount || 'all discovered'} model(s) from "${modelSearchRoot.trim()}" ` +
          `to "${modelDestinationRoot.trim()}"? Existing files will not be overwritten.`,
      )
    ) {
      return;
    }

    setMovingModelFiles(true);
    try {
      const response = await fetch(`${API_BASE}/api/model-files/move`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sourceRoot: modelSearchRoot.trim(),
          destinationRoot: modelDestinationRoot.trim(),
        }),
      });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setModelFileScan({ ...result, models: [] });
      appendLog(
        `MODEL MOVE: Moved ${result.moved.length} model(s) to ${result.destinationRoot}. ` +
          'Each model remains in its own subfolder; name collisions use numbered suffixes.',
      );
      setAgentModelRoot(result.destinationRoot);
      await Promise.all([
        refreshAgentModels(result.destinationRoot),
        refreshCentralModelInventory(result.destinationRoot),
      ]);
    } catch (error) {
      appendLog(`ERROR: Could not finish moving all models. (${error.message})`);
      await scanModelFiles();
    } finally {
      setMovingModelFiles(false);
    }
  };

  const runAcceleratorSelfTest = async () => {
    setSelfTesting(true);
    appendLog('ACCELERATOR: Running real DirectML LoRA and Qualcomm QNN workloads...');
    try {
      const response = await fetch(`${API_BASE}/api/accelerator-self-test`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      appendLog(
        `ACCELERATOR: GPU verified=${result.gpuVerified} (${result.gpuDevice}); ` +
          `NPU verified=${result.npuVerified} (${result.npuProvider}). ${result.details}`,
      );
    } catch (error) {
      appendLog(`ERROR: Accelerator self-test failed. (${error.message})`);
    } finally {
      setSelfTesting(false);
    }
  };

  useEffect(() => {
    runSystemCheck();
    loadSamples();
    refreshAgentModels();
    if (modelDestinationRoot) refreshCentralModelInventory(modelDestinationRoot);
    return () => {
      eventSourceRef.current?.close();
    };
  }, []);

  useEffect(() => {
    if (modelDestinationRoot) {
      window.localStorage.setItem(MODEL_LIBRARY_STORAGE_KEY, modelDestinationRoot);
    } else {
      window.localStorage.removeItem(MODEL_LIBRARY_STORAGE_KEY);
      setCentralModelInventory([]);
    }
  }, [modelDestinationRoot]);

  useEffect(() => {
    setSelectedCheckpoint('');
    setCheckpoints([{ path: '', displayName: 'Start new training', step: null, savedAt: null }]);
  }, [name]);

  // Video adapters can't go through GGUF/Ollama/smoke-test yet (see the
  // honest refusal in convert_to_gguf.py), so those steps are force-unchecked
  // whenever video mode is on, rather than letting them run and fail late.
  useEffect(() => {
    if (datasetKind !== 'video') return;
    setStages((old) =>
      old.map((s) => (VIDEO_UNSUPPORTED_STAGE_IDS.has(s.id) ? { ...s, selected: false } : s)),
    );
  }, [datasetKind]);

  const chooseDatasetFile = async (event) => {
    const [file] = event.target.files;
    if (!file) return;

    if (!file.name.toLowerCase().endsWith('.jsonl')) {
      appendLog('ERROR: Please pick a JSONL training data file.');
      event.target.value = '';
      return;
    }

    setUploading(true);
    setValidation(null);
    try {
      const formData = new FormData();
      formData.append('file', file);
      const response = await fetch(`${API_BASE}/api/upload-dataset`, { method: 'POST', body: formData });
      const result = await response.json();

      if (!response.ok || !result.ok) {
        appendLog(`ERROR: Upload failed. ${result.error || 'Unknown error.'}`);
        return;
      }

      setDataset({ path: result.path, name: file.name, sizeBytes: result.sizeBytes });
      appendLog(`DATA: Uploaded "${file.name}" (${result.sizeBytes} bytes) to the backend at ${result.path}.`);
    } catch (error) {
      appendLog(`ERROR: Could not upload the file. Is the backend running? (${error.message})`);
    } finally {
      setUploading(false);
      event.target.value = '';
    }
  };

  const chooseSampleDataset = (samplePath) => {
    if (!samplePath) return;
    const sample = samples.find((s) => s.path === samplePath);
    setDataset(sample ? { path: sample.path, name: sample.name, sizeBytes: sample.sizeBytes } : null);
    setValidation(null);
    appendLog(`DATA: Picked sample "${sample?.name}" from the backend's samples folder.`);
  };

  const chooseVideoManifestFile = (event) => {
    const [file] = event.target.files;
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.jsonl')) {
      appendLog('ERROR: The video manifest must be a .jsonl file.');
      event.target.value = '';
      return;
    }
    setVideoManifestFile(file);
    setValidation(null);
    setDataset(null);
  };

  const chooseVideoClipFiles = (event) => {
    const files = Array.from(event.target.files || []);
    setVideoClipFiles(files);
    setValidation(null);
    setDataset(null);
  };

  const uploadVideoDataset = async () => {
    if (!videoManifestFile) {
      appendLog('ERROR: Pick a manifest .jsonl file first.');
      return;
    }
    if (videoClipFiles.length === 0) {
      appendLog('ERROR: Pick the real video clip files referenced by the manifest.');
      return;
    }

    setUploading(true);
    setValidation(null);
    try {
      const formData = new FormData();
      formData.append('manifest', videoManifestFile);
      videoClipFiles.forEach((file) => formData.append('videos', file));

      const response = await fetch(`${API_BASE}/api/upload-video-dataset`, { method: 'POST', body: formData });
      const result = await response.json();

      if (!response.ok || !result.ok) {
        appendLog(`ERROR: Video dataset upload failed. ${result.error || 'Unknown error.'}`);
        return;
      }

      setDataset({
        path: result.manifestPath,
        videosDir: result.videosDir,
        name: videoManifestFile.name,
        videoCount: result.videoCount,
      });
      appendLog(
        `DATA: Uploaded manifest "${videoManifestFile.name}" and ${result.videoCount} real video file(s) to ${result.videosDir}.`,
      );
    } catch (error) {
      appendLog(`ERROR: Could not upload the video dataset. Is the backend running? (${error.message})`);
    } finally {
      setUploading(false);
    }
  };

  const validateDataset = async () => {
    if (!dataset) {
      appendLog('ERROR: Pick a training data file first.');
      return;
    }

    setValidating(true);
    try {
      const response = await fetch(`${API_BASE}/api/validate-dataset`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          datasetPath: dataset.path,
          kind: datasetKind,
          videosDir: dataset.videosDir,
        }),
      });
      const result = await response.json();
      setValidation(result);
      appendLog(
        result.ok
          ? `VALIDATE: "${dataset.name}" is real and valid — ${result.valid_records}/${result.total_records} records passed.`
          : `VALIDATE: "${dataset.name}" has real problems — ${result.invalid_records}/${result.total_records} records failed. First error: ${result.errors?.[0] || 'unknown'}`,
      );
    } catch (error) {
      appendLog(`ERROR: Could not validate the dataset. Is the backend running? (${error.message})`);
    } finally {
      setValidating(false);
    }
  };

  const toggleStage = (id) => {
    if (datasetKind === 'video' && VIDEO_UNSUPPORTED_STAGE_IDS.has(id)) return;
    setStages((old) => old.map((s) => (s.id === id ? { ...s, selected: !s.selected } : s)));
  };

  const streamRun = (runId, selectedStages) => {
    const source = new EventSource(`${API_BASE}/api/run/${runId}/events`);
    eventSourceRef.current = source;
    let completedCount = 0;

    source.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'stage-start') {
        setStages((old) => old.map((s) => (s.id === data.stageId ? { ...s, status: 'running' } : s)));
        appendLog(`START ${data.name}`);
      } else if (data.type === 'stage-log') {
        appendLog(`  ${data.line}`);
      } else if (data.type === 'training-progress') {
        setProgress(Math.max(0, Math.min(100, Number(data.percent) || 0)));
        setProgressText(
          data.currentStep == null
            ? `${data.message} (${Number(data.percent).toFixed(1)}%)`
            : `${data.message} (${Number(data.percent).toFixed(1)}%)`,
        );
      } else if (data.type === 'stage-end') {
        setStages((old) => old.map((s) => (s.id === data.stageId ? { ...s, status: data.status } : s)));
        appendLog(`${data.status === 'complete' ? 'DONE ' : 'FAILED'} ${data.stageId} (real exit code: ${data.exitCode})`);
        completedCount += 1;
        setProgress(Math.round((completedCount / selectedStages.length) * 100));
      } else if (data.type === 'run-complete') {
        const anyFailed = Object.values(data.stageResults).some((status) => status === 'failed');
        appendLog(anyFailed ? `RUN FINISHED WITH REAL FAILURES: ${name}` : `RUN SUCCEEDED FOR REAL: ${name}`);
        setRunning(false);
        setProgressText(anyFailed ? 'Training or pipeline failed' : 'Pipeline complete');
        refreshCheckpoints();
        refreshAgentModels();
        source.close();
      }
    };

    source.onerror = () => {
      appendLog('ERROR: Lost connection to the real-time event stream from the backend.');
      setRunning(false);
      source.close();
    };
  };

  const runPipeline = async () => {
    if (!dataset) {
      appendLog(
        datasetKind === 'video'
          ? 'ERROR: Upload a video manifest and video clips before starting.'
          : 'ERROR: Pick a JSONL training dataset before starting.',
      );
      return;
    }

    if (!effectiveModel) {
      appendLog('ERROR: Enter a base model ID before starting.');
      return;
    }

    if (backendOnline === false) {
      appendLog(`ERROR: The real backend is not reachable at ${API_BASE}. Nothing can run without it.`);
      return;
    }

    const selected = stages.filter((s) => s.selected);
    if (selected.length === 0) {
      appendLog('ERROR: Turn on at least one step before running.');
      return;
    }

    setRunning(true);
    setProgress(0);
    setProgressText('Starting dependency and environment checks');
    setStages((old) => old.map((s) => (s.selected ? { ...s, status: 'ready' } : s)));

    try {
      const response = await fetch(`${API_BASE}/api/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          stages: selected.map((s) => ({ id: s.id, name: s.name })),
          config: {
            name,
            model: effectiveModel,
            datasetPath: dataset.path,
            datasetKind,
            videosDir: dataset.videosDir,
            maxFrames: Number(maxFrames),
            fps: Number(fps),
            maxLength: Number(maxLength),
            batchSize: Number(batchSize),
            gradAccum: Number(gradAccum),
            rank: Number(rank),
            epochs: Number(epochs),
            learningRate: Number(learningRate),
            resumeCheckpoint: datasetKind === 'text' ? selectedCheckpoint || null : null,
            quickTraining,
          },
        }),
      });
      const result = await response.json();

      if (!response.ok || !result.ok) {
        appendLog(`ERROR: Could not start the real run. ${result.error || 'Unknown error.'}`);
        setRunning(false);
        return;
      }

      appendLog(`RUN STARTED for real (run id ${result.runId}). Streaming real output below.`);
      streamRun(result.runId, selected);
    } catch (error) {
      appendLog(`ERROR: Could not reach the backend to start the run. (${error.message})`);
      setRunning(false);
    }
  };

  const retryFailed = () => {
    const failedIds = stages.filter((s) => s.status === 'failed').map((s) => s.id);
    if (failedIds.length === 0) {
      appendLog('No real failed steps to retry.');
      return;
    }
    setStages((old) => old.map((s) => ({ ...s, selected: failedIds.includes(s.id) })));
    appendLog(`Selected the real failed step(s) for a retry: ${failedIds.join(', ')}. Press "Run now" to retry them for real.`);
  };

  return (
    <div className="min-h-screen bg-slate-950 p-5 text-slate-100">
      <div className="mx-auto max-w-7xl space-y-5">
        <header className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="flex items-center gap-2 text-blue-400">
              <Cpu className="h-5 w-5" />
              <span className="text-sm font-semibold uppercase tracking-[0.2em]">Local Model Lab</span>
            </div>
            <h1 className="mt-2 text-3xl font-bold">QLoRA Training Pipeline Manager</h1>
            <p className="mt-1 text-slate-400">
              Every button here calls a real local backend and runs real commands. Nothing is simulated.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Badge variant="outline" className={backendOnline === false ? 'border-red-700 text-red-300' : ''}>
              Backend: {backendOnline === null ? 'checking...' : backendOnline ? 'online' : 'offline'}
            </Badge>
            {systemCheck && (
              <>
                <Badge variant="outline">
                  GPU: {systemCheck.gpu.names[0] || 'none detected'}
                  {systemCheck.gpu.recommendedBackend ? ` (${systemCheck.gpu.recommendedBackend})` : ''}
                </Badge>
                <Badge variant="outline">WSL2: {systemCheck.tools.wsl ? 'installed' : 'not installed'}</Badge>
              </>
            )}
          </div>
        </header>

        {backendOnline === false && (
          <Card>
            <CardContent className="text-sm text-red-300">
              The real backend is not running. Open a terminal, run <code className="rounded bg-slate-950 px-1 py-0.5">cd server</code> then{' '}
              <code className="rounded bg-slate-950 px-1 py-0.5">npm start</code>, then click "Check this laptop again" below.
            </CardContent>
          </Card>
        )}

        <div className="grid gap-5 lg:grid-cols-3">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle>Pick your settings</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2 md:col-span-2">
                <Label>Training data type</Label>
                <div className="flex gap-2">
                  <Button
                    variant={datasetKind === 'text' ? 'default' : 'secondary'}
                    onClick={() => setDatasetKind('text')}
                  >
                    Text (chat JSONL)
                  </Button>
                  <Button
                    variant={datasetKind === 'video' ? 'default' : 'secondary'}
                    onClick={() => setDatasetKind('video')}
                  >
                    Video (vision-language)
                  </Button>
                </div>
                {datasetKind === 'video' && (
                  <p className="text-xs text-amber-300">
                    Real vision-language fine-tuning. Needs ffmpeg installed for real frame extraction, a separate
                    Python environment, and GGUF/Ollama/smoke-test are not available yet for these models (see below).
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label>Model name</Label>
                <Input value={name} onChange={(e) => setName(e.target.value)} />
              </div>

              {datasetKind === 'video' ? (
                <div className="space-y-2">
                  <Label>Base video-language model</Label>
                  <Select value={videoModel} onValueChange={setVideoModel}>
                    {videoModelOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                    {trainableCentralModels.length > 0 && (
                      <optgroup label="Central model library">
                        {trainableCentralModels.map((candidate) => (
                          <SelectItem key={candidate.sourcePath} value={candidate.sourcePath}>
                            {candidate.name} - local {candidate.format}
                          </SelectItem>
                        ))}
                      </optgroup>
                    )}
                    {nonTrainableCentralModels.length > 0 && (
                      <optgroup label="Central library (not usable as a training base)">
                        {nonTrainableCentralModels.map((candidate) => (
                          <SelectItem key={candidate.sourcePath} value={candidate.sourcePath} disabled>
                            {candidate.name} - {candidate.format}
                          </SelectItem>
                        ))}
                      </optgroup>
                    )}
                  </Select>
                  {videoModel === 'custom' && (
                    <Input
                      value={customVideoModel}
                      onChange={(e) => setCustomVideoModel(e.target.value)}
                      placeholder="e.g. org-name/model-name"
                      className="mt-2"
                    />
                  )}
                </div>
              ) : (
                <div className="space-y-2">
                  <Label>Base model</Label>
                  <Select value={model} onValueChange={setModel}>
                    {baseModelOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                    {trainableCentralModels.length > 0 && (
                      <optgroup label="Central model library">
                        {trainableCentralModels.map((candidate) => (
                          <SelectItem key={candidate.sourcePath} value={candidate.sourcePath}>
                            {candidate.name} - local {candidate.format}
                          </SelectItem>
                        ))}
                      </optgroup>
                    )}
                    {nonTrainableCentralModels.length > 0 && (
                      <optgroup label="Central library (not usable as a training base)">
                        {nonTrainableCentralModels.map((candidate) => (
                          <SelectItem key={candidate.sourcePath} value={candidate.sourcePath} disabled>
                            {candidate.name} - {candidate.format}
                          </SelectItem>
                        ))}
                      </optgroup>
                    )}
                  </Select>
                  {model === 'custom' && (
                    <Input
                      value={customModel}
                      onChange={(e) => setCustomModel(e.target.value)}
                      placeholder="e.g. org-name/model-name"
                      className="mt-2"
                    />
                  )}
                </div>
              )}

              {datasetKind === 'video' ? (
                <div className="space-y-2 md:col-span-2">
                  <Label>Video manifest + clips</Label>
                  <div className="flex flex-col gap-2 sm:flex-row">
                    <Input
                      value={videoManifestFile ? videoManifestFile.name : ''}
                      readOnly
                      placeholder="No manifest picked yet"
                    />
                    <input
                      ref={videoManifestInput}
                      type="file"
                      accept=".jsonl,application/jsonl,application/x-ndjson"
                      className="sr-only"
                      onChange={chooseVideoManifestFile}
                    />
                    <Button variant="secondary" onClick={() => videoManifestInput.current?.click()}>
                      <Upload className="mr-2 h-4 w-4" />
                      Browse manifest
                    </Button>
                  </div>
                  <div className="flex flex-col gap-2 sm:flex-row">
                    <Input
                      value={videoClipFiles.length ? `${videoClipFiles.length} video file(s) picked` : ''}
                      readOnly
                      placeholder="No video clips picked yet"
                    />
                    <input
                      ref={videoClipsInput}
                      type="file"
                      accept=".mp4,.mov,.mkv,.webm,.avi,.m4v,.mpeg,.mpg,video/*"
                      multiple
                      className="sr-only"
                      onChange={chooseVideoClipFiles}
                    />
                    <Button variant="secondary" onClick={() => videoClipsInput.current?.click()}>
                      <Upload className="mr-2 h-4 w-4" />
                      Browse video clips
                    </Button>
                    <Button
                      variant="ghost"
                      className="border border-slate-800"
                      onClick={uploadVideoDataset}
                      disabled={uploading || !videoManifestFile || videoClipFiles.length === 0}
                    >
                      <Upload className="mr-2 h-4 w-4" />
                      {uploading ? 'Uploading...' : 'Upload dataset'}
                    </Button>
                    <Button
                      variant="ghost"
                      className="border border-slate-800"
                      onClick={validateDataset}
                      disabled={!dataset || validating}
                    >
                      <FileCheck2 className="mr-2 h-4 w-4" />
                      {validating ? 'Checking...' : 'Check this dataset'}
                    </Button>
                  </div>
                  <p className="text-xs text-slate-400">
                    Manifest is a real JSONL file where each line has <code>video</code> (a clip filename), and{' '}
                    <code>messages</code> (system/user/assistant chat turns describing that clip). Pick the real
                    manifest, then every real video clip file it references, then "Upload dataset".
                  </p>
                  {dataset?.videoCount && (
                    <p className="text-xs text-emerald-300">
                      Uploaded: "{dataset.name}" referencing {dataset.videoCount} real video file(s) at{' '}
                      {dataset.videosDir}.
                    </p>
                  )}
                  {validation && (
                    <div
                      className={`rounded-lg border p-2 text-xs ${
                        validation.ok
                          ? 'border-emerald-800 bg-emerald-950/30 text-emerald-300'
                          : 'border-red-800 bg-red-950/30 text-red-300'
                      }`}
                    >
                      {validation.ok ? (
                        <span>
                          Real check passed: {validation.valid_records}/{validation.total_records} records are valid.
                        </span>
                      ) : (
                        <div>
                          <p>
                            Real check found problems: {validation.invalid_records}/{validation.total_records} records
                            failed.
                          </p>
                          <ul className="mt-1 list-disc pl-4">
                            {validation.errors?.slice(0, 5).map((err) => (
                              <li key={err}>{err}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ) : (
                <div className="space-y-2 md:col-span-2">
                  <Label>Training data file</Label>
                  <div className="flex flex-col gap-2 sm:flex-row">
                    <Input value={dataset ? `${dataset.name} (${dataset.sizeBytes} bytes)` : ''} readOnly placeholder="No file picked yet" />
                    <input
                      ref={datasetFileInput}
                      type="file"
                      accept=".jsonl,application/jsonl,application/x-ndjson"
                      className="sr-only"
                      onChange={chooseDatasetFile}
                    />
                    <Button variant="secondary" onClick={() => datasetFileInput.current?.click()} disabled={uploading}>
                      <Upload className="mr-2 h-4 w-4" />
                      {uploading ? 'Uploading...' : 'Browse files'}
                    </Button>
                    <Button variant="ghost" className="border border-slate-800" onClick={validateDataset} disabled={!dataset || validating}>
                      <FileCheck2 className="mr-2 h-4 w-4" />
                      {validating ? 'Checking...' : 'Check this file'}
                    </Button>
                  </div>
                  <p className="text-xs text-slate-400">
                    "Browse files" opens your real file manager and really uploads the file to the local backend on this laptop.
                  </p>
                  <Select value="" onValueChange={chooseSampleDataset} aria-label="Choose a sample data file">
                    <SelectItem value="" disabled>
                      Or pick a real sample data file from this repo
                    </SelectItem>
                    {samples.map((sample) => (
                      <SelectItem key={sample.path} value={sample.path}>
                        {sample.name} ({sample.sizeBytes} bytes)
                      </SelectItem>
                    ))}
                  </Select>
                  {validation && (
                    <div
                      className={`rounded-lg border p-2 text-xs ${
                        validation.ok ? 'border-emerald-800 bg-emerald-950/30 text-emerald-300' : 'border-red-800 bg-red-950/30 text-red-300'
                      }`}
                    >
                      {validation.ok ? (
                        <span>
                          Real check passed: {validation.valid_records}/{validation.total_records} records are valid.
                        </span>
                      ) : (
                        <div>
                          <p>
                            Real check found problems: {validation.invalid_records}/{validation.total_records} records failed.
                          </p>
                          <ul className="mt-1 list-disc pl-4">
                            {validation.errors?.slice(0, 5).map((err) => (
                              <li key={err}>{err}</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {datasetKind === 'video' ? (
                <>
                  <div className="space-y-2">
                    <Label>Max frames per clip</Label>
                    <Input value={maxFrames} onChange={(e) => setMaxFrames(e.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <Label>Sampling FPS</Label>
                    <Input value={fps} onChange={(e) => setFps(e.target.value)} />
                  </div>
                </>
              ) : (
                <>
                  <div className="space-y-2">
                    <Label>Max length</Label>
                    <Input value={maxLength} onChange={(e) => setMaxLength(e.target.value)} />
                  </div>

                  <div className="space-y-2">
                    <Label>Batch size</Label>
                    <Input value={batchSize} onChange={(e) => setBatchSize(e.target.value)} />
                  </div>
                </>
              )}

              <div className="space-y-2">
                <Label>Grad. steps</Label>
                <Input value={gradAccum} onChange={(e) => setGradAccum(e.target.value)} />
              </div>

              <div className="space-y-2">
                <Label>LoRA rank</Label>
                <Input value={rank} onChange={(e) => setRank(e.target.value)} />
              </div>

              <div className="space-y-2">
                <Label>Epochs</Label>
                <Input value={epochs} onChange={(e) => setEpochs(e.target.value)} />
              </div>

              <div className="space-y-2">
                <Label>Learning rate</Label>
                <Input value={learningRate} onChange={(e) => setLearningRate(e.target.value)} />
              </div>

              {datasetKind === 'text' && (
                <div className="space-y-2 md:col-span-2">
                  <Label>Resume checkpoint</Label>
                  <div className="flex gap-2">
                    <Select
                      value={selectedCheckpoint}
                      onValueChange={setSelectedCheckpoint}
                      aria-label="Resume checkpoint"
                    >
                      {checkpoints.map((checkpoint) => (
                        <SelectItem key={checkpoint.path || 'new'} value={checkpoint.path}>
                          {checkpoint.displayName}
                        </SelectItem>
                      ))}
                    </Select>
                    <Button
                      variant="secondary"
                      onClick={refreshCheckpoints}
                      disabled={running || refreshingCheckpoints || !name.trim()}
                    >
                      <RefreshCw className={`mr-2 h-4 w-4 ${refreshingCheckpoints ? 'animate-spin' : ''}`} />
                      Refresh
                    </Button>
                  </div>
                  <p className="text-xs text-slate-400">
                    Start new training deliberately ignores existing checkpoints. Resumes restore adapter,
                    optimizer, scheduler, RNG, and completed-step state.
                  </p>
                </div>
              )}

              <label className="md:col-span-2 flex items-start gap-3 rounded-lg border border-slate-800 bg-slate-950 p-3 text-sm text-slate-300">
                <input
                  type="checkbox"
                  checked={quickTraining}
                  onChange={(e) => setQuickTraining(e.target.checked)}
                  className="mt-1"
                />
                <span>
                  <span className="block font-medium text-slate-100">Quick smoke training</span>
                  <span className="text-xs text-slate-400">
                    Recommended on this laptop. Runs a real tiny LoRA training pass with 8 records, max length 512, and 1 training step so you can prove the pipeline works before starting a long full run.
                  </span>
                </span>
              </label>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MemoryStick className="h-5 w-5 text-violet-400" />
                Real system check
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {systemCheck ? (
                <>
                  <div className="grid gap-2 text-sm text-slate-300">
                    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                      <span>CPU</span>
                      <span className="text-right text-slate-400">
                        {systemCheck.cpu.model} ({systemCheck.cpu.vendor})
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                      <span>RAM</span>
                      <span className="text-slate-400">
                        {systemCheck.memory.totalGb} GB total, {systemCheck.memory.freeGb} GB free
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                      <span>Graphics</span>
                      <span className="text-right text-slate-400">{systemCheck.gpu.names.join(', ') || 'none detected'}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                      <span>NPU inference</span>
                      <span className="text-right text-slate-400">
                        {systemCheck.npu.names.join(', ') || 'none detected'}
                      </span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                      <span>Python</span>
                      <span className="text-slate-400">{systemCheck.tools.python || 'not installed'}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                      <span>Ollama</span>
                      <span className="text-slate-400">{systemCheck.tools.ollama || 'not installed'}</span>
                    </div>
                    <div className="flex items-center justify-between rounded-lg border border-slate-800 bg-slate-950 px-3 py-2">
                      <span>WSL2</span>
                      <span className="text-slate-400">{systemCheck.tools.wsl ? 'installed' : 'not installed'}</span>
                    </div>
                  </div>

                  <div
                    className={`rounded-xl border p-3 text-sm ${
                      systemCheck.canTrainLocally
                        ? 'border-emerald-800 bg-emerald-950/30 text-emerald-300'
                        : 'border-amber-800 bg-amber-950/30 text-amber-300'
                    }`}
                  >
                    {backendDescription(systemCheck.gpu.recommendedBackend)}
                  </div>
                </>
              ) : (
                <p className="text-sm text-slate-400">Checking this laptop for real hardware and tools...</p>
              )}

              <Button variant="secondary" className="w-full" onClick={runSystemCheck}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Check this laptop again
              </Button>
              <Button
                variant="secondary"
                className="w-full"
                onClick={runAcceleratorSelfTest}
                disabled={selfTesting}
              >
                <Cpu className="mr-2 h-4 w-4" />
                {selfTesting ? 'Testing real accelerators...' : 'Run real accelerator self-test'}
              </Button>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Bot className="h-5 w-5 text-fuchsia-400" />
              Post-training Agent / Inference model
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <Label className="mb-2">Discovery starting folder</Label>
              <div className="grid gap-3 md:grid-cols-[1fr_auto_auto]">
                <Input
                  value={agentModelRoot}
                  onChange={(event) => setAgentModelRoot(event.target.value)}
                  placeholder="Choose the parent folder containing trained models"
                  aria-label="Agent model discovery starting folder"
                />
                <Button
                  variant="secondary"
                  onClick={() =>
                    chooseFolder(agentModelRoot, (selectedPath) => {
                      setAgentModelRoot(selectedPath);
                      refreshAgentModels(selectedPath);
                    })
                  }
                >
                  <FolderSearch className="mr-2 h-4 w-4" />
                  Browse
                </Button>
                <Button
                  variant="secondary"
                  onClick={() => refreshAgentModels()}
                  disabled={!agentModelRoot.trim() || refreshingAgentModels}
                >
                  <RefreshCw className={`mr-2 h-4 w-4 ${refreshingAgentModels ? 'animate-spin' : ''}`} />
                  Discover recursively
                </Button>
              </div>
              <p className="mt-2 text-xs text-slate-400">
                Select any parent folder. All complete Hugging Face adapters and training checkpoints beneath it remain distinct.
              </p>
            </div>
            <div className="grid gap-3 md:grid-cols-[1fr_auto]">
              <div>
              <Label className="mb-2">Trained adapter used by an agent after training</Label>
              <Select
                value={selectedAgentModel}
                onValueChange={setSelectedAgentModel}
                aria-label="Agent inference model"
              >
                <SelectItem value="" disabled>
                  No trained adapters discovered
                </SelectItem>
                {agentModels.map((candidate) => (
                  <SelectItem key={candidate.adapterPath} value={candidate.adapterPath}>
                    {candidate.displayName} - {candidate.baseModel}
                    {candidate.trainingStep ? ` (step ${candidate.trainingStep})` : ''}
                  </SelectItem>
                ))}
              </Select>
              <p className="mt-2 text-xs text-slate-400">
                This is independent of the Base model field above, which only chooses what to fine-tune.
              </p>
              <p className="mt-1 text-xs text-emerald-300">
                Active: {activeAgentModel
                  ? `${activeAgentModel.baseModel} + ${activeAgentModel.adapterPath}`
                  : 'No post-training model selected'}
              </p>
              </div>
              <Button className="self-end" onClick={activateAgentModel} disabled={!selectedAgentModel}>
                Activate for inference
              </Button>
            </div>
            {selectedAgentCandidate && (
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <div className="mb-3 flex flex-wrap items-center gap-2">
                  <Badge>{selectedAgentCandidate.trainingMetadata.modality || 'Unknown modality'}</Badge>
                  <Badge variant="outline">{selectedAgentCandidate.artifactType}</Badge>
                  {selectedAgentCandidate.trainingMetadata.trainingKind && (
                    <Badge variant="outline">{selectedAgentCandidate.trainingMetadata.trainingKind}</Badge>
                  )}
                </div>
                <div className="grid gap-2 text-xs sm:grid-cols-2 xl:grid-cols-4">
                  {[
                    ['Base model', selectedAgentCandidate.baseModel],
                    ['Dataset', selectedAgentCandidate.trainingMetadata.datasetFileName],
                    ['Dataset SHA-256', selectedAgentCandidate.trainingMetadata.datasetSha256],
                    ['Backend', selectedAgentCandidate.trainingMetadata.backend],
                    ['Device', selectedAgentCandidate.trainingMetadata.deviceName],
                    ['Progress', selectedAgentCandidate.trainingMetadata.globalStep != null
                      ? `${selectedAgentCandidate.trainingMetadata.globalStep}/${selectedAgentCandidate.trainingMetadata.maxSteps ?? '?'} steps`
                      : null],
                    ['Epochs', selectedAgentCandidate.trainingMetadata.epochs],
                    ['Last loss', selectedAgentCandidate.trainingMetadata.lastLoss],
                    ['LoRA rank / alpha', selectedAgentCandidate.trainingMetadata.loraRank != null
                      ? `${selectedAgentCandidate.trainingMetadata.loraRank} / ${selectedAgentCandidate.trainingMetadata.loraAlpha ?? '?'}`
                      : null],
                    ['Batch / accumulation', selectedAgentCandidate.trainingMetadata.batchSize != null
                      ? `${selectedAgentCandidate.trainingMetadata.batchSize} / ${selectedAgentCandidate.trainingMetadata.gradientAccumulationSteps ?? '?'}`
                      : null],
                    ['Learning rate', selectedAgentCandidate.trainingMetadata.learningRate],
                    ['Context length', selectedAgentCandidate.trainingMetadata.maxLength],
                    ['Video frames / FPS', selectedAgentCandidate.trainingMetadata.maxFrames != null
                      ? `${selectedAgentCandidate.trainingMetadata.maxFrames} / ${selectedAgentCandidate.trainingMetadata.fps ?? '?'}`
                      : null],
                    ['Saved', new Date(selectedAgentCandidate.savedAt).toLocaleString()],
                  ].filter(([, value]) => value !== null && value !== undefined && value !== '').map(([label, value]) => (
                    <div key={label} className="rounded-lg border border-slate-800 px-3 py-2">
                      <div className="text-slate-500">{label}</div>
                      <div className="mt-1 break-all text-slate-200">{String(value)}</div>
                    </div>
                  ))}
                </div>
                {selectedAgentCandidate.trainingMetadata.targetModules?.length > 0 && (
                  <p className="mt-3 text-xs text-slate-400">
                    Target modules: {selectedAgentCandidate.trainingMetadata.targetModules.join(', ')}
                  </p>
                )}
                {selectedAgentCandidate.trainingMetadata.warnings.map((warning) => (
                  <p key={warning} className="mt-2 text-xs text-amber-300">{warning}</p>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FolderSearch className="h-5 w-5 text-cyan-400" />
              Find and consolidate local models
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <div>
                <Label className="mb-2">Folder to crawl</Label>
                <div className="flex gap-2">
                  <Input
                    value={modelSearchRoot}
                    onChange={(event) => {
                      setModelSearchRoot(event.target.value);
                      setModelFileScan(null);
                    }}
                    placeholder="C:\path\to\search"
                    aria-label="Folder to crawl for models"
                  />
                  <Button
                    variant="secondary"
                    onClick={() =>
                      chooseFolder(modelSearchRoot, (selectedPath) => {
                        setModelSearchRoot(selectedPath);
                        setModelFileScan(null);
                      })
                    }
                  >
                    Browse
                  </Button>
                </div>
              </div>
              <div>
                <Label className="mb-2">Single destination folder</Label>
                <div className="flex gap-2">
                  <Input
                    value={modelDestinationRoot}
                    onChange={(event) => {
                      setModelDestinationRoot(event.target.value);
                      setModelFileScan(null);
                    }}
                    placeholder="C:\path\to\model-library"
                    aria-label="Model destination folder"
                  />
                  <Button
                    variant="secondary"
                    onClick={() =>
                      chooseFolder(modelDestinationRoot, (selectedPath) => {
                        setModelDestinationRoot(selectedPath);
                        setModelFileScan(null);
                        refreshCentralModelInventory(selectedPath);
                      })
                    }
                  >
                    Browse
                  </Button>
                </div>
              </div>
            </div>
            <p className="text-xs text-slate-400">
              Finds Hugging Face, adapter, ONNX, TensorFlow, OpenVINO, Core ML, GGUF, SafeTensors,
              PyTorch, and TensorFlow Lite models. Every model stays distinct in its own destination
              subfolder, model bundles stay together, symbolic links are skipped, and existing items
              are never overwritten.
            </p>
            <div className="flex flex-wrap gap-3">
              <Button
                variant="secondary"
                onClick={scanModelFiles}
                disabled={!modelSearchRoot.trim() || scanningModelFiles || movingModelFiles}
              >
                <FolderSearch className={`mr-2 h-4 w-4 ${scanningModelFiles ? 'animate-pulse' : ''}`} />
                {scanningModelFiles ? 'Crawling folders...' : 'Scan for models'}
              </Button>
              <Button
                onClick={moveModelFiles}
                disabled={
                  !modelSearchRoot.trim() ||
                  !modelDestinationRoot.trim() ||
                  !modelFileScan?.models?.length ||
                  scanningModelFiles ||
                  movingModelFiles
                }
              >
                <Move className="mr-2 h-4 w-4" />
                {movingModelFiles ? 'Moving models...' : 'Move all found models'}
              </Button>
              <Button
                variant="secondary"
                onClick={() => refreshCentralModelInventory()}
                disabled={!modelDestinationRoot.trim() || inventoryingCentralModels || movingModelFiles}
              >
                <RefreshCw className={`mr-2 h-4 w-4 ${inventoryingCentralModels ? 'animate-spin' : ''}`} />
                {inventoryingCentralModels ? 'Inventorying...' : 'Inventory destination'}
              </Button>
            </div>
            {modelDestinationRoot && (
              <p className="text-xs text-slate-400">
                Central library inventory: {centralModelInventory.length} model(s), including{' '}
                {trainableCentralModels.length} Hugging Face base model(s) available in the training
                model selectors. Other artifact types remain visible there but are disabled because
                they cannot be loaded as a Transformers training base.
              </p>
            )}
            {modelFileScan && (
              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3">
                <p className="text-sm text-slate-200">
                  Found {modelFileScan.models?.length ?? modelFileScan.foundCount ?? 0} model(s).
                  {modelFileScan.skippedSymlinks > 0
                    ? ` Skipped ${modelFileScan.skippedSymlinks} symbolic link(s).`
                    : ''}
                  {modelFileScan.skippedDirectories?.length > 0
                    ? ` Skipped ${modelFileScan.skippedDirectories.length} protected folder(s).`
                    : ''}
                </p>
                {modelFileScan.models?.length > 0 && (
                  <ul className="mt-3 max-h-56 space-y-2 overflow-auto text-xs text-slate-400">
                    {modelFileScan.models.map((candidate) => (
                      <li key={candidate.sourcePath}>
                        <span className="text-cyan-300">{candidate.format}</span>
                        {' - '}
                        {candidate.sourcePath}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </CardContent>
        </Card>

        <div className="grid gap-5 xl:grid-cols-[1.2fr_0.8fr]">
          <Card>
            <CardHeader>
              <CardTitle>Steps to run</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {stages.map((stage) => {
                const videoUnsupported = datasetKind === 'video' && VIDEO_UNSUPPORTED_STAGE_IDS.has(stage.id);
                return (
                  <motion.div
                    key={stage.id}
                    layout
                    className={`flex items-center justify-between rounded-xl border border-slate-800 bg-slate-950 px-3 py-2 ${videoUnsupported ? 'opacity-40' : ''}`}
                    title={videoUnsupported ? 'Not available for video models yet: llama.cpp/GGUF has no export path for this vision-language architecture.' : undefined}
                  >
                    <div className="flex items-center gap-3">
                      <Checkbox checked={stage.selected} onChange={() => toggleStage(stage.id)} className={videoUnsupported ? 'cursor-not-allowed' : ''} />
                      <span className="text-sm text-slate-200">
                        {stage.name}
                        {videoUnsupported && <span className="ml-2 text-xs text-amber-400">(text-only for now)</span>}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      {statusIcon(stage.status)}
                      <span className="text-xs uppercase tracking-wide text-slate-400">{stage.status}</span>
                    </div>
                  </motion.div>
                );
              })}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Square className="h-5 w-5 text-amber-400" />
                  Quick start
                </span>
                <Badge variant="default" className="bg-emerald-500/10 text-emerald-300">
                  {selectedCount} on
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <ol className="list-decimal space-y-2 pl-5 text-sm text-slate-300">
                <li>Pick a data file (uploaded for real).</li>
                <li>Pick a base model.</li>
                <li>Turn on the steps you want.</li>
                <li>Press the run button to run real commands.</li>
              </ol>

              <Button className="w-full" onClick={runPipeline} disabled={running || backendOnline === false}>
                <Play className="mr-2 h-4 w-4" />
                {running ? 'Running for real...' : 'Run now'}
              </Button>

              <Button variant="ghost" className="w-full border border-slate-800" onClick={retryFailed} disabled={running}>
                <RefreshCw className="mr-2 h-4 w-4" />
                Retry real failed steps
              </Button>

              <div className="rounded-xl border border-slate-800 bg-slate-950 p-3 text-sm text-slate-300">
                <div className="mb-2 flex items-center justify-between">
                  <span>Work done</span>
                  <span>{progress}%</span>
                </div>
                <Progress value={progress} />
                <p className="mt-2 text-xs text-slate-400">{progressText}</p>
              </div>
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Terminal className="h-5 w-5 text-cyan-400" />
              Real command output
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="max-h-96 overflow-y-auto rounded-xl border border-slate-800 bg-slate-950 p-4 font-mono text-sm text-slate-300">
              <div className="mb-2 flex items-center gap-2 text-slate-400">
                <FileCheck2 className="h-4 w-4" />
                <span>Model lab log (real backend responses and real process output)</span>
              </div>
              <div className="space-y-1">
                {logs.map((line, index) => (
                  <div key={`${line}-${index}`} className="whitespace-pre-wrap">
                    {line}
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
