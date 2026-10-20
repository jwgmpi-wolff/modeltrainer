import React, { useEffect, useRef, useState } from 'react';
import { Download, RefreshCw, Square } from 'lucide-react';

const STORAGE_KEY = 'local-model-lab.conversations.v1';
const controlClass = 'w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100';

function loadHistory() {
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { threads: [], prompts: [], error: '' };
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.threads) || !Array.isArray(parsed.prompts) ||
        parsed.prompts.some((prompt) => typeof prompt !== 'string') ||
        parsed.threads.some((thread) => typeof thread.id !== 'string' || typeof thread.title !== 'string' ||
          typeof thread.model !== 'string' || !Array.isArray(thread.messages) ||
          thread.messages.some((message) => !['user', 'assistant'].includes(message.role) ||
            typeof message.content !== 'string'))) {
      throw new Error('Saved conversation data has an invalid format.');
    }
    return { ...parsed, error: '' };
  } catch (error) {
    return { threads: [], prompts: [], error: `Could not restore local history: ${error.message}` };
  }
}

export default function Conversation({ apiBase }) {
  const [initial] = useState(loadHistory);
  const [threads, setThreads] = useState(initial.threads);
  const [prompts, setPrompts] = useState(initial.prompts);
  const [threadId, setThreadId] = useState(initial.threads[0]?.id || '');
  const [models, setModels] = useState([]);
  const [model, setModel] = useState(initial.threads[0]?.model || '');
  const [prompt, setPrompt] = useState('');
  const [busy, setBusy] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [partial, setPartial] = useState('');
  const [status, setStatus] = useState('Ready');
  const [error, setError] = useState(initial.error);
  const [storageError, setStorageError] = useState(initial.error);
  const [historyBlocked, setHistoryBlocked] = useState(Boolean(initial.error));
  const [exporting, setExporting] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const abortRef = useRef(null);
  const tailRef = useRef(null);
  const thread = threads.find((item) => item.id === threadId);
  const messages = thread?.messages || [];

  useEffect(() => {
    // Leave unreadable history intact until the user explicitly clears it.
    if (historyBlocked) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify({ threads, prompts }));
      setStorageError('');
    } catch (failure) {
      setStorageError(`History could not be saved on this device: ${failure.message}. Export important responses before closing.`);
    }
  }, [threads, prompts, historyBlocked]);

  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    if (!busy) return;
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [busy]);
  useEffect(() => { tailRef.current?.scrollIntoView({ block: 'nearest' }); }, [partial, threadId]);

  async function refreshModels() {
    setRefreshing(true);
    setError('');
    try {
      const response = await fetch(`${apiBase}/api/inference/models`);
      const result = await response.json();
      if (!response.ok || !result.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setModels(result.models);
      const smallest = [...result.models].sort((a, b) => (a.size || 0) - (b.size || 0))[0];
      setModel((current) => current || smallest?.name || '');
      if (!result.models.length) setError('No Ollama models are installed. Install/register a chat-capable model, then refresh.');
      else if ((thread?.model || model) && !result.models.some((item) => item.name === (thread?.model || model))) {
        setError('The selected conversation model is not installed. Install/register it, or start a new conversation with another model.');
      }
    } catch (failure) {
      setError(failure.message);
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => { refreshModels(); }, [apiBase]);

  async function send(event) {
    event.preventDefault();
    if (busy || !prompt.trim()) return;
    const submittedPrompt = prompt.trim();
    const history = [...messages, { role: 'user', content: submittedPrompt }];
    const id = threadId || crypto.randomUUID();
    const selectedModel = thread?.model || model;
    setPrompts((old) => [submittedPrompt, ...old.filter((item) => item !== submittedPrompt)].slice(0, 50));
    setError('');
    setPartial('');
    setBusy(true);
    setElapsed(0);
    setStatus('Loading model / waiting for first token');
    const controller = new AbortController();
    abortRef.current = controller;
    let answer = '';
    let complete = false;
    try {
      const response = await fetch(`${apiBase}/api/inference/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: selectedModel, messages: history }),
        signal: controller.signal,
      });
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.error || `HTTP ${response.status}`);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      const consume = (line) => {
        if (!line.trim()) return;
        const result = JSON.parse(line);
        if (result.type === 'error') throw new Error(result.error);
        if (result.type === 'token') {
          answer += result.content;
          setPartial(answer);
          setStatus('Generating response');
        }
        if (result.type === 'complete') complete = true;
      };
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let newline;
        while ((newline = buffer.indexOf('\n')) !== -1) {
          consume(buffer.slice(0, newline));
          buffer = buffer.slice(newline + 1);
        }
      }
      consume(buffer + decoder.decode());
      if (!complete || !answer.trim()) throw new Error('The model did not return a complete text response. Try a chat-capable model.');
      const updated = {
        id, title: thread?.title || submittedPrompt.slice(0, 80), model: selectedModel,
        messages: [...history, { role: 'assistant', content: answer }],
      };
      setThreads((old) => [updated, ...old.filter((item) => item.id !== id)].slice(0, 30));
      setThreadId(id);
      setPrompt('');
      setPartial('');
      setStatus('Response complete - continue this conversation below');
    } catch (failure) {
      controller.abort();
      setStatus('Not completed');
      setError(failure.name === 'AbortError'
        ? 'Inference stopped. Your prompt is preserved; any partial response below was not added to the conversation.'
        : `${failure.message} Your prompt is preserved; partial output is not added to the conversation.`);
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  async function download(format) {
    setExporting(true);
    setError('');
    try {
      const response = await fetch(`${apiBase}/api/inference/export`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format, title: thread.title, model: thread.model, messages }),
      });
      if (!response.ok) {
        const result = await response.json();
        throw new Error(result.error || `HTTP ${response.status}`);
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement('a');
      link.href = url;
      link.download = `conversation.${format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (failure) {
      setError(`Export failed: ${failure.message}`);
    } finally {
      setExporting(false);
    }
  }

  return (
    <section className="rounded-2xl border border-slate-800 bg-slate-900 text-slate-100" aria-label="Model inference">
      <div className="border-b border-slate-800 px-5 py-4">
        <h2 className="text-lg font-semibold">Model inference and conversations</h2>
        <p className="text-sm text-slate-400">Independent of training. Both can run at once if RAM/VRAM permits. Chat uses an installed Ollama model, not live training weights.</p>
      </div>
      <div className="p-5 space-y-3">
        <div className="grid gap-3 md:grid-cols-2">
          <label className="space-y-2">Inference connector
            <select className={controlClass} aria-label="Inference connector" value="ollama" onChange={() => {}}>
              <option value="ollama">Local Ollama (chat / registered GGUF models)</option>
            </select>
          </label>
          <div className="space-y-2">
            <label htmlFor="chat-model">Inference model</label>
            <select id="chat-model" className={controlClass} value={thread?.model || model}
              disabled={busy || Boolean(thread)} onChange={(event) => setModel(event.target.value)}>
              <option value="">Select an installed model</option>
              {(thread?.model || model) && !models.some((item) => item.name === (thread?.model || model)) &&
                <option value={thread?.model || model}>{thread?.model || model} (refresh to verify)</option>}
              {models.map((item) => <option key={item.name} value={item.name}>{item.name}</option>)}
            </select>
            <button type="button" className="action-button" disabled={busy || refreshing} onClick={refreshModels}>
              <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} /> {refreshing ? 'Checking Ollama...' : 'Refresh installed models'}
            </button>
          </div>
          <label className="space-y-2">Saved conversation
            <select className={controlClass} aria-label="Saved conversation" value={threadId} disabled={busy}
              onChange={(event) => {
                const selected = threads.find((item) => item.id === event.target.value);
                setThreadId(event.target.value);
                if (selected) setModel(selected.model);
                setPartial(''); setPrompt(''); setError(''); setStatus('Ready');
              }}>
              <option value="">New conversation</option>
              {threads.map((item) => <option key={item.id} value={item.id}>{item.title} ({item.model})</option>)}
            </select>
          </label>
          <label className="space-y-2">Retrieve a recent prompt
            <select className={controlClass} aria-label="Recent prompts" value="" disabled={busy}
              onChange={(event) => setPrompt(event.target.value)}>
              <option value="">Choose a saved prompt</option>
              {prompts.map((item, index) => <option key={index} value={item}>{item.slice(0, 120)}</option>)}
            </select>
          </label>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="action-button" disabled={busy} onClick={() => {
            setThreadId(''); setPrompt(''); setPartial(''); setError(''); setStatus('Ready');
          }}>New conversation</button>
          <button type="button" className="action-button" disabled={busy || exporting || !messages.length} onClick={() => download('pdf')}>
            <Download className="h-4 w-4" /> Export PDF
          </button>
          <button type="button" className="action-button" disabled={busy || exporting || !messages.length} onClick={() => download('docx')}>
            <Download className="h-4 w-4" /> Export Word (.docx)
          </button>
          <button type="button" className="action-button" disabled={busy} onClick={() => {
            if (!window.confirm('Delete all saved conversations and recent prompts from this device?')) return;
            try {
              window.localStorage.removeItem(STORAGE_KEY);
              setHistoryBlocked(false);
              setThreads([]); setPrompts([]); setThreadId(''); setPartial(''); setPrompt(''); setError('');
            } catch (failure) { setStorageError(`Could not clear local history: ${failure.message}`); }
          }}>Clear local history</button>
        </div>
        <p className="text-xs text-slate-400">Stores the latest 30 conversations and 50 prompts on this device, unencrypted. New conversations select their own model; existing threads keep their original model. Text chat only; video adapters require a vision-capable runtime and are not loaded by this connector.</p>
        {messages.length > 0 && <div className="conversation-history">
          {messages.map((message, index) => <article key={index} className={`conversation-message ${message.role}`}>
            <h3>{message.role === 'user' ? 'You' : 'Model'}</h3>
            <div className="whitespace-pre-wrap">{message.content}</div>
          </article>)}
          <div ref={tailRef} />
        </div>}
        {partial && <article className="conversation-message assistant">
          <h3>{busy ? 'Model (generating)' : 'Partial response (not saved)'}</h3>
          <div className="whitespace-pre-wrap">{partial}</div>
        </article>}
        <form onSubmit={send} className="space-y-3">
          <label htmlFor="chat-prompt">Prompt / continue conversation</label>
          <textarea id="chat-prompt" className={controlClass} rows={4} value={prompt} disabled={busy}
            onChange={(event) => setPrompt(event.target.value)} placeholder="Ask a question or follow up on the previous response..." />
          <div className="flex flex-wrap gap-2">
            <button className="action-button primary" disabled={busy || !prompt.trim() || !models.some((item) => item.name === (thread?.model || model))}>
              {busy && <RefreshCw className="h-4 w-4 animate-spin" />} {busy ? 'Working...' : 'Send prompt'}
            </button>
            <button type="button" className="action-button" disabled={!busy} onClick={() => abortRef.current?.abort()}>
              <Square className="h-4 w-4" /> Stop inference
            </button>
            <span role="status" aria-live="polite">{exporting ? 'Preparing export...' : status}{busy ? ` (${elapsed}s)` : ''}</span>
          </div>
        </form>
        {error && <p role="alert" className="text-red-300">{error}</p>}
        {storageError && <p role="alert" className="text-red-300">{storageError}</p>}
      </div>
    </section>
  );
}
