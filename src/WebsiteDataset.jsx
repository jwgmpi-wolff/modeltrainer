import React, { useEffect, useRef, useState } from 'react';
import { RefreshCw } from 'lucide-react';

const controlClass = 'w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100';
const JOB_KEY = 'local-model-lab.website-crawl';

export default function WebsiteDataset({ apiBase, onDataset, disabled }) {
  const [url, setUrl] = useState('');
  const [maxPages, setMaxPages] = useState('10');
  const [permission, setPermission] = useState(false);
  const [jobId, setJobId] = useState('');
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const callback = useRef(onDataset);
  callback.current = onDataset;

  useEffect(() => {
    try { setJobId(window.sessionStorage.getItem(JOB_KEY) || ''); }
    catch (failure) { setError(`Could not restore crawl status: ${failure.message}`); }
  }, []);

  useEffect(() => {
    if (!jobId) return;
    let disposed = false;
    const controller = new AbortController();
    const timer = { current: null };
    setBusy(true);
    async function poll() {
      try {
        const response = await fetch(`${apiBase}/api/datasets/crawl/${encodeURIComponent(jobId)}`, { signal: controller.signal });
        const status = await response.json();
        if (!response.ok || !status.ok) throw new Error(status.error || `HTTP ${response.status}`);
        if (disposed) return;
        setProgress(status);
        if (status.status === 'complete') {
          setResult(status.result);
          setBusy(false);
          setStopping(false);
          return;
        }
        if (status.status === 'failed') throw new Error(status.error);
        timer.current = setTimeout(poll, 1000);
      } catch (failure) {
        if (disposed) return;
        setBusy(false);
        setStopping(false);
        setError(`Crawl failed or disconnected: ${failure.message}. Use Reconnect to check a crawl still running on the service.`);
      }
    }
    poll();
    return () => { disposed = true; controller.abort(); clearTimeout(timer.current); };
  }, [apiBase, jobId]);

  async function start() {
    setStarting(true); setError(''); setResult(null); setProgress(null);
    try {
      const response = await fetch(`${apiBase}/api/datasets/crawl`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url, maxPages: Number(maxPages), permission }),
      });
      const status = await response.json();
      if (!response.ok || !status.ok) throw new Error(status.error || `HTTP ${response.status}`);
      setJobId(status.id);
      try { window.sessionStorage.setItem(JOB_KEY, status.id); }
      catch (failure) { setError(`Crawl started, but reconnect information could not be saved: ${failure.message}`); }
    } catch (failure) { setError(failure.message); }
    finally { setStarting(false); }
  }

  async function stop() {
    setStopping(true);
    try {
      const response = await fetch(`${apiBase}/api/datasets/crawl/${encodeURIComponent(jobId)}/stop`, { method: 'POST' });
      const status = await response.json();
      if (!response.ok || !status.ok) throw new Error(status.error || `HTTP ${response.status}`);
    } catch (failure) { setStopping(false); setError(`Could not stop crawl: ${failure.message}`); }
  }

  return (
    <section className="option-frame space-y-2 md:col-span-2" aria-label="Website dataset source">
      <h3 className="font-semibold">Internet-hosted data: crawl a site, pages and documents</h3>
      <p className="text-xs text-slate-400">
        Collect readable HTML page text (screen scrape), PDF, Word (.docx), and plain-text documents from one public HTTPS site.
        Start at a page or document URL; links stay on the same site. No login, JavaScript rendering, screenshot OCR or image-only scan extraction.
      </p>
      <label htmlFor="website-url">Site or document URL</label>
      <input id="website-url" className={controlClass} type="url" value={url} placeholder="https://your-site.example/docs/"
        disabled={busy || starting} onChange={(event) => setUrl(event.target.value)} />
      <label htmlFor="website-limit">Maximum pages/documents to attempt (1-100)</label>
      <input id="website-limit" className={controlClass} type="number" min="1" max="100" value={maxPages}
        disabled={busy || starting} onChange={(event) => setMaxPages(event.target.value)} />
      <label className="flex gap-2 items-start text-sm">
        <input type="checkbox" checked={permission} disabled={busy || starting} onChange={(event) => setPermission(event.target.checked)} />
        I own this content or have permission to collect and use it for model training.
      </label>
      <div className="flex flex-wrap gap-2">
        <button type="button" className="action-button" disabled={disabled || busy || starting || !permission || !url.trim()} onClick={start}>
          {(busy || starting) && <RefreshCw className="h-4 w-4 animate-spin" />} {busy ? 'Crawling...' : 'Crawl site'}
        </button>
        <button type="button" className="action-button" disabled={!jobId || stopping || !busy} onClick={stop}>
          {stopping ? 'Stopping...' : 'Stop crawl'}
        </button>
        {jobId && !busy && <button type="button" className="action-button" onClick={() => {
          const id = jobId; setJobId(''); setTimeout(() => setJobId(id), 0); setError('');
        }}>Reconnect / refresh crawl</button>}
      </div>
      {progress && <p role="status" className="text-sm">
        {progress.status}: {progress.attempted} attempted, {progress.collected} collected, {progress.failed} unavailable/skipped.
        {progress.truncated ? ` ${progress.truncated} source(s) truncated.` : ''}
      </p>}
      {result && <div className="space-y-2">
        <p className="text-sm">Collected {result.sources} source(s), {result.records} training excerpt(s), at {new Date(result.collectedAt).toLocaleString()}.
          {result.limitReached ? ' Crawl limit reached; additional links were not visited.' : ''}
          {result.truncated ? ` ${result.truncated} source(s) exceed the 100,000-character limit and were truncated.` : ''}</p>
        <p className="text-xs text-slate-400">Readable text is split into 1,500-character source-question/answer excerpts with URL, collection date and content hash. Review the JSONL and use a suitable training context length; scraped content is untrusted and is not automatically a high-quality instruction dataset. Downloads are limited to 8 MB per source.</p>
        <button type="button" className="action-button primary" disabled={disabled} onClick={() => callback.current(result)}>
          Add collected website dataset to queue
        </button>
        <p className="text-xs text-slate-400">Dataset to review: {result.path}</p>
        {!!result.issues.length && <details><summary>Unavailable or skipped sources ({result.issues.length})</summary>
          <ul className="text-xs">{result.issues.map((issue, index) => <li key={index}>{issue.url}: {issue.error}</li>)}</ul>
        </details>}
      </div>}
      {error && <p role="alert" className="text-red-300">{error}</p>}
    </section>
  );
}
