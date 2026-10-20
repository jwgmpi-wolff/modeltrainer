const OLLAMA_BASE = 'http://127.0.0.1:11434';

export function validateConversation(body) {
  if (!body || typeof body.model !== 'string' || !body.model.trim()) {
    throw new Error('Select an installed Ollama model.');
  }
  if (!Array.isArray(body.messages) || !body.messages.length || body.messages.length > 200) {
    throw new Error('A conversation must contain between 1 and 200 messages. Start a new conversation if full.');
  }
  const messages = body.messages.map((message, index) => {
    const expectedRole = index % 2 === 0 ? 'user' : 'assistant';
    if (!message || message.role !== expectedRole || typeof message.content !== 'string' ||
        !message.content.trim() || message.content.length > 100000) {
      throw new Error('Conversation messages must alternate user and assistant, with non-empty text up to 100,000 characters.');
    }
    return { role: message.role, content: message.content };
  });
  if (messages.at(-1).role !== 'user') throw new Error('The last message must be a user prompt.');
  return { model: body.model.trim(), messages };
}

export async function listInferenceModels() {
  const response = await fetch(`${OLLAMA_BASE}/api/tags`, { signal: AbortSignal.timeout(10000) });
  if (!response.ok) throw new Error(`Ollama model listing failed (HTTP ${response.status}).`);
  const result = await response.json();
  if (!Array.isArray(result.models)) throw new Error('Ollama returned an invalid model list.');
  return result.models.map((model) => ({ name: model.name, size: model.size }));
}

export async function streamInference(conversation, signal, onEvent) {
  const response = await fetch(`${OLLAMA_BASE}/api/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...conversation, stream: true }),
    signal,
  });
  if (!response.ok) throw new Error(`Ollama inference failed (HTTP ${response.status}). Check the installed model and available RAM/VRAM.`);
  let buffer = '';
  let done = false;
  const consume = (line) => {
    if (!line.trim()) return;
    const event = JSON.parse(line);
    if (event.error) throw new Error(event.error);
    if (event.message?.content) onEvent({ type: 'token', content: event.message.content });
    if (event.done) {
      done = true;
      onEvent({ type: 'complete' });
    }
  };
  const decoder = new TextDecoder();
  for await (const chunk of response.body) {
    buffer += decoder.decode(chunk, { stream: true });
    let newline;
    while ((newline = buffer.indexOf('\n')) !== -1) {
      consume(buffer.slice(0, newline));
      buffer = buffer.slice(newline + 1);
    }
  }
  buffer += decoder.decode();
  consume(buffer);
  if (!done) throw new Error('Ollama disconnected before completing the response. Retry the prompt.');
}
