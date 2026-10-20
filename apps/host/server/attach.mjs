/**
 * strata-nvfp4: attach mode. The host uses a model server that something else started and owns - this fork's
 * start-server.cmd (its serve starts the host, serve/app_ui.py) or the tray - as its loaded model, and starts no
 * engine itself. It stands in for native-control.mjs: the same calls, the same state shape, one instance "attached".
 *
 *   STRATA_ATTACH_URL        the server's API, http://127.0.0.1:8080/v1 (a /v1 is added when missing)
 *   STRATA_ATTACH_KEY        its API key, or STRATA_ATTACH_KEY_FILE (a file holding it), or the run config
 *   STRATA_ATTACH_CONFIG     a run config: its "api_key", and its "port" when no URL is given; else STRATA_API_KEY
 *   STRATA_ATTACH_NAME       the name the page shows (default: the server's own model name)
 *
 * The doorbell rule. Behind the tray's proxy a request that carries the key loads the model, and any request through
 * it counts as use, so it keeps a loaded model from unloading when idle. So this host sends:
 *   - keyless GET /health (this fork's server answers it without the key: model, max_context, images, loaded) to
 *     learn whether the server is up: at start with a growing pause until it first answers, and every 2 s while a
 *     chat of the person's is on its way;
 *   - keyed requests (GET /metrics for the Monitor, GET /v1/models when /health does not name the model) only within
 *     30 s of the server answering /health or of a chat that finished, and the Monitor's /metrics (while a chat runs
 *     and 30 s after it) each time right after /health answered: never while the server is not known to be up;
 *   - nothing at all while the page sits idle. Only the person's chat (the native chat path, index.mjs) wakes a model.
 * Unload detaches (the host stops using the server; choosing the model again re-attaches it); the server keeps running.
 */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

export const ATTACHED_ID = 'attached';
export const UP_WINDOW_MS = 30_000;
const TICK_MS = 2000;
const PROBE_MIN_MS = 2000;
const PROBE_MAX_MS = 300_000;
const NOT_HERE = 'This interface uses the model that start-server.cmd or the tray runs. Models, downloads and tuning are done there (prepare-model.cmd, setup.py), not here.';

function readJson(path) { try { return JSON.parse(readFileSync(path, 'utf8').replace(/^﻿/, '')); } catch { return null; } }

/** The attach settings from the environment, or null when attach mode is off. The key is never logged or returned to the page. */
export function attachSettings(env = process.env, read = readJson) {
  const config = env.STRATA_ATTACH_CONFIG ? read(env.STRATA_ATTACH_CONFIG) : null;
  let url = String(env.STRATA_ATTACH_URL ?? '').trim();
  if (!url && config) url = `http://127.0.0.1:${Number(config.port) || 8080}/v1`;
  if (!url) return null;
  let parsed;
  try { parsed = new URL(url); } catch { throw new Error('STRATA_ATTACH_URL is not a URL.'); }
  if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password || parsed.search || parsed.hash) throw new Error('STRATA_ATTACH_URL must be a plain http(s) address without credentials.');
  const base = parsed.href.replace(/\/+$/, '').replace(/\/v1$/, '');
  let key = String(env.STRATA_ATTACH_KEY ?? '').trim();
  if (!key && env.STRATA_ATTACH_KEY_FILE) { try { key = readFileSync(env.STRATA_ATTACH_KEY_FILE, 'utf8').trim(); } catch { throw new Error('STRATA_ATTACH_KEY_FILE cannot be read.'); } }
  if (!key && config) key = Array.isArray(config.api_key) ? String(config.api_key[0] ?? '').trim() : String(config.api_key ?? '').trim();
  if (!key) key = String(env.STRATA_API_KEY ?? '').split(',')[0].trim();
  const name = String(env.STRATA_ATTACH_NAME ?? '').trim() || (config && typeof config.model_name === 'string' ? config.model_name : '');
  return { base, url: `${base}/v1`, key, name };
}

/** Only the sampling the person set in the page (Expert values they saved); the server's own config applies to the rest. Never the projection switch. */
export function samplingSetByPerson(expert = {}) {
  const out = {};
  const num = (id, key) => { const v = Number(expert[id]); if (expert[id] !== undefined && expert[id] !== null && expert[id] !== '' && Number.isFinite(v)) out[key] = v; };
  num('sample.temp', 'temperature'); num('sample.topP', 'top_p'); num('sample.topK', 'top_k'); num('sample.repeat', 'repeat_penalty');
  if (expert['sample.seed'] !== undefined && Number(expert['sample.seed']) >= 0) out.seed = Number(expert['sample.seed']);
  if (Number(expert['sample.maxTokens']) > 0) out.max_tokens = Number(expert['sample.maxTokens']);
  return out;
}

export function createAttach({ env = process.env, dataDir, requests = () => [], changed = () => {}, appVersion = () => null, engineVersion = () => null,
  fetchImpl = fetch, now = Date.now, setTimer = setInterval, clearTimer = clearInterval, log = () => {}, settings = attachSettings(env) } = {}) {
  if (!settings) return null;
  const cacheFile = dataDir ? join(dataDir, 'attach.json') : null;
  const cached = (cacheFile && readJson(cacheFile)) || {};
  const seen = { model: cached.url === settings.url ? cached.model ?? null : null, context: cached.url === settings.url ? cached.context ?? null : null, images: cached.url === settings.url ? cached.images ?? null : null, loaded: null };
  const sent = { keyless: 0, keyed: 0 };
  const logs = [];
  let chatActive = false, upUntil = 0, everUp = false, lastHealth = 0, nextProbe = 0, probeDelay = PROBE_MIN_MS, lastMetrics = 0, lastChatEnd = 0;
  let metrics = null, detached = false, health = 'unknown', closing = false, error = null;
  const note = line => { logs.push({ at: new Date(now()).toISOString(), instanceId: ATTACHED_ID, line: String(line).slice(0, 500) }); if (logs.length > 200) logs.shift(); log(`[attach] ${line}`); };

  function saveSeen() {
    if (!cacheFile) return;
    try { mkdirSync(dataDir, { recursive: true }); writeFileSync(cacheFile, JSON.stringify({ url: settings.url, model: seen.model, context: seen.context, images: seen.images }, null, 1), { mode: 0o600 }); } catch { /* a cache */ }
  }
  const label = () => settings.name || seen.model || 'Attached model';
  const keyedAllowed = () => now() < upUntil;

  async function get(path, { keyed }) {
    if (keyed && !keyedAllowed()) throw new Error('not known to be up');       // the doorbell rule, enforced at the one place requests are made
    keyed ? sent.keyed++ : sent.keyless++;
    const headers = keyed && settings.key ? { authorization: `Bearer ${settings.key}` } : {};
    const res = await fetchImpl(`${settings.base}${path}`, { headers, redirect: 'error', signal: AbortSignal.timeout(4000) });
    if (!res.ok) { await res.body?.cancel?.(); throw Object.assign(new Error(`HTTP ${res.status}`), { status: res.status }); }
    return res.json();
  }

  async function probe() {
    lastHealth = now();
    try {
      const h = await get('/health', { keyed: false });
      if (h?.status !== 'ok' && h?.status !== 'healthy' && h?.service !== 'strata') throw new Error('not a model server');
      const first = !everUp;
      everUp = true; health = 'up'; error = null; probeDelay = PROBE_MIN_MS;
      upUntil = Math.max(upUntil, now() + UP_WINDOW_MS);
      if (typeof h.model === 'string' && h.model) seen.model = h.model;
      if (Number(h.max_context) > 0) seen.context = Number(h.max_context);
      if (typeof h.images === 'boolean') seen.images = h.images;
      if (typeof h.loaded === 'boolean') seen.loaded = h.loaded;
      if (!seen.model) {
        try {   // not this fork's /health: the model list (keyed - the server just answered, so it is up)
          const m = await get('/v1/models', { keyed: true });
          const row = m?.data?.[0];
          if (row?.id) seen.model = row.id;
          const ctx = Number(row?.meta?.n_ctx ?? row?.context_length ?? row?.max_context_length);
          if (ctx > 0) seen.context = ctx;
        } catch (e) { note(`model list: ${e.message}`); }
      }
      saveSeen();
      if (first) { note(`attached to ${settings.url}: ${seen.model ?? 'model'}${seen.context ? `, context ${seen.context}` : ''}`); await poll(true); }
      emit();
      return true;
    } catch (e) {
      health = everUp ? health : 'down';
      if (!everUp) { error = `The model server at ${settings.url} does not answer yet (${e.status ? `HTTP ${e.status}` : e.message}).`; }
      emit();
      return false;
    }
  }

  async function poll(force = false) {
    // the Monitor's numbers: while a chat runs and for 30 s after it ended (and once when the server first answers)
    if (!force && (now() - lastMetrics < TICK_MS || !(chatActive || now() < lastChatEnd + UP_WINDOW_MS))) return;
    if (!keyedAllowed()) return;
    lastMetrics = now();
    try {
      metrics = await get('/metrics', { keyed: true });
      if (Number(metrics?.engine?.max_context) > 0) seen.context = Number(metrics.engine.max_context);
      emit();
    } catch (e) { if (e.status === 401 || e.status === 403) error = 'The model server refused the API key. Check the key this interface was started with.'; }
  }

  function chatActivity() {
    let active = false, lastOk = 0;
    for (const r of requests() ?? []) {
      if (r.instanceId !== ATTACHED_ID) continue;
      if (['queued', 'generating'].includes(r.state)) active = true;
      else if (r.state === 'done') lastOk = Math.max(lastOk, Date.parse(r.at) + Number(r.durationMs || 0));
    }
    return { active, lastOk };
  }

  let running = null;
  function tick() {   // one at a time; a caller during a run waits for that run
    if (closing || detached) return Promise.resolve();
    running ??= step().finally(() => { running = null; });
    return running;
  }
  async function step() {
    try {
      const { active, lastOk } = chatActivity();
      chatActive = active;
      if (lastOk > lastChatEnd) { lastChatEnd = lastOk; upUntil = Math.max(upUntil, lastOk + UP_WINDOW_MS); }
      if (active || now() < lastChatEnd + UP_WINDOW_MS) {
        // the key only right after the server answered the keyless /health: a model stopped or unloaded since the chat
        // (a tray's Stop, its idle unload) is never woken by the Monitor
        if (now() - lastHealth >= TICK_MS) { if (await probe()) await poll(); else upUntil = 0; }
      } else if (!everUp && now() >= nextProbe) { const ok = await probe(); if (!ok) { nextProbe = now() + probeDelay; probeDelay = Math.min(probeDelay * 2, PROBE_MAX_MS); } }
    } catch (e) { note(e.message); }
  }

  function instance() {
    const live = metrics?.live?.state;
    return { id: ATTACHED_ID, apiModelId: seen.model || 'default', model: seen.model || ATTACHED_ID, label: label(), aliases: [], context: seen.context ?? 0,
      vision: seen.images !== false, resourceProfile: null, gpus: [], status: 'ready', phase: typeof live === 'string' ? live : 'idle', parallel: metrics?.live?.slots?.length || 1,
      mcpServers: {}, loadedAt: null, metrics, error: null, attached: true, url: settings.url };
  }
  function state() {
    const ver = metrics?.engine?.version ?? engineVersion() ?? null;
    return {
      available: true, repo: null, version: appVersion(), summary: detached ? 'Detached from the model server.' : `Using the model server at ${settings.url}.`,
      catalog: { families: [], contexts: [] }, engineInfo: { version: ver, source: 'attach' }, visionReader: true,
      installed: { strata_folder: null, data_folder: null, engine: { version: ver, folder: null }, models: [] }, hardware: null, error: detached ? null : error,
      install: null, external: [], downloadDirectory: null, fields: [], instances: detached ? [] : [instance()], defaultId: detached ? null : ATTACHED_ID, jobs: [],
      capabilities: { responses: true, updates: false, rollback: false, componentUpdates: false, calibration: false },
      attached: { url: settings.url, model: seen.model, name: label(), context: seen.context, images: seen.images, state: detached ? 'detached' : health === 'up' ? (keyedAllowed() ? 'up' : 'standby') : health,
        keyedAllowed: keyedAllowed(), sent: { ...sent }, keySet: !!settings.key },
    };
  }
  let lastSent = '';
  function emit() { const value = state(); const sig = JSON.stringify(value); if (sig !== lastSent) { lastSent = sig; changed(value); } }

  const timer = setTimer(() => { void tick(); }, TICK_MS);
  timer?.unref?.();
  void tick();

  const refuse = async () => { throw new Error(NOT_HERE); };
  return {
    state, refresh: async () => state(), tick, probe,
    resolveInstance(id) {
      if (detached) throw new Error('The model is detached. Choose it again in the chat header to use it.');
      if (id && ![ATTACHED_ID, seen.model].includes(id)) throw new Error('Choose a loaded model ID.');
      return { ...instance(), baseUrl: settings.url, key: settings.key };
    },
    hasLive: () => !detached, readyInstances: () => detached ? [] : [{ ...instance(), baseUrl: settings.url, key: settings.key }],
    async unload() { detached = true; metrics = null; note('detached (the server keeps running)'); emit(); return state(); },
    setDefault() { if (detached) { detached = false; note('attached again'); } emit(); return state(); },
    operation: () => { throw new Error(NOT_HERE); }, cancel: refuse, plan: refuse, configure: refuse, setDownloadDirectory: () => { throw new Error(NOT_HERE); },
    settings: () => { throw new Error(NOT_HERE); }, save: () => { throw new Error(NOT_HERE); }, settingsFile: () => { throw new Error(NOT_HERE); }, openSettingsFile: refuse, localPreview: refuse,
    tuneStatus: refuse, tuneEffective: async () => ({ applied: false, preset: null, changed: {} }), tuneOff: refuse,
    tunePresets: { list: async () => ({ presets: [] }), action: refuse, undoFor: refuse },
    logs: () => logs.slice(), benchmarkReport: () => null, poll: () => {},
    async close() { closing = true; clearTimer(timer); },
    summary: () => ({ url: settings.url, name: label(), state: state().attached.state }),
    counters: () => ({ ...sent }),
  };
}

/** The engine version of the release folder the host belongs to (engine/BUILD.json), read from disk: no request. */
export function localEngineVersion(root) {
  const b = readJson(join(root, 'engine', 'BUILD.json'));
  return b && typeof b.version === 'string' ? b.version : null;
}
