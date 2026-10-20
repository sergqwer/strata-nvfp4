// strata-nvfp4: attach mode (server/attach.mjs) and the fork's guards and update check (server/fork.mjs).
import test from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createServer, request } from 'node:http';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { attachSettings, createAttach, samplingSetByPerson, ATTACHED_ID, UP_WINDOW_MS } from '../server/attach.mjs';
import { allowedHosts, forkBindCheck, forkUpdateCheck, runForkCheck } from '../server/fork.mjs';

const root = join(import.meta.dirname, '..');

test('attach settings: URL normalised, key from env, key file, run config or STRATA_API_KEY; none means attach mode off', async () => {
  assert.equal(attachSettings({}), null);
  assert.deepEqual(attachSettings({ STRATA_ATTACH_URL: 'http://127.0.0.1:8080/' }), { base: 'http://127.0.0.1:8080', url: 'http://127.0.0.1:8080/v1', key: '', name: '' });
  assert.equal(attachSettings({ STRATA_ATTACH_URL: 'http://127.0.0.1:8080/v1', STRATA_ATTACH_KEY: ' k1 ' }).key, 'k1');
  const dir = await mkdtemp(join(tmpdir(), 'strata-attach-'));
  try {
    await writeFile(join(dir, 'key'), 'from-file\r\n');
    await writeFile(join(dir, 'cfg.json'), JSON.stringify({ port: 9001, api_key: ['cfg-key', 'other'], model_name: 'qwen3.8-flash-next-gptq-nvfp4' }));
    assert.equal(attachSettings({ STRATA_ATTACH_URL: 'http://h:1', STRATA_ATTACH_KEY_FILE: join(dir, 'key') }).key, 'from-file');
    const fromConfig = attachSettings({ STRATA_ATTACH_CONFIG: join(dir, 'cfg.json') });
    assert.equal(fromConfig.url, 'http://127.0.0.1:9001/v1');
    assert.equal(fromConfig.key, 'cfg-key');
    assert.equal(fromConfig.name, 'qwen3.8-flash-next-gptq-nvfp4');
    assert.equal(attachSettings({ STRATA_ATTACH_URL: 'http://h:1/v1', STRATA_API_KEY: 'a,b' }).key, 'a');
  } finally { await rm(dir, { recursive: true, force: true }); }
  assert.throws(() => attachSettings({ STRATA_ATTACH_URL: 'http://user:pw@h:1' }), /credentials/);
  assert.throws(() => attachSettings({ STRATA_ATTACH_URL: 'ftp://h' }), /plain http/);
});

test('sampling: only what the person set is sent, and never the projection switch', () => {
  assert.deepEqual(samplingSetByPerson({}), {});
  assert.deepEqual(samplingSetByPerson({ 'sample.temp': 0.7, 'sample.topK': 20, 'sample.speedProjection': true, 'sample.seed': -1, 'sample.maxTokens': 0 }), { temperature: 0.7, top_k: 20 });
});

/** A fake clock and a fake model server for createAttach: every request is recorded with whether it carried the key. */
function harness({ up = true, health = { status: 'ok', service: 'strata', model: 'qwen3.8-flash-next-gptq-nvfp4', max_context: 262144, images: true, loaded: true } } = {}) {
  let t = 1_000_000;
  const calls = [];
  const server = { up };
  const fetchImpl = async (url, { headers = {} } = {}) => {
    calls.push({ at: t, path: new URL(url).pathname, keyed: !!headers.authorization });
    if (!server.up) throw new Error('connect ECONNREFUSED');
    const body = new URL(url).pathname === '/health' ? health : { engine: { version: '0.2.0-beta-nvfp4.2', max_context: 262144 }, live: { state: 'idle', slots: [{}] }, requests: [] };
    return { ok: true, status: 200, json: async () => body };
  };
  let requests = [];
  const a = createAttach({ settings: { base: 'http://127.0.0.1:8092', url: 'http://127.0.0.1:8092/v1', key: 'secret-key', name: '' }, fetchImpl, now: () => t,
    setTimer: () => ({ unref() {} }), clearTimer: () => {}, requests: () => requests });
  return { a, calls, server, advance: async ms => { const end = t + ms; while (t < end) { t += 2000; await a.tick(); } }, setRequests: r => { requests = r; }, now: () => t };
}

test('doorbell: a server that is not up never gets a keyed request; probes are keyless and back off', async () => {
  const h = harness({ up: false });
  await h.a.tick();
  await h.advance(30 * 60_000);                       // half an hour with nobody chatting
  assert.equal(h.calls.filter(c => c.keyed).length, 0, 'no keyed request while the server is down');
  assert.ok(h.calls.length <= 15, `keyless probes back off (${h.calls.length} in 30 min)`);
  assert.ok(h.calls.every(c => c.path === '/health'));
  assert.equal(h.a.state().attached.state, 'down');
  assert.equal(h.a.state().instances[0].status, 'ready', 'the chat box stays usable: a chat may wake the model');
});

test('doorbell: once up, one keyed /metrics, then nothing at all while the page sits idle', async () => {
  const h = harness();
  await h.a.tick();
  const first = h.calls.slice();
  assert.deepEqual(first.map(c => [c.path, c.keyed]), [['/health', false], ['/metrics', true]]);
  await h.advance(60 * 60_000);                       // an hour idle: no request, so a tray can unload its model
  assert.equal(h.calls.length, first.length, `no requests while idle (${h.calls.length - first.length} sent)`);
  const st = h.a.state();
  assert.equal(st.instances[0].apiModelId, 'qwen3.8-flash-next-gptq-nvfp4');
  assert.equal(st.instances[0].context, 262144);
  assert.equal(st.attached.state, 'standby');
  assert.ok(!JSON.stringify(st).includes('secret-key'), 'the key never reaches the page');
});

test('doorbell: a chat opens the window - keyless /health while it runs, keyed /metrics until 30 s after it ends, then silence', async () => {
  const h = harness({ up: false });
  await h.a.tick();
  h.calls.length = 0;
  h.server.up = true;
  h.setRequests([{ instanceId: ATTACHED_ID, state: 'generating', at: new Date(h.now()).toISOString(), durationMs: 0 }]);
  await h.advance(10_000);
  assert.ok(h.calls.some(c => c.path === '/health' && !c.keyed), 'keyless /health while the chat runs');
  assert.ok(h.calls.some(c => c.path === '/metrics' && c.keyed), 'the Monitor is fed once the server answered');
  const end = h.now();
  h.setRequests([{ instanceId: ATTACHED_ID, state: 'done', at: new Date(end - 8000).toISOString(), durationMs: 8000 }]);
  await h.advance(UP_WINDOW_MS + 4000);
  const lastKeyed = Math.max(...h.calls.filter(c => c.keyed).map(c => c.at));
  assert.ok(lastKeyed <= end + UP_WINDOW_MS, 'no keyed request after the window');
  const before = h.calls.length;
  await h.advance(20 * 60_000);
  assert.equal(h.calls.length, before, 'silent afterwards');
});

test('doorbell: a model stopped right after a chat (a tray\'s Stop) gets no keyed /metrics - /health goes first each time', async () => {
  const h = harness();
  await h.a.tick();
  h.calls.length = 0;
  const end = h.now();
  h.setRequests([{ instanceId: ATTACHED_ID, state: 'done', at: new Date(end - 5000).toISOString(), durationMs: 5000 }]);
  await h.advance(4000);
  assert.ok(h.calls.some(c => c.path === '/metrics' && c.keyed), 'the Monitor reads /metrics after the chat');
  h.server.up = false;                                // stopped within the 30 s window
  const before = h.calls.filter(c => c.keyed).length;
  await h.advance(UP_WINDOW_MS);
  assert.equal(h.calls.filter(c => c.keyed).length, before, 'no keyed request once /health stopped answering');
  assert.ok(h.calls.slice(-1)[0].path === '/health' && !h.calls.slice(-1)[0].keyed);
});

test('unload detaches without stopping anything; choosing the model again re-attaches', async () => {
  const h = harness();
  await h.a.tick();
  const inst = h.a.resolveInstance(ATTACHED_ID);
  assert.equal(inst.baseUrl, 'http://127.0.0.1:8092/v1');
  assert.equal(inst.key, 'secret-key');
  await h.a.unload(ATTACHED_ID);
  assert.equal(h.a.state().instances.length, 0);
  assert.equal(h.a.hasLive(), false);
  assert.throws(() => h.a.resolveInstance(ATTACHED_ID), /detached/);
  h.a.setDefault(ATTACHED_ID);
  assert.equal(h.a.hasLive(), true);
  assert.throws(() => h.a.operation('load', {}), /start-server\.cmd or the tray/);
});

test('bind: loopback only unless the opt-in; Host allowlist', () => {
  assert.equal(forkBindCheck('127.0.0.1', {}), null);
  assert.throws(() => forkBindCheck('0.0.0.0', {}), /127\.0\.0\.1 only/);
  assert.match(forkBindCheck('0.0.0.0', { STRATA_ALLOW_LAN_BIND: '1' }), /unsupported/);
  assert.deepEqual([...allowedHosts(8791, '127.0.0.1', {})], ['127.0.0.1:8791', 'localhost:8791', '[::1]:8791']);
  assert.ok(allowedHosts(8791, '192.168.1.5', { STRATA_ALLOW_LAN_BIND: '1' }).has('192.168.1.5:8791'));
});

test('update check: this fork\'s release (fork_release.py), never upstream\'s feed', async () => {
  const at = () => new Date('2026-10-20T00:00:00Z');
  const off = await forkUpdateCheck({ run: async () => ({ disabled: 'STRATA_UPDATE_CHECK=0' }), now: at });
  assert.equal(off.strata.status, 'unavailable'); assert.match(off.strata.message, /off \(STRATA_UPDATE_CHECK=0\)/);
  const newer = await forkUpdateCheck({ run: async () => ({ check: { status: 'available', installed: '0.2.0-beta-nvfp4.2', latest: '0.2.0-beta-nvfp4.3', tag: 'v0.2.0-beta-nvfp4.3',
    url: 'https://github.com/sergqwer/strata-nvfp4/releases/tag/v0.2.0-beta-nvfp4.3', command: 'close the server and run update.cmd in C:\\x', source: 'cache' } }), now: at });
  assert.equal(newer.strata.status, 'available'); assert.equal(newer.strata.canInstall, false);
  assert.match(newer.strata.message, /v0\.2\.0-beta-nvfp4\.3 is out .*update\.cmd/);
  assert.equal(newer.engine.status, 'available');
  assert.ok(!JSON.stringify(newer).includes('Niko1221'));
  const failed = await forkUpdateCheck({ run: async () => { throw new Error('boom'); }, now: at });
  assert.equal(failed.strata.status, 'error');
  // the real script in this checkout, with the check switched off: no network, an answer from fork_release itself
  let real = null;
  try { real = await runForkCheck({ env: { ...process.env, STRATA_UPDATE_CHECK: '0' }, root: join(root, '..', '..') }); } catch { /* no Python here */ }
  if (real) assert.deepEqual(real, { disabled: 'STRATA_UPDATE_CHECK=0' });
});

async function freePort() { const s = createServer(); await new Promise(r => s.listen(0, '127.0.0.1', r)); const p = s.address().port; await new Promise(r => s.close(r)); return p; }

/** A model server double: /health keyless (as this fork's serve), everything else needs the key; `down` answers 502 like the tray's proxy without a model. */
async function fakeServe({ key = 'serve-key', down = false } = {}) {
  const seen = [];
  const srv = createServer((req, res) => {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      const keyed = req.headers.authorization === `Bearer ${key}`;
      seen.push({ path: req.url, keyed, body });
      const send = (code, obj, type = 'application/json') => { res.writeHead(code, { 'content-type': type }); res.end(typeof obj === 'string' ? obj : JSON.stringify(obj)); };
      if (down) return send(502, { error: 'upstream: connection refused' });
      if (req.url === '/health') return send(200, { status: 'ok', service: 'strata', model: 'qwen3.8-flash-next-gptq-nvfp4', max_context: 262144, images: true, loaded: true });
      if (!keyed) return send(401, { error: { message: 'missing or wrong API key' } });
      if (req.url.startsWith('/metrics')) return send(200, { engine: { version: '0.2.0-beta-nvfp4.2', max_context: 262144 }, live: { state: 'idle', slots: [{}] }, requests: [] });
      if (req.url === '/v1/chat/completions') {
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: 'Kyiv.' } }] })}\n\n`);
        res.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: {}, finish_reason: 'stop' }], usage: { prompt_tokens: 9, completion_tokens: 2 } })}\n\n`);
        return res.end('data: [DONE]\n\n');
      }
      send(404, { error: 'not found' });
    });
  });
  await new Promise(r => srv.listen(0, '127.0.0.1', r));
  return { srv, seen, port: srv.address().port, close: () => new Promise(r => srv.close(r)) };
}

async function startHost(extraEnv) {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-fork-'));
  const port = await freePort();
  const host = spawn(process.execPath, ['server/index.mjs'], {
    cwd: root, stdio: 'ignore',
    env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_PORT: String(port), STRATA_SHARE_PORT: String(await freePort()), STRATA_TUNNEL_PORT: String(await freePort()),
      STRATA_NO_ACQUIRE: '1', STRATA_NO_BUNDLED_ENGINE: '1', STRATA_REAL_ENGINE: '1', STRATA_UI_MODE: 'browser', STRATA_UPDATE_CHECK: '0', ...extraEnv },
  });
  let up = false;
  for (let i = 0; i < 150 && !up; i++) { try { up = (await fetch(`http://127.0.0.1:${port}/api/health`)).ok; } catch { await delay(100); } }
  assert.ok(up, 'the host answers');
  return { port, host, home, base: `http://127.0.0.1:${port}` };
}

/** A raw request with any Host header (fetch does not let a test set it). */
function raw(port, path, { method = 'GET', headers = {} } = {}) {
  return new Promise((resolve, reject) => {
    const r = request({ host: '127.0.0.1', port, path, method, headers }, res => { let b = ''; res.on('data', c => { b += c; }); res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: b })); });
    r.on('error', reject); r.end();
  });
}

test('host over HTTP, attached: the serve is the loaded model, chat goes there with its key and no sampling of ours; guards refuse rebinding, other sites, /v1 and sharing', async () => {
  const serve = await fakeServe();
  const { port, host, home, base } = await startHost({ STRATA_ATTACH_URL: `http://127.0.0.1:${serve.port}/v1`, STRATA_ATTACH_KEY: 'serve-key' });
  try {
    let native;
    for (let i = 0; i < 50; i++) { native = await (await fetch(`${base}/api/native`)).json(); if (native.attached?.model) break; await delay(100); }
    assert.equal(native.instances[0].id, ATTACHED_ID);
    assert.equal(native.instances[0].label, 'qwen3.8-flash-next-gptq-nvfp4');
    assert.equal(native.instances[0].context, 262144);
    assert.ok(!JSON.stringify(native).includes('serve-key'));
    const mode = await (await fetch(`${base}/api/app/mode`)).json();
    assert.equal(mode.fork.attach.url, `http://127.0.0.1:${serve.port}/v1`);
    assert.equal((await (await fetch(`${base}/api/setup`)).json()).complete, true);
    // idle page: the UI polls /api/native; the serve sees nothing keyed beyond the first /metrics
    for (let i = 0; i < 10; i++) { await fetch(`${base}/api/native`); await delay(100); }
    assert.ok(serve.seen.filter(s => s.keyed).every(s => s.path.startsWith('/metrics')));
    // a chat
    const reply = await fetch(`${base}/api/chat/stream`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ message: 'capital of Ukraine?', history: [], nativeModelId: ATTACHED_ID }) });
    const text = await reply.text();
    assert.match(text, /Kyiv/);
    const chat = serve.seen.find(s => s.path === '/v1/chat/completions');
    assert.ok(chat?.keyed, 'the chat carries the key');
    const sent = JSON.parse(chat.body);
    for (const k of ['temperature', 'top_p', 'top_k', 'repeat_penalty', 'seed', 'experimental_speed_projection']) assert.equal(k in sent, false, `${k} is left to the server`);
    // guards
    assert.equal((await raw(port, '/api/native', { headers: { host: `evil.example:${port}` } })).status, 421);
    assert.equal((await raw(port, '/api/local-models/browse?directory=C%3A%5C', { headers: { host: `evil.example:${port}` } })).status, 421);
    assert.equal((await raw(port, '/api/native', { headers: { host: `127.0.0.1:${port}`, origin: 'https://evil.example' } })).status, 403);
    const v1 = await raw(port, '/v1/models', { headers: { host: `127.0.0.1:${port}`, origin: 'https://evil.example' } });
    assert.equal(v1.status, 404);
    assert.equal(v1.headers['access-control-allow-origin'], undefined);
    assert.equal((await raw(port, '/api/health', { headers: { host: `localhost:${port}` } })).headers['access-control-allow-origin'], undefined);
    assert.equal((await raw(port, '/api/serving/share', { method: 'POST', headers: { host: `127.0.0.1:${port}` } })).status, 403);
    assert.equal((await raw(port, '/api/serving/tunnel', { method: 'POST', headers: { host: `127.0.0.1:${port}` } })).status, 403);
    const upd = await (await fetch(`${base}/api/updates/check`)).json();
    assert.equal(upd.fork, true);
  } finally { host.kill(); await serve.close(); await rm(home, { recursive: true, force: true }); }
});

test('host over HTTP, attached to a server without its model (the tray\'s proxy after an idle unload): only keyless /health, never the key', async () => {
  const proxy = await fakeServe({ down: true });
  const { host, home, base } = await startHost({ STRATA_ATTACH_URL: `http://127.0.0.1:${proxy.port}/v1`, STRATA_ATTACH_KEY: 'serve-key' });
  try {
    for (let i = 0; i < 40; i++) { await fetch(`${base}/api/native`); await fetch(`${base}/api/app/mode`); await delay(100); }
    assert.ok(proxy.seen.length >= 1);
    assert.equal(proxy.seen.filter(s => s.keyed).length, 0, 'no keyed request reaches a server that is not up');
    assert.ok(proxy.seen.every(s => s.path === '/health'));
    const native = await (await fetch(`${base}/api/native`)).json();
    assert.equal(native.attached.state, 'down');
    assert.equal(native.instances[0].status, 'ready');
  } finally { host.kill(); await proxy.close(); await rm(home, { recursive: true, force: true }); }
});

test('host refuses a non-loopback bind unless opted in', async () => {
  const home = await mkdtemp(join(tmpdir(), 'strata-host-bind-'));
  try {
    const child = spawn(process.execPath, ['server/index.mjs'], { cwd: root, stdio: 'ignore', env: { ...process.env, STRATA_HOME: home, STRATA_MODELS_DIR: join(home, 'models'), STRATA_BIND: '0.0.0.0', STRATA_PORT: String(await freePort()) } });
    const code = await Promise.race([new Promise(r => child.on('exit', r)), delay(30000).then(() => 'running')]);
    if (code === 'running') child.kill();
    assert.equal(code, 2);
  } finally { await rm(home, { recursive: true, force: true }); }
});
