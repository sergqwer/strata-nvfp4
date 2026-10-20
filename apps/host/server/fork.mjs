/**
 * strata-nvfp4: what this fork changes in the App Host, in one file (index.mjs calls it in a few marked places).
 *
 *   - The host listens on the loopback address only. STRATA_BIND with any other address stops the start unless
 *     STRATA_ALLOW_LAN_BIND=1 (unsupported for now: the host has no login of its own).
 *   - Requests whose Host header is not this host's loopback address are refused (DNS rebinding: a web page whose
 *     name resolves to 127.0.0.1 reaches the host with its own name). No CORS headers: the page is same-origin.
 *   - The host's own OpenAI endpoint (/v1/*) is off (STRATA_HOST_V1=1 turns it on), and the LAN share and the
 *     Cloudflare tunnel cannot be started: the model's own server is the API.
 *   - The update check is this fork's (strata_control/fork_release.py: its GitHub release, its daily cache,
 *     STRATA_UPDATE_CHECK=0), never upstream's release feed.
 */
import { spawn } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const LOOPBACK = new Set(['127.0.0.1', 'localhost', '::1', '[::1]']);

/** The version the page shows is the release's (engine/BUILD.json of this folder), unless a launcher named one. */
function releaseVersion(root) {
  try { const b = JSON.parse(readFileSync(join(root, 'engine', 'BUILD.json'), 'utf8')); return b.fork && typeof b.version === 'string' ? b.version : null; } catch { return null; }
}

/** Throws when the address is not a loopback one and the opt-in is not set. */
export function forkBindCheck(host, env = process.env) {
  if (LOOPBACK.has(String(host).toLowerCase())) return null;
  if (env.STRATA_ALLOW_LAN_BIND === '1') return `strata-nvfp4: listening on ${host} (STRATA_ALLOW_LAN_BIND=1, unsupported: the interface has no login; anyone who reaches this address can use it)`;
  throw new Error(`strata-nvfp4: the interface listens on 127.0.0.1 only; STRATA_BIND=${host} was refused (STRATA_ALLOW_LAN_BIND=1 overrides it, unsupported)`);
}

/** The Host header values this host answers to. */
export function allowedHosts(port, bind = '127.0.0.1', env = process.env) {
  const names = ['127.0.0.1', 'localhost', '[::1]'];
  const b = String(bind).toLowerCase();
  if (env.STRATA_ALLOW_LAN_BIND === '1' && !LOOPBACK.has(b) && b !== '0.0.0.0' && b !== '::') names.push(b.includes(':') && !b.startsWith('[') ? `[${b}]` : b);
  return new Set(names.map(n => `${n}:${port}`));
}

/** Answers the request itself and returns true when the fork refuses it; else false (the host goes on). */
export function forkGuard(req, res, { port, bind = '127.0.0.1', env = process.env } = {}) {
  // The page is served from this origin: no cross-origin reader needs CORS headers, so none are sent.
  const setHeader = res.setHeader.bind(res);
  res.setHeader = (name, value) => (/^access-control-/i.test(String(name)) ? res : setHeader(name, value));
  const refuse = (status, message) => {
    res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
    res.end(JSON.stringify({ error: message }));
    return true;
  };
  const host = String(req.headers.host ?? '').toLowerCase();
  if (!allowedHosts(port, bind, env).has(host)) return refuse(421, `This interface answers at http://127.0.0.1:${port}/ only.`);
  const path = new URL(req.url ?? '/', 'http://localhost').pathname;
  if (path.startsWith('/v1/') && env.STRATA_HOST_V1 !== '1') return refuse(404, "This interface's own /v1 is off in strata-nvfp4: use the model's server (start-server.cmd prints its address).");
  if ((path === '/api/serving/share' || path === '/api/serving/tunnel') && req.method !== 'GET') return refuse(403, 'Sharing on the network and Cloudflare tunnels are not part of the strata-nvfp4 interface.');
  return false;
}

/** The release folder (or clone) this host belongs to: <root>/app/host/server in a release, <root>/apps/host/server in a clone. */
export function forkRoot(env = process.env) {
  return resolve(env.STRATA_FORK_ROOT || join(HERE, '..', '..', '..'));
}

function forkPython(root, env) {
  if (env.STRATA_FORK_PYTHON) return env.STRATA_FORK_PYTHON;
  for (const venv of ['.venv-serve', '.venv']) {
    const p = join(root, venv, process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
    if (existsSync(p)) return p;
  }
  return process.platform === 'win32' ? 'python' : 'python3';
}

// argv: the folder, then the run config (its "update_check": false turns the check off, as in the server) or ""
const CHECK = 'import sys,json\nr=sys.argv[1]\nsys.path.insert(0,r)\nfrom strata_control import fork_release as f\n'
  + 'c=None\ntry:\n c=json.load(open(sys.argv[2],encoding="utf-8-sig")) if sys.argv[2] else None\nexcept (OSError,ValueError):\n pass\n'
  + 'w=f.disabled(c if isinstance(c,dict) else None)\nprint(json.dumps({"disabled":w} if w else {"check":f.check(r)}))';

/** Runs fork_release.check() in the folder's Python (standard library only): {disabled} or {check}. */
export function runForkCheck({ env = process.env, root = forkRoot(env), spawnProcess = spawn, timeoutMs = 20000 } = {}) {
  return new Promise((done, fail) => {
    let out = '', err = '';
    let child;
    try {
      child = spawnProcess(forkPython(root, env), ['-I', '-c', CHECK, root, env.STRATA_ATTACH_CONFIG || ''], { cwd: root, windowsHide: true, stdio: ['ignore', 'pipe', 'pipe'], env });
    } catch (e) { fail(e); return; }
    const alarm = setTimeout(() => { try { child.kill(); } catch { /* gone */ } fail(new Error('The update check did not answer in time.')); }, timeoutMs);
    child.stdout.on('data', c => { out += c; });
    child.stderr.on('data', c => { err += c; });
    child.on('error', e => { clearTimeout(alarm); fail(e); });
    child.on('close', code => {
      clearTimeout(alarm);
      try { done(JSON.parse(out.trim().split(/\r?\n/).pop())); }
      catch { fail(new Error(err.trim().split(/\r?\n/).pop() || `The update check ended with code ${code}.`)); }
    });
  });
}

if (!process.env.STRATA_APP_VERSION) { const v = releaseVersion(forkRoot()); if (v) process.env.STRATA_APP_VERSION = v; }

/** GET /api/updates/check in this fork: the result in the shape the page reads (strata + engine rows), from fork_release. */
export async function forkUpdateCheck({ env = process.env, run = runForkCheck, now = () => new Date() } = {}) {
  const checkedAt = now().toISOString();
  // The engine is part of the release: its row follows the release's.
  const engineRow = (version, status = 'unavailable') => ({ installedVersion: version, status, canInstall: false, message: 'The engine comes with the strata-nvfp4 release and updates with it.' });
  let answer;
  try { answer = await run({ env }); }
  catch (e) {
    return { checkedAt, fork: true, strata: { installedVersion: null, status: 'error', canInstall: false, message: `Could not check for a newer strata-nvfp4 release: ${e.message}` }, engine: engineRow(null) };
  }
  if (answer.disabled) {
    return { checkedAt, fork: true, strata: { installedVersion: null, status: 'unavailable', canInstall: false, message: `The update check is off (${answer.disabled}).` }, engine: engineRow(null) };
  }
  const c = answer.check ?? {};
  const how = c.command ? ` To update: ${c.command}.` : '';
  const strata = c.status === 'available'
    ? { installedVersion: c.installed, status: 'available', canInstall: false, latestVersion: c.latest, url: c.url, message: `strata-nvfp4 ${c.tag} is out (this is ${c.installed}).${how} Notes: ${c.url}` }
    : c.status === 'current'
      ? { installedVersion: c.installed, status: 'current', canInstall: false, message: `strata-nvfp4 ${c.installed} is the newest release.` }
      : { installedVersion: c.installed ?? null, status: c.error ? 'error' : 'unavailable', canInstall: false, message: c.error ? `Could not ask GitHub (${c.error}); asked again in a day.` : 'No version to compare (not a release folder or clone of strata-nvfp4).' };
  return { checkedAt, fork: true, source: c.source ?? null, strata, engine: engineRow(c.installed ?? null, ['available', 'current'].includes(strata.status) ? strata.status : 'unavailable') };
}
