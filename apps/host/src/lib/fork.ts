/**
 * strata-nvfp4: what this fork's page hides, from one place (styles/fork.css does the hiding).
 *
 * The host says in /api/app/mode whether it is attached to a model server another process owns (start-server.cmd's,
 * the tray's) and whether the agent component is installed. Attached: no Models, Tuning or Serving (the server is
 * managed where it runs), no update or roll-back buttons (update.cmd updates the release folder). Without the agent:
 * no Subagents, no permission modes, no Tools settings. Always in this fork: no "Speed projection" switch - in this
 * fork's server that request field is the uncensor projection, which this page never offers (and never for NVIDIA's
 * checkpoints, whose licence forbids it).
 */
import { loadAppMode } from './app-mode';

interface ForkMode { attach: { url: string; name: string; state: string } | null; agent: boolean }

const HIDDEN_ROUTES: Record<string, string> = { models: 'monitor', tuning: 'monitor', serving: 'chat', setup: 'chat' };

/** True when the host is attached to a model server (read synchronously by components after the first load). */
export function attachMode(): boolean {
  return typeof document !== 'undefined' && document.documentElement.dataset.strataAttach === '1';
}

function redirect() {
  if (!attachMode()) return;
  const route = window.location.hash.replace(/^#\/?/, '').split(/[/?]/)[0];
  const to = HIDDEN_ROUTES[route];
  if (to) window.location.replace(`#/${to}`);
}

export function installFork() {
  const root = document.documentElement;
  root.dataset.strataFork = '1';
  void loadAppMode().then(mode => {
    const fork = (mode as unknown as { fork?: ForkMode }).fork;
    if (!fork) return;
    if (fork.attach) root.dataset.strataAttach = '1';
    if (!fork.agent) root.dataset.strataNoAgent = '1';
    redirect();
    window.addEventListener('hashchange', redirect);
  });
}
