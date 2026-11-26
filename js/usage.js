// Usage ids and events for the GPU server (rules and formats are in usage-lib.js). On only while the server's
// GET /status has `usage: true`. Nothing here changes what the page shows, and every failure is silent.
import { usageOn, visitorId, newId, withIds, makeThrottle, loadEvent, uiEvent } from './usage-lib.js';

const st = { serverUrl: null, on: false, visitor: null, session: null, loaded: false };
const allow = makeThrottle();

/** Called when the GPU server's status has been read (body null when unreachable). Sends the load event once. */
export function usageStatus(B, body) {
  st.serverUrl = B.url;
  st.on = usageOn(body);
  if (st.on && !st.visitor) {
    let storage = null;
    try { storage = localStorage; } catch { /* blocked */ }
    st.visitor = storage ? visitorId(storage) : newId();
    st.session = newId();
  }
  if (st.on && !st.loaded) {
    st.loaded = true;
    send(loadEvent(st, { referrer: document.referrer }));
  }
}

/** The body for a request to `url` at `path`, with the ids when it goes to the GPU server and usage is on. */
export function usageBody(url, path, body) {
  return withIds(body, { url, path, serverUrl: st.serverUrl, on: st.on, visitor: st.visitor, session: st.session });
}

/** Report a UI action (names in usage-lib.js). At most one per name per 10 s. */
export function usageUi(name, value) {
  if (!st.on) return;
  const e = uiEvent(st, name, value);
  if (e && allow(name, Date.now())) send(e);
}

function send(body) {
  try {
    fetch(`${st.serverUrl}/event`, { method: 'POST', keepalive: true, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }).catch(() => {});
  } catch { /* never block the page */ }
}

/** Listen for the reported actions at the document level, so the controls themselves need no hooks. Call once. */
export function initUsage() {
  document.addEventListener('click', (e) => {
    const b = e.target?.closest?.('button');
    if (b?.classList.contains('pick-btn') && b.dataset.mode && b.closest('.step-2')) usageUi('mode', b.dataset.mode);
  }, true);
  document.addEventListener('toggle', (e) => { if (e.target?.id === 'specmenu' && e.target.open) usageUi('spectrum_open'); }, true);
}
