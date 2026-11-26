import * as d3 from 'd3';
import { eigSym, kdpp, greedyMAPLog, quality, diversity, mulberry32 } from './dpp.js';
import { Plot3D } from './plot3d.js';
import { initExplainers } from './explain.js';
import { authEnabled, getAuth, signIn, signOut, finishSignIn } from './hfauth.js';
import { usageStatus, usageBody, initUsage } from './usage.js';

const $ = (s) => document.querySelector(s);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const pct = (x) => (!Number.isFinite(x) ? '—' : x >= 0.995 ? '100%' : x < 0.005 ? '<1%' : `${Math.round(100 * x)}%`);
const clampInt = (v, lo, hi) => Math.min(hi, Math.max(lo, Math.round(+v || lo)));
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

// ---------------------------------------------------------------- limits (every backend enforces them too)
const LIMITS = { maxN: 200, maxTokens: 256, bufferMax: 1000, timeoutMs: 5 * 60 * 1000 };

// ---------------------------------------------------------------- state
const S = {
  pool: null,        // { prompt, samples, ... } of the slice on screen
  G: null,           // its N×N similarity matrix (unit embeddings), Float64Array
  view: null,        // { buf, k, N, embedder }: which slice of which buffer is on screen
  eig: null,         // eigendecomposition of G (for the spectrum panel and PCA axes)
  logp: null, logq: null,
  proj: null,        // { coords: [N][3], frac: number[] }
  dims: 3,
  mode: 'map',
  T: NaN,            // one temperature for both modes; its starting value is the T slider's, in the HTML
  target: 'global',  // target quality: 'global' κ = p^{1/T}; 'gl' κ = p^{1/T}·p°_{T′}
  K: 3,
  pinned: -1,
  rng: mulberry32(12345),
  sel: [], pi: null, logP: NaN, fit: NaN, c: null,
  hover: -1,
  poolNote: '',
};

// T slider: log scale over [0.1, 1000], and the right end is T = ∞ (only diversity counts).
const T_MIN = -1, T_MAX = 3;
const sliderToT = (v) => (v >= 1000 ? Infinity : 10 ** (T_MIN + ((T_MAX - T_MIN) * v) / 1000));
const tToSlider = (T) => (T === Infinity ? 1000 : Math.round(((Math.log10(T) - T_MIN) / (T_MAX - T_MIN)) * 1000));
const fmtT = (T) => (T === Infinity ? '∞' : T < 10 ? T.toFixed(2) : T < 100 ? T.toFixed(1) : T.toFixed(0));

// ---------------------------------------------------------------- binary helpers
function b64ToF32(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(bytes.buffer);
}
// Similarity matrix from unit-normalised vectors (servers that still send vectors), in float64.
function gramFromVectors(f, N, D) {
  const V = Array.from({ length: N }, (_, i) => {
    const v = Float64Array.from(f.subarray(i * D, (i + 1) * D));
    const r = Math.hypot(...v) || 1;
    return v.map((x) => x / r);
  });
  const G = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
    let s = 0; const a = V[i], b = V[j];
    for (let d = 0; d < D; d++) s += a[d] * b[d];
    G[i * N + j] = G[j * N + i] = i === j ? 1 : s;
  }
  return G;
}
// The G a backend returned: an N×N float32 matrix ("G"), or vectors ("emb", "D").
function gramOf(resp, N) {
  if (resp.G) return Float64Array.from(b64ToF32(resp.G).subarray(0, N * N));
  return gramFromVectors(b64ToF32(resp.emb), N, resp.D);
}
const sub = (G, n, idx) => {
  const m = idx.length, out = new Float64Array(m * m);
  for (let a = 0; a < m; a++) for (let b = 0; b < m; b++) out[a * m + b] = G[idx[a] * n + idx[b]];
  return out;
};

// ---------------------------------------------------------------- buffers of draws
// A buffer holds i.i.d. draws from p°_{T′} for one (prompt, weights, T′, max tokens), in the order they
// were drawn, from one backend. A pool of N is a slice of it: draw k = completions (k−1)N+1 … kN.
// Similarity matrices are kept per embedder: for precomputed buffers one matrix over all draws; for
// your runs one per batch (draws sampled together), so a slice that spans batches is re-embedded.
let BUFFERS = [];
const bufFor = (cfg) => BUFFERS.filter((b) => b.prompt === cfg.prompt && b.model === cfg.model &&
  Math.abs(b.Tprime - cfg.Tprime) < 1e-9 && b.maxNewTokens === cfg.maxNewTokens);
const bufSize = (b) => (b.samples ? b.samples.length : b.size);

async function loadData() {
  const [presets, cache] = await Promise.all([fetch('data/presets.json').then((r) => r.json()), fetch('data/cache/index.json').then((r) => r.json())]);
  BUFFERS = cache.buffers.map((b) => ({ ...b, from: 'precomputed', samples: null, batches: null }));
  const el = $('#examples');
  el.innerHTML = '<div class="menu-head">Example prompts</div>' + presets.map((g, i) => `<button type="button" role="menuitem" data-i="${i}"><b>${escapeHtml(g.label)}</b><span>${escapeHtml(g.prompt)}</span></button>`).join('');
  // A preset only fills in the prompt; nothing else in the configuration changes.
  el.querySelectorAll('button').forEach((b) => (b.onclick = () => {
    $('#exmenu').open = false;
    $('#prompt').value = presets[+b.dataset.i].prompt;
    $('#prompt').dispatchEvent(new Event('input'));
  }));
  if (!$('#prompt').value.trim()) { $('#prompt').value = presets[0].prompt; $('#prompt').dispatchEvent(new Event('input')); }
}

// Every buffer is a list of batches (draws sampled together, or completions embedded together), each
// with its similarity matrix per embedder. A precomputed buffer is one batch whose matrices are files.
async function ensureLoaded(b) {
  if (b.samples) return;
  b.samples = (await (await fetch(`data/cache/${b.pool}`)).json()).samples;
  b.batches = [{ start: 0, n: b.size, G: {}, files: b.G }];
}
async function batchG(batch, embedder) {
  if (!batch.G[embedder] && batch.files?.[embedder]) {
    batch.G[embedder] = Float64Array.from(new Float32Array(await (await fetch(`data/cache/${batch.files[embedder]}`)).arrayBuffer()));
  }
  return batch.G[embedder] ?? null;
}
// The slice [start, end) of buffer b with its similarity matrix under `embedder` (a repo id), or null if
// the buffer has fewer draws, or no single batch with that embedder covers the slice.
async function sliceOf(b, start, end, embedder) {
  if (end > bufSize(b)) return null;
  await ensureLoaded(b);
  for (const batch of b.batches) {
    if (batch.start > start || end > batch.start + batch.n) continue;
    const G = await batchG(batch, embedder);
    if (!G) continue;
    const idx = d3.range(start - batch.start, end - batch.start);
    return { samples: b.samples.slice(start, end), G: sub(G, batch.n, idx) };
  }
  return null;
}

// ---------------------------------------------------------------- configuration
// What determines the pool on screen: prompt, model weights, embedder, N, T′, max tokens, and which
// draw. The plot always shows a pool made with exactly these, or keeps the previous one with a note.
// ?server=local: reach the GPU server through the local mirror (scripts/dev-server.py), for testing
// from localhost, which the GPU server's CORS rule does not allow.
const DEMO_SERVER = new URLSearchParams(location.search).get('server') === 'local'
  ? `${location.origin}/dpp_api` : 'https://zueri-monstro.rufous-hamlet.ts.net/dpp_api';
const MODAL_URL = 'https://timvieira--dpp-lm-sampler-web.modal.run';
const SPACE_URL = 'https://xtimv-dpp-lm-sampler.hf.space';
const QEMB4 = { id: 'qwen3-emb-4b', repo: 'Qwen/Qwen3-Embedding-4B', label: 'Qwen3-Embedding-4B' };
const QEMB = { id: 'qwen3-emb', repo: 'Qwen/Qwen3-Embedding-0.6B', label: 'Qwen3-Embedding-0.6B' };
const OWN = { id: 'own', repo: 'own:Qwen/Qwen3-14B-AWQ', label: 'Qwen3 14B’s own states' };
const BACKENDS = [
  { id: 'server', where: 'GPU server', the: 'the GPU server', label: 'Qwen3 14B, 4-bit (AWQ) · GPU server', short: 'Qwen3 14B 4-bit', repo: 'Qwen/Qwen3-14B-AWQ',
    url: DEMO_SERVER, auth: false, embedders: [QEMB4, QEMB, OWN], up: null },
  { id: 'modal', where: 'Modal', the: 'Modal', label: 'Qwen3 14B, 4-bit (AWQ) · Modal', short: 'Qwen3 14B 4-bit', repo: 'Qwen/Qwen3-14B-AWQ',
    url: MODAL_URL, auth: true, embedders: [QEMB4, QEMB], up: null,
    // Listed so the settings look the same everywhere, but not runnable yet (TODO: Modal and the 4B embedder).
    unavailable: { 'qwen3-emb-4b': 'Modal’s GPU (48 GB) cannot hold Qwen3 14B and Qwen3-Embedding-4B together yet.' } },
  { id: 'space', where: 'Hugging Face Space', the: 'the Hugging Face Space', label: 'Qwen3 14B, 4-bit (AWQ) · HF Space', short: 'Qwen3 14B 4-bit', repo: 'Qwen/Qwen3-14B-AWQ',
    url: SPACE_URL, auth: true, embedders: [QEMB4], up: null },
];
const cap = (t) => t[0].toUpperCase() + t.slice(1);
const backend = () => BACKENDS.find((b) => b.id === $('#model').value) ?? BACKENDS[0];
const embedderOf = (B) => B.embedders.find((e) => e.id === $('#embedder').value) ?? B.embedders[0];
const TARGETS = { global: { label: 'Global (paper)', short: 'global' }, gl: { label: 'Global and local', short: 'global × local' } };

function readConfig() {
  const B = backend();
  return {
    prompt: $('#prompt').value.trim(), model: B.repo, backend: B.id, embedder: embedderOf(B).repo, embedderId: embedderOf(B).id,
    N: clampInt($('#live-N').value, 2, LIMITS.maxN), Tprime: Math.round(10 * Math.min(2, Math.max(0.2, +$('#live-Tp').value || 1))) / 10,
    maxNewTokens: clampInt($('#live-max').value, 8, LIMITS.maxTokens),
  };
}
const viewKey = (cfg, k) => JSON.stringify([cfg.prompt, cfg.model, cfg.embedder, cfg.N, cfg.Tprime, cfg.maxNewTokens, k]);
let drawK = 1;   // which draw the visitor asked for (1 = the first N completions)
let pendingK = null;   // a draw to show once restored settings settle (restoreConfig)
// A panel (failure, sign-in) stays until the settings it is about change; '*' keeps it until dismissed.
let stickyFor = null;
const cfgKey = (cfg) => viewKey(cfg, 0);

// Called whenever the pool-determining settings change. Shows a stored slice that matches exactly,
// otherwise keeps the current plot and says it is from earlier settings.
async function configChanged() {
  const cfg = readConfig();
  syncFieldNotes();
  const keep = !$('#stage-msg').hidden && stickyFor != null && (stickyFor === '*' || stickyFor === cfgKey(cfg));
  if (pendingK == null && S.view && S.view.key === viewKey(cfg, drawK)) { if (!keep) setStatus(S.poolNote, { drawNext: true }); return; }
  const want = pendingK ?? 1;
  pendingK = null;
  drawK = 1;
  const key = cfgKey(cfg);
  for (const k of want > 1 ? [want, 1] : [1]) {
    for (const b of bufFor(cfg)) {
      const sl = await sliceOf(b, (k - 1) * cfg.N, k * cfg.N, cfg.embedder).catch(() => null);
      if (cfgKey(readConfig()) !== key) return;   // settings changed while loading
      if (sl) { if (!keep) stageMsg(null); drawK = k; showSlice(b, sl, cfg, k); return; }
    }
  }
  if (keep) return;
  if (S.pool) { setStatus('The plot is from earlier settings. Press ↑.'); return; }
  emptyMsg(cfg);
  setStatus('');
}

function showSlice(b, sl, cfg, k) {
  const N = cfg.N, a = (k - 1) * N + 1, z = k * N;
  const bt = b.batches?.find((x) => x.start <= a - 1 && z <= x.start + x.n && x.files);
  const B = BACKENDS.find((x) => x.id === b.backend) ?? BACKENDS[0];
  const made = bt ? `Precomputed on ${B.the}` : `Sampled on ${B.the}`;
  S.poolNote = `${made} · draw ${k} (completions ${a}–${z}).`;
  setPool({ prompt: b.prompt, samples: sl.samples }, sl.G, { buf: b, k, N, embedder: cfg.embedder, key: viewKey(cfg, k) });
  setStatus(S.poolNote, { drawNext: true });
}

// "Draw new ones": the next N draws of the buffer; samples only what is missing.
async function drawNext() {
  const cfg = readConfig();
  const k = (S.view?.key === viewKey(cfg, drawK) ? drawK : 1) + 1;
  for (const b of bufFor(cfg)) {
    const sl = await sliceOf(b, (k - 1) * cfg.N, k * cfg.N, cfg.embedder).catch(() => null);
    if (sl) { drawK = k; showSlice(b, sl, cfg, k); return; }
  }
  runLive({ k });
}

// ---------------------------------------------------------------- pool on screen
// φ is a function of the string, so identical completions must have identical rows of G. Batched
// embedding on a GPU can make copies differ by ~1e-5 (padding changes the arithmetic), which would let
// a set of two copies of one string have a tiny positive volume. Use the first copy's row for every copy.
function sameStringsSameRows(samples, G) {
  const N = samples.length, first = new Map(), c = samples.map((s, i) => (first.has(s.text) ? first.get(s.text) : (first.set(s.text, i), i)));
  if (c.every((x, i) => x === i)) return G;
  const H = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) H[i * N + j] = G[c[i] * N + c[j]];
  return H;
}
function setPool(pool, G, view) {
  G = sameStringsSameRows(pool.samples, G);
  $('#labels').innerHTML = '';
  S.pinned = -1;
  S.pool = pool;
  S.G = G;
  S.view = view;
  S.logp = pool.samples.map((s) => s.logp);
  S.logq = pool.samples.map((s) => s.logq);
  const N = S.logp.length;
  S.eig = eigSym(G, N);
  S.proj = null;
  S.K = Math.min(S.K, N - 1);
  $('#K').max = String(Math.min(20, N - 1));
  $('#K').value = S.K;
  S.rng = mulberry32(12345);
  plot.reset();
  recompute({ newDraw: true });
}

// ---------------------------------------------------------------- inference
// log κ for the chosen target, and the importance weights w = κ / p°_{T′} (the pool's proposal).
function logKappa() {
  const T = S.T, gl = S.target === 'gl';
  return S.logp.map((lp, i) => (T === Infinity ? 0 : lp / T) + (gl ? S.logq[i] : 0));
}
function recompute({ newDraw = true } = {}) {
  if (!S.pool) return;
  const N = S.logp.length, K = S.K, G = S.G;
  const lk = logKappa();
  const lw = lk.map((x, i) => x - S.logq[i]);
  // Most likely scores sets by Π κ · det G_S; Sample draws from the K-DPP with the importance weights.
  const scale = S.mode === 'map' ? lk : lw;
  const dist = kdpp(G, N, scale, K);
  // Greedy stops at the rank of G, so it also tells whether any K-subset has positive volume.
  const greedy = greedyMAPLog(G, N, lk, K).set;
  S.hasK = greedy.length === K;
  if (S.mode === 'map') S.sel = greedy;
  else if (newDraw || !S.sel.length) S.sel = dist.sample(S.rng) ?? [];
  S.pi = S.hasK ? dist.pi : null;   // no K-DPP exists without a K-subset of positive volume
  S.logP = S.sel.length === K ? dist.logProb(S.sel) : NaN;
  // Pool fit to the target (paper, Thm 5.2): θ̂ = K²ρ̂/(N−K+1), ρ̂ = N Σ (π_i/K)² with π the inclusion
  // probabilities of the importance-weighted K-DPP, i.e. the pool's estimate of the target.
  const piW = S.mode === 'sample' ? dist.pi : kdpp(G, N, lw, K).pi;
  let s2 = 0; for (const p of piW) s2 += (p / K) ** 2;
  S.fit = (K * K * N * s2) / (N - K + 1);
  // Drawn length of each point: √κ (its quality under the target) relative to the largest, in both
  // modes. The importance weights Sample uses only correct for how the pool was drawn; they are not a
  // property of the completion, so the points do not move when the mode changes.
  const m = Math.max(...lk);
  S.c = lk.map((s) => Math.exp(0.5 * (s - m)));
  // The axes are set once per pool (and per 1-D/2-D/3-D choice), when the points are first drawn, and
  // then left alone: the selection, the mode, T and K never turn them. Only the visitor does, with
  // "Fit axes to selection" (fitAxes).
  if (!S.proj || S.proj.pool !== S.pool) S.proj = project();
  render();
}

// Axes. For K ≤ 3 (and K ≤ the drawn dimensions) the first K axes span the selected completions, so
// the drawn parallelepiped is exactly the volume being scored; the remaining axes are the main
// directions of what is left. Otherwise plain uncentred PCA.
// Computed from G alone.
function project() {
  const N = S.logp.length, G = S.G, sel = S.sel, k = sel.length;
  const vals = S.eig.values, vecs = S.eig.vectors;
  const tot = vals.reduce((a, b) => a + Math.max(0, b), 0);
  const frac = vals.slice(0, 8).map((v) => Math.max(0, v) / tot);
  let coords, fitted = false;
  if (k >= 1 && k <= 3 && k <= S.dims && k === S.K) {
    // Cholesky of G_SS, then a_i = R⁻¹ G_{S,i}.
    const R = new Float64Array(k * k);
    let ok = true;
    for (let i = 0; i < k; i++) for (let j = 0; j <= i; j++) {
      let s = G[sel[i] * N + sel[j]];
      for (let m = 0; m < j; m++) s -= R[i * k + m] * R[j * k + m];
      if (i === j) { if (!(s > 1e-12)) ok = false; R[i * k + i] = Math.sqrt(Math.max(s, 1e-300)); } else R[i * k + j] = s / R[j * k + j];
    }
    if (ok) {
      const A = Array.from({ length: N }, (_, n) => {
        const a = new Float64Array(k);
        for (let i = 0; i < k; i++) { let s = G[sel[i] * N + n]; for (let m = 0; m < i; m++) s -= R[i * k + m] * a[m]; a[i] = s / R[i * k + i]; }
        return a;
      });
      const rest = 3 - k;
      let resid = null;
      if (rest > 0) {
        const Gr = new Float64Array(N * N);
        for (let i = 0; i < N; i++) for (let j = i; j < N; j++) {
          let s = G[i * N + j]; for (let m = 0; m < k; m++) s -= A[i][m] * A[j][m];
          Gr[i * N + j] = Gr[j * N + i] = s;
        }
        resid = eigSym(Gr, N);
      }
      coords = A.map((a, n) => [0, 1, 2].map((d) => (d < k ? a[d] : Math.sqrt(Math.max(0, resid.values[d - k])) * resid.vectors[d - k][n])));
      fitted = true;
    }
  }
  if (!coords) coords = Array.from({ length: N }, (_, n) => [0, 1, 2].map((d) => Math.sqrt(Math.max(0, vals[d])) * vecs[d][n]));
  // Align to the previous view of the same pool with a proper rotation of the drawn dimensions.
  const prev = S.proj?.pool === S.pool ? S.proj.coords : null;
  if (prev) coords = kabsch(coords, prev, S.dims);
  return { coords, frac, fitted, k, sel: [...sel], pool: S.pool };
}
function kabsch(P, Q, d) {
  // Rotation R (d×d, det +1) minimising Σ‖R p_i − q_i‖² over the first d coordinates.
  if (d === 1) { let s = 0; P.forEach((p, i) => (s += p[0] * Q[i][0])); return s < 0 ? P.map((p) => [-p[0], p[1], p[2]]) : P; }
  const H = Array.from({ length: d }, () => new Float64Array(d));
  P.forEach((p, i) => { for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) H[a][b] += p[a] * Q[i][b]; });
  // SVD of H via the eigendecomposition of HᵀH: H = U Σ Vᵀ, R = V Uᵀ.
  const HtH = new Float64Array(d * d);
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) { let s = 0; for (let c = 0; c < d; c++) s += H[c][a] * H[c][b]; HtH[a * d + b] = s; }
  const { values, vectors } = eigSym(HtH, d);
  const V = vectors.map((v) => Array.from(v));          // columns of V = eigenvectors
  if (values.some((x) => !(x > 1e-12))) return P;
  const U = V.map((v, j) => { const u = new Array(d).fill(0); for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) u[a] += H[a][b] * v[b]; const r = Math.sqrt(values[j]); return u.map((x) => x / r); });
  const R = Array.from({ length: d }, (_, a) => Array.from({ length: d }, (_, b) => V.reduce((s, v, j) => s + v[a] * U[j][b], 0)));
  const det = d === 2 ? R[0][0] * R[1][1] - R[0][1] * R[1][0]
    : R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  if (det < 0) { // flip the axis with the smallest singular value
    const j = d - 1;
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) R[a][b] -= 2 * V[j][a] * U[j][b];
  }
  return P.map((p) => { const out = p.slice(); for (let a = 0; a < d; a++) { let s = 0; for (let b = 0; b < d; b++) s += R[a][b] * p[b]; out[a] = s; } return out; });
}

// ---------------------------------------------------------------- rendering
function render() {
  const distinct = new Set(S.pool.samples.map((x) => x.text)).size;
  const N = S.pool.samples.length;
  $('#stage-note').textContent = `${N} completions${distinct < N ? ` (${distinct} distinct)` : ''} · ${S.dims === 3 ? 'drag to rotate · ' : ''}${matchMedia('(pointer: coarse)').matches ? 'two fingers to move' : 'Shift-drag to move'} · tap a point to read it`;
  $('#subset-meta').textContent = `K = ${S.sel.length} · ${S.mode === 'map' ? 'most likely (greedy)' : 'sampled'} at T = ${fmtT(S.T)} · target ${TARGETS[S.target].short}`;
  $('#T-out').textContent = fmtT(S.T);
  plot.update();
  renderSpectrum();
  renderReadout();
  renderCards();
  renderVolHint();
  renderInspect();
  renderLabels();
}

// The spectrum panel (opened from the strip under the plot): the share of the pool's variation along
// each of the first eight principal axes, and the 1-D/2-D/3-D switch.
function renderSpectrum() {
  const el = $('#spectrum');
  const f = S.proj.frac, max = f[0] || 1;
  el.innerHTML = `<p class="sp-head"><span class="fn">Spectrum<button type="button" class="xbtn" data-x="pca" aria-label="About the spectrum">?</button></span></p>` +
    f.map((v, k) => `<div class="bar"><span>${k + 1}</span><span class="fill" style="width:${(100 * v) / max}%"></span><span class="pv">${(100 * v).toFixed(1)}%</span></div>`).join('') +
    `<p class="sp-dims"><span class="lab">Drawn in</span><span class="pick" role="group" aria-label="Dimensions">${[1, 2, 3].map((d) => `<button type="button" class="pick-btn" data-dims="${d}" aria-pressed="${S.dims === d}">${d}-D</button>`).join('')}</span></p>` +
    `<p class="sp-note">${S.proj.fitted ? 'Axes 1–' + S.proj.k + ' of the plot span the completions selected when the axes were fitted; the rest show the main remaining directions.' : 'The plot shows the first principal axes.'}</p>`;
  el.querySelectorAll('[data-dims]').forEach((b) => (b.onclick = () => setDims(+b.dataset.dims)));
}
// Whether the axes were fitted to exactly the current selection, so the drawn box is the scored volume.
function axesFitSel() {
  const p = S.proj;
  return !!p?.fitted && p.sel.length === S.sel.length && S.sel.every((i) => p.sel.includes(i));
}
// Turn the axes to contain the current selection (aligned to the current view, so points move little).
function fitAxes() { S.proj = project(); render(); }
function setDims(k) { S.dims = k; S.proj = null; plot.reset(); S.proj = project(); render(); }

function renderReadout() {
  const N = S.logp.length, K = S.K;
  const ok = S.sel.length === K;
  const stat = (k, v, x, cls = '') => `<span class="stat ${cls}"><span class="fn">${k}<button type="button" class="xbtn" data-x="${x}" aria-label="What is ${k}?">?</button></span> <span class="v">${v}</span></span>`;
  $('#readout').innerHTML =
    stat('log P(S)', ok ? fmt(S.logP) : '—', 'logp') +
    stat('quality', ok ? fmt(quality(S.logp, S.sel)) : '—', 'quality') +
    stat('diversity', ok ? fmt(diversity(S.G, N, S.sel)) : '—', 'diversity') +
    stat('pool fit', ok ? `${fmt(S.fit)} nats` : '— why?', 'fit');   // a plain number: no cut-off for 'poor'
  $('#readout').title = ok ? '' : `No subset of size ${K}: the pool spans fewer than ${K} independent directions.`;
  const note = $('#fit-note');
  if (!ok) {
    note.hidden = false;
    note.textContent = `No subset of ${K} completions has positive volume in this pool: lower K, or sample more varied completions (raise T′).`;
  } else note.hidden = true;
  // The Details button carries a dot when the note inside has something to say.
  for (const b of [$('#details-btn'), $('#details-toggle')]) b.classList.toggle('warnflag', !note.hidden);
  // The note (only when no K-subset exists, an exact condition) opens from the pool-fit "— why?".
  const fitStat = $('#readout').querySelector('.stat:last-child');
  if (!note.hidden) { fitStat.classList.add('warn', 'can-open'); fitStat.querySelector('.v').id = 'fit-why'; fitStat.querySelector('.v').title = 'Why?'; }
  else note.classList.remove('show');
}

function renderVolHint() {
  const el = $('#vol-hint');
  // The box is exact only when the plot's axes contain every selected vector: K ≤ 2 in 2-D, K ≤ 3 in 3-D.
  if (S.sel.length < 2) { el.innerHTML = ''; return; }
  if (S.sel.length > S.dims) { el.innerHTML = `<span class="muted">${S.dims === 1 ? 'volume drawn in 2-D and 3-D' : `volume drawn for K ≤ ${S.dims}`}</span>`; return; }
  const tried = S.proj.sel.length === S.sel.length && S.sel.every((i) => S.proj.sel.includes(i));
  if (!axesFitSel()) {
    // Fitting fails only when the selected vectors are linearly dependent: the volume is then zero.
    el.innerHTML = tried ? '<span class="muted">zero volume</span>'
      : '<button type="button" class="link vol-fit" title="Turn the axes to contain the selected completions, so the box drawn is the volume being scored">Fit axes to selection</button>';
    el.querySelector('.vol-fit')?.addEventListener('click', fitAxes);
    return;
  }
  el.innerHTML = `<svg class="sw-vol" width="18" height="12" viewBox="0 0 18 12" aria-hidden="true"><path d="M1 11L5 1H17L13 11Z"/></svg><span class="fn">spanned volume<button type="button" class="xbtn" data-x="volume" aria-label="What is the spanned volume?">?</button></span>`;
}

// A completion that hit the max-token limit, rather than ending with the end-of-text token: said in words.
const cutLabel = (s) => (s.ended ? '' : `<span class="cutoff" title="Stopped at the max-token limit (${s.nTokens} tokens), before the model ended it">cut off at ${s.nTokens} tokens</span>`);
function renderCards() {
  // Most likely to be included first, like All completions; log p breaks ties (and orders them when
  // inclusion is undefined because no K-subset exists).
  const incl = inclusionOfText(), by = (i) => (Number.isFinite(incl(i)) ? incl(i) : 0);
  const order = S.sel.slice().sort((a, b) => (by(b) - by(a)) || (S.logp[b] - S.logp[a]));
  $('#selected').innerHTML = order.map((i) => {
    const s = S.pool.samples[i];
    return `<li class="card${s.ended ? '' : ' cut'}" data-i="${i}">
      <div class="txt">${escapeHtml(s.text)}${s.ended ? '' : ' …'}</div>
      <div class="meta"><span class="dot" style="background:${css('--accent')}"></span>log p ${fmt(s.logp, 1)}<span>·</span><span class="fn">inclusion ${pct(incl(i))}<button type="button" class="xbtn" data-x="inclusion" aria-label="What is the inclusion probability?">?</button></span>${cutLabel(s)}</div></li>`;
  }).join('');
  renderAllCards();
  $('#selected').querySelectorAll('.card').forEach((el) => {
    el.onmouseenter = () => plot.highlight(+el.dataset.i);
    el.onmouseleave = () => plot.highlight(-1);
    el.onclick = (e) => { if (!e.target.closest('.xbtn')) inspect(+el.dataset.i); };
  });
}
// All completions, in the rail under the selected ones and in the same card style: one card per distinct
// string (copies counted), most likely to be included first. Re-rendered at most every 150 ms, so
// dragging T stays smooth.
let allTimer = 0;
function renderAllCards() {
  clearTimeout(allTimer);
  allTimer = setTimeout(() => {
    if (!S.pool) return;
    const distinct = new Set(S.pool.samples.map((x) => x.text)).size;
    if (!$('#all-list').open) { $('#all-meta').textContent = `${S.pool.samples.length} completions, ${distinct} distinct`; $('#all-cards').innerHTML = ''; return; }   // built only when opened
    const incl = inclusionOfText(), sel = new Set(S.sel.map((i) => S.pool.samples[i].text));
    const groups = new Map();
    S.pool.samples.forEach((s, i) => { const g = groups.get(s.text); if (g) g.n++; else groups.set(s.text, { i, n: 1 }); });
    const rows = [...groups.values()].sort((a, b) => (incl(b.i) - incl(a.i)) || (S.logp[b.i] - S.logp[a.i]));
    const cut = S.pool.samples.filter((s) => !s.ended).length;
    $('#all-meta').textContent = `${S.pool.samples.length} completions, ${rows.length} distinct${cut ? `, ${cut} cut off at the token limit` : ''} · ${S.pi ? 'by inclusion' : 'by log p'}`;
    $('#all-cards').innerHTML = rows.map(({ i, n }) => {
      const s = S.pool.samples[i], on = sel.has(s.text);
      return `<li class="card${s.ended ? '' : ' cut'}" data-i="${i}">
      <div class="txt">${escapeHtml(s.text)}${s.ended ? '' : ' …'}</div>
      <div class="meta"><span class="dot" style="background:${on ? css('--accent') : css('--ramp-lo')}"></span>${n > 1 ? `×${n}<span>·</span>` : ''}log p ${fmt(s.logp, 1)}<span>·</span>inclusion ${pct(incl(i))}${cutLabel(s)}</div></li>`;
    }).join('');
    $('#all-cards').querySelectorAll('.card').forEach((el) => {
      el.onmouseenter = () => plot.highlight(+el.dataset.i);
      el.onmouseleave = () => plot.highlight(-1);
      el.onclick = () => inspect(+el.dataset.i);
    });
  }, 150);
}
// Inclusion probability of a completion's text: copies of the same string can never be selected
// together, so the probability that the string appears is the sum over its copies.
function inclusionOfText() {
  const byText = new Map();
  S.pool.samples.forEach((s, i) => byText.set(s.text, (byText.get(s.text) ?? 0) + (S.pi ? S.pi[i] : NaN)));
  return (i) => byText.get(S.pool.samples[i].text);
}

// ---------------------------------------------------------------- tooltip
const tip = $('#tooltip');
function showTip(i, x, y) {
  if (i < 0) { tip.hidden = true; return; }
  const s = S.pool.samples[i];
  tip.innerHTML = `${escapeHtml(s.text)}${s.ended ? '' : '…'}<div class="meta">log p ${fmt(s.logp, 2)} · inclusion ${pct(inclusionOfText()(i))}${S.sel.includes(i) ? ' · <b>selected</b>' : ''}${s.ended ? '' : ` · ${cutLabel(s)}`}</div>`;
  tip.hidden = false;
  const r = tip.getBoundingClientRect();
  tip.style.left = Math.min(window.innerWidth - r.width - 8, x + 14) + 'px';
  tip.style.top = Math.min(window.innerHeight - r.height - 8, y + 14) + 'px';
}

// ---------------------------------------------------------------- 1-D / 2-D plot
function parallelotope(O, V) {
  const add = (a, b) => a.map((x, d) => x + b[d] - O[d]);
  const corner = (mask) => V.reduce((acc, v, k) => (mask & (1 << k) ? add(acc, v) : acc), O.slice());
  const faces = [];
  for (let a = 0; a < V.length; a++) for (let b = a + 1; b < V.length; b++) {
    const others = [...Array(V.length).keys()].filter((k) => k !== a && k !== b);
    const offs = others.length ? [0, 1 << others[0]] : [0];
    for (const o of offs) faces.push([o, o | (1 << a), o | (1 << a) | (1 << b), o | (1 << b)].map(corner));
  }
  const full = (1 << V.length) - 1, corners = [];
  for (let mask = 1; mask <= full; mask++) if (mask & (mask - 1)) corners.push({ p: corner(mask) });
  return { faces, corners };
}

class Plot2D {
  constructor(el) {
    this.el = el; this.svg = null;
    // When the plot's box really changes size, resize in place and keep the visitor's pan and zoom: what
    // was drawn stays where it was on screen (anchored at the top left). On phones the box is sized in vh,
    // which does not change when the address bar shows or hides while scrolling, so scrolling never
    // reaches here.
    new ResizeObserver(() => this.resize()).observe(el);
  }
  resize() {
    if (!this.svg) return;
    const W = this.el.clientWidth, H = this.el.clientHeight;
    if (!W || !H || (W === this.W && H === this.H)) return;
    this.W = W; this.H = H;
    this.svg.attr('viewBox', `0 0 ${W} ${H}`);
    this.svg.select('#grid-mask rect').attr('width', W).attr('height', H);
    if (this.bx) this.draw(false);
  }
  reset() { this.el.innerHTML = ''; this.svg = null; this.zt = d3.zoomIdentity; this.bx = null; }
  build() {
    const el = this.el;
    this.W = el.clientWidth || 600;
    this.H = el.clientHeight || 450;
    const svg = d3.select(el).append('svg').attr('viewBox', `0 0 ${this.W} ${this.H}`);
    // The background grid (2-D): the 3-D view's floor seen flat. Cells of 1/8, fading out with distance
    // from the origin (gone by 6 units) through a radial mask.
    const defs = svg.append('defs');
    this.fade = defs.append('radialGradient').attr('id', 'grid-fade').attr('gradientUnits', 'userSpaceOnUse');
    this.fade.append('stop').attr('offset', 0).attr('stop-color', '#fff').attr('stop-opacity', 1);
    this.fade.append('stop').attr('offset', 1).attr('stop-color', '#fff').attr('stop-opacity', 0);
    defs.append('mask').attr('id', 'grid-mask').append('rect').attr('width', this.W).attr('height', this.H).attr('fill', 'url(#grid-fade)');
    this.gGrid = svg.append('g').attr('mask', 'url(#grid-mask)');
    this.gOrigin = svg.append('g');
    this.gOrigin.append('path').attr('stroke', css('--muted')).attr('stroke-width', 1);
    this.gPara = svg.append('g');
    // Arrowhead for the selected vectors, drawn at the tip in the line's own colour.
    defs.append('marker').attr('id', 'arrowhead').attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
      .attr('markerWidth', 4).attr('markerHeight', 4).attr('orient', 'auto-start-reverse')
      .append('path').attr('d', 'M0,0L10,5L0,10Z').attr('fill', css('--accent'));
    this.gArrow = svg.append('g');
    this.gDot = svg.append('g');
    this.svg = svg;
    this.jit = S.logp.map((_, i) => ((i * 0.6180339887) % 1) - 0.5);
    // Pan and zoom, with the same gestures as the 3-D view: the wheel or a pinch zooms; Shift-drag,
    // right-drag or two fingers move; double-click resets. (A plain drag rotates in 3-D; here it does nothing.)
    this.zt = d3.zoomIdentity;
    this.zoom = d3.zoom().scaleExtent([0.2, 50])
      .filter((e) => e.type === 'wheel' || (e.type === 'mousedown' && (e.shiftKey || e.button === 2)) || (e.type === 'touchstart' && e.touches.length >= 2))
      .on('zoom', (e) => { this.zt = e.transform; if (this.bx) this.draw(false); });
    svg.call(this.zoom).on('dblclick.zoom', null)
      .on('dblclick', () => svg.call(this.zoom.transform, d3.zoomIdentity))
      .on('contextmenu', (e) => e.preventDefault());
    const nearest = (ev, r) => {
      const [px, py] = d3.pointer(ev);
      let best = -1, bd = r * r;
      this.pos.forEach(([a, b], i) => { const d = (a - px) ** 2 + (b - py) ** 2; if (d < bd) { bd = d; best = i; } });
      return best;
    };
    svg.on('pointermove', (ev) => {
      if (ev.pointerType !== 'mouse') return;
      const best = nearest(ev, 14);
      this.highlight(best);
      showTip(best, ev.clientX, ev.clientY);
    }).on('pointerleave', () => { this.highlight(-1); showTip(-1); })
      .on('click', (ev) => inspect(nearest(ev, ev.pointerType === 'mouse' ? 14 : 28)));
  }
  dataXY(i, scale) {
    const c = S.proj.coords[i];
    return S.dims === 1 ? [c[0] * scale, this.jit[i] * 1.6] : [c[0] * scale, c[1] * scale];
  }
  fit(points) {
    // The drawing goes in the open area between the floating panels; the plot itself fills the window.
    const dims = S.dims, r = openArea(), w = r.R - r.L, h = r.B - r.T;
    const pad = (a) => { const lo = d3.min(a), hi = d3.max(a), p = 0.06 * (hi - lo || 1); return [lo - p, hi + p]; };
    let [x0, x1] = pad(points.map((q) => q[0]).concat([0]));
    let [y0, y1] = dims === 1 ? [-1, 1] : pad(points.map((q) => q[1]).concat([0]));
    if (dims === 2) {
      const s = Math.min(w / (x1 - x0), h / (y1 - y0));
      const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      x0 = cx - w / s / 2; x1 = cx + w / s / 2;
      y0 = cy - h / s / 2; y1 = cy + h / s / 2;
    }
    this.bx = d3.scaleLinear().domain([x0, x1]).range([r.L, r.R]);
    this.by = d3.scaleLinear().domain([y0, y1]).range([r.B, r.T]);
  }
  update() {
    if (!this.svg) this.build();
    const N = S.logp.length, dims = S.dims, H = this.H;
    const sel = new Set(S.sel);
    const idx = d3.range(N);
    const posD = idx.map((i) => (dims === 1 ? [this.dataXY(i, S.c[i])[0], this.dataXY(i, 1)[1]] : this.dataXY(i, S.c[i])));
    const vol = dims === 2 && S.sel.length === 2 && axesFitSel();
    const volD = vol ? parallelotope([0, 0], S.sel.map((i) => posD[i])) : { faces: [], corners: [] };
    if (!this.bx) this.fit([...posD, ...volD.corners.map((c) => c.p)]);   // framed once, when first drawn
    Object.assign(this, { posD, volD, sel });
    this.draw(true);
  }
  // Draw with the current pan and zoom: animated after a data change, at once while panning or zooming.
  draw(animate) {
    const { posD, volD, sel } = this, dims = S.dims, H = this.H, N = posD.length, idx = d3.range(N);
    this.x = this.zt.rescaleX(this.bx); this.y = this.zt.rescaleY(this.by);
    const px = (q) => [this.x(q[0]), this.y(q[1])];
    this.pos = posD.map(px);
    const ox = this.x(0), oy = this.y(0);
    this.drawGrid(dims === 2, ox, oy);
    const t = animate ? this.svg.transition().duration(450).ease(d3.easeCubicOut) : null;
    const go = (selection) => (t ? selection.transition(t) : selection);
    go(this.gOrigin.select('path').attr('stroke-dasharray', dims === 1 ? '3 3' : null))
      .attr('d', dims === 1 ? `M${ox},16V${H - 16}` : `M${ox - 6},${oy}H${ox + 6}M${ox},${oy - 6}V${oy + 6}`);
    const y0 = (i) => (dims === 1 ? this.pos[i][1] : oy);
    // Every selected completion is an arrow from the origin, whatever K (only the volume box needs K ≤ 2).
    this.gArrow.selectAll('line').data(S.sel, (i) => i).join(
      (enter) => enter.append('line').attr('x1', ox).attr('y1', y0).attr('x2', ox).attr('y2', y0),
    ).attr('stroke', css('--accent')).attr('stroke-width', 2).attr('marker-end', 'url(#arrowhead)');
    go(this.gArrow.selectAll('line')).attr('x1', ox).attr('y1', y0)
      .attr('x2', (i) => this.pos[i][0]).attr('y2', (i) => this.pos[i][1]);
    const face = (f) => 'M' + f.map((q) => px(q).join(',')).join('L') + 'Z';
    go(this.gPara.selectAll('path').data(volD.faces).join('path')
      .attr('fill', css('--text')).attr('fill-opacity', 0.04)
      .attr('stroke', css('--text-2')).attr('stroke-opacity', 0.7).attr('stroke-width', 1).attr('stroke-linejoin', 'round'))
      .attr('d', face);
    go(this.gDot.selectAll('circle').data(idx).join('circle')
      .attr('fill', (i) => (sel.has(i) ? css('--accent') : css('--ink')))
      .attr('stroke', css('--surface')).attr('stroke-width', 0.75)
      .attr('r', 1.6)
      .sort((a, b) => sel.has(a) - sel.has(b)))
      .attr('cx', (i) => this.pos[i][0]).attr('cy', (i) => this.pos[i][1]);
    if (!animate) renderLabels(false);   // at once while panning or zooming, never trailing
  }
  drawGrid(on, ox, oy) {
    if (!on) { this.gGrid.selectAll('line').remove(); return; }
    const R = 6, unit = Math.abs(this.x(1) - this.x(0));   // pixels per unit length
    this.fade.attr('cx', ox).attr('cy', oy).attr('r', R * unit);
    let step = 1 / 8;
    while (step * unit < 6) step *= 2;   // never denser than one line per 6 px
    const lines = [], xa = Math.max(-R, this.x.invert(0)), xb = Math.min(R, this.x.invert(this.W));
    const ya = Math.max(-R, this.y.invert(this.H)), yb = Math.min(R, this.y.invert(0));
    for (let v = Math.ceil(xa / step) * step; v <= xb; v += step) lines.push([this.x(v), this.y(-R), this.x(v), this.y(R)]);
    for (let v = Math.ceil(ya / step) * step; v <= yb; v += step) lines.push([this.x(-R), this.y(v), this.x(R), this.y(v)]);
    this.gGrid.selectAll('line').data(lines).join('line')
      .attr('x1', (d) => d[0]).attr('y1', (d) => d[1]).attr('x2', (d) => d[2]).attr('y2', (d) => d[3])
      .attr('stroke', css('--text-2')).attr('stroke-opacity', 0.13).attr('stroke-width', 1);   // as soft as the 3-D floor
  }
  highlight(i) {
    if (!this.svg) return;
    if (i < 0) i = S.pinned;
    S.hover = i;
    this.gDot.selectAll('circle').attr('stroke', (j) => (j === i ? css('--text') : css('--surface'))).attr('stroke-width', (j) => (j === i ? 1.5 : 0.75));
    document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i));
  }
}

const plot = {
  p2: new Plot2D($('#plot')),
  p3: null,
  reset() { this.p2.reset(); this.p3?.dispose(); this.p3 = null; },
  update() {
    if (S.dims === 3) {
      if (!this.p3) {
        $('#plot').innerHTML = '';
        this.p3 = new Plot3D($('#plot'), {
          onHover: (i, x, y) => { showTip(i, x, y); S.hover = i < 0 ? S.pinned : i; this.highlightCards(S.hover); },
          onTap: (i) => inspect(i),
          safeRect: openArea,
          // Labels follow the points in the same frame the scene is drawn, so they never trail it.
          onFrame: () => { if ($('#labels').childElementCount) placeLabels(); },
        });
      }
      this.p3.update({
        coords: S.proj.coords, c: S.c, sel: S.sel, fitted: S.proj.fitted, exact: axesFitSel(),
        color: () => css('--ink'), selColor: () => css('--accent'), accent: css('--accent'), text: css('--text-2'), muted: css('--muted'), bg: css('--surface'),
      });
    } else this.p2.update();
  },
  highlight(i) {
    if (i < 0) i = S.pinned;
    S.hover = i;
    if (S.dims === 3) { this.p3?.highlight(i); this.highlightCards(i); } else this.p2.highlight(i);
  },
  highlightCards(i) { document.querySelectorAll('.card').forEach((c) => c.classList.toggle('hl', +c.dataset.i === i)); },
};

// ---------------------------------------------------------------- labels
function labelText(s) { const w = s.text.replace(/\s+/g, ' ').trim().split(' '); return w.length > 5 ? w.slice(0, 5).join(' ') + '…' : w.join(' '); }
function renderLabels(animate = true) {
  const layer = $('#labels');
  const show = $('#label-toggle').checked && S.sel.length;   // every selected point, whatever K; the checkbox turns them off
  if (!show) { layer.innerHTML = ''; return; }
  const want = new Set(S.sel.map(String));
  layer.querySelectorAll('.lbl').forEach((el) => { if (!want.has(el.dataset.i)) el.remove(); });
  for (const i of S.sel) {
    if (layer.querySelector(`.lbl[data-i="${i}"]`)) continue;
    const el = document.createElement('div');
    el.className = 'lbl';
    el.dataset.i = i;
    el.appendChild(document.createElement('span')).textContent = labelText(S.pool.samples[i]);
    el.onclick = () => inspect(i);
    el.onmouseenter = () => plot.highlight(i);
    el.onmouseleave = () => plot.highlight(-1);
    layer.appendChild(el);
  }
  layer.classList.toggle('anim', animate && S.dims !== 3);
  placeLabels();
}
function placeLabels() {
  const layer = $('#labels');
  const els = [...layer.querySelectorAll('.lbl')];
  if (!els.length) return;
  const W = layer.clientWidth, H = layer.clientHeight;
  const items = els.map((el) => {
    const i = +el.dataset.i;
    const p = S.dims === 3 ? plot.p3?.screenXY(i) : plot.p2.pos?.[i];
    return p && { el, x: p[0], y: p[1], w: el.offsetWidth, h: el.offsetHeight };
  }).filter(Boolean).sort((a, b) => a.y - b.y);
  const placed = [];
  for (const it of items) {
    let left = it.x + 8 + it.w > W - 4 ? it.x - 8 - it.w : it.x + 8;
    left = Math.max(4, left);
    let top = Math.min(Math.max(4, it.y - it.h / 2), H - it.h - 4);
    for (const q of placed) {
      const overlapX = left < q.left + q.w && q.left < left + it.w;
      if (overlapX && top < q.top + q.h + 3 && q.top < top + it.h) top = q.top + q.h + 3;
    }
    placed.push({ left, top, w: it.w, h: it.h });
    it.el.style.transform = `translate(${Math.round(left)}px, ${Math.round(top)}px)`;
  }
}
const wide = matchMedia('(min-width: 901px)');
// The part of the plot not covered by the floating panels (wide screens), in the plot's own pixels.
function openArea() {
  const p = $('#plot').getBoundingClientRect();
  if (!wide.matches) return { L: 8, T: 8, R: p.width - 8, B: p.height - 8 };
  const left = Math.max($('.dock').getBoundingClientRect().right, $('.top').getBoundingClientRect().right);
  return { L: left + 20 - p.left, T: 20 - p.top, R: $('.rail').getBoundingClientRect().left - 20 - p.left, B: $('#details').getBoundingClientRect().top - 20 - p.top };
}

// ---------------------------------------------------------------- inspector
function inspect(i) {
  S.pinned = i;
  showTip(-1);
  renderInspect();
  plot.highlight(i);
}
function renderInspect() {
  const el = $('#inspect');
  const i = S.pinned;
  if (i < 0 || !S.pool || i >= S.pool.samples.length) { el.hidden = true; return; }
  const s = S.pool.samples[i];
  const inSel = S.sel.includes(i);
  el.hidden = false;
  el.innerHTML = `<div class="ins-txt">${escapeHtml(s.text)}${s.ended ? '' : ' …'}</div>
    <div class="ins-meta">${inSel ? '<span class="pill">selected</span>' : ''}
      <span>log p ${fmt(s.logp, 2)}</span><span>log p°<sub>T′</sub> ${fmt(s.logq, 2)}</span><span>inclusion ${pct(inclusionOfText()(i))}</span>${cutLabel(s)}</div>
    <button class="ins-close" aria-label="Close">×</button>`;
  el.querySelector('.ins-close').onclick = () => inspect(-1);
}

// ---------------------------------------------------------------- messages
function setStatus(text, { drawNext: showDraw = false } = {}) {
  const el = $('#live-status');
  el.textContent = text;
  if (showDraw && S.pool) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'link'; b.textContent = 'Draw new ones';
    b.onclick = drawNext;
    el.append(' ', b);
  }
}
function stageMsg(kind, { title = '', detail = '', actions = '', sticky = null } = {}) {
  const el = $('#stage-msg');
  stickyFor = kind ? sticky : null;
  if (!kind) { el.hidden = true; el.innerHTML = ''; return; }
  el.hidden = false;
  el.className = `stage-msg ${kind}${S.pool ? ' over' : ''}`;
  el.innerHTML = `${kind === 'running' ? '<span class="spinner" aria-hidden="true"></span>' : ''}<p class="t">${title}</p>` +
    (detail ? `<p class="d">${detail}</p>` : '') + (actions ? `<p class="a">${actions}</p>` : '');
}
function emptyMsg(cfg) {
  stageMsg('empty', cfg.prompt
    ? { title: 'No completions for these settings yet.', detail: `Press ↑ to sample ${cfg.N} completions.` }
    : { title: 'Type a prompt, or pick one with +.' });
}
const useServerBtn = () => (BACKENDS[0].up === false ? '' : ' <button type="button" class="btn" data-act="use-server">Use the GPU server</button>');
const otherModelBtn = () => '<button type="button" class="btn" data-act="settings">Use another model</button>';

// ---------------------------------------------------------------- running
let cloudAbort = null;
// A failure the page can explain: `code` follows docs/BACKEND.md.
class RunError extends Error { constructor(code, message) { super(message); this.code = code; } }

async function callBackend(B, path, body, { signal, auth }) {
  const headers = { 'Content-Type': 'application/json', ...(auth ? { Authorization: `Bearer ${auth.token}` } : {}) };
  let r;
  try { r = await fetch(`${B.url}${path}`, { method: 'POST', signal, headers, body: JSON.stringify(usageBody(B.url, path, body)) }); }   // ids only for the GPU server
  catch (e) { if (e.name === 'AbortError') throw e; throw new RunError('unreachable', ''); }
  const out = await r.json().catch(() => null);
  if (r.ok && out) return out;
  if (r.ok) throw new RunError('bad_answer', 'The server sent an answer the page could not read.');
  const msg = out?.detail || out?.message || '';
  const code = out?.code || ({ 400: 'bad_request', 401: 'auth', 403: 'auth', 429: 'busy', 503: 'starting' }[r.status] ?? 'server');
  throw new RunError(code, msg);
}
// The Space speaks Gradio's queue protocol: join the queue, then read the event stream. (Gradio's
// simpler /call endpoint drops error messages, and its JS client sends credentials, which the Space's
// CORS answer does not allow for other sites.)
async function callSpace(B, api, data, { signal, auth }) {
  const hdr = auth ? { Authorization: `Bearer ${auth.token}` } : {};
  const net = async (url, init) => {
    try { return await fetch(url, { ...init, signal }); }
    catch (e) { if (e.name === 'AbortError') throw e; throw new RunError('unreachable', ''); }
  };
  let r = await net(`${B.url}/config`, { headers: hdr });
  if (!r.ok) throw new RunError(r.status === 401 || r.status === 403 ? 'auth' : 'starting', '');
  const cfg = await r.json().catch(() => null);
  const fn = cfg?.dependencies?.findIndex((d) => d.api_name === api) ?? -1;
  if (fn < 0) throw new RunError('bad_answer', `The Space has no "${api}" function.`);
  const session = Math.random().toString(36).slice(2);
  r = await net(`${B.url}/gradio_api/queue/join`, {
    method: 'POST', headers: { 'Content-Type': 'application/json', ...hdr },
    body: JSON.stringify({ data, fn_index: fn, trigger_id: cfg.dependencies[fn].id ?? null, session_hash: session, event_data: null }),
  });
  if (!r.ok) throw new RunError(r.status === 401 || r.status === 403 ? 'auth' : r.status === 429 ? 'busy' : 'server', '');
  r = await net(`${B.url}/gradio_api/queue/data?session_hash=${session}`, { headers: hdr });
  if (!r.ok || !r.body) throw new RunError('server', '');
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += value;
    let k;
    while ((k = buf.indexOf('\n\n')) >= 0) {
      const line = buf.slice(0, k).split('\n').find((l) => l.startsWith('data:'));
      buf = buf.slice(k + 2);
      if (!line) continue;
      const m = JSON.parse(line.slice(5));
      if (m.msg !== 'process_completed') continue;
      if (m.success && m.output?.data) return m.output.data[0];
      const msg = String(m.output?.error ?? m.output?.title ?? '');
      // The Space prefixes its own errors with a code; ZeroGPU's quota errors have none.
      const c = /^(bad_request|server|busy|limit): (.*)$/s.exec(msg);
      if (c) throw new RunError(c[1], c[2]);
      throw new RunError(/quota|runs limit/i.test(msg) ? 'quota' : 'server', msg);
    }
  }
  throw new RunError('bad_answer', 'The Space ended without an answer.');
}

// Sample `n` new draws for cfg on backend B; returns { samples, G (n×n), timing }.
async function sampleDraws(B, cfg, n, opts) {
  const seed = Math.floor(Math.random() * 2 ** 31);
  let out;
  if (B.id === 'space') out = await callSpace(B, 'sample', [cfg.prompt, 'qwen3-14b', n, cfg.Tprime, cfg.maxNewTokens, seed], opts);
  else {
    const body = { prompt: cfg.prompt, n, t_prime: cfg.Tprime, max_new_tokens: cfg.maxNewTokens, seed };
    if (B.id === 'server') Object.assign(body, { model: 'qwen3-14b', embedder: cfg.embedderId });
    out = await callBackend(B, '/sample', body, opts);
  }
  if (out.N !== n || out.maxNewTokens !== cfg.maxNewTokens || (out.Tprime != null && Math.abs(out.Tprime - cfg.Tprime) > 1e-9) || !Array.isArray(out.samples) || out.samples.length !== n) throw new RunError('mismatch', 'The server did not run the requested settings.');
  return { samples: out.samples, G: gramOf(out, n), timing: out.timing };
}
// Similarity matrix of given completions under cfg's embedder, without sampling (for slices that span batches).
async function embedTexts(B, cfg, samples, opts) {
  const texts = samples.map((s) => s.text);
  let out;
  if (B.id === 'space') out = await callSpace(B, 'embed', [texts, cfg.embedderId], opts);
  else out = await callBackend(B, '/embed', { texts, embedder: cfg.embedderId, model: 'qwen3-14b', prompt: cfg.prompt, ended: samples.map((s) => s.ended) }, opts);
  return gramOf(out, texts.length);
}

async function runLive({ k = drawK } = {}) {
  const cfg = readConfig(), B = backend();
  if (!cfg.prompt) return;
  const auth = B.auth ? getAuth() : null;
  if (B.auth && !auth) {
    // Never leave the page without asking: explain and offer the two ways forward.
    stageMsg('error', {
      title: `Running on ${B.the} needs a free Hugging Face account.`,
      detail: 'You will be sent to Hugging Face to sign in, then back here, where the run starts.',
      actions: '<button type="button" class="btn" data-act="signin">Sign in with Hugging Face</button>' + useServerBtn() + (S.pool ? ' <button type="button" class="btn" data-act="dismiss">Cancel</button>' : ''),
      sticky: cfgKey(cfg),
    });
    pendingSignIn = { ...cfg, k };
    return;
  }
  if (B.unavailable?.[cfg.embedderId]) {
    // An embedder this backend lists but cannot run yet: say why, and offer the two ways forward.
    stageMsg('error', {
      title: `${embedderOf(B).label} isn’t available on ${B.the} yet.`,
      detail: escapeHtml(B.unavailable[cfg.embedderId]),
      actions: '<button type="button" class="btn" data-act="use-small-emb">Use Qwen3-Embedding-0.6B on Modal</button>' + useServerBtn() + (S.pool ? ' <button type="button" class="btn" data-act="dismiss">Cancel</button>' : ''),
      sticky: cfgKey(cfg),
    });
    return;
  }
  const start = (k - 1) * cfg.N, end = k * cfg.N;
  if (end > LIMITS.bufferMax) { fail(B, new RunError('limit', `At most ${LIMITS.bufferMax} completions are kept per prompt and settings.`)); return; }
  // Draws are only combined with draws from the same backend (same weights, same arithmetic).
  let buf = bufFor(cfg).find((b) => b.backend === B.id);
  if (buf) await ensureLoaded(buf).catch(() => {});
  const need = Math.max(0, end - (buf?.samples ? buf.samples.length : 0));
  cloudAbort = new AbortController();
  const signal = cloudAbort.signal;
  let timedOut = false;
  const timer = setTimeout(() => { timedOut = true; cloudAbort?.abort(); }, LIMITS.timeoutMs);
  $('#live-go').hidden = true; $('#live-stop').hidden = false;
  const t0 = performance.now(), secs = () => Math.round((performance.now() - t0) / 1000);
  const slowAfter = B.id === 'server' ? 20 : 90;
  const show = () => stageMsg('running', {
    title: need ? `Sampling ${need} completions… ${secs()} s` : `Embedding the completions… ${secs()} s`,
    // The GPU server keeps a log of runs (prompts and completions, nothing about the visitor).
    detail: `${escapeHtml(B.label)}.${B.id === 'server' ? ' Runs here are logged for research.' : ' The first run starts a GPU, which can take a minute or two.'}` +
      (secs() >= slowAfter ? ' <b>This is taking longer than usual:</b> the server may be busy with other requests, or a GPU is starting.' : ''),
    actions: '<button type="button" class="btn" data-act="stop">Stop</button>',
  });
  show();
  setStatus('');
  const tick = setInterval(show, 1000);
  const opts = { signal, auth };
  try {
    const timing = { generate_s: 0, embed_s: 0, reported: false };
    if (!buf) { buf = { prompt: cfg.prompt, model: cfg.model, backend: B.id, Tprime: cfg.Tprime, maxNewTokens: cfg.maxNewTokens, from: 'run', samples: [], batches: [] }; BUFFERS.push(buf); }
    // Sample what is missing, at most LIMITS.maxN per request, appended in order.
    for (let left = need; left > 0;) {
      const n = Math.min(left, LIMITS.maxN);
      const r = await sampleDraws(B, cfg, n, opts);
      buf.batches.push({ start: buf.samples.length, n, G: { [cfg.embedder]: r.G } });
      buf.samples.push(...r.samples);
      if (r.timing?.generate_s != null) { timing.generate_s += r.timing.generate_s; timing.embed_s += r.timing.embed_s ?? 0; timing.reported = true; }
      left -= n;
    }
    let sl = await sliceOf(buf, start, end, cfg.embedder);
    if (!sl) {
      const samples = buf.samples.slice(start, end);
      const G = await embedTexts(B, cfg, samples, opts);
      buf.batches.push({ start, n: end - start, G: { [cfg.embedder]: G } });   // an embedding-only batch over this slice
      sl = { samples, G };
    }
    const took = (need ? `Sampled in ${secs()} s` : `Embedded in ${secs()} s`) + (timing.reported ? ` (${Math.round(timing.generate_s)} s generating, ${Math.round(timing.embed_s)} s embedding)` : '') + '.';
    if (viewKey(readConfig(), k) !== viewKey(cfg, k)) {
      // The settings changed during the run: keep the result for its own settings, offer to show it.
      stageMsg(null);
      setStatus(`Done, but for earlier settings. `);
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'link'; b.textContent = 'Show it';
      b.onclick = () => restoreConfig({ ...cfg, k });
      $('#live-status').append(b);
      return;
    }
    drawK = k;
    stageMsg(null);
    showSlice(buf, sl, cfg, k);
    S.poolNote = `${took} Draw ${k} (completions ${start + 1}–${end}).`;
    setStatus(S.poolNote, { drawNext: true });
  } catch (e) {
    if (e.name === 'AbortError' && !timedOut) {
      if (S.pool) stageMsg(null); else emptyMsg(cfg);
      setStatus('Stopped.');
    } else fail(B, timedOut ? new RunError('timeout', '') : e);
  } finally {
    clearInterval(tick);
    clearTimeout(timer);
    cloudAbort = null;
    $('#live-go').hidden = false; $('#live-stop').hidden = true;
  }
}

// A failure: what happened, in plain words, and the next step that works.
function fail(B, e) {
  // Anything that is not a RunError was thrown while reading the answer (bad JSON, missing fields).
  const code = e instanceof RunError ? e.code : 'bad_answer';
  const msg = e.message ? escapeHtml(e.message) : '';
  const retry = '<button type="button" class="btn" data-act="retry">Try again</button>';
  const dismiss = S.pool ? ' <button type="button" class="btn" data-act="dismiss">Dismiss</button>' : '';
  const M = {
    unreachable: [`${cap(B.the)} isn’t responding.`, B.id === 'server' ? 'It may be offline, or not yet open to this page.' : 'It may be offline.', retry + ' ' + otherModelBtn()],
    busy: ['Many people are sampling right now.', msg || 'Try again in a minute.', retry + ' ' + otherModelBtn()],
    rate_limit: ['You have reached this server’s limit for now.', msg, otherModelBtn()],
    budget: ['This month’s GPU budget for the demo is used up.', msg, otherModelBtn()],
    quota: ['Your daily Hugging Face GPU time is used up.', msg || 'Free accounts get a few minutes a day.', otherModelBtn()],
    auth: ['Your Hugging Face sign-in has expired.', '', '<button type="button" class="btn" data-act="signin">Sign in again</button> ' + otherModelBtn()],
    starting: [`${cap(B.the)} is starting up.`, 'Try again in a minute.', retry],
    timeout: ['No answer after 5 minutes.', 'Try again, or lower N or max tokens.', retry],
    limit: ['That is more than the demo allows.', msg, ''],
    mismatch: ['The server did not run the requested settings.', '', retry],
    bad_answer: ['The server sent an answer the page could not read.', '', retry],
    bad_request: ['The server did not accept these settings.', msg, ''],
    server: ['The server hit an error while sampling.', msg, retry + ' ' + otherModelBtn()],
  }[code] ?? ['Sampling failed.', msg, retry];
  if (code === 'auth' && B.auth) { signOut(); refreshNote(); }
  // A server's detail that only repeats the title adds nothing.
  const detail = M[1] && M[1].replace(/&#?\w+;/g, '').startsWith(M[0].replace(/[.’']/g, '').slice(0, 20)) ? '' : M[1];
  stageMsg('error', { title: M[0], detail, actions: M[2] + dismiss, sticky: cfgKey(readConfig()) });
  setStatus('');
}

let pendingSignIn = null;
let refreshNote = () => {};
function restoreConfig(c) {
  // Set every field first, then let the page react once to the complete settings.
  pendingK = c.k > 1 ? c.k : null;
  $('#prompt').value = c.prompt;
  $('#model').value = c.backend;
  window.__fillEmbedders?.();
  $('#embedder').value = c.embedderId;
  $('#live-N').value = c.N; $('#live-Tp').value = c.Tprime; $('#live-max').value = c.maxNewTokens;
  $('#prompt').dispatchEvent(new Event('input'));   // the send button's state
  window.__settingsChanged?.();
}

// ---------------------------------------------------------------- settings notes and limits
function syncFieldNotes() {
  const N = clampInt($('#live-N').value, 2, LIMITS.maxN), M = clampInt($('#live-max').value, 8, LIMITS.maxTokens);
  const capN = +$('#live-N').value > LIMITS.maxN, capM = +$('#live-max').value > LIMITS.maxTokens;
  $('#limits-note').textContent = capN || capM
    ? `The demo allows at most ${LIMITS.maxN} samples and ${LIMITS.maxTokens} tokens per completion; using ${N} and ${M}.`
    : `Up to ${LIMITS.maxN} samples and ${LIMITS.maxTokens} tokens.`;
}

// ---------------------------------------------------------------- wiring
function setMode(m) {
  S.mode = m;
  document.querySelectorAll('.step-2 .pick-btn[data-mode]').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.mode === m)));
  recompute({ newDraw: true });
}
function setK(v) {
  if (!S.pool) return;
  S.K = clampInt(v, 1, Math.min(20, S.logp.length - 1));
  $('#K').value = S.K;
  recompute({ newDraw: true });
}

async function checkServer() {
  const B = BACKENDS[0];
  try {
    const r = await fetch(`${B.url}/status`, { signal: AbortSignal.timeout(6000) });
    B.up = r.ok;
    usageStatus(B, r.ok ? await r.json().catch(() => null) : null);   // usage ids and the load event (js/usage.js)
  } catch { B.up = false; }
  return B.up;
}

function wire() {
  initExplainers();
  initUsage();
  const fillModels = () => {
    $('#model').innerHTML = BACKENDS.map((B) => `<option value="${B.id}">${escapeHtml(B.label)}${B.up === false ? ' — not available right now' : ''}</option>`).join('');
  };
  fillModels();
  const fillEmbedders = () => {
    const B = backend(), keep = $('#embedder').value;
    $('#embedder').innerHTML = B.embedders.map((e) => `<option value="${e.id}">${escapeHtml(e.label)}</option>`).join('');
    if (B.embedders.some((e) => e.id === keep)) $('#embedder').value = keep;
  };
  const modelNote = () => {
    const B = backend(), a = getAuth();
    const where = B.id === 'server'
      ? (B.up === false ? 'The GPU server can’t be reached from this page right now. Precomputed examples still show.' : 'Runs on a GPU server shared by everyone using this demo.')
      : B.id === 'modal' ? 'Runs on a cloud GPU (Modal); needs a free Hugging Face sign-in.'
        : 'Runs on a Hugging Face ZeroGPU Space and uses your account’s daily GPU quota; needs a free Hugging Face sign-in.';
    $('#model-note').innerHTML = escapeHtml(where) + (B.auth && a ? ` Signed in as <b>${escapeHtml(a.user)}</b>. <button type="button" class="link" id="hf-signout">Sign out</button>` : '');
    $('#hf-signout')?.addEventListener('click', (e) => { e.stopPropagation(); signOut(); modelNote(); });
  };
  refreshNote = modelNote;
  const advSum = () => ($('#adv-sum').textContent = `${backend().short} · ${readConfig().N} samples`);
  const onSettings = () => { fillEmbedders(); advSum(); modelNote(); configChanged(); };
  window.__settingsChanged = onSettings;
  window.__fillEmbedders = fillEmbedders;
  if (window.__dbg) Object.assign(window, { __S: S, __plot: plot, __callSpace: (...a) => callSpace(BACKENDS[2], ...a) });
  $('#model').addEventListener('change', onSettings);
  $('#embedder').addEventListener('change', onSettings);
  ['#live-N', '#live-Tp', '#live-max'].forEach((id) => $(id).addEventListener('change', onSettings));
  // The target quality is part of κ, like T: it reweights the same completions, so it sits with T and K.
  $('#target').innerHTML = Object.entries(TARGETS).map(([k, t]) => `<button type="button" class="pick-btn" data-target="${k}" aria-pressed="${k === S.target}">${t.short}</button>`).join('');
  $('#target').querySelectorAll('.pick-btn').forEach((b) => (b.onclick = () => {
    S.target = b.dataset.target;
    $('#target').querySelectorAll('.pick-btn').forEach((x) => x.setAttribute('aria-pressed', String(x === b)));
    recompute({ newDraw: true });
  }));
  let pt = 0;
  $('#prompt').addEventListener('input', () => { clearTimeout(pt); pt = setTimeout(configChanged, 250); });
  fillEmbedders(); advSum(); modelNote(); syncFieldNotes();
  // The GPU server: if it can't be reached when the page opens, say so and default to Modal for this visit.
  checkServer().then((up) => {
    const keep = $('#model').value;
    fillModels();
    $('#model').value = !up && keep === 'server' ? 'modal' : keep;
    onSettings();
  });

  S.T = sliderToT(+$('#T-slider').value);   // the HTML is the one source of the starting T
  let raf = 0;
  $('#T-slider').addEventListener('input', (e) => {
    S.T = sliderToT(+e.target.value);
    $('#T-out').textContent = fmtT(S.T);
    cancelAnimationFrame(raf);
    raf = requestAnimationFrame(() => recompute({ newDraw: true }));
  });
  // "Most likely" shows the greedy MAP subset; "Sample" draws a new subset every time it is pressed.
  document.querySelectorAll('.step-2 .pick-btn[data-mode]').forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $('#K').onchange = () => setK($('#K').value);
  $('#K-minus').onclick = () => setK(S.K - 1);
  $('#K-plus').onclick = () => setK(S.K + 1);
  $('#label-toggle').onchange = () => renderLabels();
  // Pop-up panels close on an outside click or Escape.
  const pops = ['#viewmenu', '#specmenu', '#adv', '#exmenu'];
  document.addEventListener('click', (e) => { for (const id of pops) { const m = $(id); if (m?.open && !e.composedPath().includes(m) && !e.target.closest('.xpop')) m.open = false; } });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { const was = pops.some((id) => $(id)?.open); for (const id of pops) if ($(id)) $(id).open = false; if (!was && wide.matches) setDrawer(false); } });
  // Details: a drawer on wide screens, a collapsible section on phones.
  const setDrawer = (open) => {
    $('#details').classList.toggle('open', open);
    for (const b of [$('#details-btn'), $('#details-toggle')]) b.setAttribute('aria-expanded', String(open));
  };
  $('#details-btn').onclick = () => setDrawer(!$('#details').classList.contains('open'));
  $('#details-toggle').onclick = () => { if (!wide.matches) setDrawer(!$('#details').classList.contains('open')); };
  $('#details-close').onclick = () => setDrawer(false);
  $('#all-list').addEventListener('toggle', () => renderAllCards());
  $('#readout').addEventListener('click', (e) => {
    const v = e.target.closest('#fit-why');
    if (!v) return;
    if (!wide.matches) { $('#fit-note').classList.toggle('show'); return; }   // phones: shown in the bar, under the readouts
    // Open just above the value that was clicked, kept on screen.
    const note = $('#fit-note'), bar = $('#details').getBoundingClientRect(), r = v.getBoundingClientRect();
    note.classList.toggle('show');
    note.style.left = `${Math.max(8, Math.min(r.left - bar.left - 16, bar.width - Math.min(520, bar.width - 16) - 8))}px`;
  });
  document.addEventListener('click', (e) => { if (!e.target.closest('#fit-note, #fit-why')) $('#fit-note').classList.remove('show'); });
  // The controls column starts under the title block, whatever its height: keep --top-h equal to it.
  const top = document.querySelector('.top');
  new ResizeObserver(() => {
    const h = Math.ceil(top.getBoundingClientRect().bottom);
    if (h) document.documentElement.style.setProperty('--top-h', `${h}px`);
  }).observe(top);
  // The plot ends where the Details bar begins (wide screens): keep --bar-h equal to the bar's height.
  const bar = $('#details');
  new ResizeObserver(() => {
    const h = wide.matches ? Math.ceil(bar.getBoundingClientRect().height) : 0;
    if (h) document.documentElement.style.setProperty('--bar-h', `${h}px`);
  }).observe(bar);
  for (const [a, b] of [['#adv', '#exmenu'], ['#exmenu', '#adv']]) $(a).addEventListener('toggle', () => { if ($(a).open) $(b).open = false; });
  $('#live-go').onclick = () => runLive({ k: 1 });
  const syncGo = () => ($('#live-go').disabled = !$('#prompt').value.trim());
  $('#prompt').addEventListener('input', syncGo);
  $('#prompt').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!$('#live-go').hidden && !$('#live-go').disabled) runLive({ k: 1 }); }
  });
  syncGo();
  $('#live-stop').onclick = () => cloudAbort?.abort();
  $('#stage-msg').addEventListener('click', (e) => {
    const act = e.target.closest('[data-act]')?.dataset.act;
    if (act === 'stop') cloudAbort?.abort();
    if (act === 'retry') runLive({ k: drawK });
    if (act === 'dismiss') { stageMsg(null); if (!S.pool) emptyMsg(readConfig()); }
    if (act === 'settings') { stageMsg(S.pool ? null : 'empty', S.pool ? {} : { title: 'Pick another model, then press ↑.' }); $('#adv').open = true; $('#model').focus(); }
    if (act === 'use-small-emb') { stageMsg(null); $('#embedder').value = 'qwen3-emb'; onSettings(); runLive({ k: 1 }); }
    if (act === 'use-server') { stageMsg(null); $('#model').value = 'server'; onSettings(); if (!S.pool) emptyMsg(readConfig()); }
    if (act === 'signin') {
      // The settings on screen now (a panel may have been up while they changed).
      const now = readConfig(), cfg = pendingSignIn && cfgKey(pendingSignIn) === cfgKey(now) ? pendingSignIn : { ...now, k: drawK };
      Promise.resolve().then(() => signIn({ cfg })).catch((err) => stageMsg('error', {
        title: 'Could not start the Hugging Face sign-in.', detail: escapeHtml(err?.message || ''),
        actions: '<button type="button" class="btn" data-act="signin">Try again</button>' + useServerBtn(), sticky: cfgKey(now),
      }));
    }
  });
  // Both plots follow their box's size themselves and keep the visitor's view (no rebuild on window
  // resize: on phones that event fires whenever scrolling shows or hides the address bar, and a rebuild
  // replayed the opening turn and discarded the view). Only a switch between the phone and the wide
  // layout, where the panels move entirely, frames the plot afresh.
  wide.addEventListener('change', () => { if (!S.pool) return; plot.reset(); render(); });
}

wire();
// Coming back from Hugging Face: restore the settings; run if the sign-in succeeded.
finishSignIn().then((r) => {
  if (!r) return;
  const c = r.stash?.cfg;
  if (c) restoreConfig(c);
  if (r.error) {
    stageMsg('error', { title: 'Sign-in was cancelled.', detail: escapeHtml(String(r.error)), actions: '<button type="button" class="btn" data-act="signin">Sign in with Hugging Face</button>' + useServerBtn() + ' <button type="button" class="btn" data-act="dismiss">Dismiss</button>', sticky: cfgKey(readConfig()) });
    return;
  }
  refreshNote();
  if (c) { setStatus(`Signed in to Hugging Face as ${r.auth.user}.`); runLive({ k: c.k ?? 1 }); }
  else stageMsg('empty', { title: `Signed in to Hugging Face as ${escapeHtml(r.auth.user)}.`, detail: 'Your settings could not be kept through the sign-in. Press ↑ to run.', actions: '<button type="button" class="btn" data-act="dismiss">OK</button>', sticky: '*' });
}).catch((e) => setStatus('Sign-in failed: ' + (e.message || e)));
// The page stays hidden (html.booting) until its first complete render, so no half-built state shows:
// then it appears in one step. Never longer than 6 s, whatever happens.
const reveal = () => document.documentElement.classList.remove('booting');
setTimeout(reveal, 6000);
loadData().then(configChanged).catch((e) => {
  stageMsg('error', { title: 'The examples could not be loaded.', detail: escapeHtml(e.message || '') + ' You can still type a prompt and press ↑.', actions: '<button type="button" class="btn" data-act="reload" onclick="location.reload()">Try again</button>', sticky: '*' });
}).then(() => Promise.race([document.fonts?.ready, new Promise((r) => setTimeout(r, 1500))])).then(() => requestAnimationFrame(reveal));
