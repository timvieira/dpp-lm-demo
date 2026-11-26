# To do

## Done: the full-page layout
The plot fills the page between the controls (left) and the selected subset and all completions as
cards (right rail); readouts, Spectrum and View in a bar at the bottom. The 3-D view turns about the
origin, opens with one 30° turn (no auto-rotation), starts with the drawing centred, pans with
Shift/right-drag or two fingers, and resets on double-click. Floor grid and x, y, z axes with arrows.

## Done (the batch agreed in the review of Manuel's demo)

Log-space DPP routines with brute-force reference tests; Householder + QL eigensolver; exact G from
every backend and in the cache; buffers of i.i.d. draws with draw slices; Target quality (Global /
Global and local), one T, T = ∞; limits (N ≤ 200, max tokens ≤ 256, buffer ≤ 1000; the N × tokens
limit was dropped: 200 × 256 = 51,200, so it could never bind); /embed on Modal and the Space;
docs/BACKEND.md and scripts/conformance.mjs; pool fit θ̂; fitted axes for K ≤ 3 with rotation;
inclusion on the cards; chip and status line; run-flow fixes (docs/run-flow.md); the GPU server
marked unavailable at load; in-browser code deleted; spectrum panel in the strip.

Cut (not doing): Manuel's Expert mode (paper presets, T′ picker, shared-T switch, weight switch,
vector-length lock, point size for quality, unit positions, open the cone, axes toggle), the
multi-model rail and morphing, his extra data files, in-browser models, Gemma, mpnet, owner-only
routes in the page, the drawn-vs-true volume readout, paper T ticks, the T = T′ marker.

## Next: the matrix views (in discussion)
Full notes: docs/matrix-views.md (the views, the kernel / eigendecomposition / PCA relationships, open questions).
- A, high priority: the pool's similarity matrix, reordered so similar completions form blocks (VAT,
  Bezdek & Hathaway 2002; Manuel's "pairs of completions" panel is the start). Details under discussion.
  A dendrogram only if its ordering is coherent with the DPP's geometry (no arbitrary linkage rule).
- Themes: eigenvectors of the kernel as themes over the completions (the dual view is where they live);
  effective dimension Σ λ/(λ+1) (also explains "K too large").
- Conditioning (Schur complement): shared by greedy MAP and sampling; value as an exploration tool TBD.
- Pairwise repulsion: π_ij/(π_i π_j) (the marginal-kernel picture; a K-DPP needs π_ij computed directly).
- Decided (Tim): always L, the matrix the algorithm uses (G is L for the Global target at T = ∞).
- The lab, a separate page for playing with orderings: https://timvieira.github.io/dpp-lm-demo/lab/matrix.html
  (lab/; tests in tests/orderings.test.mjs, run before each deploy). Findings so far (details in the notes):
  - Average linkage (Manuel's) draws the cleanest blocks, but cannot see L: the cosine of L is G.
  - Pivoted Cholesky (greedy volume) gets the coarse blocks right; the insides are striped. The nested tree
    puts outliers first at T = ∞ (with no quality, greedy volume favours them).
  - Themes, by the exact split π_i = Σ_a p_a u_a(i)²: clean blocks on the joke preset; on the story preset
    the first theme is a large, noisier block.
- Decided (Tim): show block strength honestly. Each completion's exact profile over the themes,
  r_i(a) = p_a u_a(i)² / π_i, says how cleanly it belongs to its block; the lab shows its own theme's share
  beside each row and the pool's mean (joke 66%, story 33%: story openings have no clean blocks).
- Open: which ordering to use (themes are the principled candidate); the order inside a block (proposal: by
  the rest of the profile); whether conditioning earns a place on the main page (built in the lab: click up to K rows; the
  matrix shows the conditioned kernel, and hover shows π before and after).
- Theme profiles as stacked colour bars beside the rows (lab/matrix.html): absolute scale (length π_i, segments
  p_a u_a(i)², each colour sums to p_a), three validated muted colours clear of the accent, rest in grey with its
  share, near-degenerate eigenvalues merged (5% knob, open), colours matched to the previous eigenvectors.
- Theme cards (lab/matrix.html): each coloured theme explained by the completions it would produce, not a name.
  Theme 1 is the shared core (one-signed by Perron–Frobenius; L is positive in all 18 presets); later themes are
  contrasts shown by their two ends' most representative completions (largest u_a(i)²) and shares. Hover a card
  to pick out its rows.
- Prototype "Building a set" (https://timvieira.github.io/dpp-lm-demo/lab/set.html, lab/set.html), kept separate from
  the main page for now: the matrix and the main page's 3-D plot, linked. The set builds in a rail of cards (×
  removes, Ctrl/⌘+Z undoes); its members are the red vectors in the plot, live; hovering a cell draws the pair of
  vectors it compares (with the angle); hovering a row shows it as a candidate card (volume factor, π before and
  after). Open: what to do when the set already has K (today: the candidate card says to remove one first).

## Prototype: the ensemble demo (prompt intersection, arXiv:2603.05432 Fig. 1)
- Brief: docs/handoff-ensemble-demo.md. Built here as an unlinked lab page (lab/ensemble.html), to move
  to its own repository once it works.
- Decided (Tim): Qwen3 14B (the GPU server's model); cached pools for the presets during development
  (data/ensemble/presets.json).
- Waiting on Manuel: `/score` (texts scored under given prompts) first; the token-level product sampler
  with local scores second. Today's server has neither (checked 26 Nov). With `/score` alone, a 50/50
  mixture of A and B draws gives the global ensemble exactly; local scores wait for the sampler.
- Meanwhile: stand-in pools from Qwen3-0.6B on CPU (scripts/ensemble_pool.py), marked as stand-ins.

## Run log and shared cache (for us, not a visitor feature)
- Done on Manuel's server (25 Nov): every run is logged (nothing about the visitor), and the shared draw
  cache exists, opt-in through `offset` (see "From Manuel's update" below). Our page says "Runs here are
  logged for research." while a GPU-server run is in progress.
- Open: an export of the log to a private place we can download from (ask Manuel); Modal (on hold).
- Open question: should ↑ on a cached setting add fresh draws, or show the cached ones? (Same decision as
  whether our page sends `offset`.)

## From Manuel's update (25 Nov): decide what to take
- His server, beyond contract v1:
  - a queue wait over 600 s returns 503 `busy`, and a restarting engine returns 503 `starting`, so BACKEND.md's
    error table should gain both;
  - `emb` is sent only with `"emb": true`;
  - a run log (nothing about the visitor; `logsRuns` in /status);
  - a private usage store (no prompt text or addresses; our visitors are counted per day by a salted hash
    that is never stored);
  - an opt-in shared draw cache: send an integer `offset` and `/sample` returns stream positions
    [offset, offset + n), sampling only the shortfall (server/API.md, "Draw cache").
- His server logs runs, including runs from our page. Decided (Tim): only a minimal note, in the message shown
  while a run on the GPU server is in progress ("Runs here are logged for research.").
- Our page does not send `offset` yet. Decide whether "Draw new ones" should read the stream (this answers the
  run-log item's open question).
- His page features are his own exploration, not proposals for this page. This page's experience is
  curated, so nothing from his fork comes in unless Tim picks it. For the record:
  - "Sample again" as a refresh icon on the Sample button;
  - "Color by inclusion probability" in the View menu.

Done (25 Nov): Manuel merged the own-states PR and moved his server to contract v1. Checked here: /status
says version 1 with limits 200 / 256, and n = 201 and max_new_tokens = 300 get 400 bad_request.

## The LM's own states as an embedding
- Decided: exact own states, no random projection. The sampler sums each sequence's raw logits; the
  server recovers the full 5120-d mean hidden state ((AᵀA)⁻¹Aᵀ l̄, A = [W | 1]), centres it, and sends G.
  Pooling is the mean over generated steps, including the step that samples EOS (from his code).
- PR opened on Manuel's fork (branch own-exact on timvieira/dpp-lm-demo, one commit on his main
  e769c13). Draft, note and numpy check in docs/manuel/ on the scratch branch (own-exact.patch, own-exact-PR.md,
  own_exact_check.py). Tested on the GPU by Manuel (below).
- Done on Manuel's side (25 Nov): merged and live. M built, cond(A) = 300; the hook sees raw logits (no
  −inf, not divided by T′); recovery error 1.7e-4 per step (vLLM's logits are float16), max |ΔG| 7.7e-6
  against vLLM's own final-norm states. His fix: /embed_own refuses more than 400 texts.
- Done (25 Nov): data/cache/*.own.G.bin replaced by his exact files (18; pools identical to ours; symmetric,
  diagonal 1; max |ΔG| 0.165 against the 256-d ones; correlation with Qwen3-Embedding-4B up on 15 of 18).
- Next: check the own states against Qwen3-Embedding (the picks on the joke and sorting presets).
- Modal and the Space: no own states for now (his P files are the projection we are dropping).

## Later: Qwen3-Embedding-4B on Modal
- Modal's L40S (48 GB) cannot hold Qwen3 14B (≈30 GB) plus the 4B embedder in float32 (16 GB) and the
  generation memory. Options: an 80 GB A100 (≈25–30% more per GPU-second; exact, like the GPU server), or
  the embedder in bfloat16 on the L40S (similarities off by ~1e-3). Until then the page lists 4B on Modal
  and explains, with a way forward, if it is chosen.

## Later: toggling the embedding as a knob on the same completions
- Interesting for the future (Tim): switching the embedding keeps the completions and only changes G,
  so it is a re-embedding, not a new draw; it could sit with T, K and the target as a way to compare
  what different notions of similarity pick from the same pool. For now it stays in the upper panel.
- Within one backend this already holds: a missing embedding re-embeds the existing completions
  (runLive, embedTexts). Two gaps if it becomes a lower-panel knob: the empty-pool message says
  "Press ↑ to sample N completions" when only embedding is needed; and on another backend than the one
  that drew the completions, ↑ draws N new ones (draws are reused only per backend) instead of
  re-embedding those on screen.

## Untested
- Hugging Face sign-in end to end, now that the page is back at the main address.
- Two-finger panning on a real touch screen.

## Later: pool fit and the default T (see docs/figures/pool-fit-vs-T.png)
- Report the fit as an effective number of completions, N/ρ̂ = 1/Σ(π_i/K)² (between K and N by
  construction), e.g. "the target effectively uses 4 of these 200 completions". Keep θ̂ in nats in the
  popover. (The 1-nat warning threshold is already gone.) The note could show the count and which way raises it
  (for the Global target, move T toward 1). Optionally translate θ̂ into a total-variation bound,
  TV ≤ min(√(θ/2), √(1 − e^(−θ))).
- The default T. T and T′ are unrelated (T′ tempers each token, T the whole completion), so T = T′ is
  not a matched point. Global matches the pool only at T = 1 with T′ = 1 (then κ = p = q); for T′ ≠ 1 it
  never matches. Global and local matches at T = ∞ for every T′ (κ = p°_{T′} = q). Today's default,
  Global at T = 20, sits on the poor-fit plateau. Options: Global at T = 1, or Global and local at large T.
- What the options give (the 9 presets drawn at T′ = 1, K = 3, Global):
  - T = 1: fit at the floor (Sample uses 95–196 of 200 completions), but Most likely picks near-duplicates
    (closest pair's cosine 0.76–0.99; e.g. two "The last letter from her brother…" openings, two skeleton jokes).
  - T = 20: diverse picks (closest pair 0.38–0.81), but Sample effectively uses only 3–10 completions.
  - Global and local at T = 20 gives the same picks as T = 1 here (with T′ = 1, κ = p^{1/20}·p ≈ p).
  - The tension is built in: the pool can only represent targets close to the distribution it was drawn
    from, which is dominated by quality; a target flat enough for diversity leans on the pool's rare
    completions, where the fit breaks.
- Undecided. Claude's recommendation: keep T = 20 (the default mode is Most likely, and the fit is about
  Sample; the pool-fit readout already flags Sample there). The effective-count readout above would make
  that flag easy to read.

## Later: a "follow the action" camera (decide later)
- Today the view is set once, when the points are first drawn (axes, zoom, framing), and then only the
  visitor moves it. An optional mode could re-frame or re-fit the axes as the selection changes.

## Later: the bouquet (all completions point one way from the origin)
- Completions of one prompt share a large common direction (cosines 0.5–0.9; on the story prompt the
  first direction carries 52% of the spectrum), so the cloud is a narrow bouquet away from the origin.
  Under the paper's cosine kernel this is honest: diversity is volume measured from the origin.
- Options: (A) keep it and add a "?" explaining it; (B) centre the kernel on the pool's mean: pool-
  dependent, leaves the paper's fixed kernel and Theorem 5.2, so avoid; (C) the RBF kernel
  S_β = exp∘(β(G − 1)): origin-free (small β: det ≈ β^(K−1)·det(EᵀE), E = φ_j − φ_1), still a fixed kernel
  over strings, centred picture. Decide together with whether option C ships (see the DPP/GP handoff).

## Later (from the review of Manuel's demo; the matrix views above now cover the kernel-matrix items)
- Clustering: colour completions by cluster, and/or order completions by cluster (Manuel's demo has
  four methods, a group count and LLM-written group names). Decide whether any of it is faithful and
  useful enough; colour must follow the house style.
- Views of the similarity structure: show the embedding space's notion of similarity directly
  (the spectrum; the distribution of pairwise similarities as a whole, not thresholded; possibly
  a monochrome kernel matrix ordered without clustering).
- Kernel matrix visualizations: we want a version of these (Manuel's primal L = BBᵀ and dual C = BᵀB
  heatmaps are the starting point). Design it to fit the demo: house colours, compact, no hand-picking of the subset.
- When K is too large for the pool, explain it from the spectrum (effective dimension), not a threshold.


## Question: warm up the Modal GPU when a visitor arrives?
- It would cut the cold start (loading Qwen3 14B onto the GPU) from the first run's wait, a clear win for
  the visitor's experience.
- The cost: every visit, including the many that never run anything, would pay for the GPU's start-up plus
  its idle time before it shuts down, all out of the $30 credit.
- Options to weigh: warm on arrival; warm on a sign of intent instead (choosing Modal, focusing the prompt,
  signing in), which pays only for likely runs; or no warm-up, with an honest "starting the GPU" message.
  Measure the cold-start time and the cost per warm-up before deciding.
