"""Sampling backend for the diverse-sampling demo, on Modal.

Same job as the Hugging Face Space (space/app.py): draw N completions i.i.d. from the locally
annealed proposal q = p°_{T'} and return, for each one, the exact log p(y) and log q(y)
(end-of-string included), plus unit-norm MiniLM embeddings. The page does the DPP part.

Spending guards (the app must stay inside Modal's free monthly credit):
  * only visitors signed in with Hugging Face may call it (their token is checked with HF);
  * each visitor gets RUNS_PER_USER_PER_DAY runs a day;
  * GPU time is metered and the endpoint refuses work once GPU_SECONDS_PER_MONTH is used;
  * one GPU container at most, shut down after SCALEDOWN seconds idle; inputs are capped.

Deploy:   modal deploy modal/app.py
Weights:  modal run modal/app.py::download   (once; fills the volume on a CPU machine)
"""
import datetime
import os
import pathlib
import time

import modal

MODEL_KEY = "qwen3-14b"
MODEL_REPO = "Qwen/Qwen3-14B-AWQ"   # 4-bit; the same weights as the demo's GPU server
CHAT_KW = {"enable_thinking": False}
EMBEDDER = "Qwen/Qwen3-Embedding-0.6B"   # the same embedder as the demo's GPU server
EMBEDDER_ID = "qwen3-emb"
CONTRACT_VERSION = 1
GPU = "L40S"                    # 48 GB: Qwen3-14B (4-bit weights unpacked to fp16, 30 GB) plus the KV cache
GPU_DOLLARS_PER_SEC = 0.000542  # modal.com/pricing, L40S
# $30/month free credit. Budget ~$21 of GPU time and leave the rest for CPU, memory and slack.
GPU_SECONDS_PER_MONTH = int(os.environ.get("GPU_SECONDS_PER_MONTH", 21 / GPU_DOLLARS_PER_SEC))
RUNS_PER_USER_PER_DAY = 20
SCALEDOWN = 120                 # seconds an idle GPU container stays up
MAX_N, MAX_TOKENS = 200, 256   # docs/BACKEND.md: the limits every backend enforces
ORIGINS = ["https://timvieira.github.io", "http://localhost:8765"]

app = modal.App("dpp-lm-sampler")
cache = modal.Volume.from_name("dpp-lm-hf-cache", create_if_missing=True)
usage = modal.Dict.from_name("dpp-lm-usage", create_if_missing=True)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "transformers==5.24.0", "sentence-transformers==6.3.0", "numpy")
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    # The ensemble prototype's pool code (plain PyTorch, shared with scripts/ensemble_pool.py).
    .add_local_file(pathlib.Path(__file__).parent / "ensemble_lib.py", "/root/ensemble_lib.py")
)
web_image = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]")


def _month():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m")


def _charge(seconds):
    """Add GPU seconds to this month's meter."""
    key = f"gpu-seconds:{_month()}"
    usage[key] = usage.get(key, 0) + float(seconds)


@app.function(image=gpu_image, volumes={"/cache": cache}, timeout=3600)
def download():
    """Fetch the weights into the volume (CPU only, so it costs almost nothing)."""
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_REPO, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    snapshot_download(EMBEDDER)
    cache.commit()
    print("weights cached")


@app.cls(image=gpu_image, gpu=GPU, volumes={"/cache": cache}, max_containers=1,
         scaledown_window=SCALEDOWN, timeout=600)
class Sampler:
    @modal.enter()
    def load(self):
        import torch
        from sentence_transformers import SentenceTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.started = time.time()
        # Charge the start-up and the final idle window up front; exit() settles the rest.
        self.charged = 60 + SCALEDOWN
        _charge(self.charged)
        self.tok = AutoTokenizer.from_pretrained(MODEL_REPO)
        self.model = load_awq_unpacked(MODEL_REPO, "cuda")
        # float32 throughout, so the similarity matrix is as accurate as the embedder allows.
        self.embedder = SentenceTransformer(EMBEDDER, device="cuda", model_kwargs={"dtype": torch.float32},
                                            processor_kwargs={"padding_side": "left"})

    @modal.exit()
    def settle(self):
        _charge(max(0.0, time.time() - self.started - self.charged))

    @modal.method()
    def score(self, prompt, texts):
        """log p(y | prompt) of finished completions, end-of-turn token included (for checks)."""
        import torch
        head = self.tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                            add_generation_prompt=True, **CHAT_KW)
        h = self.tok(head, add_special_tokens=False).input_ids
        eos = self.model.generation_config.eos_token_id
        eos = eos[0] if isinstance(eos, (list, tuple)) else eos
        out = []
        with torch.no_grad():
            for t in texts:
                y = self.tok(t, add_special_tokens=False).input_ids + [eos]
                ids = torch.tensor([h + y], device="cuda")
                lp = torch.log_softmax(self.model(input_ids=ids).logits[0, len(h) - 1:-1].float(), -1)
                out.append(lp.gather(1, ids[0, len(h):, None]).sum().item())
        return out

    def _gram(self, texts):
        """Similarity matrix of unit embeddings: float64 product of float32 vectors, sent as float32."""
        import base64
        import numpy as np
        emb = self.embedder.encode([t or "(empty)" for t in texts], normalize_embeddings=True, batch_size=16)
        E = np.asarray(emb, dtype=np.float64)
        E /= np.linalg.norm(E, axis=1, keepdims=True)
        G = E @ E.T
        np.fill_diagonal(G, 1.0)
        return base64.b64encode(G.astype("<f4").tobytes()).decode("ascii"), int(E.shape[1])

    @modal.method()
    def sample_pool(self, prompt, n, t_prime, max_new_tokens, seed):
        t0 = time.time()
        try:
            samples = sample(self.tok, self.model, CHAT_KW, prompt, n, t_prime, max_new_tokens, 100, seed)
            t1 = time.time()
            G, D = self._gram([s["text"] for s in samples])
        finally:
            dt = time.time() - t0
            self.charged += dt
            _charge(dt)
        return {
            "prompt": prompt, "model": MODEL_REPO, "modelName": MODEL_REPO.split("/")[-1],
            "embedder": EMBEDDER, "embedderId": EMBEDDER_ID, "Tprime": t_prime, "maxNewTokens": max_new_tokens,
            "N": len(samples), "D": D, "seed": seed, "samples": samples, "G": G,
            "timing": {"generate_s": round(t1 - t0, 2), "embed_s": round(time.time() - t1, 2)},
        }

    @modal.method()
    def ensemble_pool(self, preset, n, t_prime, max_new_tokens, seed, stop="."):
        """A pool for the ensemble prototype (modal/ensemble_lib.py; docs/handoff-ensemble-demo.md)."""
        from ensemble_lib import ensemble_pool
        t0 = time.time()
        try:
            return ensemble_pool(self.tok, self.model, preset, n=n, t_prime=t_prime, max_new_tokens=max_new_tokens,
                                 seed=seed, stop=stop, model_name=MODEL_REPO)
        finally:
            dt = time.time() - t0
            self.charged += dt
            _charge(dt)

    @modal.method()
    def embed(self, texts):
        t0 = time.time()
        try:
            G, D = self._gram(texts)
        finally:
            dt = time.time() - t0
            self.charged += dt
            _charge(dt)
        return {"embedder": EMBEDDER, "embedderId": EMBEDDER_ID, "N": len(texts), "D": D, "G": G}


def unpack_awq(qweight, qzeros, scales, group_size, bits=4):
    """AWQ "gemm" packing (AutoAWQ): 8 4-bit values per int32, in the order 0 2 4 6 1 3 5 7.
    Returns the linear layer's weight [out, in] = ((q - zero) * scale)^T, in the scales' dtype."""
    import torch
    shifts = torch.arange(0, 32, bits, device=qweight.device)
    def unpack(x):
        v = torch.bitwise_right_shift(x[:, :, None], shifts[None, None, :]).view(x.shape[0], -1)
        order = torch.arange(v.shape[1], device=x.device).view(-1, 32 // bits)[:, [0, 4, 1, 5, 2, 6, 3, 7]].reshape(-1)
        return torch.bitwise_and(v[:, order], 2**bits - 1)
    q, z = unpack(qweight), unpack(qzeros)
    w = (q - z.repeat_interleave(group_size, 0)).to(scales.dtype) * scales.repeat_interleave(group_size, 0)
    return w.t().contiguous()


def load_awq_unpacked(repo, device, path=None):
    """The AWQ checkpoint as a plain fp16 model: every 4-bit weight unpacked to (q - zero) * scale.
    These are exactly the 4-bit model's weights; only the matrix kernels differ from an int4 engine.
    The model is built on `device` without random initialisation, then filled tensor by tensor."""
    import glob, json, os
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.initialization import no_init_weights
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo)
    qc = json.load(open(os.path.join(path, "config.json")))["quantization_config"]
    assert qc["quant_method"] == "awq" and qc["version"] == "gemm" and qc["bits"] == 4 and qc["zero_point"]
    cfg = AutoConfig.from_pretrained(path)
    cfg.__dict__.pop("quantization_config", None)
    with no_init_weights(), torch.device(device):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16)
    params = dict(model.named_parameters())
    done, packed = set(), {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with safe_open(f, framework="pt", device=str(device)) as st:
            for k in st.keys():
                if k.endswith((".qweight", ".qzeros", ".scales")):
                    base, part = k.rsplit(".", 1)
                    packed.setdefault(base, {})[part] = st.get_tensor(k)
                    t = packed[base]
                    if len(t) == 3:
                        w = unpack_awq(t["qweight"], t["qzeros"], t["scales"].to(torch.float16), qc["group_size"])
                        params[base + ".weight"].data.copy_(w)
                        done.add(base + ".weight")
                        del packed[base], w
                else:
                    params[k].data.copy_(st.get_tensor(k))
                    done.add(k)
    if getattr(cfg, "tie_word_embeddings", False):
        model.tie_weights()
        done |= {k for k in params if k.startswith("lm_head")}
    missing = set(params) - done
    assert not missing and not packed, (sorted(missing)[:5], list(packed)[:5])
    return model.eval()


def sample(tok, model, chat, prompt, n, t_prime, max_new_tokens, batch_size, seed):
    """Exact ancestral sampling from p°_{T'} with per-string log p and log q (as in space/app.py)."""
    import inspect
    import torch
    with torch.no_grad():
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True, **chat)
        dev = next(model.parameters()).device
        prompt_ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
        ids = model.generation_config.eos_token_id
        ids = ids if isinstance(ids, (list, tuple)) else [ids if ids is not None else tok.eos_token_id]
        eos = {int(i) for i in ids if i is not None}
        eos_t = torch.tensor(sorted(eos), device=dev)
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        cache_kw = "cache" if "cache" in inspect.signature(type(model).forward).parameters else "past_key_values"
        out = []
        while len(out) < n:
            B = min(batch_size, n - len(out))
            logp = torch.zeros(B, dtype=torch.float64, device=dev)
            logq = torch.zeros(B, dtype=torch.float64, device=dev)
            done = torch.zeros(B, dtype=torch.bool, device=dev)
            ntok = torch.zeros(B, dtype=torch.long, device=dev)
            steps, past, step_in = [], None, prompt_ids.repeat(B, 1)
            for _ in range(max_new_tokens):
                res = model(input_ids=step_in, use_cache=True, **{cache_kw: past})
                past = getattr(res, "cache", None) or getattr(res, "past_key_values", None)
                logits = res.logits[:, -1, :].float()
                lp1 = torch.log_softmax(logits, dim=-1)            # log p(. | prefix)
                lpT = torch.log_softmax(logits / t_prime, dim=-1)  # log p°_{T'}(. | prefix)
                nxt = torch.multinomial(lpT.exp(), 1, generator=gen).squeeze(1)
                active = ~done
                nxt = torch.where(active, nxt, eos_t[0])
                logp += torch.where(active, lp1.gather(1, nxt[:, None]).squeeze(1).double(), 0.0)
                logq += torch.where(active, lpT.gather(1, nxt[:, None]).squeeze(1).double(), 0.0)
                ntok += active.long()
                steps.append(nxt)
                done |= active & torch.isin(nxt, eos_t)
                if bool(done.all()):
                    break
                step_in = nxt[:, None]
            toks = torch.stack(steps, 1).cpu().tolist()
            ntok, done, logp, logq = ntok.cpu().tolist(), done.cpu().tolist(), logp.cpu().tolist(), logq.cpu().tolist()
            for b in range(B):
                seq = toks[b][: ntok[b]]
                out.append({
                    "text": tok.decode([t for t in seq if t not in eos], skip_special_tokens=True).strip(),
                    "logp": logp[b], "logq": logq[b], "nTokens": ntok[b], "ended": bool(done[b]),
                })
        return out


def hf_user(token):
    """The Hugging Face username a token belongs to, or None."""
    import json
    import urllib.request
    for url in ("https://huggingface.co/oauth/userinfo", "https://huggingface.co/api/whoami-v2"):
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
            with urllib.request.urlopen(req, timeout=10) as r:
                info = json.load(r)
            name = info.get("preferred_username") or info.get("name")
            if name:
                return name
        except Exception:
            continue
    return None


@app.function(image=web_image, scaledown_window=60, timeout=900)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    import asyncio
    import random
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware

    api = FastAPI()
    api.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"],
                       allow_headers=["authorization", "content-type"])

    from fastapi.responses import JSONResponse

    class Fail(Exception):
        def __init__(self, status, code, detail):
            self.status, self.code, self.detail = status, code, detail

    @api.exception_handler(Fail)
    async def on_fail(_req, e):
        return JSONResponse({"code": e.code, "detail": e.detail}, status_code=e.status)

    async def budget_left():
        return GPU_SECONDS_PER_MONTH - await usage.get.aio(f"gpu-seconds:{_month()}", 0)

    async def visitor(req):
        auth = req.headers.get("authorization", "")
        token = auth[7:] if auth.lower().startswith("bearer ") else ""
        user = await asyncio.to_thread(hf_user, token) if token else None
        if not user:
            raise Fail(401, "auth", "Sign in with Hugging Face to use this model.")
        if await budget_left() <= 0:
            raise Fail(429, "budget", "This month's GPU budget for the demo is used up.")
        return user

    async def body_of(req):
        try:
            return await req.json()
        except Exception:
            raise Fail(400, "bad_request", "The request body is not JSON.")

    @api.get("/status")
    async def status():
        return {
            "version": CONTRACT_VERSION, "model": MODEL_REPO, "modelKey": MODEL_KEY,
            "embedders": {EMBEDDER_ID: EMBEDDER}, "defaultEmbedder": EMBEDDER_ID,
            "limits": {"maxN": MAX_N, "maxNewTokens": MAX_TOKENS},
            "auth": "hf", "gpu_minutes_left_this_month": round(max(0, await budget_left()) / 60),
        }

    @api.post("/sample")
    async def sample_endpoint(req: Request):
        user = await visitor(req)
        body = await body_of(req)
        prompt = str(body.get("prompt", "")).strip()
        if not prompt or len(prompt) > 2000:
            raise Fail(400, "bad_request", "The prompt must have 1 to 2000 characters.")
        try:
            n, t_prime, max_new = int(body.get("n", 30)), float(body.get("t_prime", 1.0)), int(body.get("max_new_tokens", 128))
        except (TypeError, ValueError):
            raise Fail(400, "bad_request", "n, t_prime and max_new_tokens must be numbers.")
        if not (2 <= n <= MAX_N and 8 <= max_new <= MAX_TOKENS and 0.2 <= t_prime <= 2.0):
            raise Fail(400, "bad_request", f"Allowed: 2 ≤ n ≤ {MAX_N}, 8 ≤ max_new_tokens ≤ {MAX_TOKENS}, 0.2 ≤ t_prime ≤ 2.")
        if body.get("embedder", EMBEDDER_ID) != EMBEDDER_ID:
            raise Fail(400, "bad_request", f"This backend offers the embedder {EMBEDDER_ID!r} only.")
        day = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        key = f"runs:{day}:{user}"
        runs = await usage.get.aio(key, 0)
        if runs >= RUNS_PER_USER_PER_DAY:
            raise Fail(429, "rate_limit", f"You have used today's {RUNS_PER_USER_PER_DAY} runs. Please come back tomorrow.")
        await usage.put.aio(key, runs + 1)
        seed = int(body.get("seed", random.randrange(2**31)))
        print(f"run by {user}: n={n} T'={t_prime} max={max_new}")
        try:
            return await Sampler().sample_pool.remote.aio(prompt, n, t_prime, max_new, seed)
        except Exception as e:
            raise Fail(500, "server", f"{type(e).__name__}: {e}")

    @api.post("/embed")
    async def embed_endpoint(req: Request):
        await visitor(req)
        body = await body_of(req)
        texts = body.get("texts")
        if not isinstance(texts, list) or not 1 <= len(texts) <= MAX_N or not all(isinstance(t, str) and len(t) <= 20000 for t in texts):
            raise Fail(400, "bad_request", f"texts must be a list of 1 to {MAX_N} strings.")
        if body.get("embedder", EMBEDDER_ID) != EMBEDDER_ID:
            raise Fail(400, "bad_request", f"This backend offers the embedder {EMBEDDER_ID!r} only.")
        try:
            return await Sampler().embed.remote.aio(texts)
        except Exception as e:
            raise Fail(500, "server", f"{type(e).__name__}: {e}")

    return api


@app.local_entrypoint()
def smoke():
    """One small GPU run, for checking a deploy: modal run modal/app.py::smoke"""
    t = time.time()
    pool = Sampler().sample_pool.remote("Name a color in one word.", 6, 1.0, 16, 0)
    import base64, struct
    n = pool["N"]
    G = struct.unpack(f"<{n * n}f", base64.b64decode(pool["G"]))   # no numpy on the runner
    diag = [G[i * n + i] for i in range(n)]
    asym = max(abs(G[i * n + j] - G[j * n + i]) for i in range(n) for j in range(n))
    assert n == 6 and all(d == 1.0 for d in diag) and asym == 0.0, (n, diag, asym)
    E = Sampler().embed.remote([s["text"] for s in pool["samples"]])
    G2 = struct.unpack(f"<{n * n}f", base64.b64decode(E["G"]))
    same = max(abs(a - b) for a, b in zip(G, G2))
    print(f"{n} samples, D={pool['D']}, G {n}x{n} symmetric with unit diagonal; embed() agrees to {same:.1e}; "
          f"timing {pool['timing']}; {time.time() - t:.0f}s including start-up")
    assert same < 1e-5
    for s in pool["samples"]:
        print(f"  logp={s['logp']:8.3f} logq={s['logq']:8.3f} n={s['nTokens']} ended={s['ended']} {s['text']!r}")


@app.local_entrypoint()
def ensemble_cache(only: str = "", n: int = 200, t_prime: float = 1.0, max_new_tokens: int = 32, seed: int = 0):
    """Qwen3 14B pools for the ensemble prototype: modal run modal/app.py::ensemble_cache [--only id,id]
    Writes data/ensemble/<id>.json for each preset in data/ensemble/presets.json."""
    import json
    root = pathlib.Path(__file__).parent.parent
    presets = json.load(open(root / "data/ensemble/presets.json"))
    want = {x for x in only.split(",") if x}
    for p in presets:
        if want and p["id"] not in want:
            continue
        rec = Sampler().ensemble_pool.remote(p, n, t_prime, max_new_tokens, seed)
        rec["standIn"] = False
        path = root / "data/ensemble" / f"{p['id']}.json"
        path.write_text(json.dumps(rec, ensure_ascii=False, separators=(",", ":")))
        ended = sum(s["ended"] for s in rec["samples"])
        print(f"{p['id']}: {rec['N']} samples ({ended} ended), check {rec['checkMaxAbsErr']:.2e} nats, {rec['timing']}")
        # A bug shows up as whole nats; fp16 on the GPU gives ~0.1 between incremental and one-pass scoring.
        assert rec["checkMaxAbsErr"] < 0.5, "sampling sums disagree with teacher-forced scores"


@app.function(image=gpu_image, volumes={"/cache": cache}, timeout=1800, memory=65536)
def check_unpacking():
    """CPU check: unpacked 4-bit weights against the original bf16 weights (relative error)."""
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import glob, json, os
    def tensors(repo, names):
        p, out = snapshot_download(repo, allow_patterns=["*.json"]), {}
        idx = json.load(open(os.path.join(p, "model.safetensors.index.json")))["weight_map"]
        for n in names:
            f = snapshot_download(repo, allow_patterns=[idx[n]])
            with safe_open(os.path.join(f, idx[n]), framework="pt") as st:
                out[n] = st.get_tensor(n)
        return out
    for layer in (0, 20, 39):
        for mod in ("self_attn.q_proj", "mlp.down_proj"):
            b = f"model.layers.{layer}.{mod}"
            a = tensors(MODEL_REPO, [b + ".qweight", b + ".qzeros", b + ".scales"])
            w = unpack_awq(a[b + ".qweight"], a[b + ".qzeros"], a[b + ".scales"].float(), 128)
            ref = tensors("Qwen/Qwen3-14B", [b + ".weight"])[b + ".weight"].float()
            rel = ((w - ref).norm() / ref.norm()).item()
            cos = torch.nn.functional.cosine_similarity(w.flatten(), ref.flatten(), dim=0).item()
            print(f"{b}: shape {tuple(w.shape)} vs {tuple(ref.shape)}, relative error {rel:.4f}, cosine {cos:.5f}")


@app.local_entrypoint()
def verify(pool_file: str = "data/cache/story_T1.0.json", k: int = 30):
    """Recompute log p of the demo server's precomputed completions (sampled with the same 4-bit
    weights in vLLM) and compare with the values it recorded."""
    import json
    check_unpacking.remote()
    rec = json.load(open(pool_file))
    S = [s for s in rec["samples"] if s["ended"]][:k]
    got = Sampler().score.remote(rec["prompt"], [s["text"] for s in S])
    d = [g - s["logp"] for g, s in zip(got, S)]
    for g, s in list(zip(got, S))[:8]:
        print(f"  server {s['logp']:9.4f}  here {g:9.4f}  {s['text'][:60]!r}")
    import statistics
    print(f"{len(d)} completions: mean diff {statistics.mean(d):+.4f}, max |diff| {max(map(abs, d)):.4f} nats")
