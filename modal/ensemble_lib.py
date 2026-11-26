"""Pools for the ensemble ("prompt intersection") prototype: plain PyTorch, no Modal.

Used by modal/app.py on the GPU (Qwen3 14B) and by scripts/ensemble_pool.py on a CPU (stand-in models).
See docs/handoff-ensemble-demo.md (A3, A7) for what the page does with these numbers.

For prompts A and B (and an optional explicit-intersection prompt C), draw N completions i.i.d. from the
token-level product of A and B at temperature T':

    r(y_t | y_<t) ∝ (p_A(y_t | A, y_<t) · p_B(y_t | B, y_<t))^(1/T')

and record, for each completion y (the same token ids under every prompt):

    logpA, logpB, logpC   Σ_t log p_X(y_t | X, y_<t): string-level scores (the global ensemble's inputs)
    logr                  Σ_t log r(y_t | y_<t): the proposal, for importance weights
    loglocal              Σ_t log normalise(p_A · p_B)(y_t | ...): the local (token-level) product ensemble
                          (equal to logr at T' = 1)

A completion ends at the first token containing the stop string (default "."), at an end-of-text token,
or at max_new_tokens (then "ended" is false). The sums include the token that ended it. The prompts are
plain text continued by the model (no chat template), as in the paper's Fig. 1, unless chat=True.
"""


def prompt_ids(tok, prompt, chat=False, chat_kw=None):
    if chat:
        prompt = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                         add_generation_prompt=True, **(chat_kw or {}))
    return tok(prompt, add_special_tokens=False).input_ids


def _eos_ids(tok, model):
    ids = model.generation_config.eos_token_id
    ids = ids if isinstance(ids, (list, tuple)) else [ids]
    ids = [i for i in ids if i is not None] + ([tok.eos_token_id] if tok.eos_token_id is not None else [])
    return sorted(set(int(i) for i in ids))


def _stop_ids(tok, stop):
    """Token ids whose text contains the stop string (e.g. ".", ".\\n", "!.")."""
    if not stop:
        return []
    return [i for i in range(len(tok)) if stop in tok.decode([i])]


def ensemble_sample(tok, model, A, B, n, t_prime=1.0, max_new_tokens=32, seed=0, batch_size=100,
                    stop=".", chat=False, chat_kw=None):
    """N completions i.i.d. from the token-level product of prompts A and B (see the module docstring)."""
    import inspect
    import torch
    dev = next(model.parameters()).device
    ia = torch.tensor([prompt_ids(tok, A, chat, chat_kw)], device=dev)
    ib = torch.tensor([prompt_ids(tok, B, chat, chat_kw)], device=dev)
    eos = _eos_ids(tok, model)
    end_t = torch.tensor(sorted(set(eos) | set(_stop_ids(tok, stop))), device=dev)
    eos_t = torch.tensor(eos, device=dev)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    cache_kw = "cache" if "cache" in inspect.signature(type(model).forward).parameters else "past_key_values"
    out = []
    with torch.no_grad():
        while len(out) < n:
            Bn = min(batch_size, n - len(out))
            acc = {k: torch.zeros(Bn, dtype=torch.float64, device=dev) for k in ("A", "B", "r", "local")}
            done = torch.zeros(Bn, dtype=torch.bool, device=dev)
            ntok = torch.zeros(Bn, dtype=torch.long, device=dev)
            steps, pa, pb = [], None, None
            sa, sb = ia.repeat(Bn, 1), ib.repeat(Bn, 1)
            for _ in range(max_new_tokens):
                ra = model(input_ids=sa, use_cache=True, **{cache_kw: pa})
                rb = model(input_ids=sb, use_cache=True, **{cache_kw: pb})
                pa = getattr(ra, "cache", None) or ra.past_key_values
                pb = getattr(rb, "cache", None) or rb.past_key_values
                la = torch.log_softmax(ra.logits[:, -1, :].float(), -1)
                lb = torch.log_softmax(rb.logits[:, -1, :].float(), -1)
                local = torch.log_softmax(la + lb, -1)                  # normalised token-level product
                lr = torch.log_softmax((la + lb) / t_prime, -1)         # the proposal at T'
                nxt = torch.multinomial(lr.exp(), 1, generator=gen).squeeze(1)
                active = ~done
                nxt = torch.where(active, nxt, eos_t[0])
                for k, l in (("A", la), ("B", lb), ("r", lr), ("local", local)):
                    acc[k] += torch.where(active, l.gather(1, nxt[:, None]).squeeze(1).double(), 0.0)
                ntok += active.long()
                steps.append(nxt)
                done |= active & torch.isin(nxt, end_t)
                if bool(done.all()):
                    break
                sa = sb = nxt[:, None]
            toks = torch.stack(steps, 1).cpu().tolist()
            ntok, done = ntok.cpu().tolist(), done.cpu().tolist()
            acc = {k: v.cpu().tolist() for k, v in acc.items()}
            for b in range(Bn):
                seq = toks[b][: ntok[b]]
                out.append({
                    "ids": seq,
                    "text": tok.decode([t for t in seq if t not in eos], skip_special_tokens=True),
                    "logpA": acc["A"][b], "logpB": acc["B"][b], "logr": acc["r"][b], "loglocal": acc["local"][b],
                    "nTokens": ntok[b], "ended": bool(done[b]),
                    "end": "eos" if seq and seq[-1] in eos else ("stop" if done[b] else "max"),
                })
    return out


def score_ids(tok, model, prompt, seqs, batch_size=50, chat=False, chat_kw=None):
    """Σ_t log p(y_t | prompt, y_<t) for each token-id sequence (teacher forcing, right padding)."""
    import torch
    dev = next(model.parameters()).device
    h = prompt_ids(tok, prompt, chat, chat_kw)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    out = []
    with torch.no_grad():
        for s in range(0, len(seqs), batch_size):
            chunk = seqs[s:s + batch_size]
            L = len(h) + max(len(y) for y in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long, device=dev)
            mask = torch.zeros((len(chunk), L), dtype=torch.long, device=dev)
            for b, y in enumerate(chunk):
                ids[b, : len(h) + len(y)] = torch.tensor(h + y, device=dev)
                mask[b, : len(h) + len(y)] = 1
            lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), -1)
            for b, y in enumerate(chunk):
                if not y:
                    out.append(0.0)
                    continue
                pos = torch.arange(len(h) - 1, len(h) - 1 + len(y), device=dev)
                tgt = torch.tensor(y, device=dev)
                out.append(lp[b, pos, tgt].double().sum().item())
    return out


def ensemble_pool(tok, model, preset, n=200, t_prime=1.0, max_new_tokens=32, seed=0, stop=".",
                  chat=False, chat_kw=None, model_name="", check=True):
    """One preset's pool: samples from the A×B product, scored under A, B and (if given) C."""
    import time
    t0 = time.time()
    S = ensemble_sample(tok, model, preset["A"], preset["B"], n, t_prime, max_new_tokens, seed,
                        stop=stop, chat=chat, chat_kw=chat_kw)
    t1 = time.time()
    seqs = [s["ids"] for s in S]
    if preset.get("C"):
        for s, v in zip(S, score_ids(tok, model, preset["C"], seqs, chat=chat, chat_kw=chat_kw)):
            s["logpC"] = v
    check_err = None
    if check:
        # The sums accumulated while sampling must match teacher-forced scores of the same tokens.
        ra = score_ids(tok, model, preset["A"], seqs, chat=chat, chat_kw=chat_kw)
        rb = score_ids(tok, model, preset["B"], seqs, chat=chat, chat_kw=chat_kw)
        check_err = max(max(abs(a - s["logpA"]), abs(b - s["logpB"])) for a, b, s in zip(ra, rb, S))
    for s in S:
        del s["ids"]
    return {
        "schema": 1, "kind": "ensemble-pool", "id": preset["id"], "label": preset.get("label", ""),
        "prompts": {k: preset[k] for k in ("A", "B", "C") if preset.get(k)},
        "model": model_name, "chat": chat, "proposal": "local_product", "Tprime": t_prime,
        "maxNewTokens": max_new_tokens, "stop": stop, "N": len(S), "seed": seed,
        "checkMaxAbsErr": check_err,
        "timing": {"sample_s": round(t1 - t0, 2), "score_s": round(time.time() - t1, 2)},
        "samples": S,
    }
