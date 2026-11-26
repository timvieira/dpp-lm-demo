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

A completion ends at the end of its first sentence (see sentence_period below: abbreviations, initials and
decimals do not end it), at an end-of-text token, or at max_new_tokens (then "ended" is false). A sentence end
contributes the probability that the next token starts with whitespace or ends the text. The prompts are
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


# A completion ends at the end of its first sentence. A period ends a sentence when the next token starts with
# whitespace, or is the end of text, and the period does not belong to an abbreviation or an initial:
# "Dr. Richard Feynman." ends at "Feynman.", "J. Robert Oppenheimer." at "Oppenheimer.", and "0.5" never at "0.".
# Ending is then the event "the next token starts with whitespace or ends the text"; its probability (summed over
# all such tokens) is part of every score, so each score stays the exact probability of the string ending there.
ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "prof", "st", "jr", "sr", "mt", "vs", "gen", "sen", "rev", "fr", "lt",
                 "col", "capt", "sgt", "e.g", "i.e", "u.s", "u.k", "a.k.a"}


def sentence_period(text):
    """Whether text ends with a period that can end a sentence (not an abbreviation or an initial)."""
    import re
    if not text.endswith("."):
        return False
    m = re.search(r"(\S+)\.$", text)
    w = (m.group(1) if m else "").lstrip("\"'([“‘").lower()
    if w in ABBREVIATIONS:
        return False
    if len(w) == 1 and w.isalpha():          # an initial: "J."
        return False
    if re.fullmatch(r"[a-z](\.[a-z])+", w):   # dotted initials: "J.R.R", "U.S"
        return False
    return True


def _boundary_mask(tok, model, eos, dev):
    """Tokens that can follow a sentence-ending period: those starting with whitespace, and end-of-text.
    Sized to the model's output layer, which can be padded beyond the tokenizer's vocabulary."""
    import torch
    m = torch.zeros(model.get_output_embeddings().weight.shape[0], dtype=torch.bool)
    for i in range(len(tok)):
        t = tok.decode([i])
        if t[:1].isspace():
            m[i] = True
    m[list(eos)] = True
    return m.to(dev)


def ensemble_sample(tok, model, A, B, n, t_prime=1.0, max_new_tokens=32, seed=0, batch_size=100,
                    chat=False, chat_kw=None):
    """N completions i.i.d. from the token-level product of prompts A and B (see the module docstring)."""
    import inspect
    import torch
    dev = next(model.parameters()).device
    ia = torch.tensor([prompt_ids(tok, A, chat, chat_kw)], device=dev)
    ib = torch.tensor([prompt_ids(tok, B, chat, chat_kw)], device=dev)
    eos = _eos_ids(tok, model)
    eos_set = set(eos)
    bmask = _boundary_mask(tok, model, eos, dev)
    gen = torch.Generator(device=dev).manual_seed(int(seed))
    cache_kw = "cache" if "cache" in inspect.signature(type(model).forward).parameters else "past_key_values"
    lse = lambda l: torch.logsumexp(l.masked_fill(~bmask, float("-inf")), -1)   # log P(next token is a boundary)
    out = []
    with torch.no_grad():
        while len(out) < n:
            Bn = min(batch_size, n - len(out))
            acc = {k: torch.zeros(Bn, dtype=torch.float64) for k in ("A", "B", "r", "local")}
            done, pending = [False] * Bn, [False] * Bn
            end = ["max"] * Bn
            seqs = [[] for _ in range(Bn)]
            pa = pb = None
            sa, sb = ia.repeat(Bn, 1), ib.repeat(Bn, 1)
            for _ in range(max_new_tokens + 1):
                ra = model(input_ids=sa, use_cache=True, **{cache_kw: pa})
                rb = model(input_ids=sb, use_cache=True, **{cache_kw: pb})
                pa = getattr(ra, "cache", None) or ra.past_key_values
                pb = getattr(rb, "cache", None) or rb.past_key_values
                la = torch.log_softmax(ra.logits[:, -1, :].float(), -1)
                lb = torch.log_softmax(rb.logits[:, -1, :].float(), -1)
                local = torch.log_softmax(la + lb, -1)                  # normalised token-level product
                lr = torch.log_softmax((la + lb) / t_prime, -1)         # the proposal at T'
                nxt = torch.multinomial(lr.exp(), 1, generator=gen).squeeze(1)
                L = {"A": la, "B": lb, "r": lr, "local": local}
                tokp = {k: l.gather(1, nxt[:, None]).squeeze(1).double().cpu() for k, l in L.items()}
                bnd = {k: lse(l).double().cpu() for k, l in L.items()} if any(pending) else None
                nl, isb = nxt.cpu().tolist(), bmask[nxt].cpu().tolist()
                for b in range(Bn):
                    if done[b]:
                        continue
                    if pending[b] and isb[b]:
                        # The sentence ended: the event "a boundary token comes next", not the token itself.
                        for k in acc:
                            acc[k][b] += bnd[k][b]
                        done[b], end[b] = True, "sentence"
                        continue
                    if len(seqs[b]) >= max_new_tokens:
                        done[b] = True                                   # cut off ("max")
                        continue
                    for k in acc:
                        acc[k][b] += tokp[k][b]
                    seqs[b].append(nl[b])
                    if nl[b] in eos_set:
                        done[b], end[b] = True, "eos"
                        continue
                    pending[b] = sentence_period(tok.decode(seqs[b]))
                if all(done):
                    break
                sa = sb = nxt[:, None]
            acc = {k: v.tolist() for k, v in acc.items()}
            for b in range(Bn):
                out.append({
                    "ids": seqs[b], "boundary": end[b] == "sentence",
                    "text": tok.decode([t for t in seqs[b] if t not in eos_set], skip_special_tokens=True),
                    "logpA": acc["A"][b], "logpB": acc["B"][b], "logr": acc["r"][b], "loglocal": acc["local"][b],
                    "nTokens": len(seqs[b]), "ended": end[b] != "max", "end": end[b],
                })
    return out


def score_ids(tok, model, prompt, seqs, boundary=None, batch_size=50, chat=False, chat_kw=None):
    """Σ_t log p(y_t | prompt, y_<t) for each token-id sequence (teacher forcing, right padding), plus, where
    boundary[b] is true, log P(the next token is a boundary) at the end (the sentence ended there)."""
    import torch
    dev = next(model.parameters()).device
    h = prompt_ids(tok, prompt, chat, chat_kw)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    bmask = _boundary_mask(tok, model, _eos_ids(tok, model), dev) if boundary and any(boundary) else None
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
                v = 0.0
                if y:
                    pos = torch.arange(len(h) - 1, len(h) - 1 + len(y), device=dev)
                    v = lp[b, pos, torch.tensor(y, device=dev)].double().sum().item()
                if boundary and boundary[s + b]:
                    last = lp[b, len(h) + len(y) - 1]
                    v += torch.logsumexp(last.masked_fill(~bmask, float("-inf")), -1).item()
                out.append(v)
    return out


def ensemble_pool(tok, model, preset, n=200, t_prime=1.0, max_new_tokens=32, seed=0,
                  chat=False, chat_kw=None, model_name="", check=True):
    """One preset's pool: samples from the A×B product, scored under A, B and (if given) C."""
    import time
    t0 = time.time()
    S = ensemble_sample(tok, model, preset["A"], preset["B"], n, t_prime, max_new_tokens, seed,
                        chat=chat, chat_kw=chat_kw)
    t1 = time.time()
    seqs, bnd = [s["ids"] for s in S], [s["boundary"] for s in S]
    if preset.get("C"):
        for s, v in zip(S, score_ids(tok, model, preset["C"], seqs, bnd, chat=chat, chat_kw=chat_kw)):
            s["logpC"] = v
    check_err = None
    if check:
        # The sums accumulated while sampling must match teacher-forced scores of the same tokens.
        ra = score_ids(tok, model, preset["A"], seqs, bnd, chat=chat, chat_kw=chat_kw)
        rb = score_ids(tok, model, preset["B"], seqs, bnd, chat=chat, chat_kw=chat_kw)
        check_err = max(max(abs(a - s["logpA"]), abs(b - s["logpB"])) for a, b, s in zip(ra, rb, S))
    for s in S:
        del s["ids"], s["boundary"]
    return {
        "schema": 1, "kind": "ensemble-pool", "id": preset["id"], "label": preset.get("label", ""),
        "prompts": {k: preset[k] for k in ("A", "B", "C") if preset.get(k)},
        "model": model_name, "chat": chat, "proposal": "local_product", "Tprime": t_prime,
        "maxNewTokens": max_new_tokens, "stop": "sentence", "N": len(S), "seed": seed,
        "checkMaxAbsErr": check_err,
        "timing": {"sample_s": round(t1 - t0, 2), "score_s": round(time.time() - t1, 2)},
        "samples": S,
    }
