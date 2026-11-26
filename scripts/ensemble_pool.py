"""Make cached pools for the ensemble prototype on this machine (CPU or local GPU).

    python scripts/ensemble_pool.py --model Qwen/Qwen3-0.6B --n 200 --only physicist_author

Writes data/ensemble/<preset id>.json (see modal/ensemble_lib.py for the record). Pools from a model other
than the demo's Qwen3 14B are stand-ins for development: the file records the model, and the page says so.
The real pools come from Modal (modal run modal/app.py::ensemble_cache).
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "modal"))
from ensemble_lib import ensemble_pool  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--presets", default=os.path.join(ROOT, "data/ensemble/presets.json"))
    ap.add_argument("--only", nargs="*", help="preset ids (default: all)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--tprime", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stop", default=".")
    ap.add_argument("--out", default=os.path.join(ROOT, "data/ensemble"))
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev).eval()
    presets = json.load(open(args.presets))
    for p in presets:
        if args.only and p["id"] not in args.only:
            continue
        rec = ensemble_pool(tok, model, p, n=args.n, t_prime=args.tprime, max_new_tokens=args.max_new_tokens,
                            seed=args.seed, stop=args.stop, model_name=args.model)
        rec["standIn"] = args.model != "Qwen/Qwen3-14B-AWQ"
        path = os.path.join(args.out, f"{p['id']}.json")
        with open(path, "w") as f:
            json.dump(rec, f, ensure_ascii=False, separators=(",", ":"))
        ended = sum(s["ended"] for s in rec["samples"])
        print(f"{p['id']}: {rec['N']} samples ({ended} ended), check {rec['checkMaxAbsErr']:.2e} nats, "
              f"{rec['timing']} -> {os.path.relpath(path, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
