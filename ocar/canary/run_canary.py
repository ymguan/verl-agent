"""Canary probe runner: compute per-token NLL of ALFWorld probes under a given LM.

Usage:
  python run_canary.py --model Qwen/Qwen2.5-1.5B-Instruct --gpu 0
  python run_canary.py --model Qwen/Qwen2.5-7B-Instruct  --gpu 1
  python run_canary.py --model Qwen/Qwen3-8B             --gpu 2

Output: ocar/canary/results/<model_slug>.json with per-probe NLL and per-split aggregates.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROBES = Path(__file__).parent / "probes.jsonl"
OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def compute_nll(model, tokenizer, text: str, device):
    """Per-token NLL (nats/token) on full sentence."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    if input_ids.shape[1] < 2:
        return float("nan"), 0
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    n_tokens = input_ids.shape[1] - 1  # labels shift
    return out.loss.item(), n_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    dtype = getattr(torch, args.dtype)

    slug = args.model.replace("/", "_")
    out_path = OUT_DIR / f"{slug}.json"

    print(f"[canary] loading {args.model} on gpu={args.gpu} dtype={dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, trust_remote_code=True,
    ).to(device).eval()

    probes = [json.loads(ln) for ln in PROBES.read_text().splitlines() if ln.strip()]
    print(f"[canary] {len(probes)} probes loaded")

    results = []
    for i, p in enumerate(probes):
        nll, n_tok = compute_nll(model, tokenizer, p["text"], device)
        results.append({**p, "nll": nll, "n_tok": n_tok})
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(probes)}] split={p['split']} nll={nll:.3f} n_tok={n_tok}")

    # Aggregate per split (token-weighted mean NLL)
    agg = {}
    for split in ["original", "shuffled", "swapped", "nonsense", "generic"]:
        rs = [r for r in results if r["split"] == split]
        if not rs:
            continue
        total_nll = sum(r["nll"] * r["n_tok"] for r in rs)
        total_tok = sum(r["n_tok"] for r in rs)
        mean_nll = total_nll / total_tok if total_tok else float("nan")
        mean_nll_per_sample = sum(r["nll"] for r in rs) / len(rs)
        agg[split] = {
            "n_samples": len(rs),
            "mean_nll_token_weighted": mean_nll,
            "mean_nll_per_sample": mean_nll_per_sample,
            "total_tokens": total_tok,
        }

    out = {"model": args.model, "per_probe": results, "aggregate": agg}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[canary] wrote {out_path}")
    print("[canary] aggregate:")
    for split, a in agg.items():
        print(f"  {split:9s}  mean_nll={a['mean_nll_token_weighted']:.4f} (n={a['n_samples']}, toks={a['total_tokens']})")


if __name__ == "__main__":
    main()
