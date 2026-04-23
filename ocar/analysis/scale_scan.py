"""E3.2: Zero-shot scale scan of obs surprise signals across Qwen model family.

Loads each model, scores a fixed set of ALFWorld trajectories with:
  - obs_nll:  P(obs_t | full history)              — the "grounding" signal
  - wm_A:     P(obs_{t+1} | obs_t, action_t)       — state+action world model
  - wm_B:     P(obs_{t+1} | action_t)              — action-only world model
  - wm_gap = wm_B - wm_A  (context-info gain)
  - action_nll: P(action_t | history)              — policy NLL

Since we use shared trajectories, NLL values are directly comparable across models.

Usage:
  python ocar/analysis/scale_scan.py --model /local_nvme/rs/models/Qwen2.5-0.5B-Instruct --gpu 0
"""
import argparse
import json
import os
from pathlib import Path

import torch

OCAR_DIR = Path(__file__).parent.parent
TRAJ_DEFAULT = str(OCAR_DIR.parent / "checkpoints/ocar_alfworld_20260414_140127/ocar_tau1.0_dstrue/global_step_25/ocar_trajectories.json")
OUT_DIR = OCAR_DIR / "analysis_results" / "scale_scan"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--traj_path", default=TRAJ_DEFAULT)
    ap.add_argument("--n_traj", type=int, default=12)
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda:0"
    dtype = getattr(torch, args.dtype)

    # Local imports after CUDA_VISIBLE_DEVICES is set
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    sys.path.insert(0, str(OCAR_DIR))
    from track_surprise_across_steps import compute_signals_for_trajectory

    slug = args.model.replace("/", "_")
    out_path = OUT_DIR / f"{slug}.json"

    # Deterministic traj selection: half success + half failure
    import random
    random.seed(args.seed)
    with open(args.traj_path) as f:
        data = json.load(f)
    trajs = data["trajectories"]
    success = [t for t in trajs if t.get("success")]
    failure = [t for t in trajs if not t.get("success")]
    random.shuffle(success); random.shuffle(failure)
    n_each = args.n_traj // 2
    selected = (success[:n_each] + failure[:args.n_traj - n_each])[:args.n_traj]
    print(f"[scale_scan] {args.model} gpu={args.gpu}")
    print(f"[scale_scan] {len(selected)} trajs ({sum(t['success'] for t in selected)} success)")

    print(f"[scale_scan] loading model in {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, trust_remote_code=True
    ).to(device).eval()

    # Aggregate signals
    agg = {"obs_nll_last": [], "wm_A": [], "wm_B": [], "action_nll": []}
    per_traj = []
    for i, traj in enumerate(selected):
        r = compute_signals_for_trajectory(model, tokenizer, traj["steps"], args.max_steps, device)
        for k in agg:
            agg[k].extend(r[k])
        per_traj.append({
            "idx": i,
            "success": bool(traj.get("success")),
            **{k: [float(x) for x in r[k]] for k in agg},
        })
        if (i + 1) % 3 == 0:
            print(f"  [{i+1}/{len(selected)}] done")

    import numpy as np
    def stats(vals):
        a = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if len(a) == 0:
            return None
        return {"mean": float(a.mean()), "std": float(a.std()), "n": int(len(a))}

    summary = {k: stats(v) for k, v in agg.items()}
    # wm_gap = wm_B - wm_A per-step
    wmA = np.asarray(agg["wm_A"]); wmB = np.asarray(agg["wm_B"])
    summary["wm_gap"] = stats(list(wmB - wmA))

    # Success/failure breakdown for obs_nll
    succ_obs, fail_obs = [], []
    for t in per_traj:
        (succ_obs if t["success"] else fail_obs).extend(t["obs_nll_last"])
    summary["obs_nll_success"] = stats(succ_obs)
    summary["obs_nll_failure"] = stats(fail_obs)

    out = {
        "model": args.model,
        "n_traj": len(selected),
        "n_success": sum(t["success"] for t in per_traj),
        "max_steps": args.max_steps,
        "dtype": args.dtype,
        "summary": summary,
        "per_traj": per_traj,
    }
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[scale_scan] wrote {out_path}")
    print("[scale_scan] summary:")
    for k, v in summary.items():
        if v:
            print(f"  {k:18s}  mean={v['mean']:.4f}  std={v['std']:.4f}  n={v['n']}")


if __name__ == "__main__":
    main()
