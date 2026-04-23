#!/usr/bin/env python3
"""
Batch merge FSDP checkpoints -> HF format and upload to HF Hub with per-step revisions.

Layout:
  merged_models/<repo_name>/step_<N>/    (local HF-format dirs)

HF Hub layout (per repo):
  main branch:  step_150 contents (latest)
  step_<N> branches: one per training step

Usage:
  python ocar/merge_and_upload.py --run grpo_1_5b
  python ocar/merge_and_upload.py --run gigpo_1_5b
  python ocar/merge_and_upload.py --run grpo_7b
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/local_nvme/guanyiming/project/verl-agent")
MERGED = ROOT / "merged_models"

RUNS = {
    "grpo_1_5b": {
        "fsdp_root": ROOT / "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hf_repo": "Ricardo-H/ocar-grpo-observe-alfworld-1.5b",
        "steps": [20, 40, 60, 80, 100, 120, 140, 150],
    },
    "gigpo_1_5b": {
        "fsdp_root": ROOT / "checkpoints/gigpo_observe_alfworld_1.5b_20260420_162642/gigpo_observe_qwen2.5_1.5b_seed0",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "hf_repo": "Ricardo-H/ocar-gigpo-observe-alfworld-1.5b",
        "steps": [20, 40, 60, 80, 100, 120, 140, 150],
    },
    "grpo_7b": {
        "fsdp_root": ROOT / "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "hf_repo": "Ricardo-H/ocar-grpo-observe-alfworld-7b",
        "steps": [150],
    },
}


def merge_step(run_cfg, step: int, out_root: Path):
    ckpt_dir = run_cfg["fsdp_root"] / f"global_step_{step}" / "actor"
    out_dir = out_root / f"step_{step}"
    if (out_dir / "model.safetensors.index.json").exists() or (out_dir / "model.safetensors").exists():
        print(f"[skip] {out_dir} already merged")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "ocar/merge_fsdp_to_hf.py"),
        "--ckpt_dir", str(ckpt_dir),
        "--model_name", run_cfg["base_model"],
        "--output_dir", str(out_dir),
    ]
    print(f"[merge] step={step} -> {out_dir}")
    subprocess.run(cmd, check=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUNS.keys()))
    parser.add_argument("--steps", default=None, help="comma-separated override (e.g. 20,40)")
    args = parser.parse_args()

    cfg = RUNS[args.run]
    steps = [int(s) for s in args.steps.split(",")] if args.steps else cfg["steps"]
    out_root = MERGED / cfg["hf_repo"].split("/")[-1]

    for s in steps:
        merge_step(cfg, s, out_root)
    print(f"\n[done] merged runs for {args.run} -> {out_root}")


if __name__ == "__main__":
    main()
