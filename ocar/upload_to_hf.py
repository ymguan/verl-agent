#!/usr/bin/env python3
"""
Upload merged HF-format checkpoints to HuggingFace Hub with per-step revisions (branches).

For each run:
  - Create repo if missing (public)
  - For each step, create a branch `step_<N>` and upload files there
  - Also upload `step_150` contents to `main` branch (latest pointer)
  - Generate README.md (model card) on main

Usage:
  python ocar/upload_to_hf.py --run grpo_1_5b
  python ocar/upload_to_hf.py --run gigpo_1_5b
  python ocar/upload_to_hf.py --run grpo_7b
  python ocar/upload_to_hf.py --run grpo_7b_already_merged  # for checkpoints/merged_hf
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, create_branch

ROOT = Path("/local_nvme/guanyiming/project/verl-agent")
MERGED = ROOT / "merged_models"

RUNS = {
    "grpo_1_5b": {
        "local_dir": MERGED / "ocar-grpo-observe-alfworld-1.5b",
        "hf_repo": "Ricardo-H/ocar-grpo-observe-alfworld-1.5b",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "algo": "GRPO + observe",
    },
    "gigpo_1_5b": {
        "local_dir": MERGED / "ocar-gigpo-observe-alfworld-1.5b",
        "hf_repo": "Ricardo-H/ocar-gigpo-observe-alfworld-1.5b",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "algo": "GiGPO + observe",
    },
    "grpo_7b": {
        "local_dir": MERGED / "ocar-grpo-observe-alfworld-7b",
        "hf_repo": "Ricardo-H/ocar-grpo-observe-alfworld-7b",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "algo": "GRPO + observe",
    },
    "grpo_7b_ocar_v3": {
        "local_dir": ROOT / "checkpoints/merged_hf",
        "hf_repo": "Ricardo-H/ocar-v3-alfworld-7b",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "algo": "OCAR v3 (Δs-based credit, adaptive τ)",
    },
}


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
base_model: {base_model}
tags:
- agent-rl
- alfworld
- archived
- ocar
- research-post-mortem
---

# {repo_short} — Archived Checkpoints

> ⚠️ **Research line terminated (2026-04-22).** These checkpoints are retained for
> inference / analysis reproducibility only. See the
> [post-mortem document](https://github.com/ymguan/verl-agent/blob/master/ocar/docs/POSTMORTEM_SURPRISE.md)
> for why we do not recommend building on this method.

## What this is

Fine-tuned from `{base_model}` on **ALFWorld** with **{algo}** (verl-agent stack),
as part of the OCAR (Observation-grounded Credit Advantage Redistribution)
research line investigating free policy-forward-pass signals for agent RL
credit assignment.

## Checkpoints (per-step revisions)

Each training step is stored as a separate git branch / revision. Load a
specific step via `revision=`:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{repo}", revision="step_150", torch_dtype="bfloat16"
)
tokenizer = AutoTokenizer.from_pretrained("{repo}", revision="step_150")
```

Available revisions: {revisions_str}

## Results summary

See `ocar/docs/POSTMORTEM_SURPRISE.md` in the companion repo for full results.
Key points:

- 6-seed peak SR (ALFWorld paper-config, t=0.4): around 80% — **did not match GiGPO 90.8**
- Δs signal shown to be causally circular (reads back GRPO's own updates)
- Step-level AUC ≈ 0.5 across 4 heterogeneous base scorers
- Cross-environment direction flip on WebShop (r(Δs, succ): −0.53 ↔ +0.65)

## Companion resources

- Code & analysis: <https://github.com/ymguan/verl-agent>
- Training trajectories: `data/trajectories/` in companion repo
- Analysis JSONs: `ocar/analysis_results/` in companion repo
- Post-mortem: [`ocar/docs/POSTMORTEM_SURPRISE.md`](https://github.com/ymguan/verl-agent/blob/master/ocar/docs/POSTMORTEM_SURPRISE.md)

## Citation / attribution

These artifacts are shared in an "as-is" state. If you find the negative
results useful, please reference the post-mortem document.
"""


def upload_run(run_key: str):
    cfg = RUNS[run_key]
    api = HfApi()
    repo_id = cfg["hf_repo"]
    local_dir = cfg["local_dir"]

    if not local_dir.exists():
        raise FileNotFoundError(f"Local dir missing: {local_dir}")

    # 1. create repo
    print(f"[repo] ensuring {repo_id}")
    create_repo(repo_id, private=False, exist_ok=True)

    # 2. find step dirs
    step_dirs = sorted(
        [d for d in local_dir.iterdir() if d.is_dir() and d.name.startswith("step_")],
        key=lambda d: int(d.name.split("_")[1]),
    )
    step_names = [d.name for d in step_dirs]
    print(f"[repo] {repo_id}: {len(step_dirs)} steps -> {step_names}")

    # 3. upload each step to its own revision
    for d in step_dirs:
        rev = d.name
        print(f"  [upload] {rev} ({sum(f.stat().st_size for f in d.rglob('*') if f.is_file()) / 1e9:.1f} GB)")
        # ensure branch exists
        try:
            create_branch(repo_id, branch=rev, exist_ok=True)
        except Exception as e:
            print(f"    branch create: {e}")
        api.upload_folder(
            folder_path=str(d),
            repo_id=repo_id,
            revision=rev,
            commit_message=f"Upload {rev}",
        )

    # 4. Upload latest (highest step) to main + model card
    latest = step_dirs[-1]
    print(f"[main] uploading {latest.name} to main branch as pointer")
    card_path = latest / "README.md"
    card_path.write_text(MODEL_CARD_TEMPLATE.format(
        base_model=cfg["base_model"],
        algo=cfg["algo"],
        repo=repo_id,
        repo_short=repo_id.split("/")[-1],
        revisions_str=", ".join(f"`{n}`" for n in step_names),
    ))
    api.upload_folder(
        folder_path=str(latest),
        repo_id=repo_id,
        commit_message=f"Update main to {latest.name} + model card",
    )
    # cleanup temp README so it doesn't persist in other revisions later
    card_path.unlink(missing_ok=True)
    print(f"[done] {repo_id} uploaded. https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=list(RUNS.keys()))
    args = parser.parse_args()
    upload_run(args.run)


if __name__ == "__main__":
    main()
