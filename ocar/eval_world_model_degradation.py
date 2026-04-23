#!/usr/bin/env python3
"""
World Model Degradation Evaluation

Evaluates observation NLL across training checkpoints to test whether
RL training degrades the model's ability to predict environment observations.

Usage:
    # Step 1: Merge FSDP checkpoints (run once)
    python3 ocar/eval_world_model_degradation.py merge

    # Step 2: Evaluate obs NLL across checkpoints
    python3 ocar/eval_world_model_degradation.py eval

    # Or do both:
    python3 ocar/eval_world_model_degradation.py all
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ── Configuration ──
BASE_MODEL = "/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct"
CKPT_ROOT = "/local_nvme/guanyiming/project/verl-agent/checkpoints/ocar_alfworld/ocar_tau1.0_dstrue"
MERGED_ROOT = "/local_nvme/guanyiming/project/verl-agent/checkpoints/merged_hf"
TRAJ_FILE = "/local_nvme/guanyiming/project/verl-agent/checkpoints/ocar_alfworld_20260414_094253/ocar_tau1.0_dstrue/global_step_1/ocar_trajectories.json"
STEPS = [50, 75, 100, 125, 150]


def merge_checkpoints():
    """Merge FSDP sharded checkpoints to HF format."""
    os.makedirs(MERGED_ROOT, exist_ok=True)

    for step in STEPS:
        actor_dir = f"{CKPT_ROOT}/global_step_{step}/actor"
        target_dir = f"{MERGED_ROOT}/step_{step}"

        if os.path.exists(f"{target_dir}/config.json"):
            print(f"[step {step}] Already merged, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"[step {step}] Merging FSDP checkpoint → {target_dir}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "scripts/model_merger.py", "merge",
            "--backend", "fsdp",
            "--local_dir", actor_dir,
            "--target_dir", target_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[step {step}] FAILED:\n{result.stderr}")
        else:
            print(f"[step {step}] Done.")


def load_trajectories(max_trajs=20):
    """Load fixed trajectories for evaluation."""
    with open(TRAJ_FILE) as f:
        data = json.load(f)

    trajs = data["trajectories"]
    # Mix of success and failure for balanced evaluation
    success = [t for t in trajs if t["success"]]
    failure = [t for t in trajs if not t["success"]]

    selected = success[:min(10, len(success))] + failure[:min(10, len(failure))]
    print(f"Loaded {len(selected)} trajectories ({sum(1 for t in selected if t['success'])} success, "
          f"{sum(1 for t in selected if not t['success'])} failure)")
    return selected


def build_conversation_tokens(traj, tokenizer):
    """Build the full conversation token sequence and identify observation token positions.

    Returns:
        input_ids: tensor of shape (seq_len,)
        obs_mask: tensor of shape (seq_len,) with 1 at observation token positions
    """
    # Build chat messages from trajectory steps
    messages = []
    messages.append({
        "role": "system",
        "content": "You are a helpful assistant in a text-based game."
    })

    for step_data in traj["steps"]:
        obs = step_data["observation"]
        action = step_data["action"]

        # Observation = user turn (environment response)
        messages.append({"role": "user", "content": obs})
        # Action = assistant turn (model response)
        messages.append({"role": "assistant", "content": action})

    # Tokenize the full conversation
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Now identify observation token positions
    # Strategy: tokenize incrementally, mark obs regions
    obs_mask = torch.zeros(len(full_ids), dtype=torch.bool)

    # Build prefixes to find obs boundaries
    prefix_len = 0
    for i, step_data in enumerate(traj["steps"]):
        obs = step_data["observation"]

        # Build messages up to just before this observation
        msgs_before_obs = []
        msgs_before_obs.append({"role": "system", "content": "You are a helpful assistant in a text-based game."})
        for j in range(i):
            msgs_before_obs.append({"role": "user", "content": traj["steps"][j]["observation"]})
            msgs_before_obs.append({"role": "assistant", "content": traj["steps"][j]["action"]})

        # Build messages up to and including this observation
        msgs_with_obs = msgs_before_obs.copy()
        msgs_with_obs.append({"role": "user", "content": obs})

        text_before = tokenizer.apply_chat_template(msgs_before_obs, tokenize=False, add_generation_prompt=False)
        text_with = tokenizer.apply_chat_template(msgs_with_obs, tokenize=False, add_generation_prompt=True)

        ids_before = tokenizer.encode(text_before, add_special_tokens=False)
        ids_with = tokenizer.encode(text_with, add_special_tokens=False)

        # The observation tokens are between len(ids_before) and len(ids_with)
        # But we want only the content tokens, not the role markers
        # Simple approach: mark the range, exclude first few role-marker tokens

        obs_start = len(ids_before)
        obs_end = len(ids_with)

        # Skip role markers (e.g., <|im_start|>user\n) - typically ~3-4 tokens
        # Find where the actual obs content starts by tokenizing just the role prefix
        role_prefix = "<|im_start|>user\n"
        role_tokens = tokenizer.encode(role_prefix, add_special_tokens=False)
        content_start = obs_start + len(role_tokens)

        # Mark observation content tokens (exclude the <|im_end|> and generation prompt at the end)
        gen_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        gen_tokens = tokenizer.encode(gen_suffix, add_special_tokens=False)
        content_end = obs_end - len(gen_tokens)

        if content_start < content_end and content_end <= len(full_ids):
            obs_mask[content_start:content_end] = True

    return torch.tensor(full_ids), obs_mask


def compute_obs_nll(model, tokenizer, trajectories, device, max_length=2048):
    """Compute mean observation NLL for a set of trajectories."""
    model.eval()
    all_obs_nlls = []
    per_traj_nlls = []

    with torch.no_grad():
        for i, traj in enumerate(trajectories):
            input_ids, obs_mask = build_conversation_tokens(traj, tokenizer)

            # Truncate to max_length
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                obs_mask = obs_mask[:max_length]

            n_obs_tokens = obs_mask.sum().item()
            if n_obs_tokens < 5:
                continue

            input_ids = input_ids.unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

            # Shift for autoregressive: predict token t from position t-1
            shift_logits = logits[0, :-1, :]  # (seq_len-1, vocab)
            shift_labels = input_ids[0, 1:]    # (seq_len-1,)
            shift_obs_mask = obs_mask[1:]       # (seq_len-1,)

            # Compute per-token NLL
            log_probs = torch.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            token_nll = -token_log_probs

            # Average NLL over observation tokens
            obs_nll = token_nll[shift_obs_mask].mean().item()
            all_obs_nlls.append(obs_nll)
            per_traj_nlls.append({
                "traj_id": traj["traj_id"][:8],
                "success": traj["success"],
                "n_obs_tokens": n_obs_tokens,
                "obs_nll": obs_nll,
            })

            if (i + 1) % 5 == 0:
                print(f"  [{i+1}/{len(trajectories)}] running mean obs NLL = {np.mean(all_obs_nlls):.4f}")

    mean_nll = np.mean(all_obs_nlls) if all_obs_nlls else float("nan")
    std_nll = np.std(all_obs_nlls) if all_obs_nlls else float("nan")

    # Split by success/failure
    success_nlls = [t["obs_nll"] for t in per_traj_nlls if t["success"]]
    failure_nlls = [t["obs_nll"] for t in per_traj_nlls if not t["success"]]

    return {
        "mean_nll": mean_nll,
        "std_nll": std_nll,
        "n_trajs": len(all_obs_nlls),
        "success_nll": np.mean(success_nlls) if success_nlls else float("nan"),
        "failure_nll": np.mean(failure_nlls) if failure_nlls else float("nan"),
        "per_traj": per_traj_nlls,
    }


def run_evaluation():
    """Evaluate obs NLL across base model + all checkpoints."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    trajectories = load_trajectories(max_trajs=20)

    # Models to evaluate: base + checkpoints
    model_paths = [("base (step 0)", BASE_MODEL)]
    for step in STEPS:
        merged_path = f"{MERGED_ROOT}/step_{step}"
        if os.path.exists(f"{merged_path}/config.json"):
            model_paths.append((f"step {step}", merged_path))
        else:
            print(f"WARNING: Merged checkpoint not found for step {step} at {merged_path}")

    results = []

    for name, path in model_paths:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"  Path: {path}")
        print(f"{'='*60}")

        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

        result = compute_obs_nll(model, tokenizer, trajectories, device)
        result["checkpoint"] = name
        results.append(result)

        print(f"\n  >> {name}: obs NLL = {result['mean_nll']:.4f} ± {result['std_nll']:.4f}")
        print(f"     success NLL = {result['success_nll']:.4f}, failure NLL = {result['failure_nll']:.4f}")

        del model
        torch.cuda.empty_cache()

    # ── Print summary ──
    print(f"\n{'='*60}")
    print("WORLD MODEL DEGRADATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'Checkpoint':<20} {'Obs NLL':>10} {'± std':>8} {'Succ NLL':>10} {'Fail NLL':>10} {'n':>5}")
    print("-" * 65)
    for r in results:
        print(f"{r['checkpoint']:<20} {r['mean_nll']:>10.4f} {r['std_nll']:>8.4f} "
              f"{r['success_nll']:>10.4f} {r['failure_nll']:>10.4f} {r['n_trajs']:>5}")

    # ── Trend analysis ──
    nlls = [r["mean_nll"] for r in results]
    if len(nlls) >= 3:
        trend = nlls[-1] - nlls[0]
        print(f"\nTrend (last - first): {trend:+.4f}")
        if trend > 0.05:
            print(">>> WORLD MODEL DEGRADATION DETECTED: obs NLL increases over training")
            print(">>> This supports the Dual-Token Training motivation (paper_v6)")
        elif trend < -0.05:
            print(">>> World model IMPROVES during training (unexpected)")
        else:
            print(">>> No significant degradation detected (|trend| < 0.05)")

    # Save results
    output_path = "/local_nvme/guanyiming/project/verl-agent/ocar/wm_degradation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="World Model Degradation Evaluation")
    parser.add_argument("mode", choices=["merge", "eval", "all"],
                        help="merge: convert FSDP checkpoints; eval: compute obs NLL; all: both")
    args = parser.parse_args()

    if args.mode in ("merge", "all"):
        merge_checkpoints()

    if args.mode in ("eval", "all"):
        run_evaluation()


if __name__ == "__main__":
    main()
