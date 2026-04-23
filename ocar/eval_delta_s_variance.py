#!/usr/bin/env python3
"""
Compute per-step delta_s (S_theta - S_ref) within-trajectory variance
across training checkpoints.

Answers: does delta_s variance grow fast enough to make OCAR effective
beyond the first few training steps?
"""

import json
import sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct"
MERGED_ROOT = "/local_nvme/guanyiming/project/verl-agent/checkpoints/merged_hf"
TRAJ_FILE = "/local_nvme/guanyiming/project/verl-agent/checkpoints/ocar_alfworld_20260414_094253/ocar_tau1.0_dstrue/global_step_1/ocar_trajectories.json"
STEPS = [50, 75, 100, 125, 150]


def load_trajectories(max_trajs=20):
    with open(TRAJ_FILE) as f:
        data = json.load(f)
    trajs = data["trajectories"]
    success = [t for t in trajs if t["success"]][:10]
    failure = [t for t in trajs if not t["success"]][:10]
    return success + failure


def compute_per_step_obs_nll(model, tokenizer, traj, device, max_length=2048):
    """Compute obs NLL for EACH step separately in a trajectory."""
    messages = [{"role": "system", "content": "You are a helpful assistant in a text-based game."}]

    for step_data in traj["steps"]:
        messages.append({"role": "user", "content": step_data["observation"]})
        messages.append({"role": "assistant", "content": step_data["action"]})

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]

    input_ids = torch.tensor(full_ids).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]

    # Compute per-token NLL
    shift_logits = logits[:-1]
    shift_labels = input_ids[0, 1:]
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_nll = -log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    # Now find each step's observation token range and compute per-step NLL
    step_nlls = []
    role_prefix_tokens = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    gen_suffix_tokens = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

    for i, step_data in enumerate(traj["steps"]):
        # Build up to before this obs
        msgs_before = [{"role": "system", "content": "You are a helpful assistant in a text-based game."}]
        for j in range(i):
            msgs_before.append({"role": "user", "content": traj["steps"][j]["observation"]})
            msgs_before.append({"role": "assistant", "content": traj["steps"][j]["action"]})

        msgs_with = msgs_before.copy()
        msgs_with.append({"role": "user", "content": step_data["observation"]})

        text_before = tokenizer.apply_chat_template(msgs_before, tokenize=False, add_generation_prompt=False)
        text_with = tokenizer.apply_chat_template(msgs_with, tokenize=False, add_generation_prompt=True)

        ids_before = tokenizer.encode(text_before, add_special_tokens=False)
        ids_with = tokenizer.encode(text_with, add_special_tokens=False)

        content_start = len(ids_before) + len(role_prefix_tokens)
        content_end = len(ids_with) - len(gen_suffix_tokens)

        # Adjust for shift (shifted by 1 for autoregressive)
        shift_start = max(0, content_start - 1)
        shift_end = min(len(token_nll), content_end - 1)

        if shift_end > shift_start and shift_end <= len(token_nll):
            step_nll = token_nll[shift_start:shift_end].mean().item()
            step_nlls.append(step_nll)

        if content_end >= max_length:
            break

    return step_nlls


def main():
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    trajectories = load_trajectories(max_trajs=20)

    print(f"Loaded {len(trajectories)} trajectories")
    print(f"{'='*80}")

    # First load ref model (base) and compute per-step s_ref for all trajectories
    print("\nLoading ref model (base)...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

    all_ref_nlls = {}  # traj_idx -> list of per-step NLLs
    for i, traj in enumerate(trajectories):
        nlls = compute_per_step_obs_nll(ref_model, tokenizer, traj, device)
        all_ref_nlls[i] = nlls
        if (i+1) % 10 == 0:
            print(f"  Ref model: {i+1}/{len(trajectories)} trajectories done")
    del ref_model
    torch.cuda.empty_cache()

    # Now for each checkpoint, compute per-step s_theta and delta_s
    print(f"\n{'='*80}")
    print(f"{'Checkpoint':<15} {'ΔS mean':>10} {'ΔS std':>10} {'within-traj std':>16} {'raw S_θ std':>12} {'ratio':>8}")
    print("-" * 73)

    all_results = []

    for step in STEPS:
        model_path = f"{MERGED_ROOT}/step_{step}"
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

        all_delta_s = []
        all_s_theta = []
        within_traj_stds_delta = []
        within_traj_stds_raw = []

        for i, traj in enumerate(trajectories):
            theta_nlls = compute_per_step_obs_nll(model, tokenizer, traj, device)
            ref_nlls = all_ref_nlls[i]

            n = min(len(theta_nlls), len(ref_nlls))
            if n < 2:
                continue

            step_delta_s = [theta_nlls[j] - ref_nlls[j] for j in range(n)]
            step_raw_s = theta_nlls[:n]

            all_delta_s.extend(step_delta_s)
            all_s_theta.extend(step_raw_s)

            within_traj_stds_delta.append(np.std(step_delta_s))
            within_traj_stds_raw.append(np.std(step_raw_s))

        delta_mean = np.mean(all_delta_s)
        delta_std = np.std(all_delta_s)
        within_delta_std = np.mean(within_traj_stds_delta)
        within_raw_std = np.mean(within_traj_stds_raw)
        ratio = within_raw_std / within_delta_std if within_delta_std > 0 else float('inf')

        print(f"step {step:<9} {delta_mean:>10.4f} {delta_std:>10.4f} {within_delta_std:>16.4f} {within_raw_std:>12.4f} {ratio:>8.1f}x")

        all_results.append({
            "step": step,
            "delta_s_mean": delta_mean,
            "delta_s_global_std": delta_std,
            "delta_s_within_traj_std": within_delta_std,
            "raw_s_theta_within_traj_std": within_raw_std,
            "ratio_raw_over_delta": ratio,
            "n_trajs": len(within_traj_stds_delta),
        })

        del model
        torch.cuda.empty_cache()

    # Also show v2 step 1-2 data for comparison
    print(f"\n(v2 step 1-2 reference: ΔS within-traj std ≈ 0.009, raw S_θ within-traj std ≈ 0.147)")

    # OCAR weight simulation: what would softmax weights look like?
    print(f"\n{'='*80}")
    print("OCAR Weight Simulation (tau=1.0)")
    print(f"{'Checkpoint':<15} {'weight std (ΔS)':>16} {'weight std (raw)':>16} {'weight range (ΔS)':>20} {'weight range (raw)':>20}")
    print("-" * 90)

    for r in all_results:
        # Simulate softmax weights for a typical trajectory
        # Use within-traj std to generate synthetic steps
        n_steps = 10  # typical trajectory
        np.random.seed(42)

        # Delta-S weights
        ds_signal = np.random.normal(0, r["delta_s_within_traj_std"], n_steps)
        ds_weights = n_steps * np.exp(-ds_signal / 1.0) / np.exp(-ds_signal / 1.0).sum()

        # Raw S_theta weights
        raw_signal = np.random.normal(0, r["raw_s_theta_within_traj_std"], n_steps)
        raw_weights = n_steps * np.exp(-raw_signal / 1.0) / np.exp(-raw_signal / 1.0).sum()

        print(f"step {r['step']:<9} {np.std(ds_weights):>16.4f} {np.std(raw_weights):>16.4f} "
              f"[{ds_weights.min():.3f}, {ds_weights.max():.3f}]{'':>7} "
              f"[{raw_weights.min():.3f}, {raw_weights.max():.3f}]")

    # Save
    output_path = "/local_nvme/guanyiming/project/verl-agent/ocar/delta_s_variance_analysis.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
