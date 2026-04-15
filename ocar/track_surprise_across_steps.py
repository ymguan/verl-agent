"""
Track surprise signals across training checkpoints, including delta_s.

Loads a reference model (base) once, then each checkpoint, computes:
  - obs_nll_last:  S_theta of current obs given full history
  - obs_nll_ref:   S_ref of current obs (same context, ref model)
  - delta_s:       S_theta - S_ref
  - wm_A:          P(obs_{t+1} | obs_t, action_t)  — state+action world model
  - wm_B:          P(obs_{t+1} | action_t)          — action-only world model
  - action_nll:    NLL of action tokens given history

Usage:
    python ocar/track_surprise_across_steps.py \
        --ckpt_dir checkpoints/merged_hf \
        --ref_model /local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct \
        --traj_path checkpoints/ocar_alfworld_*/global_step_25/ocar_trajectories.json \
        --device cuda:0
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_nll_of_target(model, input_ids_full, ctx_len, device):
    """Compute mean NLL of target tokens (positions ctx_len onwards)."""
    with torch.no_grad():
        logits = model(input_ids=input_ids_full, use_cache=False).logits
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids_full[:, 1:]
    token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)
    tgt_lp = token_lp[0, ctx_len - 1:]
    if tgt_lp.numel() == 0:
        return 0.0, 0
    return -tgt_lp.mean().item(), tgt_lp.numel()


def build_chatml_ids(tokenizer, role, content):
    """Encode a single chatml turn."""
    text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
    return tokenizer.encode(text, add_special_tokens=False)


def build_obs_context_and_target(tokenizer, steps, t):
    """Build context (history up to act_{t-1}) and target (obs_t as tool turn).

    Returns (ctx_ids, tgt_ids) for computing obs NLL at step t.
    For t=0 returns None (no meaningful obs surprise).
    """
    if t == 0:
        return None, None

    ctx_parts = []
    ctx_parts.extend(build_chatml_ids(tokenizer, "user", steps[0]["observation"]))
    for j in range(t):
        ctx_parts.extend(build_chatml_ids(tokenizer, "assistant", steps[j]["action"]))
        if j + 1 < t:
            ctx_parts.extend(build_chatml_ids(tokenizer, "tool", steps[j + 1]["observation"]))
    # Target: obs_t as tool response
    tgt_ids = build_chatml_ids(tokenizer, "tool", steps[t]["observation"])
    return ctx_parts, tgt_ids


def build_action_context_and_target(tokenizer, steps, t):
    """Build context (history up to obs_t) and target (action_t content).

    Returns (ctx_ids, tgt_ids).
    """
    ctx_parts = []
    ctx_parts.extend(build_chatml_ids(tokenizer, "user", steps[0]["observation"]))
    for j in range(t):
        ctx_parts.extend(build_chatml_ids(tokenizer, "assistant", steps[j]["action"]))
        ctx_parts.extend(build_chatml_ids(tokenizer, "tool", steps[j + 1]["observation"]))
    act_prefix = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    ctx_parts.extend(act_prefix)
    tgt_ids = tokenizer.encode(steps[t]["action"], add_special_tokens=False)
    return ctx_parts, tgt_ids


def compute_signals_for_trajectory(model, tokenizer, steps, max_steps, device):
    """Compute obs_nll_last, wm_A, wm_B, action_nll for one trajectory with one model."""
    n = min(len(steps), max_steps + 1)
    results = {"obs_nll_last": [], "wm_A": [], "wm_B": [], "action_nll": []}

    for t in range(n - 1):
        obs_t = steps[t]["observation"]
        action_t = steps[t]["action"]
        obs_next = steps[t + 1]["observation"]

        # --- obs_nll_last ---
        ctx, tgt = build_obs_context_and_target(tokenizer, steps, t)
        if ctx is not None and tgt:
            full = torch.tensor([ctx + tgt], dtype=torch.long, device=device)
            nll, _ = compute_nll_of_target(model, full, len(ctx), device)
            results["obs_nll_last"].append(nll)
        else:
            results["obs_nll_last"].append(0.0)

        # --- wm_A: P(obs_{t+1} | obs_t, action_t) ---
        ctx_A = tokenizer.encode(f"{obs_t}\n{action_t}", add_special_tokens=True)
        tgt_A = tokenizer.encode(obs_next, add_special_tokens=False)
        if tgt_A:
            full_A = torch.tensor([ctx_A + tgt_A], dtype=torch.long, device=device)
            wm_a, _ = compute_nll_of_target(model, full_A, len(ctx_A), device)
        else:
            wm_a = 0.0
        results["wm_A"].append(wm_a)

        # --- wm_B: P(obs_{t+1} | action_t) ---
        ctx_B = tokenizer.encode(action_t, add_special_tokens=True)
        if tgt_A:
            full_B = torch.tensor([ctx_B + tgt_A], dtype=torch.long, device=device)
            wm_b, _ = compute_nll_of_target(model, full_B, len(ctx_B), device)
        else:
            wm_b = 0.0
        results["wm_B"].append(wm_b)

        # --- action_nll ---
        ctx_act, tgt_act = build_action_context_and_target(tokenizer, steps, t)
        full_act = torch.tensor([ctx_act + tgt_act], dtype=torch.long, device=device)
        act_nll, _ = compute_nll_of_target(model, full_act, len(ctx_act), device)
        results["action_nll"].append(act_nll)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", required=True, help="Dir with step_* subdirs")
    parser.add_argument("--ref_model", required=True, help="Path to reference/base model")
    parser.add_argument("--traj_path", required=True, help="Trajectory JSON for test data")
    parser.add_argument("--n_traj", type=int, default=6)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # Find checkpoints
    step_dirs = sorted(glob.glob(os.path.join(args.ckpt_dir, "step_*")))
    steps_available = []
    for d in step_dirs:
        m = re.search(r"step_(\d+)", d)
        if m and os.path.exists(os.path.join(d, "config.json")):
            steps_available.append((int(m.group(1)), d))
    steps_available.sort()
    print(f"Found checkpoints: {[s[0] for s in steps_available]}")

    # Load trajectory data
    with open(args.traj_path) as f:
        traj_data = json.load(f)
    trajectories = traj_data["trajectories"]
    success = [t for t in trajectories if t["success"]]
    failure = [t for t in trajectories if not t["success"]]
    selected = (success[:args.n_traj // 2] + failure[:args.n_traj // 2])[:args.n_traj]
    if not selected:
        selected = trajectories[:args.n_traj]
    print(f"Using {len(selected)} trajectories ({sum(t['success'] for t in selected)} success)\n")

    # ── Step 0: compute ref model signals (once) ──
    print("=" * 80)
    print(f"Loading REFERENCE model from {args.ref_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=True)
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.ref_model, dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()

    ref_signals = {"obs_nll_last": [], "wm_A": [], "wm_B": [], "action_nll": []}
    for traj in selected:
        r = compute_signals_for_trajectory(ref_model, tokenizer, traj["steps"], args.max_steps, args.device)
        for k in ref_signals:
            ref_signals[k].extend(r[k])

    # Store per-step ref values for delta computation
    ref_obs_nll_per_step = ref_signals["obs_nll_last"]  # flat list across all trajs
    ref_wm_A_per_step = ref_signals["wm_A"]

    ref_summary = {k: (np.mean(v), np.std(v)) for k, v in ref_signals.items() if v}
    print(f"  Ref: obs_nll_last={ref_summary['obs_nll_last'][0]:.4f}  "
          f"wm_A={ref_summary['wm_A'][0]:.4f}  "
          f"wm_B={ref_summary['wm_B'][0]:.4f}  "
          f"action_nll={ref_summary['action_nll'][0]:.4f}")

    del ref_model
    torch.cuda.empty_cache()

    # ── Step 1..N: compute each checkpoint ──
    all_results = {"ref": ref_summary}

    for step_num, ckpt_path in steps_available:
        print(f"\n{'=' * 80}")
        print(f"Loading checkpoint step_{step_num}...")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_path, dtype=torch.bfloat16, trust_remote_code=True
        ).to(args.device).eval()

        theta_signals = {"obs_nll_last": [], "wm_A": [], "wm_B": [], "action_nll": []}
        for traj in selected:
            r = compute_signals_for_trajectory(model, tokenizer, traj["steps"], args.max_steps, args.device)
            for k in theta_signals:
                theta_signals[k].extend(r[k])

        # Compute delta_s = S_theta - S_ref (per-step then average)
        delta_obs = np.array(theta_signals["obs_nll_last"]) - np.array(ref_obs_nll_per_step)
        delta_wm_A = np.array(theta_signals["wm_A"]) - np.array(ref_wm_A_per_step)

        step_result = {k: (np.mean(v), np.std(v)) for k, v in theta_signals.items() if v}
        step_result["delta_obs"] = (float(np.mean(delta_obs)), float(np.std(delta_obs)))
        step_result["delta_wm_A"] = (float(np.mean(delta_wm_A)), float(np.std(delta_wm_A)))
        all_results[step_num] = step_result

        print(f"  Step {step_num}: "
              f"obs={step_result['obs_nll_last'][0]:.4f}  "
              f"delta_obs={step_result['delta_obs'][0]:+.4f}  "
              f"wm_A={step_result['wm_A'][0]:.4f}  "
              f"delta_wm_A={step_result['delta_wm_A'][0]:+.4f}  "
              f"action_nll={step_result['action_nll'][0]:.4f}")

        del model
        torch.cuda.empty_cache()

    # ── Summary table ──
    print(f"\n{'=' * 90}")
    print("TRAINING PROGRESSION SUMMARY (with delta_s = S_theta - S_ref)")
    print(f"{'=' * 90}")
    header = (f"{'Step':>6} | {'obs_nll':>9} | {'delta_obs':>10} | "
              f"{'wm_A':>9} | {'delta_wm':>10} | "
              f"{'wm_B':>9} | {'act_nll':>9} | {'A-B':>7}")
    print(header)
    print("-" * len(header))

    # Print ref first
    r = all_results["ref"]
    print(f"{'ref':>6} | {r['obs_nll_last'][0]:>9.4f} | {'   ---':>10} | "
          f"{r['wm_A'][0]:>9.4f} | {'   ---':>10} | "
          f"{r['wm_B'][0]:>9.4f} | {r['action_nll'][0]:>9.4f} | "
          f"{r['wm_A'][0] - r['wm_B'][0]:>+7.3f}")

    for step_num in sorted(k for k in all_results if k != "ref"):
        r = all_results[step_num]
        a_b = r['wm_A'][0] - r['wm_B'][0]
        print(f"{step_num:>6} | {r['obs_nll_last'][0]:>9.4f} | {r['delta_obs'][0]:>+10.4f} | "
              f"{r['wm_A'][0]:>9.4f} | {r['delta_wm_A'][0]:>+10.4f} | "
              f"{r['wm_B'][0]:>9.4f} | {r['action_nll'][0]:>9.4f} | "
              f"{a_b:>+7.3f}")

    # Save raw results
    out_path = os.path.join(os.path.dirname(args.traj_path), "surprise_trend_with_delta.json")
    serializable = {}
    for k, v in all_results.items():
        serializable[str(k)] = {kk: list(vv) for kk, vv in v.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
