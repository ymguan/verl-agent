"""Probe: does ||Δh|| (hidden state shift at observation boundaries)
carry more informational signal than obs NLL?

Uses a base model (no training checkpoint needed) to test the concept.
For each trajectory step, computes:
  - h_pre:  hidden state at last action token (before obs)
  - h_post: hidden state at last obs token (after obs)
  - delta_h = ||h_post - h_pre||_2
  - cos_sim = cosine(h_post, h_pre)

Then tests whether delta_h discriminates:
  1. state_change vs revisit vs nothing_happens
  2. success vs failure trajectories
  3. informative vs non-informative steps
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def classify_obs(obs, seen_in_traj):
    obs = obs.strip()
    if obs == "Nothing happens.":
        return "nothing"
    if obs in seen_in_traj:
        return "revisit"
    if obs.startswith(("You pick up", "You put", "You open", "You close",
                       "You turn", "You heat", "You cool", "You clean")):
        return "state_change"
    if obs.startswith("-= Welcome"):
        return "welcome"
    if obs.startswith("You are carrying"):
        return "inventory"
    return "new_location"


def build_sequence(tokenizer, steps, max_len=2048):
    """Build a single token sequence from trajectory steps (simplified ChatML)."""
    parts = []
    boundaries = []  # (action_end_pos, obs_end_pos, obs_category)

    seen_obs = set()

    for s in steps:
        action_text = s['action']
        obs_text = s['observation']

        cat = classify_obs(obs_text, seen_obs)
        if obs_text.strip() not in ("Nothing happens.",) and obs_text.strip().startswith(("-= Welcome",)) is False:
            seen_obs.add(obs_text.strip())

        act_part = f"<|im_start|>assistant\n{action_text}<|im_end|>\n"
        obs_part = f"<|im_start|>user\n{obs_text}<|im_end|>\n"

        parts.append(('action', act_part, cat))
        parts.append(('obs', obs_part, cat))

    full_text = ""
    token_boundaries = []

    for role, text, cat in parts:
        start_pos = len(tokenizer.encode(full_text, add_special_tokens=False)) if full_text else 0
        full_text += text
        end_pos = len(tokenizer.encode(full_text, add_special_tokens=False))
        token_boundaries.append((role, start_pos, end_pos, cat))

    input_ids = tokenizer.encode(full_text, add_special_tokens=False, truncation=True, max_length=max_len)
    return input_ids, token_boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--traj_file", default="checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json")
    parser.add_argument("--max_trajs", type=int, default=30)
    parser.add_argument("--layer", type=int, default=-1, help="Which layer's hidden state to use (-1=last)")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    n_layers = model.config.num_hidden_layers
    layer_idx = args.layer if args.layer >= 0 else n_layers + args.layer
    print(f"Using layer {layer_idx}/{n_layers}, hidden_size={model.config.hidden_size}")

    with open(args.traj_file) as f:
        data = json.load(f)
    trajs = data['trajectories'][:args.max_trajs]
    print(f"Processing {len(trajs)} trajectories...")

    results = []

    for ti, traj in enumerate(trajs):
        steps = traj['steps']
        if len(steps) < 2:
            continue

        seen_obs = set()

        for si, step in enumerate(steps):
            action_text = step['action']
            obs_text = step['observation']
            cat = classify_obs(obs_text, seen_obs)
            seen_obs.add(obs_text.strip())

            # Use sliding window: only last N steps of context to avoid truncation
            context_window = 3
            context_before = ""
            start_idx = max(0, si - context_window)
            for prev_s in steps[start_idx:si]:
                context_before += f"<|im_start|>assistant\n{prev_s['action']}<|im_end|>\n"
                context_before += f"<|im_start|>user\n{prev_s['observation']}<|im_end|>\n"

            seq_with_action = context_before + f"<|im_start|>assistant\n{action_text}<|im_end|>\n"
            seq_with_obs = seq_with_action + f"<|im_start|>user\n{obs_text}<|im_end|>\n"

            ids_action = tokenizer.encode(seq_with_action, add_special_tokens=False, truncation=True, max_length=2048)
            ids_full = tokenizer.encode(seq_with_obs, add_special_tokens=False, truncation=True, max_length=2048)

            n_obs_tok = len(ids_full) - len(ids_action)
            if n_obs_tok < 2 or len(ids_action) < 2:
                continue

            with torch.no_grad():
                out_action = model(
                    torch.tensor([ids_action], device=device),
                    output_hidden_states=True
                )
                h_pre = out_action.hidden_states[layer_idx][0, -1].float().cpu()

                out_full = model(
                    torch.tensor([ids_full], device=device),
                    output_hidden_states=True
                )
                h_post = out_full.hidden_states[layer_idx][0, -1].float().cpu()

            delta_h = (h_post - h_pre).norm().item()
            cos_sim = torch.nn.functional.cosine_similarity(h_pre.unsqueeze(0), h_post.unsqueeze(0)).item()

            results.append({
                'traj_idx': ti,
                'step': si,
                'traj_id': traj['traj_id'],
                'success': traj['success'],
                'obs_category': cat,
                'delta_h_l2': delta_h,
                'cos_sim': cos_sim,
                's_theta': step['s_theta_mean'],
                'delta_s': step['delta_s_mean'],
                'entropy': step['entropy_mean'],
                'obs_text': obs_text[:100],
                'n_obs_tokens': len(ids_full) - len(ids_action),
            })

        if (ti + 1) % 5 == 0:
            print(f"  {ti+1}/{len(trajs)} trajectories done")

    # === Analysis ===
    print(f"\n{'='*70}")
    print(f"Total step-observations: {len(results)}")

    # 1. By observation category
    print(f"\n--- By observation category ---")
    cats = {}
    for r in results:
        c = r['obs_category']
        if c not in cats:
            cats[c] = []
        cats[c].append(r)

    print(f"{'category':<15} {'n':>5} {'δh_L2':>8} {'cos_sim':>8} {'s_θ':>8} {'ΔS':>8} {'ent':>8}")
    print("-" * 75)
    for cat in ['state_change', 'new_location', 'revisit', 'nothing', 'welcome', 'inventory']:
        if cat not in cats:
            continue
        items = cats[cat]
        dh = np.array([r['delta_h_l2'] for r in items])
        cs = np.array([r['cos_sim'] for r in items])
        st = np.array([r['s_theta'] for r in items])
        ds = np.array([r['delta_s'] for r in items])
        en = np.array([r['entropy'] for r in items])
        print(f"{cat:<15} {len(items):>5} {dh.mean():>8.2f} {cs.mean():>8.4f} {st.mean():>8.3f} {ds.mean():>+8.3f} {en.mean():>8.3f}")

    # 2. Success vs Failure
    print(f"\n--- Success vs Failure (per-step means) ---")
    for label, filt in [("SUCC", True), ("FAIL", False)]:
        items = [r for r in results if r['success'] == filt]
        if not items:
            continue
        dh = np.array([r['delta_h_l2'] for r in items])
        st = np.array([r['s_theta'] for r in items])
        en = np.array([r['entropy'] for r in items])
        print(f"{label}: n={len(items):>4}, δh={dh.mean():.2f}±{dh.std():.2f}, s_θ={st.mean():.3f}, ent={en.mean():.3f}")

    # 3. Discrimination AUC: δh vs s_θ vs entropy for state_change vs nothing
    print(f"\n--- AUC: state_change vs nothing_happens ---")
    from scipy.stats import mannwhitneyu
    def manual_auc(pos_scores, neg_scores):
        stat, p = mannwhitneyu(pos_scores, neg_scores, alternative='two-sided')
        auc = stat / (len(pos_scores) * len(neg_scores))
        return auc, p

    sc = [r for r in results if r['obs_category'] == 'state_change']
    nh = [r for r in results if r['obs_category'] == 'nothing']
    if sc and nh:
        for signal_name in ['delta_h_l2', 's_theta', 'entropy']:
            pos = [r[signal_name] for r in sc]
            neg = [r[signal_name] for r in nh]
            auc, p = manual_auc(pos, neg)
            best_auc = max(auc, 1-auc)
            print(f"  {signal_name:<12}: AUC={best_auc:.3f} (p={p:.2e}, direction: {'high=state_change' if auc>0.5 else 'low=state_change'})")

    # 4. AUC: informative (state_change + new_location) vs non-informative (revisit + nothing)
    print(f"\n--- AUC: informative vs non-informative ---")
    info = [r for r in results if r['obs_category'] in ('state_change', 'new_location')]
    noinfo = [r for r in results if r['obs_category'] in ('revisit', 'nothing')]
    if info and noinfo:
        for signal_name in ['delta_h_l2', 's_theta', 'entropy']:
            pos = [r[signal_name] for r in info]
            neg = [r[signal_name] for r in noinfo]
            auc, p = manual_auc(pos, neg)
            best_auc = max(auc, 1-auc)
            print(f"  {signal_name:<12}: AUC={best_auc:.3f} (p={p:.2e}, direction: {'high=informative' if auc>0.5 else 'low=informative'})")

    # 5. Correlation matrix
    print(f"\n--- Correlations (Spearman) ---")
    from scipy.stats import spearmanr
    signals = ['delta_h_l2', 's_theta', 'delta_s', 'entropy']
    for i, s1 in enumerate(signals):
        for s2 in signals[i+1:]:
            v1 = [r[s1] for r in results]
            v2 = [r[s2] for r in results]
            rho, p = spearmanr(v1, v2)
            print(f"  ρ({s1}, {s2}) = {rho:+.3f} (p={p:.2e})")

    # Save raw results
    out_path = Path("ocar/analysis_results/hidden_state_probe.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
