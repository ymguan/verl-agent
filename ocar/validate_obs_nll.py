"""
Validate obs NLL computation using a saved checkpoint.

Loads a model, constructs multi-turn sequences from saved trajectory data,
runs forward pass, and computes obs NLL with the last-block extraction.

Usage:
    python ocar/validate_obs_nll.py \
        --model_path checkpoints/merged_hf/step_150 \
        --traj_path checkpoints/ocar_alfworld_*/ocar_tau*/global_step_25/ocar_trajectories.json \
        --n_traj 4 --device cuda:0
"""
import argparse
import json
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_chatml_sequence(tokenizer, obs_list, action_list):
    """Build a multi-turn chatml sequence from obs/action pairs.

    Returns input_ids, attention_mask, loss_mask as lists.
    """
    # System message
    messages_for_prompt = []

    input_ids = []
    loss_mask = []

    # Build turn by turn using chatml format
    # <|im_start|>system\n...<|im_end|>\n
    # <|im_start|>user\n...<|im_end|>\n
    # <|im_start|>assistant\n...<|im_end|>\n
    # <|im_start|>tool\n...<|im_end|>\n

    im_start = tokenizer.encode("<|im_start|>", add_special_tokens=False)
    im_end = tokenizer.encode("<|im_end|>", add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)

    def add_role_content(role, content, is_action_content=False):
        # <|im_start|>role\ncontent<|im_end|>\n
        role_tokens = tokenizer.encode(role, add_special_tokens=False)
        content_tokens = tokenizer.encode(content, add_special_tokens=False)

        # prefix: <|im_start|>role\n
        prefix = im_start + role_tokens + nl
        # suffix: <|im_end|>\n
        suffix = im_end + nl

        input_ids.extend(prefix)
        loss_mask.extend([0] * len(prefix))

        input_ids.extend(content_tokens)
        if is_action_content:
            loss_mask.extend([1] * len(content_tokens))
        else:
            loss_mask.extend([0] * len(content_tokens))

        input_ids.extend(suffix)
        if is_action_content:
            loss_mask.extend([1] * len(suffix))
        else:
            loss_mask.extend([0] * len(suffix))

    # First obs is the user message (task description)
    add_role_content("user", obs_list[0], is_action_content=False)

    for step_i in range(len(action_list)):
        # Assistant action
        add_role_content("assistant", action_list[step_i], is_action_content=True)

        # Tool response (next obs), if exists
        if step_i + 1 < len(obs_list):
            add_role_content("tool", obs_list[step_i + 1], is_action_content=False)

    attention_mask = [1] * len(input_ids)
    return input_ids, attention_mask, loss_mask


def find_last_obs_block(loss_mask, prompt_len):
    """Find (start, end) of the last contiguous 0-block in loss_mask[:prompt_len]."""
    end = prompt_len
    start = end
    for j in range(end - 1, -1, -1):
        if loss_mask[j] == 1:
            start = j + 1
            break
    else:
        start = 0
    return (start, end)


def compute_obs_nll(model, tokenizer, obs_list, action_list, device):
    """Compute per-step obs NLL for a single trajectory.

    Returns list of dicts, one per step, with:
        step, obs_preview, n_obs_tokens_last, nll_last_mean, nll_last_sum,
        n_obs_tokens_all, nll_all_mean, nll_all_sum
    """
    results = []

    # For each step t, build the sequence up to step t and compute NLL
    for t in range(len(action_list)):
        # Build sequence: obs_0, action_0, obs_1, action_1, ..., obs_t, action_t
        obs_sub = obs_list[:t + 2] if t + 1 < len(obs_list) else obs_list[:t + 1]
        act_sub = action_list[:t + 1]

        input_ids_list, attn_list, lm_list = build_chatml_sequence(tokenizer, obs_sub, act_sub)

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.tensor([attn_list], dtype=torch.long, device=device)
        loss_mask_t = torch.tensor([lm_list], dtype=torch.long, device=device)

        seq_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Compute log probs: log p(token[i+1] | token[0:i+1])
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)  # (1, seq_len-1, vocab_size)
        # Gather actual token log probs
        target_ids = input_ids[:, 1:]  # (1, seq_len-1)
        full_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

        # Find response length (last action)
        # The response is the last action's tokens. Find where the last 1-block starts.
        lm_np = np.array(lm_list)
        last_1_end = len(lm_np)
        # Find start of last 1-block
        last_1_start = last_1_end
        for j in range(last_1_end - 1, -1, -1):
            if lm_np[j] == 0:
                last_1_start = j + 1
                break
        response_length = last_1_end - last_1_start
        prompt_len = seq_len - response_length

        # --- Last obs block ---
        start, end = find_last_obs_block(lm_list, prompt_len)
        if start < end and start > 0:
            flp_start = start - 1
            flp_end = end - 1
            block_lp = full_log_probs[0, flp_start:flp_end]
            block_attn = attention_mask[0, start:end]
            valid = block_attn.bool()
            if valid.any():
                obs_lp = block_lp[valid]
                nll_last_sum = -obs_lp.sum().item()
                nll_last_mean = nll_last_sum / obs_lp.numel()
                n_last = obs_lp.numel()
            else:
                nll_last_mean = nll_last_sum = 0.0
                n_last = 0
        else:
            nll_last_mean = nll_last_sum = 0.0
            n_last = 0

        # --- All obs tokens ---
        obs_mask_all = (attention_mask[0, 1:] == 1) & (loss_mask_t[0, 1:] == 0)
        if obs_mask_all.any():
            all_lp = full_log_probs[0][obs_mask_all]
            nll_all_sum = -all_lp.sum().item()
            nll_all_mean = nll_all_sum / all_lp.numel()
            n_all = all_lp.numel()
        else:
            nll_all_mean = nll_all_sum = 0.0
            n_all = 0

        # Current obs preview
        if t + 1 < len(obs_list):
            obs_preview = obs_list[t + 1][:80]
        else:
            obs_preview = obs_list[t][:80] if t < len(obs_list) else ""

        results.append({
            "step": t,
            "obs_preview": obs_preview,
            "n_obs_tokens_last": n_last,
            "nll_last_mean": round(nll_last_mean, 4),
            "nll_last_sum": round(nll_last_sum, 2),
            "n_obs_tokens_all": n_all,
            "nll_all_mean": round(nll_all_mean, 4),
            "nll_all_sum": round(nll_all_sum, 2),
        })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to HF model checkpoint")
    parser.add_argument("--traj_path", required=True, help="Path to trajectory JSON")
    parser.add_argument("--n_traj", type=int, default=4, help="Number of trajectories to analyze")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps per trajectory")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")

    print(f"Loading trajectories from {args.traj_path}...")
    with open(args.traj_path) as f:
        traj_data = json.load(f)

    trajectories = traj_data["trajectories"]
    # Pick a mix of success and failure
    success = [t for t in trajectories if t["success"]]
    failure = [t for t in trajectories if not t["success"]]
    selected = (success[:args.n_traj // 2] + failure[:args.n_traj // 2])[:args.n_traj]
    if not selected:
        selected = trajectories[:args.n_traj]

    print(f"Selected {len(selected)} trajectories ({sum(t['success'] for t in selected)} success)")
    print("=" * 100)

    for traj in selected:
        obs_list = [s["observation"] for s in traj["steps"]]
        action_list = [s["action"] for s in traj["steps"]]

        # Limit steps
        n_steps = min(len(action_list), args.max_steps)
        obs_list = obs_list[:n_steps + 1]
        action_list = action_list[:n_steps]

        print(f"\nTrajectory: {traj['traj_id'][:8]}... | success={traj['success']} | "
              f"steps={traj['n_steps']} (showing {n_steps})")
        print(f"Task: {obs_list[0][:120]}...")
        print("-" * 100)
        print(f"{'Step':>4} | {'NLL_last':>9} {'n_tok':>5} | {'NLL_all':>9} {'n_tok':>6} | Obs preview")
        print("-" * 100)

        results = compute_obs_nll(model, tokenizer, obs_list, action_list, args.device)

        for r in results:
            print(f"{r['step']:>4} | {r['nll_last_mean']:>9.4f} {r['n_obs_tokens_last']:>5} | "
                  f"{r['nll_all_mean']:>9.4f} {r['n_obs_tokens_all']:>6} | {r['obs_preview']}")

        # Summary
        last_means = [r["nll_last_mean"] for r in results if r["n_obs_tokens_last"] > 0]
        all_means = [r["nll_all_mean"] for r in results if r["n_obs_tokens_all"] > 0]
        if last_means:
            print(f"  Summary: NLL_last avg={np.mean(last_means):.4f} std={np.std(last_means):.4f}")
        if all_means:
            print(f"           NLL_all  avg={np.mean(all_means):.4f} std={np.std(all_means):.4f}")
        print("=" * 100)


if __name__ == "__main__":
    main()
