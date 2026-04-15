"""
Compare two world-model surprise formulations:
  A) P(obs_{t+1} | obs_t, action_t)   — state + action context
  B) P(obs_{t+1} | action_t)          — action only context

Usage:
    python ocar/compare_wm_context.py \
        --model_path checkpoints/merged_hf/step_150 \
        --traj_path checkpoints/ocar_alfworld_*/global_step_25/ocar_trajectories.json
"""
import argparse
import json

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def compute_nll_of_target(model, tokenizer, context_str, target_str, device):
    """Compute NLL of target tokens given context.

    Returns (nll_mean, nll_sum, n_tokens).
    """
    ctx_ids = tokenizer.encode(context_str, add_special_tokens=True)
    tgt_ids = tokenizer.encode(target_str, add_special_tokens=False)

    if not tgt_ids:
        return 0.0, 0.0, 0

    input_ids = torch.tensor([ctx_ids + tgt_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, use_cache=False).logits

    # log probs shifted: position i predicts token i+1
    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, seq_len-1)

    # Only take the target portion: positions [len(ctx_ids)-1 : len(ctx_ids)-1+len(tgt_ids)]
    # Because token at position len(ctx_ids) is first target token,
    # its log prob is at shifted position len(ctx_ids)-1
    start = len(ctx_ids) - 1
    end = start + len(tgt_ids)
    tgt_lp = token_log_probs[0, start:end]

    nll_sum = -tgt_lp.sum().item()
    nll_mean = nll_sum / tgt_lp.numel()
    return nll_mean, nll_sum, tgt_lp.numel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--traj_path", required=True)
    parser.add_argument("--n_traj", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(args.device).eval()
    print("Model loaded.\n")

    with open(args.traj_path) as f:
        traj_data = json.load(f)

    trajectories = traj_data["trajectories"]
    success = [t for t in trajectories if t["success"]]
    failure = [t for t in trajectories if not t["success"]]
    selected = (success[:args.n_traj // 2] + failure[:args.n_traj // 2])[:args.n_traj]
    if not selected:
        selected = trajectories[:args.n_traj]

    all_A, all_B = [], []

    for traj in selected:
        steps = traj["steps"]
        n = min(len(steps), args.max_steps + 1)

        print(f"Traj {traj['traj_id'][:8]}... | success={traj['success']} | steps={traj['n_steps']}")
        print(f"{'step':>4} | {'A: s+a→s\'':>12} {'n':>4} | {'B: a→s\'':>12} {'n':>4} | {'A-B':>7} | obs_t+1 preview")
        print("-" * 100)

        for t in range(n - 1):
            obs_t = steps[t]["observation"]
            action_t = steps[t]["action"]
            obs_next = steps[t + 1]["observation"]

            if not obs_next.strip():
                continue

            # A: state + action → next state
            context_A = f"{obs_t}\n{action_t}"
            nll_A, _, n_A = compute_nll_of_target(model, tokenizer, context_A, obs_next, args.device)

            # B: action only → next state
            context_B = action_t
            nll_B, _, n_B = compute_nll_of_target(model, tokenizer, context_B, obs_next, args.device)

            diff = nll_A - nll_B
            all_A.append(nll_A)
            all_B.append(nll_B)

            preview = obs_next[:60].replace('\n', ' ')
            print(f"{t:>4} | {nll_A:>12.4f} {n_A:>4} | {nll_B:>12.4f} {n_B:>4} | {diff:>+7.3f} | {preview}")

        print("=" * 100)

    print(f"\nOverall summary ({len(all_A)} steps):")
    print(f"  A (state+action→state'): mean={np.mean(all_A):.4f} ± {np.std(all_A):.4f}")
    print(f"  B (action→state'):       mean={np.mean(all_B):.4f} ± {np.std(all_B):.4f}")
    print(f"  A - B:                   mean={np.mean(np.array(all_A) - np.array(all_B)):.4f}")
    print(f"  Correlation(A, B):       {np.corrcoef(all_A, all_B)[0,1]:.4f}")


if __name__ == "__main__":
    main()
