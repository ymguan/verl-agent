"""
Compute delta_s (obs NLL) and hidden state info_gain using Qwen2.5-1.5B-Instruct
on both ALFWorld and WebShop trajectories.

Goal: test whether a small external model can provide step-level succ/fail signal.
- s_1.5B: per-step obs NLL (same as s_theta but from 1.5B)
- info_gain_1.5B: ||h_post - h_pre|| from 1.5B hidden states
"""

import json, os, sys
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path, device="cuda"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def compute_obs_nll(model, tokenizer, context_before_obs, obs_text):
    """Compute NLL of observation tokens given context."""
    full_text = context_before_obs + obs_text
    full_ids = tokenizer.encode(full_text, return_tensors="pt", add_special_tokens=True,
                                 truncation=True, max_length=8192).to(model.device)
    ctx_ids = tokenizer.encode(context_before_obs, return_tensors="pt", add_special_tokens=True,
                                truncation=True, max_length=8192).to(model.device)

    ctx_len = ctx_ids.shape[1]
    obs_len = full_ids.shape[1] - ctx_len

    if obs_len <= 0:
        return None, 0

    with torch.no_grad():
        logits = model(full_ids).logits[0]  # (seq_len, vocab)

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    obs_start = max(ctx_len - 1, 0)
    obs_nlls = []
    for i in range(obs_start, full_ids.shape[1] - 1):
        target = full_ids[0, i + 1]
        nll = -log_probs[i, target].item()
        if i >= ctx_len - 1:
            obs_nlls.append(nll)

    if not obs_nlls:
        return None, 0
    return np.mean(obs_nlls), len(obs_nlls)


def compute_info_gain(model, tokenizer, pre_obs_text, post_obs_text, layers=(-1,)):
    """Compute hidden state info gain. Uses max_length=8192 to reduce truncation artifacts."""
    pre_ids = tokenizer.encode(pre_obs_text, return_tensors="pt", add_special_tokens=True,
                                truncation=True, max_length=8192).to(model.device)
    post_ids = tokenizer.encode(post_obs_text, return_tensors="pt", add_special_tokens=True,
                                 truncation=True, max_length=8192).to(model.device)

    if pre_ids.shape[1] == 0 or post_ids.shape[1] == 0:
        return {l: 0.0 for l in layers}, {l: 1.0 for l in layers}

    if pre_ids.shape[1] == post_ids.shape[1]:
        if torch.equal(pre_ids, post_ids):
            return {l: 0.0 for l in layers}, {l: 1.0 for l in layers}

    with torch.no_grad():
        out_pre = model(pre_ids, output_hidden_states=True)
        out_post = model(post_ids, output_hidden_states=True)

    ig, cos = {}, {}
    for layer in layers:
        h_pre = out_pre.hidden_states[layer][0, -1].float()
        h_post = out_post.hidden_states[layer][0, -1].float()
        ig[layer] = (h_post - h_pre).norm().item()
        cos[layer] = torch.nn.functional.cosine_similarity(
            h_pre.unsqueeze(0), h_post.unsqueeze(0)
        ).item()
    return ig, cos


def process_trajectories(model, tokenizer, trajs, layers=(-1,)):
    results = []
    for ti, traj in enumerate(trajs):
        if ti % 5 == 0:
            print(f"  Traj {ti+1}/{len(trajs)}...")
        steps = traj["steps"]
        traj_result = {"success": traj["success"], "n_steps": len(steps), "steps": []}
        context_parts = []

        for si, step in enumerate(steps):
            obs = step["observation"]
            action = step["action"]

            pre_obs_text = "".join(context_parts)
            post_obs_text = pre_obs_text + obs
            ctx_before_obs = pre_obs_text  # for NLL: context before obs

            step_info = {
                "step": si,
                "orig_delta_s": step.get("delta_s_mean"),
                "orig_entropy": step.get("entropy_mean"),
                "orig_s_theta": step.get("s_theta_mean"),
            }

            # Compute obs NLL from 1.5B
            nll, n_tok = compute_obs_nll(model, tokenizer, ctx_before_obs, obs)
            step_info["s_1_5b"] = nll
            step_info["obs_n_tokens_1_5b"] = n_tok

            # Compute info gain (skip step 0)
            if si == 0:
                step_info["info_gain"] = {str(l): 0.0 for l in layers}
                step_info["cosine"] = {str(l): 1.0 for l in layers}
            else:
                with torch.no_grad():
                    ig, cos = compute_info_gain(model, tokenizer, pre_obs_text, post_obs_text, layers)
                step_info["info_gain"] = {str(l): v for l, v in ig.items()}
                step_info["cosine"] = {str(l): v for l, v in cos.items()}

            traj_result["steps"].append(step_info)
            context_parts.append(obs)
            context_parts.append(action)

        results.append(traj_result)
    return results


def analyze(results, label):
    print(f"\n{'='*60}")
    print(f"Analysis: {label}")
    print(f"{'='*60}")

    n_succ = sum(1 for r in results if r["success"])
    n_fail = sum(1 for r in results if not r["success"])
    print(f"Trajectories: {len(results)} total, {n_succ} succ, {n_fail} fail")

    if n_succ == 0 or n_fail == 0:
        print("Cannot compare (one group empty)")
        return

    # Collect step-level signals (skip step 0)
    succ_nll, fail_nll = [], []
    succ_ig, fail_ig = [], []
    succ_orig_ds, fail_orig_ds = [], []
    succ_orig_ent, fail_orig_ent = [], []

    succ_traj_nll, fail_traj_nll = [], []
    succ_traj_ig, fail_traj_ig = [], []

    for r in results:
        traj_nlls, traj_igs = [], []
        for s in r["steps"]:
            if s["step"] == 0:
                continue
            target_nll = succ_nll if r["success"] else fail_nll
            target_ig = succ_ig if r["success"] else fail_ig

            if s["s_1_5b"] is not None:
                target_nll.append(s["s_1_5b"])
                traj_nlls.append(s["s_1_5b"])

            ig_val = s["info_gain"].get("-1", s["info_gain"].get(str(-1), 0))
            if ig_val > 0:
                target_ig.append(ig_val)
                traj_igs.append(ig_val)

            if s["orig_delta_s"] is not None:
                (succ_orig_ds if r["success"] else fail_orig_ds).append(s["orig_delta_s"])
            if s["orig_entropy"] is not None:
                (succ_orig_ent if r["success"] else fail_orig_ent).append(s["orig_entropy"])

        if traj_nlls:
            (succ_traj_nll if r["success"] else fail_traj_nll).append(np.mean(traj_nlls))
        if traj_igs:
            (succ_traj_ig if r["success"] else fail_traj_ig).append(np.mean(traj_igs))

    print(f"\nStep-level comparison (excl step 0):")
    if succ_nll and fail_nll:
        print(f"  s_1.5B (obs NLL): succ={np.mean(succ_nll):.4f}±{np.std(succ_nll):.4f} (n={len(succ_nll)})  "
              f"fail={np.mean(fail_nll):.4f}±{np.std(fail_nll):.4f} (n={len(fail_nll)})  "
              f"gap={np.mean(succ_nll)-np.mean(fail_nll):+.4f}")
    if succ_ig and fail_ig:
        print(f"  info_gain (L-1):  succ={np.mean(succ_ig):.1f}±{np.std(succ_ig):.1f} (n={len(succ_ig)})  "
              f"fail={np.mean(fail_ig):.1f}±{np.std(fail_ig):.1f} (n={len(fail_ig)})  "
              f"gap={np.mean(succ_ig)-np.mean(fail_ig):+.1f}")
    if succ_orig_ds and fail_orig_ds:
        print(f"  orig ΔS (7B):     succ={np.mean(succ_orig_ds):.4f}  fail={np.mean(fail_orig_ds):.4f}  "
              f"gap={np.mean(succ_orig_ds)-np.mean(fail_orig_ds):+.4f}")
    if succ_orig_ent and fail_orig_ent:
        print(f"  orig entropy (7B): succ={np.mean(succ_orig_ent):.4f}  fail={np.mean(fail_orig_ent):.4f}  "
              f"gap={np.mean(succ_orig_ent)-np.mean(fail_orig_ent):+.4f}")

    print(f"\nTraj-level comparison:")
    if succ_traj_nll and fail_traj_nll:
        t_val, p_val = stats.ttest_ind(succ_traj_nll, fail_traj_nll)
        print(f"  s_1.5B:     succ={np.mean(succ_traj_nll):.4f}  fail={np.mean(fail_traj_nll):.4f}  "
              f"t={t_val:.3f} p={p_val:.4f}")
    if succ_traj_ig and fail_traj_ig:
        t_val, p_val = stats.ttest_ind(succ_traj_ig, fail_traj_ig)
        print(f"  info_gain:  succ={np.mean(succ_traj_ig):.1f}  fail={np.mean(fail_traj_ig):.1f}  "
              f"t={t_val:.3f} p={p_val:.4f}")

    # Step-level correlations
    all_nll = succ_nll + fail_nll
    all_ig = succ_ig + fail_ig
    all_labels_nll = [1]*len(succ_nll) + [0]*len(fail_nll)
    all_labels_ig = [1]*len(succ_ig) + [0]*len(fail_ig)

    print(f"\nStep-level correlations:")
    if all_nll:
        print(f"  corr(s_1.5B, success) = {np.corrcoef(all_nll, all_labels_nll)[0,1]:.4f}")
    if all_ig:
        print(f"  corr(info_gain, success) = {np.corrcoef(all_ig, all_labels_ig)[0,1]:.4f}")
    if len(all_nll) == len(all_ig) and all_nll:
        print(f"  corr(s_1.5B, info_gain) = {np.corrcoef(all_nll, all_ig)[0,1]:.4f}")


def main():
    model_path = "/local_nvme/guanyiming/models/Qwen/Qwen2.5-1.5B-Instruct"
    layers = (-1,)
    out_dir = Path("ocar/analysis_results/model_1_5b")
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(model_path)

    # === ALFWorld ===
    alfworld_traj = "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"
    if os.path.exists(alfworld_traj):
        print("\n" + "="*60)
        print("ALFWorld (1.5B scorer)")
        print("="*60)
        with open(alfworld_traj) as f:
            data = json.load(f)
        trajs = data["trajectories"]
        fail_trajs = [t for t in trajs if not t["success"]]
        succ_trajs = [t for t in trajs if t["success"]]
        np.random.seed(42)
        succ_idx = list(np.random.choice(len(succ_trajs), size=min(20, len(succ_trajs)), replace=False))
        sample = fail_trajs + [succ_trajs[i] for i in succ_idx]
        np.random.shuffle(sample)
        print(f"Sampled {len(sample)} trajs ({len(fail_trajs)} fail + {min(20, len(succ_trajs))} succ)")

        results = process_trajectories(model, tokenizer, sample, layers)
        analyze(results, "ALFWorld 1.5B")
        with open(out_dir / "alfworld.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    # === WebShop (step 240 and 640) ===
    ws_base = "data/trajectories/grpo_observe_webshop_20260418_070828"
    for step in [240, 640]:
        traj_path = f"{ws_base}/global_step_{step}_observe_trajectories.json"
        if not os.path.exists(traj_path):
            print(f"Skipping WebShop step {step}")
            continue
        print(f"\n{'='*60}")
        print(f"WebShop step {step} (1.5B scorer)")
        print("="*60)
        with open(traj_path) as f:
            data = json.load(f)
        results = process_trajectories(model, tokenizer, data["trajectories"], layers)
        analyze(results, f"WebShop step {step} 1.5B")
        with open(out_dir / f"webshop_step{step}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\nDone!")


if __name__ == "__main__":
    main()
