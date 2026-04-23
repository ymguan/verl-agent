"""
Hidden State Information Gain Analysis

For each trajectory step, compute:
  h_pre  = hidden state at the last token BEFORE observation
  h_post = hidden state at the last token OF observation (before action)
  info_gain = ||h_post - h_pre||_2

Then measure succ/fail discriminability at step level and traj level,
and compare with delta_s and entropy signals.

Uses the merged HF checkpoint (step 150) for ALFWorld,
and the base model (Qwen2.5-7B-Instruct) for WebShop analysis.
"""

import json
import os
import sys
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

def load_model(model_path, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_hidden_states(model, tokenizer, trajectories, max_trajs=None, layers=(-1,)):
    """Extract hidden state info gain for each step in each trajectory.

    For each step t:
      - Build context = obs_0 + act_0 + obs_1 + act_1 + ... + obs_t + act_t
      - h_pre = hidden at last token of (... + act_{t-1}), i.e. just before obs_t
      - h_post = hidden at last token of obs_t, i.e. just before act_t
      - info_gain = ||h_post - h_pre||

    Returns list of dicts per trajectory.
    """
    results = []
    trajs = trajectories[:max_trajs] if max_trajs else trajectories

    for ti, traj in enumerate(trajs):
        if ti % 10 == 0:
            print(f"  Processing trajectory {ti+1}/{len(trajs)}...")

        steps = traj["steps"]
        traj_result = {
            "success": traj["success"],
            "n_steps": len(steps),
            "steps": [],
        }

        context_parts = []

        for si, step in enumerate(steps):
            obs = step["observation"]
            action = step["action"]

            # Build pre-obs context (everything before this observation)
            pre_obs_text = "".join(context_parts)

            # Build post-obs context (including this observation)
            post_obs_text = pre_obs_text + obs

            step_info = {
                "step": si,
                "delta_s_mean": step.get("delta_s_mean", None),
                "entropy_mean": step.get("entropy_mean", None),
                "s_theta_mean": step.get("s_theta_mean", None),
                "advantage": step.get("advantage", None),
                "step_reward": step.get("step_reward", None),
                "obs_n_tokens": step.get("obs_n_tokens", None),
            }

            if si == 0:
                # Step 0: no pre-obs context, skip info gain
                step_info["info_gain"] = {layer: 0.0 for layer in layers}
                step_info["h_pre_norm"] = {layer: 0.0 for layer in layers}
                step_info["h_post_norm"] = {layer: 0.0 for layer in layers}
                step_info["cosine_sim"] = {layer: 1.0 for layer in layers}
            else:
                with torch.no_grad():
                    # Tokenize both contexts
                    pre_ids = tokenizer.encode(pre_obs_text, return_tensors="pt",
                                               add_special_tokens=True, truncation=True,
                                               max_length=4096).to(model.device)
                    post_ids = tokenizer.encode(post_obs_text, return_tensors="pt",
                                                add_special_tokens=True, truncation=True,
                                                max_length=4096).to(model.device)

                    if pre_ids.shape[1] == 0 or post_ids.shape[1] == 0:
                        step_info["info_gain"] = {layer: 0.0 for layer in layers}
                        step_info["h_pre_norm"] = {layer: 0.0 for layer in layers}
                        step_info["h_post_norm"] = {layer: 0.0 for layer in layers}
                        step_info["cosine_sim"] = {layer: 1.0 for layer in layers}
                    else:
                        out_pre = model(pre_ids, output_hidden_states=True)
                        out_post = model(post_ids, output_hidden_states=True)

                        ig = {}
                        h_pre_norms = {}
                        h_post_norms = {}
                        cosines = {}

                        for layer in layers:
                            h_pre = out_pre.hidden_states[layer][0, -1].float()
                            h_post = out_post.hidden_states[layer][0, -1].float()

                            diff = h_post - h_pre
                            ig[layer] = diff.norm().item()
                            h_pre_norms[layer] = h_pre.norm().item()
                            h_post_norms[layer] = h_post.norm().item()
                            cos = torch.nn.functional.cosine_similarity(
                                h_pre.unsqueeze(0), h_post.unsqueeze(0)
                            ).item()
                            cosines[layer] = cos

                        step_info["info_gain"] = ig
                        step_info["h_pre_norm"] = h_pre_norms
                        step_info["h_post_norm"] = h_post_norms
                        step_info["cosine_sim"] = cosines

            traj_result["steps"].append(step_info)

            # Extend context for next step
            context_parts.append(obs)
            context_parts.append(action)

        results.append(traj_result)

    return results


def analyze_results(results, label=""):
    """Compute succ/fail discriminability for info_gain vs delta_s vs entropy."""
    print(f"\n{'='*60}")
    print(f"Analysis: {label}")
    print(f"{'='*60}")

    n_succ = sum(1 for r in results if r["success"])
    n_fail = sum(1 for r in results if not r["success"])
    print(f"Trajectories: {len(results)} total, {n_succ} succ, {n_fail} fail")

    if n_succ == 0 or n_fail == 0:
        print("Cannot compute succ/fail comparison (one group empty)")
        return

    # Collect step-level signals
    layers = list(results[0]["steps"][1]["info_gain"].keys()) if len(results[0]["steps"]) > 1 else [-1]

    for layer in layers:
        succ_ig, fail_ig = [], []
        succ_ds, fail_ds = [], []
        succ_ent, fail_ent = [], []
        succ_cos, fail_cos = [], []

        # Traj-level means
        succ_traj_ig, fail_traj_ig = [], []
        succ_traj_ds, fail_traj_ds = [], []
        succ_traj_ent, fail_traj_ent = [], []

        for r in results:
            traj_igs = []
            traj_ds = []
            traj_ent = []

            for s in r["steps"]:
                if s["step"] == 0:
                    continue
                ig = s["info_gain"].get(layer, 0)
                ds = s.get("delta_s_mean")
                ent = s.get("entropy_mean")
                cos = s["cosine_sim"].get(layer, 1.0)

                if r["success"]:
                    succ_ig.append(ig)
                    if ds is not None: succ_ds.append(ds)
                    if ent is not None: succ_ent.append(ent)
                    succ_cos.append(cos)
                else:
                    fail_ig.append(ig)
                    if ds is not None: fail_ds.append(ds)
                    if ent is not None: fail_ent.append(ent)
                    fail_cos.append(cos)

                traj_igs.append(ig)
                if ds is not None: traj_ds.append(ds)
                if ent is not None: traj_ent.append(ent)

            if traj_igs:
                target = succ_traj_ig if r["success"] else fail_traj_ig
                target.append(np.mean(traj_igs))
            if traj_ds:
                target = succ_traj_ds if r["success"] else fail_traj_ds
                target.append(np.mean(traj_ds))
            if traj_ent:
                target = succ_traj_ent if r["success"] else fail_traj_ent
                target.append(np.mean(traj_ent))

        print(f"\n--- Layer {layer} ---")
        print(f"\nStep-level comparison (excluding step 0):")
        print(f"  info_gain:  succ={np.mean(succ_ig):.4f}±{np.std(succ_ig):.4f} (n={len(succ_ig)})  "
              f"fail={np.mean(fail_ig):.4f}±{np.std(fail_ig):.4f} (n={len(fail_ig)})  "
              f"gap={np.mean(succ_ig)-np.mean(fail_ig):+.4f}")
        print(f"  cosine_sim: succ={np.mean(succ_cos):.4f}  fail={np.mean(fail_cos):.4f}  "
              f"gap={np.mean(succ_cos)-np.mean(fail_cos):+.4f}")
        if succ_ds and fail_ds:
            print(f"  delta_s:    succ={np.mean(succ_ds):.4f}  fail={np.mean(fail_ds):.4f}  "
                  f"gap={np.mean(succ_ds)-np.mean(fail_ds):+.4f}")
        if succ_ent and fail_ent:
            print(f"  entropy:    succ={np.mean(succ_ent):.4f}  fail={np.mean(fail_ent):.4f}  "
                  f"gap={np.mean(succ_ent)-np.mean(fail_ent):+.4f}")

        print(f"\nTraj-level comparison:")
        if succ_traj_ig and fail_traj_ig:
            from scipy import stats
            t_ig, p_ig = stats.ttest_ind(succ_traj_ig, fail_traj_ig)
            print(f"  info_gain:  succ={np.mean(succ_traj_ig):.4f}  fail={np.mean(fail_traj_ig):.4f}  "
                  f"t={t_ig:.3f} p={p_ig:.4f}")
        if succ_traj_ds and fail_traj_ds:
            t_ds, p_ds = stats.ttest_ind(succ_traj_ds, fail_traj_ds)
            print(f"  delta_s:    succ={np.mean(succ_traj_ds):.4f}  fail={np.mean(fail_traj_ds):.4f}  "
                  f"t={t_ds:.3f} p={p_ds:.4f}")
        if succ_traj_ent and fail_traj_ent:
            t_ent, p_ent = stats.ttest_ind(succ_traj_ent, fail_traj_ent)
            print(f"  entropy:    succ={np.mean(succ_traj_ent):.4f}  fail={np.mean(fail_traj_ent):.4f}  "
                  f"t={t_ent:.3f} p={p_ent:.4f}")

        # Correlation between info_gain and other signals (step level)
        all_ig = succ_ig + fail_ig
        all_ds = succ_ds + fail_ds
        all_ent = succ_ent + fail_ent
        all_labels = [1]*len(succ_ig) + [0]*len(fail_ig)

        print(f"\nStep-level correlations:")
        if len(all_ig) == len(all_ds) and all_ds:
            r_ig_ds = np.corrcoef(all_ig, all_ds)[0, 1]
            print(f"  corr(info_gain, delta_s) = {r_ig_ds:.4f}")
        if len(all_ig) == len(all_ent) and all_ent:
            r_ig_ent = np.corrcoef(all_ig, all_ent)[0, 1]
            print(f"  corr(info_gain, entropy) = {r_ig_ent:.4f}")
        if all_ig and all_labels:
            r_ig_succ = np.corrcoef(all_ig, all_labels)[0, 1]
            print(f"  corr(info_gain, success) = {r_ig_succ:.4f}")


def main():
    # Use multiple layers: middle, second-to-last, last
    layers_to_check = (-1, -16, -28)

    out_dir = Path("ocar/analysis_results/hidden_state")
    out_dir.mkdir(parents=True, exist_ok=True)

    # === ALFWorld: use merged step 150 checkpoint ===
    alfworld_model_path = "checkpoints/merged_hf/step_150"
    alfworld_traj_path = "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"

    if os.path.exists(alfworld_model_path) and os.path.exists(alfworld_traj_path):
        print("\n" + "="*60)
        print("ALFWorld Analysis (step 150 checkpoint)")
        print("="*60)

        model, tokenizer = load_model(alfworld_model_path)

        with open(alfworld_traj_path) as f:
            data = json.load(f)

        # Sample: all fail (11) + 20 random succ
        trajs = data["trajectories"]
        fail_trajs = [t for t in trajs if not t["success"]]
        succ_trajs = [t for t in trajs if t["success"]]

        np.random.seed(42)
        succ_sample = list(np.random.choice(len(succ_trajs), size=min(20, len(succ_trajs)), replace=False))
        sample_trajs = fail_trajs + [succ_trajs[i] for i in succ_sample]
        np.random.shuffle(sample_trajs)

        print(f"Sampled {len(sample_trajs)} trajectories ({len(fail_trajs)} fail + {min(20, len(succ_trajs))} succ)")

        results = extract_hidden_states(model, tokenizer, sample_trajs, layers=layers_to_check)
        analyze_results(results, label="ALFWorld step 150")

        with open(out_dir / "alfworld_step150.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        del model
        torch.cuda.empty_cache()

    # === WebShop: use base model (Qwen2.5-7B-Instruct) ===
    webshop_base_path = "/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct"
    webshop_traj_base = "data/trajectories/grpo_observe_webshop_20260418_070828"

    # Check if base model exists, try alternative paths
    if not os.path.exists(webshop_base_path):
        alt_paths = [
            os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct"),
            "Qwen/Qwen2.5-7B-Instruct",
        ]
        for p in alt_paths:
            if os.path.exists(p):
                webshop_base_path = p
                break

    # Use step 240 (best succ/fail balance) and step 640 (late training)
    for step in [240, 640]:
        traj_path = f"{webshop_traj_base}/global_step_{step}_observe_trajectories.json"
        if not os.path.exists(traj_path):
            print(f"Skipping WebShop step {step}: {traj_path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"WebShop Analysis (step {step}, using base model)")
        print("="*60)

        if 'model' not in dir() or model is None:
            model, tokenizer = load_model(webshop_base_path)

        with open(traj_path) as f:
            data = json.load(f)

        trajs = data["trajectories"]
        print(f"All {len(trajs)} trajectories")

        results = extract_hidden_states(model, tokenizer, trajs, layers=layers_to_check)
        analyze_results(results, label=f"WebShop step {step}")

        with open(out_dir / f"webshop_step{step}.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

    print("\n\nDone! Results saved to ocar/analysis_results/hidden_state/")


if __name__ == "__main__":
    main()
