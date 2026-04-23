"""Analyze surprise/entropy signals by observation type (state_change, new_location, revisit, nothing_happens)."""
import json, re, sys
from collections import defaultdict
import numpy as np

def classify_observation(obs: str, prev_observations: list[str]) -> str:
    obs_lower = obs.lower().strip()
    nothing_patterns = [
        "nothing happens",
        "nothing to",
        "can't",
        "don't see",
        "no more",
    ]
    for p in nothing_patterns:
        if p in obs_lower:
            return "nothing_happens"

    state_change_patterns = [
        "you pick up", "you put", "you open", "you close",
        "you turn on", "you turn off", "you heat", "you cool",
        "you clean", "you slice", "you use",
        "is now in/on",
    ]
    for p in state_change_patterns:
        if p in obs_lower:
            return "state_change"

    if obs in prev_observations:
        return "revisit"

    if "you arrive at" in obs_lower or "on the" in obs_lower:
        return "new_location"

    if obs in prev_observations:
        return "revisit"

    return "new_location"


def analyze_file(path: str):
    with open(path) as f:
        data = json.load(f)

    print(f"=== {path.split('/')[-3]} step {data['global_step']} ===")
    print(f"Trajectories: {data['n_trajectories']} (success={data['n_success']}, fail={data['n_failure']})")

    records = defaultdict(lambda: defaultdict(list))

    for traj in data["trajectories"]:
        success = traj["success"]
        outcome = "success" if success else "failure"
        prev_obs = []

        for step in traj["steps"]:
            obs = step["observation"]
            obs_type = classify_observation(obs, prev_obs)
            prev_obs.append(obs)

            key = (outcome, obs_type)
            for metric in ["s_theta_mean", "delta_s_mean", "entropy_mean", "wm_s", "wm_s_B"]:
                val = step.get(metric)
                if val is not None:
                    records[key][metric].append(val)
            records[key]["count"].append(1)

    obs_types = ["state_change", "new_location", "revisit", "nothing_happens"]
    metrics = ["s_theta_mean", "delta_s_mean", "entropy_mean", "wm_s", "wm_s_B"]

    print(f"\n{'':30s} | {'s_theta':>12s} | {'delta_s':>12s} | {'entropy':>12s} | {'wm_s':>12s} | {'wm_s_B':>12s} | {'wm_gap':>12s} | {'count':>6s}")
    print("-" * 130)

    for ot in obs_types:
        for outcome in ["success", "failure"]:
            key = (outcome, ot)
            n = len(records[key]["count"])
            if n == 0:
                continue
            row = f"  {outcome:8s} / {ot:18s}"
            vals = {}
            for m in metrics:
                arr = records[key][m]
                if arr:
                    vals[m] = np.mean(arr)
                    row += f" | {np.mean(arr):12.4f}"
                else:
                    vals[m] = 0
                    row += f" | {'n/a':>12s}"
            wm_gap = vals.get("wm_s_B", 0) - vals.get("wm_s", 0)
            row += f" | {wm_gap:12.4f}"
            row += f" | {n:6d}"
            print(row)
        print()

    # Also print the delta between success and failure for each obs type
    print("\n--- Signal difference (success - failure) by obs type ---")
    print(f"{'obs_type':20s} | {'Δ s_theta':>12s} | {'Δ delta_s':>12s} | {'Δ entropy':>12s} | {'Δ wm_s':>12s} | {'Δ wm_gap':>12s}")
    print("-" * 95)
    for ot in obs_types:
        sk = ("success", ot)
        fk = ("failure", ot)
        sn = len(records[sk]["count"])
        fn = len(records[fk]["count"])
        if sn == 0 or fn == 0:
            continue
        row = f"{ot:20s}"
        for m in ["s_theta_mean", "delta_s_mean", "entropy_mean", "wm_s"]:
            diff = np.mean(records[sk][m]) - np.mean(records[fk][m])
            row += f" | {diff:12.4f}"
        s_gap = np.mean(records[sk]["wm_s_B"]) - np.mean(records[sk]["wm_s"])
        f_gap = np.mean(records[fk]["wm_s_B"]) - np.mean(records[fk]["wm_s"])
        row += f" | {s_gap - f_gap:12.4f}"
        print(row)


if __name__ == "__main__":
    paths = [
        "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_20/observe_trajectories.json",
        "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_40/observe_trajectories.json",
        "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_60/observe_trajectories.json",
        "checkpoints/gigpo_observe_alfworld_1.5b_20260420_162642/gigpo_observe_qwen2.5_1.5b_seed0/global_step_20/observe_trajectories.json",
        "checkpoints/gigpo_observe_alfworld_1.5b_20260420_162642/gigpo_observe_qwen2.5_1.5b_seed0/global_step_40/observe_trajectories.json",
        "checkpoints/gigpo_observe_alfworld_1.5b_20260420_162642/gigpo_observe_qwen2.5_1.5b_seed0/global_step_60/observe_trajectories.json",
        "checkpoints/gigpo_observe_alfworld_1.5b_20260420_162642/gigpo_observe_qwen2.5_1.5b_seed0/global_step_80/observe_trajectories.json",
        "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json",
    ]
    for p in paths:
        try:
            analyze_file(p)
            print("\n" + "=" * 130 + "\n")
        except FileNotFoundError:
            print(f"SKIP: {p} not found\n")
