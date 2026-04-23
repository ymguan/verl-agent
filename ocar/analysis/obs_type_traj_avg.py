"""Analyze signals by obs type — trajectory-level averaging to avoid length bias."""
import json
import numpy as np
from collections import defaultdict


def classify_observation(obs: str, prev_observations: list[str]) -> str:
    obs_lower = obs.lower().strip()
    nothing_patterns = ["nothing happens", "nothing to", "can't", "don't see", "no more"]
    for p in nothing_patterns:
        if p in obs_lower:
            return "nothing_happens"
    state_change_patterns = [
        "you pick up", "you put", "you open", "you close",
        "you turn on", "you turn off", "you heat", "you cool",
        "you clean", "you slice", "you use",
    ]
    for p in state_change_patterns:
        if p in obs_lower:
            return "state_change"
    if obs in prev_observations:
        return "revisit"
    return "new_location"


def analyze_file(path: str):
    with open(path) as f:
        data = json.load(f)

    label = path.split("/")[-3]
    print(f"\n=== {label} step {data['global_step']} ===")
    print(f"Trajectories: {data['n_trajectories']} (success={data['n_success']}, fail={data['n_failure']})")

    metrics = ["s_theta_mean", "delta_s_mean", "entropy_mean", "wm_s", "wm_s_B"]
    obs_types = ["state_change", "new_location", "revisit", "nothing_happens"]

    # Per-trajectory: compute mean of each metric for each obs_type
    # traj_stats[outcome][obs_type] = list of per-traj means for each metric
    traj_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    traj_obs_frac = defaultdict(list)  # fraction of steps in each obs type

    for traj in data["trajectories"]:
        outcome = "success" if traj["success"] else "failure"
        prev_obs = []
        # Collect steps by obs type within this trajectory
        by_type = defaultdict(lambda: defaultdict(list))
        total_steps = len(traj["steps"])

        for step in traj["steps"]:
            obs = step["observation"]
            ot = classify_observation(obs, prev_obs)
            prev_obs.append(obs)
            for m in metrics:
                val = step.get(m)
                if val is not None and val != 0.0:
                    by_type[ot][m].append(val)
            by_type[ot]["_count"].append(1)

        # Now average within trajectory
        for ot in obs_types:
            if by_type[ot]["_count"]:
                frac = len(by_type[ot]["_count"]) / total_steps
                traj_obs_frac[outcome + "/" + ot].append(frac)
                for m in metrics:
                    if by_type[ot][m]:
                        traj_stats[outcome][ot][m].append(np.mean(by_type[ot][m]))

    # Print results
    header = f"{'':30s} | {'s_theta':>10s} | {'delta_s':>10s} | {'entropy':>10s} | {'wm_s':>10s} | {'wm_s_B':>10s} | {'wm_gap':>10s} | {'n_traj':>6s} | {'obs_frac':>8s}"
    print(header)
    print("-" * len(header))

    for ot in obs_types:
        for outcome in ["success", "failure"]:
            vals = traj_stats[outcome][ot]
            n = len(vals.get("wm_s", vals.get("s_theta_mean", [])))
            if n == 0:
                continue
            row = f"  {outcome:8s} / {ot:18s}"
            computed = {}
            for m in metrics:
                arr = vals.get(m, [])
                if arr:
                    mu = np.mean(arr)
                    computed[m] = mu
                    row += f" | {mu:10.4f}"
                else:
                    computed[m] = None
                    row += f" | {'n/a':>10s}"
            wm_gap = (computed.get("wm_s_B") or 0) - (computed.get("wm_s") or 0)
            row += f" | {wm_gap:10.4f}"
            row += f" | {n:6d}"
            frac_key = outcome + "/" + ot
            avg_frac = np.mean(traj_obs_frac[frac_key]) if traj_obs_frac[frac_key] else 0
            row += f" | {avg_frac:8.1%}"
            print(row)
        print()

    # Difference table
    print("--- Δ(success - failure), trajectory-averaged ---")
    print(f"{'obs_type':20s} | {'Δ s_theta':>10s} | {'Δ delta_s':>10s} | {'Δ entropy':>10s} | {'Δ wm_s':>10s} | {'Δ wm_gap':>10s} | {'effect_d':>10s}")
    print("-" * 100)
    for ot in obs_types:
        s_vals = traj_stats["success"][ot]
        f_vals = traj_stats["failure"][ot]
        if not s_vals or not f_vals:
            continue
        row = f"{ot:20s}"
        for m in ["s_theta_mean", "delta_s_mean", "entropy_mean", "wm_s"]:
            sa, fa = s_vals.get(m, []), f_vals.get(m, [])
            if sa and fa:
                diff = np.mean(sa) - np.mean(fa)
                row += f" | {diff:10.4f}"
            else:
                row += f" | {'n/a':>10s}"
        # wm_gap diff
        s_gap_arr = [b - a for a, b in zip(s_vals.get("wm_s", []), s_vals.get("wm_s_B", [])) if True] if s_vals.get("wm_s") and s_vals.get("wm_s_B") else []
        f_gap_arr = [b - a for a, b in zip(f_vals.get("wm_s", []), f_vals.get("wm_s_B", [])) if True] if f_vals.get("wm_s") and f_vals.get("wm_s_B") else []
        if s_gap_arr and f_gap_arr:
            row += f" | {np.mean(s_gap_arr) - np.mean(f_gap_arr):10.4f}"
            # Cohen's d on wm_s as example
            pooled_std = np.sqrt((np.var(s_vals["wm_s"]) + np.var(f_vals["wm_s"])) / 2) if s_vals.get("wm_s") and f_vals.get("wm_s") else 1
            d = (np.mean(s_vals["wm_s"]) - np.mean(f_vals["wm_s"])) / pooled_std if pooled_std > 0 else 0
            row += f" | {d:10.2f}"
        else:
            row += f" | {'n/a':>10s} | {'n/a':>10s}"
        print(row)


if __name__ == "__main__":
    paths = [
        "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_60/observe_trajectories.json",
        "checkpoints/gigpo_observe_alfworld_1.5b_20260420_162642/gigpo_observe_qwen2.5_1.5b_seed0/global_step_80/observe_trajectories.json",
        "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json",
    ]
    for p in paths:
        try:
            analyze_file(p)
            print("\n" + "=" * 120 + "\n")
        except FileNotFoundError:
            print(f"SKIP: {p}\n")
