"""Correlation analysis between step-level signals, per trajectory."""
import json
import numpy as np
from collections import defaultdict

def analyze_correlations(path: str):
    with open(path) as f:
        data = json.load(f)
    label = path.split("/")[-3]
    print(f"\n=== {label} step {data['global_step']} ===")

    signals = ["entropy_mean", "delta_s_mean", "wm_s", "wm_s_B", "wm_gap"]

    # Collect per-step vectors (skip steps with missing data)
    rows = []
    for traj in data["trajectories"]:
        for step in traj["steps"]:
            e = step.get("entropy_mean")
            ds = step.get("delta_s_mean")
            wm = step.get("wm_s")
            wmb = step.get("wm_s_B")
            # skip if key signals are missing/zero (GiGPO has no entropy/delta_s)
            if wm is None or wm == 0:
                continue
            wm_gap = (wmb or 0) - wm
            row = {
                "entropy_mean": e if e and e != 0 else np.nan,
                "delta_s_mean": ds if ds is not None else np.nan,
                "wm_s": wm,
                "wm_s_B": wmb or np.nan,
                "wm_gap": wm_gap,
                "success": float(traj["success"]),
            }
            rows.append(row)

    print(f"Total steps with wm data: {len(rows)}")

    # Build matrix
    keys = ["entropy_mean", "delta_s_mean", "wm_s", "wm_gap", "success"]
    available = []
    arrays = {}
    for k in keys:
        arr = np.array([r[k] for r in rows])
        if np.all(np.isnan(arr)):
            continue
        arrays[k] = arr
        available.append(k)

    print(f"\nPairwise Pearson correlations:")
    header = f"{'':>15s}" + "".join(f" | {k:>12s}" for k in available)
    print(header)
    print("-" * len(header))
    for k1 in available:
        row_str = f"{k1:>15s}"
        for k2 in available:
            a1, a2 = arrays[k1], arrays[k2]
            mask = ~(np.isnan(a1) | np.isnan(a2))
            if mask.sum() < 10:
                row_str += f" | {'n/a':>12s}"
            else:
                r = np.corrcoef(a1[mask], a2[mask])[0, 1]
                row_str += f" | {r:12.3f}"
        print(row_str)

    # Also: per-trajectory level correlations
    print(f"\n--- Per-trajectory mean correlations ---")
    traj_means = defaultdict(list)
    for traj in data["trajectories"]:
        vals = defaultdict(list)
        for step in traj["steps"]:
            wm = step.get("wm_s")
            wmb = step.get("wm_s_B")
            if wm is None or wm == 0:
                continue
            vals["wm_s"].append(wm)
            vals["wm_gap"].append((wmb or 0) - wm)
            e = step.get("entropy_mean")
            ds = step.get("delta_s_mean")
            if e and e != 0:
                vals["entropy_mean"].append(e)
            if ds is not None:
                vals["delta_s_mean"].append(ds)
        for k in ["entropy_mean", "delta_s_mean", "wm_s", "wm_gap"]:
            if vals[k]:
                traj_means[k].append(np.mean(vals[k]))
            else:
                traj_means[k].append(np.nan)
        traj_means["success"].append(float(traj["success"]))

    available2 = []
    arrays2 = {}
    for k in keys:
        arr = np.array(traj_means[k])
        if np.all(np.isnan(arr)):
            continue
        arrays2[k] = arr
        available2.append(k)

    header = f"{'':>15s}" + "".join(f" | {k:>12s}" for k in available2)
    print(header)
    print("-" * len(header))
    for k1 in available2:
        row_str = f"{k1:>15s}"
        for k2 in available2:
            a1, a2 = arrays2[k1], arrays2[k2]
            mask = ~(np.isnan(a1) | np.isnan(a2))
            if mask.sum() < 10:
                row_str += f" | {'n/a':>12s}"
            else:
                r = np.corrcoef(a1[mask], a2[mask])[0, 1]
                row_str += f" | {r:12.3f}"
        print(row_str)


if __name__ == "__main__":
    for p in [
        "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_60/observe_trajectories.json",
        "checkpoints/gigpo_observe_alfworld_1.5b_20260420_162642/gigpo_observe_qwen2.5_1.5b_seed0/global_step_80/observe_trajectories.json",
        "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json",
    ]:
        try:
            analyze_correlations(p)
        except FileNotFoundError:
            print(f"SKIP: {p}")
        print()
