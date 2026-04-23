"""Check if z-scored delta_s (per batch) fixes the direction instability."""
import json, glob
import numpy as np
from collections import defaultdict


def analyze(path: str):
    with open(path) as f:
        data = json.load(f)

    # Collect per-step delta_s for all trajectories in this batch
    all_ds = []
    traj_info = []  # (start_idx, n_steps, success)
    for traj in data["trajectories"]:
        steps_ds = []
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            if ds is not None:
                steps_ds.append(ds)
            else:
                steps_ds.append(np.nan)
        traj_info.append((len(all_ds), len(steps_ds), traj["success"]))
        all_ds.extend(steps_ds)

    all_ds = np.array(all_ds)
    valid = ~np.isnan(all_ds)
    if valid.sum() < 10:
        return None

    # Z-score within this batch
    mu = np.nanmean(all_ds)
    std = np.nanstd(all_ds)
    if std < 1e-8:
        return None
    z_ds = (all_ds - mu) / std

    # Per-trajectory: raw mean, z-scored mean
    results = []
    for start, n, succ in traj_info:
        raw = all_ds[start:start+n]
        z = z_ds[start:start+n]
        raw_valid = raw[~np.isnan(raw)]
        z_valid = z[~np.isnan(z)]
        if len(raw_valid) == 0:
            continue
        results.append({
            "success": float(succ),
            "raw_mean": np.mean(raw_valid),
            "z_mean": np.mean(z_valid),
        })

    succ = np.array([r["success"] for r in results])
    raw = np.array([r["raw_mean"] for r in results])
    z = np.array([r["z_mean"] for r in results])

    r_raw = np.corrcoef(raw, succ)[0, 1]
    r_z = np.corrcoef(z, succ)[0, 1]

    s_mask = succ == 1
    f_mask = succ == 0
    if s_mask.sum() == 0 or f_mask.sum() == 0:
        return None

    raw_s, raw_f = np.mean(raw[s_mask]), np.mean(raw[f_mask])
    z_s, z_f = np.mean(z[s_mask]), np.mean(z[f_mask])

    return {
        "step": data["global_step"],
        "n": len(results),
        "sr": s_mask.sum() / len(results),
        "r_raw": r_raw,
        "r_z": r_z,
        "raw_s": raw_s, "raw_f": raw_f,
        "z_s": z_s, "z_f": z_f,
        "batch_mu": mu, "batch_std": std,
    }


alfworld = sorted(glob.glob("checkpoints/grpo_observe_alfworld_1.5b_*/grpo_observe_*/global_step_*/observe_trajectories.json"),
                  key=lambda x: int(x.split("global_step_")[1].split("/")[0]))
alfworld7b = ["checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"]
webshop = sorted(glob.glob("data/trajectories/grpo_observe_webshop_*/global_step_*_observe_trajectories.json"),
                 key=lambda x: int(x.split("global_step_")[1].split("_")[0]))

for name, files in [("ALFWorld-1.5B", alfworld), ("ALFWorld-7B", alfworld7b), ("WebShop", webshop)]:
    print(f"\n### {name} ###")
    print(f"{'step':>6s} | {'SR':>5s} | {'r_raw':>7s} | {'r_zscore':>8s} | {'raw(S)':>8s} | {'raw(F)':>8s} | {'z(S)':>8s} | {'z(F)':>8s} | {'μ_batch':>8s} | {'σ_batch':>8s}")
    print("-" * 105)
    for f in files:
        try:
            res = analyze(f)
        except FileNotFoundError:
            continue
        if res is None:
            continue
        print(f"{res['step']:6d} | {res['sr']:5.1%} | {res['r_raw']:7.3f} | {res['r_z']:8.3f} | {res['raw_s']:8.4f} | {res['raw_f']:8.4f} | {res['z_s']:8.3f} | {res['z_f']:8.3f} | {res['batch_mu']:8.4f} | {res['batch_std']:8.4f}")
