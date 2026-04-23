"""Cross-dataset comparison: delta_s vs wm_gap signal strength (trajectory-level)."""
import json, glob, os
import numpy as np
from collections import defaultdict


def traj_level_corr(path: str):
    with open(path) as f:
        data = json.load(f)

    traj_means = defaultdict(list)
    for traj in data["trajectories"]:
        vals = defaultdict(list)
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            if ds is not None:
                vals["delta_s"].append(ds)
            e = step.get("entropy_mean")
            if e is not None and e != 0:
                vals["entropy"].append(e)
            wm = step.get("wm_s")
            wmb = step.get("wm_s_B")
            if wm and wm > 0:
                vals["wm_s"].append(wm)
                vals["wm_gap"].append((wmb or 0) - wm)

        for k in ["delta_s", "entropy", "wm_s", "wm_gap"]:
            traj_means[k].append(np.mean(vals[k]) if vals[k] else np.nan)
        traj_means["success"].append(float(traj["success"]))

    results = {}
    succ = np.array(traj_means["success"])
    n_succ = int(succ.sum())
    n_fail = len(succ) - n_succ
    for k in ["delta_s", "entropy", "wm_s", "wm_gap"]:
        arr = np.array(traj_means[k])
        mask = ~np.isnan(arr)
        if mask.sum() >= 10:
            r = np.corrcoef(arr[mask], succ[mask])[0, 1]
            s_mean = np.nanmean(arr[succ == 1]) if n_succ > 0 else np.nan
            f_mean = np.nanmean(arr[succ == 0]) if n_fail > 0 else np.nan
            s_std = np.nanstd(arr[succ == 1]) if n_succ > 1 else 1
            f_std = np.nanstd(arr[succ == 0]) if n_fail > 1 else 1
            pooled = np.sqrt((s_std**2 + f_std**2) / 2)
            cohen_d = (s_mean - f_mean) / pooled if pooled > 0 else 0
            results[k] = {"r": r, "s_mean": s_mean, "f_mean": f_mean, "d": cohen_d, "n": int(mask.sum())}
        else:
            results[k] = None
    return data["n_trajectories"], n_succ, n_fail, data["global_step"], results


# Collect all files
alfworld_files = sorted(glob.glob("checkpoints/grpo_observe_alfworld_1.5b_*/grpo_observe_*/global_step_*/observe_trajectories.json"))
alfworld_files += sorted(glob.glob("checkpoints/grpo_observe_alfworld_20260415*/grpo_observe_*/global_step_*/observe_trajectories.json"))
webshop_files = sorted(glob.glob("data/trajectories/grpo_observe_webshop_*/global_step_*_observe_trajectories.json"),
                       key=lambda x: int(x.split("global_step_")[1].split("_")[0]))

print("=" * 130)
print("CROSS-DATASET: delta_s vs wm_gap vs entropy — trajectory-level correlation with success")
print("=" * 130)

for dataset_name, files in [("ALFWorld 1.5B", alfworld_files), ("ALFWorld 7B", ["checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"]), ("WebShop", webshop_files)]:
    print(f"\n### {dataset_name} ###")
    print(f"{'step':>6s} | {'n':>4s} | {'SR':>5s} | {'r(Δs)':>8s} | {'d(Δs)':>8s} | {'r(ent)':>8s} | {'d(ent)':>8s} | {'r(wm_s)':>8s} | {'d(wm_s)':>8s} | {'r(gap)':>8s} | {'d(gap)':>8s}")
    print("-" * 110)
    for f in files:
        try:
            n, ns, nf, step, res = traj_level_corr(f)
        except FileNotFoundError:
            continue
        sr = ns / n if n > 0 else 0
        row = f"{step:6d} | {n:4d} | {sr:5.1%}"
        for k in ["delta_s", "entropy", "wm_s", "wm_gap"]:
            if res.get(k):
                row += f" | {res[k]['r']:8.3f} | {res[k]['d']:8.2f}"
            else:
                row += f" | {'n/a':>8s} | {'n/a':>8s}"
        print(row)
