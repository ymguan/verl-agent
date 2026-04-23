"""Analyze s_theta_mean (observation NLL) as a standalone causal signal.
Compare with delta_s, entropy, and wm_s across all datasets."""
import json, glob
import numpy as np


def load(path):
    with open(path) as f:
        return json.load(f)


def analyze(data):
    results = []
    for traj in data["trajectories"]:
        s_theta_vals, ds_vals, ent_vals, wm_vals = [], [], [], []
        for step in traj["steps"]:
            st = step.get("s_theta_mean")
            ds = step.get("delta_s_mean")
            ent = step.get("entropy_mean")
            wm = step.get("wm_s")
            if st is not None: s_theta_vals.append(st)
            if ds is not None: ds_vals.append(ds)
            if ent is not None: ent_vals.append(ent)
            if wm is not None and wm > 0: wm_vals.append(wm)
        results.append({
            "success": float(traj["success"]),
            "s_theta": np.mean(s_theta_vals) if s_theta_vals else None,
            "delta_s": np.mean(ds_vals) if ds_vals else None,
            "entropy": np.mean(ent_vals) if ent_vals else None,
            "wm_s": np.mean(wm_vals) if wm_vals else None,
        })

    def corr(key):
        vals = [(r["success"], r[key]) for r in results if r[key] is not None]
        if len(vals) < 5:
            return float('nan'), float('nan'), float('nan')
        s = np.array([v[0] for v in vals])
        d = np.array([v[1] for v in vals])
        if np.std(s) < 1e-8 or np.std(d) < 1e-8:
            return float('nan'), float('nan'), float('nan')
        r = np.corrcoef(s, d)[0, 1]
        s_mean = np.mean(d[s == 1]) if (s == 1).any() else float('nan')
        f_mean = np.mean(d[s == 0]) if (s == 0).any() else float('nan')
        return r, s_mean, f_mean

    return {k: corr(k) for k in ["s_theta", "delta_s", "entropy", "wm_s"]}


alfworld_1_5b = sorted(
    glob.glob("checkpoints/grpo_observe_alfworld_1.5b_*/grpo_observe_*/global_step_*/observe_trajectories.json"),
    key=lambda x: int(x.split("global_step_")[1].split("/")[0]))
alfworld_7b = ["checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"]
webshop = sorted(
    glob.glob("data/trajectories/grpo_observe_webshop_*/global_step_*_observe_trajectories.json"),
    key=lambda x: int(x.split("global_step_")[1].split("_")[0]))

for name, files in [("ALFWorld-1.5B", alfworld_1_5b), ("ALFWorld-7B", alfworld_7b), ("WebShop", webshop)]:
    print(f"\n{'#'*110}")
    print(f"### {name} ###")
    print(f"{'#'*110}")
    print(f"{'step':>6s} | {'SR':>5s} | {'r(s_θ)':>8s} {'S':>8s} {'F':>8s} | {'r(Δs)':>8s} {'S':>8s} {'F':>8s} | {'r(ent)':>8s} {'S':>8s} {'F':>8s} | {'r(wm)':>8s} {'S':>8s} {'F':>8s}")
    print("-" * 120)
    for f in files:
        try:
            data = load(f)
        except FileNotFoundError:
            continue
        step = data["global_step"]
        sr = data["n_success"] / data["n_trajectories"]
        res = analyze(data)
        parts = [f"{step:6d} | {sr:5.0%}"]
        for k in ["s_theta", "delta_s", "entropy", "wm_s"]:
            r, s_m, f_m = res[k]
            parts.append(f"{r:8.3f} {s_m:8.4f} {f_m:8.4f}")
        print(" | ".join(parts))
