"""Check if |delta_s| (absolute deviation from ref) is a stable signal."""
import json, glob
import numpy as np


def analyze(path: str):
    with open(path) as f:
        data = json.load(f)

    results = []
    for traj in data["trajectories"]:
        vals = []
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            if ds is not None:
                vals.append(abs(ds))
        if vals:
            results.append({"success": float(traj["success"]), "abs_ds": np.mean(vals),
                            "raw_ds": np.mean([step.get("delta_s_mean", 0) for step in traj["steps"]])})

    succ = np.array([r["success"] for r in results])
    abs_ds = np.array([r["abs_ds"] for r in results])
    raw_ds = np.array([r["raw_ds"] for r in results])
    s_mask = succ == 1
    f_mask = succ == 0
    if s_mask.sum() == 0 or f_mask.sum() == 0:
        return None

    r_abs = np.corrcoef(abs_ds, succ)[0, 1]
    r_raw = np.corrcoef(raw_ds, succ)[0, 1]

    return {
        "step": data["global_step"],
        "sr": s_mask.sum() / len(succ),
        "r_raw": r_raw,
        "r_abs": r_abs,
        "abs_s": np.mean(abs_ds[s_mask]),
        "abs_f": np.mean(abs_ds[f_mask]),
        "raw_s": np.mean(raw_ds[s_mask]),
        "raw_f": np.mean(raw_ds[f_mask]),
    }


alfworld = sorted(glob.glob("checkpoints/grpo_observe_alfworld_1.5b_*/grpo_observe_*/global_step_*/observe_trajectories.json"),
                  key=lambda x: int(x.split("global_step_")[1].split("/")[0]))
alfworld7b = ["checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"]
webshop = sorted(glob.glob("data/trajectories/grpo_observe_webshop_*/global_step_*_observe_trajectories.json"),
                 key=lambda x: int(x.split("global_step_")[1].split("_")[0]))

for name, files in [("ALFWorld-1.5B", alfworld), ("ALFWorld-7B", alfworld7b), ("WebShop", webshop)]:
    print(f"\n### {name} ###")
    print(f"{'step':>6s} | {'SR':>5s} | {'r(raw)':>8s} | {'r(|Δs|)':>8s} | {'|Δs| succ':>10s} | {'|Δs| fail':>10s} | {'direction':>10s}")
    print("-" * 85)
    for f in files:
        try:
            res = analyze(f)
        except FileNotFoundError:
            continue
        if res is None:
            continue
        direction = "S > F" if res["abs_s"] > res["abs_f"] else "S < F"
        print(f"{res['step']:6d} | {res['sr']:5.1%} | {res['r_raw']:8.3f} | {res['r_abs']:8.3f} | {res['abs_s']:10.4f} | {res['abs_f']:10.4f} | {direction:>10s}")
