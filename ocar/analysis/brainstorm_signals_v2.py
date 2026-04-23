"""
Re-analyze D3 (surprise-gated credit) using raw delta_s and wm_gap as the gate,
instead of |delta_s|.

Gate logic: split steps by median of the gate signal.
- "high gate" steps = exploratory
- "low gate" steps = decisive
Check which group has better correlation with trajectory outcome.
"""
import json, glob
import numpy as np


def load_trajs(path):
    with open(path) as f:
        return json.load(f)


def gate_analysis(data, gate_name, gate_fn):
    """Split steps into high/low by median of gate signal.
    For each group, compute trajectory-level mean delta_s and correlate with success."""
    all_gate_vals = []
    for traj in data["trajectories"]:
        for step in traj["steps"]:
            v = gate_fn(step)
            if v is not None:
                all_gate_vals.append(v)
    if len(all_gate_vals) < 10:
        return None
    median_gate = np.median(all_gate_vals)

    results_low = []
    results_high = []
    for traj in data["trajectories"]:
        low_ds, high_ds = [], []
        low_ent, high_ent = [], []
        low_gate, high_gate = [], []
        for step in traj["steps"]:
            v = gate_fn(step)
            ds = step.get("delta_s_mean")
            ent = step.get("entropy_mean")
            if v is None or ds is None:
                continue
            if v <= median_gate:
                low_ds.append(ds)
                if ent is not None: low_ent.append(ent)
                low_gate.append(v)
            else:
                high_ds.append(ds)
                if ent is not None: high_ent.append(ent)
                high_gate.append(v)

        succ = float(traj["success"])
        if low_ds:
            results_low.append({"success": succ, "mean_ds": np.mean(low_ds),
                                "mean_ent": np.mean(low_ent) if low_ent else None,
                                "mean_gate": np.mean(low_gate), "n": len(low_ds)})
        if high_ds:
            results_high.append({"success": succ, "mean_ds": np.mean(high_ds),
                                 "mean_ent": np.mean(high_ent) if high_ent else None,
                                 "mean_gate": np.mean(high_gate), "n": len(high_ds)})

    def corr(results, key="mean_ds"):
        vals = [(r["success"], r[key]) for r in results if r[key] is not None]
        if len(vals) < 5:
            return float('nan')
        s = np.array([v[0] for v in vals])
        d = np.array([v[1] for v in vals])
        if np.std(s) < 1e-8 or np.std(d) < 1e-8:
            return float('nan')
        return np.corrcoef(s, d)[0, 1]

    def mean_by_outcome(results, key="mean_ds"):
        succ_vals = [r[key] for r in results if r["success"] == 1.0 and r[key] is not None]
        fail_vals = [r[key] for r in results if r["success"] == 0.0 and r[key] is not None]
        return (np.mean(succ_vals) if succ_vals else float('nan'),
                np.mean(fail_vals) if fail_vals else float('nan'))

    low_s, low_f = mean_by_outcome(results_low)
    high_s, high_f = mean_by_outcome(results_high)

    return {
        "median_gate": median_gate,
        "r_low_ds": corr(results_low, "mean_ds"),
        "r_high_ds": corr(results_high, "mean_ds"),
        "r_low_ent": corr(results_low, "mean_ent"),
        "r_high_ent": corr(results_high, "mean_ent"),
        "low_ds_S": low_s, "low_ds_F": low_f,
        "high_ds_S": high_s, "high_ds_F": high_f,
        "n_low": np.mean([r["n"] for r in results_low]) if results_low else 0,
        "n_high": np.mean([r["n"] for r in results_high]) if results_high else 0,
    }


def analyze_raw_signals(data):
    """Also show raw trajectory-level correlation of delta_s, entropy, wm_gap with success."""
    results = []
    for traj in data["trajectories"]:
        ds_vals, ent_vals, wm_gap_vals = [], [], []
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            ent = step.get("entropy_mean")
            wm_s = step.get("wm_s")
            wm_b = step.get("wm_s_B")
            if ds is not None: ds_vals.append(ds)
            if ent is not None: ent_vals.append(ent)
            if wm_s is not None and wm_b is not None:
                wm_gap_vals.append(wm_b - wm_s)
        results.append({
            "success": float(traj["success"]),
            "ds": np.mean(ds_vals) if ds_vals else None,
            "ent": np.mean(ent_vals) if ent_vals else None,
            "wm_gap": np.mean(wm_gap_vals) if wm_gap_vals else None,
        })

    def corr(key):
        vals = [(r["success"], r[key]) for r in results if r[key] is not None]
        if len(vals) < 5:
            return float('nan')
        s = np.array([v[0] for v in vals])
        d = np.array([v[1] for v in vals])
        if np.std(s) < 1e-8 or np.std(d) < 1e-8:
            return float('nan')
        return np.corrcoef(s, d)[0, 1]

    return {"r_ds": corr("ds"), "r_ent": corr("ent"), "r_wm_gap": corr("wm_gap")}


alfworld_1_5b = sorted(
    glob.glob("checkpoints/grpo_observe_alfworld_1.5b_*/grpo_observe_*/global_step_*/observe_trajectories.json"),
    key=lambda x: int(x.split("global_step_")[1].split("/")[0]))
alfworld_7b = ["checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"]
webshop = sorted(
    glob.glob("data/trajectories/grpo_observe_webshop_*/global_step_*_observe_trajectories.json"),
    key=lambda x: int(x.split("global_step_")[1].split("_")[0]))

datasets = [("ALFWorld-1.5B", alfworld_1_5b), ("ALFWorld-7B", alfworld_7b), ("WebShop", webshop)]

for name, files in datasets:
    print(f"\n{'#'*100}")
    print(f"### {name} ###")
    print(f"{'#'*100}")

    for f in files:
        try:
            data = load_trajs(f)
        except FileNotFoundError:
            continue

        step = data["global_step"]
        sr = data["n_success"] / data["n_trajectories"]
        print(f"\n--- step {step} | SR={data['n_success']}/{data['n_trajectories']} ({sr:.0%}) ---")

        # Raw correlations
        raw = analyze_raw_signals(data)
        print(f"  [原始相关性] r(Δs)={raw['r_ds']:.3f}  r(entropy)={raw['r_ent']:.3f}  r(wm_gap)={raw['r_wm_gap']:.3f}")

        # Gate by raw delta_s (not |delta_s|)
        r_ds = gate_analysis(data, "delta_s", lambda s: s.get("delta_s_mean"))
        if r_ds:
            print(f"  [D3 门控 by Δs] median={r_ds['median_gate']:.4f}")
            print(f"    低Δs半(更surprised): r(Δs)={r_ds['r_low_ds']:.3f}, r(ent)={r_ds['r_low_ent']:.3f}  "
                  f"Δs: S={r_ds['low_ds_S']:.4f} F={r_ds['low_ds_F']:.4f}  ({r_ds['n_low']:.0f} steps)")
            print(f"    高Δs半(更confident): r(Δs)={r_ds['r_high_ds']:.3f}, r(ent)={r_ds['r_high_ent']:.3f}  "
                  f"Δs: S={r_ds['high_ds_S']:.4f} F={r_ds['high_ds_F']:.4f}  ({r_ds['n_high']:.0f} steps)")

        # Gate by entropy
        r_ent = gate_analysis(data, "entropy", lambda s: s.get("entropy_mean"))
        if r_ent:
            print(f"  [D3 门控 by entropy] median={r_ent['median_gate']:.4f}")
            print(f"    低entropy半(更确定): r(Δs)={r_ent['r_low_ds']:.3f}, r(ent)={r_ent['r_low_ent']:.3f}  "
                  f"Δs: S={r_ent['low_ds_S']:.4f} F={r_ent['low_ds_F']:.4f}  ({r_ent['n_low']:.0f} steps)")
            print(f"    高entropy半(更不确定): r(Δs)={r_ent['r_high_ds']:.3f}, r(ent)={r_ent['r_high_ent']:.3f}  "
                  f"Δs: S={r_ent['high_ds_S']:.4f} F={r_ent['high_ds_F']:.4f}  ({r_ent['n_high']:.0f} steps)")

        # Gate by wm_gap (if available)
        def get_wm_gap(step):
            wm_s = step.get("wm_s")
            wm_b = step.get("wm_s_B")
            if wm_s is not None and wm_b is not None:
                return wm_b - wm_s
            return None
        r_wm = gate_analysis(data, "wm_gap", get_wm_gap)
        if r_wm:
            print(f"  [D3 门控 by wm_gap] median={r_wm['median_gate']:.4f}")
            print(f"    低wm_gap半(state有用): r(Δs)={r_wm['r_low_ds']:.3f}, r(ent)={r_wm['r_low_ent']:.3f}  "
                  f"Δs: S={r_wm['low_ds_S']:.4f} F={r_wm['low_ds_F']:.4f}  ({r_wm['n_low']:.0f} steps)")
            print(f"    高wm_gap半(state无用): r(Δs)={r_wm['r_high_ds']:.3f}, r(ent)={r_wm['r_high_ent']:.3f}  "
                  f"Δs: S={r_wm['high_ds_S']:.4f} F={r_wm['high_ds_F']:.4f}  ({r_wm['n_high']:.0f} steps)")
