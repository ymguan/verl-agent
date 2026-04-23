#!/usr/bin/env python3
"""
Deep analysis of surprise signal characteristics for OCAR credit assignment.

1. Within-trajectory surprise distribution shape (skew, kurtosis)
2. Per-task-type surprise variance (clean vs heat vs others)
3. Correlation between surprise ranking and step importance
4. Signal growth rate and collapse risk simulation
5. Normalization schemes comparison
"""

import json
import numpy as np
from collections import defaultdict
from scipy import stats as sp_stats

TRAJ_FILE = "/local_nvme/guanyiming/project/verl-agent/checkpoints/ocar_alfworld_20260414_094253/ocar_tau1.0_dstrue/global_step_1/ocar_trajectories.json"
TRAJ_FILE_2 = "/local_nvme/guanyiming/project/verl-agent/checkpoints/ocar_alfworld_20260414_094253/ocar_tau1.0_dstrue/global_step_2/ocar_trajectories.json"
DELTA_S_FILE = "/local_nvme/guanyiming/project/verl-agent/ocar/delta_s_variance_analysis.json"


def load_trajs(path):
    with open(path) as f:
        return json.load(f)["trajectories"]


def softmax_weights(signal, tau, success):
    """Compute OCAR weights for a trajectory."""
    signal = np.array(signal)
    T = len(signal)
    if success:
        logits = -signal / tau
    else:
        logits = signal / tau
    logits -= logits.max()  # numerical stability
    exp_logits = np.exp(logits)
    weights = T * exp_logits / exp_logits.sum()
    return weights


def analyze_trajectories(trajs, label=""):
    """Comprehensive analysis of surprise signal within trajectories."""
    print(f"\n{'='*80}")
    print(f"ANALYSIS: {label}")
    print(f"{'='*80}")

    # Collect per-trajectory stats
    by_task = defaultdict(lambda: {
        "s_theta": [], "s_ref": [], "delta_s": [],
        "within_std_raw": [], "within_std_delta": [],
        "n_steps": [], "success": [], "n": 0
    })

    all_within_std_raw = []
    all_within_std_delta = []
    all_s_theta_flat = []
    all_delta_s_flat = []

    for t in trajs:
        steps = t["steps"]
        if len(steps) < 2:
            continue

        s_thetas = [s["s_theta"] for s in steps]
        s_refs = [s["s_ref"] for s in steps]
        delta_ss = [s["delta_s"] for s in steps]

        task = t.get("task_type", "unknown")
        d = by_task[task]
        d["n"] += 1
        d["success"].append(t["success"])
        d["n_steps"].append(len(steps))
        d["within_std_raw"].append(np.std(s_thetas))
        d["within_std_delta"].append(np.std(delta_ss))
        d["s_theta"].extend(s_thetas)
        d["delta_s"].extend(delta_ss)

        all_within_std_raw.append(np.std(s_thetas))
        all_within_std_delta.append(np.std(delta_ss))
        all_s_theta_flat.extend(s_thetas)
        all_delta_s_flat.extend(delta_ss)

    # ── 1. Overall signal statistics ──
    print(f"\n--- 1. Overall Signal Statistics ---")
    print(f"{'Metric':<30} {'Raw S_θ':>12} {'ΔS':>12}")
    print(f"{'Global mean':<30} {np.mean(all_s_theta_flat):>12.4f} {np.mean(all_delta_s_flat):>12.4f}")
    print(f"{'Global std':<30} {np.std(all_s_theta_flat):>12.4f} {np.std(all_delta_s_flat):>12.4f}")
    print(f"{'Within-traj std (mean)':<30} {np.mean(all_within_std_raw):>12.4f} {np.mean(all_within_std_delta):>12.4f}")
    print(f"{'Within-traj std (median)':<30} {np.median(all_within_std_raw):>12.4f} {np.median(all_within_std_delta):>12.4f}")
    print(f"{'Within-traj std (min)':<30} {np.min(all_within_std_raw):>12.4f} {np.min(all_within_std_delta):>12.4f}")
    print(f"{'Within-traj std (max)':<30} {np.max(all_within_std_raw):>12.4f} {np.max(all_within_std_delta):>12.4f}")

    # ── 2. Per-task-type analysis ──
    print(f"\n--- 2. Per-Task-Type Signal Quality ---")
    print(f"{'Task':<18} {'n':>4} {'SR':>6} {'S_θ within-std':>15} {'ΔS within-std':>15} {'S_θ mean':>10}")
    print("-" * 72)
    for task in sorted(by_task.keys()):
        d = by_task[task]
        sr = np.mean(d["success"]) * 100
        raw_std = np.mean(d["within_std_raw"])
        ds_std = np.mean(d["within_std_delta"])
        s_mean = np.mean(d["s_theta"])
        print(f"{task:<18} {d['n']:>4} {sr:>5.1f}% {raw_std:>15.4f} {ds_std:>15.4f} {s_mean:>10.4f}")

    # ── 3. Success vs Failure signal comparison ──
    print(f"\n--- 3. Success vs Failure Within-Trajectory Variance ---")
    succ_raw = [all_within_std_raw[i] for i, t in enumerate(trajs) if t["success"] and len(t["steps"]) >= 2]
    fail_raw = [all_within_std_raw[i] for i, t in enumerate(trajs) if not t["success"] and len(t["steps"]) >= 2]
    succ_ds = [all_within_std_delta[i] for i, t in enumerate(trajs) if t["success"] and len(t["steps"]) >= 2]
    fail_ds = [all_within_std_delta[i] for i, t in enumerate(trajs) if not t["success"] and len(t["steps"]) >= 2]

    if succ_raw and fail_raw:
        print(f"{'':>20} {'Success':>12} {'Failure':>12} {'Ratio F/S':>12}")
        print(f"{'S_θ within-std':<20} {np.mean(succ_raw):>12.4f} {np.mean(fail_raw):>12.4f} {np.mean(fail_raw)/np.mean(succ_raw):>12.2f}x")
        print(f"{'ΔS within-std':<20} {np.mean(succ_ds):>12.4f} {np.mean(fail_ds):>12.4f} {np.mean(fail_ds)/np.mean(succ_ds) if np.mean(succ_ds)>0 else float('nan'):>12.2f}x")

    # ── 4. OCAR weight distribution under different normalization ──
    print(f"\n--- 4. OCAR Weight Distributions (tau=1.0) ---")
    print(f"{'Method':<25} {'weight std':>12} {'weight min':>12} {'weight max':>12} {'% clamped':>12}")
    print("-" * 75)

    methods = {
        "raw ΔS": lambda t: [s["delta_s"] for s in t["steps"]],
        "raw S_θ": lambda t: [s["s_theta"] for s in t["steps"]],
        "ΔS / S_ref (relative)": lambda t: [s["delta_s"] / max(s["s_ref"], 0.01) for s in t["steps"]],
        "log(S_θ/S_ref)": lambda t: [np.log(max(s["s_theta"], 0.01) / max(s["s_ref"], 0.01)) for s in t["steps"]],
        "z-norm ΔS (ε=0.05)": None,  # special handling
        "z-norm S_θ (ε=0.05)": None,
        "z-norm ΔS (ε=0.1)": None,
    }

    for method_name in methods:
        all_w_stds = []
        all_w_mins = []
        all_w_maxs = []
        n_clamped = 0
        n_total = 0

        for t in trajs:
            if len(t["steps"]) < 2:
                continue

            if method_name.startswith("z-norm ΔS"):
                eps = float(method_name.split("ε=")[1].rstrip(")"))
                signal = np.array([s["delta_s"] for s in t["steps"]])
                std = max(signal.std(), eps)
                signal = (signal - signal.mean()) / std
            elif method_name.startswith("z-norm S_θ"):
                eps = float(method_name.split("ε=")[1].rstrip(")"))
                signal = np.array([s["s_theta"] for s in t["steps"]])
                std = max(signal.std(), eps)
                signal = (signal - signal.mean()) / std
            else:
                signal = methods[method_name](t)

            signal = np.array(signal)
            w = softmax_weights(signal, tau=1.0, success=t["success"])
            all_w_stds.append(np.std(w))
            all_w_mins.append(np.min(w))
            all_w_maxs.append(np.max(w))
            clamped = np.sum((w < 0.1) | (w > 10.0))
            n_clamped += clamped
            n_total += len(w)

        print(f"{method_name:<25} {np.mean(all_w_stds):>12.4f} {np.mean(all_w_mins):>12.4f} "
              f"{np.mean(all_w_maxs):>12.4f} {100*n_clamped/max(n_total,1):>11.2f}%")

    # ── 5. Collapse risk simulation ──
    print(f"\n--- 5. Collapse Risk: Weight Extremes Growth Over Training ---")
    with open(DELTA_S_FILE) as f:
        delta_s_data = json.load(f)

    print(f"{'Step':<10} {'ΔS within-std':>15} {'Simulated w_max (ΔS)':>22} {'Simulated w_max (raw)':>22}")
    print("-" * 72)

    # Also add v2 step 1 data point
    v2_ds_std = np.mean([np.std([s["delta_s"] for s in t["steps"]]) for t in trajs if len(t["steps"]) >= 2])
    v2_raw_std = np.mean([np.std([s["s_theta"] for s in t["steps"]]) for t in trajs if len(t["steps"]) >= 2])

    all_points = [{"step": "1 (v2)", "delta_s_within_traj_std": v2_ds_std, "raw_s_theta_within_traj_std": v2_raw_std}]
    all_points.extend(delta_s_data)

    for r in all_points:
        T = 10
        np.random.seed(42)
        ds_signal = np.random.normal(0, r["delta_s_within_traj_std"], T)
        raw_signal = np.random.normal(0, r["raw_s_theta_within_traj_std"], T)
        ds_w = softmax_weights(ds_signal, 1.0, False)
        raw_w = softmax_weights(raw_signal, 1.0, False)
        step_label = r["step"] if isinstance(r["step"], str) else f"{r['step']}"
        print(f"{step_label:<10} {r['delta_s_within_traj_std']:>15.4f} {ds_w.max():>22.3f} {raw_w.max():>22.3f}")

    # ── 6. Extrapolation: what happens at step 200, 300? ──
    print(f"\n--- 6. Extrapolation: ΔS Growth Trend ---")
    steps = [d["step"] for d in delta_s_data]
    ds_stds = [d["delta_s_within_traj_std"] for d in delta_s_data]
    # Fit log curve (ΔS std grows roughly logarithmically)
    log_steps = np.log(steps)
    coeffs = np.polyfit(log_steps, ds_stds, 1)
    print(f"Fit: ΔS_std ≈ {coeffs[0]:.3f} * ln(step) + {coeffs[1]:.3f}")
    for future_step in [200, 300, 500]:
        predicted_std = coeffs[0] * np.log(future_step) + coeffs[1]
        # Simulate worst-case weight
        T = 10
        np.random.seed(42)
        signal = np.random.normal(0, predicted_std, T)
        w = softmax_weights(signal, 1.0, False)
        print(f"  step {future_step}: predicted ΔS_std = {predicted_std:.3f}, w_max = {w.max():.2f}, w_min = {w.min():.3f}")

    # ── 7. Z-normalization stability check ──
    print(f"\n--- 7. Z-Normalization: Stability Across Training ---")
    print(f"With z-norm(ε=0.1), weight distribution is invariant to signal magnitude:")
    print(f"{'Signal std':>12} {'After z-norm std':>18} {'w_max':>8} {'w_min':>8} {'w_std':>8}")
    print("-" * 58)
    for sig_std in [0.009, 0.05, 0.1, 0.3, 0.5, 0.7, 1.0, 2.0]:
        T = 10
        np.random.seed(42)
        signal = np.random.normal(0, sig_std, T)
        eps = 0.1
        z = (signal - signal.mean()) / max(signal.std(), eps)
        w = softmax_weights(z, 1.0, False)
        print(f"{sig_std:>12.3f} {z.std():>18.4f} {w.max():>8.3f} {w.min():>8.3f} {w.std():>8.4f}")


def main():
    trajs1 = load_trajs(TRAJ_FILE)
    trajs2 = load_trajs(TRAJ_FILE_2)

    analyze_trajectories(trajs1, "v2 Step 1 (128 trajectories)")

    # Quick cross-step comparison
    print(f"\n{'='*80}")
    print("STEP 1 vs STEP 2 SIGNAL EVOLUTION (v2 early training)")
    print(f"{'='*80}")

    for step_label, trajs in [("Step 1", trajs1), ("Step 2", trajs2)]:
        ds_stds = [np.std([s["delta_s"] for s in t["steps"]]) for t in trajs if len(t["steps"]) >= 2]
        raw_stds = [np.std([s["s_theta"] for s in t["steps"]]) for t in trajs if len(t["steps"]) >= 2]
        print(f"{step_label}: ΔS within-std = {np.mean(ds_stds):.4f}, raw S_θ within-std = {np.mean(raw_stds):.4f}")


if __name__ == "__main__":
    main()
