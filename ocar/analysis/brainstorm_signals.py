"""
Brainstorm analysis: test 4 candidate step-level signal ideas on real trajectories.

Direction 1: Consecutive exploration detector — consecutive high-|Δs| steps without state_change
Direction 2: Entropy trajectory shape — does entropy decrease over time in success vs stay high in failure?
Direction 3: Surprise-gated credit assignment — high-|Δs| steps get uniform weight, low-|Δs| steps get outcome weight
Direction 5: Surprise velocity (ΔΔs) — is |Δs| increasing (drifting) or decreasing (converging)?
"""
import json, glob, re
import numpy as np
from collections import defaultdict


def extract_action(text):
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    return m.group(1).strip() if m else text.replace("\n", " ").strip()[:80]


def classify_obs(obs, prev_obs_list):
    obs_lower = obs.lower()
    if any(p in obs_lower for p in ["nothing happens", "nothing to", "can't", "don't see"]):
        return "nothing_happens"
    elif any(p in obs_lower for p in ["you pick up", "you put", "you open", "you close",
                                       "you turn", "you heat", "you cool", "you clean",
                                       "you slice", "you use"]):
        return "state_change"
    elif obs in prev_obs_list:
        return "revisit"
    else:
        return "new_location"


def load_trajs(path):
    with open(path) as f:
        data = json.load(f)
    return data


# ============================================================
# Direction 1: Consecutive exploration detector
# ============================================================
def analyze_direction1(data, ds_threshold_pct=75):
    """Count max consecutive high-|Δs| steps without state_change.
    Hypothesis: failure trajs have longer such streaks."""
    all_ds = []
    for traj in data["trajectories"]:
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            if ds is not None:
                all_ds.append(abs(ds))
    if not all_ds:
        return None
    threshold = np.percentile(all_ds, ds_threshold_pct)

    results = []
    for traj in data["trajectories"]:
        prev_obs = []
        max_streak = 0
        curr_streak = 0
        total_streaks = 0
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            obs_type = classify_obs(step["observation"], prev_obs)
            prev_obs.append(step["observation"])

            if ds is not None and abs(ds) > threshold and obs_type != "state_change":
                curr_streak += 1
            else:
                if curr_streak > 0:
                    total_streaks += 1
                max_streak = max(max_streak, curr_streak)
                curr_streak = 0
        max_streak = max(max_streak, curr_streak)

        results.append({
            "success": traj["success"],
            "max_streak": max_streak,
            "n_steps": traj["n_steps"],
            "streak_ratio": max_streak / max(traj["n_steps"], 1),
        })

    succ = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    if not succ or not fail:
        return None

    return {
        "succ_max_streak": np.mean([r["max_streak"] for r in succ]),
        "fail_max_streak": np.mean([r["max_streak"] for r in fail]),
        "succ_streak_ratio": np.mean([r["streak_ratio"] for r in succ]),
        "fail_streak_ratio": np.mean([r["streak_ratio"] for r in fail]),
        "threshold": threshold,
    }


# ============================================================
# Direction 2: Entropy trajectory shape
# ============================================================
def analyze_direction2(data):
    """Compare entropy in first half vs second half of trajectory.
    Hypothesis: success trajs have decreasing entropy, failure have flat/increasing."""
    results = []
    for traj in data["trajectories"]:
        entropies = []
        for step in traj["steps"]:
            ent = step.get("entropy_mean")
            if ent is not None:
                entropies.append(ent)
        if len(entropies) < 4:
            continue
        mid = len(entropies) // 2
        first_half = np.mean(entropies[:mid])
        second_half = np.mean(entropies[mid:])
        # Linear trend: positive slope = increasing entropy
        x = np.arange(len(entropies))
        slope = np.polyfit(x, entropies, 1)[0]

        results.append({
            "success": traj["success"],
            "ent_first": first_half,
            "ent_second": second_half,
            "ent_delta": second_half - first_half,
            "ent_slope": slope,
            "n_steps": traj["n_steps"],
        })

    succ = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    if not succ or not fail:
        return None

    return {
        "succ_ent_delta": np.mean([r["ent_delta"] for r in succ]),
        "fail_ent_delta": np.mean([r["ent_delta"] for r in fail]),
        "succ_slope": np.mean([r["ent_slope"] for r in succ]),
        "fail_slope": np.mean([r["ent_slope"] for r in fail]),
        "succ_first": np.mean([r["ent_first"] for r in succ]),
        "succ_second": np.mean([r["ent_second"] for r in succ]),
        "fail_first": np.mean([r["ent_first"] for r in fail]),
        "fail_second": np.mean([r["ent_second"] for r in fail]),
    }


# ============================================================
# Direction 3: Surprise-gated credit assignment
# ============================================================
def analyze_direction3(data, ds_threshold_pct=50):
    """Split steps into high-|Δs| (exploratory) and low-|Δs| (decisive).
    Check if low-|Δs| steps have better outcome-correlation than all steps."""
    all_ds = []
    for traj in data["trajectories"]:
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            if ds is not None:
                all_ds.append(abs(ds))
    if not all_ds:
        return None
    threshold = np.percentile(all_ds, ds_threshold_pct)

    results_low = []  # decisive steps
    results_high = []  # exploratory steps
    for traj in data["trajectories"]:
        low_ds = []
        high_ds = []
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            if ds is None:
                continue
            if abs(ds) <= threshold:
                low_ds.append(ds)
            else:
                high_ds.append(ds)
        if low_ds:
            results_low.append({"success": float(traj["success"]), "mean_ds": np.mean(low_ds), "n": len(low_ds)})
        if high_ds:
            results_high.append({"success": float(traj["success"]), "mean_ds": np.mean(high_ds), "n": len(high_ds)})

    def corr(results):
        if len(results) < 5:
            return float('nan')
        s = np.array([r["success"] for r in results])
        d = np.array([r["mean_ds"] for r in results])
        if np.std(s) < 1e-8 or np.std(d) < 1e-8:
            return float('nan')
        return np.corrcoef(s, d)[0, 1]

    return {
        "r_low_ds": corr(results_low),
        "r_high_ds": corr(results_high),
        "n_low": np.mean([r["n"] for r in results_low]) if results_low else 0,
        "n_high": np.mean([r["n"] for r in results_high]) if results_high else 0,
        "threshold": threshold,
    }


# ============================================================
# Direction 5: Surprise velocity (ΔΔs)
# ============================================================
def analyze_direction5(data):
    """Track |Δs| trend within each trajectory.
    Hypothesis: success trajs have decreasing |Δs| (converging), failure have increasing (drifting)."""
    results = []
    for traj in data["trajectories"]:
        abs_ds = []
        for step in traj["steps"]:
            ds = step.get("delta_s_mean")
            if ds is not None:
                abs_ds.append(abs(ds))
        if len(abs_ds) < 4:
            continue
        x = np.arange(len(abs_ds))
        slope = np.polyfit(x, abs_ds, 1)[0]
        mid = len(abs_ds) // 2
        first = np.mean(abs_ds[:mid])
        second = np.mean(abs_ds[mid:])

        # Also: consecutive differences
        diffs = np.diff(abs_ds)
        mean_diff = np.mean(diffs)

        results.append({
            "success": traj["success"],
            "slope": slope,
            "mean_diff": mean_diff,
            "ds_delta": second - first,
            "n_steps": traj["n_steps"],
        })

    succ = [r for r in results if r["success"]]
    fail = [r for r in results if not r["success"]]
    if not succ or not fail:
        return None

    all_s = np.array([r["success"] for r in results], dtype=float)
    all_slope = np.array([r["slope"] for r in results])
    r_slope = np.corrcoef(all_s, all_slope)[0, 1] if np.std(all_slope) > 1e-8 else float('nan')

    return {
        "succ_slope": np.mean([r["slope"] for r in succ]),
        "fail_slope": np.mean([r["slope"] for r in fail]),
        "succ_ds_delta": np.mean([r["ds_delta"] for r in succ]),
        "fail_ds_delta": np.mean([r["ds_delta"] for r in fail]),
        "r_slope_success": r_slope,
    }


# ============================================================
# Main
# ============================================================
alfworld_1_5b = sorted(
    glob.glob("checkpoints/grpo_observe_alfworld_1.5b_*/grpo_observe_*/global_step_*/observe_trajectories.json"),
    key=lambda x: int(x.split("global_step_")[1].split("/")[0]))
alfworld_7b = ["checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"]
webshop = sorted(
    glob.glob("data/trajectories/grpo_observe_webshop_*/global_step_*_observe_trajectories.json"),
    key=lambda x: int(x.split("global_step_")[1].split("_")[0]))

datasets = [("ALFWorld-1.5B", alfworld_1_5b), ("ALFWorld-7B", alfworld_7b), ("WebShop", webshop)]

for name, files in datasets:
    print(f"\n{'#'*80}")
    print(f"### {name} ###")
    print(f"{'#'*80}")

    for f in files:
        try:
            data = load_trajs(f)
        except FileNotFoundError:
            continue

        step = data["global_step"]
        sr = data["n_success"] / data["n_trajectories"]
        print(f"\n--- step {step} | SR={data['n_success']}/{data['n_trajectories']} ({sr:.0%}) ---")

        # Direction 1
        r1 = analyze_direction1(data)
        if r1:
            print(f"  [D1 循环检测] 成功轨迹最长探索连续段: {r1['succ_max_streak']:.1f}, 失败: {r1['fail_max_streak']:.1f}  "
                  f"(占比: 成功 {r1['succ_streak_ratio']:.1%}, 失败 {r1['fail_streak_ratio']:.1%})")

        # Direction 2
        r2 = analyze_direction2(data)
        if r2:
            print(f"  [D2 Entropy趋势] 成功: {r2['succ_first']:.4f}→{r2['succ_second']:.4f} (slope={r2['succ_slope']:.6f}), "
                  f"失败: {r2['fail_first']:.4f}→{r2['fail_second']:.4f} (slope={r2['fail_slope']:.6f})")

        # Direction 3
        r3 = analyze_direction3(data)
        if r3:
            print(f"  [D3 门控信用] 低|Δs|步骤(decisive) r={r3['r_low_ds']:.3f} (avg {r3['n_low']:.0f} steps), "
                  f"高|Δs|步骤(exploratory) r={r3['r_high_ds']:.3f} (avg {r3['n_high']:.0f} steps)")

        # Direction 5
        r5 = analyze_direction5(data)
        if r5:
            print(f"  [D5 Surprise速度] 成功轨迹|Δs|斜率: {r5['succ_slope']:.6f}, 失败: {r5['fail_slope']:.6f}  "
                  f"(r(slope,success)={r5['r_slope_success']:.3f})")
