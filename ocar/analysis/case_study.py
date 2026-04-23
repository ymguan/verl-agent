"""Extract concrete cases: high/low delta_s steps, grouped by success/failure and obs type."""
import json, glob
import numpy as np


def truncate(s, n=120):
    s = s.replace("\n", " ").strip()
    return s[:n] + "..." if len(s) > n else s


def extract_action(action_text):
    """Extract the <action>...</action> part."""
    import re
    m = re.search(r"<action>(.*?)</action>", action_text, re.DOTALL)
    return m.group(1).strip() if m else truncate(action_text, 80)


def analyze_cases(path: str, top_n=5):
    with open(path) as f:
        data = json.load(f)

    label = path.split("/")[-3] if "checkpoints" in path else path.split("/")[-1]
    print(f"\n{'='*120}")
    print(f"=== {label} step {data['global_step']} | SR={data['n_success']}/{data['n_trajectories']} ===")
    print(f"{'='*120}")

    # Collect all steps with context
    all_steps = []
    for traj in data["trajectories"]:
        prev_obs = []
        for step in traj["steps"]:
            obs = step["observation"]
            ds = step.get("delta_s_mean")
            ent = step.get("entropy_mean")
            wm = step.get("wm_s")
            if ds is None:
                continue

            # classify obs
            obs_lower = obs.lower()
            if any(p in obs_lower for p in ["nothing happens", "nothing to", "can't", "don't see"]):
                obs_type = "nothing_happens"
            elif any(p in obs_lower for p in ["you pick up", "you put", "you open", "you close", "you turn", "you heat", "you cool", "you clean", "you slice", "you use"]):
                obs_type = "state_change"
            elif obs in prev_obs:
                obs_type = "revisit"
            else:
                obs_type = "new_location"
            prev_obs.append(obs)

            all_steps.append({
                "traj_id": traj["traj_id"][:8],
                "success": traj["success"],
                "step_num": step["step"],
                "n_steps": traj["n_steps"],
                "obs": obs,
                "action": step["action"],
                "delta_s": ds,
                "entropy": ent,
                "wm_s": wm,
                "obs_type": obs_type,
                "s_theta": step.get("s_theta_mean", 0),
            })

    if not all_steps:
        return

    ds_arr = np.array([s["delta_s"] for s in all_steps])
    p10, p25, p75, p90 = np.percentile(ds_arr, [10, 25, 75, 90])

    print(f"\nDelta_s distribution: min={ds_arr.min():.4f}, p10={p10:.4f}, p25={p25:.4f}, median={np.median(ds_arr):.4f}, p75={p75:.4f}, p90={p90:.4f}, max={ds_arr.max():.4f}")

    # Show extreme cases
    sorted_steps = sorted(all_steps, key=lambda x: x["delta_s"])

    for label, steps in [("LOWEST delta_s (most negative = policy much more surprised than ref)", sorted_steps[:top_n]),
                          ("HIGHEST delta_s (most positive = policy much less surprised than ref)", sorted_steps[-top_n:][::-1])]:
        print(f"\n--- {label} ---")
        for s in steps:
            outcome = "SUCCESS" if s["success"] else "FAIL"
            action = extract_action(s["action"])
            print(f"  [{outcome}] traj={s['traj_id']} step {s['step_num']}/{s['n_steps']} | obs_type={s['obs_type']}")
            print(f"    delta_s={s['delta_s']:.4f}  entropy={s['entropy']:.4f}  s_theta={s['s_theta']:.4f}  wm_s={s['wm_s']:.4f}")
            print(f"    obs: {truncate(s['obs'], 150)}")
            print(f"    action: {action}")
            print()

    # The key question: cases where direction is "wrong"
    # i.e. successful steps with very negative delta_s, or failed steps with very positive delta_s
    print(f"\n--- COUNTER-INTUITIVE: SUCCESS steps with delta_s < p10 ({p10:.4f}) ---")
    counter_succ = [s for s in all_steps if s["success"] and s["delta_s"] < p10]
    for s in counter_succ[:top_n]:
        action = extract_action(s["action"])
        print(f"  traj={s['traj_id']} step {s['step_num']}/{s['n_steps']} | obs_type={s['obs_type']}")
        print(f"    delta_s={s['delta_s']:.4f}  entropy={s['entropy']:.4f}  s_theta={s['s_theta']:.4f}")
        print(f"    obs: {truncate(s['obs'], 150)}")
        print(f"    action: {action}")
        print()

    print(f"--- COUNTER-INTUITIVE: FAIL steps with delta_s > p90 ({p90:.4f}) ---")
    counter_fail = [s for s in all_steps if not s["success"] and s["delta_s"] > p90]
    for s in counter_fail[:top_n]:
        action = extract_action(s["action"])
        print(f"  traj={s['traj_id']} step {s['step_num']}/{s['n_steps']} | obs_type={s['obs_type']}")
        print(f"    delta_s={s['delta_s']:.4f}  entropy={s['entropy']:.4f}  s_theta={s['s_theta']:.4f}")
        print(f"    obs: {truncate(s['obs'], 150)}")
        print(f"    action: {action}")
        print()

    # Count stats
    n_succ_low = len(counter_succ)
    n_succ_total = sum(1 for s in all_steps if s["success"])
    n_fail_high = len(counter_fail)
    n_fail_total = sum(1 for s in all_steps if not s["success"])
    print(f"Counter-intuitive rate: {n_succ_low}/{n_succ_total} ({n_succ_low/n_succ_total:.1%}) success steps below p10, {n_fail_high}/{n_fail_total} ({n_fail_high/n_fail_total:.1%}) fail steps above p90")


# Run on key checkpoints
for p in [
    "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_60/observe_trajectories.json",
    "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json",
    "data/trajectories/grpo_observe_webshop_20260418_070828/global_step_240_observe_trajectories.json",
    "data/trajectories/grpo_observe_webshop_20260418_070828/global_step_320_observe_trajectories.json",
]:
    try:
        analyze_cases(p)
    except FileNotFoundError:
        print(f"SKIP: {p}")
