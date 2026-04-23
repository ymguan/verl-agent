"""Credit assignment via |Δs|: which steps get most/least credit, and is it correct?"""
import json, re
import numpy as np


def extract_action(text):
    m = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    return m.group(1).strip() if m else text.replace("\n", " ").strip()[:80]


def truncate(s, n=120):
    return s.replace("\n", " ").strip()[:n]


def analyze_credit(path: str, n_trajs=4):
    with open(path) as f:
        data = json.load(f)

    label = path.split("/")[-3] if "checkpoints" in path else path.split("/")[-1]
    print(f"\n{'='*130}")
    print(f"=== {label} step {data['global_step']} | SR={data['n_success']}/{data['n_trajectories']} ===")
    print(f"{'='*130}")

    # Credit assignment scheme: weight_i = |Δs_i| / Σ|Δs_j|
    # For success traj: high weight steps should be "good key actions"
    # For failure traj: high weight steps should be "bad key actions"

    for outcome_label, target_success in [("SUCCESS", True), ("FAILURE", False)]:
        trajs = [t for t in data["trajectories"] if t["success"] == target_success]
        if not trajs:
            continue
        # Pick a few representative trajectories (varied lengths)
        trajs_sorted = sorted(trajs, key=lambda t: t["n_steps"])
        picks = []
        if len(trajs_sorted) >= n_trajs:
            indices = np.linspace(0, len(trajs_sorted)-1, n_trajs, dtype=int)
            picks = [trajs_sorted[i] for i in indices]
        else:
            picks = trajs_sorted

        for traj in picks:
            steps = traj["steps"]
            ds_vals = []
            for s in steps:
                ds = s.get("delta_s_mean")
                ds_vals.append(ds if ds is not None else 0)
            ds_arr = np.array(ds_vals)
            abs_ds = np.abs(ds_arr)
            total = abs_ds.sum()
            if total < 1e-8:
                continue
            weights = abs_ds / total

            print(f"\n--- [{outcome_label}] traj={traj['traj_id'][:8]} | {traj['n_steps']} steps ---")
            print(f"{'step':>4s} | {'weight':>7s} | {'Δs':>8s} | {'entropy':>8s} | {'action':40s} | {'observation (truncated)'}")
            print("-" * 140)

            # Sort by weight descending to highlight top credit steps
            indexed = list(enumerate(steps))
            indexed_sorted = sorted(indexed, key=lambda x: weights[x[0]], reverse=True)

            for rank, (i, s) in enumerate(indexed_sorted):
                action = extract_action(s["action"])
                obs = truncate(s["observation"], 80)
                ds = ds_vals[i]
                ent = s.get("entropy_mean", 0)
                marker = " <<<" if rank < 3 else ""
                print(f"{s['step']:4d} | {weights[i]:7.1%} | {ds:8.4f} | {ent:8.4f} | {action:40s} | {obs}{marker}")

            # Summary: top 3 credit steps
            top3 = indexed_sorted[:3]
            print(f"\n  TOP-3 credit steps (get {sum(weights[i] for i,_ in top3):.0%} of total credit):")
            for rank, (i, s) in enumerate(top3):
                action = extract_action(s["action"])
                obs = truncate(s["observation"], 60)
                correct = ""
                if target_success:
                    # For success: is this step actually a good action?
                    obs_lower = s["observation"].lower()
                    if any(p in obs_lower for p in ["you pick up", "you put", "you open", "you close", "you turn", "you heat", "you cool", "you clean"]):
                        correct = "✓ state_change"
                    elif "nothing happens" in obs_lower:
                        correct = "✗ nothing_happens"
                    elif s["observation"] in [st["observation"] for st in steps[:i]]:
                        correct = "? revisit"
                    else:
                        correct = "~ new_location"
                else:
                    obs_lower = s["observation"].lower()
                    if "nothing happens" in obs_lower:
                        correct = "✓ correctly penalizes nothing_happens"
                    elif any(p in obs_lower for p in ["you pick up", "you put", "you open"]):
                        correct = "✗ wrongly penalizes state_change"
                    else:
                        correct = "~ navigation"
                print(f"    #{rank+1} step {s['step']}: weight={weights[i]:.1%}, Δs={ds_vals[i]:.4f}, action='{action}' | {correct}")


if __name__ == "__main__":
    for p in [
        "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json",
        "checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_60/observe_trajectories.json",
        "data/trajectories/grpo_observe_webshop_20260418_070828/global_step_240_observe_trajectories.json",
        "data/trajectories/grpo_observe_webshop_20260418_070828/global_step_320_observe_trajectories.json",
    ]:
        try:
            analyze_credit(p, n_trajs=3)
        except FileNotFoundError:
            print(f"SKIP: {p}")
