"""ALFWorld hidden state info gain using BASE model (Qwen2.5-7B-Instruct) for fair comparison with WebShop."""
import sys
sys.path.insert(0, ".")
from ocar.analysis.hidden_state_info_gain import load_model, extract_hidden_states, analyze_results
import json, os, numpy as np, torch
from pathlib import Path

def main():
    layers_to_check = (-1, -16, -28)
    out_dir = Path("ocar/analysis_results/hidden_state")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_model_path = "/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct"
    traj_path = "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"

    print("="*60)
    print("ALFWorld Analysis (BASE model: Qwen2.5-7B-Instruct)")
    print("="*60)

    model, tokenizer = load_model(base_model_path)

    with open(traj_path) as f:
        data = json.load(f)

    trajs = data["trajectories"]
    fail_trajs = [t for t in trajs if not t["success"]]
    succ_trajs = [t for t in trajs if t["success"]]

    np.random.seed(42)
    succ_sample = list(np.random.choice(len(succ_trajs), size=min(20, len(succ_trajs)), replace=False))
    sample_trajs = fail_trajs + [succ_trajs[i] for i in succ_sample]
    np.random.shuffle(sample_trajs)

    print(f"Sampled {len(sample_trajs)} trajectories ({len(fail_trajs)} fail + {min(20, len(succ_trajs))} succ)")

    results = extract_hidden_states(model, tokenizer, sample_trajs, layers=layers_to_check)
    analyze_results(results, label="ALFWorld BASE model")

    with open(out_dir / "alfworld_base_model.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\nDone! Saved to ocar/analysis_results/hidden_state/alfworld_base_model.json")

if __name__ == "__main__":
    main()
