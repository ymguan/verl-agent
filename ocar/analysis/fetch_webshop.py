"""Fetch webshop grpo+observe run history from wandb."""
import os, wandb, pandas as pd

os.environ.setdefault("WANDB_API_KEY",
    "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")

PROJ = "guanyiming290-alibaba/grpo_observe_webshop_20260418_070828"
RUN  = "42rxhh6f"

KEYS = [
    "_step",
    "observe/obs_s_theta_mean_mean","observe/obs_s_theta_mean_std",
    "observe/obs_s_ref_mean_mean","observe/obs_s_ref_mean_std",
    "observe/obs_delta_s_mean_mean","observe/obs_delta_s_mean_std",
    "observe/obs_delta_s_sum_mean","observe/obs_delta_s_sum_std",
    "observe/obs_consecutive_s_mean","observe/obs_consecutive_s_std",
    "observe/obs_wm_s_mean","observe/obs_wm_s_B_mean",
    "observe/obs_step_entropy_mean_mean","observe/obs_step_entropy_mean_std",
    "observe/success_s_theta_mean","observe/failure_s_theta_mean",
    "observe/success_entropy_mean","observe/failure_entropy_mean",
    "observe/obs_n_tokens_mean",
    "episode/success_rate","episode/reward/mean",
    "episode/webshop_task_score (not success_rate)",
    "val/success_rate","val/webshop_task_score (not success_rate)",
    "actor/entropy_loss","actor/kl_loss","actor/ppo_kl",
]

def main():
    api = wandb.Api()
    r = api.run(f"{PROJ}/{RUN}")
    print("state:", r.state, "name:", r.name)
    h = r.history(keys=KEYS, samples=5000, pandas=True)
    print("rows:", len(h), "step range:", h["_step"].min(), "→", h["_step"].max())
    out = os.path.join(os.path.dirname(__file__), "..", "analysis_results", "webshop", "history.csv")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    h.to_csv(out, index=False)
    print("saved:", out)

if __name__ == "__main__":
    main()
