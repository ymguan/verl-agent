"""Side-by-side comparison: observe vs dual-token temporal decoupling."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/local_nvme/guanyiming/project/verl-agent/ocar/analysis_results")

RUNS = [
    ("observe (GRPO+obs metric only)", "observe_lmlyvpa6"),
    ("dual-token (λ=0.1, NLL on obs)", "dualtoken_l49ikuco"),
]


def z(x):
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x); sd = np.nanstd(x)
    return x - mu if sd == 0 or not np.isfinite(sd) else (x - mu) / sd


fig, axes = plt.subplots(len(RUNS), 4, figsize=(20, 4.2 * len(RUNS)))

for row, (title, tag) in enumerate(RUNS):
    csv = OUT / f"a1a2_{tag}.csv"
    js = json.loads((OUT / f"a1a2_{tag}.json").read_text())
    df = pd.read_csv(csv)
    step = df["step"].to_numpy()
    obs_key = js["keys"]["obs"]; sr_key = js["keys"]["sr"]
    val_key = js["keys"]["val"]
    obs = df[obs_key].to_numpy(dtype=float)
    sr = df[sr_key].to_numpy(dtype=float)

    # ---- panel 1: z-trajectories
    ax = axes[row, 0]
    ax.plot(step, z(obs), color="#d62728", label="obs_s_theta")
    ax.plot(step, z(sr),  color="#1f77b4", label="episode_SR")
    if val_key and val_key in df.columns:
        val = df[val_key].to_numpy(dtype=float)
        ax.plot(step, z(val), color="#2ca02c", alpha=0.6, label="val_SR")
    for k, c in [("observe/obs_wm_s_mean", "#8c564b"),
                 ("observe/obs_wm_s_B_mean", "#e377c2"),
                 ("observe/obs_delta_s_mean_mean", "#9467bd")]:
        if k in df.columns:
            ax.plot(step, z(df[k].to_numpy(dtype=float)),
                    color=c, alpha=0.55, lw=1.0, label=k.split("/")[-1])
    ax.set_title(f"[{title}] z-trajectories")
    ax.set_xlabel("training step"); ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower right")

    # ---- panel 2: rolling Pearson
    ax = axes[row, 1]
    rw = js["A1"]["rolling_window"]["series"]
    rs = [r[0] for r in rw]; rv = [r[1] if r[1] is not None else np.nan for r in rw]
    ax.plot(rs, rv, color="#9467bd")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(f"rolling Pearson(obs,SR), W={js['A1']['rolling_window']['window']}")
    ax.set_xlabel("step (center)"); ax.set_ylabel("r"); ax.grid(alpha=0.3)

    # ---- panel 3: CCF
    ax = axes[row, 2]
    ccf = js["A1"]["ccf"]
    lags = np.array(ccf["lags"]); vals = np.array([np.nan if v is None else v for v in ccf["ccf"]])
    ax.stem(lags, vals, basefmt=" ")
    k_best = ccf["arg_max_abs_lag"]
    ax.axvline(k_best, color="red", ls="--", label=f"argmax|.|={k_best}")
    ax.set_title(f"CCF (ccf@peak={ccf['ccf_at_arg_max']:+.3f})")
    ax.set_xlabel("lag (obs leads SR by k)")
    ax.set_ylabel("norm cov"); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # ---- panel 4: stage-wise slopes (bar)
    ax = axes[row, 3]
    stage_names = ["early", "mid", "late"]
    slopes = [js["A2"]["stages"][s]["levels_SR_on_obs"]["slope"] or 0
              for s in stage_names]
    r2s = [js["A2"]["stages"][s]["levels_SR_on_obs"]["r2"] or 0
           for s in stage_names]
    bars = ax.bar(stage_names, slopes,
                  color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    for b, r2 in zip(bars, r2s):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"R²={r2:.2f}", ha="center",
                va="bottom" if b.get_height() >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("A2 stage-wise slope  SR = a + b·obs_s_theta")
    ax.set_ylabel("slope b"); ax.grid(alpha=0.3, axis="y")

fig.suptitle("Temporal decoupling — observe vs dual-token  (ALFWorld, Qwen2.5-1.5B, 150 GRPO steps)",
             y=1.005, fontsize=13)
fig.tight_layout()
out = OUT / "a1a2_compare_observe_vs_dualtoken.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
