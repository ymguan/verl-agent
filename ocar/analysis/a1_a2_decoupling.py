"""A1 + A2: Temporal decoupling analysis between surprise signal s_theta and SR.

A1 — Correlation & lead-lag:
  * Pearson / Spearman over the full run (levels and first-differences).
  * Rolling-window Pearson (window=W) to visualize how correlation evolves.
  * Cross-correlation function (CCF) of z-scored series over lag range
    [-L, +L]; the arg-max lag is reported as "lead" (positive lag = surprise
    signal LEADS SR by that many training steps).

A2 — Stage-wise regression:
  * Split the training trajectory into early / mid / late thirds by step index.
  * Per stage, fit OLS on levels:   SR_t       = a + b * s_theta_t + eps
    and on first-differences:       dSR_t      = a + b * d(s_theta_t) + eps
  * Report (b, R^2, n) per stage. Also fit a single global model as baseline.

Inputs are pulled OFFLINE from the local `output.log` of a wandb run, using
the existing `parse_wandb_log.parse_output_log` utility; no W&B API key or
network access is required.

Outputs (written under ocar/analysis_results/):
  * a1a2_<run_tag>.json               — all numbers
  * a1a2_<run_tag>.png                — 4-panel figure (series, rolling corr,
                                         CCF, stage scatters)
  * a1a2_<run_tag>.csv                — the aligned, filtered series used
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

# reuse the local parser (no wandb API)
import sys
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from ocar.analysis.parse_wandb_log import parse_output_log  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent.parent / "analysis_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------

def _corr(a: np.ndarray, b: np.ndarray):
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 3:
        return {"n": n, "pearson": None, "pearson_p": None,
                "spearman": None, "spearman_p": None}
    pr, pp = pearsonr(a[m], b[m])
    sr, sp = spearmanr(a[m], b[m])
    return {"n": n,
            "pearson": float(pr), "pearson_p": float(pp),
            "spearman": float(sr), "spearman_p": float(sp)}


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd == 0:
        return x - mu
    return (x - mu) / sd


def _ccf(x: np.ndarray, y: np.ndarray, max_lag: int):
    """Normalized cross-correlation at lags -L..+L.

    Sign convention: lag k > 0 means y(t) correlates with x(t - k),
    i.e. x leads y by k steps.  Returns (lags, corrs).
    """
    x = _zscore(x)
    y = _zscore(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)
    lags = np.arange(-max_lag, max_lag + 1)
    out = np.full(len(lags), np.nan)
    for i, k in enumerate(lags):
        if k >= 0:
            a = x[: n - k]
            b = y[k:]
        else:
            a = x[-k:]
            b = y[: n + k]
        if len(a) > 2:
            out[i] = float(np.mean(a * b))
    return lags, out


def _ols(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    n = len(x)
    if n < 3 or np.std(x) == 0:
        return {"n": n, "slope": None, "intercept": None, "r2": None}
    X = np.vstack([x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return {"n": int(n), "slope": float(beta[0]),
            "intercept": float(beta[1]),
            "r2": (float(r2) if r2 is not None else None)}


# ---------- main ----------

def run(df: pd.DataFrame, run_tag: str,
        obs_key: str, sr_key: str, val_key: str | None,
        extras: dict,
        window: int, max_lag: int) -> dict:
    # align to a dense integer step axis
    steps = df.index.to_numpy()
    obs = df[obs_key].to_numpy(dtype=float)
    sr = df[sr_key].to_numpy(dtype=float)
    val = df[val_key].to_numpy(dtype=float) if val_key and val_key in df else None

    # save aligned csv
    aligned = pd.DataFrame({"step": steps, obs_key: obs, sr_key: sr})
    for k, arr in extras.items():
        aligned[k] = arr
    if val is not None:
        aligned[val_key] = val
    aligned.to_csv(OUT_DIR / f"a1a2_{run_tag}.csv", index=False)

    results: dict = {"run_tag": run_tag, "n_steps": int(len(df)),
                     "keys": {"obs": obs_key, "sr": sr_key, "val": val_key},
                     "A1": {}, "A2": {}}

    # ---------- A1.1: global correlations (levels + first-diff) ----------
    d_obs = np.diff(obs, prepend=np.nan)
    d_sr = np.diff(sr, prepend=np.nan)
    results["A1"]["global_levels_obs_vs_sr"] = _corr(obs, sr)
    results["A1"]["global_diffs_obs_vs_sr"] = _corr(d_obs, d_sr)
    if val is not None:
        results["A1"]["global_levels_obs_vs_val"] = _corr(obs, val)
    for k, arr in extras.items():
        results["A1"][f"global_levels_{k}_vs_sr"] = _corr(arr, sr)

    # ---------- A1.2: rolling Pearson ----------
    rw = []
    half = window // 2
    for i in range(len(steps)):
        lo = max(0, i - half)
        hi = min(len(steps), i + half + 1)
        if hi - lo >= max(5, window // 2):
            c = _corr(obs[lo:hi], sr[lo:hi])
            rw.append((int(steps[i]), c["pearson"], c["n"]))
        else:
            rw.append((int(steps[i]), None, hi - lo))
    results["A1"]["rolling_window"] = {"window": window, "series": rw}

    # ---------- A1.3: cross-correlation / lead-lag ----------
    lags, ccf = _ccf(obs, sr, max_lag=max_lag)
    # arg-max |ccf|; the strongest relation may be negative (obs drops while SR rises)
    absccf = np.where(np.isfinite(ccf), np.abs(ccf), -np.inf)
    k_best = int(lags[int(np.argmax(absccf))])
    results["A1"]["ccf"] = {
        "max_lag": max_lag,
        "lags": lags.tolist(),
        "ccf": [float(v) if np.isfinite(v) else None for v in ccf],
        "arg_max_abs_lag": k_best,
        "ccf_at_arg_max": float(ccf[int(np.argmax(absccf))]),
        "lag_sign_convention": "positive lag k => obs_s_theta leads SR by k steps",
    }

    # ---------- A2: stage-wise regression ----------
    n = len(steps)
    cuts = [(0, n // 3, "early"),
            (n // 3, 2 * n // 3, "mid"),
            (2 * n // 3, n, "late"),
            (0, n, "global")]
    stage_res = {}
    for lo, hi, name in cuts:
        stage_res[name] = {
            "step_range": [int(steps[lo]), int(steps[hi - 1])],
            "levels_SR_on_obs": _ols(obs[lo:hi], sr[lo:hi]),
            "diffs_dSR_on_dobs": _ols(d_obs[lo:hi], d_sr[lo:hi]),
        }
    results["A2"]["stages"] = stage_res

    # ---------- figure ----------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(steps, _zscore(obs), label=f"z({obs_key.split('/')[-1]})", color="#d62728")
    ax.plot(steps, _zscore(sr), label=f"z({sr_key.split('/')[-1]})", color="#1f77b4")
    if val is not None:
        ax.plot(steps, _zscore(val), label=f"z({val_key.split('/')[-1]})",
                color="#2ca02c", alpha=0.6)
    ax.set_title("A1 — z-scored trajectories")
    ax.set_xlabel("training step")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    rw_steps = [r[0] for r in rw]
    rw_vals = [r[1] if r[1] is not None else np.nan for r in rw]
    ax.plot(rw_steps, rw_vals, color="#9467bd")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(f"A1 — rolling Pearson(obs, SR), window={window}")
    ax.set_xlabel("training step (center)")
    ax.set_ylabel("r")
    ax.set_ylim(-1.05, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.stem(lags, ccf, basefmt=" ")
    ax.axvline(k_best, color="red", ls="--",
               label=f"argmax|ccf|: lag={k_best}")
    ax.set_title("A1 — cross-correlation (obs_s_theta vs SR)")
    ax.set_xlabel("lag k  (k>0: obs leads SR by k steps)")
    ax.set_ylabel("normalized cov")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    colors = {"early": "#1f77b4", "mid": "#ff7f0e", "late": "#2ca02c"}
    for name in ["early", "mid", "late"]:
        lo, hi, _ = cuts[["early", "mid", "late"].index(name)]
        ax.scatter(obs[lo:hi], sr[lo:hi], s=14, alpha=0.7,
                   color=colors[name], label=name)
        r = stage_res[name]["levels_SR_on_obs"]
        if r["slope"] is not None:
            xs = np.linspace(np.nanmin(obs[lo:hi]), np.nanmax(obs[lo:hi]), 50)
            ax.plot(xs, r["slope"] * xs + r["intercept"],
                    color=colors[name], lw=1.2)
    ax.set_title("A2 — stage-wise SR vs obs_s_theta")
    ax.set_xlabel(obs_key.split("/")[-1])
    ax.set_ylabel(sr_key.split("/")[-1])
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Temporal decoupling — {run_tag}", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"a1a2_{run_tag}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--log",
        default="/local_nvme/guanyiming/project/verl-agent/wandb/"
                "run-20260415_105029-lmlyvpa6/files/output.log",
        help="path to wandb output.log containing 'step:N - k:v ...' lines",
    )
    ap.add_argument("--csv", default=None,
                    help="optional CSV (wandb scan_history export); overrides --log")
    ap.add_argument("--tag", default="observe_lmlyvpa6")
    ap.add_argument("--obs_key", default="observe/obs_s_theta_mean_mean")
    ap.add_argument("--sr_key", default="episode/success_rate")
    ap.add_argument("--val_key", default="val/success_rate")
    ap.add_argument("--window", type=int, default=30,
                    help="rolling window size (training steps)")
    ap.add_argument("--max_lag", type=int, default=30,
                    help="maximum lag magnitude for CCF")
    args = ap.parse_args()

    if args.csv:
        print(f"[A1A2] reading csv {args.csv}")
        df = pd.read_csv(args.csv)
        if "_step" in df.columns:
            df = df.dropna(subset=["_step"]).sort_values("_step").reset_index(drop=True)
            df = df.set_index(df["_step"].astype(int).rename("step"))
    else:
        print(f"[A1A2] parsing {args.log}")
        df = parse_output_log(args.log)
    print(f"[A1A2] shape={df.shape} steps=[{df.index.min()}..{df.index.max()}]")
    if df.empty:
        raise SystemExit("empty dataframe — cannot run analysis")

    extras = {}
    for k in ["observe/obs_wm_s_mean", "observe/obs_wm_s_B_mean",
              "observe/obs_delta_s_mean_mean",
              "observe/obs_step_entropy_mean_mean"]:
        if k in df.columns:
            extras[k] = df[k].to_numpy(dtype=float)

    val_key = args.val_key if args.val_key in df.columns else None
    results = run(df, args.tag,
                  obs_key=args.obs_key, sr_key=args.sr_key,
                  val_key=val_key, extras=extras,
                  window=args.window, max_lag=args.max_lag)

    out_path = OUT_DIR / f"a1a2_{args.tag}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[A1A2] wrote {out_path}")
    print(f"[A1A2] wrote {OUT_DIR / f'a1a2_{args.tag}.png'}")

    # short text digest to stdout
    print("\n=== A1 (global) ===")
    for k, v in results["A1"].items():
        if k in ("rolling_window", "ccf"):
            continue
        print(f"  {k}: n={v['n']}  r_pearson={v['pearson']}  r_spearman={v['spearman']}")
    ccf = results["A1"]["ccf"]
    print(f"  CCF argmax|.|: lag={ccf['arg_max_abs_lag']} (ccf={ccf['ccf_at_arg_max']:+.3f})")
    print("\n=== A2 (stage-wise SR ~ obs_s_theta) ===")
    for name in ["early", "mid", "late", "global"]:
        r = results["A2"]["stages"][name]["levels_SR_on_obs"]
        rng = results["A2"]["stages"][name]["step_range"]
        print(f"  {name:6s} steps={rng[0]:>4d}..{rng[1]:<4d}  "
              f"n={r['n']:>3d}  slope={r['slope']}  R^2={r['r2']}")


if __name__ == "__main__":
    main()
