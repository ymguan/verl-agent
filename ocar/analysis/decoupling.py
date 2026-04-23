"""E3.4: Temporal decoupling of obs_s_theta vs SR.

Uses local wandb output.log (no API call needed).

Computes:
  - Pearson/Spearman correlations between grounding signals and SR
  - reach-step (x% of total change) to establish lead/lag
  - endpoint summary
"""
import argparse
import json
from pathlib import Path
import numpy as np
from parse_wandb_log import parse_output_log


OUT_DIR = Path(__file__).parent.parent / "analysis_results"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def reach_step(values, steps, fraction, direction, smooth_w=5, end_w=10):
    """Step at which a smoothed signal first reaches `fraction` of (start → end) change.
    start = mean of first `smooth_w` finite values
    end   = mean of last `end_w` finite values
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(steps, dtype=float)
    mask = np.isfinite(v)
    v, s = v[mask], s[mask]
    if len(v) < smooth_w + end_w:
        return None
    start = v[:smooth_w].mean()
    end   = v[-end_w:].mean()
    # Smoothed trajectory for crossing detection
    v_ma = np.array([v[max(0, i - smooth_w + 1):i + 1].mean() for i in range(len(v))])
    if direction == "down":
        target = start - fraction * (start - end)
        hit = np.where(v_ma <= target)[0]
    else:
        target = start + fraction * (end - start)
        hit = np.where(v_ma >= target)[0]
    if len(hit) == 0:
        return None
    return float(s[hit[0]])


def moving_avg(x, w=5):
    x = np.asarray(x, dtype=float)
    if len(x) < w:
        return x
    # Simple nan-aware MA: use nanmean over window
    out = np.full_like(x, np.nan)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        seg = x[lo:i + 1]
        seg = seg[np.isfinite(seg)]
        if len(seg) > 0:
            out[i] = seg.mean()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="/local_nvme/guanyiming/project/verl-agent/wandb/run-20260415_105029-lmlyvpa6/files/output.log")
    ap.add_argument("--tag", default="observe_lmlyvpa6")
    args = ap.parse_args()

    df = parse_output_log(args.log)
    if df.empty:
        print("no data"); return
    steps = df.index.to_numpy()
    print(f"[parsed] {len(df)} steps, {df.shape[1]} keys")

    keys = {
        "obs_s_theta": "observe/obs_s_theta_mean_mean",
        "wm_A":       "observe/obs_wm_s_mean",
        "wm_B":       "observe/obs_wm_s_B_mean",
        "delta_s":    "observe/obs_delta_s_mean_mean",
        "entropy":    "observe/obs_step_entropy_mean_mean",
        "train_sr":   "episode/success_rate",
        "val_sr":     "val/success_rate",
    }
    data = {}
    for name, col in keys.items():
        if col in df.columns:
            data[name] = df[col].to_numpy()
        else:
            print(f"  missing: {col}")
            data[name] = np.full(len(df), np.nan)
    data["wm_gap"] = data["wm_B"] - data["wm_A"]

    from scipy.stats import pearsonr, spearmanr

    def corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        n = int(mask.sum())
        if n < 3:
            return dict(n=n, pearson=None, spearman=None)
        pr, pp = pearsonr(a[mask], b[mask])
        sr, sp = spearmanr(a[mask], b[mask])
        return dict(n=n, pearson=float(pr), pearson_p=float(pp),
                    spearman=float(sr), spearman_p=float(sp))

    # Use smoothed SR (train_sr is very noisy batch-to-batch)
    train_sr_sm = moving_avg(data["train_sr"], 5)

    print("\n[correlation vs train_sr (5-step MA)]")
    results = {"run": args.tag, "n_steps": int(len(df)), "correlations": {}, "reach_steps": {}, "endpoints": {}}
    def fmt(c):
        pr = c.get("pearson"); sr = c.get("spearman"); pp = c.get("pearson_p")
        pr_s = f"{pr:+.3f}" if pr is not None else "  n/a"
        sr_s = f"{sr:+.3f}" if sr is not None else "  n/a"
        pp_s = f"{pp:.1e}" if pp is not None else " n/a"
        return pr_s, sr_s, pp_s

    for name in ["obs_s_theta", "wm_A", "wm_B", "wm_gap", "delta_s", "entropy"]:
        c = corr(data[name], train_sr_sm)
        results["correlations"][name] = c
        pr_s, sr_s, pp_s = fmt(c)
        print(f"  {name:13s}  n={c['n']:3d}  pearson={pr_s} (p={pp_s})  spearman={sr_s}")

    print("\n[correlation vs val_sr]")
    results["correlations_vs_val"] = {}
    for name in ["obs_s_theta", "wm_A", "wm_gap"]:
        c = corr(data[name], data["val_sr"])
        results["correlations_vs_val"][name] = c
        pr_s, sr_s, _ = fmt(c)
        print(f"  {name:13s}  n={c['n']:3d}  pearson={pr_s}  spearman={sr_s}")

    # Lead/lag via reach-step
    print("\n[reach-step 50% / 80%]")
    directions = {"obs_s_theta": "down", "wm_A": "down", "wm_B": "down",
                  "wm_gap": "up", "delta_s": "down", "entropy": "up",
                  "train_sr": "up", "val_sr": "up"}
    for name, d in directions.items():
        r50 = reach_step(data[name], steps, 0.5, d)
        r80 = reach_step(data[name], steps, 0.8, d)
        results["reach_steps"][name] = {"r50": r50, "r80": r80, "direction": d}
        print(f"  {name:13s}  dir={d:4s}  50%-step={r50}  80%-step={r80}")

    # Summary lead/lag
    r_obs = results["reach_steps"]["obs_s_theta"]["r50"]
    r_gap = results["reach_steps"]["wm_gap"]["r50"]
    r_tr  = results["reach_steps"]["train_sr"]["r50"]
    r_val = results["reach_steps"]["val_sr"]["r50"]
    if r_obs and r_tr:
        print(f"\n  train_SR[50%] - obs_s_theta[50%] = {r_tr - r_obs:+.0f} steps  (positive = grounding leads SR)")
        results["lead_lag_obs_vs_train_sr"] = r_tr - r_obs
    if r_gap and r_val:
        print(f"  val_SR[50%]   - wm_gap[50%]     = {r_val - r_gap:+.0f} steps")
        results["lead_lag_wmgap_vs_val_sr"] = r_val - r_gap

    # Endpoints
    for name in keys.keys():
        a = data[name][np.isfinite(data[name])]
        if len(a) == 0:
            continue
        results["endpoints"][name] = {"first": float(a[0]), "last": float(a[-1]),
                                       "min": float(a.min()), "max": float(a.max())}

    # Save time series too
    ts = {"step": steps.tolist()}
    for name in keys.keys():
        ts[name] = [None if not np.isfinite(v) else float(v) for v in data[name]]
    ts["wm_gap"] = [None if not np.isfinite(v) else float(v) for v in data["wm_gap"]]
    results["time_series"] = ts

    out = OUT_DIR / f"decoupling_{args.tag}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[wrote] {out}")


if __name__ == "__main__":
    main()
