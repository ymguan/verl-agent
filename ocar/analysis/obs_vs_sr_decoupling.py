"""E3.4: obs_s_theta vs SR temporal decoupling analysis.

Pulls step-wise metrics from wandb and computes:
  - Pearson / Spearman correlation between obs_s_theta and success_rate
  - "50%-reach step": step at which each metric reaches 50% of its final change
  - Lead/lag: does grounding (obs_s_theta drop) precede policy improvement (SR rise)?

Usage:
    WANDB_API_KEY=xxx python ocar/analysis/obs_vs_sr_decoupling.py \
        --run grpo_observe_alfworld_20260415_104816/lmlyvpa6
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).parent.parent / "analysis_results"
OUT_DIR.mkdir(exist_ok=True, parents=True)


def fetch_wandb_history(entity: str, project: str, run_id: str, keys):
    import wandb
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    hist = run.history(keys=["_step"] + list(keys), pandas=True)
    return hist


def reach_step(values, steps, target_fraction, monotone_dir="down"):
    """Step at which a signal first reaches `target_fraction` of its total change.

    monotone_dir="down": signal should decrease. target = start - frac*(start-end)
    monotone_dir="up":   signal should increase. target = start + frac*(end-start)
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(steps, dtype=float)
    mask = np.isfinite(v)
    v, s = v[mask], s[mask]
    if len(v) < 2:
        return None
    start, end = v[0], v[-1]
    if monotone_dir == "down":
        target = start - target_fraction * (start - end)
        hit = np.where(v <= target)[0]
    else:
        target = start + target_fraction * (end - start)
        hit = np.where(v >= target)[0]
    if len(hit) == 0:
        return None
    return float(s[hit[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entity", default="guanyiming290-alibaba")
    ap.add_argument("--project", default="grpo_observe_alfworld_20260415_104816")
    ap.add_argument("--run_id", default="lmlyvpa6")
    ap.add_argument("--obs_key", default="observe/obs_s_theta_mean_mean")
    ap.add_argument("--sr_key", default="episode/success_rate")
    ap.add_argument("--val_sr_key", default="val/success_rate")
    ap.add_argument("--wm_a_key", default="observe/obs_wm_s_mean")
    ap.add_argument("--wm_b_key", default="observe/obs_wm_s_B_mean")
    ap.add_argument("--entropy_key", default="observe/obs_step_entropy_mean_mean")
    args = ap.parse_args()

    keys = [args.obs_key, args.sr_key, args.val_sr_key,
            args.wm_a_key, args.wm_b_key, args.entropy_key]
    print(f"[wandb] fetching {args.entity}/{args.project}/{args.run_id}")
    print(f"[wandb] keys={keys}")
    df = fetch_wandb_history(args.entity, args.project, args.run_id, keys)
    print(f"[wandb] {len(df)} rows")
    print(df.head())

    # Attempt correlations (pairwise drop-NaN)
    from scipy.stats import pearsonr, spearmanr

    def corr(a, b, label_a, label_b):
        mask = np.isfinite(a) & np.isfinite(b)
        n = mask.sum()
        if n < 3:
            print(f"  [{label_a} vs {label_b}] insufficient data (n={n})")
            return None
        pr, pp = pearsonr(a[mask], b[mask])
        sr, sp = spearmanr(a[mask], b[mask])
        print(f"  [{label_a:30s} vs {label_b:25s}] n={n:3d}  pearson={pr:+.3f} (p={pp:.1e})  spearman={sr:+.3f} (p={sp:.1e})")
        return dict(n=int(n), pearson=float(pr), pearson_p=float(pp),
                    spearman=float(sr), spearman_p=float(sp))

    print("\n[correlations]")
    obs = df[args.obs_key].to_numpy()
    sr  = df[args.sr_key].to_numpy()
    valsr = df[args.val_sr_key].to_numpy()
    wmA = df[args.wm_a_key].to_numpy()
    wmB = df[args.wm_b_key].to_numpy()
    ent = df[args.entropy_key].to_numpy()
    steps = df["_step"].to_numpy()

    results = {
        "run": f"{args.entity}/{args.project}/{args.run_id}",
        "n_steps": int(len(df)),
        "correlations": {},
        "reach_steps": {},
        "endpoints": {},
    }

    results["correlations"]["obs_s_theta__vs__train_sr"] = corr(obs, sr,  "obs_s_theta", "train_ep_sr")
    results["correlations"]["obs_s_theta__vs__val_sr"]   = corr(obs, valsr,"obs_s_theta", "val_sr")
    results["correlations"]["wm_A__vs__train_sr"]        = corr(wmA, sr,  "wm_A",        "train_ep_sr")
    results["correlations"]["wm_B__vs__train_sr"]        = corr(wmB, sr,  "wm_B",        "train_ep_sr")
    results["correlations"]["entropy__vs__train_sr"]     = corr(ent, sr,  "entropy",     "train_ep_sr")
    results["correlations"]["wm_gap__vs__train_sr"]      = corr(wmB - wmA, sr, "wm_B - wm_A", "train_ep_sr")

    # Reach steps — obs goes down, SR goes up
    print("\n[reach-50% step]")
    for label, vals, direction in [
        ("obs_s_theta", obs, "down"),
        ("wm_A",       wmA, "down"),
        ("wm_B - wm_A",wmB - wmA, "up"),
        ("train_ep_sr", sr, "up"),
        ("val_sr",     valsr, "up"),
    ]:
        r50 = reach_step(vals, steps, 0.5, direction)
        r80 = reach_step(vals, steps, 0.8, direction)
        print(f"  {label:15s}  50%-reach={r50}  80%-reach={r80}")
        results["reach_steps"][label] = {"r50": r50, "r80": r80}

    # Endpoints
    def endpoint(a, name):
        a = np.asarray(a, dtype=float)
        a = a[np.isfinite(a)]
        if len(a) == 0:
            return None
        return {"first": float(a[0]), "last": float(a[-1]), "min": float(a.min()), "max": float(a.max())}
    for k, a in [("obs_s_theta", obs), ("wm_A", wmA), ("wm_B", wmB),
                 ("entropy", ent), ("train_sr", sr), ("val_sr", valsr)]:
        results["endpoints"][k] = endpoint(a, k)

    # Lead/lag summary
    r_obs = results["reach_steps"]["obs_s_theta"]["r50"]
    r_sr  = results["reach_steps"]["train_ep_sr"]["r50"]
    if r_obs is not None and r_sr is not None:
        lead = r_sr - r_obs
        print(f"\n[lead/lag] obs_s_theta leads SR by {lead:+.0f} steps (positive = grounding precedes success)")
        results["lead_lag"] = {"obs_minus_sr_at_r50": lead}

    out_path = OUT_DIR / f"decoupling_{args.run_id}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[wrote] {out_path}")


if __name__ == "__main__":
    main()
