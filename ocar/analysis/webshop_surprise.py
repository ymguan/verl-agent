"""Analyze surprise signal dynamics on Webshop grpo+observe run.

Compares against ALFWorld observe run (§2, §6) to see if the 3-component
structure ($f_\\text{text}$/$g(\\theta)$/$h(\\theta,o_t)$) transfers.
"""
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

CSV = Path(__file__).parent.parent / "analysis_results" / "webshop" / "history_full.csv"

def main():
    df = pd.read_csv(CSV).sort_values("_step").reset_index(drop=True)
    step = df["_step"].values
    print(f"=== Webshop grpo+observe (42rxhh6f) — {len(df)} logged points, step {step.min()}→{step.max()} ===\n")

    # ── 1. Signal magnitude trajectory ──
    print("── 1. Raw $s_\\theta$, $s_\\text{ref}$, $\\Delta S$ over training ──")
    cols = {
        "s_theta": "observe/obs_s_theta_mean_mean",
        "s_ref":   "observe/obs_s_ref_mean_mean",
        "delta_s": "observe/obs_delta_s_mean_mean",
        "delta_s_std": "observe/obs_delta_s_mean_std",
        "consec_s": "observe/obs_consecutive_s_mean",
        "wm_s":   "observe/obs_wm_s_mean",
        "wm_s_B": "observe/obs_wm_s_B_mean",
        "ent":    "observe/obs_step_entropy_mean_mean",
        "sr":     "episode/success_rate",
        "reward": "episode/reward/mean",
        "val_sr": "val/success_rate",
        "actor_ent": "actor/entropy_loss",
        "kl":     "actor/kl_loss",
    }
    print(f"{'step':>6} {'s_theta':>8} {'s_ref':>8} {'ΔS':>8} {'ΔS_std':>8} "
          f"{'consec':>8} {'wm_s':>8} {'ent':>6} {'SR':>6} {'reward':>7} {'val_SR':>7}")
    for i in range(len(df)):
        row = df.iloc[i]
        print(f"{int(row['_step']):>6} "
              f"{row[cols['s_theta']]:>8.3f} {row[cols['s_ref']]:>8.3f} "
              f"{row[cols['delta_s']]:>+8.3f} {row[cols['delta_s_std']]:>8.3f} "
              f"{row[cols['consec_s']]:>+8.3f} "
              f"{row[cols['wm_s']]:>8.3f} {row[cols['ent']]:>6.3f} "
              f"{row[cols['sr']]:>6.3f} {row[cols['reward']]:>7.3f} "
              f"{row[cols['val_sr']] if not pd.isna(row[cols['val_sr']]) else -1:>7.3f}")

    # ── 2. U-shape check for obs NLL ──
    s_theta = df[cols["s_theta"]].values
    sr = df[cols["sr"]].values
    print(f"\n── 2. Obs NLL trajectory ──")
    print(f"  s_theta @ step {step[0]} : {s_theta[0]:.3f}")
    print(f"  s_theta min          : {s_theta.min():.3f} @ step {step[np.argmin(s_theta)]}")
    print(f"  s_theta @ step {step[-1]}: {s_theta[-1]:.3f}")
    print(f"  range: {s_theta.max()-s_theta.min():.3f}")
    if s_theta.min() < s_theta[0] and s_theta.min() < s_theta[-1]:
        print(f"  ✅ U-shape: low in middle")
    elif s_theta[-1] < s_theta[0]:
        print(f"  monotone decrease")
    else:
        print(f"  monotone increase / irregular")

    # ── 3. ΔS cold-start and growth ──
    delta_s = df[cols["delta_s"]].values
    delta_s_std = df[cols["delta_s_std"]].values
    print(f"\n── 3. $\\Delta S$ trajectory ──")
    print(f"  ΔS @ step {step[0]}  : {delta_s[0]:+.3f} (std={delta_s_std[0]:.3f})")
    print(f"  ΔS @ step {step[len(step)//4]}: {delta_s[len(step)//4]:+.3f} (std={delta_s_std[len(step)//4]:.3f})")
    print(f"  ΔS @ step {step[len(step)//2]}: {delta_s[len(step)//2]:+.3f} (std={delta_s_std[len(step)//2]:.3f})")
    print(f"  ΔS @ step {step[-1]}: {delta_s[-1]:+.3f} (std={delta_s_std[-1]:.3f})")
    print(f"  ΔS sign: {'negative throughout (s_theta < s_ref)' if np.all(delta_s<0) else 'mixed/positive'}")

    # ── 4. Success / failure separation (within-batch) ──
    succ_s = df["observe/success_s_theta_mean"].dropna().values
    fail_s = df["observe/failure_s_theta_mean"].dropna().values
    # align on step that has both
    mask = (~df["observe/success_s_theta_mean"].isna()) & (~df["observe/failure_s_theta_mean"].isna())
    both = df.loc[mask]
    succ_paired = both["observe/success_s_theta_mean"].values
    fail_paired = both["observe/failure_s_theta_mean"].values
    gap = succ_paired - fail_paired
    print(f"\n── 4. Within-batch succ vs fail s_theta gap (n={len(gap)} batches with both) ──")
    print(f"  gap mean : {gap.mean():+.3f}  std={gap.std():.3f}")
    print(f"  gap sign distribution: {(gap>0).sum()}+ / {(gap<0).sum()}− / {(gap==0).sum()}=")
    print(f"  early batches (first 1/3): gap mean = {gap[:len(gap)//3].mean():+.3f}")
    print(f"  late  batches (last  1/3): gap mean = {gap[-len(gap)//3:].mean():+.3f}")

    # entropy gap
    es = both["observe/success_entropy_mean"].values
    ef = both["observe/failure_entropy_mean"].values
    egap = es - ef
    print(f"  entropy gap mean: {egap.mean():+.3f}")

    # ── 5. Correlation with training-level SR ──
    print(f"\n── 5. Training-level correlations (n={len(df)}) ──")
    for n1, n2 in [
        ("s_theta","sr"),("s_theta","reward"),
        ("delta_s","sr"),("delta_s","reward"),
        ("consec_s","sr"),
        ("ent","sr"),
        ("wm_s","sr"),("wm_s_B","sr"),
        ("s_theta","val_sr"),("delta_s","val_sr"),
    ]:
        a = df[cols[n1]].values
        b = df[cols[n2]].values
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 5:
            print(f"  {n1:>10} vs {n2:>10}: (n={m.sum()} insufficient)")
            continue
        rs, ps = spearmanr(a[m], b[m])
        rp, pp = pearsonr(a[m], b[m])
        print(f"  {n1:>10} vs {n2:>10}: Spearman={rs:+.3f} (p={ps:.3f})  Pearson={rp:+.3f}")

    # ── 6. Stage-wise ΔS growth (cold start?) ──
    n = len(df)
    early = slice(0, n//3)
    mid   = slice(n//3, 2*n//3)
    late  = slice(2*n//3, n)
    print(f"\n── 6. Stage-wise stats ──")
    for name, sl in [("early", early), ("mid", mid), ("late", late)]:
        s = df[cols["s_theta"]].iloc[sl].values
        d = df[cols["delta_s"]].iloc[sl].values
        ds = df[cols["delta_s_std"]].iloc[sl].values
        srv = df[cols["sr"]].iloc[sl].values
        print(f"  {name:5s} step {step[sl][0]:3d}-{step[sl][-1]:3d}: "
              f"s_theta={s.mean():.3f}  ΔS={d.mean():+.3f}  ΔS_std={ds.mean():.3f}  "
              f"SR={srv.mean():.3f}")

if __name__ == "__main__":
    main()
