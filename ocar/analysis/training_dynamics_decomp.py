"""Training-dynamics analysis of ΔS signal.

Uses wm_degradation_results.json (20 fixed trajs × 6 checkpoints: base + step 50/75/100/125/150)
to decompose where the ΔS signal's information content actually comes from during training.

Questions:
  Q1. Cross-checkpoint Spearman of per-traj obs_nll — is traj ranking stable across training?
      (If yes → most variance is "traj-intrinsic"; training drift is just a uniform shift)
  Q2. Per-checkpoint succ/fail AUC on obs_nll — when does outcome info emerge/disappear?
  Q3. Δ_ckpt = obs_nll(step_k) - obs_nll(base) — is this Δ a better outcome predictor than raw?
      (This is the training-induced drift component, which §12 claims is the only useful source)
  Q4. Variance decomposition: obs_nll total variance split into
      (a) between-traj  (b) between-checkpoint  (c) traj × ckpt interaction
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).parent.parent
DATA = ROOT / "wm_degradation_results.json"

def auc_mwu(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return None, None
    u, p = mannwhitneyu(pos, neg, alternative="two-sided")
    return u / (len(pos) * len(neg)), p

def main():
    with open(DATA) as f:
        ckpts = json.load(f)
    labels = ["base", "step50", "step75", "step100", "step125", "step150"]
    assert len(ckpts) == len(labels)

    # Build (n_ckpt, n_traj) matrix keyed by traj_id
    traj_ids = [t["traj_id"] for t in ckpts[0]["per_traj"]]
    tid2succ = {t["traj_id"]: t["success"] for t in ckpts[0]["per_traj"]}
    mat = np.zeros((len(ckpts), len(traj_ids)))
    for i, ck in enumerate(ckpts):
        by = {t["traj_id"]: t["obs_nll"] for t in ck["per_traj"]}
        for j, tid in enumerate(traj_ids):
            mat[i, j] = by[tid]
    succ_mask = np.array([tid2succ[tid] for tid in traj_ids], dtype=bool)

    # ── Q1: cross-checkpoint Spearman ──
    print("\n=== Q1: Spearman of per-traj obs_nll across checkpoints (n=20) ===")
    print(f"{'':10s} " + " ".join(f"{l:>8s}" for l in labels))
    for i, a in enumerate(labels):
        row = [f"{a:10s}"]
        for j, b in enumerate(labels):
            if i == j:
                row.append(f"{'1.000':>8s}")
            else:
                r, _ = spearmanr(mat[i], mat[j])
                row.append(f"{r:>8.3f}")
        print(" ".join(row))

    # ── Q2: per-ckpt succ/fail AUC ──
    print("\n=== Q2: per-checkpoint obs_nll succ/fail AUC (n=10+10) ===")
    print(f"{'ckpt':10s} {'succ_mean':>10s} {'fail_mean':>10s} {'gap':>8s} {'AUC':>8s} {'p':>8s}")
    for i, l in enumerate(labels):
        s = mat[i, succ_mask]; f = mat[i, ~succ_mask]
        auc, p = auc_mwu(s, f)
        print(f"{l:10s} {s.mean():>10.3f} {f.mean():>10.3f} "
              f"{s.mean()-f.mean():>+8.3f} {auc:>8.3f} {p:>8.3f}")

    # ── Q3: Δ_ckpt = step_k - base ──
    print("\n=== Q3: Δ(step_k − base) as 'training-induced drift' signal ===")
    print(f"{'delta':20s} {'mean':>8s} {'succ_mean':>10s} {'fail_mean':>10s} {'gap':>8s} {'AUC':>8s} {'p':>8s}")
    base = mat[0]
    for i in range(1, len(labels)):
        delta = mat[i] - base
        s = delta[succ_mask]; f = delta[~succ_mask]
        auc, p = auc_mwu(s, f)
        print(f"{labels[i]+' − base':20s} {delta.mean():>+8.3f} "
              f"{s.mean():>+10.3f} {f.mean():>+10.3f} "
              f"{s.mean()-f.mean():>+8.3f} {auc:>8.3f} {p:>8.3f}")

    print("\n=== Q3b: Δ(step_k − step_{k-1}) — consecutive-ckpt drift ===")
    print(f"{'delta':20s} {'mean':>8s} {'gap':>8s} {'AUC':>8s} {'p':>8s}")
    for i in range(1, len(labels)):
        delta = mat[i] - mat[i-1]
        s = delta[succ_mask]; f = delta[~succ_mask]
        auc, p = auc_mwu(s, f)
        print(f"{labels[i]+' − '+labels[i-1]:20s} {delta.mean():>+8.3f} "
              f"{s.mean()-f.mean():>+8.3f} {auc:>8.3f} {p:>8.3f}")

    # ── Q4: variance decomposition ──
    print("\n=== Q4: variance decomposition of obs_nll (6 ckpts × 20 trajs) ===")
    gm = mat.mean()
    ckpt_mean = mat.mean(axis=1, keepdims=True)  # (6,1)
    traj_mean = mat.mean(axis=0, keepdims=True)  # (1,20)
    ss_tot = ((mat - gm) ** 2).sum()
    ss_ckpt = 20 * ((ckpt_mean - gm) ** 2).sum()
    ss_traj = 6 * ((traj_mean - gm) ** 2).sum()
    ss_int = ss_tot - ss_ckpt - ss_traj
    print(f"  total SS         : {ss_tot:7.2f}")
    print(f"  between-traj     : {ss_traj:7.2f}  ({100*ss_traj/ss_tot:5.1f}%)  <- traj-intrinsic (text difficulty)")
    print(f"  between-ckpt     : {ss_ckpt:7.2f}  ({100*ss_ckpt/ss_tot:5.1f}%)  <- uniform training drift")
    print(f"  traj × ckpt      : {ss_int:7.2f}  ({100*ss_int/ss_tot:5.1f}%)  <- 'useful' drift signal")

    # ── Q5: Is the "drift rank" (within-traj time ordering) preserved? ──
    # i.e. does step 100's traj-ordering still resemble base's?
    print("\n=== Q5: rank stability — corr of step_k ranks vs base ranks per succ/fail subset ===")
    for i, l in enumerate(labels[1:], 1):
        r_all, _ = spearmanr(mat[0], mat[i])
        r_s, _ = spearmanr(mat[0, succ_mask], mat[i, succ_mask])
        r_f, _ = spearmanr(mat[0, ~succ_mask], mat[i, ~succ_mask])
        print(f"  {l:10s} all={r_all:+.3f}  succ={r_s:+.3f}  fail={r_f:+.3f}")

if __name__ == "__main__":
    main()
