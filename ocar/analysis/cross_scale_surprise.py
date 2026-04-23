"""Cross-scale analysis of obs-NLL surprise signals.

Uses the 4 scale_scan JSONs (same 12 trajectories scored by different models)
to answer:
  Q1. Are surprise signals *consistent across model scales*?
      (Spearman corr of per-step obs_nll between models)
  Q2. How much variance is model-dependent vs step-dependent?
      (two-way ANOVA-like decomposition)
  Q3. Does success/failure discrimination scale with model size?
  Q4. Are delta-type signals (model - model) cleaner than raw?
  Q5. wm_gap vs scale: is the "context marginal value" monotone?
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr

OUT = Path(__file__).parent.parent / "analysis_results" / "scale_scan"
FILES = {
    "Qwen2.5-0.5B":      OUT / "_local_nvme_rs_models_Qwen2.5-0.5B-Instruct.json",
    "Qwen2.5-7B-Inst":   OUT / "_local_nvme_guanyiming_models_Qwen_Qwen2.5-7B-Instruct.json",
    "Qwen3-8B":          OUT / "Qwen_Qwen3-8B.json",
    "Qwen3-14B":         OUT / "Qwen_Qwen3-14B.json",
}

def flatten(data, key):
    """Return (step_vec, traj_id_vec, step_idx_vec, success_vec)."""
    vals, tid, sidx, succ = [], [], [], []
    for t in data["per_traj"]:
        for i, v in enumerate(t[key]):
            vals.append(v)
            tid.append(t["idx"])
            sidx.append(i)
            succ.append(int(t["success"]))
    return (np.array(vals), np.array(tid), np.array(sidx), np.array(succ))

def main():
    models = {}
    for name, path in FILES.items():
        with open(path) as f:
            models[name] = json.load(f)

    # ── Q1: cross-scale consistency of raw obs_nll ──
    print("\n=== Q1: Spearman corr of per-step obs_nll across scales (n=93) ===")
    obs_matrix = {}
    for n, d in models.items():
        obs_matrix[n], tid, sidx, succ = flatten(d, "obs_nll_last")
    names = list(obs_matrix.keys())
    print(f"{'':20s} " + " ".join(f"{n:>15s}" for n in names))
    for a in names:
        row = [f"{a:20s}"]
        for b in names:
            if a == b:
                row.append(f"{'1.000':>15s}")
            else:
                r, _ = spearmanr(obs_matrix[a], obs_matrix[b])
                row.append(f"{r:>15.3f}")
        print(" ".join(row))

    # ── Q2: variance decomposition ──
    # Stack: each step has 4 obs_nll values (one per model)
    stacked = np.stack([obs_matrix[n] for n in names])  # (4, 93)
    grand_mean = stacked.mean()
    model_mean = stacked.mean(axis=1, keepdims=True)    # (4,1) — per-model baseline
    step_mean  = stacked.mean(axis=0, keepdims=True)    # (1,93) — per-step avg across models
    ss_total = ((stacked - grand_mean) ** 2).sum()
    ss_model = 93 * ((model_mean - grand_mean) ** 2).sum()
    ss_step  = 4  * ((step_mean  - grand_mean) ** 2).sum()
    ss_resid = ss_total - ss_model - ss_step
    print(f"\n=== Q2: variance decomposition of obs_nll (total={ss_total:.1f}) ===")
    print(f"  between-model    : {ss_model:7.1f}  ({100*ss_model/ss_total:5.1f}%)")
    print(f"  between-step     : {ss_step:7.1f}  ({100*ss_step/ss_total:5.1f}%)  <- 'signal' component")
    print(f"  residual (inter) : {ss_resid:7.1f}  ({100*ss_resid/ss_total:5.1f}%)  <- model-specific noise")

    # ── Q3: per-model success/failure discrimination (AUC via Mann-Whitney) ──
    from scipy.stats import mannwhitneyu
    print("\n=== Q3: per-step succ vs fail obs_nll (step-level label = traj outcome) ===")
    print(f"{'model':20s} {'succ_mean':>10s} {'fail_mean':>10s} {'gap':>8s} {'U-AUC':>8s} {'p':>8s}")
    for n in names:
        v = obs_matrix[n]
        v_s = v[succ == 1]; v_f = v[succ == 0]
        # AUC = U / (n1*n2)
        u, p = mannwhitneyu(v_s, v_f, alternative="two-sided")
        auc = u / (len(v_s) * len(v_f))
        print(f"{n:20s} {v_s.mean():>10.3f} {v_f.mean():>10.3f} "
              f"{v_s.mean()-v_f.mean():>+8.3f} {auc:>8.3f} {p:>8.3f}")

    # ── Q4: delta signals (larger - smaller) vs raw ──
    print("\n=== Q4: Δ-NLL across scales as a 'de-biased' surprise ===")
    print("     (ref = smaller model, query = larger model; Δ = query - ref)")
    pairs = [("Qwen2.5-0.5B","Qwen2.5-7B-Inst"),
             ("Qwen2.5-0.5B","Qwen3-14B"),
             ("Qwen2.5-7B-Inst","Qwen3-14B")]
    for ref, qry in pairs:
        delta = obs_matrix[qry] - obs_matrix[ref]
        v_s = delta[succ == 1]; v_f = delta[succ == 0]
        u, p = mannwhitneyu(v_s, v_f, alternative="two-sided")
        auc = u / (len(v_s) * len(v_f))
        # also raw
        raw = obs_matrix[qry]
        v_s_r = raw[succ==1]; v_f_r = raw[succ==0]
        u_r, p_r = mannwhitneyu(v_s_r, v_f_r, alternative="two-sided")
        auc_r = u_r / (len(v_s_r) * len(v_f_r))
        print(f"  Δ({qry} - {ref}): mean={delta.mean():+.3f}  "
              f"succ-fail={v_s.mean()-v_f.mean():+.3f}  AUC={auc:.3f} (raw_qry AUC={auc_r:.3f})")

    # ── Q5: wm_gap vs scale ──
    print("\n=== Q5: wm_gap (P(o|a) - P(o|ctx,a)) per model ===")
    for n, d in models.items():
        s = d["summary"]["wm_gap"]
        print(f"  {n:20s} mean={s['mean']:+.3f}  std={s['std']:.3f}")

    # ── Q6: within-traj std (signal strength regardless of mean) ──
    print("\n=== Q6: within-trajectory std of obs_nll (signal per traj) ===")
    print(f"{'model':20s} {'mean_within_std':>16s} {'over_trajs':>12s}")
    for n, d in models.items():
        stds = [np.std(t["obs_nll_last"][1:]) for t in d["per_traj"]]  # skip step 0 (often 0)
        print(f"  {n:20s} {np.mean(stds):>16.3f} {np.std(stds):>12.3f}")

    # ── Q7: canary probe cross-check ──
    print("\n=== Q7: canary orig/nonsense gap vs scale ===")
    canary_dir = OUT.parent.parent / "canary" / "results"
    canary_files = {
        "Qwen2.5-0.5B":    "_local_nvme_rs_models_Qwen2.5-0.5B-Instruct.json",
        "Qwen2.5-7B-Inst": "_local_nvme_guanyiming_models_Qwen_Qwen2.5-7B-Instruct.json",
        "Qwen3-8B":        "Qwen_Qwen3-8B.json",
        "Qwen3-14B":       "Qwen_Qwen3-14B.json",
    }
    for n, fname in canary_files.items():
        cpath = canary_dir / fname
        if not cpath.exists():
            print(f"  {n:20s} (missing)")
            continue
        with open(cpath) as f:
            c = json.load(f)
        from collections import defaultdict
        agg = defaultdict(list)
        for it in c.get("per_probe", []):
            agg[it["split"]].append(it["nll"])
        vals = {k: float(np.mean(v)) for k, v in agg.items() if v}
        orig = vals.get("original"); nons = vals.get("nonsense"); swap = vals.get("swapped"); shuf = vals.get("shuffled")
        if orig is not None and nons is not None:
            print(f"  {n:20s} orig={orig:.3f}  shuf={shuf:.3f}  swap={swap:.3f}  nons={nons:.3f}  "
                  f"nons-orig={nons-orig:+.3f}  swap-orig={swap-orig:+.3f}")

if __name__ == "__main__":
    main()
