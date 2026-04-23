"""Cross-generation × scale analysis of obs-NLL surprise.

Processes scale_scan JSONs for 10 post-trained Qwen models (Qwen2.5-Instruct
and Qwen3) across scales 0.5B/0.6B, 1.5B/1.7B, 3B/4B, 7B/8B, 14B/14B.

Goals:
  1. Replace the old 4-model analysis (confounded by generation/scale/SFT)
     with a clean 2-generation x 5-scale design.
  2. Two-way ANOVA-style decomposition of obs_nll variance.
  3. Succ/fail AUC per model + trend by (generation, scale).
  4. Cross-model Spearman consistency.

Outputs to: ocar/analysis_results/cross_gen_scale/report.md
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

ROOT = Path(__file__).parent.parent
SCAN_DIR = ROOT / "analysis_results" / "scale_scan"
OUT_DIR = ROOT / "analysis_results" / "cross_gen_scale"
OUT_DIR.mkdir(exist_ok=True, parents=True)

MODELS = [
    ("Qwen2.5-0.5B-Instruct", "Qwen2.5", 0.5),
    ("Qwen2.5-1.5B-Instruct", "Qwen2.5", 1.5),
    ("Qwen2.5-3B-Instruct",   "Qwen2.5", 3.0),
    ("Qwen2.5-7B-Instruct",   "Qwen2.5", 7.0),
    ("Qwen2.5-14B-Instruct",  "Qwen2.5", 14.0),
    ("Qwen3-0.6B",            "Qwen3",   0.6),
    ("Qwen3-1.7B",            "Qwen3",   1.7),
    ("Qwen3-4B",              "Qwen3",   4.0),
    ("Qwen3-8B",              "Qwen3",   8.0),
    ("Qwen3-14B",             "Qwen3",   14.0),
]


def find_scan(name):
    """Match scale_scan JSON filename (slug with / replaced by _)."""
    for p in SCAN_DIR.glob("*.json"):
        if name in p.stem:
            return p
    return None


def flatten(data, key):
    vals, tid, sidx, succ = [], [], [], []
    for t in data["per_traj"]:
        for i, v in enumerate(t[key]):
            vals.append(v)
            tid.append(t["idx"])
            sidx.append(i)
            succ.append(int(t["success"]))
    return np.array(vals), np.array(tid), np.array(sidx), np.array(succ)


def main():
    loaded = {}
    for name, gen, scale in MODELS:
        p = find_scan(name)
        if p is None:
            print(f"  [missing] {name}")
            continue
        loaded[name] = (json.load(open(p)), gen, scale)

    lines = []
    lines.append("# Cross-Generation × Scale Analysis (obs-NLL)\n")
    lines.append(f"Models loaded: {len(loaded)}/{len(MODELS)}\n")

    # ── Per-model summary + succ/fail AUC ──
    lines.append("## Per-model summary\n")
    lines.append("| model | gen | scale (B) | obs_nll_mean | succ | fail | gap | AUC | p |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    per_model_obs = {}
    for name, gen, scale in MODELS:
        if name not in loaded:
            continue
        data, _, _ = loaded[name]
        vals, tid, sidx, succ = flatten(data, "obs_nll_last")
        per_model_obs[name] = (vals, succ, tid, sidx)
        v_s = vals[succ == 1]
        v_f = vals[succ == 0]
        if len(v_s) > 0 and len(v_f) > 0:
            u, p = mannwhitneyu(v_s, v_f, alternative="two-sided")
            auc = u / (len(v_s) * len(v_f))
        else:
            auc, p = float("nan"), float("nan")
        lines.append(f"| {name} | {gen} | {scale} | {vals.mean():.3f} | "
                     f"{v_s.mean():.3f} | {v_f.mean():.3f} | "
                     f"{v_s.mean() - v_f.mean():+.3f} | {auc:.3f} | {p:.3f} |")

    # ── Cross-model Spearman (all pairs) ──
    lines.append("\n## Spearman corr of per-step obs_nll across models\n")
    names = [n for n, _, _ in MODELS if n in loaded]
    lines.append("| model | " + " | ".join(names) + " |")
    lines.append("|---|" + "|".join([":-:"] * len(names)) + "|")
    for a in names:
        row = [a]
        va, _, _, _ = per_model_obs[a]
        for b in names:
            if a == b:
                row.append("1.000")
            else:
                vb, _, _, _ = per_model_obs[b]
                r, _ = spearmanr(va, vb)
                row.append(f"{r:.3f}")
        lines.append("| " + " | ".join(row) + " |")

    # ── Two-way ANOVA: generation × scale-bucket ──
    # Align obs matrix: shape (n_model, n_step)
    if len(names) >= 2:
        obs_matrix = np.stack([per_model_obs[n][0] for n in names])
        gens = np.array([loaded[n][1] for n in names])
        scales = np.array([loaded[n][2] for n in names])
        grand = obs_matrix.mean()
        model_mean = obs_matrix.mean(axis=1, keepdims=True)
        step_mean = obs_matrix.mean(axis=0, keepdims=True)
        n_step = obs_matrix.shape[1]
        ss_total = ((obs_matrix - grand) ** 2).sum()
        ss_model = n_step * ((model_mean - grand) ** 2).sum()
        ss_step = len(names) * ((step_mean - grand) ** 2).sum()
        ss_resid = ss_total - ss_model - ss_step

        # Further split ss_model into generation + scale + gen*scale residual
        gen_means = {g: obs_matrix[gens == g].mean() for g in np.unique(gens)}
        scale_means = {s: obs_matrix[scales == s].mean() for s in np.unique(scales)}
        ss_gen = sum(((gen_means[g] - grand) ** 2) * (gens == g).sum() * n_step
                     for g in gen_means)
        ss_scale_ = sum(((scale_means[s] - grand) ** 2) * (scales == s).sum() * n_step
                        for s in scale_means)

        lines.append("\n## Variance decomposition\n")
        lines.append("| source | SS | % of total |")
        lines.append("|---|---:|---:|")
        lines.append(f"| total | {ss_total:.1f} | 100.0 |")
        lines.append(f"| between-model (model id) | {ss_model:.1f} | {100*ss_model/ss_total:.1f} |")
        lines.append(f"|   \u2192 attributable to generation | {ss_gen:.1f} | {100*ss_gen/ss_total:.1f} |")
        lines.append(f"|   \u2192 attributable to scale | {ss_scale_:.1f} | {100*ss_scale_/ss_total:.1f} |")
        lines.append(f"| between-step (text intrinsic) | {ss_step:.1f} | {100*ss_step/ss_total:.1f} |")
        lines.append(f"| model \u00d7 step residual | {ss_resid:.1f} | {100*ss_resid/ss_total:.1f} |")

    # ── AUC trend by scale per generation ──
    lines.append("\n## AUC trend by scale within each generation\n")
    lines.append("| generation | scale | AUC |")
    lines.append("|---|---:|---:|")
    for name, gen, scale in MODELS:
        if name not in per_model_obs:
            continue
        vals, succ, _, _ = per_model_obs[name]
        v_s = vals[succ == 1]; v_f = vals[succ == 0]
        if len(v_s) == 0 or len(v_f) == 0:
            continue
        u, _ = mannwhitneyu(v_s, v_f, alternative="two-sided")
        auc = u / (len(v_s) * len(v_f))
        lines.append(f"| {gen} | {scale} | {auc:.3f} |")

    out_path = OUT_DIR / "report.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"[cross_gen_scale] wrote {out_path}")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
