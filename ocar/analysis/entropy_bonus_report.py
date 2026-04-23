"""Auto-report generator for entropy bonus ablation (Experiment A).

Pulls wandb histories for grpo_entropy_bonus_ablation project + compares
against dual-token `l49ikuco` and observe `lmlyvpa6` baselines.

Outputs: ocar/analysis_results/entropy_bonus/report.md
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "analysis_results" / "entropy_bonus"
OUT_DIR.mkdir(exist_ok=True, parents=True)

DUAL_CSV = ROOT / "analysis_results" / "wandb_dualtoken_l49ikuco_full.csv"
OBSERVE_CSV = ROOT / "analysis_results" / "webshop" / "alfworld_observe_history.csv"

ENTITY = os.environ.get("WANDB_ENTITY", "")
PROJECT = "grpo_entropy_bonus_ablation"


def pull_run(api, run):
    hist = run.history(samples=100000, pandas=True)
    hist["_run_name"] = run.name
    hist["_run_id"] = run.id
    return hist


def summarize(df, label, val_key="val/alfworld/AlfredTWEnv/success_rate",
              train_key="train/alfworld/AlfredTWEnv/success_rate",
              ent_key="actor/entropy", s_theta_key=None):
    """Return final-3-step mean + last-value for key metrics."""
    out = {"label": label}
    for key, short in [(val_key, "val_sr"), (train_key, "train_sr"),
                       (ent_key, "ent"), (s_theta_key, "s_theta")]:
        if key is None or key not in df.columns:
            out[short + "_end3"] = float("nan")
            out[short + "_last"] = float("nan")
            continue
        v = df[key].dropna().tail(5)
        out[short + "_last"] = float(v.iloc[-1]) if len(v) else float("nan")
        out[short + "_end3"] = float(v.tail(3).mean()) if len(v) else float("nan")
    return out


def main():
    lines = ["# GRPO + Entropy Bonus Ablation Report (Experiment A)\n"]

    entropy_summaries = []
    try:
        import wandb
        api = wandb.Api()
        runs = list(api.runs(f"{ENTITY + '/' if ENTITY else ''}{PROJECT}"))
        print(f"Found {len(runs)} runs in {PROJECT}")
        for r in runs:
            try:
                df = pull_run(api, r)
                df.to_csv(OUT_DIR / f"hist_{r.name}.csv", index=False)
                entropy_summaries.append(summarize(df, r.name))
            except Exception as e:
                print(f"  skip {r.name}: {e}")
    except Exception as e:
        lines.append(f"\n> wandb fetch failed: {e}\n")
        print(f"wandb fetch failed: {e}")

    # ── Entropy bonus runs table ──
    if entropy_summaries:
        lines.append("## Entropy bonus runs (final state)\n")
        lines.append("| run | val_sr_end3 | val_sr_last | train_sr_end3 | train_sr_last | ent_end3 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for s in sorted(entropy_summaries, key=lambda d: d["label"]):
            lines.append(f"| {s['label']} | {s['val_sr_end3']:.3f} | {s['val_sr_last']:.3f} | "
                         f"{s['train_sr_end3']:.3f} | {s['train_sr_last']:.3f} | {s['ent_end3']:.3f} |")

    # ── Baseline comparison ──
    lines.append("\n## Baseline comparison (dual-token vs observe, from prior logs)\n")
    baselines = []
    if DUAL_CSV.exists():
        df = pd.read_csv(DUAL_CSV)
        baselines.append(summarize(df, "dual-token (l49ikuco)"))
    if OBSERVE_CSV.exists():
        df = pd.read_csv(OBSERVE_CSV)
        baselines.append(summarize(df, "observe (lmlyvpa6)"))
    lines.append("| run | val_sr_end3 | val_sr_last | train_sr_end3 | ent_end3 |")
    lines.append("|---|---:|---:|---:|---:|")
    for s in baselines:
        lines.append(f"| {s['label']} | {s['val_sr_end3']:.3f} | {s['val_sr_last']:.3f} | "
                     f"{s['train_sr_end3']:.3f} | {s['ent_end3']:.3f} |")

    # ── Interpretation hook ──
    lines.append("\n## Interpretation hook\n")
    lines.append("If entropy_bonus runs reach val_sr_end3 >= dual-token (~0.81-0.88 end5 range),")
    lines.append("framing (b) 'entropy regularizer' is supported. If they stay near observe baseline")
    lines.append("(~0.79-0.81), framing (a) 'observation-grounded LM' is supported. See EXPERIMENT_LOG")
    lines.append("§4.2 for the three framings.")

    out = OUT_DIR / "report.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[entropy_bonus] wrote {out}")


if __name__ == "__main__":
    main()
