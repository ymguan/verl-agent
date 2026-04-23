"""Joint entropy × surprise analysis over three wandb runs.

Inputs (pre-exported CSVs; re-export via wandb API if missing):
- ocar/analysis_results/webshop/alfworld_observe_history.csv   (ALFW GRPO+observe, n=30)
- ocar/analysis_results/webshop/history_full.csv               (Webshop GRPO+observe, n=31)
- ocar/analysis_results/wandb_dualtoken_l49ikuco_full.csv      (ALFW dual-token, n=151)

Outputs:
- ocar/analysis_results/entropy_surprise/report.md
- ocar/analysis_results/entropy_surprise/overview.png
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "ocar" / "analysis_results"
OUT = RES / "entropy_surprise"
OUT.mkdir(parents=True, exist_ok=True)

AF_CSV = RES / "webshop" / "alfworld_observe_history.csv"
WS_CSV = RES / "webshop" / "history_full.csv"
DT_CSV = RES / "wandb_dualtoken_l49ikuco_full.csv"

ENT = "observe/obs_step_entropy_mean_mean"
STH = "observe/obs_s_theta_mean_mean"
DSM = "observe/obs_delta_s_mean_mean"
SR = "episode/success_rate"
SUCC_ENT = "observe/success_entropy_mean"
FAIL_ENT = "observe/failure_entropy_mean"
SUCC_STH = "observe/success_s_theta_mean"
FAIL_STH = "observe/failure_s_theta_mean"


def _rho(a: pd.Series, b: pd.Series):
    m = a.notna() & b.notna()
    if m.sum() < 5:
        return np.nan, np.nan, int(m.sum())
    r, p = spearmanr(a[m], b[m])
    return float(r), float(p), int(m.sum())


def _detrend(s: pd.Series, step: pd.Series) -> pd.Series:
    m = s.notna() & step.notna()
    if m.sum() < 5:
        return s * np.nan
    coef = np.polyfit(step[m], s[m], 1)
    return s - np.polyval(coef, step)


def _ccf_argmax(x: np.ndarray, y: np.ndarray, maxlag: int):
    x = (x - x.mean()) / x.std()
    y = (y - y.mean()) / y.std()
    lags = list(range(-maxlag, maxlag + 1))
    cc = []
    for k in lags:
        if k < 0:
            xs, ys = x[:k], y[-k:]
        elif k > 0:
            xs, ys = x[k:], y[:-k]
        else:
            xs, ys = x, y
        if len(xs) < 5:
            cc.append(np.nan)
            continue
        cc.append(float(np.corrcoef(xs, ys)[0, 1]))
    arr = np.array(cc)
    i = int(np.nanargmax(np.abs(arr)))
    return lags[i], float(arr[i])


def q1_training_spearman(runs: dict) -> pd.DataFrame:
    rows = []
    for name, df in runs.items():
        r_es = _rho(df[ENT], df[STH])
        r_ed = _rho(df[ENT], df[DSM])
        r_er = _rho(df[ENT], df[SR])
        r_sr = _rho(df[STH], df[SR])
        r_dr = _rho(df[DSM], df[SR])
        rows.append(
            {
                "run": name,
                "n": r_es[2],
                "rho(ent,s_th)": r_es[0],
                "p(ent,s_th)": r_es[1],
                "rho(ent,dS)": r_ed[0],
                "p(ent,dS)": r_ed[1],
                "rho(ent,SR)": r_er[0],
                "p(ent,SR)": r_er[1],
                "rho(s_th,SR)": r_sr[0],
                "p(s_th,SR)": r_sr[1],
                "rho(dS,SR)": r_dr[0],
                "p(dS,SR)": r_dr[1],
            }
        )
    return pd.DataFrame(rows)


def q5_detrended(runs: dict) -> pd.DataFrame:
    rows = []
    for name, df in runs.items():
        ent = _detrend(df[ENT], df["_step"])
        sth = _detrend(df[STH], df["_step"])
        dsm = _detrend(df[DSM], df["_step"])
        sr = _detrend(df[SR], df["_step"])
        r_er = _rho(ent, sr)
        r_sr = _rho(sth, sr)
        r_dr = _rho(dsm, sr)
        r_es = _rho(ent, sth)
        r_ed = _rho(ent, dsm)
        rows.append(
            {
                "run": name,
                "n": r_er[2],
                "rhodet(ent,SR)": r_er[0],
                "p(ent,SR)": r_er[1],
                "rhodet(s_th,SR)": r_sr[0],
                "rhodet(dS,SR)": r_dr[0],
                "rhodet(ent,s_th)": r_es[0],
                "rhodet(ent,dS)": r_ed[0],
            }
        )
    return pd.DataFrame(rows)


def q2_within_batch_gap(runs: dict) -> pd.DataFrame:
    rows = []
    for name, df in runs.items():
        if SUCC_ENT not in df.columns or FAIL_ENT not in df.columns:
            continue
        d = df[[SUCC_ENT, FAIL_ENT, SUCC_STH, FAIL_STH]].dropna()
        if len(d) < 5:
            continue
        gap_e = d[FAIL_ENT] - d[SUCC_ENT]
        gap_s = d[FAIL_STH] - d[SUCC_STH]
        t_e = ttest_1samp(gap_e, 0)
        t_s = ttest_1samp(gap_s, 0)
        rows.append(
            {
                "run": name,
                "n": len(d),
                "ent_gap_mean": gap_e.mean(),
                "ent_P(f>s)": (gap_e > 0).mean(),
                "ent_t": t_e.statistic,
                "ent_p": t_e.pvalue,
                "s_th_gap_mean": gap_s.mean(),
                "s_th_P(f>s)": (gap_s > 0).mean(),
                "s_th_t": t_s.statistic,
                "s_th_p": t_s.pvalue,
            }
        )
    return pd.DataFrame(rows)


def q6_gap_quartiles(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["_step", SUCC_ENT, FAIL_ENT, SUCC_STH, FAIL_STH, SR]].dropna().reset_index(drop=True)
    d["gap_ent"] = d[FAIL_ENT] - d[SUCC_ENT]
    d["gap_sth"] = d[FAIL_STH] - d[SUCC_STH]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(d)), 4)):
        q = d.iloc[idx]
        rows.append(
            {
                "quartile": f"Q{i+1}",
                "step_lo": int(q["_step"].min()),
                "step_hi": int(q["_step"].max()),
                "n": len(q),
                "ent_gap_mean": q["gap_ent"].mean(),
                "ent_gap_std": q["gap_ent"].std(),
                "s_th_gap_mean": q["gap_sth"].mean(),
                "s_th_gap_std": q["gap_sth"].std(),
                "SR_mean": q[SR].mean(),
            }
        )
    return pd.DataFrame(rows)


def q10_cumulative_gap(df: pd.DataFrame) -> pd.DataFrame:
    d = df[[SUCC_ENT, FAIL_ENT]].dropna().reset_index(drop=True)
    gap = (d[FAIL_ENT] - d[SUCC_ENT]).values
    rows = []
    for k in [5, 10, 15, 20, 30, 50, 75, 100, 150]:
        if k > len(gap):
            continue
        t = ttest_1samp(gap[:k], 0)
        rows.append({"first_k": k, "mean_gap": gap[:k].mean(), "t": t.statistic, "p": t.pvalue})
    return pd.DataFrame(rows)


def q3_aligned_trajectory(af: pd.DataFrame, dt: pd.DataFrame) -> pd.DataFrame:
    cols = [
        ("s_th", STH),
        ("dS", DSM),
        ("ent", ENT),
        ("SR", SR),
        ("wm_s", "observe/obs_wm_s_mean"),
        ("wm_s_B", "observe/obs_wm_s_B_mean"),
    ]
    keep = [c for _, c in cols if c in dt.columns]
    dt_sub = dt[["_step", *keep]].dropna(subset=[STH])
    m = pd.merge(af[["_step", *keep]], dt_sub, on="_step", how="inner", suffixes=("_obs", "_dt"))
    rows = []
    for name, col in cols:
        if col not in dt.columns:
            continue
        a = m[f"{col}_obs"]
        b = m[f"{col}_dt"]
        rows.append(
            {
                "metric": name,
                "obs_mean": a.mean(),
                "dt_mean": b.mean(),
                "obs_end3": a.iloc[-3:].mean(),
                "dt_end3": b.iloc[-3:].mean(),
                "delta_end3": b.iloc[-3:].mean() - a.iloc[-3:].mean(),
            }
        )
    return pd.DataFrame(rows)


def q8_entropy_by_phase(af: pd.DataFrame, dt: pd.DataFrame) -> pd.DataFrame:
    m = pd.merge(
        af[["_step", ENT, SR]].rename(columns={ENT: "ent_obs", SR: "sr_obs"}),
        dt[["_step", ENT, "actor/entropy_loss"]].rename(
            columns={ENT: "ent_dt", "actor/entropy_loss": "aent_dt"}
        ),
        on="_step",
        how="inner",
    )
    rows = []
    for lo, hi in [(5, 50), (55, 100), (105, 150)]:
        sub = m[(m["_step"] >= lo) & (m["_step"] <= hi)]
        rows.append(
            {
                "step_lo": lo,
                "step_hi": hi,
                "n": len(sub),
                "ent_obs": sub["ent_obs"].mean(),
                "ent_dt": sub["ent_dt"].mean(),
                "delta_dt_minus_obs": sub["ent_dt"].mean() - sub["ent_obs"].mean(),
                "actor_ent_loss_dt": sub["aent_dt"].mean(),
            }
        )
    return pd.DataFrame(rows)


def q_per_task(dt: pd.DataFrame) -> pd.DataFrame:
    tasks = ["clean", "cool", "heat", "examine", "pick_place", "other"]
    rows = []
    for t in tasks:
        ec = f"observe_task/{t}_entropy_mean"
        sc = f"observe_task/{t}_s_theta_mean"
        rc = f"observe_task/{t}_success_rate"
        if not all(c in dt.columns for c in (ec, sc, rc)):
            continue
        d = dt[[ec, sc, rc]].dropna()
        if len(d) < 5:
            continue
        r_es = _rho(d[ec], d[rc])
        r_ss = _rho(d[sc], d[rc])
        r_et = _rho(d[ec], d[sc])
        rows.append(
            {
                "task": t,
                "n": len(d),
                "ent_end5": d[ec].tail(5).mean(),
                "s_th_end5": d[sc].tail(5).mean(),
                "SR_end5": d[rc].tail(5).mean(),
                "rho(ent,SR)": r_es[0],
                "p(ent,SR)": r_es[1],
                "rho(s_th,SR)": r_ss[0],
                "rho(ent,s_th)": r_et[0],
            }
        )
    return pd.DataFrame(rows).sort_values("SR_end5", ascending=False)


def q_lead_lag(runs: dict) -> pd.DataFrame:
    rows = []
    for name, df in runs.items():
        d = df[[ENT, DSM, STH, SR]].dropna().reset_index(drop=True)
        if len(d) < 15:
            continue
        maxlag = min(15, len(d) // 3)
        lag1, cc1 = _ccf_argmax(d[ENT].values, d[SR].values, maxlag)
        lag2, cc2 = _ccf_argmax(d[DSM].values, d[SR].values, maxlag)
        lag3, cc3 = _ccf_argmax(d[ENT].values, d[DSM].values, maxlag)
        rows.append(
            {
                "run": name,
                "lag(ent->SR)": lag1,
                "cc(ent,SR)": cc1,
                "lag(dS->SR)": lag2,
                "cc(dS,SR)": cc2,
                "lag(ent->dS)": lag3,
                "cc(ent,dS)": cc3,
            }
        )
    return pd.DataFrame(rows)


def q_residual_hardness(dt: pd.DataFrame):
    d = dt[[SR, SUCC_ENT, FAIL_ENT]].dropna()
    gap = d[FAIL_ENT] - d[SUCC_ENT]
    r, p = spearmanr(d[SR], gap)
    return float(r), float(p), int(len(d)), float(gap.mean()), float(d[SR].mean())


def q_val_contrast(af: pd.DataFrame, dt: pd.DataFrame) -> dict:
    af_v = af[["_step", "val/success_rate"]].dropna()
    dt_v = dt[["_step", "val/success_rate"]].dropna()
    return {
        "af_val_end3": af_v["val/success_rate"].tail(3).mean(),
        "dt_val_end3": dt_v["val/success_rate"].tail(3).mean(),
        "af_val_last": af_v["val/success_rate"].iloc[-1],
        "dt_val_last": dt_v["val/success_rate"].iloc[-1],
        "af_train_end3": af[SR].dropna().tail(3).mean(),
        "dt_train_end3": dt[SR].dropna().tail(3).mean(),
    }


def plot_overview(af: pd.DataFrame, ws: pd.DataFrame, dt: pd.DataFrame, out_png: Path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 1: ALFW observe vs dual-token
    ax = axes[0, 0]
    ax.plot(af["_step"], af[ENT], label="GRPO+observe", color="tab:blue")
    ax.plot(dt["_step"], dt[ENT], label="dual-token", color="tab:orange", alpha=0.8)
    ax.set_title("ALFW step_entropy trajectory")
    ax.set_xlabel("step")
    ax.set_ylabel("step entropy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(af["_step"], af[STH], label="GRPO+observe", color="tab:blue")
    ax.plot(dt["_step"], dt[STH], label="dual-token", color="tab:orange", alpha=0.8)
    ax.set_title("ALFW s_theta trajectory")
    ax.set_xlabel("step")
    ax.set_ylabel("obs NLL (nats/tok)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    ax.plot(af["_step"], af[SR], label="train SR obs", color="tab:blue")
    ax.plot(dt["_step"], dt[SR], label="train SR dual", color="tab:orange", alpha=0.8)
    if "val/success_rate" in af.columns:
        vaf = af[["_step", "val/success_rate"]].dropna()
        ax.plot(vaf["_step"], vaf["val/success_rate"], "--", color="tab:blue", label="val SR obs")
    vdt = dt[["_step", "val/success_rate"]].dropna()
    ax.plot(vdt["_step"], vdt["val/success_rate"], "--", color="tab:orange", label="val SR dual")
    ax.set_title("ALFW SR (train vs val)")
    ax.set_xlabel("step")
    ax.set_ylabel("SR")
    ax.legend()
    ax.grid(alpha=0.3)

    # Row 2: succ/fail entropy gap (dual-token), per-task, webshop entropy
    ax = axes[1, 0]
    d = dt[["_step", SUCC_ENT, FAIL_ENT]].dropna()
    ax.plot(d["_step"], d[SUCC_ENT], label="success entropy", color="tab:green")
    ax.plot(d["_step"], d[FAIL_ENT], label="failure entropy", color="tab:red")
    ax.fill_between(d["_step"], d[SUCC_ENT], d[FAIL_ENT], color="tab:red", alpha=0.1)
    ax.set_title("Dual-token: fail vs succ entropy (gap widens)")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    tasks = ["heat", "examine", "pick_place", "cool", "clean", "other"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    for t, c in zip(tasks, colors):
        col = f"observe_task/{t}_entropy_mean"
        if col in dt.columns:
            dd = dt[["_step", col]].dropna()
            ax.plot(dd["_step"], dd[col], label=t, color=c, alpha=0.8)
    ax.set_title("Dual-token: per-task entropy")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    ax.plot(ws["_step"], ws[ENT], label="entropy", color="tab:purple")
    ax2 = ax.twinx()
    ax2.plot(ws["_step"], ws[SR], label="SR", color="tab:gray", alpha=0.5)
    ax.set_title("Webshop observe: entropy vs SR")
    ax.set_xlabel("step")
    ax.set_ylabel("entropy", color="tab:purple")
    ax2.set_ylabel("SR", color="tab:gray")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _fmt_df(df: pd.DataFrame, floats: int = 4) -> str:
    def _cell(v):
        if isinstance(v, float):
            if np.isnan(v):
                return "nan"
            return f"{v:.{floats}f}"
        return str(v)
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_cell(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    af = pd.read_csv(AF_CSV)
    ws = pd.read_csv(WS_CSV)
    dt = pd.read_csv(DT_CSV)
    runs = {"ALFW observe": af, "Webshop observe": ws, "ALFW dual": dt}

    q1 = q1_training_spearman(runs)
    q5 = q5_detrended(runs)
    q2 = q2_within_batch_gap(runs)
    q6_dt = q6_gap_quartiles(dt)
    q6_ws = q6_gap_quartiles(ws)
    q10 = q10_cumulative_gap(dt)
    q3 = q3_aligned_trajectory(af, dt)
    q8 = q8_entropy_by_phase(af, dt)
    qt = q_per_task(dt)
    ql = q_lead_lag(runs)
    rh = q_residual_hardness(dt)
    vc = q_val_contrast(af, dt)

    plot_overview(af, ws, dt, OUT / "overview.png")

    report_lines = [
        "# Entropy × Surprise Joint Analysis (3 wandb runs)",
        "",
        "Inputs:",
        "- ALFW observe: `ocar/analysis_results/webshop/alfworld_observe_history.csv`",
        "- Webshop observe: `ocar/analysis_results/webshop/history_full.csv`",
        "- ALFW dual-token: `ocar/analysis_results/wandb_dualtoken_l49ikuco_full.csv`",
        "",
        "## Q1. Training-level Spearman",
        _fmt_df(q1, 3),
        "",
        "## Q2. Within-batch succ/fail gaps (fail minus succ)",
        _fmt_df(q2, 4),
        "",
        "## Q3. Aligned trajectory: dual-token vs GRPO+observe on ALFW",
        _fmt_df(q3, 4),
        "",
        "## Q5. Detrended Spearman (linear _step trend removed)",
        _fmt_df(q5, 3),
        "",
        "## Q6. Gap quartiles — dual-token (ALFW)",
        _fmt_df(q6_dt, 4),
        "",
        "## Q6. Gap quartiles — webshop observe",
        _fmt_df(q6_ws, 4),
        "",
        "## Q8. Entropy phase means: dual-token vs observe (ALFW)",
        _fmt_df(q8, 4),
        "",
        "## Q10. Cumulative t-stat: entropy succ/fail gap (dual-token)",
        _fmt_df(q10, 4),
        "",
        "## Per-task end-of-training (dual-token, last 5 logged steps)",
        _fmt_df(qt, 4),
        "",
        "## Lead-lag (argmax |CCF|)",
        _fmt_df(ql, 3),
        "",
        "## Residual hardness: SR vs ent_gap (dual-token)",
        f"rho(SR, fail_ent - succ_ent) = {rh[0]:+.3f}  p = {rh[1]:.2g}  n = {rh[2]}",
        f"mean gap = {rh[3]:+.4f}, mean SR = {rh[4]:.3f}",
        "",
        "## Val vs train SR contrast (ALFW)",
        f"observe train_end3 = {vc['af_train_end3']:.4f}, val_end3 = {vc['af_val_end3']:.4f}, val_last = {vc['af_val_last']:.4f}",
        f"dual    train_end3 = {vc['dt_train_end3']:.4f}, val_end3 = {vc['dt_val_end3']:.4f}, val_last = {vc['dt_val_last']:.4f}",
        "",
        "Figure: `overview.png`",
    ]
    (OUT / "report.md").write_text("\n".join(report_lines))
    print(f"[done] wrote {OUT/'report.md'} and {OUT/'overview.png'}")


if __name__ == "__main__":
    main()
