"""F1 — Within-obs_type bucketed correlations of step-level signals with success.

For each (dataset, ckpt_step):
  - classify each step by obs_type
  - compute trajectory-level mean signal (overall and per-bucket, only counting that bucket's steps)
  - report Pearson r(traj_mean_signal, success) overall and per-bucket
  - also: step-level AUC (signal vs traj-level success label) within each bucket

Signals: delta_s_mean, entropy_mean, s_theta_mean, wm_s, wm_gap (= wm_s_B - wm_s)
Hidden-state info_gain joined via (success,n_steps,step0 delta_s) when possible.
"""
import json, os, glob, re
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr

# --------------- obs-type classifier ---------------
def classify_alfworld(obs: str, prev: list) -> str:
    s = obs.lower()
    if any(p in s for p in ["nothing happens", "nothing to", "can't", "don't see"]):
        return "nothing_happens"
    if any(p in s for p in ["you pick up", "you put", "you open", "you close",
                              "you turn", "you heat", "you cool", "you clean",
                              "you slice", "you use", "you move"]):
        return "state_change"
    if obs in prev:
        return "revisit"
    return "new_location"


def classify_webshop(obs: str, prev: list) -> str:
    if not obs or obs.strip() == "'Search'":
        return "search_page"
    s = obs.lower()
    if "nothing happens" in s or "invalid" in s:
        return "nothing_happens"
    # product detail page has 'description' or 'features' section markers
    if "[sep] 'description'" in s or "[sep] 'features'" in s or "[sep] 'reviews'" in s or "[sep] 'attributes'" in s:
        return "product_detail"
    # options page: lists sizes/colors after < Prev in URL but with many option lines
    if "[sep] 'buy now'" in s:
        return "options_page"
    # results listing: many ASIN-like codes
    if re.search(r"'b0[0-9a-z]{8}'", obs.lower()):
        return "search_results"
    if obs in prev:
        return "revisit"
    return "other"


# --------------- signal extraction ---------------
SIGNALS = ["delta_s_mean", "entropy_mean", "s_theta_mean", "wm_s", "wm_gap"]


def build_step_records(data: dict, env: str):
    """Return list of dicts per step with signals + obs_type + traj success + traj_idx."""
    classify = classify_alfworld if env == "alfworld" else classify_webshop
    recs = []
    for ti, traj in enumerate(data["trajectories"]):
        prev = []
        for step in traj["steps"]:
            obs = step.get("observation", "")
            otype = classify(obs, prev)
            prev.append(obs)
            wm_s = step.get("wm_s")
            wm_s_B = step.get("wm_s_B")
            wm_gap = (wm_s_B - wm_s) if (wm_s is not None and wm_s_B is not None) else None
            rec = {
                "traj_idx": ti,
                "success": int(bool(traj["success"])),
                "obs_type": otype,
                "delta_s_mean": step.get("delta_s_mean"),
                "entropy_mean": step.get("entropy_mean"),
                "s_theta_mean": step.get("s_theta_mean"),
                "wm_s": wm_s,
                "wm_gap": wm_gap,
            }
            recs.append(rec)
    return recs


def traj_bucket_means(recs: list, signals=SIGNALS):
    """Return dict: (traj_idx, obs_type) -> {signal: mean}, plus overall per-traj means."""
    by_traj_type = {}
    by_traj = {}
    # overall
    traj_idx_set = set(r["traj_idx"] for r in recs)
    for ti in traj_idx_set:
        traj_recs = [r for r in recs if r["traj_idx"] == ti]
        succ = traj_recs[0]["success"]
        by_traj[ti] = {"success": succ, "n": len(traj_recs)}
        for sig in signals:
            vals = [r[sig] for r in traj_recs if r[sig] is not None]
            by_traj[ti][sig] = float(np.mean(vals)) if vals else None
        types = set(r["obs_type"] for r in traj_recs)
        for t in types:
            sub = [r for r in traj_recs if r["obs_type"] == t]
            d = {"success": succ, "n": len(sub)}
            for sig in signals:
                vals = [r[sig] for r in sub if r[sig] is not None]
                d[sig] = float(np.mean(vals)) if vals else None
            by_traj_type[(ti, t)] = d
    return by_traj, by_traj_type


def corr(xs, ys):
    xs, ys = list(xs), list(ys)
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and not np.isnan(x)]
    if len(pairs) < 3:
        return None, None, len(pairs)
    x, y = zip(*pairs)
    if np.std(x) == 0 or np.std(y) == 0:
        return None, None, len(pairs)
    r, p = pearsonr(x, y)
    return r, p, len(pairs)


def analyze_file(path: str, env: str):
    data = json.load(open(path))
    step = data["global_step"]
    recs = build_step_records(data, env)
    if not recs:
        return None

    by_traj, by_traj_type = traj_bucket_means(recs)
    n_traj = len(by_traj)
    sr = np.mean([v["success"] for v in by_traj.values()])

    out = {"env": env, "step": step, "n_traj": n_traj, "SR": sr,
           "path": path, "overall": {}, "buckets": {}}

    # Overall traj-level r
    success = [v["success"] for v in by_traj.values()]
    for sig in SIGNALS:
        xs = [by_traj[ti].get(sig) for ti in by_traj]
        r, p, n = corr(xs, success)
        out["overall"][sig] = {"r": r, "p": p, "n": n}

    # Per-bucket
    types = sorted(set(r["obs_type"] for r in recs))
    for t in types:
        entries = [v for (ti, tt), v in by_traj_type.items() if tt == t]
        # only keep trajectories with at least 1 step of that type
        trajs_with = [(ti, v) for (ti, tt), v in by_traj_type.items() if tt == t]
        if len(trajs_with) < 6:
            continue
        ys = [v["success"] for (ti, v) in trajs_with]
        bucket_info = {"n_traj_with_type": len(trajs_with),
                        "frac_succ": float(np.mean(ys)),
                        "mean_steps_per_traj": float(np.mean([v["n"] for (ti, v) in trajs_with]))}
        for sig in SIGNALS:
            xs = [v.get(sig) for (ti, v) in trajs_with]
            r, p, n = corr(xs, ys)
            bucket_info[sig] = {"r": r, "p": p, "n": n}
            # also means succ vs fail
            succ_vals = [v.get(sig) for (ti, v) in trajs_with if v["success"] and v.get(sig) is not None]
            fail_vals = [v.get(sig) for (ti, v) in trajs_with if not v["success"] and v.get(sig) is not None]
            bucket_info[sig]["succ_mean"] = float(np.mean(succ_vals)) if succ_vals else None
            bucket_info[sig]["fail_mean"] = float(np.mean(fail_vals)) if fail_vals else None
        out["buckets"][t] = bucket_info
    return out


# --------------- hidden-state joining ---------------
def join_hidden(hidden_path: str, observe_path: str):
    """Return augmented list: for each hidden traj, try to match to observe traj by (success,n_steps) and by step-0 delta_s when ambiguous.
    Augment hidden step records with obs text from observe."""
    hd = json.load(open(hidden_path))
    od = json.load(open(observe_path))
    obs_trajs = od["trajectories"]

    # index obs trajs by (success, n_steps)
    from collections import defaultdict
    key2obs = defaultdict(list)
    for t in obs_trajs:
        key2obs[(bool(t["success"]), t["n_steps"])].append(t)

    matched = []
    for ht in hd:
        key = (bool(ht["success"]), ht["n_steps"])
        cands = key2obs.get(key, [])
        # further disambiguate by step-0 delta_s if available
        if not cands:
            matched.append(None)
            continue
        best = None; best_diff = 1e9
        h_step0 = ht["steps"][0].get("delta_s_mean")
        for c in cands:
            c_step0 = c["steps"][0].get("delta_s_mean")
            if h_step0 is not None and c_step0 is not None:
                diff = abs(h_step0 - c_step0)
            else:
                diff = 0
            if diff < best_diff:
                best_diff = diff; best = c
        matched.append(best)
    return hd, matched


def analyze_hidden(hidden_path: str, observe_path: str, env: str):
    if not os.path.exists(hidden_path) or not os.path.exists(observe_path):
        return None
    hd, matched = join_hidden(hidden_path, observe_path)
    classify = classify_alfworld if env == "alfworld" else classify_webshop

    # Build traj-level IG means overall & per bucket
    per_traj = []  # list of dicts
    per_traj_bucket = {}  # (ti,bucket)->list of ig values
    for ti, (ht, ot) in enumerate(zip(hd, matched)):
        succ = int(bool(ht["success"]))
        obs_list = []
        if ot is not None:
            obs_list = [s.get("observation","") for s in ot["steps"]]
        # IG layer -1
        igs_by_type = {}
        prev = []
        for si, hs in enumerate(ht["steps"]):
            ig = hs.get("info_gain", {}).get("-1", 0.0)
            if ig == 0.0:
                prev.append(obs_list[si] if si < len(obs_list) else "")
                continue  # skip truncation artefacts
            obs = obs_list[si] if si < len(obs_list) else ""
            ot_type = classify(obs, prev)
            prev.append(obs)
            igs_by_type.setdefault(ot_type, []).append(ig)
        entry = {"traj_idx": ti, "success": succ,
                 "ig_overall": float(np.mean([ig for igs in igs_by_type.values() for ig in igs])) if igs_by_type else None}
        for t, vals in igs_by_type.items():
            entry[f"ig_{t}"] = float(np.mean(vals))
        per_traj.append(entry)

    # Overall r
    ys = [t["success"] for t in per_traj]
    xs = [t["ig_overall"] for t in per_traj]
    r_overall, p_overall, n_overall = corr(xs, ys)

    # Per-bucket r
    buckets = {}
    types = set()
    for t in per_traj:
        for k in t:
            if k.startswith("ig_") and k != "ig_overall":
                types.add(k[3:])
    for tt in types:
        xs, ys = [], []
        for t in per_traj:
            v = t.get(f"ig_{tt}")
            if v is not None:
                xs.append(v); ys.append(t["success"])
        if len(xs) < 6:
            continue
        r, p, n = corr(xs, ys)
        succ_vals = [x for x, y in zip(xs, ys) if y]
        fail_vals = [x for x, y in zip(xs, ys) if not y]
        buckets[tt] = {"n": n, "r": r, "p": p,
                       "succ_mean": float(np.mean(succ_vals)) if succ_vals else None,
                       "fail_mean": float(np.mean(fail_vals)) if fail_vals else None,
                       "n_succ": len(succ_vals), "n_fail": len(fail_vals)}
    return {"env": env, "hidden_path": hidden_path, "n_traj": len(per_traj),
            "overall_r_ig": r_overall, "overall_p_ig": p_overall, "overall_n": n_overall,
            "buckets": buckets}


# --------------- main ---------------
def format_r(x):
    if x is None:
        return "  —  "
    return f"{x:+.3f}"


def print_report(results):
    print("\n" + "=" * 120)
    print("=== FILE-LEVEL SIGNAL CORRELATIONS (traj-level mean vs success) ===")
    print("=" * 120)
    for res in results:
        print(f"\n[{res['env']} step={res['step']} n_traj={res['n_traj']} SR={res['SR']:.2f}]")
        print(f"  OVERALL:")
        hdr = f"    {'signal':12s}  {'r':>7s}  {'p':>7s}  {'n':>4s}"
        print(hdr)
        for sig in SIGNALS:
            v = res["overall"][sig]
            r = format_r(v['r']); p = f"{v['p']:.3f}" if v['p'] is not None else "  —  "
            print(f"    {sig:12s}  {r:>7s}  {p:>7s}  {v['n']:>4d}")
        print(f"  PER obs_type (bucket):")
        print(f"    {'bucket':18s} {'n':>3s} {'SR':>5s} " + " ".join(f"{s[:11]:>11s}" for s in SIGNALS))
        for t, info in sorted(res["buckets"].items()):
            row = [f"{t:18s}", f"{info['n_traj_with_type']:>3d}", f"{info['frac_succ']:>5.2f}"]
            for sig in SIGNALS:
                v = info[sig]
                row.append(f"{format_r(v['r']):>11s}")
            print("    " + " ".join(row))
        print(f"  (succ_mean / fail_mean per bucket, delta_s):")
        for t, info in sorted(res["buckets"].items()):
            v = info["delta_s_mean"]
            if v["succ_mean"] is not None and v["fail_mean"] is not None:
                gap = v["succ_mean"] - v["fail_mean"]
                print(f"    {t:18s} Δs(S)={v['succ_mean']:+.4f}  Δs(F)={v['fail_mean']:+.4f}  gap={gap:+.4f}")


def print_hidden(hres_list):
    print("\n" + "=" * 120)
    print("=== HIDDEN-STATE INFO GAIN: per obs_type ===")
    print("=" * 120)
    for r in hres_list:
        if r is None:
            continue
        print(f"\n[{r['env']}] n_traj={r['n_traj']}  overall r(IG, succ)={format_r(r['overall_r_ig'])} "
              f"p={r['overall_p_ig']:.3f} n={r['overall_n']}" if r['overall_p_ig'] is not None else f"\n[{r['env']}] n_traj={r['n_traj']}  overall r(IG, succ)={format_r(r['overall_r_ig'])}")
        print(f"    {'bucket':18s} {'n':>3s} {'r(IG,succ)':>11s} {'p':>7s} {'IG(S)':>8s} {'IG(F)':>8s}")
        for t, info in sorted(r["buckets"].items()):
            pv = f"{info['p']:.3f}" if info['p'] is not None else "  —  "
            sm = f"{info['succ_mean']:.1f}" if info['succ_mean'] is not None else "  —  "
            fm = f"{info['fail_mean']:.1f}" if info['fail_mean'] is not None else "  —  "
            print(f"    {t:18s} {info['n']:>3d} {format_r(info['r']):>11s} {pv:>7s} {sm:>8s} {fm:>8s}")


if __name__ == "__main__":
    # Define file sets
    alfw_1_5b = [
        ("alfworld", f"checkpoints/grpo_observe_alfworld_1.5b_20260420_162642/grpo_observe_qwen2.5_1.5b_seed0/global_step_{s}/observe_trajectories.json")
        for s in [20, 40, 60, 80, 100, 120, 140, 150]
    ]
    alfw_7b = [("alfworld", "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json")]
    webshop = [
        ("webshop", f"data/trajectories/grpo_observe_webshop_20260418_070828/global_step_{s}_observe_trajectories.json")
        for s in [80, 160, 240, 320, 400, 480, 560, 640]
    ]

    results = []
    for env, path in alfw_1_5b + alfw_7b + webshop:
        if not os.path.exists(path):
            continue
        try:
            r = analyze_file(path, env)
            if r:
                results.append(r)
        except Exception as e:
            print(f"ERROR {path}: {e}")

    print_report(results)

    # Save JSON
    out_dir = Path("ocar/analysis_results/bucketed")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "signal_corr_by_bucket.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Hidden state
    hidden_pairs = [
        ("alfworld",
         "ocar/analysis_results/hidden_state/alfworld_step150.json",
         "checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0/global_step_150/observe_trajectories.json"),
        ("webshop",
         "ocar/analysis_results/hidden_state/webshop_step240.json",
         "data/trajectories/grpo_observe_webshop_20260418_070828/global_step_240_observe_trajectories.json"),
        ("webshop",
         "ocar/analysis_results/hidden_state/webshop_step640.json",
         "data/trajectories/grpo_observe_webshop_20260418_070828/global_step_640_observe_trajectories.json"),
    ]
    hres = []
    for env, hp, op in hidden_pairs:
        try:
            r = analyze_hidden(hp, op, env)
            if r:
                hres.append(r)
        except Exception as e:
            print(f"ERROR hidden {hp}: {e}")
    print_hidden(hres)
    with open(out_dir / "hidden_ig_by_bucket.json", "w") as f:
        json.dump(hres, f, indent=2, default=str)
    print(f"\nSaved to {out_dir}/")
