"""
STRIDE Detector Validation (E1)
===============================
Evaluates stuck-detection and useful-exploration detection across
multiple detector baselines using step-level rule-based labels.

Usage:
    python ocar/stride_detector_validation.py [--ckpt_dir PATH] [--out_dir PATH]

Outputs:
    - Console: F1/Precision/Recall/FPR tables
    - {out_dir}/detector_comparison.png
    - {out_dir}/signal_orthogonality.png
    - {out_dir}/temporal_structure.png
    - {out_dir}/case_studies.md
"""
import json, os, argparse
import numpy as np
from collections import defaultdict
from pathlib import Path

# ─────────────────── Data Loading ───────────────────

def load_checkpoint(ckpt_dir, step):
    path = os.path.join(ckpt_dir, f"global_step_{step}", "observe_trajectories.json")
    with open(path) as f:
        return json.load(f)

def parse_act(raw):
    if '<action>' in raw:
        return raw.split('<action>')[-1].split('</action>')[0].strip()
    return raw.strip().replace('\n', ' ')[:100]

def build_dataset(ckpt_dir, steps=None):
    if steps is None:
        steps = sorted([int(d.split('_')[-1])
                        for d in os.listdir(ckpt_dir)
                        if d.startswith('global_step_') and os.path.isdir(os.path.join(ckpt_dir, d))])
    rows = []
    for step in steps:
        d = load_checkpoint(ckpt_dir, step)
        for t in d['trajectories']:
            n = len(t['steps'])
            obs_list = [s['observation'] for s in t['steps']]
            act_list = [parse_act(s['action']) for s in t['steps']]
            s_theta = np.array([s['s_theta_mean'] for s in t['steps']])
            delta_s = np.array([s['delta_s_mean'] for s in t['steps']])
            ent_mean = np.array([s['entropy_mean'] for s in t['steps']])

            # EMA of delta_s
            ema_d = np.zeros(n)
            ema_d[0] = delta_s[0]
            lam = 0.5
            for i in range(1, n):
                ema_d[i] = lam * ema_d[i-1] + (1-lam) * delta_s[i]

            # window-3 mean of delta_s
            win3_d = np.array([delta_s[max(0,i-2):i+1].mean() for i in range(n)])

            for i in range(n):
                # ── Step-level ground truth labels ──
                # Stuck: future 3 obs have ≥2 "Nothing happens" OR future 4 acts have ≥3 repeats
                stuck = False
                future_obs = obs_list[i+1:i+4]
                if len(future_obs) >= 2 and sum('nothing happens' in o.lower() for o in future_obs) >= 2:
                    stuck = True
                future_acts = act_list[i:i+4]
                if len(future_acts) >= 3:
                    from collections import Counter
                    mc = Counter(future_acts).most_common(1)[0][1]
                    if mc >= 3:
                        stuck = True

                # Useful exploration: future 3 obs contain ≥3 novel words (>3 chars)
                seen_words = set()
                for j in range(i+1):
                    for w in obs_list[j].lower().split()[:50]:
                        seen_words.add(w)
                future_words = ' '.join(obs_list[i+1:i+4]).lower().split()
                novel = sum(1 for w in future_words if w not in seen_words and len(w) > 3)
                useful = novel >= 3

                rows.append({
                    'ckpt': step, 'tid': t['traj_id'], 'step_idx': i,
                    'n_steps': n, 'traj_success': int(t['success']),
                    'stuck': stuck, 'useful': useful,
                    's_theta': s_theta[i], 'delta_s': delta_s[i],
                    'ent_mean': ent_mean[i],
                    'ema_delta_s': ema_d[i], 'win3_delta_s': win3_d[i],
                    'obs': obs_list[i][:120], 'act': act_list[i][:80],
                })
    return rows

# ─────────────────── Detectors ───────────────────

def detector_random(rows, seed=42):
    rng = np.random.RandomState(seed)
    rate = np.mean([r['stuck'] for r in rows])
    return rng.random(len(rows)) < rate

def detector_action_repetition(rows, k=3, min_repeat=2):
    traj_rows = defaultdict(list)
    for i, r in enumerate(rows):
        traj_rows[(r['ckpt'], r['tid'])].append((i, r))

    preds = np.zeros(len(rows), dtype=bool)
    for key, trows in traj_rows.items():
        trows_sorted = sorted(trows, key=lambda x: x[1]['step_idx'])
        for j, (gi, r) in enumerate(trows_sorted):
            recent = [trows_sorted[m][1]['act'] for m in range(max(0, j-k+1), j+1)]
            if len(recent) >= min_repeat:
                from collections import Counter
                mc = Counter(recent).most_common(1)[0][1]
                if mc >= min_repeat:
                    preds[gi] = True
    return preds

def detector_single_signal(rows, signal_key, percentile=85):
    vals = np.array([r[signal_key] for r in rows])
    tau = np.percentile(vals, percentile)
    return vals > tau

def detector_stride_2signal(rows, k_S=3, pct_S=85, pct_H=85):
    """Stuck = consecutive S_θ ≥ k_S AND H_a > τ_H"""
    s_theta = np.array([r['s_theta'] for r in rows])
    ent = np.array([r['ent_mean'] for r in rows])
    tau_S = np.percentile(s_theta, pct_S)
    tau_H = np.percentile(ent, pct_H)

    traj_rows = defaultdict(list)
    for i, r in enumerate(rows):
        traj_rows[(r['ckpt'], r['tid'])].append((i, r))

    preds = np.zeros(len(rows), dtype=bool)
    for key, trows in traj_rows.items():
        trows_sorted = sorted(trows, key=lambda x: x[1]['step_idx'])
        run = 0
        for gi, r in trows_sorted:
            if r['s_theta'] > tau_S:
                run += 1
            else:
                run = 0
            if run >= k_S and r['ent_mean'] > tau_H:
                preds[gi] = True
    return preds

def detector_stride_3signal(rows, k_S=3, pct_S=85, pct_H=85, pct_D=85):
    """Stuck = consecutive S_θ ≥ k_S AND H_a > τ_H AND NOT exploring(EMA_ΔS > τ_Δ)"""
    s_theta = np.array([r['s_theta'] for r in rows])
    ent = np.array([r['ent_mean'] for r in rows])
    ema_d = np.array([r['ema_delta_s'] for r in rows])
    tau_S = np.percentile(s_theta, pct_S)
    tau_H = np.percentile(ent, pct_H)
    tau_D = np.percentile(ema_d, pct_D)

    traj_rows = defaultdict(list)
    for i, r in enumerate(rows):
        traj_rows[(r['ckpt'], r['tid'])].append((i, r))

    preds = np.zeros(len(rows), dtype=bool)
    for key, trows in traj_rows.items():
        trows_sorted = sorted(trows, key=lambda x: x[1]['step_idx'])
        run = 0
        for gi, r in trows_sorted:
            if r['s_theta'] > tau_S:
                run += 1
            else:
                run = 0
            stuck = run >= k_S and r['ent_mean'] > tau_H
            exploring = r['ema_delta_s'] > tau_D
            if stuck and not exploring:
                preds[gi] = True
    return preds

# ─────────────────── Metrics ───────────────────

def compute_metrics(y_true, y_pred):
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {'prec': prec, 'rec': rec, 'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}

def compute_useful_fpr(y_useful, y_pred):
    """Among steps labeled 'useful exploration', how many are falsely flagged as stuck?"""
    n_useful = y_useful.sum()
    if n_useful == 0:
        return 0.0
    return ((y_useful == 1) & (y_pred == 1)).sum() / n_useful

# ─────────────────── Figures ───────────────────

def plot_detector_comparison(results, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        names = list(results.keys())
        f1s = [results[n]['f1'] for n in names]
        fprs = [results[n]['useful_fpr'] for n in names]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.barh(names, f1s, color=['#ccc' if 'STRIDE' not in n else '#e74c3c' for n in names])
        ax1.set_xlabel('Stuck Detection F1')
        ax1.set_title('(a) Stuck Detection F1')
        ax1.set_xlim(0, max(f1s) * 1.3 + 0.05)
        for i, v in enumerate(f1s):
            ax1.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

        ax2.barh(names, fprs, color=['#ccc' if 'STRIDE' not in n else '#2ecc71' for n in names])
        ax2.set_xlabel('Useful-Exploration False Positive Rate')
        ax2.set_title('(b) Exploration Disruption Rate (lower is better)')
        ax2.set_xlim(0, max(fprs) * 1.3 + 0.05)
        for i, v in enumerate(fprs):
            ax2.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot")

def plot_signal_orthogonality(rows, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        s_theta = np.array([r['s_theta'] for r in rows])
        delta_s = np.array([r['delta_s'] for r in rows])
        ent = np.array([r['ent_mean'] for r in rows])

        signals = {'$S_\\theta$': s_theta, '$\\Delta S$': delta_s, '$H_a$': ent}
        names = list(signals.keys())
        n = len(names)

        fig, axes = plt.subplots(n, n, figsize=(10, 10))
        for i in range(n):
            for j in range(n):
                ax = axes[i][j]
                if i == j:
                    ax.hist(signals[names[i]], bins=50, color='steelblue', alpha=0.7)
                    ax.set_ylabel('Count')
                else:
                    subsample = np.random.RandomState(42).choice(len(s_theta), min(2000, len(s_theta)), replace=False)
                    stuck_mask = np.array([rows[k]['stuck'] for k in subsample])
                    ax.scatter(signals[names[j]][subsample[~stuck_mask]],
                               signals[names[i]][subsample[~stuck_mask]],
                               alpha=0.15, s=3, c='steelblue', label='normal')
                    ax.scatter(signals[names[j]][subsample[stuck_mask]],
                               signals[names[i]][subsample[stuck_mask]],
                               alpha=0.4, s=8, c='red', label='stuck')
                    r = np.corrcoef(signals[names[j]], signals[names[i]])[0, 1]
                    ax.set_title(f'r={r:.3f}', fontsize=9)
                if i == n-1:
                    ax.set_xlabel(names[j])
                if j == 0:
                    ax.set_ylabel(names[i])

        plt.suptitle('Signal Orthogonality Matrix', fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot")

def plot_temporal_structure(rows, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        traj_rows = defaultdict(list)
        for r in rows:
            traj_rows[(r['ckpt'], r['tid'])].append(r)

        # Find one success and one fail trajectory with interesting patterns
        best_succ, best_fail = None, None
        for key, trows in traj_rows.items():
            trows = sorted(trows, key=lambda r: r['step_idx'])
            if len(trows) < 15:
                continue
            n_stuck = sum(r['stuck'] for r in trows)
            if trows[0]['traj_success'] and n_stuck >= 2 and best_succ is None:
                best_succ = trows
            if not trows[0]['traj_success'] and n_stuck >= 5 and best_fail is None:
                best_fail = trows
            if best_succ and best_fail:
                break

        if not best_succ or not best_fail:
            print("  [WARN] couldn't find suitable trajectories for temporal plot")
            return

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=False)
        for ax, trows, label in [(axes[0], best_succ, 'Success'), (axes[1], best_fail, 'Failure')]:
            steps = [r['step_idx'] for r in trows]
            s_t = [r['s_theta'] for r in trows]
            d_s = [r['delta_s'] for r in trows]
            h_a = [r['ent_mean'] for r in trows]
            stuck = [r['stuck'] for r in trows]

            ax.plot(steps, s_t, 'b-o', markersize=3, label='$S_\\theta$ (obs surprise)', alpha=0.8)
            ax.plot(steps, d_s, 'g-s', markersize=3, label='$\\Delta S$ (policy divergence)', alpha=0.8)
            ax.plot(steps, h_a, 'r-^', markersize=3, label='$H_a$ (action entropy)', alpha=0.8)

            for i, s in enumerate(stuck):
                if s:
                    ax.axvspan(steps[i]-0.4, steps[i]+0.4, alpha=0.2, color='red')

            ax.set_title(f'{label} Trajectory (tid={trows[0]["tid"][:8]}, n_steps={len(trows)})')
            ax.set_ylabel('Signal Value')
            ax.legend(loc='upper right', fontsize=8)
            ax.set_xlabel('Step')

        plt.suptitle('Temporal Structure of Three Uncertainty Signals', fontsize=14)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")
    except ImportError:
        print("  [WARN] matplotlib not available, skipping plot")

# ─────────────────── Case Studies ───────────────────

def write_case_studies(rows, out_path):
    traj_rows = defaultdict(list)
    for r in rows:
        traj_rows[(r['ckpt'], r['tid'])].append(r)

    lines = ["# STRIDE Detector Case Studies\n"]

    # Find a STRIDE-detected stuck case and a correctly-guarded exploration case
    s_theta_all = np.array([r['s_theta'] for r in rows])
    ent_all = np.array([r['ent_mean'] for r in rows])
    ema_d_all = np.array([r['ema_delta_s'] for r in rows])
    tau_S = np.percentile(s_theta_all, 85)
    tau_H = np.percentile(ent_all, 85)
    tau_D = np.percentile(ema_d_all, 85)

    lines.append("## Case 1: Correctly Detected Stuck State\n")
    found = 0
    for key, trows in traj_rows.items():
        trows = sorted(trows, key=lambda r: r['step_idx'])
        run = 0
        for r in trows:
            if r['s_theta'] > tau_S:
                run += 1
            else:
                run = 0
            if run >= 3 and r['ent_mean'] > tau_H and r['stuck'] and found < 2:
                idx = r['step_idx']
                lines.append(f"### tid={r['tid'][:8]}, ckpt={r['ckpt']}, step={idx}\n")
                window = [rr for rr in trows if abs(rr['step_idx'] - idx) <= 2]
                for rr in window:
                    mark = "**★ TRIGGER**" if rr['step_idx'] == idx else ""
                    lines.append(f"- step {rr['step_idx']}: S_θ={rr['s_theta']:.2f} ΔS={rr['delta_s']:+.3f} H_a={rr['ent_mean']:.2f} stuck={rr['stuck']} {mark}")
                    lines.append(f"  - obs: {rr['obs']}")
                    lines.append(f"  - act: {rr['act']}\n")
                found += 1
        if found >= 2:
            break

    lines.append("\n## Case 2: Correctly Guarded Exploration (NOT triggered)\n")
    found = 0
    for key, trows in traj_rows.items():
        trows = sorted(trows, key=lambda r: r['step_idx'])
        run = 0
        for r in trows:
            if r['s_theta'] > tau_S:
                run += 1
            else:
                run = 0
            if run >= 3 and r['ema_delta_s'] > tau_D and r['useful'] and not r['stuck'] and found < 2:
                idx = r['step_idx']
                lines.append(f"### tid={r['tid'][:8]}, ckpt={r['ckpt']}, step={idx}\n")
                window = [rr for rr in trows if abs(rr['step_idx'] - idx) <= 2]
                for rr in window:
                    mark = "**★ GUARDED** (exploring)" if rr['step_idx'] == idx else ""
                    lines.append(f"- step {rr['step_idx']}: S_θ={rr['s_theta']:.2f} ΔS={rr['delta_s']:+.3f} EMA_ΔS={rr['ema_delta_s']:+.3f} useful={rr['useful']} {mark}")
                    lines.append(f"  - obs: {rr['obs']}")
                    lines.append(f"  - act: {rr['act']}\n")
                found += 1
        if found >= 2:
            break

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"  Saved: {out_path}")

# ─────────────────── Main ───────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str,
                        default='checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0')
    parser.add_argument('--out_dir', type=str, default='ocar/analysis_results/stride_e1')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    rows = build_dataset(args.ckpt_dir)
    N = len(rows)
    y_stuck = np.array([int(r['stuck']) for r in rows])
    y_useful = np.array([int(r['useful']) for r in rows])
    print(f"  N={N}, stuck={y_stuck.sum()} ({100*y_stuck.mean():.1f}%), useful={y_useful.sum()} ({100*y_useful.mean():.1f}%)")

    # ── Run all detectors ──
    print("\nRunning detectors...")
    detectors = {
        'Random': detector_random(rows),
        'Action-Repetition': detector_action_repetition(rows),
        'Single H_a (p85)': detector_single_signal(rows, 'ent_mean', 85),
        'Single S_θ (p85)': detector_single_signal(rows, 's_theta', 85),
        'Single ΔS (p85)': detector_single_signal(rows, 'delta_s', 85),
        'STRIDE 2-signal': detector_stride_2signal(rows),
        'STRIDE 3-signal': detector_stride_3signal(rows),
    }

    results = {}
    print(f"\n{'Detector':30s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'Useful FPR':>12s} {'N_pred':>8s}")
    print("-" * 80)
    for name, preds in detectors.items():
        preds = np.asarray(preds)
        m = compute_metrics(y_stuck, preds)
        ufpr = compute_useful_fpr(y_useful, preds)
        results[name] = {**m, 'useful_fpr': ufpr}
        print(f"  {name:28s} {m['prec']:7.3f} {m['rec']:7.3f} {m['f1']:7.3f} {ufpr:12.3f} {preds.sum():8d}")

    # ── Figures ──
    print("\nGenerating figures...")
    plot_detector_comparison(results, os.path.join(args.out_dir, 'detector_comparison.png'))
    plot_signal_orthogonality(rows, os.path.join(args.out_dir, 'signal_orthogonality.png'))
    plot_temporal_structure(rows, os.path.join(args.out_dir, 'temporal_structure.png'))

    # ── Case studies ──
    print("\nWriting case studies...")
    write_case_studies(rows, os.path.join(args.out_dir, 'case_studies.md'))

    # ── Sweep k_S for STRIDE ──
    print("\n\nAblation: k_S sweep for STRIDE 3-signal")
    print(f"  {'k_S':>4s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Useful FPR':>12s}")
    for k in [2, 3, 4, 5]:
        preds = detector_stride_3signal(rows, k_S=k)
        m = compute_metrics(y_stuck, preds)
        ufpr = compute_useful_fpr(y_useful, preds)
        print(f"  {k:4d} {m['f1']:7.3f} {m['prec']:7.3f} {m['rec']:7.3f} {ufpr:12.3f}")

    # ── Sweep percentile thresholds ──
    print("\nAblation: percentile sweep for STRIDE 3-signal (k_S=3)")
    print(f"  {'pct':>4s} {'F1':>7s} {'Prec':>7s} {'Rec':>7s} {'Useful FPR':>12s}")
    for pct in [75, 80, 85, 90, 95]:
        preds = detector_stride_3signal(rows, pct_S=pct, pct_H=pct, pct_D=pct)
        m = compute_metrics(y_stuck, preds)
        ufpr = compute_useful_fpr(y_useful, preds)
        print(f"  {pct:4d} {m['f1']:7.3f} {m['prec']:7.3f} {m['rec']:7.3f} {ufpr:12.3f}")

    print(f"\nDone. Results saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
