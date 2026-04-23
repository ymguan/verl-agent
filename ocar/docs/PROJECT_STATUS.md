# OCAR Project Status

> **Last Updated:** 2026-04-14
> **Goal:** Beat GiGPO on ALFWorld, prove observation surprise is a superior credit assignment signal

---

## 1. What is OCAR

**OCAR (Observation-grounded Credit Advantage Redistribution)** uses observation token log-probabilities — already computed but discarded during standard GRPO training — as a step-level credit assignment signal for agent RL.

### Core Formula

$$\phi_t = \text{softmax}\left(\frac{-\text{sign}(A_i) \cdot \Delta S(o_t)}{\tau}\right)$$

- Success trajectories: low surprise → high credit (reinforce grounded behavior)
- Failure trajectories: high surprise → high blame (penalize ungrounded behavior)
- $\Delta S = S_\theta - S_{\text{ref}}$, where $S = -\frac{1}{|o_t|}\sum_{j} \log P(w_j | w_{<j})$

### Zero Cost Claim

OCAR reuses `ref_log_prob` already computed for GRPO's KL penalty. Only extra work: extract obs-position log-probs + softmax reweighting (CPU). Verified <0.5% wall-clock overhead.

### Code Integration

- Advantage estimator: `algorithm.adv_estimator=ocar` in training config
- Core implementation: `ocar/core_ocar.py`
- Config: `ocar.tau=1.0`, `ocar.use_delta_s=true`

---

## 2. Current Results (ALFWorld, Qwen2.5-7B-Instruct)

### 2.1 OCAR Training (3 seeds: 123/456/789)

| Step | Seed 123 | Seed 456 | Seed 789 | Mean ± Std |
|:----:|:--------:|:--------:|:--------:|:----------:|
| 50   | 59.4     | 55.5     | 54.7     | 56.5 ± 2.5 |
| 75   | 68.0     | 64.8     | 69.5     | 67.4 ± 2.5 |
| **100** | **82.8** | **85.2** | **79.7** | **82.6 ± 2.8** |

⚠️ Peak around step ~120 not recorded. Collapse after step 100.

### 2.2 Comparison with Published Methods

| Method | Source | ALFWorld All | vs GRPO | Extra Cost |
|--------|--------|:-----------:|:-------:|:----------:|
| GRPO | GiGPO Tab.1 | 77.6 | — | 0 |
| GRPO+EMPG | EMPG Tab.1 | 78.5 | +0.9 | ~0 |
| **OCAR (ours)** | — | **82.6 ± 2.8** | **+5.0** | **~0** |
| GiGPO | GiGPO Tab.1 | 90.8 | +13.2 | ~0 |
| HCAPO | HCAPO Tab.1 | 91.4 | +13.8 | +8.3% |

### 2.3 Sub-task Breakdown (seed 123)

| Method | Pick | Clean | Cool | Look | Pick2 | Heat | All |
|--------|:----:|:-----:|:----:|:----:|:-----:|:----:|:---:|
| GRPO | 90.8 | 89.3 | 72.5 | 66.1 | 64.7 | 74.7 | 77.6 |
| OCAR | 89.7 | 74.2 | 87.5 | 82.4 | 70.6 | **100.0** | 82.8 |
| GiGPO | 97.7 | 98.8 | 89.3 | 82.7 | 79.2 | 83.7 | 90.8 |

---

## 3. Key Findings (Pre-training Signal Analysis)

### 3.1 Signal Quality

| Metric | Obs Surprise | Action Entropy | Ratio |
|--------|:----:|:----:|:----:|
| Spearman ρ (ALFWorld) | **0.530** | 0.187 | 2.8× |
| Spearman ρ (WebShop) | **0.594** | 0.206 | 2.9× |

Entropy blind spot: 14.0% (ALFWorld) / 48.6% (WebShop) of low-entropy steps are actually Harmful.

### 3.2 Phase Transition

- WebShop: 1 high-surprise step → 75% Useful+Critical (good exploration)
- WebShop: 2 consecutive high-surprise steps → 2% Useful+Critical (73pp cliff!)
- "One surprised step is exploration; two is collapse."
- Action entropy shows no such phase transition (gradual 25pp decline).

### 3.3 Offline Credit-Mass Audit

- Success trajectories: OCAR reduces Harmful credit 20.8% → **3.3%**, boosts Useful+Critical 27.9% → **39.7%**
- Entropy Reweight: Harmful stays at 21.6%, Useful+Critical only 29.7%
- Failure trajectories: OCAR concentrates blame on Harmful steps (**98.0%**)

### 3.4 Inference-time Applications

| Application | Result |
|-------------|--------|
| Best-of-N (WebShop N=4) | **99.1% SR = Oracle** |
| Best-of-N (ALFWorld N=4) | Random 52.2% → **81.3%** (+29.1pp) |
| Early termination (WebShop) | Saves **62.6%** compute, 6% false positive |

---

## 4. Known Issues & Next Steps

### 4.1 Issues to Fix

| Issue | Impact | Root Cause Hypothesis |
|-------|--------|----------------------|
| **Clean subtask 74.2%** (vs GRPO 89.3%) | -15.1pp, largest anomaly | Clean task obs may have low surprise variance → OCAR signal too weak |
| **Collapse after step 100** | Peak not captured | KL penalty insufficient / surprise signal flips on-policy / no lr decay |
| **ΔS very small** (~0.005-0.015) | OCAR weights near uniform in early training | τ=1.0 too large for ΔS ∈ [-0.02, 0]; try τ=0.1 or 0.01 |
| **On-policy surprise flip** | Model predicts own failures perfectly (S≈0.07 for invalid), surprised by success (S≈2.15) | Expected to self-correct as model improves; ΔS may help but needs verification |

### 4.2 Priority Action Items

1. **Diagnose Clean subtask**: output real trajectories + surprise distributions for clean tasks
2. **Tune τ**: sweep τ ∈ {0.01, 0.1, 0.5, 1.0} — current τ=1.0 is too blunt for small ΔS
3. **More checkpoints**: save every 10 steps, capture true peak
4. **Training stability**: try lr decay / higher KL penalty / clip surprise weights
5. **WebShop experiment**: GiGPO degrades on dynamic environments; OCAR should maintain advantage

### 4.3 Strategic Goal

**Beat GiGPO.** Not necessarily on ALFWorld (GiGPO's best case — deterministic env with perfect string matching), but show:
- ALFWorld: close to GiGPO (≤3pp gap acceptable)
- WebShop: surpass GiGPO (its string matching degrades on dynamic content)

GiGPO's structural weakness: requires deterministic environments with identical observation strings for same states. OCAR has no such constraint.

---

## 5. Experiment Configuration

### Training Hyperparameters (aligned to GiGPO/HCAPO papers)

| Param | ALFWorld | WebShop |
|-------|----------|---------|
| Model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| LR | 1e-6 | 1e-6 |
| Group size (G) | 8 | 8 |
| Rollout groups | 16 | 16 |
| Max prompt tokens | 2048 | 4096 |
| Max response tokens | 512 | 512 |
| Rollout temperature | 1.0 | 1.0 |
| KL penalty (β) | 0.01 | 0.01 |
| Reward (success) | +10 | score ∈ [0,1] |
| Reward (failure) | -0.1 | 0 |
| Invalid penalty | -0.1 | — |
| Training steps | 200 | 200 |

### OCAR-Specific

| Param | Current | Notes |
|-------|---------|-------|
| τ (temperature) | 1.0 | **Needs tuning** — too high for ΔS scale |
| use_delta_s | true | ΔS = S_θ - S_ref |
| Signal | obs token mean NLL | Per-step average over observation tokens |

---

## 6. Competitor Landscape

| Method | Core Idea | Extra Cost | Env Constraint | ALFWorld |
|--------|-----------|:----------:|:--------------:|:--------:|
| **GRPO** | Uniform episode advantage | 0 | None | 77.6 |
| **EMPG** | Action entropy reweighting | ~0 | None | 78.5 |
| **OCAR** | Obs surprise reweighting | ~0 | **None** | 82.6 |
| **GiGPO** | State string matching → micro-groups | ~0 | **Deterministic** | 90.8 |
| **HCAPO** | Hindsight verification FP | +8.3% | None | 91.4 |
| **CoPO** | Cognitive depth grouping | +40-50% | None | 92.5* |

\* CoPO uses CoSFT init (GPT-4o annotation), not directly comparable.

---

## 7. Future Ideas (Parked)

- **Dual-Token Training**: Add NLL loss on obs tokens to prevent world-model degradation during RL. Parked until OCAR standalone is competitive. Pre-experiment: check if GRPO obs NLL increases over training.
- **OCAR + GiGPO combination**: Orthogonal methods, technically composable. Only as bonus experiment after standalone OCAR is strong.
