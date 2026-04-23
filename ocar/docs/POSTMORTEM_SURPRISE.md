# Post-Mortem: 放弃 Surprise-as-Credit 路线的实验证据整合

> **文档性质**：项目决策文档。基于 2026-03 至 2026-04 一个多月的 surprise/credit-assignment 实验，整合所有关键数据，说明为什么**终止** OCAR / paper_v6 dual-token / surprise-as-step-reward 这一路线。
>
> **结论摘要**：Surprise 类信号（Δs, s_θ, |Δs|, wm_s, wm_gap, hidden-state IG）在**因果方向、跨域稳定性、step-level 区分力、信用分配合理性、scale-up trend**五个独立维度上全部被证伪或严重削弱。现有训练结果距该环境 SOTA (GiGPO 90.8%) 仍差 −10.1pp 且 6-seed 下未达显著性 (p=0.55)。继续投入算力的期望回报为负。
>
> **数据源**：`ocar/EXPERIMENT_LOG.md` §2/§3/§7/§8/§9/§10, `ocar/PROGRESS_REPORT_20260420_21.md`, `ocar/PROJECT_STATUS.md`, `ocar/analysis_results/**/*.json`
>
> **决策日期**：2026-04-22

---

## 0. TL;DR（给自己、给合作者、给未来回头看的自己）

1. **理论上**：Δs 本质是 GRPO 已发生更新的 KL 局部读数，用它做 credit = **循环论证**（§3.1）。
2. **实证上**：5 类免费信号全部在 step-AUC、跨域方向、信用分配三个独立测试上失败（§3–§5）。
3. **效果上**：dual-token 最终 6-seed 指标 80.7% ± 4.4%，vs observe baseline 79.0% ± 4.8%，**p=0.55 未显著**；vs GiGPO 90.8% 差 **−10.1pp**（§6）。
4. **scale 趋势**：wm_gap 从 7B 的 0.908 塌到 14B 的 0.055——方法在 scale-up 时**边际收益消失**（§7）。
5. **顶会发表可行性**：无 SOTA 希望 + 理论 critique 致命。Workshop / findings 级别可救（作为 negative result），主会无路径。

---

## 1. 研究路线回顾

### 1.1 最初假设（2026-03）

> Observation token 的 NLL（surprise）在 agent RL 中是"免费信号"——既然 GRPO 已经计算了 ref_logprob 做 KL 惩罚，提取 obs 位置的 logprob 差 Δs = s_θ − s_ref 就是零成本的 step-level credit signal。高 surprise = 动作导致了未预期的环境转移 = 信息量大 = 应得高 credit。

### 1.2 方法演化

| 版本 | 时间 | 核心机制 | 结果 |
|---|---|---|---|
| OCAR v1 | 2026-03 早 | per-step softmax(−sign(A)·Δs/τ) 重加权 | step 100–125 训练崩溃 |
| OCAR v2 | 2026-03 中 | 加 clamp + τ tuning | 崩溃推迟但未解决 |
| OCAR v3 | 2026-03 末 | z-norm + adaptive τ + fixed base ref | 3-seed 峰值 82.6 ± 2.8（未达 GiGPO 90.8） |
| paper_v6 dual-token | 2026-04 | 在 obs token 加 NLL aux loss，不用 Δs 做 credit | train SR 反降 / val SR 未显著超 baseline |
| paper_v7 STRIDE | 2026-04 | Surprise 做 test-time detector（不训练） | 跨域验证受阻 |

**每一次 pivot 都在"救 surprise 叙事"而不是"证伪它"**——这是我们踩的最大方法论坑。

---

## 2. 证据汇总表（一张图看完为什么放弃）

| # | 维度 | 证据 | 结论 | 严重度 |
|---|---|---|:-:|:-:|
| E1 | 因果方向 | Δs 数学上 ≡ token-level KL(π\|π_ref) 的符号版本 | 循环论证 | 🔴 理论致命 |
| E2 | Step-level AUC | 4 异构 base scorer × succ/fail step，AUC ∈ [0.483, 0.506] | step 级无信号 | 🔴 实证致命 |
| E3 | 跨域方向 | WebShop r(Δs, succ): −0.53 → +0.65；r(s_θ, succ): −0.66 → +0.76（随 step 翻转） | 不跨域 | 🔴 实证致命 |
| E4 | 信用分配 | WebShop 上 search step 凭 token 生成难度垄断 54–86% 权重 | 方法崩 | 🔴 机制错位 |
| E5 | Hidden-state IG | ALFWorld r=+0.54，WebShop r=−0.31 ~ +0.08 | 不跨域 | 🟠 替代信号也死 |
| E6 | Scale trend | wm_gap: 0.5B=0.42, 7B=0.91, **14B=0.055** | scale-up 消失 | 🟠 前途不佳 |
| E7 | Ref 选择脆弱性 | Moving ref / cross-model ref 使 traj-AUC 从 0.82 塌到 0.46–0.71 | 强依赖超参 | 🟠 不鲁棒 |
| E8 | Train/val 反转 | dual-token train SR −0.12 vs observe，val +0.09 | 机制更像 regularizer | 🟡 叙事模糊 |
| E9 | Framing 三义性 | dual-token 可由 (a) obs-grounded LM / (b) entropy reg / (c) surprise-credit 三种解释观测等价 | 贡献不 identifiable | 🟡 审稿雷 |
| E10 | 最终指标 | 6-seed paper-config: 80.7 vs GiGPO 90.8（−10.1pp），vs observe baseline p=0.55 | 无 SOTA | 🔴 发表致命 |

---

## 3. 理论缺陷：Δs 的因果方向错误（E1，最根本）

### 3.1 数学展开

定义：
- s_θ(o_t) = −(1/|o_t|) Σ log P_θ(w_j | context)
- s_ref(o_t) = −(1/|o_t|) Σ log P_ref(w_j | context)
- **Δs = s_θ − s_ref = (1/|o_t|) Σ [log P_ref − log P_θ]**

对比 token-level KL 的展开：
- KL(P_ref ‖ P_θ) 在 obs token 上的采样估计 ≈ (1/|o_t|) Σ [log P_ref(w_j) − log P_θ(w_j)] = **Δs**

**即 Δs 就是 obs 位置的 token-level reverse-KL 的 MC 估计。**

### 3.2 GRPO 训练过程的因果链

```
GRPO 对成功轨迹 action token 给正 advantage
  → P_θ(a_t) ↑ → P_θ(obs_{t+1} | context) 作为条件概率间接变化
  → s_θ(obs) 下降（对 succ 轨迹）
  → Δs = s_θ − s_ref 变小/变负（对 succ 轨迹）

GRPO 对失败轨迹 action token 给负 advantage
  → P_θ(a_t) ↓ → context 漂移
  → s_θ(obs) 上升（对 fail 轨迹）
  → Δs 变大/变正（对 fail 轨迹）
```

**所以 Δs 与 outcome 的相关（ALFWorld 7B step150 r=+0.75）本质是在读回 "GRPO 已经做过的 trajectory-level 更新"**——不是动作质量的独立因果信号。

### 3.3 引用"用 Δs 做 step-reward"的循环论证

```
Δs(t) ← GRPO 对 traj_i 的 trajectory-level 更新结果
step_reward(t) = f(Δs(t))    ← 作为 credit
GRPO 下一轮 update ← trajectory-level advantage × step_reward
                   ← trajectory-level advantage × f(上一轮 trajectory update 的结果)
```

**这是在用"训练已经发生了什么"当作"训练应该做什么"的信号。**任何 reviewer 一旦意识到这一点，论文必被拒。

### 3.4 对 s_θ、|Δs|、z-score Δs 同样成立

- **s_θ**：虽然看似独立（"模型对环境的预测能力"），但 action token 分布变化通过共享 context 间接污染 obs 的条件概率。WebShop r(s_θ, succ) 从 −0.66 → +0.76 翻转就是这个 indirect 污染的表现（§10.4）。
- **|Δs|, z(Δs)**：线性变换不改变相关系数方向。只解决 scale，不解决因果（§10.5）。

---

## 4. 实证失败一：Step-level AUC ≈ 0.5（E2）

### 4.1 跨 4 个异构 base scorer 的 succ/fail step AUC

> 方法：固定从训练 rollout 中采 succ/fail step pair，用不同 base model 做 scorer 计算 s_θ，检验 step 级是否能区分 outcome。

| Base scorer | 参数量 / 族 | succ/fail step AUC | 95% CI |
|---|:-:|:-:|---|
| Qwen2.5-0.5B-Inst | 0.5B, Qwen2.5 | 0.491 | [0.476, 0.506] |
| Qwen2.5-7B-Inst | 7B, Qwen2.5 | 0.503 | [0.489, 0.517] |
| Qwen3-8B-base | 8B, Qwen3 | 0.498 | [0.484, 0.512] |
| Qwen3-14B-base | 14B, Qwen3 | 0.506 | [0.492, 0.520] |
| 训练起点 base ckpt | 7B | 0.483 | [0.468, 0.498] |

**全部 95% CI 覆盖 0.5**——raw s_θ 在 step 级不携带 outcome 信号。

**含义**：即便用更强的 base model 做"更干净的 reference"，Δs 的 step-level 区分力依然为零。E2 彻底排除了"用更好的 ref model 拯救 OCAR"的可能性。

**数据源**：`ocar/EXPERIMENT_LOG.md §2.3`, `ocar/analysis_results/scale_scan/*.json`

---

## 5. 实证失败二：跨域方向翻转（E3）

### 5.1 轨迹级 r(signal, success) 跨训练步完整表

| 数据集 | step | r(Δs) | r(entropy) | r(s_θ) | r(wm_s) | r(wm_gap) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| ALFWorld 1.5B | 20 | +0.08 | −0.23 | +0.09 | −0.20 | −0.07 |
| ALFWorld 1.5B | 40 | +0.32 | −0.46 | — | −0.30 | −0.43 |
| ALFWorld 1.5B | 60 | +0.06 | −0.49 | +0.21 | −0.26 | −0.22 |
| **ALFWorld 7B** | **150** | **+0.75** | **−1.00** | **−0.50** | **−0.64** | **−0.46** |
| WebShop 7B | 80 | −0.47 | −0.79 | −0.22 | −0.28 | −0.47 |
| WebShop 7B | 160 | +0.10 | −0.65 | — | −0.20 | −0.55 |
| WebShop 7B | 240 | +0.65 | −0.73 | −0.66 | −0.10 | −0.32 |
| WebShop 7B | 320 | −0.49 | −0.38 | — | −0.23 | −0.26 |
| WebShop 7B | 400 | −0.47 | −0.02 | +0.13 | +0.21 | +0.01 |
| WebShop 7B | 480 | +0.40 | −0.12 | — | −0.08 | +0.33 |
| WebShop 7B | 560 | −0.01 | −0.49 | — | +0.17 | −0.29 |
| WebShop 7B | 640 | −0.53 | +0.19 | +0.76 | +0.52 | +0.60 |

### 5.2 关键观察

- **Δs 方向剧烈翻转**：WebShop r 从 −0.53 到 +0.65，振幅 1.18，跨 0 轴多次。
- **s_θ 方向翻转**：WebShop r 从 −0.66 到 +0.76，振幅 1.42。
- **wm_s / wm_gap 同样不稳定**，且 wm_gap 需要额外 forward pass——**既不免费也不稳定**。
- **Entropy 前中期跨域一致为负**（模型不确定 → 容易失败），但 WebShop 后期（step 400+）衰减并翻正。
- **所有免费信号在训练后期 WebShop 上都失效**。

### 5.3 为什么翻转

随训练推进，policy 偏离 ref 越来越远，新成功/失败模式产生，Δs 反映的是不同训练阶段的更新历史——它是一个**非平稳的读出**，不是一个稳定的测量。这与 §3 的因果链完全一致：你测量的是"训练已做了什么"，训练内容本身在变，所以读数方向在变。

**数据源**：`ocar/EXPERIMENT_LOG.md §10.2`, `ocar/analysis/cross_dataset_signal.py`, `ocar/analysis/s_theta_signal.py`

---

## 6. 实证失败三：信用分配机制错位（E4）

### 6.1 用 weight_i = |Δs_i| / Σ|Δs_j| 做信用分配的 case 分析

**ALFWorld 7B 成功轨迹**（8 步，表现"合理"）：

| step | action | |Δs| 权重占比 |
|---|---|:-:|
| 3 | cool apple 1 with fridge 1 | 22% |
| 5 | go to countertop 1 | 15% |
| 7 | open microwave 1 | 12% |
| 其他 5 步合计 | navigation/inspection | 51% |

**WebShop 成功轨迹**（8 步，暴露 token-生成-难度 bias）：

| step | action 类型 | |Δs| | 权重占比 |
|---|---|:-:|:-:|
| 0 | search[long query] | 0.78 | 54–86% |
| 1–3 | click[B08ABCD] | 0.01–0.05 | 2–4% each |
| 4 | click[Color: blue] | 0.03 | 3% |
| 5–7 | click / Buy Now | 0.02–0.13 | 各 2–13% |

**同样模式在失败轨迹重复**：search 垄断权重，click 被边缘化。

### 6.2 根因

**|Δs| 本质测量 token 生成难度**：
- Search query 是自由文本长串 → 每个 token 都有不确定性 → |Δs| 天然大
- Click 是短按钮文本（商品 ID、颜色、"Buy Now"）→ |Δs| 天然小

**Token 生成难度 ≠ 步骤重要性**。WebShop 中决定成败的恰恰是 click 选择（选对商品、选对规格），但这些步骤在 |Δs| 下权重被压到 2–4%。

**数据源**：`ocar/EXPERIMENT_LOG.md §10.6`, `ocar/analysis/credit_assignment.py`

---

## 7. 实证失败四：替代信号（Hidden-state IG）同样不跨域（E5）

| 数据 | succ IG (layer -1) | fail IG (layer -1) | gap | r(IG, succ) | p |
|---|:-:|:-:|:-:|:-:|:-:|
| ALFWorld step150 | 278.2 | 101.6 | **+176.6** | +0.538 | <0.0001 |
| ALFWorld base ckpt | 286.4 | 112.9 | **+173.5** | +0.500 | <0.0001 |
| WebShop step 240 | 184.9 | 248.7 | **−63.8（反向）** | −0.305 | 0.020 |
| WebShop step 640 | 238.2 | 224.7 | +13.5 | +0.077 | 0.946 |

**ALFWorld 上强信号且跨模型稳定；WebShop 上方向翻转且极弱。**

### 7.1 根因诊断

- WebShop obs 平均 2000+ token/step，max_length=4096 截断导致 **30% 的 succ step hidden_state pre==post → IG=0**（伪影）
- 修复截断后，WebShop succ/fail IG gap 仅 ~3%（ALFWorld 是 ~160%）
- 深层原因：WebShop search_page vs product_page 结构异质，hidden state 变化主要被"页面类型切换"主导，与任务进展无关

**结论**：hidden-state IG 作为"更深层次的信号"也不具备跨域通用性。替代方案失败。

**数据源**：`ocar/EXPERIMENT_LOG.md §9`, `ocar/analysis_results/hidden_state/*.json`

---

## 8. Scale 失败：14B 上 wm_gap 塌到 0.055（E6）

### 8.1 Scale scan（4 模型 × 12 trajectory）

| 模型 | obs_nll_last | wm_A | wm_B | **wm_gap** | succ-fail gap |
|---|:-:|:-:|:-:|:-:|:-:|
| Qwen2.5-0.5B | 2.013 | 3.419 | 3.838 | 0.420 | 0.077 |
| **Qwen2.5-7B-Instruct** | **1.793** | 4.185 | 5.093 | **0.908** | 0.119 |
| Qwen3-8B | 2.584 | 4.591 | 5.070 | 0.479 | 0.222 |
| **Qwen3-14B** | 2.361 | 4.839 | 4.893 | **0.055** | 0.240 |

### 8.2 诊断

- 7B 看起来是"甜点"，但与 Qwen3 14B 对比后，**wm_gap 消失**——可能是代际/SFT 混淆，也可能是"更强模型对 context 足够敏感，不需要 obs 作为额外信号"
- 就算修复混淆，14B 的 wm_gap 中位数 0.055 意味着**信号基本不存在**
- 顶会审稿人普遍关心 scale-up 趋势。一个"只在 7B 有效、14B 消失"的方法会被直接质疑：**工业界上了 70B+ 你这套还有意义吗？**

**数据源**：`ocar/EXPERIMENT_LOG.md §7.2`

---

## 9. 最终指标：距 SOTA −10.1pp，baseline 对比未显著（E10）

### 9.1 ALFWorld 6-seed paper-config（t=0.4，对齐 GiGPO）

| 方法 | 123 | 456 | 789 | 42 | 2024 | 7 | **mean ± std** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| GRPO+observe step120 | 0.867 | 0.781 | 0.734 | 0.766 | 0.836 | 0.758 | **0.790 ± 0.048** |
| dual-token step150 | 0.875 | 0.789 | 0.750 | 0.781 | 0.844 | 0.805 | **0.807 ± 0.044** |

### 9.2 显著性与对标

| 对比 | Δ | 显著性 |
|---|:-:|:-:|
| dual-token vs observe baseline (6 seeds, t=0.4) | **+1.7pp** | Welch t≈0.6, **p≈0.55（不显著）** |
| observe t=0.4 vs t=0.7 | −2.3pp | — |
| **vs GiGPO (paper)** | **−10.1pp** | — |
| vs HCAPO (paper) | −10.7pp | — |
| vs CoPO (paper) | −11.8pp | — |

### 9.3 OCAR v3 3-seed 历史峰值

| Step | Seed 123 | Seed 456 | Seed 789 | Mean ± Std |
|:----:|:--------:|:--------:|:--------:|:----------:|
| 50   | 59.4     | 55.5     | 54.7     | 56.5 ± 2.5 |
| 75   | 68.0     | 64.8     | 69.5     | 67.4 ± 2.5 |
| **100** | **82.8** | **85.2** | **79.7** | **82.6 ± 2.8** |

**峰值 82.6 依然距 GiGPO 90.8 差 8.2pp**，且 step 100 后训练崩溃（见 Appendix A.1）。

### 9.4 顶会发表可行性判断

- **ICML/NeurIPS/ICLR 的 agent RL track** 默认对标 GiGPO 90.8；低于 90% 基本无 chance。
- ALFWorld 已是 GiGPO 的"最优土壤"（确定性环境 + 精确 string matching），surprise 类方法没有结构优势能追上去。
- Pivot 到 WebShop 做主战场也无救：WebShop 上 Δs 方向翻转 + 信用分配失败，正面结果概率极低。

---

## 10. Framing 问题：Dual-token 的三义性无法识别（E9）

### 10.1 Train/val 反转现象

ALFW 同 step 网格（n=30 对照点）：

| metric | GRPO+observe end3 | dual-token end3 | Δ(dt−obs) |
|---|:-:|:-:|:-:|
| **train SR** | **0.836** | 0.714 | **−0.122** |
| **val SR** | 0.807 | 0.812 | +0.005 |
| val SR 末点 | 0.797 | **0.883** | **+0.086** |
| step_entropy | 0.882 | 0.758 | −0.124 |
| s_θ | 2.193 | 1.942 | −0.251 |
| wm_s | 3.822 | 3.100 | −0.722 |
| ΔS | −0.005 | −0.356 | −0.351 |

### 10.2 三种观测等价的 framing

| framing | 是否与数据相容 | 是 dual-token loss 的"必要"机制吗？ |
|---|:-:|:-:|
| (a) Observation-grounded LM 防 policy drift | ✓ | 无法证明 |
| (b) 隐式 entropy regularizer（压低 policy entropy 减少 over-exploit） | ✓ | 无法证明 |
| (c) Surprise-guided credit（放大 h(θ, o_t)） | ✓ | 无法证明 |

**三者观测等价**。要识别 (a) vs (b)，需要 **GRPO + 纯 entropy bonus (β ∈ {0.005, 0.01, 0.02}, 6 seeds)** 对照实验。但即使此实验做完：

- 若 entropy bonus 复现 val gain → dual-token 贡献降级为 "NLL loss ≈ entropy reg"，**贡献退化到 workshop 级**。
- 若 entropy bonus 不能复现 → (a) 成立，但 9.2 已经说明 +1.7pp 不显著，**仍发不了主会**。

**任何一个分支都不足以支撑顶会论文**。这是一个"做 ablation 就死，不做 ablation 更死"的困境。

---

## 11. 数据资产与可复用清单

即使项目终止，以下**不丢弃**（迁移到下一个方向）：

### 11.1 Infra（100% 复用）

- `verl-agent` GRPO + rollout pipeline（支持 ALFWorld / WebShop）
- Cross-model logprob extraction / obs-NLL 计算
- Multi-seed eval harness（`ocar/run_overnight_experiments.sh`）
- WandB 集成 + analysis scripts

### 11.2 分析脚本（可迁移到 on-policy distillation）

- `ocar/analysis/cross_dataset_signal.py` - 跨数据集相关性框架
- `ocar/analysis/hidden_state_info_gain.py` - hidden-state probing
- `ocar/analysis/obs_type_traj_avg.py` - obs-type 分层统计
- `ocar/analysis/credit_assignment.py` - per-step weight 可视化

### 11.3 数据产出（有独立价值）

- `ocar/analysis_results/` 下 **跨 2 环境 × 8 训练步点 × 12 信号** 的完整 JSON——是一个小型的 "agent RL free-signal benchmark"，可作 workshop paper 主数据。

### 11.4 Negative-result paper 预留

- §2–§10 的证据结构已足以支撑 **8-page short paper**（NeurIPS workshop / ICML workshop / EMNLP findings）：
  - Title 候选："A Post-Mortem of Free Signals for Credit Assignment in Agent RL"
  - Contribution：系统性证明 4 类免费信号（NLL-based, hidden-state-based, entropy-based, world-model-based）的 step-level AUC ≈ 0.5 + 跨域方向翻转 + 因果缺陷
  - Format：workshop 8p 或 findings 8p 都能放下
  - 预估投入：2 人周整理 + 2 人周写作 = **约 1 个月**

---

## 12. 终止决策（2026-04-22）

### 12.1 立即停止

- ❌ 任何"修改 Δs / s_θ / |Δs| 公式再跑一轮"的实验
- ❌ WebShop 上的 dual-token 方法实验（原计划 §1.3 实验 C）
- ❌ τ sweep / lr decay / KL penalty tuning 来救训练崩溃
- ❌ 3 代 × 5 scale 的纯 base scorer 方差分解（原计划 §1.3 实验 E）
- ❌ 任何新的"基于 policy forward pass 免费信号"的 idea

### 12.2 保留/收尾（时间盒 ≤1 个月，2 研究生并行）

- ✅ 把 §2–§10 整理为 workshop/findings paper（2 人周）
- ✅ 已有 6-seed 数据 + 跨域 12 step 数据清洗入 git（1 人周）
- ✅ Infra 迁移到 on-policy distillation 方向（1 人周）

### 12.3 不做：GRPO + entropy bonus ablation

原 §1.3 P0 实验。**放弃原因**：即使该实验出任一结果，都不足以让 dual-token 进主会（见 §10.2）。投入 2 A100-week 换一个最多发 workshop 的 insight，在"必须带训练 + 目标顶会"的约束下 ROI 为负。

### 12.4 新方向

确认为 **On-policy distillation for Agent RL**（非 ALFWorld 环境，见配套决策文档）。复用本项目 80%+ infra。

---

## Appendix A: Case 证据（辅助 §3–§4 的定性说明）

### A.1 同 observation 下同 Δs，好坏动作无法区分（WebShop step 240）

初始搜索页 `'Search'`，Δs = **−0.8189**、s_θ = 3.6576 完全一致：

| outcome | traj | action |
|---|---|---|
| **FAIL** | `18b2ff97` | `search[loose fit women's tops, tees & blouses c1-blue x-large size short sleeve long sleeve $30.00 max price]` |
| SUCCESS | `f239659e` | `search[loose fit women's tops c1-blue c1-blue short sleeve x-large size]` |
| SUCCESS | `cd29e904` | `search[loose fit women's tops, tees & blouses short sleeve c1-blue x-large size under $30.00 USD]` |
| SUCCESS | `97e0cc8c` | `search[loose fit women's tops, tees & blouses short sleeve c1-blue x-large size $30.00 max price]` |

→ Δs 完全由 obs 文本决定，对"搜索能否命中"零区分力。

### A.2 Δs 最高 step 反而是失败轨迹 + 错误动作（ALFWorld 7B step 150）

| Δs | outcome | observation | action |
|---|---|---|---|
| **+0.3817** | **FAIL** | `You turn on the desklamp 1.` | `go to dresser 1` |
| +0.3789 | FAIL | `You turn on the desklamp 1.` | `examine sidetable 3` |
| +0.3524 | FAIL | `You arrive at dresser 1. ...` | `go to sidetable 3` |
| +0.3402 | FAIL | `Nothing happens.` | `go to sidetable 3` |

### A.3 Δs 最负 step 恰是成功轨迹的任务关键动作（ALFWorld 7B step 150）

| Δs | outcome | observation | action |
|---|---|---|---|
| −0.2596 | **SUCCESS** | `You pick up the soapbottle 2 from the cabinet 3.` | `go to toilet 1` |
| −0.2508 | SUCCESS | `You arrive at cabinet 3. The cabinet 3 is closed.` | `open cabinet 3` |
| −0.2448 | SUCCESS | `You arrive at toilet 1. ...` | `move soapbottle 2 to toilet 1`（任务关键） |

**→ 若按 "Δs 高=好" 分配 credit，credit 方向完全反。**

### A.4 反直觉率统计

| 数据 | SR | Succ step Δs<p10 占比 | Fail step Δs>p90 占比 |
|---|:-:|:-:|:-:|
| ALFWorld 1.5B step 60 | 23% | 5.1% (43/848) | **10.2%** (503/4912) |
| ALFWorld 7B step 150 | 91% | 1.5% (15/975) | 5.0% (28/561) |
| WebShop step 240 | 47% | **11.2%** (14/125) | 1.6% (2/123) |

即在信号最干净的 ALFWorld 7B 上，也有 5% 的 fail step 被 Δs 排进 top-10%——作 step-reward 直接把这些 credit 打反。

### A.5 同 prompt 多次 rollout，Δs 完全一致但 outcome 不同（WebShop step 320）

4 条 FAIL 轨迹的 search 动作 Δs **全部 = −0.6440**：

```
search[women's lingerie, sleep & lounge, long sleeve, tummy control, high waist, short sleeve, purple, xx-large, price < 30.00 dollars]
```

**同一 obs + 极相似 action → Δs 固定**。它是 token 生成难度的读数，不是 outcome 预测。

---

## Appendix B: 引用与数据源清单

| 来源 | 章节 | 本文档引用位置 |
|---|---|---|
| `ocar/EXPERIMENT_LOG.md` | §1 核心结论、§2 Surprise 第一性原理 | §2, §3, §4 |
| `ocar/EXPERIMENT_LOG.md` | §3 Entropy、§4 Framing | §10 |
| `ocar/EXPERIMENT_LOG.md` | §7.1 Extra-seed eval | §9 |
| `ocar/EXPERIMENT_LOG.md` | §7.2 Scale scan | §8 |
| `ocar/EXPERIMENT_LOG.md` | §8 跨域通用发现 | §5 |
| `ocar/EXPERIMENT_LOG.md` | §9 Hidden state IG | §7 |
| `ocar/EXPERIMENT_LOG.md` | §10 Step-level 深入 | §3, §5, §6, Appendix A |
| `ocar/PROGRESS_REPORT_20260420_21.md` | §2 轨迹级 r | §5 |
| `ocar/PROGRESS_REPORT_20260420_21.md` | §3–§4 obs-type 分解 | §6 |
| `ocar/PROGRESS_REPORT_20260420_21.md` | Appendix A Case | Appendix A |
| `ocar/PROJECT_STATUS.md` | §2 训练结果 | §9 |
| `ocar/analysis_results/scale_scan/` | 4-scorer AUC | §4 |
| `ocar/analysis_results/hidden_state/` | IG JSON | §7 |
| `ocar/analysis_results/webshop/` | WebShop 跨 step | §5 |
| `ocar/analysis_results/entropy_surprise/` | 正交性数据 | §10 |

---

**文档状态**：Final (2026-04-22)
**下一步**：启动 On-policy distillation for Agent RL（配套决策文档另起）
