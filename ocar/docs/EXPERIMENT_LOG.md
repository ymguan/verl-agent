# OCAR Experiment Log — Surprise & Entropy 双信号

> **文档定位**：从第一性原理出发，把 observation-NLL surprise 和 action entropy 两个信号刻画清楚，并说明它们对方法（paper_v6 dual-token）的真实含义。
>
> **阅读顺序**：§1 是 punch list 入口，读完就知道我们相信什么、不相信什么、还要做什么。§2-3 是两个信号的第一性原理刻画（互相独立）。§4 把这些含义接到方法上。§5-7 是跨环境、证据分级、实验账本。Appendix 收纳方法演化史和原始数据。
>
> **历史备份**：`ocar/EXPERIMENT_LOG.md.orig`（2026-04-19 重构前 1236 行原文档）。

---

## 1. 核心结论一览（进门先看）

### 1.1 强证据（✅ 已成立）

- **Raw s_θ 在 step 级不携带 outcome 信号**。4 个异构 base scorer（Qwen2.5-0.5B/7B-Inst, Qwen3-8B/14B-base）+ 训练起点 base ckpt，succ/fail AUC 全部 ∈ [0.483, 0.506]。→ §2.3, 表 §C.1 Q3
- **ΔS 相对固定 base ref 有 traj 级信号**：AUC 0.82–0.86（p<0.05），只来自"累积训练偏移"，moving ref / cross-model ref 全部退化到 0.46–0.71。→ §2.3, §2.5, 表 §C.2 Q3
- **Entropy 是比 ΔS 更强、更早的 within-batch outcome 信号**。dual-token n=150 step 内，`fail_ent − succ_ent` 恒正（P(f>s)=1.000, t=17.24, p=1e-37）；前 5 个 log step 就达 p<0.01。→ §3.1, 表 §3.1
- **Dual-token 在 train SR 上退步、在 val SR 上前进**：train end3 0.714 vs observe 0.836（Δ −0.07），val 末点 0.883 vs 0.797（Δ +0.09）。→ §4.1
- **Residual hardness**：随 SR 升高，fail 轨迹 entropy 发散，ρ(SR, ent_gap) = +0.655（p<1e-19, n=150）。→ §3.2
- **Obs NLL 在 ALFWorld / Webshop 两域都呈 U 型**，但 ΔS 累积性在 webshop 上消失——**方法适用性由 ΔS drift 决定**。→ §5.1-5.3, §8.1
- **Entropy × ΔS step 级正交**（corr ≈ −0.12 / +0.23），信号互补性跨域成立。→ §8.2
- **Residual hardness 跨域成立**：fail entropy 随 SR 升高而发散。→ §3.2, §8.3
- **14B wm_gap ≈ 0**（0.055 vs 7B 的 0.908）——dual-token 类方法在更大模型上边际收益预期显著下降。→ §7.2, 表 §C.1 Q6
- **Δs 因果方向错误**：Δs 本质是 KL divergence 局部版本，记录训练偏移而非动作质量。WebShop 上 r(Δs, success) 在 −0.53 到 +0.65 之间翻转。→ §10.3
- **所有 NLL 差异信号（Δs, s_theta, |Δs|）均受训练污染**，方向跨域不稳定。→ §10.3–10.5
- **|Δs| 信用分配在 WebShop 上失败**：search 步骤垄断 54–86% 权重（token 生成难度 ≠ 步骤重要性）。→ §10.6
- **Entropy 是 policy forward pass 中唯一因果方向正确且跨域稳定的免费信号**。→ §10.8
- **Entropy slope（轨迹内趋势）方向跨域一致**：失败轨迹 slope 始终 > 成功轨迹，但信号强度弱。→ §10.7

### 1.2 已撤回 claim（❌ 不再 assert）

- **OCAR 做通用 per-step credit assignment** —— §2.3 跨 4 异构 base 反证
- **Surprise density 作 intrinsic reward** —— §6.2 分析 + §3.2 label 非平稳
- **更大 base model 会让 surprise 成为 outcome 信号** —— §2.3 AUC 全部 ≈ 0.5 反证
- **wm_gap 7B 甜点** —— 代际×SFT 混淆（§C.1 Q5）
- **Hidden state info gain 做通用 intrinsic reward** —— ALFWorld 强但 WebShop 退化为噪声，不跨域（§9）
- **Stride / isolated-consecutive 规则检测 stuck** —— §A.5 反证
- **Dual-token 优势 = "observation-grounded language modeling"** —— §4.2 揭示此解释和 "entropy regularizer" 观测等价，需对照实验才能分开
- **OCAR v1 崩溃 = clamp 缺失** —— §A.1 更新：根因是 selective fail reconstruction
- **Δs 做 step-level reward / 信用分配权重** —— §10.3 因果方向错误（训练结果回读），§10.6 WebShop 上 search 垄断信用
- **s_theta_mean（obs NLL）做独立信号** —— §10.4 间接受训练污染，方向跨域翻转
- **|Δs| 或 z-score Δs 修复方向不稳定** —— §10.5 线性变换不改变相关系数方向

### 1.3 未决实验（🔬 待跑）

| # | 实验 | 回答的问题 | 优先级 |
|---|---|---|:-:|
| A | GRPO + entropy bonus（β ∈ {0.005, 0.01, 0.02}, 6 seeds） | Dual-token 优势是 observation-grounded LM 还是 entropy regularizer？ | P0 |
| B | Entropy 作 scorer 在 §C.2 20-traj 集上方差分解 + AUC | Entropy 和 ΔS 在 traj 级是 orthogonal 还是 redundant？ | P1 |
| C | Dual-token 在 webshop 上训练 | 方法跨域性 + 检验 §4.1 train/val 反转是否环境独立 | P1 |
| D | Nats/token 绝对尺度参照系（random / n-gram / in-domain fit ceiling） | 当前 s_θ ≈ 1.8 是大是小？ | P2 |
| E | 3 代 × 5 scale 纯 Base scorer 方差分解（14 模型） | 分开 §C.1 的 scale×代×SFT 混淆 | P2 |

---

## 2. Surprise 信号的第一性原理 (data: 2026-04-19)

> 目的：**不引入新假设**，从定义出发刻画 obs-NLL surprise。所有结论标注证据出处。

### 2.1 定义

$$s_\theta(o_t \mid c_{<t}, a_t) = -\frac{1}{|o_t|}\sum_k \log \pi_\theta(o_t^k \mid c_{<t}, a_t)$$

**就是且仅是** policy 当前权重下 observation tokens 的平均 NLL。
- 不是 value，不是 reward
- 不是独立 transition model（没有独立 WM head）
- 不是 action 评估

后续性质由"LM head 条件 NLL"这一本质决定。

### 2.2 方差分解：三成分结构 ⭐

同一批轨迹只变 scorer，两套独立分解同向：

| 数据 | between-step/traj | between-scorer | scorer×step 交互 | 证据 |
|---|:-:|:-:|:-:|:-:|
| 4 模型 × 93 step（SS=1443.7） | **87.6 %** | 2.4 % | 10.0 % | §C.1 Q2 |
| 6 ckpt × 20 traj（SS=17.25） | **74.5 %** | 21.6 % | **3.9 %** | §C.2 Q4 |

$$s_\theta(o_t) = \underbrace{f_\text{text}(o_t)}_{74\text{–}88\%} + \underbrace{g(\theta)}_{2\text{–}22\%} + \underbrace{h(\theta, o_t)}_{4\text{–}10\%}$$

- $f_\text{text}$：**文本/step 内在难度**。跨 0.5B→14B、Qwen2.5→Qwen3、Instruct→base，step 排序 Spearman ≥ 0.89（§C.1 Q1），跨 ckpt traj 排序 ≥ 0.85（§C.2 Q1）。**不随 scorer 变化。**
- $g(\theta)$：**均值偏移**。跨异构模型 2.4%，跨训练 21.6%（主要是整体 NLL 下降）。**z-norm 或 $s_\theta - s_\text{ref}$ 消除。**
- $h(\theta, o_t)$：**scorer × step 交互**，唯一携带模型特异信息的成分。

### 2.3 Outcome 信号只存在于 $h(\theta, o_t)$

**Step 级，任何 base scorer AUC ≈ 0.5**：

| scorer | succ/fail AUC | p |
|---|:-:|:-:|
| Qwen2.5-0.5B-Instruct | 0.495 | 0.94 |
| Qwen2.5-7B-Instruct | 0.498 | 0.98 |
| Qwen3-8B-base | 0.483 | 0.78 |
| Qwen3-14B-base | 0.506 | 0.92 |
| Training base ckpt（traj 级） | 0.355 | 0.29 |

**outcome 信号与模型容量、代际、SFT 状态无关——它只能来自训练诱导的 drift。**

**Traj 级，只有"固定 base ref 累积 drift"携带强信号**：

| signal | traj AUC | p |
|---|:-:|:-:|
| Δ(step50 − base) | 0.820 | 0.017 |
| Δ(step100 − base) | 0.835 | 0.013 |
| Δ(step150 − base) | **0.860** | **0.007** |
| Δ(step75 − step50)（moving ref） | 0.640 | 0.31 |
| Δ(step125 − step100) | 0.620 | 0.38 |
| Δ(Qwen3-14B − Qwen2.5-7B)（cross-model） | 0.459 | — |

→ $h(\theta, o_t)$ 信息来自**累积训练偏移量**，不是**当前 drift 速度**，更不是**跨 scorer 容量差**。

### 2.4 机制：训练 selectively 重构 fail 轨迹

Rank stability 按 succ/fail 拆解（§C.2 Q5）：

| ckpt vs base | all | succ-only | fail-only |
|---|:-:|:-:|:-:|
| step 50 | 0.926 | 0.900 | 0.827 |
| step 100 | 0.854 | **0.903** | **0.692** |
| step 125 | 0.867 | **0.903** | **0.680** |
| step 150 | 0.899 | 0.900 | 0.768 |

训练对 succ 轨迹排序几乎不变（≈0.90），对 fail 轨迹排序在 step 100–125（v1 崩溃窗口）大幅重构（0.69）。结合 Δ 的均值（fail: −0.57 vs succ: −0.35）：

**训练对 fail 轨迹的 NLL 降得更多 → 抹平 base 本有的 −0.23 gap → AUC 从 0.355 趋近 0.510。**

"succ-fail gap 消失"不是信号失效，而是**训练主动消灭**了这个区分度。

### 2.5 Ref model 选择由证据强约束

基于 §2.3，$\Delta S = s_\theta - s_\text{ref}$ 的有效性完全取决于 ref：

| ref 类型 | traj AUC | 机制 |
|---|:-:|---|
| **固定 base model（训练起点）** | **0.82–0.86** | 捕获累积训练偏移 |
| Moving ref（上一 ckpt） | 0.46–0.71 | 只剩局部速度，几无 outcome |
| 跨模型 ref（其他 scale/代际） | 0.46–0.54 | 无共享 drift 方向 |

**必要条件**：ref 必须是**同一训练起点的 frozen base**。这直接反驳"用更大模型做 ref 可以增强信号"的直觉。

### 2.6 本质刻画与优缺点

> **$\Delta S$ 相对 frozen base 是"θ 的累积训练偏移在 obs token 上的投影"，占 $s_\theta$ 总方差约 4%。通过 ref 减法消除 $f_\text{text}$ 和 $g(\theta)$ 均值偏移，保留唯一携带 outcome 的 $h(\theta, o_t)$。该成分"稀薄但有力"：traj 级 AUC 0.82+，step 级 ≈ 0.5。机制来源于 §2.4 selective 重构，不是模型容量、代际或 SFT 状态。**

| 优点 | 缺点 |
|---|---|
| 正交于 entropy（step-level corr −0.12） | 信号稀薄：step 级 4–10%，traj 级需聚合 |
| 长度鲁棒（partial-out 后仍有效） | 冷启动慢，可用窗口 step 50–150 |
| 零额外模型（复用 LM head + frozen base） | 不编码 reward（AUC ≈ 0.5） |
| z-norm 后幅度自稳 | 任务方向偏差不可超参修复 |
|  | Succ/fail 区分度随训练消失 |
|  | 对 ref 选择极度敏感 |

---

## 3. Action Entropy：并列信号 (data: 2026-04-19)

> §9 原版把 obs-NLL 当"唯一 observation-derived outcome 信号"，§3 升级为 entropy 并列——它来自 policy head 的 step-wise Shannon entropy，和 surprise 不共用计算路径。

### 3.1 Entropy 比 ΔS 更强、更早的 within-batch 信号

dual-token n=150 log step：

| signal | mean gap | P(f>s) | t | p |
|---|:-:|:-:|:-:|:-:|
| `fail_ent − succ_ent`（ALFW dual） | **+0.452** | **1.000** | **17.24** | **2.5e-37** |
| `fail_s_θ − succ_s_θ`（ALFW dual） | +0.021 | 0.513 | 2.06 | 0.041 |
| `fail_ent − succ_ent`（Webshop obs） | +0.093 | 0.871 | 6.95 | 1e-07 |
| `fail_s_θ − succ_s_θ`（Webshop obs） | +0.083 | 0.613 | 1.96 | 0.059 |

**早期可用性**（cumulative t-test, ALFW dual）：

| first k steps | mean gap | t | p |
|---|:-:|:-:|:-:|
| 5 | +0.060 | 5.80 | 4e-3 |
| 10 | +0.063 | 11.09 | 2e-6 |
| 30 | +0.087 | 16.87 | 2e-16 |
| 100 | +0.266 | 13.17 | 2e-23 |

**entropy gap 在前 5 个 log step 就显著**；ΔS 要 ckpt step 50 才有 traj-AUC 0.82。**快 5–10 倍，且 step 级就成立**——这是 §2.3 "step-level AUC ≈ 0.5" 死线的旁路。

含义：**paper_v6 dual-token 叙事需要主动回应"为什么不用 entropy"**。

### 3.2 Residual hardness 现象

ALFW dual-token 按训练进度四等分：

| quartile | step | ent_gap(f−s) | s_θ_gap(f−s) | SR |
|---|:-:|:-:|:-:|:-:|
| Q1 | 1–38 | +0.099 | +0.016 | 0.435 |
| Q2 | 39–76 | +0.240 | +0.003 | 0.629 |
| Q3 | 77–113 | **+0.638** | +0.031 | 0.659 |
| Q4 | 114–150 | **+0.848** | +0.036 | 0.711 |

ρ(SR, ent_gap) = **+0.655**, p=9.6e-20, n=150。

**解读**：早期 fail ≈ "随机失败"，entropy 不特别高；后期 SR 升 → 剩下的 fail 是 hard residual（policy 无法 commit），entropy 自然高。这是 §2.4 selective 重构的**entropy 面镜像**。

**含义（§6.2 intrinsic reward 风险补充第 5 条）**：任何用 "high-entropy fail" 做 credit 的方法，**label 分布非平稳**——step 30 和 step 130 看到的"高 entropy fail"意义完全不同。

### 3.3 Per-task entropy 动态（dual-token 独有）

| task | SR_end5 | ent_end5 | s_θ_end5 | ρ(ent, SR) |
|---|:-:|:-:|:-:|:-:|
| heat | **1.000** | **0.110** | 1.702 | −0.806 |
| examine | 0.900 | 0.499 | 2.013 | −0.792 |
| pick_place | 0.837 | 0.646 | 2.021 | −0.681 |
| cool | 0.811 | 0.632 | 1.623 | −0.727 |
| clean | 0.796 | 0.773 | 1.717 | −0.489 |
| **other** | **0.679** | **0.891** | 1.962 | **+0.124** |

- **heat 是唯一 solve 的任务**（entropy 崩到 0.11），也是原 §3.1 "OCAR 帮助 heat/examine" 的 entropy 面证据。
- **other 反向**（SR 与 entropy 同涨）——"per-task 方向异质"在 entropy 面的第一次动态观察：policy 看到 "other" 类 prompt 时 entropy 反而上升（异质集合的两端放大）。

### 3.4 Entropy vs Surprise 正交性与 detrended 关系

**Step-level**（原 §9.6 #4）：corr(ent, ΔS) = −0.12；corr(ent, wm_s_B) ≈ 0。

**Training-level Spearman after detrend（linear _step removed）**：

| run | ρdet(ent, SR) | ρdet(s_θ, SR) | ρdet(ΔS, SR) | ρdet(ent, s_θ) | ρdet(ent, ΔS) |
|---|:-:|:-:|:-:|:-:|:-:|
| ALFW observe (n=30) | **−0.539** (p=.002) | −0.171 | −0.175 | +0.106 | −0.223 |
| Webshop observe (n=31) | −0.192 | −0.178 | −0.073 | +0.206 | +0.231 |
| ALFW dual (n=150) | **−0.668** (p<1e-19) | −0.221 | **−0.319** | +0.215 | +0.199 |

**去时间后，entropy → SR 的负相关两 ALFW run 上都显著，ΔS 则不稳**——entropy 是更"outcome-proximal"的信号。§2.3 的 ΔS traj-AUC 0.82 在 *training-level* 上不直接成立；entropy 在 training-level 上反而比 ΔS 强。

---

## 4. 方法含义：Dual-Token 叙事的三种解释 (data: 2026-04-19)

### 4.1 Train/val 反转的实证

ALFW 同 step 网格（step 5–150, n=30 对照点，dual-token `l49ikuco` vs GRPO+observe `lmlyvpa6`）：

| metric | GRPO+observe end3 | dual-token end3 | Δ(dt−obs) |
|---|:-:|:-:|:-:|
| **train SR** | **0.836** | 0.714 | **−0.122** |
| **val SR** | 0.807 | **0.812** | +0.005 |
| val SR 末点 | 0.797 | **0.883** | **+0.086** |
| step_entropy | 0.882 | 0.758 | −0.124 |
| s_θ | 2.193 | 1.942 | −0.251 |
| wm_s | 3.822 | 3.100 | −0.722 |
| ΔS | −0.005 | −0.356 | −0.351 |

**结构化观察**：wm loss 把 s_θ / wm_s / ΔS / step_entropy 四个量同时压低 → train SR 掉 → val SR 没掉甚至涨 → 净效应更像 regularizer。

### 4.2 三种观测等价的 framing

| framing | 预测 | 当前证据 |
|---|---|---|
| (a) Observation-grounded LM（paper_v6 原叙事） | wm loss 防 policy drift → val 稳 | 与数据相容 |
| (b) Entropy regularizer（§4.1 新 framing） | wm loss 压低 entropy → 减少 over-exploit → val 稳 | 与数据相容 |
| (c) Surprise-guided credit（原始直觉） | wm loss 放大 $h(\theta,o_t)$ 的 3.9% → 改善 credit | 与 §2.3 一致但量级小 |

**三者当前观测等价，需实验 §1.3 A 分开**：GRPO + 纯 entropy bonus 对照 dual-token。若 entropy bonus 复现 val 提升，(b) 成立；若失败，(a) 获支持。

### 4.3 对 paper_v6 叙事的修正

- **合理性来自 "observation-grounded language modeling"**——这是最诚实的 framing，且与证据方向一致（raw $s_\theta$ 不必有 outcome 信号，LM 监督本身防 drift 是独立动机）
- **Dual-token loss 恰好放大 $h(\theta,o_t)$ 那 3.9%**——**事后解释**，不是 design motivation
- **Ref model 必须 frozen base**（§2.5）——dual-token KL/ref 默认如此，一致
- **Base 容量/代际/SFT 状态不会"自动"让 surprise 成为 outcome 信号**——这是训练动态独有现象

---

## 5. 跨环境稳健性 (data: 2026-04-19)

> **⚠️ 身份澄清**：`grpo_observe_webshop_20260418_070828/42rxhh6f` 是**纯 GRPO + observe 指标仪表盘**（不参与梯度），**不是 dual-token 方法的跨域验证**。Dual-token 在 webshop 上的方法实验**尚未进行**（§1.3 实验 C）。

### 5.1 Webshop 训练轨迹概览（n=31，每 20 step）

| stage | step | s_θ | ΔS | ΔS_std | SR | val_SR |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| early | 20–200 | 2.19 | −0.164 | 0.198 | 0.33 | 0.39 |
| mid | 220–420 | 2.15 | −0.130 | 0.211 | 0.59 | 0.63 |
| late | 440–640 | 2.24 | −0.132 | 0.205 | 0.61 | 0.68 |

### 5.2 稳健发现（唯一可 paper 化的跨域内容）

| 结论 | ALFWorld | Webshop | 与 §2 关系 |
|---|:-:|:-:|---|
| ✅ Raw $s_\theta$ 与 SR 无/弱相关 | +0.32 (p=.08) | −0.06 (p=.76) | 验证 §2.3 step 不编码 outcome |
| ✅ Obs NLL U 型 | ✓ | ✓ | 均源于训练 uniform drift $g(\theta)$ |
| ✅ `fail_ent − succ_ent > 0` | — | t=6.95 p=1e-7 | §3.1 跨域 |

**所有其他"跨域模式"都是环境特异 artefact**（见原 §12.3 的 Spearman 表，已下沉至 §C.4）。

### 5.3 方法适用性的可观测前提

1. **ΔS 累积性**：若训练中 $|E[\Delta S]|$ 不随 step 增长（webshop 所示），θ 相对 ref 没真正偏离——obs NLL aux loss 的"放大 $h(\theta,o_t)$"机制无东西可放大，收益小。
2. **Obs NLL 量级差**：webshop $s_\theta$ 起点高（2.4）且 U 型振幅大（0.76 vs ALFW 0.50），obs 文本本身更难——aux loss 的 gradient 可能主导训练，需重新调权重。

**下一步**：在 webshop 跑 dual-token（§1.3 实验 C）。若超过 GRPO baseline，跨域叙事成立；若不超过，写成 "方法适用性由 ΔS drift rate 决定" 的**限定性证据**。

---

## 6. 证据分级与合理用法

### 6.1 按证据分级

✅ **强证据（traj 级 + fixed base ref）**
- **Trajectory-level reweighting**：把 $\Delta S_\text{traj-mean}$ 作为 outcome advantage 的 per-traj 权重（GRPO-style）
- **Dual-token NLL aux loss**：选择性强化 $h(\theta, o_t)$，与证据方向一致（但见 §4.2 framing 模糊）
- **Entropy succ/fail discrimination**：P(f>s)=1.0 的 within-batch outcome proxy

⚠️ **中等证据**
- Per-step reweighting（OCAR v3: z-norm + softmax）：仅限 heat/examine 等高交互任务族，且配合 fixed base ref（见 §A.3）
- 早期失败预测器（前 5 步 AUC 0.73）→ early-stop / curriculum

❌ **不支持（有反证）**
- Raw $s_\theta$ 直接使用（§2.3）
- Moving ref 的 $\Delta S$（§2.5）
- 跨模型 Δ（§2.5）
- Step 级通用 credit assignment（§2.3）
- Isolated/consecutive 规则做 stuck 检测（§A.5）
- 声称反映 "world model quality"（U 型 + 无独立 WM）
- 声称 "base model scale 提升信号有效性"（§2.3 AUC 全部 ≈ 0.5）

### 6.2 Surprise 作 intrinsic reward 的可行性

> 问题：$\Delta S$（或其密集程度）的连续出现次数可否作 intrinsic reward？

**现有证据**（§4.5–4.6 原测试）：

| 量 | 预测 useful-exploration AUC | 预测 stuck AUC |
|---|:-:|:-:|
| delta_s 单步 | 0.58 | 0.48 |
| ema_delta_s (λ=0.5) | **0.59** | 0.48 |
| win3_delta_s | **0.59** | 0.48 |
| sustained_pos_delta | 0.55 | 0.49 |
| d_delta_s | 0.49 | 0.52 |

**5 条风险**：
1. "Useful" label 是本地启发式（"3 新 obs token in next 3 steps"），AUC 0.59 解释极小方差，和最终 success 关系未证实
2. **Reward hacking 直接可达**：agent 做能翻新 obs 文本但无用的动作（反复 examine、open/close receptacle）即可刷高密集度。§4.7 定性案例：fail traj `f4c9dacf` 在 countertop 间来回 → consec ΔS 高但任务失败
3. **非平稳**：ΔS 幅度随训练对数增长，密集度阈值漂移
4. **Succ/Fail 同形**（§2.4, §3.2）：密集度在成功与失败轨迹分布几乎一致
5. **Residual hardness 非平稳**（§3.2 新增）：label 分布随 SR 上升漂移，同样的 "high-ent/high-S fail step" 在不同训练阶段意义不同

**与已知 intrinsic reward 的关系**：ΔS 密集度 ≈ prediction-error based curiosity（ICM/RND 的文本域等价）。ICM 结论完全适用——reward hacking + 依赖环境结构。ALFWorld 充满"翻新文本但无用"动作，高风险。

**推荐做法**：继续走 loss / advantage 路线（OCAR v3 / dual-token），不做 intrinsic reward。如果要做，必须先过 A/B/d 三种消融（见 §A.4）。

---

## 7. 实验账本

### 7.1 Extra-seed eval（24 runs, 2026-04-19）

**Phase A（t=0.7，3 new seeds）**

| 方法 / step | 42 | 2024 | 7 | mean ± std |
|---|:-:|:-:|:-:|:-:|
| observe step120 | 0.797 | 0.852 | 0.789 | **0.813 ± 0.028** |
| observe step140 | 0.820 | 0.852 | 0.773 | **0.815 ± 0.033** |
| dual step140 | 0.820 | 0.828 | 0.781 | **0.810 ± 0.021** |
| dual step150 | 0.820 | 0.812 | 0.789 | **0.807 ± 0.016** |

**Phase B（t=0.4，GiGPO paper-config, 6 seeds）**

| 方法 / step | 123 | 456 | 789 | 42 | 2024 | 7 | **mean ± std** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| observe step120 | 0.867 | 0.781 | 0.734 | 0.766 | 0.836 | 0.758 | **0.790 ± 0.048** |
| dual step150 | 0.875 | 0.789 | 0.750 | 0.781 | 0.844 | 0.805 | **0.807 ± 0.044** |

| 对比 | mean diff | 显著性 |
|---|:-:|:-:|
| dual vs observe @ t=0.4 (6 seeds) | +1.7pp | Welch t≈0.6, p≈0.55 **未显著** |
| observe t=0.4 vs t=0.7 | −2.3pp（低温反降） | — |
| dual t=0.4 vs t=0.7 | 0.0pp | — |

**与 GiGPO 论文对标（t=0.4）**：

| 方法 | GiGPO 论文 | 我们 | gap |
|---|:-:|:-:|:-:|
| GRPO 类 baseline | 77.6% | observe **79.0%** | +1.4pp（复现成功） |
| GiGPO | 90.8% | dual **80.7%** | **−10.1pp** |

**→ dual-token 在 6-seed paper-config 未显著超过 observe 基线，且与 GiGPO 主方法还差 10pp。**

### 7.2 Canary + Scale scan 主要数据

**Canary（5 模型，原 §11.2）**：nonsense − orig gap 在 7B-Instruct 最大（+0.257），在 14B 其次（+0.285），0.5B 最小（+0.181）——结构敏感性随 scale 单调提升。

**Scale scan（4 模型 × 12 traj，原 §11.3）**：

| 模型 | obs_nll_last | wm_A | wm_B | **wm_gap** | succ-fail gap |
|---|:-:|:-:|:-:|:-:|:-:|
| Qwen2.5-0.5B | 2.013 | 3.419 | 3.838 | 0.420 | 0.077 |
| **Qwen2.5-7B-Instruct** | **1.793** | 4.185 | 5.093 | **0.908** | 0.119 |
| Qwen3-8B | 2.584 | 4.591 | 5.070 | 0.479 | 0.222 |
| Qwen3-14B | 2.361 | 4.839 | 4.893 | **0.055** | 0.240 |

**⚠️ 代际×scale×SFT 混淆**——需 §1.3 实验 E 的 14 模型纯 Base scan 才能解除。

### 7.3 A1/A2 decoupling（observe `lmlyvpa6`, 原 §11.4）

- **CCF argmax**：lag=−24, ccf=+0.573 → obs 信号滞后 SR ~24 step
- **Stage-wise SR ~ obs_s_θ**：early R²=0.35, mid/late R²≈0.05 → 相关性集中在训练早期
- Spearman(delta_s_mean, SR) = +0.496（global levels）

---

## 8. 跨域通用发现汇总 (data: 2026-04-20)

> 目的：将 ALFWorld 和 WebShop 两个环境上的通用发现整理为可直接写入 paper 的 claim，明确区分"跨域成立"与"环境特异"。

### 8.1 通用发现 1：Obs NLL U 型趋势（World Model Degradation）

**现象**：在纯 GRPO 训练中，actor 对 observation token 的 NLL（$s_\theta$）呈"先降后升"的 U 形。

| 环境 | 训练区间 | $s_\theta$ 变化 | U 形拐点 |
|---|---|---|---|
| ALFWorld | step 1–150 | 2.4 → 1.8 → 2.3 | ~step 80 |
| WebShop | step 20–640 | 2.4 → 1.9 → 2.5 | ~step 300 |

**Per-token ΔS_mean 控制了轨迹长度混淆**：ALFWorld per-token ΔS 从 −0.38 升至 +0.05（partial corr = +0.546 controlling for SR）。Dual-token run 在相同 SR 增长下 ΔS 保持 −0.35——说明 U 形不是 SR/长度的 artifact。

**机制解释**：GRPO 仅在 action token 上提供梯度信号，obs token 的 LM 能力只通过 shared weights 间接受影响。早期 action loss 下降带动整体 NLL 降（$g(\theta)$ 下行），后期 action-specific 优化破坏了对 obs token 的泛化（$g(\theta)$ 上行）= **world model degradation**。

### 8.2 通用发现 2：Action Entropy 与 Obs Surprise 正交

**Step 级正交性**（不同 run、不同环境一致）：

| 数据 | corr(entropy, ΔS) | corr(entropy, wm_s_B) |
|---|---|---|
| ALFWorld dual-token (n=150 steps) | −0.12 | ≈ 0 |
| WebShop observe (n=31 steps) | +0.23 | ≈ 0 |

**两个信号捕捉不同维度**：

| 维度 | Action Entropy ($H_t$) | Obs Surprise ($\Delta S_t$) |
|---|---|---|
| **来源** | Policy head softmax 分布 | Obs token NLL vs frozen ref |
| **衡量** | Agent 对当前 action 的确信度 | 环境转移与 agent 世界模型的偏差 |
| **Succ/Fail 区分** | Step 级即成立（P(f>s)=1.0） | 仅 traj 级成立（AUC 0.82+） |
| **信号可用性** | 前 5 log step 就显著 | 需 ~50 training step 累积 |
| **跨域稳健性** | 两域均成立 | ALFWorld 强，WebShop 弱 |
| **与 outcome 关系** | 直接：高 entropy → 低 success | 间接：通过累积 drift |

**含义**：两信号信息互补、计算路径不共享、方向不冗余。这为联合使用两个信号做 credit assignment 提供了实证基础。

### 8.3 通用发现 3：Residual Hardness 非平稳性

两个环境中都观察到：随训练推进，fail 轨迹从"随机失败"演变为"hard residual"。

- ALFWorld：ρ(SR, ent_gap) = +0.655（p<1e-19），Q4 fail entropy 是 Q1 的 8.6 倍
- WebShop：early fail entropy 0.93 vs late fail entropy 1.12（+20%）

**对方法设计的约束**：任何用 "high entropy = bad" 做 credit 的方法必须处理 **label 分布随训练漂移**——同样的 entropy 阈值在 step 30 和 step 130 意义不同。z-normalization per batch 是最小必要条件。

### 8.4 环境特异 vs 通用信号总结

| 信号/现象 | ALFWorld | WebShop | 通用？ |
|---|---|---|---|
| ✅ Obs NLL U 型 | ✓ | ✓ | **是** |
| ✅ Entropy succ/fail gap > 0 | t=17.24 | t=6.95 | **是** |
| ✅ Entropy × ΔS 正交 | −0.12 | +0.23 | **是**（均弱相关） |
| ✅ Residual hardness | ρ=+0.655 | +20% late | **是** |
| ✅ Raw $s_\theta$ 不编码 outcome | AUC≈0.50 | ρ=−0.06 | **是** |
| ❌ ΔS 累积 drift | 强（+0.43） | 弱（≈0） | 否 |
| ❌ consec_s 方向 | +0.691 | −0.664 | 否（反向） |
| ❌ wm_s 预测 success | ρ=−0.04 | ρ=−0.43 | 否 |

### 8.5 对新方法设计的启示

基于上述跨域通用发现，一个有效的新方法需要：

1. **利用 entropy 和 ΔS 的正交互补性**——单独用任一信号都有已有工作（EPO/CARL/UEC-RL 用 entropy，OCAR 用 ΔS），联合使用是新贡献
2. **处理 residual hardness 非平稳性**——per-batch z-normalization 而非固定阈值
3. **不依赖 ΔS 累积性**——WebShop 上 ΔS drift 弱，方法不能假设 ΔS 随训练单调变化
4. **不使用 step 级 raw signal 做 credit**——step 级 AUC ≈ 0.5，必须聚合或与 entropy 联合

---

## 9. Hidden State Information Gain 预验证 (data: 2026-04-20)

### 9.1 动机与定义

**目标**：评估表征空间信号是否比 token 级 ΔS 更好的跨域通用 step-level reward。

**定义**：对于 step t (t>0)，构建 context 到 obs_t 之前和之后：
- $h_\text{pre}$ = 模型在 obs_t 之前最后一个 token 的 hidden state
- $h_\text{post}$ = 模型在 obs_t 最后一个 token 的 hidden state
- info_gain = $\|h_\text{post} - h_\text{pre}\|_2$

直觉：成功步骤的 observation 应包含更多"有用信息"，导致 hidden state 变化更大。

**提取层**：Layer -1（最后层）、-16（中间层）、-28（早期层）。

### 9.2 ALFWorld 结果（31 traj: 11 fail + 20 succ）

| 模型 | succ IG (L-1) | fail IG (L-1) | gap | corr(IG, success) | traj p |
|---|:-:|:-:|:-:|:-:|:-:|
| **Step 150 ckpt** | 278.2 | 101.6 | +176.6 | 0.538 | <0.0001 |
| **Base (Qwen2.5-7B-Inst)** | 286.4 | 112.9 | +173.5 | 0.500 | <0.0001 |

- **两种模型结果几乎相同**——info gain 区分力不依赖训练状态
- 与其他信号正交：corr(IG, ΔS) ≈ 0.21–0.23，corr(IG, entropy) ≈ -0.23 ~ -0.25

### 9.3 WebShop 结果（base model, step 240 & 640 轨迹）

| 数据 | succ IG (L-1) | fail IG (L-1) | gap | corr(IG, success) | traj p |
|---|:-:|:-:|:-:|:-:|:-:|
| **step 240** | 184.9 | 248.7 | **-63.8 (反向!)** | -0.305 | 0.020 |
| **step 640** | 238.2 | 224.7 | +13.5 | +0.077 | 0.946 |

**⚠️ 方向不一致且信号弱。**

### 9.4 根因分析：截断伪影 + 环境结构差异

**截断 bug**：脚本 `max_length=4096` 导致长 context 被截断后 `pre_ids == post_ids`，产生 info_gain = 0.0。

- SUCC 轨迹受影响更严重：33/110 step 为 0.0（30%），因为成功轨迹更长（多步浏览产品页）
- FAIL 仅 3/106 step 为 0.0（3%）
- 去除 0.0 后：SUCC ≈ 264 vs FAIL ≈ 256，gap 仅 +8（~3%），方向正确但极弱

**环境结构差异**：
- ALFWorld obs 短（几十 tokens），全部 context 能放入 4096 → 无截断 → 信号干净
- WebShop obs 长（2000+ tokens/step），后期 step 必然被截断
- 即使修复截断，WebShop succ/fail IG gap 也只有 ~3%，远不如 ALFWorld 的 ~160%

### 9.5 结论

| 维度 | ALFWorld | WebShop |
|---|---|---|
| step-level corr(success) | **0.50–0.54** | -0.31 ~ +0.08 |
| 跨模型稳定 | ✅ base ≈ trained | — |
| 截断鲁棒 | ✅ obs 短 | ❌ obs 长导致伪影 |
| 作为通用 reward | ❌ **不通用** | ❌ |

**Hidden state info gain 不具备跨域通用性，放弃作为 intrinsic reward 候选。** ALFWorld 上的强信号来源于环境 obs 短且结构简单（hidden state 变化直接反映 task progress），WebShop 上 obs 长且异构（search page vs product page）使 info gain 退化为噪声。

脚本：`ocar/analysis/hidden_state_info_gain.py`，`ocar/analysis/hidden_state_ig_alfworld_base.py`
数据：`ocar/analysis_results/hidden_state/{alfworld_step150,alfworld_base_model,webshop_step240,webshop_step640}.json`

---

## Appendix A: 方法演化史（OCAR v1/v2/v3 与其他撤回路线）

> 主线 §2-4 只讲最终结论；以下收纳历史实验的精简要点。完整原文见 `ocar/EXPERIMENT_LOG.md.orig`。

### A.1 OCAR v1 崩溃诊断（2026-04 初）

- **时间线**：v1 per-step softmax 在 step 100–125 出现训练崩溃（SR 骤降、loss 发散）
- **原诊断**：clamp 缺失导致 per-step weight 爆炸
- **✅ 更新后根因（§2.4）**：selective fail reconstruction——训练在 step 100–125 对 fail 轨迹 NLL 降得特别多，fail rank 大幅重构（0.69 vs succ 0.90），per-step softmax 权重被这批大位移 fail step 主导

### A.2 OCAR v2 Clamp 修复的局限

Clamp 治标不治本：限制 per-step weight 上限避免爆炸，但不解决"fail 被 selectively 重构"这个更深的非平稳性。v2 稳定但性能提升有限。

### A.3 OCAR v3 Z-score 归一化提案（未跑）

- 每条轨迹内 z-score ΔS 后再 softmax
- 消除 $f_\text{text}$ 和 $g(\theta)$ 的基线漂移
- ε 推荐值 0.1
- **仅限 heat/examine 等高交互任务族，且配合 fixed base ref**（§6.1 中等证据条目）
- 代码改动 3 行：`core_ocar.py` 中 `compute_ocar_outcome_advantage` 的 Step 3

### A.4 Surprise-as-reward 探索（早期放弃）

- OCAR v1/v2/v3 都不把 surprise 直接作 reward
- §6.2 给出 5 条风险分析
- 若要做，最低要求：potential-based shaping + episode 级 z-norm + outcome 非零时衰减到 0

### A.5 Stride / Stuck 规则检测（证伪）

- 原假设：ΔS consecutive 高点 → stuck 行为
- 实验结果（原 §4.6）：consecutive 高点 useful% 反而更高（45.8 vs 36.2），stuck 无差异
- Isolated vs consecutive 用 step-level 标签后，方向反转
- **结论**：规则-based stuck 检测不工作，放弃

### A.6 World Model Dynamics 初步（2026-04-14）

- 实验：比较 $P_\theta(o|\text{ctx},a)$ 与 $P_\theta(o|a)$ 的差值 `wm_gap`
- 结果：训练中 wm_gap U 型（早期升，step 100 后降）
- 含义（已修正）：这不是"world model quality"升降，而是训练对 obs token 的 uniform drift（$g(\theta)$）占据主导

---

## Appendix B: 跨模型原始数据核查 (原 §C, 数据：2026-04-19 复跑)

> §2 的第一性原理结论都从下表导出。脚本：[ocar/analysis/cross_scale_surprise.py](ocar/analysis/cross_scale_surprise.py), [ocar/analysis/training_dynamics_decomp.py](ocar/analysis/training_dynamics_decomp.py)。

### B.1 Cross-model inference (4 异构 Qwen × 93 step)

**⚠️ 变量混淆**：4 模型同时变 scale、代际、SFT 状态，**不是**纯 scale scan；Q5–Q7 跨族趋势不可单独归因于 scale。

详细 Q1–Q7 数据表见 `EXPERIMENT_LOG.md.orig` §C.1（行 983–1062）。

### B.2 Training dynamics decomposition (6 ckpt × 20 traj)

详细 Q1–Q5 数据表见 `EXPERIMENT_LOG.md.orig` §C.2（行 1063–1131）。

Rank stability（§2.4 引用源）：

| ckpt vs base | all | succ-only | fail-only |
|---|:-:|:-:|:-:|
| step50 | 0.926 | 0.900 | 0.827 |
| step75 | 0.905 | 0.879 | 0.806 |
| **step100** | 0.854 | **0.903** | **0.692** |
| **step125** | 0.867 | **0.903** | **0.680** |
| step150 | 0.899 | 0.900 | 0.768 |

### B.3 数据完整性核查

- **C.1**：4 模型评分的 traj_ids = `[0..11]`、per-traj step lens = `[8,8,6,7,8×9]`、success flags 完全一致 → 93 step / 45 succ / 48 fail
- **C.2**：6 ckpt 的 traj_id 集合完全一致，同 20 traj / 10 succ + 10 fail
- 两脚本输出与本附录表格**逐位复现**

### B.4 小字注

C.2 n=20 使 AUC 0.82+ 的 95% CI 较宽；若论文需硬性 CI，补一次 1000-bootstrap。C.1 step-level Mann-Whitney 以 traj outcome 作 step-label，引入 step 间相关性 → p 值偏乐观；但 4 AUC 全在 0.483–0.506，定性结论不翻转。

### B.5 Webshop 描述性 Spearman（n=31，**不作信号 claim**）

| signal | Webshop vs SR | ALFWorld vs SR | 评语 |
|---|:-:|:-:|---|
| raw $s_\theta$ | −0.057 (p=.76) | +0.320 (p=.08) | 两侧都弱，与 §2.3 一致 |
| $\Delta S$ | +0.139 (p=.46) | +0.722 (p<.0001) | ALFW 强，webshop 无 |
| consec_s | **−0.664** (p<.0001) | **+0.691** (p<.0001) | **方向相反**，环境特异 |
| wm_s_B | −0.534 (p=.002) | −0.036 (p=.85) | 环境特异 |
| wm_s | −0.431 (p=.015) | +0.077 (p=.69) | 环境特异 |
| step_entropy | +0.285 (p=.12) | +0.118 (p=.53) | 两侧都弱 |

**3 个重要警告**：(1) 时间耦合主导——Spearman 捕捉两条曲线随训练同向还是反向，不是 within-batch 预测力；(2) `consec_s` 两环境符号相反是 obs 文本结构差异的产物，非"consec_s 能预测 stuck"；(3) n=31 CI 约 ±0.3，不作 paper 级 claim。

---

## Appendix C: 文件与脚本

### C.1 关键文件路径

| 用途 | 路径 |
|---|---|
| 第一性原理 / 跨模型扫描 | [ocar/analysis/cross_scale_surprise.py](ocar/analysis/cross_scale_surprise.py) |
| 训练动态分解 | [ocar/analysis/training_dynamics_decomp.py](ocar/analysis/training_dynamics_decomp.py) |
| Entropy × Surprise 联合分析 | [ocar/analysis/entropy_surprise_joint.py](ocar/analysis/entropy_surprise_joint.py) |
| Webshop surprise | [ocar/analysis/webshop_surprise.py](ocar/analysis/webshop_surprise.py) |
| Scale scan 数据 | `ocar/analysis_results/scale_scan/*.json` |
| WM degradation 数据 | `ocar/wm_degradation_results.json` |
| ΔS variance 数据 | `ocar/delta_s_variance_analysis.json` |
| Webshop wandb history | `ocar/analysis_results/webshop/history_full.csv` |
| ALFW observe wandb history | `ocar/analysis_results/webshop/alfworld_observe_history.csv` |
| Dual-token wandb history | `ocar/analysis_results/wandb_dualtoken_l49ikuco_full.csv` |
| Entropy-surprise 报告 | `ocar/analysis_results/entropy_surprise/report.md` |
| §10 obs 类型信号分解 | `ocar/analysis/obs_type_traj_avg.py` |
| §10 跨数据集信号稳定性 | `ocar/analysis/cross_dataset_signal.py` |
| §10 s_theta 独立信号分析 | `ocar/analysis/s_theta_signal.py` |
| §10 z-score / |Δs| 验证 | `ocar/analysis/zscore_delta_s.py`, `ocar/analysis/abs_delta_s.py` |
| §10 |Δs| 信用分配 case study | `ocar/analysis/credit_assignment.py` |
| §10 头脑风暴信号对比 | `ocar/analysis/brainstorm_signals.py`, `ocar/analysis/brainstorm_signals_v2.py` |
| §10 信号相关性矩阵 | `ocar/analysis/signal_correlation.py` |

### C.2 分析脚本快速使用

```bash
# 环境
source /local_nvme/guanyiming/env/verl-agent-06x-py312/bin/activate

# 第一性原理复现
python ocar/analysis/cross_scale_surprise.py
python ocar/analysis/training_dynamics_decomp.py

# Entropy 联合分析（§3 + §4.1）
python ocar/analysis/entropy_surprise_joint.py

# Webshop 描述性分析（§5）
python ocar/analysis/webshop_surprise.py
```

---

## 10. Step-Level 信号深入分析：Observation 类型分解与因果方向 (data: 2026-04-21)

> 目的：系统评估所有 policy forward pass 中可获取的信号作为 step-level reward 的可行性。通过 observation 类型分解、跨环境相关性、门控信用分配等多角度分析，得出明确结论。

### 10.1 Observation 类型分类与信号分解

将每步 observation 分为四类：
- **state_change**：环境状态改变（pick up, put, open, close, heat, cool, clean, slice, use）
- **new_location**：到达新位置/页面（首次出现的 obs）
- **revisit**：重复访问已见过的 obs
- **nothing_happens**：无效动作（nothing happens, can't, don't see）

**ALFWorld 7B（step 150, SR=91%）按 obs 类型的信号差异**（轨迹级平均，消除长度偏差）：

| obs_type | frac(S) | frac(F) | entropy(S) | entropy(F) | Δs(S) | Δs(F) | wm_s(S) | wm_s(F) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| state_change | 33.3% | 5.2% | 0.023 | 0.816 | +0.12 | −0.14 | 3.15 | 4.42 |
| new_location | 48.3% | 47.3% | 0.031 | 2.40 | +0.12 | −0.02 | 3.66 | 5.00 |
| revisit | 3.6% | 9.0% | 0.040 | 2.52 | +0.13 | −0.09 | 3.29 | 5.53 |
| nothing_happens | 14.8% | 38.5% | 0.033 | 2.53 | +0.15 | +0.08 | 3.81 | 5.75 |

**关键观察**：
1. **Entropy 在所有 obs 类型上都有极强区分力**：成功轨迹 entropy 全部 <0.04，失败轨迹 0.8–2.5
2. **obs 类型分布本身就是强信号**：成功轨迹 state_change 占 33% vs 失败仅 5%；nothing_happens 失败占 39% vs 成功 15%
3. **Δs 区分力弱**：成功/失败轨迹的 Δs 差异远小于 entropy

**ALFWorld 1.5B（step 60, SR=23%）信号差异微弱**：

| obs_type | entropy(S) | entropy(F) | gap |
|---|:-:|:-:|:-:|
| state_change | 0.70 | 0.77 | 0.07 |
| new_location | 0.72 | 0.82 | 0.10 |
| nothing_happens | 0.74 | 0.80 | 0.06 |

1.5B 模型 entropy gap 仅 0.06–0.10，远弱于 7B 的 0.8–2.5。**信号强度与模型能力正相关**。

脚本：`ocar/analysis/obs_type_traj_avg.py`

### 10.2 跨数据集信号稳定性

**轨迹级相关性 r(signal, success) 跨训练步**：

| 数据集 | step | r(Δs) | r(entropy) | r(wm_s) | r(wm_gap) |
|---|:-:|:-:|:-:|:-:|:-:|
| ALFWorld 1.5B | 20 | +0.08 | −0.23 | −0.20 | −0.07 |
| ALFWorld 1.5B | 40 | +0.32 | −0.46 | −0.30 | −0.43 |
| ALFWorld 1.5B | 60 | +0.06 | −0.49 | −0.26 | −0.22 |
| **ALFWorld 7B** | **150** | **+0.75** | **−1.00** | **−0.64** | **−0.46** |
| WebShop | 80 | −0.47 | −0.79 | −0.28 | −0.47 |
| WebShop | 160 | +0.10 | −0.65 | −0.20 | −0.55 |
| WebShop | 240 | +0.65 | −0.73 | −0.10 | −0.32 |
| WebShop | 320 | −0.49 | −0.38 | −0.23 | −0.26 |
| WebShop | 400 | −0.47 | −0.02 | +0.21 | +0.01 |
| WebShop | 480 | +0.40 | −0.12 | −0.08 | +0.33 |
| WebShop | 560 | −0.01 | −0.49 | +0.17 | −0.29 |
| WebShop | 640 | −0.53 | +0.19 | +0.52 | +0.60 |

**结论**：
- **Δs 方向剧烈翻转**：WebShop 上 r 从 −0.53 到 +0.65，完全不可靠
- **Entropy 前中期最稳**（r 一致为负），但后期 WebShop 也崩溃（step 400: −0.02, step 640: +0.19）
- **wm_s 和 wm_gap 同样不稳定**，且 wm_gap 需要额外 forward pass，非免费信号
- **所有信号在训练后期 WebShop 上都失效**

脚本：`ocar/analysis/cross_dataset_signal.py`, `ocar/analysis/s_theta_signal.py`

### 10.3 Δs 的因果方向问题 ⭐

**核心论断**：Δs 作为 step-level reward 存在根本性的因果方向错误。

$$\Delta s = s_\theta - s_{ref} = (-\log P_\theta) - (-\log P_{ref})$$

Δs 记录的是**训练已经发生的结果**，不是动作质量的因果信号：
- 成功轨迹中的 token 被 GRPO 强化 → $P_\theta$ 上升 → $s_\theta$ 下降 → Δs 变负
- 失败轨迹中的 token 被 GRPO 抑制 → $P_\theta$ 下降 → $s_\theta$ 上升 → Δs 变正

因此 Δs 与 success 的相关性（7B r=+0.75）**本质是在读回 GRPO 已做的 trajectory-level 更新**。用 Δs 做 step-level reward 来指导训练 = "用训练的结果来指导训练本身" = **循环论证**。

**WebShop 上方向翻转的解释**：随训练推进，policy 偏移远离 ref，新的成功/失败模式出现，Δs 反映的是不同阶段的更新历史，导致方向不一致。

**与 §2 的关系**：§2.3 已证明 step 级 AUC ≈ 0.5，§2.5 已证明 ref 选择强约束。本节进一步揭示了根本原因——**Δs 本质上是 KL divergence 的局部版本，测量的是训练偏移而非动作质量**。

### 10.4 s_theta_mean（obs NLL）同样不可靠

$s_\theta^{obs}$ 虽看似有因果方向（"模型对环境的预测能力"），但实际也被训练间接污染：

| 数据集 | step | r(s_theta, success) |
|---|:-:|:-:|
| ALFWorld 1.5B | 20 | +0.09 |
| ALFWorld 1.5B | 60 | +0.21 |
| ALFWorld 7B | 150 | −0.50 |
| WebShop | 80 | −0.22 |
| WebShop | 240 | −0.66 |
| WebShop | 400 | +0.13 |
| WebShop | 640 | +0.76 |

**方向同样剧烈翻转**（WebShop: −0.66 到 +0.76）。原因：GRPO 强化 action tokens → 改变 context 中 action 分布 → 间接影响 obs tokens 的条件概率。$s_\theta$ 是训练的**间接结果**。

这解释了 §2.3 "raw $s_\theta$ 不编码 outcome"的深层原因——不是信号太弱，而是信号方向随训练漂移。

脚本：`ocar/analysis/s_theta_signal.py`

### 10.5 Z-score 和 |Δs| 均无法修复方向不稳定

**Z-score normalization**：线性变换不改变 Pearson 相关系数，r_zscore ≡ r_raw。Z-score 只解决 scale 问题，不解决方向翻转。

**|Δs|（绝对偏离度）**：

| 数据集 | step | |Δs| succ | |Δs| fail | 方向 |
|---|:-:|:-:|:-:|:-:|
| ALFWorld 1.5B | 20 | 0.064 | 0.070 | S < F |
| ALFWorld 1.5B | 60 | 0.170 | 0.177 | S < F |
| ALFWorld 7B | 150 | 0.135 | 0.159 | S < F |
| WebShop | 80 | 0.209 | 0.173 | S > F |
| WebShop | 240 | 0.171 | 0.196 | S < F |
| WebShop | 640 | 0.217 | 0.195 | S > F |

|Δs| 方向在环境间不一致（ALFWorld 一致 S<F，WebShop 混合），不可作通用信号。

脚本：`ocar/analysis/zscore_delta_s.py`, `ocar/analysis/abs_delta_s.py`

### 10.6 |Δs| 信用分配在 WebShop 上失败

用 weight_i = |Δs_i| / Σ|Δs_j| 做信用分配的 case 分析：

**ALFWorld 7B 成功轨迹**（8 步）：
- top-3 步骤获得 49% 权重，分别是关键动作（cool, go to, open）→ ✓ 合理

**WebShop 成功轨迹**（8 步）：
- search 步骤获得 **54–86%** 权重（|Δs| ≈ 0.75–0.82）
- click 步骤仅获得 2–13% 权重（|Δs| ≈ 0.01–0.13）
- **失败轨迹也是相同模式**：search 垄断权重，click 被边缘化

**根因**：|Δs| 本质是 token-level 生成难度。search query 是自由文本（高 |Δs|），click 是短选择（低 |Δs|）。这与动作质量无关——**token 生成难度 ≠ 步骤重要性**。

脚本：`ocar/analysis/credit_assignment.py`

### 10.7 Entropy 趋势（D2）：方向一致但信号弱

分析轨迹内 entropy 的变化趋势（前半 vs 后半，线性 slope）：

| 数据集 | step | slope(S) | slope(F) | 方向一致？ |
|---|:-:|:-:|:-:|:-:|
| ALFWorld 1.5B | 40 | −0.002 | +0.001 | ✓ S 下降 F 上升 |
| ALFWorld 7B | 150 | −0.001 | **+0.040** | ✓ **失败轨迹 entropy 急剧上升** |
| WebShop | 80 | −0.018 | +0.018 | ✓ |
| WebShop | 240 | +0.004 | +0.037 | ✓（F slope 更高） |
| WebShop | 400 | +0.019 | +0.046 | ✓（F slope 更高） |
| WebShop | 640 | +0.041 | +0.119 | ✓（F slope 更高） |

**方向始终一致**：失败轨迹的 entropy slope > 成功轨迹。但在 1.5B 和 WebShop 上差异较小，单独作为 step-level reward 不够强。

脚本：`ocar/analysis/brainstorm_signals.py`

### 10.8 Policy Forward Pass 信号因果性总结 ⭐

| 信号 | 定义 | 免费？ | 因果方向 | 跨域稳定 | 区分力 | 可用性 |
|---|---|:-:|---|---|---|---|
| **entropy** | $H(P_\theta(\cdot\|c_t))$ | ✓ | ✓ 模型当前不确定性 | 较稳定 | 最强 | ✓ |
| Δs | $s_\theta - s_{ref}$ | ✓ | ✗ 训练结果回读 | 不稳定 | 不可靠 | ✗ |
| s_theta | $-\log P_\theta(o_t\|c_{<t})$ | ✓ | ✗ 间接受训练污染 | 不稳定 | 不可靠 | ✗ |
| |Δs| | 偏离程度 | ✓ | ✗ 同上 | 不稳定 | 不可靠 | ✗ |
| wm_s | $-\log P_\theta(o_{t+1}\|o_t,a_t)$ | ✗ 额外 pass | ✓ 环境预测 | 不稳定 | 中等 | ✗ |
| wm_gap | wm_B − wm_s | ✗ 2 次额外 pass | ✓ 信息增益 | 不稳定 | 弱 | ✗ |
| entropy slope | 轨迹内 entropy 趋势 | ✓ | ✓ 收敛/发散 | **稳定** | 弱 | ⚠️ |

**核心结论**：在 policy forward pass 的免费信号中，**entropy 是唯一具备正确因果方向且跨域稳定的信号**。所有基于 NLL 差异的信号（Δs, s_theta, |Δs|）都受训练过程直接或间接污染，因果方向错误——它们记录的是"训练做了什么"而非"动作质量如何"。

### 10.9 对方法设计的更新约束

基于 §10 的分析，更新 §8.5 的约束：

1. **Δs 不可作 step-level reward 或信用分配权重**——因果方向错误（§10.3），且 token 生成难度 ≠ 步骤重要性（§10.6）
2. **Entropy 是唯一可用的免费 step-level 信号**，但已有大量工作（EPO, CARL, UEC-RL 等），单独使用缺乏新颖性
3. **Entropy slope（轨迹内趋势）方向稳定但信号弱**——可能需要与其他机制结合
4. **任何新方法需要引入新的信息源**——纯 policy forward pass 的免费信号已穷尽，要么引入额外计算（learned critic, self-evaluation），要么利用环境交互结构（轨迹拓扑、obs 转移模式）

---

**文档版本**：2026-04-21 添加 §10 step-level 信号深入分析。历史版本见 `ocar/EXPERIMENT_LOG.md.orig`。
