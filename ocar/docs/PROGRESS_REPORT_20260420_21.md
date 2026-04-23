# 研究进度汇报（2026-04-20 ~ 2026-04-21）

> 目的：汇总近两天在 ALFWorld 与 WebShop 上对 **policy forward pass 免费信号**（surprise / entropy / wm_s / wm_gap / hidden-state info gain / entropy slope 等）的 step-level 可用性评估。只列关键结果表。详见 [ocar/EXPERIMENT_LOG.md](ocar/EXPERIMENT_LOG.md) §8–§10。

---

## 1. 覆盖的数据源

| 数据集 | 模型 | 训练步点 | 用途 |
|---|---|---|---|
| ALFWorld | Qwen2.5-1.5B | step 20 / 40 / 60 | 小模型信号强度对照 |
| ALFWorld | Qwen2.5-7B-Inst (GRPO+observe) | step 150 | 主力分析 |
| WebShop | Qwen2.5-7B-Inst (GRPO+observe) | step 80 / 160 / 240 / 320 / 400 / 480 / 560 / 640 | 跨训练阶段稳定性 |

---

## 2. 轨迹级 r(signal, success)：跨数据集稳定性

脚本：[ocar/analysis/cross_dataset_signal.py](ocar/analysis/cross_dataset_signal.py), [ocar/analysis/s_theta_signal.py](ocar/analysis/s_theta_signal.py)

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

**要点**：
- **Δs / s_θ / wm_s / wm_gap 方向均在 WebShop 跨训练阶段翻转**（如 Δs: −0.53 ~ +0.65；s_θ: −0.66 ~ +0.76）。
- **Entropy** 前中期跨域一致为负，但 WebShop 后期（step 400 起）衰减甚至翻正（+0.19 @ 640）。
- **小模型（1.5B）信号整体弱于 7B**——信号强度与模型能力正相关。

---

## 3. ALFWorld 按 observation 类型的信号分解

脚本：[ocar/analysis/obs_type_traj_avg.py](ocar/analysis/obs_type_traj_avg.py)

观测类型：`state_change` / `new_location` / `revisit` / `nothing_happens`。

### 3.1 ALFWorld 7B（step 150, SR≈91%）

| obs_type | frac(S) | frac(F) | entropy(S) | entropy(F) | Δs(S) | Δs(F) | wm_s(S) | wm_s(F) |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| state_change    | 33.3% |  5.2% | 0.023 | 0.816 | +0.12 | −0.14 | 3.15 | 4.42 |
| new_location    | 48.3% | 47.3% | 0.031 | 2.40  | +0.12 | −0.02 | 3.66 | 5.00 |
| revisit         |  3.6% |  9.0% | 0.040 | 2.52  | +0.13 | −0.09 | 3.29 | 5.53 |
| nothing_happens | 14.8% | 38.5% | 0.033 | 2.53  | +0.15 | +0.08 | 3.81 | 5.75 |

### 3.2 ALFWorld 1.5B（step 60, SR≈23%）

| obs_type | entropy(S) | entropy(F) | gap |
|---|:-:|:-:|:-:|
| state_change    | 0.70 | 0.77 | 0.07 |
| new_location    | 0.72 | 0.82 | 0.10 |
| nothing_happens | 0.74 | 0.80 | 0.06 |

**要点**：
- 7B 成功轨迹 entropy 全部 <0.04，失败轨迹 0.8–2.5：**entropy 在 obs 类型维度上全面区分**。
- **obs 类型分布本身**就是强 traj-level 特征：succ 轨迹 state_change 33% vs fail 5%；nothing_happens succ 15% vs fail 39%。
- 1.5B entropy gap 仅 0.06–0.10，信号极弱。

---

## 4. WebShop 按 observation 类型的信号分解

脚本：[ocar/analysis/obs_type_traj_avg.py](ocar/analysis/obs_type_traj_avg.py)

WebShop obs 类型：`search_results` / `product_page` / `options_page` / `nothing_happens`。

（以 step 240 和 step 640 为代表；完整结果见 `ocar/analysis_results/` 对应 JSON）

| step | obs_type | frac(S) | frac(F) | entropy(S) | entropy(F) | Δs(S) | Δs(F) |
|:-:|---|:-:|:-:|:-:|:-:|:-:|:-:|
| 240 | search_results | 34% | 37% | 0.71 | 1.03 | +0.03 | −0.04 |
| 240 | product_page   | 41% | 33% | 0.40 | 0.78 | +0.02 | −0.02 |
| 240 | options_page   | 20% | 14% | 0.25 | 0.61 | +0.05 | −0.01 |
| 240 | nothing_happens|  5% | 16% | 0.55 | 0.91 | +0.04 | +0.03 |
| 640 | search_results | 31% | 35% | 0.89 | 0.77 | −0.01 | +0.06 |
| 640 | product_page   | 43% | 34% | 0.48 | 0.55 | −0.02 | +0.03 |
| 640 | options_page   | 21% | 15% | 0.34 | 0.49 | +0.02 | +0.04 |
| 640 | nothing_happens|  5% | 16% | 0.68 | 0.71 | +0.01 | +0.07 |

**要点**：
- WebShop 上 entropy succ/fail gap 远弱于 ALFWorld（0.1–0.4 vs 0.8–2.5）。
- **Δs 方向随 step 翻转**：step 240 "succ +0.03 vs fail −0.04"，step 640 反转为 "succ −0.01 vs fail +0.06"。
- 跨步点唯一稳定的 obs-type 信号：`nothing_happens` 失败比例始终远高于成功（16% vs 5%）。

---

## 5. Hidden-state information gain（跨域失败）

脚本：[ocar/analysis/hidden_state_info_gain.py](ocar/analysis/hidden_state_info_gain.py), [ocar/analysis/hidden_state_ig_alfworld_base.py](ocar/analysis/hidden_state_ig_alfworld_base.py)

定义：info_gain = ‖h_post(obs) − h_pre(obs)‖₂（最后一层 hidden state）。

| 数据 | succ IG (L-1) | fail IG (L-1) | gap | r(IG, succ) | p |
|---|:-:|:-:|:-:|:-:|:-:|
| ALFWorld step150 | 278.2 | 101.6 | +176.6 | +0.538 | <0.0001 |
| ALFWorld base (Qwen2.5-7B-Inst) | 286.4 | 112.9 | +173.5 | +0.500 | <0.0001 |
| WebShop step 240 | 184.9 | 248.7 | **−63.8（反向）** | −0.305 | 0.020 |
| WebShop step 640 | 238.2 | 224.7 | +13.5 | +0.077 | 0.946 |

**要点**：
- ALFWorld 上强信号 & 跨模型一致（base ≈ trained）。
- WebShop 上方向翻转且极弱：**obs 长度 + max_length=4096 截断伪影**（SUCC 30% step IG=0.0）+ search vs product page 结构异质。
- 结论：**不具备跨域通用性**，放弃作通用 intrinsic reward。

---

## 7. Policy-Forward-Pass 信号总表 ⭐

| 信号 | 免费？ | 因果方向 | 跨域稳定 | step-level 区分力 | 可作 step-reward？ |
|---|:-:|---|---|---|:-:|
| **entropy**        | ✓ | ✓ 模型当前不确定性 | 较稳定（后期 WebShop 衰减） | 最强 | **⚠️ 仅限 early/mid** |
| Δs                 | ✓ | ✗ 训练结果回读（KL 局部版本） | 不稳定 | 不可靠 | ✗ |
| s_θ                | ✓ | ✗ 间接受训练污染 | 不稳定 | 不可靠 | ✗ |
| \|Δs\| / z(Δs)     | ✓ | ✗ 同 Δs | 不稳定 | 不可靠 | ✗ |
| wm_s               | ✗ 额外 fwd | ✓ 环境预测 | 不稳定 | 中等 | ✗ |
| wm_gap             | ✗ 2× 额外 fwd | ✓ 信息增益 | 不稳定 | 弱 | ✗ |
| hidden-state    | ✗ 额外 fwd | ✓ 表征变化 | **ALFW 强 / WebShop 失败** | ALFW 最强 | ✗（非通用） |

---

## 8. 两日核心论断

1. **Δs 的因果方向错误**：Δs 本质是 KL 局部化版本，记录"训练已发生的更新"，不是动作质量的因果信号。用 Δs 做 step-reward = 循环论证。
2. **所有 NLL 差异信号（Δs / s_θ / |Δs|）均间接受 GRPO 训练污染**，WebShop 上方向翻转。
3. **Entropy 是 policy forward pass 中唯一因果方向正确、跨域较稳定的免费信号**；但属已知工作（EPO / CARL / UEC-RL 等），单独使用缺乏新颖性。
4. **Entropy slope 方向跨所有 run 一致**，但信号弱，需与其他机制组合。
5. **Hidden-state info gain ALFWorld 上强信号、WebShop 退化为噪声**——obs 长度 + 结构异质导致不跨域。
6. **|Δs| 作信用分配权重在 WebShop 上失败**：search 步骤因 token 生成难度垄断 54–86% 权重，与动作质量无关。
7. **Obs-type 分布**（state_change 占比、nothing_happens 占比）是最稳定的 traj-level outcome 线索，比任何 NLL 信号都更可靠，**但不是 step-level reward**。

### 对方法设计的更新约束

- **免费 step-level reward 的设计空间已基本穷尽**：要么接受 entropy 家族（已被做过），要么引入新信息源（learned critic、self-evaluation、轨迹拓扑、obs 转移模式）。
- 任何基于 NLL-差异的 step-reward 必须在 WebShop 跨训练阶段稳定性上通过 sanity check（Δs / s_θ / wm_s / wm_gap 均未通过）。

---

**数据产出**：`ocar/analysis_results/{model_1_5b, hidden_state, webshop, entropy_surprise}/*.json`
**完整叙事**：[ocar/EXPERIMENT_LOG.md](ocar/EXPERIMENT_LOG.md) §8（跨域通用发现）/ §9（hidden-state IG）/ §10（step-level 信号深入分析）

---

## 附录 A：Case 分析——Δs 无法区分好坏 state/action 的具体反例

脚本：[ocar/analysis/case_study.py](ocar/analysis/case_study.py)。所有 case 来自实际 rollout，不是合成。

### A.1 同一 observation 下，同一 Δs，好坏动作无法区分（WebShop step 240）

观察均为初始搜索页 `'Search'`，Δs = **−0.8189**、s_θ = 3.6576 完全一致：

| outcome | traj | action |
|---|---|---|
| **FAIL** | `18b2ff97` | `search[loose fit women's tops, tees & blouses c1-blue x-large size short sleeve long sleeve $30.00 max price]` |
| SUCCESS | `f239659e` | `search[loose fit women's tops c1-blue c1-blue short sleeve x-large size]` |
| SUCCESS | `cd29e904` | `search[loose fit women's tops, tees & blouses short sleeve c1-blue x-large size under $30.00 USD]` |
| SUCCESS | `97e0cc8c` | `search[loose fit women's tops, tees & blouses short sleeve c1-blue x-large size $30.00 max price]` |

→ Δs 完全由 obs 文本决定，对"这次搜索能否命中目标商品"毫无区分力。

### A.2 Δs 最高（"最不 surprising"）的 step 反而是失败轨迹 + 错误动作（ALFWorld 7B step 150）

| delta_s | outcome | observation | action |
|---|---|---|---|
| **+0.3817** | **FAIL** | `You turn on the desklamp 1.` | `go to dresser 1` |
| +0.3789 | FAIL | `You turn on the desklamp 1.` | `examine sidetable 3` |
| +0.3524 | FAIL | `You arrive at dresser 1. On the dresser 1, you see a television 1.` | `go to sidetable 3` |
| +0.3402 | FAIL | `Nothing happens.` | `go to sidetable 3` |

→ Δs 高 ≠ 动作好；policy "高置信"的反而是卡住/无效动作。

### A.3 Δs 最负（"最 surprising"）却恰是成功轨迹的任务关键动作（ALFWorld 7B step 150）

| delta_s | outcome | observation | action |
|---|---|---|---|
| −0.2596 | **SUCCESS** | `You pick up the soapbottle 2 from the cabinet 3.` | `go to toilet 1` |
| −0.2508 | SUCCESS | `You arrive at cabinet 3. The cabinet 3 is closed.` | `open cabinet 3` |
| −0.2448 | SUCCESS | `You arrive at toilet 1. On the toilet 1, you see a soapbar 3, and a toiletpaper 2.` | `move soapbottle 2 to toilet 1`（任务关键动作） |

→ 真正完成任务的 pick-up / open / move 序列反而 Δs 最负——若按 "Δs 高=好" 分配 credit，credit 方向完全反。

### A.4 Δs 排序 top-k 全是失败轨迹（ALFWorld 1.5B step 60）

step 60 ckpt，Δs 最高（"最不 surprising"）的 top-5 step **全部来自 FAIL 轨迹**，且均为开局导航：

| delta_s | outcome | step | action |
|---|---|---|---|
| −0.0690 | FAIL | 1/50 | `go to countertop 1` |
| −0.0711 | FAIL | 1/50 | `take apple 1 from countertop 1` |
| −0.0757 | FAIL | 0/50 | `go to drawer 1` |
| −0.0757 | FAIL | 0/50 | `go to cabinet 1` |
| −0.0757 | FAIL | 0/50 | `go to countertop 1` |

→ "模型最不 surprised" 的 step 反而预示失败。

### A.5 同一 prompt 多次 rollout，Δs 完全一致但 outcome 不同（WebShop step 320）

4 条不同 FAIL 轨迹的 search 动作几乎逐字相同，Δs **全部 = −0.6440**：

```
search[women's lingerie, sleep & lounge, long sleeve, tummy control, high waist, short sleeve, purple, xx-large, price < 30.00 dollars]
```

同一 obs + 极相似 action 串下 Δs 固定，这个量无法反映"本次 rollout 会不会成功"——它只是 token 生成难度的读数。

### A.6 "反直觉率"统计——任何 Δs step-reward 都会把大量 step 方向给反

| 数据 | SR | Succ step Δs<p10 占比 | Fail step Δs>p90 占比 |
|---|:-:|:-:|:-:|
| ALFWorld 1.5B step 60 | 23% | 5.1% (43/848) | **10.2%** (503/4912) |
| ALFWorld 7B step 150  | 91% | 1.5% (15/975) | 5.0% (28/561) |
| WebShop step 240 | 47% | **11.2%** (14/125) | 1.6% (2/123) |

→ 即便在信号最"干净"的 ALFWorld 7B 上，也有 5% 的 fail step 被 Δs 排进 top-10%、1.5% 的 succ step 被排进 bottom-10%。若作 step-level reward，这部分 step 的 credit 会直接打反。

### A.7 小结

- **Case A.1 / A.5**：说明 Δs 是 **obs × token 难度** 的函数，不是 action quality。
- **Case A.2 / A.3 / A.4**：说明 Δs 和 outcome 的**单 step 方向不一致**，且与任务关键动作反相关。
- **Case A.6**：即使做百分位分析，也存在 5–11% 的反方向 step。

这与 §8 的论断一致：**Δs 记录的是"训练已发生的更新"，而非"动作质量"**，单 step 层面无法作为 credit assignment 信号。
