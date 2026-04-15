# OCAR Trainer — Observation-grounded Credit Advantage Redistribution

## 什么是 OCAR

OCAR 用 **observation surprise** 对 GRPO 的均匀 advantage 做步级重分配：

```
GRPO:  advantage_t = A_i                    (所有步相同)
OCAR:  advantage_t = A_i × T × softmax(±ΔS) (surprise 驱动的步级权重)
```

- **成功轨迹**：低 surprise 的步骤（agent 行为与环境 grounded）获更多 credit
- **失败轨迹**：高 surprise 的步骤（agent 行为 ungrounded）承担更多 blame
- **零额外开销**：surprise 信号从训练已有的 log prob 中提取，不需要额外 forward pass

## 前置准备

### 1. ALFWorld 数据

```bash
# 方式一：用下载脚本（推荐）
bash download_alfworld_data.sh

# 方式二：用 pip 包自带的命令
pip install alfworld
alfworld-download
```

数据会放在 `alfworld_data/` 下（6374 train + 251 valid_seen + 477 valid_unseen 个游戏）。

### 2. 模型

默认使用 `/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct`。可通过 `MODEL` 环境变量修改：

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct bash examples/ocar_trainer/run_alfworld.sh
```

### 3. 环境变量

脚本会自动设置以下环境变量：

| 变量 | 值 | 说明 |
|------|-----|------|
| `ALFWORLD_DATA` | `verl-agent/alfworld_data` | ALFWorld 游戏数据路径 |
| `TMPDIR` | `/local_nvme/guanyiming/tmp` | 临时文件目录（避免根分区满） |
| `VLLM_ATTENTION_BACKEND` | `FLASH_ATTN` | vLLM 注意力后端 |

## 快速开始

```bash
cd /local_nvme/guanyiming/project/verl-agent

# 完整训练（150 epoch，约 10+ 小时）
bash examples/ocar_trainer/run_alfworld.sh

# 冒烟测试（3 epoch，约 5-10 分钟，验证 pipeline 不崩溃）
bash examples/ocar_trainer/smoke_test.sh
```

## OCAR 特有参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `OCAR_TAU` | `1.0` | softmax 温度。越大权重越均匀（趋向 GRPO），越小越极端 |
| `OCAR_USE_DELTA_S` | `true` | 使用 ΔS = S_θ - S_ref 去噪。`false` 则用 raw S_θ |

```bash
# 调高温度（更保守的重分配）
OCAR_TAU=2.0 bash examples/ocar_trainer/run_alfworld.sh

# 不用 ΔS（ablation: raw surprise）
OCAR_USE_DELTA_S=false bash examples/ocar_trainer/run_alfworld.sh
```

## 训练参数说明

### 与 GRPO baseline 一致的参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `train_batch_size` | 16 | 每 epoch 采样 16 个任务 |
| `val_batch_size` | 128 | 验证用 128 个任务 |
| `group_size` (rollout.n) | 8 | 每个任务采样 8 条轨迹（用于 GRPO 分组归一化） |
| `max_steps` | 50 | 每条轨迹最多 50 步 |
| `total_epochs` | 150 | 训练 150 个 epoch |
| `test_freq` | 5 | 每 5 epoch 做一次 validation |
| `lr` | 1e-6 | 学习率 |
| `kl_loss_coef` | 0.01 | KL 惩罚系数 |
| `invalid_action_penalty_coef` | 0.1 | 格式错误扣分 |

### 因 7B 模型调整的参数

这些参数跟 GRPO baseline（1.5B）不同，是为了在 4×A100 80GB 上跑 7B 模型：

| 参数 | GRPO 1.5B | OCAR 7B | 原因 |
|------|-----------|---------|------|
| `tensor_model_parallel_size` | 2 | 4 | 7B 需要更多 GPU |
| `n_gpus_per_node` | 2 | 4 | 同上 |
| `ppo_mini_batch_size` | 256 | 128 | 降低峰值显存 |
| `ppo_micro_batch_size_per_gpu` | 32 | 8 | 降低峰值显存 |
| `param_offload` | False | True | 模型参数 offload 到 CPU |
| `optimizer_offload` | False | True | 优化器状态 offload 到 CPU |
| `gpu_memory_utilization` | 0.6 | 0.4 | vLLM KV cache 少占显存 |
| `log_prob_micro_batch_size_per_gpu` | 32 | 8 | log prob 计算用更小 batch |

如果跑 1.5B 模型，可以用原版参数：

```bash
MODEL=Qwen/Qwen2.5-1.5B-Instruct N_GPUS=2 bash examples/ocar_trainer/run_alfworld.sh
```

## 训练流程

每个 epoch：

```
1. 从 6374 个 ALFWorld 游戏中随机抽 16 个任务
2. 每个任务用当前模型跑 8 条 rollout（共 80 条轨迹）
3. 每条轨迹最多 50 步 agent-environment 交互
4. 计算 episode reward（成功=10, 失败=0）
5. Actor forward → old_log_probs → 提取 S_θ
6. Ref forward → ref_log_prob → 提取 S_ref（零额外开销，KL 惩罚已需要）
7. ΔS = S_θ - S_ref
8. GRPO 分组归一化 → episode advantage A_i
9. OCAR softmax 重分配 → per-step advantage_t = w_t × A_i
10. PPO clipped loss 更新 actor
```

### ΔS 去噪的 warm-start 性质

- **训练初期**（θ ≈ ref）：ΔS ≈ 0 → 权重均匀 → OCAR = GRPO（安全退化）
- **训练中后期**（θ 与 ref 分化）：ΔS 信号涌现 → 精准的步级信用分配

## 代码结构

```
verl-agent/
├── ocar/
│   ├── __init__.py
│   └── core_ocar.py              ← OCAR 核心算法（3 个函数）
├── verl/trainer/ppo/
│   └── ray_trainer.py            ← 修改了 2 处：
│                                    ① AdvantageEstimator 枚举加了 OCAR
│                                    ② compute_advantage() 加了 OCAR 分支
└── examples/ocar_trainer/
    ├── README.md                 ← 本文件
    ├── run_alfworld.sh           ← 完整训练脚本
    └── smoke_test.sh             ← 冒烟测试（快速验证 pipeline）
```

### core_ocar.py 三个函数

| 函数 | 作用 |
|------|------|
| `compute_obs_surprise_from_prompt_logprobs` | 从 old_log_probs / ref_log_prob 提取每步 surprise |
| `compute_ocar_outcome_advantage` | GRPO 分组归一化 + OCAR softmax 重分配 |
| `_softmax` | 带温度的数值稳定 softmax |

## Baseline 对比实验

```bash
# GRPO baseline
bash examples/grpo_trainer/run_alfworld.sh

# GiGPO baseline
bash examples/gigpo_trainer/run_alfworld.sh

# OCAR
bash examples/ocar_trainer/run_alfworld.sh

# OCAR ablation: raw S (不去噪)
OCAR_USE_DELTA_S=false bash examples/ocar_trainer/run_alfworld.sh

# OCAR ablation: 高温度
OCAR_TAU=2.0 bash examples/ocar_trainer/run_alfworld.sh
```

所有实验的核心对比指标是 wandb 上的 `val/success_rate` 曲线。

## 评测

训练过程中每 5 epoch 自动评测（`val_before_train=True` 还会在训练前评测一次 baseline）。

训练后单独评测某个 checkpoint：

```bash
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ocar \
    +algorithm.ocar.tau=1.0 \
    +algorithm.ocar.use_delta_s=true \
    actor_rollout_ref.model.path=/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct \
    trainer.default_local_dir=/path/to/checkpoint \
    trainer.val_only=True \
    trainer.val_before_train=True \
    env.env_name=alfworld/AlfredTWEnv \
    data.val_batch_size=128 \
    ...  # 其他参数同训练
```

## 常见问题

### Q: `ModuleNotFoundError: No module named 'alfworld'`

```bash
pip install alfworld
```

### Q: `OSError: No space left on device` (在 /tmp)

脚本已设置 `TMPDIR=/local_nvme/guanyiming/tmp`，如果仍有问题：

```bash
rm -rf /tmp/tmp*
```

### Q: `CUDA out of memory`

降低以下参数：
- `gpu_memory_utilization` → 0.3
- `ppo_micro_batch_size_per_gpu` → 4
- `log_prob_micro_batch_size_per_gpu` → 4

或换用更小的模型（`Qwen2.5-1.5B-Instruct`）。

### Q: `IndexError: Cannot choose from an empty sequence` (ALFWorld bug)

已修复。`envs.py` 的 `step()` 方法会捕获 ALFWorld expert 内部异常，返回 "Nothing happens." + done=True。

### Q: `AssertionError: gen_batch size X does not match obs size Y`

parquet 数据行数不匹配。确保脚本里有 `prepare` 步骤（当前脚本已包含）。

---

## 更新日志

### v3 (2026-04-14) — 自适应温度 + 环境修复

**问题背景**: v2 的 clamp `[0.1, 10.0]` 虽然能截断极端权重，但无法解决根本问题——随着训练推进，surprise 信号的绝对值持续增长（ΔS std 从 step 1 的 0.005 增长到 step 150 的 0.66），导致 softmax 输入越来越极端，clamp 触发越来越频繁，实际上退化为硬截断而非平滑的信用分配。

**核心改动**: 自适应温度 τ = max(σ(s), ε)

将 softmax 温度从固定值改为轨迹内 surprise 信号的标准差。数学上等价于 z-score 归一化后用固定 τ=1.0：

```
# 等价关系：
softmax(s / σ(s))  ≡  softmax(z-score(s))
```

这意味着无论训练到哪个阶段、surprise 信号有多大，softmax 的输入始终是零均值、单位方差的标准化信号。

**为什么这解决了崩溃问题**:
- v1 崩溃根因：surprise 绝对值增长 → softmax 饱和 → w_max >> 10 → 梯度爆炸
- 自适应 τ：信号先除以自身标准差 → 输入 softmax 的值恒定在 ~[-2, +2] 范围 → 权重永远温和
- Clamp `[0.1, 10.0]` 保留为安全网，正常情况下不应触发

**优雅性**: 一个公式、一个超参数（ε=0.1），当 surprise 信号弱时自动退化到 GRPO。

**代码改动** (`ocar/core_ocar.py`):
```python
# 新增参数: znorm_epsilon: float = 0.1
# 核心逻辑 (lines 205-216):
if np.std(traj_s) < 1e-10:
    continue  # all same → uniform

s_std = max(np.std(traj_s), znorm_epsilon)  # 自适应 τ = σ(s)
traj_z = (traj_s - np.mean(traj_s)) / s_std  # z-score

if traj_adv > 0:
    w = T * _softmax(-traj_z, temperature=tau)   # tau=1.0 固定
elif traj_adv < 0:
    w = T * _softmax(traj_z, temperature=tau)
```

**环境修复**:
1. `VLLM_USE_V1=0` → `VLLM_USE_V1=1`：vLLM 0.11.0 默认使用 V1 引擎，强制关闭会导致初始化冲突
2. `TMPDIR`/`RAY_TMPDIR` → `/local_nvme/guanyiming/tmp`：根分区 `/tmp` 仅 29G，旧 Ray session + CUDA 编译缓存会导致空间不足

**当前实验**: `ocar_alfworld_20260414_140127` / `ocar_tau1.0_dstrue`
- 配置: ΔS + adaptive τ (ε=0.1) + clamp [0.1, 10.0], 150 epochs, 4×A100
- wandb: https://wandb.ai/guanyiming290-alibaba/ocar_alfworld_20260414_140127
- 关键观察点: step 100 (v1 最佳区), step 110-125 (v1 崩溃区), step 150 (是否持续稳定)

### v2 (2026-04-14) — 稳定性 + 可观测性

**问题背景**: v1 训练在 step 120 达到 val 93%，但 step 125 后 KL 暴涨（0.3→4.0），response length 从 100 缩到 20，模型完全崩溃至 0%。根因是 OCAR softmax 在失败轨迹中给某些步分配了极端 blame weight（adv_min 飙到 -354），导致梯度爆炸。

**修复**:

1. **权重 clamp `[0.1, 10.0]`** (`ocar/core_ocar.py`)
   - OCAR softmax 权重限制在 [0.1, 10.0] 范围内，防止极端值
   - 新增 `weight_clip_min` / `weight_clip_max` 参数
   - 记录 `ocar/weight_clipped_count`，过多说明 τ 太低需调大

2. **Wandb 实时 surprise 监控** (`verl/trainer/ppo/ray_trainer.py`)
   - 每个 training step 自动记录 13 个 OCAR 指标到 wandb：

   | 指标 | 监控用途 |
   |------|---------|
   | `ocar/weight_{mean,std,min,max}` | 权重分布是否极端化 |
   | `ocar/weight_clipped_count` | clamp 触发频率 |
   | `ocar/surprise_theta_{mean,std}` | S_θ 趋势（模型 NLL） |
   | `ocar/surprise_ref_{mean,std}` | S_ref baseline |
   | `ocar/delta_s_{mean,std,min,max}` | ΔS 信号健康度 |

   **预警规则**:
   - `delta_s_std` 持续增大 → 信号发散，考虑提高 τ
   - `weight_clipped_count` > 总步数 10% → τ 太小
   - `surprise_theta_mean` 单调下降 + `response_length` 缩短 → 模型在学"少说少错"

3. **Checkpoint 轨迹 case 保存** (`verl/trainer/ppo/ray_trainer.py`)
   - 每次 `save_checkpoint` 时额外保存 `ocar_trajectories.json`
   - 包含每条轨迹的逐步 observation、action、s_theta、s_ref、delta_s、reward
   - 失败轨迹排前面，方便 debug
   - 最多 20 条轨迹，文件大小可控

   文件位置：
   ```
   checkpoints/.../global_step_100/
   ├── actor/
   ├── data.pt
   └── ocar_trajectories.json   ← 新增
   ```

   查看方法：
   ```bash
   python3 -c "
   import json
   with open('checkpoints/.../global_step_100/ocar_trajectories.json') as f:
       data = json.load(f)
   print(f'Success: {data[\"n_success\"]}, Failure: {data[\"n_failure\"]}')
   for t in data['trajectories'][:3]:
       print(f'\\nTraj {t[\"traj_id\"][:8]} success={t[\"success\"]} steps={t[\"n_steps\"]}')
       for s in t['steps'][:5]:
           print(f'  step {s[\"step\"]}: ds={s[\"delta_s\"]:.4f} | {s[\"action\"][:60]}')
   "
   ```

### v1 (2026-04-10) — 初始版本

- OCAR advantage 计算 (`ocar/core_ocar.py`)
- verl-agent 集成：`AdvantageEstimator.OCAR` 枚举 + `compute_advantage()` 分支
- ΔS = S_θ - S_ref 去噪模式
- ALFWorld 训练脚本 (`run_alfworld.sh`)

**训练结果** (Qwen2.5-7B-Instruct, 4×A100):

| Metric | 值 |
|--------|:-:|
| 初始 val/success_rate | 57.8% |
| 最佳 val/success_rate | **93.0%** (step 120, epoch 70) |
| Test set (valid_unseen) | **82.6% ± 2.8%** (step 100, 3 seeds) |
| 崩溃点 | step 125 (KL loss 0.3→3.3，weight 极端化)
