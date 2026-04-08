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
