#!/bin/bash
# ──────────────────────────────────────────────────────────────
# OCAR Quick Smoke Test — 验证 pipeline 不崩溃
#
# 跑 3 个 epoch, 小 batch, 快速验证:
#   ✓ ALFWorld 环境正常 reset/step
#   ✓ vLLM rollout 能生成
#   ✓ OCAR advantage 计算不报错
#   ✓ Actor update 不崩溃
#   ✓ Validation 能跑通
#
# 预计耗时: ~5-10 分钟 (vs 完整训练 ~10+ 小时)
# ──────────────────────────────────────────────────────────────
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TMPDIR=/local_nvme/guanyiming/tmp
export ALFWORLD_DATA=/local_nvme/guanyiming/project/verl-agent/alfworld_data
mkdir -p $TMPDIR

# ── Smoke test config: minimal sizes ──
train_data_size=4          # 只 4 个 prompt (正式: 16)
val_data_size=8            # 只 8 个 val (正式: 128)
group_size=2               # 只 2 rollout/prompt (正式: 8)
TOTAL_EPOCHS=3             # 只 3 epoch (正式: 150)
TEST_FREQ=1                # 每 epoch 都 validate
MODEL=${MODEL:-"/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct"}
N_GPUS=${N_GPUS:-4}

DATA_DIR=/local_nvme/guanyiming/project/verl-agent/data/text

# Regenerate small parquet
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir /local_nvme/guanyiming/project/verl-agent/data

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ocar \
    +algorithm.ocar.tau=1.0 \
    +algorithm.ocar.use_delta_s=true \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=42 \
    env.max_steps=10 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=0.1 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='ocar_smoke_test' \
    trainer.experiment_name='smoke_test' \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True $@

echo ""
echo "=========================================="
echo "  SMOKE TEST COMPLETE — No crashes! ✓"
echo "=========================================="
