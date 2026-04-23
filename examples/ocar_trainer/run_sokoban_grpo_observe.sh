#!/bin/bash
# ──────────────────────────────────────────────────────────────
# GRPO-only + Observational Surprise Logging
# Qwen2.5-VL-3B-Instruct on Sokoban, 2 GPUs, tp=2
# Based on gigpo_trainer/run_sokoban.sh with observe flags added
# ──────────────────────────────────────────────────────────────
set -x
ENGINE=${1:-vllm}
shift 2>/dev/null || true

VERL_PY_BIN=${VERL_PY_BIN:-/local_nvme/guanyiming/env/verl-agent-06x-py312/bin}
export PATH=$VERL_PY_BIN:$PATH

export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_ENTITY=${WANDB_ENTITY:-guanyiming290-alibaba}

export VLLM_ATTENTION_BACKEND=XFORMERS
export TMPDIR=${TMPDIR:-/local_nvme/guanyiming/tmp/grpo_observe_sokoban}
export RAY_TMPDIR=${RAY_TMPDIR:-/local_nvme/guanyiming/tmp/ray_grpo_sokoban}
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

num_cpus_per_env_worker=0.1
train_data_size=32
val_data_size=128
group_size=8

MODEL=${MODEL:-"Qwen/Qwen2.5-VL-3B-Instruct"}
N_GPUS=${N_GPUS:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
SEED=${SEED:-0}
RUN_TAG=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}

DATA_DIR=${DATA_DIR:-$HOME/data/verl-agent/visual}

python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    +algorithm.observe_surprise=true \
    +algorithm.observe_surprise_wm=true \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=Sokoban \
    env.seed=$SEED \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.sokoban.mode='rgb_array' \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="grpo_observe_sokoban_${RUN_TAG}" \
    trainer.experiment_name="grpo_observe_qwen2.5_vl_3b_seed${SEED}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True $@
