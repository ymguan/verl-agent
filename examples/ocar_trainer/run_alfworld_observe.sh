#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Pure GRPO Training on ALFWorld — with Observational Surprise Logging
#
# Uses standard GRPO advantage (no OCAR reweighting).
# Computes and logs surprise variants + entropy for analysis only.
#
# Surprise signals logged (observe-only, do NOT influence training):
#   - obs_s_theta: raw NLL from actor
#   - obs_s_ref: raw NLL from reference model
#   - obs_delta_s: S_theta - S_ref
#   - obs_consecutive_s: step-to-step surprise delta
#   - obs_step_entropy_{mean,std,min,max}: action token entropy
#
# Optional: +algorithm.observe_surprise_wm=true for world-model
# style surprise (extra forward pass cost).
# ──────────────────────────────────────────────────────────────
set -x
ENGINE=${1:-vllm}
shift 2>/dev/null || true
export WANDB_API_KEY='07d67694ce977d4e8e96369367c00af9a0becb7c'
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
export ALFWORLD_DATA=${ALFWORLD_DATA:-$PROJECT_DIR/alfworld_data}

# ── Standard config (same as OCAR baseline for fair comparison) ──
num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8
MODEL=${MODEL:-"Qwen/Qwen2.5-7B-Instruct"}
N_GPUS=${N_GPUS:-8}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
SEED=${SEED:-0}

DATA_DIR=${DATA_DIR:-$HOME/data/verl-agent/text}

# Ensure parquet data matches batch sizes
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
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
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=$SEED \
    env.max_steps=50 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="grpo_observe_alfworld_$(date +%Y%m%d_%H%M%S)" \
    trainer.experiment_name="grpo_observe_seed${SEED}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True $@
