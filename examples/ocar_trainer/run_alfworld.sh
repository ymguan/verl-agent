#!/bin/bash
# ──────────────────────────────────────────────────────────────
# OCAR RL Training on ALFWorld — Direct verl-agent Integration
#
# Uses OCAR (Observation-grounded Credit Advantage Redistribution)
# as the advantage estimator, replacing GRPO's uniform credit
# with surprise-weighted per-step redistribution.
#
# Changes vs standard GRPO:
#   algorithm.adv_estimator=ocar  (instead of grpo)
#   algorithm.ocar.tau=1.0        (softmax temperature)
#   algorithm.ocar.use_delta_s=true (ΔS = S_θ - S_ref denoising)
#
# ΔS automatically uses ref model log probs (already computed for KL)
# so this adds ZERO extra forward passes to the training loop.
# ──────────────────────────────────────────────────────────────
set -x
ENGINE=${1:-vllm}
export WANDB_API_KEY='07d67694ce977d4e8e96369367c00af9a0becb7c'
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TMPDIR=/local_nvme/guanyiming/tmp
export ALFWORLD_DATA=/local_nvme/guanyiming/project/verl-agent/alfworld_data
mkdir -p $TMPDIR

# ── OCAR-specific config ──
OCAR_TAU=${OCAR_TAU:-1.0}              # softmax temperature (higher = more uniform)
OCAR_USE_DELTA_S=${OCAR_USE_DELTA_S:-true}  # use ΔS = S_θ - S_ref

# ── Standard config ──
num_cpus_per_env_worker=0.1
train_data_size=16
val_data_size=128
group_size=8
MODEL=${MODEL:-"/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct"}
N_GPUS=${N_GPUS:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-150}
SEED=${SEED:-0}

DATA_DIR=/local_nvme/guanyiming/project/verl-agent/data/text

# Ensure parquet data matches batch sizes (same as GRPO baseline)
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size \
    --local_dir /local_nvme/guanyiming/project/verl-agent/data

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=ocar \
    +algorithm.ocar.tau=$OCAR_TAU \
    +algorithm.ocar.use_delta_s=$OCAR_USE_DELTA_S \
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
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
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
    trainer.project_name='ocar_alfworld' \
    trainer.experiment_name="ocar_tau${OCAR_TAU}_ds${OCAR_USE_DELTA_S}" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True $@
