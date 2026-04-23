#!/bin/bash
# ──────────────────────────────────────────────────────────────
# Eval ALL dual-token HF checkpoints on test set (alfworld valid_unseen)
# Reads HF model directly via actor_rollout_ref.model.path=<repo_id>
# ──────────────────────────────────────────────────────────────
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TMPDIR=/local_nvme/guanyiming/tmp
export ALFWORLD_DATA=/local_nvme/guanyiming/project/verl-agent/alfworld_data
export PATH=/local_nvme/guanyiming/env/verl-agent-06x-py312/bin:$PATH
export HF_HOME=/local_nvme/hf_cache
mkdir -p $TMPDIR

STEPS=${STEPS:-"20 40 60 80 100 120 140 150"}
SEEDS=${SEEDS:-"123 456 789"}
EVAL_DATASET=${EVAL_DATASET:-"eval_out_of_distribution"}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
N_GPUS=${N_GPUS:-4}
HF_PREFIX=${HF_PREFIX:-"Ricardo-H/ws-wm-0416-step-"}

num_cpus_per_env_worker=0.1
train_data_size=16
group_size=8

DATA_DIR=/local_nvme/guanyiming/project/verl-agent/data/text
LOG_DIR=logs/dual_token_eval
mkdir -p $LOG_DIR

for step in $STEPS; do
    MODEL="${HF_PREFIX}${step}"
    for seed in $SEEDS; do
        log_file="${LOG_DIR}/step${step}_seed${seed}.log"
        if [ -f "$log_file" ] && grep -q "val/success_rate" "$log_file" 2>/dev/null; then
            echo "SKIP step=$step seed=$seed (already done)"
            continue
        fi
        echo "=========================================="
        echo "Eval dual-token: step=${step} seed=${seed} model=${MODEL}"
        echo "=========================================="

        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=grpo \
            data.train_files=$DATA_DIR/train.parquet \
            data.val_files=$DATA_DIR/test.parquet \
            data.train_batch_size=$train_data_size \
            data.val_batch_size=$VAL_DATA_SIZE \
            data.max_prompt_length=2048 \
            data.max_response_length=512 \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            data.return_raw_chat=True \
            actor_rollout_ref.model.path="$MODEL" \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=128 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.01 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
            actor_rollout_ref.rollout.name=$ENGINE \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
            actor_rollout_ref.rollout.enable_chunked_prefill=False \
            actor_rollout_ref.rollout.enforce_eager=False \
            actor_rollout_ref.rollout.free_cache_engine=False \
            actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
            actor_rollout_ref.rollout.val_kwargs.do_sample=True \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            actor_rollout_ref.actor.use_invalid_action_penalty=True \
            actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
            algorithm.use_kl_in_reward=False \
            env.env_name=alfworld/AlfredTWEnv \
            env.alfworld.eval_dataset=$EVAL_DATASET \
            env.seed=$seed \
            env.max_steps=50 \
            env.rollout.n=$group_size \
            env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
            trainer.critic_warmup=0 \
            trainer.logger=['console'] \
            trainer.project_name='dual_token_eval' \
            trainer.experiment_name="dual_token_step${step}_seed${seed}" \
            trainer.n_gpus_per_node=$N_GPUS \
            trainer.nnodes=1 \
            trainer.save_freq=-1 \
            trainer.test_freq=-1 \
            trainer.total_epochs=1 \
            trainer.val_only=True \
            trainer.val_before_train=True 2>&1 | tee "$log_file"

        echo "Done dual-token: step=${step} seed=${seed}"
    done
done

echo "=========================================="
echo "DUAL-TOKEN EVAL COMPLETE."
for step in $STEPS; do
    echo "--- step $step ---"
    for seed in $SEEDS; do
        log="${LOG_DIR}/step${step}_seed${seed}.log"
        [ -f "$log" ] && echo "  seed=$seed: $(grep -oE 'val/success_rate:[0-9.]+' "$log" | tail -1)"
    done
done
