#!/bin/bash
# Extra seeds + paper-config (val_temp=0.4) eval for the peak checkpoints
# of both observe and dual-token methods.
#
# Phase A (match existing protocol, val_temp=0.7): 3 new seeds on peak ckpts
# Phase B (match GiGPO paper protocol, val_temp=0.4): 6 seeds on best ckpts
set -x
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TMPDIR=/local_nvme/guanyiming/tmp
export ALFWORLD_DATA=/local_nvme/guanyiming/project/verl-agent/alfworld_data
export PATH=/local_nvme/guanyiming/env/verl-agent-06x-py312/bin:$PATH
export HF_HOME=/local_nvme/hf_cache
mkdir -p $TMPDIR

DATA_DIR=/local_nvme/guanyiming/project/verl-agent/data/text
LOG_DIR=logs/extra_seed_eval
mkdir -p $LOG_DIR

OBS_CKPT_DIR="/local_nvme/guanyiming/project/verl-agent/checkpoints/grpo_observe_alfworld_20260415_104816/grpo_observe_seed0"
DT_HF_PREFIX="Ricardo-H/ws-wm-0416-step-"
N_GPUS=4
group_size=8

run_eval() {
    local method=$1
    local step=$2
    local seed=$3
    local temp=$4
    local log_file="${LOG_DIR}/${method}_step${step}_seed${seed}_t${temp}.log"

    if [ -f "$log_file" ] && grep -q "val/success_rate" "$log_file" 2>/dev/null; then
        echo "SKIP $method step=$step seed=$seed temp=$temp (done)"
        return 0
    fi

    local MODEL_ARG RESUME_ARGS=""
    if [ "$method" = "observe" ]; then
        MODEL_ARG="Qwen/Qwen2.5-7B-Instruct"
        RESUME_ARGS="trainer.default_local_dir=${OBS_CKPT_DIR} trainer.resume_mode=resume_path trainer.resume_from_path=${OBS_CKPT_DIR}/global_step_${step}"
    else
        MODEL_ARG="${DT_HF_PREFIX}${step}"
    fi

    echo "=== ${method} step=${step} seed=${seed} temp=${temp} ==="
    python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_batch_size=16 \
        data.val_batch_size=128 \
        data.max_prompt_length=2048 \
        data.max_response_length=512 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=True \
        actor_rollout_ref.model.path="$MODEL_ARG" \
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
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.enforce_eager=False \
        actor_rollout_ref.rollout.free_cache_engine=False \
        actor_rollout_ref.rollout.val_kwargs.temperature=$temp \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.use_invalid_action_penalty=True \
        actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
        algorithm.use_kl_in_reward=False \
        env.env_name=alfworld/AlfredTWEnv \
        env.alfworld.eval_dataset=eval_out_of_distribution \
        env.seed=$seed \
        env.max_steps=50 \
        env.rollout.n=$group_size \
        env.resources_per_worker.num_cpus=0.1 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='extra_seed_eval' \
        trainer.experiment_name="${method}_step${step}_seed${seed}_t${temp}" \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_epochs=1 \
        trainer.val_only=True \
        trainer.val_before_train=True \
        $RESUME_ARGS 2>&1 | tee "$log_file"
}

NEW_SEEDS="42 2024 7"

# ===== Phase A: extra seeds at val_temp=0.7 (match our previous eval), peak ckpts only =====
# observe peaks: 100, 120, 140
# dual-token peaks: 140, 150
for s in $NEW_SEEDS; do
    run_eval observe 120 $s 0.7
    run_eval observe 140 $s 0.7
    run_eval dual    140 $s 0.7
    run_eval dual    150 $s 0.7
done

# ===== Phase B: paper-config (val_temp=0.4), 6 seeds on single best ckpt each =====
PAPER_SEEDS="123 456 789 42 2024 7"
for s in $PAPER_SEEDS; do
    run_eval observe 120 $s 0.4
    run_eval dual    150 $s 0.4
done

echo "=========================================="
echo "DONE. Summary:"
for f in ${LOG_DIR}/*.log; do
    sr=$(grep -oE "val/success_rate:[0-9.]+" "$f" | tail -1)
    echo "  $(basename $f): $sr"
done
