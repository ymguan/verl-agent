#!/bin/bash
# ──────────────────────────────────────────────────────────────
# OCAR ALFWorld Evaluation on Test Set (valid_unseen)
#
# Evaluates saved checkpoints on out-of-distribution test data.
# Uses env.alfworld.eval_dataset=eval_out_of_distribution
# which maps to ALFWorld's valid_unseen split (477 games).
#
# Usage:
#   # Eval best checkpoint (step 120) on test set, 3 seeds:
#   STEPS="120" bash examples/ocar_trainer/eval_alfworld.sh
#
#   # Eval on in-distribution val set instead:
#   EVAL_DATASET=eval_in_distribution STEPS="120" bash examples/ocar_trainer/eval_alfworld.sh
# ──────────────────────────────────────────────────────────────
set -x
ENGINE=${1:-vllm}
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export TMPDIR=/local_nvme/guanyiming/tmp
export ALFWORLD_DATA=/local_nvme/guanyiming/project/verl-agent/alfworld_data
export PATH=/local_nvme/guanyiming/env/verl-agent-06x-py312/bin:$PATH
mkdir -p $TMPDIR

# ── Eval config ──
CKPT_DIR=${CKPT_DIR:-"/local_nvme/guanyiming/project/verl-agent/checkpoints/ocar_alfworld/ocar_tau1.0_dstrue"}
STEPS=${STEPS:-"100 110 120"}                  # checkpoint steps to eval (peak range)
SEEDS=${SEEDS:-"123 456 789"}                  # eval seeds
EVAL_DATASET=${EVAL_DATASET:-"eval_out_of_distribution"}  # test set by default
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
N_GPUS=${N_GPUS:-4}
MODEL=${MODEL:-"/local_nvme/guanyiming/models/Qwen/Qwen2.5-7B-Instruct"}

num_cpus_per_env_worker=0.1
train_data_size=16
group_size=8

# Prepare data
python3 -m examples.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $VAL_DATA_SIZE \
    --local_dir /local_nvme/guanyiming/project/verl-agent/data

DATA_DIR=/local_nvme/guanyiming/project/verl-agent/data/text

mkdir -p logs/ocar_eval

for step in $STEPS; do
    step_dir="${CKPT_DIR}/global_step_${step}"
    if [ ! -d "$step_dir" ]; then
        echo "WARNING: checkpoint not found: $step_dir, skipping."
        continue
    fi

    for seed in $SEEDS; do
        echo "=========================================="
        echo "Evaluating: step=${step}, seed=${seed}"
        echo "=========================================="

        log_file="logs/ocar_eval/step${step}_seed${seed}.log"

        python3 -m verl.trainer.main_ppo \
            algorithm.adv_estimator=ocar \
            +algorithm.ocar.tau=1.0 \
            +algorithm.ocar.use_delta_s=true \
            data.train_files=$DATA_DIR/train.parquet \
            data.val_files=$DATA_DIR/test.parquet \
            data.train_batch_size=$train_data_size \
            data.val_batch_size=$VAL_DATA_SIZE \
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
            env.alfworld.eval_dataset=$EVAL_DATASET \
            env.seed=$seed \
            env.max_steps=50 \
            env.rollout.n=$group_size \
            env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
            trainer.critic_warmup=0 \
            trainer.logger=['console','wandb'] \
            trainer.project_name='ocar_alfworld_eval' \
            trainer.experiment_name="ocar_${EVAL_DATASET}_step${step}_seed${seed}" \
            trainer.n_gpus_per_node=$N_GPUS \
            trainer.nnodes=1 \
            trainer.save_freq=-1 \
            trainer.test_freq=-1 \
            trainer.total_epochs=150 \
            trainer.default_local_dir=${CKPT_DIR} \
            trainer.resume_mode=resume_path \
            trainer.resume_from_path=${step_dir} \
            trainer.val_only=True \
            trainer.val_before_train=True 2>&1 | tee "$log_file"

        echo "Done: step=${step}, seed=${seed}"
        echo ""
    done
done

echo "=========================================="
echo "All evaluations complete."
echo "Logs in: logs/ocar_eval/"
echo ""
echo "Quick summary:"
for step in $STEPS; do
    echo "--- Step $step ---"
    for seed in $SEEDS; do
        log="logs/ocar_eval/step${step}_seed${seed}.log"
        if [ -f "$log" ]; then
            sr=$(grep -o "val/success_rate.*" "$log" | tail -1)
            echo "  seed=$seed: $sr"
        fi
    done
done
echo "=========================================="
