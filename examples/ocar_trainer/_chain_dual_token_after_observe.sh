#!/bin/bash
# Wait for the observe-eval runner to finish, then start dual-token HF eval.
set -e
OBSERVE_PID=${1:-835623}
LOG=logs/dual_token_eval/_chain.log
mkdir -p logs/dual_token_eval
echo "[chain] waiting for observe-eval pid=$OBSERVE_PID to finish..." | tee -a $LOG
while kill -0 $OBSERVE_PID 2>/dev/null; do
    sleep 60
done
echo "[chain] observe-eval finished at $(date), starting dual-token eval..." | tee -a $LOG
exec bash /local_nvme/guanyiming/project/verl-agent/examples/ocar_trainer/eval_dual_token_hf.sh \
    >> logs/dual_token_eval/_runner.log 2>&1
