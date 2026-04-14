# Project Context: OCAR Research

This is a **fork of verl-agent** used solely for the OCAR research project. Most files in this repository are inherited from the upstream fork and are NOT part of our work.

## What This Project Is About

We are developing **OCAR (Observation-grounded Credit Advantage Redistribution)** — a zero-cost step-level credit assignment method for LLM agent RL training. OCAR uses observation token log-probabilities (already computed during forward pass but normally discarded) as a surprise signal to redistribute GRPO's uniform episode advantage across steps.

## Relevant Directories

Focus on these when helping with the project:

| Directory | Contents |
|-----------|----------|
| `ocar/` | **PROJECT_STATUS.md** (current state, results, next steps), OCAR-specific docs |
| `ocar/core_ocar.py` | Core OCAR advantage computation |
| `examples/ocar_trainer/` | Training scripts and configs for OCAR experiments |
| `verl/trainer/ppo/ray_trainer.py` | Main trainer — OCAR integration at advantage computation |
| `verl/trainer/ppo/core_algos.py` | Loss computation (PPO-clip, KL penalty) |
| `verl/workers/rollout/` | Rollout and trajectory collection |
| `gigpo/core_gigpo.py` | GiGPO baseline (competitor method) |

## Directories to Ignore

These are from the upstream fork and not relevant to OCAR:

- `docs/` — upstream verl-agent documentation
- `docker/` — Docker configs for upstream
- `recipe/` — upstream experiment recipes (not ours)
- `tests/` — upstream tests
- `scripts/` — upstream utility scripts
- Most files in `examples/` (except `ocar_trainer/`)

## Key Technical Facts

- **Model**: Qwen2.5-7B-Instruct
- **Environments**: ALFWorld (primary), WebShop (planned)
- **Framework**: verl (FSDP + vLLM), Ray distributed training
- **Algorithm**: GRPO base + OCAR advantage redistribution
- **Config key**: `algorithm.adv_estimator=ocar`
- **Loss mask**: action tokens = RL loss (loss_mask=1), observation tokens = discarded (loss_mask=0)
- **OCAR signal**: Mean NLL of observation tokens per step, extracted from same forward pass

## Current Priority

**Beat GiGPO standalone** — fix known issues (Clean subtask regression, training collapse, τ tuning), then demonstrate advantage on non-deterministic environments (WebShop).

Read `ocar/PROJECT_STATUS.md` for full details on results, known issues, and action items.
