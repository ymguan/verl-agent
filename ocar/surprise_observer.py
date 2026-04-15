"""
Surprise Observer: observation-only surprise computation for GRPO experiments.

Computes multiple surprise signal variants WITHOUT modifying advantages.
All signals are returned as numpy arrays for logging/analysis only.

Surprise is computed from **true observation token NLL** using full_log_probs.
Two granularities:
  - "last": only the LAST obs block before the response (= current step's observation)
  - "all": all non-action tokens (attention_mask=1 & loss_mask=0) across the full sequence

Both mean and sum NLL are recorded alongside token counts so downstream
analysis can study length effects.

Variants:
    - obs_s_theta_{mean,sum}: last-obs-block NLL from actor (primary signal)
    - obs_s_all_theta_{mean,sum}: all-obs NLL from actor (secondary)
    - obs_s_ref_{mean,sum}: last-obs-block NLL from reference model
    - obs_delta_s_{mean,sum}: S_theta - S_ref (denoised)
    - obs_n_tokens / obs_n_tokens_all: token counts
    - obs_consecutive_s: step-to-step surprise delta (mean-based)
    - obs_step_entropy_{mean,std,min,max}: action token entropy stats
"""
import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _find_last_obs_block(loss_mask_row: torch.Tensor, prompt_len: int) -> tuple[int, int]:
    """Find the start and end of the last contiguous 0-block in loss_mask[:prompt_len].

    In multi-turn sequences, loss_mask alternates between 0-blocks (obs/system)
    and 1-blocks (action content). The last 0-block in the prompt is the current
    step's observation + assistant prefix tokens.

    Args:
        loss_mask_row: (seq_len,) loss mask for one sample
        prompt_len: length of the prompt portion

    Returns:
        (start, end) indices into the full sequence, end is exclusive.
        Returns (0, 0) if no valid block found.
    """
    # Scan backward from prompt_len - 1 to find the 0-block
    prompt_mask = loss_mask_row[:prompt_len]

    # Find end of last 0-block (should be prompt_len since prompt ends with 0s)
    end = prompt_len

    # Scan backward to find where this 0-block starts (first 1 going backward)
    start = end
    for j in range(end - 1, -1, -1):
        if prompt_mask[j] == 1:
            start = j + 1
            break
    else:
        # All 0s in prompt (step 0: system + obs_0 + prefix)
        start = 0

    if start >= end:
        return (0, 0)

    return (start, end)


def _compute_obs_nll_last_block(
    full_log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    response_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-step NLL using only the LAST obs block (current observation).

    Args:
        full_log_probs: (bs, seq_len - 1) full sequence log probs (shifted)
        attention_mask: (bs, seq_len)
        loss_mask: (bs, seq_len)
        response_length: length of the response portion

    Returns:
        (nll_mean, nll_sum, n_tokens) each of shape (bs,)
    """
    bs, seq_len_minus1 = full_log_probs.shape
    seq_len = seq_len_minus1 + 1
    prompt_len = seq_len - response_length

    nll_mean = np.zeros(bs, dtype=np.float64)
    nll_sum = np.zeros(bs, dtype=np.float64)
    n_tokens = np.zeros(bs, dtype=np.float64)

    for i in range(bs):
        start, end = _find_last_obs_block(loss_mask[i], prompt_len)
        if start >= end:
            continue

        # full_log_probs is shifted: token at position j -> full_log_probs[:, j-1]
        # Token at position 0 has no log prob (first token), so skip it.
        if start == 0:
            block_attn = attention_mask[i, 1:end]
            flp_indices = torch.arange(0, end - 1)
        else:
            block_attn = attention_mask[i, start:end]
            flp_indices = torch.arange(start - 1, end - 1)

        valid = block_attn.bool()
        if not valid.any():
            continue

        obs_lp = full_log_probs[i, flp_indices[valid]]
        n = obs_lp.numel()
        s = -obs_lp.sum().item()
        nll_mean[i] = s / n
        nll_sum[i] = s
        n_tokens[i] = n

    return nll_mean, nll_sum, n_tokens


def _compute_obs_nll_all(
    full_log_probs: torch.Tensor,
    attention_mask: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-step NLL over ALL non-action tokens in the sequence.

    obs_mask = (attention_mask[:, 1:] == 1) & (loss_mask[:, 1:] == 0)

    Args:
        full_log_probs: (bs, seq_len - 1)
        attention_mask: (bs, seq_len)
        loss_mask: (bs, seq_len)

    Returns:
        (nll_mean, nll_sum, n_tokens) each of shape (bs,)
    """
    obs_mask = (attention_mask[:, 1:] == 1) & (loss_mask[:, 1:] == 0)

    bs = full_log_probs.shape[0]
    nll_mean = np.zeros(bs, dtype=np.float64)
    nll_sum = np.zeros(bs, dtype=np.float64)
    n_tokens = np.zeros(bs, dtype=np.float64)

    for i in range(bs):
        mask_i = obs_mask[i]
        if mask_i.any():
            obs_lp = full_log_probs[i][mask_i]
            n = obs_lp.numel()
            s = -obs_lp.sum().item()
            nll_mean[i] = s / n
            nll_sum[i] = s
            n_tokens[i] = n

    return nll_mean, nll_sum, n_tokens


def _compute_consecutive_surprise(
    s_theta: np.ndarray,
    traj_index: np.ndarray,
) -> np.ndarray:
    """Compute step-to-step surprise change within each trajectory.

    For step t > 0: consecutive_s[t] = s_theta[t] - s_theta[t-1]
    For step t = 0: consecutive_s[t] = 0
    """
    bs = len(s_theta)
    consecutive = np.zeros(bs, dtype=np.float64)

    traj2indices = defaultdict(list)
    for i in range(bs):
        traj2indices[traj_index[i]].append(i)

    for _, indices in traj2indices.items():
        indices = sorted(indices)
        for j in range(1, len(indices)):
            consecutive[indices[j]] = s_theta[indices[j]] - s_theta[indices[j - 1]]

    return consecutive


def _compute_entropy_stats(
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
) -> dict[str, np.ndarray]:
    """Compute per-step entropy statistics from token-level entropy.

    Entropy is computed on action tokens (response_mask=1).
    """
    bs = entropys.shape[0]
    stats = {
        "obs_step_entropy_mean": np.zeros(bs, dtype=np.float64),
        "obs_step_entropy_std": np.zeros(bs, dtype=np.float64),
        "obs_step_entropy_min": np.zeros(bs, dtype=np.float64),
        "obs_step_entropy_max": np.zeros(bs, dtype=np.float64),
    }

    for i in range(bs):
        mask = response_mask[i].bool()
        step_ent = entropys[i][mask]
        if len(step_ent) > 0:
            stats["obs_step_entropy_mean"][i] = step_ent.mean().item()
            stats["obs_step_entropy_std"][i] = step_ent.std().item() if len(step_ent) > 1 else 0.0
            stats["obs_step_entropy_min"][i] = step_ent.min().item()
            stats["obs_step_entropy_max"][i] = step_ent.max().item()

    return stats


def compute_surprise_variants(data) -> dict[str, np.ndarray]:
    """Compute all surprise signal variants from a training batch.

    Uses full_log_probs (full sequence log probs) to compute true observation
    token NLL. Two granularities:
      - "last block": only the last obs block (current step's observation) — primary signal
      - "all": all non-action tokens — secondary

    Args:
        data: DataProto batch with tensor and non-tensor fields.
              Required: full_log_probs, attention_mask, loss_mask, traj_uid, responses.
              Optional: full_ref_log_probs, step_entropys, response_mask.

    Returns:
        dict mapping signal names to (bs,) numpy arrays
    """
    traj_index = data.non_tensor_batch["traj_uid"]
    result = {}

    if "full_log_probs" not in data.batch:
        logger.warning(
            "observe_surprise: full_log_probs not available. "
            "Ensure return_full_log_probs=True is set in meta_info."
        )
        return result

    attention_mask = data.batch["attention_mask"]
    response_length = data.batch["responses"].shape[1]

    # Synthesize loss_mask if not present (e.g., vllm rollout without multi-turn sglang).
    # Convention: prompt tokens = obs (0), response tokens = action (1).
    if "loss_mask" in data.batch:
        loss_mask = data.batch["loss_mask"]
    else:
        seq_len = attention_mask.shape[1]
        prompt_length = seq_len - response_length
        loss_mask = torch.zeros_like(attention_mask)
        loss_mask[:, prompt_length:] = 1
        logger.info(f"observe_surprise: synthesized loss_mask from prompt/response split "
                    f"(prompt_len={prompt_length}, response_len={response_length})")

    # ── Primary: last obs block only (current step's observation) ──
    s_theta_mean, s_theta_sum, n_tokens = _compute_obs_nll_last_block(
        data.batch["full_log_probs"], attention_mask, loss_mask, response_length
    )
    result["obs_s_theta_mean"] = s_theta_mean
    result["obs_s_theta_sum"] = s_theta_sum
    result["obs_n_tokens"] = n_tokens

    # ── Secondary: all obs tokens ──
    s_all_mean, s_all_sum, n_all = _compute_obs_nll_all(
        data.batch["full_log_probs"], attention_mask, loss_mask
    )
    result["obs_s_all_theta_mean"] = s_all_mean
    result["obs_s_all_theta_sum"] = s_all_sum
    result["obs_n_tokens_all"] = n_all

    # ── Ref model surprise (last block) ──
    if "full_ref_log_probs" in data.batch:
        s_ref_mean, s_ref_sum, _ = _compute_obs_nll_last_block(
            data.batch["full_ref_log_probs"], attention_mask, loss_mask, response_length
        )
        result["obs_s_ref_mean"] = s_ref_mean
        result["obs_s_ref_sum"] = s_ref_sum
        result["obs_delta_s_mean"] = s_theta_mean - s_ref_mean
        result["obs_delta_s_sum"] = s_theta_sum - s_ref_sum
    else:
        logger.warning("observe_surprise: full_ref_log_probs not available, skipping ref/delta_s")

    # ── Consecutive surprise (step-to-step delta, based on mean NLL) ──
    result["obs_consecutive_s"] = _compute_consecutive_surprise(s_theta_mean, traj_index)

    logger.info(
        f"Surprise observer: "
        f"s_theta_last_mean={s_theta_mean.mean():.4f}±{s_theta_mean.std():.4f} "
        f"(n_tokens={n_tokens.mean():.0f}) | "
        f"s_theta_all_mean={s_all_mean.mean():.4f} "
        f"(n_tokens_all={n_all.mean():.0f}) | "
        f"consecutive={result['obs_consecutive_s'].mean():.4f}"
    )

    # ── Per-step entropy stats (on action tokens) ──
    if "step_entropys" in data.batch:
        response_mask = data.batch["response_mask"]
        entropy_stats = _compute_entropy_stats(data.batch["step_entropys"], response_mask)
        result.update(entropy_stats)
    else:
        logger.warning("observe_surprise: step_entropys not available, skipping entropy stats")

    return result


def _build_wm_batch(prompt_strs, target_strs, tokenizer):
    """Tokenize prompt/target pairs and build padded tensors for forward pass.

    Args:
        prompt_strs: list of prompt strings (context)
        target_strs: list of target strings (to predict)
        tokenizer: tokenizer

    Returns:
        DataProto batch, response_mask, max_prompt_len, or None if empty
    """
    from verl import DataProto

    if not prompt_strs:
        return None, None, None

    prompt_ids_list = []
    target_ids_list = []
    max_prompt_len = 0
    max_target_len = 0

    for prompt_str, target_str in zip(prompt_strs, target_strs):
        p_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        t_ids = tokenizer.encode(target_str, add_special_tokens=False)
        prompt_ids_list.append(p_ids)
        target_ids_list.append(t_ids)
        max_prompt_len = max(max_prompt_len, len(p_ids))
        max_target_len = max(max_target_len, len(t_ids))

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    total_len = max_prompt_len + max_target_len
    wm_bs = len(prompt_strs)

    input_ids = torch.full((wm_bs, total_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((wm_bs, total_len), dtype=torch.long)
    position_ids = torch.zeros((wm_bs, total_len), dtype=torch.long)
    response_mask = torch.zeros((wm_bs, max_target_len), dtype=torch.float32)

    for i in range(wm_bs):
        p_len = len(prompt_ids_list[i])
        t_len = len(target_ids_list[i])
        seq_len = p_len + t_len

        offset = max_prompt_len - p_len
        input_ids[i, offset:offset + p_len] = torch.tensor(prompt_ids_list[i])
        input_ids[i, max_prompt_len:max_prompt_len + t_len] = torch.tensor(target_ids_list[i])
        attention_mask[i, offset:max_prompt_len + t_len] = 1
        position_ids[i, offset:max_prompt_len + t_len] = torch.arange(seq_len)
        response_mask[i, :t_len] = 1.0

    wm_batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, max_prompt_len:],
            "response_mask": response_mask,
        }
    )
    return wm_batch, response_mask, max_prompt_len


def _extract_nll_from_output(output, response_mask, step_indices, bs):
    """Extract per-step mean NLL from forward pass output."""
    result = np.zeros(bs, dtype=np.float64)
    log_probs = output.batch["old_log_probs"]
    wm_bs = log_probs.shape[0]
    for i in range(wm_bs):
        mask = response_mask[i].bool()
        step_lp = log_probs[i][mask]
        if len(step_lp) > 0:
            result[step_indices[i]] = -step_lp.mean().item()
    return result


def compute_wm_surprise(batch, actor_rollout_wg, tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Compute world-model surprise in two variants:
      A) P(obs_{t+1} | obs_t, action_t)  — state+action context
      B) P(obs_{t+1} | action_t)          — action-only context

    For each trajectory step t > 0, constructs prompts and computes NLL.

    Args:
        batch: DataProto training batch
        actor_rollout_wg: actor worker group for forward pass
        tokenizer: tokenizer for encoding/decoding

    Returns:
        (wm_A, wm_B): each np.ndarray of shape (bs,)
    """
    traj_index = batch.non_tensor_batch["traj_uid"]
    anchor_obs = batch.non_tensor_batch.get("anchor_obs", None)
    bs = len(traj_index)
    zeros = np.zeros(bs, dtype=np.float64)

    if anchor_obs is None:
        logger.warning("compute_wm_surprise: anchor_obs not available, returning zeros")
        return zeros.copy(), zeros.copy()

    responses_decoded = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

    # Group steps by trajectory
    traj2indices = defaultdict(list)
    for i in range(bs):
        traj2indices[traj_index[i]].append(i)

    # Build pairs for both A and B
    prompts_A, prompts_B, targets, step_indices = [], [], [], []

    for _, indices in traj2indices.items():
        indices = sorted(indices)
        for j in range(1, len(indices)):
            prev_idx = indices[j - 1]
            curr_idx = indices[j]

            prev_obs = str(anchor_obs[prev_idx]) if anchor_obs[prev_idx] is not None else ""
            prev_action = responses_decoded[prev_idx]
            curr_obs = str(anchor_obs[curr_idx]) if anchor_obs[curr_idx] is not None else ""

            if not curr_obs:
                continue

            prompts_A.append(f"{prev_obs}\n{prev_action}")  # state + action
            prompts_B.append(prev_action)                    # action only
            targets.append(curr_obs)
            step_indices.append(curr_idx)

    if not prompts_A:
        logger.warning("compute_wm_surprise: no valid WM pairs found")
        return zeros.copy(), zeros.copy()

    wm_A = zeros.copy()
    wm_B = zeros.copy()

    # Forward pass for A: P(obs_{t+1} | obs_t, action_t)
    try:
        batch_A, rmask_A, _ = _build_wm_batch(prompts_A, targets, tokenizer)
        if batch_A is not None:
            out_A = actor_rollout_wg.compute_log_prob(batch_A)
            wm_A = _extract_nll_from_output(out_A, rmask_A, step_indices, bs)
    except Exception as e:
        logger.error(f"compute_wm_surprise (A) forward pass failed: {e}")

    # Forward pass for B: P(obs_{t+1} | action_t)
    try:
        batch_B, rmask_B, _ = _build_wm_batch(prompts_B, targets, tokenizer)
        if batch_B is not None:
            out_B = actor_rollout_wg.compute_log_prob(batch_B)
            wm_B = _extract_nll_from_output(out_B, rmask_B, step_indices, bs)
    except Exception as e:
        logger.error(f"compute_wm_surprise (B) forward pass failed: {e}")

    logger.info(
        f"WM surprise: A(s+a→s')={wm_A[wm_A>0].mean():.4f} "
        f"B(a→s')={wm_B[wm_B>0].mean():.4f} "
        f"A-B={((wm_A-wm_B)[wm_A>0]).mean():.4f}"
    )

    return wm_A, wm_B
