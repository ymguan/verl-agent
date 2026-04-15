"""
Surprise Observer: observation-only surprise computation for GRPO experiments.

Computes multiple surprise signal variants WITHOUT modifying advantages.
All signals are returned as numpy arrays for logging/analysis only.

Variants:
    - obs_s_theta: raw NLL from actor log probs
    - obs_s_ref: raw NLL from reference model log probs
    - obs_delta_s: S_theta - S_ref (denoised)
    - obs_consecutive_s: S_theta[t] - S_theta[t-1] per trajectory
    - obs_step_entropy_{mean,std,min,max}: action token entropy stats
"""
import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _compute_per_step_nll(log_probs: torch.Tensor, response_mask: torch.Tensor) -> np.ndarray:
    """Compute mean negative log probability per step (batch row).

    Args:
        log_probs: (bs, response_length) token-level log probs
        response_mask: (bs, response_length) mask for valid tokens

    Returns:
        np.ndarray of shape (bs,) with mean NLL per step
    """
    bs = log_probs.shape[0]
    nlls = np.zeros(bs, dtype=np.float64)
    for i in range(bs):
        mask = response_mask[i].bool()
        step_lp = log_probs[i][mask]
        if len(step_lp) > 0:
            nlls[i] = -step_lp.mean().item()
    return nlls


def _compute_consecutive_surprise(
    s_theta: np.ndarray,
    traj_index: np.ndarray,
) -> np.ndarray:
    """Compute step-to-step surprise change within each trajectory.

    For step t > 0: consecutive_s[t] = s_theta[t] - s_theta[t-1]
    For step t = 0: consecutive_s[t] = 0

    Args:
        s_theta: (bs,) raw surprise per step
        traj_index: (bs,) trajectory uid per step

    Returns:
        np.ndarray of shape (bs,) with consecutive surprise delta
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

    Args:
        entropys: (bs, response_length) token-level entropy
        response_mask: (bs, response_length) mask

    Returns:
        dict with keys obs_step_entropy_{mean,std,min,max}, each (bs,)
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

    Uses data already available in the batch (old_log_probs, ref_log_prob,
    step_entropys, response_mask, traj_uid). Does NOT modify advantages.

    Args:
        data: DataProto batch with tensor and non-tensor fields

    Returns:
        dict mapping signal names to (bs,) numpy arrays
    """
    response_mask = data.batch["response_mask"]
    traj_index = data.non_tensor_batch["traj_uid"]

    result = {}

    # Raw surprise from actor
    s_theta = _compute_per_step_nll(data.batch["old_log_probs"], response_mask)
    result["obs_s_theta"] = s_theta

    # Raw surprise from reference model
    if "ref_log_prob" in data.batch:
        s_ref = _compute_per_step_nll(data.batch["ref_log_prob"], response_mask)
        result["obs_s_ref"] = s_ref
        result["obs_delta_s"] = s_theta - s_ref
    else:
        logger.warning("observe_surprise: ref_log_prob not available, skipping delta_s and s_ref")

    # Consecutive surprise (step-to-step delta)
    result["obs_consecutive_s"] = _compute_consecutive_surprise(s_theta, traj_index)

    # Per-step entropy stats
    if "step_entropys" in data.batch:
        entropy_stats = _compute_entropy_stats(data.batch["step_entropys"], response_mask)
        result.update(entropy_stats)
    else:
        logger.warning("observe_surprise: step_entropys not available, skipping entropy stats")

    logger.info(
        f"Surprise observer: s_theta mean={s_theta.mean():.4f} std={s_theta.std():.4f} | "
        f"consecutive mean={result['obs_consecutive_s'].mean():.4f}"
    )

    return result


def compute_wm_surprise(batch, actor_rollout_wg, tokenizer) -> np.ndarray:
    """Compute world-model style surprise: NLL(obs_t | action_{t-1}, obs_{t-1}).

    For each trajectory step t > 0, constructs a minimal prompt from
    (obs_{t-1}, action_{t-1}) and computes NLL of obs_t tokens.
    This requires one extra forward pass.

    Args:
        batch: DataProto training batch
        actor_rollout_wg: actor worker group for forward pass
        tokenizer: tokenizer for encoding/decoding

    Returns:
        np.ndarray of shape (bs,) with world-model surprise per step
    """
    from verl import DataProto

    traj_index = batch.non_tensor_batch["traj_uid"]
    anchor_obs = batch.non_tensor_batch.get("anchor_obs", None)
    if anchor_obs is None:
        logger.warning("compute_wm_surprise: anchor_obs not available, returning zeros")
        return np.zeros(len(traj_index), dtype=np.float64)

    bs = len(traj_index)
    responses_decoded = tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)

    # Group steps by trajectory
    traj2indices = defaultdict(list)
    for i in range(bs):
        traj2indices[traj_index[i]].append(i)

    # For each step t > 0, build (prev_obs + prev_action) -> curr_obs pair
    wm_prompts = []  # list of prompt strings
    wm_targets = []  # list of target strings (current obs)
    wm_step_indices = []  # which original batch index this corresponds to

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

            wm_prompts.append(f"{prev_obs}\n{prev_action}")
            wm_targets.append(curr_obs)
            wm_step_indices.append(curr_idx)

    if not wm_prompts:
        logger.warning("compute_wm_surprise: no valid WM pairs found")
        return np.zeros(bs, dtype=np.float64)

    # Tokenize and build a DataProto batch for forward pass
    wm_surprise = np.zeros(bs, dtype=np.float64)

    max_prompt_len = 0
    max_target_len = 0
    prompt_ids_list = []
    target_ids_list = []

    for prompt_str, target_str in zip(wm_prompts, wm_targets):
        p_ids = tokenizer.encode(prompt_str, add_special_tokens=True)
        t_ids = tokenizer.encode(target_str, add_special_tokens=False)
        prompt_ids_list.append(p_ids)
        target_ids_list.append(t_ids)
        max_prompt_len = max(max_prompt_len, len(p_ids))
        max_target_len = max(max_target_len, len(t_ids))

    # Pad and build tensors
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    total_len = max_prompt_len + max_target_len
    wm_bs = len(wm_prompts)

    input_ids = torch.full((wm_bs, total_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((wm_bs, total_len), dtype=torch.long)
    position_ids = torch.zeros((wm_bs, total_len), dtype=torch.long)
    response_mask = torch.zeros((wm_bs, max_target_len), dtype=torch.float32)

    for i in range(wm_bs):
        p_len = len(prompt_ids_list[i])
        t_len = len(target_ids_list[i])
        seq_len = p_len + t_len

        # Left-pad prompt, then target
        offset = max_prompt_len - p_len
        input_ids[i, offset:offset + p_len] = torch.tensor(prompt_ids_list[i])
        input_ids[i, max_prompt_len:max_prompt_len + t_len] = torch.tensor(target_ids_list[i])
        attention_mask[i, offset:max_prompt_len + t_len] = 1
        position_ids[i, offset:max_prompt_len + t_len] = torch.arange(seq_len)
        response_mask[i, :t_len] = 1.0

    # Build DataProto
    wm_batch = DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, max_prompt_len:],
            "response_mask": response_mask,
        }
    )

    try:
        wm_output = actor_rollout_wg.compute_log_prob(wm_batch)
        wm_log_probs = wm_output.batch["old_log_probs"]  # (wm_bs, max_target_len)

        for i in range(wm_bs):
            mask = response_mask[i].bool()
            step_lp = wm_log_probs[i][mask]
            if len(step_lp) > 0:
                wm_surprise[wm_step_indices[i]] = -step_lp.mean().item()
    except Exception as e:
        logger.error(f"compute_wm_surprise forward pass failed: {e}")

    return wm_surprise
