"""
OCAR: Observation-grounded Credit Advantage Redistribution.

Per-step credit assignment via observation surprise softmax reweighting.
Replaces GRPO's uniform advantage broadcast with surprise-weighted redistribution:

    A_{i,t}^{OCAR} = A_i * T * w_t

where w_t = softmax(±S(o_t)) and S is observation surprise (or ΔS = S_θ - S_ref).

Integration with verl-agent:
    - Placed alongside gigpo/core_gigpo.py
    - Called from compute_advantage() in ray_trainer.py
    - Uses data already available in the training batch:
        * prompts / responses → decode to get observation text
        * old_log_probs / ref_log_prob → can extract obs-position NLL
        * anchor_obs → raw observation text per step
        * uid / traj_uid → group steps into trajectories

ΔS mode (use_delta_s=True):
    ΔS = S_θ - S_ref removes shared base-rate surprise:
    - Length bias → both models have same bias → cancels
    - Brilliant-but-Surprising → both models surprised → ΔS ≈ 0
    - Warm-start: θ ≈ ref at training start → ΔS ≈ 0 → degrades to GRPO
"""
import logging
from collections import defaultdict
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Numerically stable softmax with temperature."""
    x = np.asarray(x, dtype=np.float64)
    x_scaled = x / max(temperature, 1e-10)
    x_scaled = x_scaled - x_scaled.max()
    exp_x = np.exp(x_scaled)
    return exp_x / exp_x.sum()


def compute_obs_surprise_from_prompt_logprobs(
    batch,
    tokenizer,
    model_key: str = "old_log_probs",
) -> np.ndarray:
    """Extract observation surprise proxy from prompt decoding.

    In the multi-turn setting, each batch row = one step.
    The 'prompts' field contains the observation text (tokenized).
    The 'responses' field contains the action text.

    We approximate S(o_t) using the per-token log probability at
    the response positions, averaged.  This is a proxy because the
    true observation surprise requires logits at prompt positions,
    which are computed but not stored by the standard forward pass.

    A better approach: use anchor_obs text + an extra forward pass.
    For now, we use log_prob variance as a lightweight proxy.

    Returns:
        np.ndarray of shape (bs,) with surprise proxy per step.
    """
    bs = batch.batch[model_key].shape[0]
    response_mask = batch.batch.get("response_mask", None)
    log_probs = batch.batch[model_key]

    surprises = np.zeros(bs, dtype=np.float64)
    for i in range(bs):
        if response_mask is not None:
            mask = response_mask[i].bool()
            step_lp = log_probs[i][mask]
        else:
            step_lp = log_probs[i]

        if len(step_lp) > 0:
            # Use mean negative log prob as a surprise proxy
            # Higher NLL → more surprising
            surprises[i] = -step_lp.mean().item()

    return surprises


def compute_obs_surprise_from_anchor(
    anchor_obs: np.ndarray,
    tokenizer,
) -> np.ndarray:
    """Compute a lightweight observation surprise proxy from anchor text.

    Uses observation text length and token diversity as heuristic signals.
    This is the fallback when per-token logprobs are not available for
    observation positions.

    For the full surprise computation, use the extra forward pass mode.

    Returns:
        np.ndarray of shape (bs,) with heuristic surprise per step.
    """
    bs = len(anchor_obs)
    surprises = np.zeros(bs, dtype=np.float64)

    for i in range(bs):
        obs = anchor_obs[i]
        if obs is None or (isinstance(obs, str) and not obs):
            surprises[i] = 0.0
            continue

        if isinstance(obs, str):
            # Use token-level features
            tokens = obs.lower().split()
            n_tokens = max(len(tokens), 1)
            unique_ratio = len(set(tokens)) / n_tokens
            surprises[i] = unique_ratio * np.log(n_tokens + 1)
        else:
            surprises[i] = 0.0

    return surprises


def compute_ocar_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    traj_index: np.ndarray,
    obs_surprise_theta: np.ndarray,
    obs_surprise_ref: Optional[np.ndarray] = None,
    tau: float = 1.0,
    use_delta_s: bool = True,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    compute_mean_std_cross_steps: bool = True,
):
    """Compute GRPO + OCAR advantage with per-step surprise reweighting.

    Step 1: Standard GRPO episode-level advantage (A_i)
    Step 2: Per-step OCAR weights via surprise softmax
    Step 3: advantage_t = w_t × A_i (mean-preserving redistribution)

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) uid per sample
        traj_index: (bs,) traj_uid per sample
        obs_surprise_theta: (bs,) S_θ(o_t) from current model
        obs_surprise_ref: (bs,) S_ref(o_t) from reference model (optional)
        tau: softmax temperature (higher = more uniform)
        use_delta_s: if True, uses ΔS = S_θ - S_ref
        epsilon: numerical stability
        norm_adv_by_std_in_grpo: GRPO normalization mode
        compute_mean_std_cross_steps: GRPO grouping mode

    Returns:
        (advantages, returns) both of shape (bs, response_length)
    """
    # ── Step 1: GRPO episode-level advantage ──
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    seen_pairs = set()

    bsz = scores.shape[0]
    with torch.no_grad():
        for i in range(bsz):
            if (index[i], traj_index[i]) in seen_pairs:
                continue
            id2score[index[i]].append(scores[i])
            if not compute_mean_std_cross_steps:
                seen_pairs.add((index[i], traj_index[i]))

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"No score for index {idx}")

        episode_adv = torch.zeros(bsz, device=scores.device)
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                episode_adv[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                episode_adv[i] = scores[i] - id2mean[index[i]]

    # ── Step 2: Compute surprise signal ──
    if use_delta_s and obs_surprise_ref is not None:
        surprise = obs_surprise_theta.astype(np.float64) - obs_surprise_ref.astype(np.float64)
    else:
        surprise = obs_surprise_theta.astype(np.float64)

    # ── Step 3: OCAR per-trajectory softmax weights ──
    traj2indices = defaultdict(list)
    for i in range(bsz):
        traj2indices[traj_index[i]].append(i)

    ocar_weights = np.ones(bsz, dtype=np.float64)

    for traj_id, indices in traj2indices.items():
        indices = sorted(indices)
        T = len(indices)
        if T <= 1:
            continue

        traj_adv = episode_adv[indices[0]].item()
        traj_s = np.array([surprise[i] for i in indices])

        if np.std(traj_s) < 1e-10:
            continue  # all same → uniform

        if traj_adv > 0:
            # Success: low surprise → high weight
            w = T * _softmax(-traj_s, temperature=tau)
        elif traj_adv < 0:
            # Failure: high surprise → high blame
            w = T * _softmax(traj_s, temperature=tau)
        else:
            continue

        for j, idx in enumerate(indices):
            ocar_weights[idx] = w[j]

    # ── Step 4: Apply weights to episode advantage ──
    w_tensor = torch.tensor(ocar_weights, device=scores.device, dtype=scores.dtype)

    with torch.no_grad():
        weighted_adv = episode_adv * w_tensor
        advantages = weighted_adv.unsqueeze(-1) * response_mask

    # Log stats
    logger.info(
        f"OCAR: weight mean={ocar_weights.mean():.3f} std={ocar_weights.std():.3f} "
        f"min={ocar_weights.min():.3f} max={ocar_weights.max():.3f}"
    )

    return advantages, advantages
