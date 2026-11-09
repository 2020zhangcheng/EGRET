"""Selection entropy of the decoder queries and its gradient w.r.t. the seeds.

The functions here are the building blocks of the ROI ranking: they turn the
decoder output into a per-seed score ``s_i = ||dH_sel/dF_i||_2``.  Nothing in
this module changes the network -- the caller differentiates its own detached
copy of the seed features (see ``PVGround._selection_entropy_seed_score``).

--------------------------------------------------------------------------
Mathematical definition (per sample)
--------------------------------------------------------------------------
Notation: ``B`` batch, ``Q = 256`` decoder queries, ``C = 256`` semantic
"soft-token" classes (``num_class`` of the checkpoint), ``N = 1024`` FPS seeds,
``M`` boxes (``M = 1`` for ScanRefer grounding: the referred object).

1) Semantic prediction of the last decoder layer::

       sem_prob (B, Q, C) = softmax(last_sem_cls_scores, dim=-1)

2) Per-query target-matching score (soft grounding score of box ``m``)::

       score_{q,m} = sum_c  p(c|q) * pmap(m, c)        ->  (B, Q, M)

   ``pmap`` is the dataset's ``positive_map`` (a phrase-token span indicator);
   it only encodes *which words* of the expression describe the object, never a
   box, so no GT geometry leaks in.  If ``pmap`` is missing the code falls back
   to ``score_q = max_c p(c|q)``.

3) Query competition distribution (softmax **over the 256 queries**, not over
   the classes), with temperature ``tau``::

       a_{q,m}   = score_{q,m} / tau
       P(q|m)    = exp(a_{q,m}) / sum_j exp(a_{j,m})      ->  (B, Q, M)

4) Instance-selection entropy::

       H_sel(m)  = - sum_q P(q|m) log P(q|m)               ->  (B, M)

   ``H_sel -> 0``      : a single query owns almost all the probability, the
                         target is unambiguously selected (in the extreme the
                         model is confidently wrong);
   ``H_sel -> log Q``  : many queries compete for the same expression, i.e. the
                         instance selection is ambiguous.

5) Entropy gradient and its magnitude (the 1024-dim score)::

       g_i = dH_sel / dF_i          (F_i = seed feature of the i-th FPS seed)
       s_i = ||g_i||_2              ->  (B, N)

   ``g_i`` is a vector telling in which feature direction (and how strongly)
   seed ``i`` would have to move to change the *selection* ambiguity, ``s_i``
   the scalar score used for ranking.  Because ``H_sel`` depends on all queries
   at once, ``s_i`` is a *competition sensitivity*, not a target relevance: its
   largest values sit on confusing same-class instances, which is exactly why
   the pure Top-K ranking is a weak ROI locator (see ``utils/roi_sampling``).

The temperature is the only free parameter.  Measured on this checkpoint
(ScanRefer val), the standard deviation of the 256 per-query target scores
varies by a factor of ~50 between samples (1.1e-3 ... 6.2e-2, median 2.8e-2),
so *no single constant works*: with ``tau = 0.02`` ten samples out of a hundred
saturate at ``H_sel = log 256 = 5.545`` (all queries equally likely, gradient
numerically zero) while ~20 others collapse to ``H_sel ~ 0``.

Therefore ``tau <= 0`` (the default) means **auto**: ``tau`` is set to one
standard deviation of the current sample's per-query scores, which makes the
softmax scale-invariant with respect to the (arbitrary) magnitude of
``score_{q,m}``.  With auto-``tau`` the hundred samples give
``H_sel in [0.0, 5.24]`` with median 2.0 and *no* sample pinned at the uniform
bound.
"""

import numpy as np
import torch

# 0 -> auto: one std of the per-query target scores of the current sample
# (see the module docstring for the measurements that motivate this)
DEFAULT_TAU = 0.0
_TAU_FLOOR = 1e-4          # guards against amplifying float noise
_EPS = 1e-12


def selection_scores(sem_prob, positive_map=None):
    """``score_{q,m} = sum_c p(c|q) pmap(m, c)`` -> ``(B, Q, M)``.

    ``positive_map`` is ``(M, C)`` / ``(B, M, C)`` with non-negative weights
    (this dataset stores a 0/1 phrase-token span).  ``None`` falls back to
    ``score_q = max_c p(c|q)`` (``M = 1``).
    """
    B, Q, C = sem_prob.shape
    if positive_map is None:
        return sem_prob.max(dim=-1, keepdim=True).values            # (B, Q, 1)

    pmap = torch.as_tensor(np.asarray(positive_map), dtype=sem_prob.dtype,
                           device=sem_prob.device)
    if pmap.dim() == 1:                        # (C,)   -> one target box
        pmap = pmap[None, None, :]
    elif pmap.dim() == 2:                      # (M, C) -> batch of one
        pmap = pmap[None]
    if pmap.shape[0] == 1 and B > 1:
        pmap = pmap.expand(B, -1, -1)
    if pmap.shape[-1] != C:                    # pad the tail with zeros
        pad = torch.zeros(pmap.shape[0], pmap.shape[1], C - pmap.shape[-1],
                          dtype=pmap.dtype, device=pmap.device)
        pmap = torch.cat([pmap, pad], dim=-1) if pmap.shape[-1] < C \
            else pmap[..., :C]
    return torch.einsum('bqc,bmc->bqm', sem_prob, pmap)


def resolve_tau(score, tau=DEFAULT_TAU):
    """Effective temperature: a constant, or ``std_q(score_q)`` when ``tau <= 0``.

    The adaptive value is **detached** (a constant w.r.t. the graph), so it only
    rescales the logits and never adds a second gradient path.

    Args:
        score: ``(B, Q, M)`` per-query target scores.
        tau: positive float -> used as is; ``<= 0`` / ``None`` -> auto.

    Returns:
        ``(B, 1, 1)`` tensor of positive temperatures.
    """
    if tau is not None and float(tau) > 0:
        return torch.full((score.shape[0], 1, 1), float(tau),
                          dtype=score.dtype, device=score.device)
    with torch.no_grad():
        s = score.detach().double()
        std = s.std(dim=1, keepdim=True, unbiased=False)          # (B, 1, M)
        std = std.clamp_min(_TAU_FLOOR)
    return std.to(dtype=score.dtype, device=score.device)


def selection_entropy(prob, reduce='sum'):
    """``H_sel(m) = -sum_q P(q|m) log P(q|m)`` -> ``(B, M)``.

    ``reduce`` only matters when several target boxes ``m`` are present.
    """
    return -(prob * torch.log(prob.clamp_min(_EPS))).sum(dim=1)       # (B, M)


def selection_entropy_scalar(prob, reduce='sum'):
    """Batch-level scalar ``H_sel`` used as the differentiation target."""
    h = selection_entropy(prob, reduce=reduce)                        # (B, M)
    if reduce == 'sum':
        return h.sum()
    if reduce == 'mean':
        return h.mean()
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
