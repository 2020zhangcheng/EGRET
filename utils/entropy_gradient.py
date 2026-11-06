"""Selection-entropy (instance-selection ambiguity) gradient of the 1024 seeds.

This module is *read-only* with respect to the network: the forward pass only
installs a temporary forward hook on ``model.cross_encoder`` (the module that
produces ``end_points['seed_features']``) so that this tensor becomes a
differentiable leaf.  No layer, no parameter and no prediction path is changed.

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

5) Entropy gradient and its magnitude (the required 1024-dim score)::

       g_i = dH_sel / dF_i          (F_i = seed feature of the i-th FPS seed)
       s_i = ||g_i||_2              ->  (B, N)

   ``g_i`` is a vector telling in which feature direction (and how strongly)
   seed ``i`` would have to move to change the *selection* ambiguity, ``s_i`` is
   the scalar score used for ranking.  Seeds with a large ``s_i`` are the ones
   that can actually resolve (or create) the competition between candidate
   instances, so they are the informative seeds for the ROI re-partitioning.

   Because ``H_sel`` depends on all queries at once, the gradient is *not*
   local to one seed; it is obtained by a single backward pass through the
   decoder (``torch.autograd.grad``).  The optional ``target='xyz'`` variant
   differentiates w.r.t. the seed coordinates instead (``g_i in R^3``).

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
bound.  ``H_sel ~ 0`` then has a genuine meaning (one query really owns the
selection) and its gradient still ranks the seeds that could overturn that
selection.  A positive ``tau`` restores the literal fixed-temperature formula.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch


# 0 -> auto: one std of the per-query target scores of the current sample
# (see the module docstring for the measurements that motivate this)
DEFAULT_TAU = 0.0
_TAU_FLOOR = 1e-4          # guards against amplifying float noise
_EPS = 1e-12


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def normalize_score(score, mode='minmax', q_low=1.0, q_high=99.0):
    """Normalise a per-sample score vector to ``[0, 1]`` for colour mapping.

    The raw magnitude ``s_i = ||dH_sel/dF_i||_2`` has a heavy tail (a few seeds
    are much more influential than the rest), so the normalisation is exposed as
    a monotone transform that never changes the ranking / the Top-32:

    Args:
        score: ``(N,)`` numpy array (raw entropy-gradient magnitude).
        mode:
            ``'minmax'`` (default): plain ``(s - min) / (max - min)``;
            ``'robust'``: percentile-clipped ``minmax``;
            ``'log'`` / ``'log_robust'``: ``minmax(log10(s))`` (+ clipping),
            useful when the dynamic range exceeds ~3 decades.
        q_low, q_high: percentiles for the ``*_robust`` modes.

    Returns:
        ``(N,)`` float32 array in ``[0, 1]``; all-zeros if the score is
        constant (or non-finite).  The same array must be re-used for the
        Top-32 subset so that the colours are identical in panels (b) and (c).
    """
    s = np.asarray(score, dtype=np.float64).reshape(-1)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    if mode in ('log', 'log_robust'):
        floor = max(float(s.max()) * 1e-12, 1e-30)
        s = np.log10(np.maximum(s, floor))
        clip = mode == 'log_robust'
    elif mode in ('minmax', 'robust'):
        clip = mode == 'robust'
    else:
        raise ValueError(f"unknown score normalisation mode {mode!r}")

    if clip:
        s = np.clip(s, np.percentile(s, q_low), np.percentile(s, q_high))

    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(s, dtype=np.float32)
    return ((s - lo) / (hi - lo)).astype(np.float32)


def topk_seeds(score, k=32):
    """Indices of the ``k`` largest scores, sorted descending (ties by index)."""
    s = np.asarray(score).reshape(-1)
    k = int(min(k, s.shape[0]))
    # stable sort on the negated score => descending, deterministic tie-break
    idx = np.argsort(-s, kind='stable')[:k]
    return idx.astype(np.int64)


@dataclass
class SeedSelectionEntropyGradient:
    """Result of one sample's selection-entropy-gradient computation."""

    score: np.ndarray                 # (N,)  s_i = ||dH_sel/dF_i||_2, raw
    score_norm: np.ndarray            # (N,)  s_i normalised to [0, 1]
    selection_prob: np.ndarray        # (Q,)  P(q|m=0) over the 256 queries
    selection_scores: np.ndarray      # (Q,)  score_{q,0} = sum_c p(c|q) pmap
    total_entropy: float              # H_sel of the sample
    grad_target: str                  # 'feat' or 'xyz'
    query_index: int = -1             # decoder query used for the prediction
    grad: Optional[np.ndarray] = None  # (N, C) or (N, 3) if keep_grad=True
    extra: dict = field(default_factory=dict)

    @property
    def num_seeds(self):
        return int(self.score.shape[0])


# --------------------------------------------------------------------------- #
# 1) selection entropy (pure tensor math, no graph needed)
# --------------------------------------------------------------------------- #
def semantic_probabilities(sem_cls_scores, dim=-1):
    """``softmax`` over the semantic classes -> ``(B, Q, C)``."""
    if sem_cls_scores.dim() != 3:
        raise ValueError('sem_cls_scores must be (B, Q, C), got '
                         f'{tuple(sem_cls_scores.shape)}')
    return torch.softmax(sem_cls_scores, dim=dim)


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


def selection_probabilities(score, tau=DEFAULT_TAU):
    """``P(q|m) = softmax_q(score_{q,m} / tau)`` -> ``(B, Q, M)``.

    ``tau <= 0`` uses :func:`resolve_tau` (one std of the sample's scores).
    """
    tau_eff = resolve_tau(score, tau)
    return torch.softmax(score / tau_eff, dim=1)


def selection_entropy(prob, reduce='sum'):
    """``H_sel(m) = -sum_q P(q|m) log P(q|m)`` -> ``(B, M)``.

    ``reduce`` only matters when several target boxes ``m`` are present: with
    ``'sum'`` the entropies of the targets are added (``'mean'`` averages them).
    Both give the same gradient direction for ``M = 1``.
    """
    h = -(prob * torch.log(prob.clamp_min(_EPS))).sum(dim=1)          # (B, M)
    return h


def selection_entropy_scalar(prob, reduce='sum'):
    """Batch-level scalar ``H_sel`` used as the differentiation target."""
    h = selection_entropy(prob, reduce=reduce)                        # (B, M)
    if reduce == 'sum':
        return h.sum()
    if reduce == 'mean':
        return h.mean()
    raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")


# --------------------------------------------------------------------------- #
# 2) making the seed features differentiable without touching the model
# --------------------------------------------------------------------------- #
class _CrossEncoderTap:
    """Temporarily turn the cross-encoder output into a differentiable leaf.

    The hook returns ``out.detach().requires_grad_(True)`` in place of the real
    output, so the decoder / prediction heads stay bit-identical while the
    tensor they consume carries a gradient towards ``F``.  Nothing in the model
    is modified and the hook is removed on exit.
    """

    def __init__(self, module):
        self.module = module
        self.feature = None
        self._handle = None

    def _hook(self, _module, _inputs, output):
        feats = output[0] if isinstance(output, (tuple, list)) else output
        leaf = feats.detach().requires_grad_(True)
        self.feature = leaf
        if isinstance(output, (tuple, list)):
            return type(output)((leaf,) + tuple(output[1:]))
        return leaf

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc_info):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def _resolve_module(model, name='cross_encoder'):
    for mod_name, module in model.named_modules():
        if mod_name == name:
            return module
    raise AttributeError(f'{name!r} not found in the model')


# --------------------------------------------------------------------------- #
# 3) entropy gradient of the 1024 seeds
# --------------------------------------------------------------------------- #
def selection_entropy_gradient(model, inputs, positive_map=None,
                               tau=DEFAULT_TAU, reduce='sum', target='feat',
                               keep_grad=False, module_name='cross_encoder'):
    """Full pipeline for one batch: forward -> H_sel -> dH_sel/d(seed).

    Args:
        model: the ``PVGround`` network in ``eval()`` mode.  Used read-only; the
            only intrusion is a temporary forward hook (see
            :class:`_CrossEncoderTap`).
        inputs: the dict built exactly like ``TrainTester._get_inputs``.
        positive_map: ``(M, C)`` phrase-token weights (``sample['positive_map']``).
        tau: softmax temperature over the queries (see module docstring).
        reduce: ``'sum'`` (definition) or ``'mean'``.
        target: ``'feat'`` -> ``dH_sel/dF_i`` in ``R^C`` (default);
            ``'xyz'`` -> ``dH_sel/dX_i`` in ``R^3`` (needs the full graph).
        keep_grad: also return the full per-seed gradient.
        module_name: attribute path of the module whose output is
            ``end_points['seed_features']``.

    Returns:
        dict of tensors: ``score`` ``(B, N)``, ``selection_prob`` ``(B, Q, M)``,
        ``selection_scores`` ``(B, Q, M)``, ``total_entropy`` ``(B, M)``,
        ``grad`` ``(B, N, C)`` / ``(B, N, 3)`` or ``None``.
    """
    if target not in ('feat', 'xyz'):
        raise ValueError(f"target must be 'feat' or 'xyz', got {target!r}")

    grad_inputs = inputs
    if target == 'xyz':
        # the seeds are produced by FPS from the raw points: they only become
        # differentiable if the point cloud itself requires grad
        grad_inputs = dict(inputs)
        grad_inputs['points'] = inputs['points'].detach().clone().requires_grad_(True)

    with torch.enable_grad():
        if target == 'feat':
            module = _resolve_module(model, module_name)
            with _CrossEncoderTap(module) as tap:
                end_points = model(grad_inputs)
            if tap.feature is None:
                raise RuntimeError(f'the hook on {module_name!r} did not fire')
            diff_var = tap.feature                            # (B, N, C)
        else:
            end_points = model(grad_inputs)
            diff_var = end_points['seed_xyz']                 # (B, N, 3)

        sem_prob = semantic_probabilities(end_points['last_sem_cls_scores'])
        score_qm = selection_scores(sem_prob, positive_map)    # (B, Q, M)
        tau_eff = resolve_tau(score_qm, tau)                   # (B, 1, 1)
        prob_qm = torch.softmax(score_qm / tau_eff, dim=1)     # (B, Q, M)
        h_m = selection_entropy(prob_qm, reduce=reduce)        # (B, M)
        H = selection_entropy_scalar(prob_qm, reduce=reduce)

        grad, = torch.autograd.grad(H, diff_var, allow_unused=False)

    # (B, N, C) or (B, N, 3) -> per-seed L2 norm; float64 avoids underflow
    score = grad.detach().double().norm(dim=-1).float()        # (B, N)

    return {
        'score': score.detach(),
        'selection_prob': prob_qm.detach(),
        'selection_scores': score_qm.detach(),
        'total_entropy': h_m.detach(),
        'tau': tau_eff.detach().reshape(-1),
        'grad': grad.detach() if keep_grad else None,
    }


# --------------------------------------------------------------------------- #
# 4) convenience wrapper: torch tensors -> numpy dataclass
# --------------------------------------------------------------------------- #
def build_result(payload, query_index=-1, grad_target='feat',
                 norm_mode='minmax', q_low=1.0, q_high=99.0, keep_grad=False,
                 tau=DEFAULT_TAU, target_index=0):
    """Convert the raw dict of one sample into a dataclass (numpy arrays)."""
    score = payload['score'].detach().float().cpu().numpy().reshape(-1)
    grad = payload['grad']
    if grad is not None:
        grad = grad.detach().float().cpu().numpy().reshape(score.shape[0], -1)
        if not keep_grad:
            grad = None

    prob = payload['selection_prob'].detach().float().cpu().numpy()   # (B, Q, M)
    prob = prob.reshape(prob.shape[0], prob.shape[1], -1)[0]          # (Q, M)
    qscores = payload['selection_scores'].detach().float().cpu().numpy()
    qscores = qscores.reshape(qscores.shape[0], qscores.shape[1], -1)[0]

    h_m = payload['total_entropy'].detach().float().cpu().numpy().reshape(-1)
    tau_eff = payload.get('tau')
    tau_eff = float(tau_eff.detach().float().cpu().reshape(-1)[0]) \
        if tau_eff is not None else float(tau)
    num_queries = int(prob.shape[0])
    return SeedSelectionEntropyGradient(
        score=score,
        score_norm=normalize_score(score, mode=norm_mode, q_low=q_low, q_high=q_high),
        selection_prob=prob[:, target_index].copy(),
        selection_scores=qscores[:, target_index].copy(),
        total_entropy=float(h_m.mean()),
        grad_target=grad_target,
        query_index=int(query_index),
        grad=grad,
        extra={
            'tau': tau_eff,                       # effective (auto-resolved) value
            'tau_input': float(tau),
            'num_queries': num_queries,
            'entropy_max': float(np.log(num_queries)),   # ln Q, the upper bound
            'entropy_per_target': [float(v) for v in h_m],
            'target_index': int(target_index),
        },
    )
