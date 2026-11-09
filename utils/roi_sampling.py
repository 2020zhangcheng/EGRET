"""ROI re-partitioning driven by the Top-K entropy-gradient seeds.

Design (kept independent from the model, no GT box involved):

* Every selected seed ``c_m`` (m = 1..K, K = 32) defines a *spherical* local ROI
  ``B(c_m, r)`` with ``r = 0.3 m`` (``--roi_radius``).  The ROI is the **union**
  of the K balls, so overlapping balls are merged automatically (a point that
  belongs to several balls is kept once -- deduplication is implicit in the
  boolean mask / the set union of indices).
* Candidate points are the *original* point cloud points that fall inside the
  union; their indices in the original cloud are kept, which makes the sampled
  points traceable (``roi_point_indices.npy``).
* 512 points are drawn from the candidates without replacement, with a fixed
  RNG seed, and sorted so the output is deterministic.
* If the union contains fewer than 512 points the radius is grown
  geometrically (``r <- r * expand_factor``, default 1.5) until enough points
  are available or ``max_radius`` is reached; as a last resort the sample is
  padded with replacement (``padded=True``) instead of raising.  ``radius_used``
  is reported so the caller can log it.

The seed centres are the *real* Top-32 FPS seeds of the sample, never
re-sampled, and the GT box is not used anywhere in this module.

--------------------------------------------------------------------------
Second part of the module: *reliable* seed selection (see the section
"Seed reliability estimation" near the bottom).
--------------------------------------------------------------------------
Two rankings are supported:

* ``entropy``  -- the legacy one: the Top-K of the entropy-gradient norm
  ``s_i = ||dH_sel/dF_i||``, each seed defining a ball of radius
  ``--roi_radius``; the ROI is the union of those balls (kept for the
  visualisation and as the ablation reference);
* ``entropy_ball`` / ``field`` -- the recommended one: one **robust ball**
  around the weighted median of the seeds whose reliability score is the
  largest.  The score is either ``rank(s_i)`` (``entropy_ball``) or the
  box-proposal field ``rank(f_i)`` (``field``), where

      f_i = sum_m w_m * exp(-0.5 * ||(x_i - c_m) / (delta_m / 2)||^2)

  soft-contains the model's own predicted boxes (``proposal_seed_field``).
  ``f_i`` is the *region level* prior that the seed-level signals do not have:
  on ScanRefer val the entropy Top-32 ROI covers 34 % of the GT+1 m region,
  the box field 54 % (GT oracle 81 %).

The ROI geometry is: greedy NMS + mass stop -> coordinate-wise **weighted
median** centre (breakdown point 50 %, a single outlier seed cannot drag it)
-> weighted ``q``-quantile radius + margin -> one ball -> FPS.  A single ball
(not the union of K balls) merges the reliable seeds of different object parts
into one region.
"""

import torch

try:  # scipy is already a dependency of the project (utils/box_util.py)
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _union_mask_torch(points_xyz, centers, radius):
    """``(B, N)`` bool mask of the points inside the union of ``K`` spheres.

    ``radius`` is either a python float (same radius for every mode) or a
    ``(B,)`` / ``(B, K)`` tensor: per-sample / per-centre radii, which is what
    the adaptive (reliability driven) ROI needs.
    """
    dist = torch.cdist(points_xyz.float(), centers.float())       # (B, N, K)
    if torch.is_tensor(radius):
        r = radius.float().to(dist.device)
        if r.dim() == 1:                                          # (B,) -> (B, 1, 1)
            return dist.min(dim=-1).values <= (r.view(-1, 1) + 1e-4)
        if r.dim() == 2:                                          # (B, K)
            return (dist <= (r.unsqueeze(1) + 1e-4)).any(dim=-1)
        if r.dim() == 3:                                          # (B, 1, K)
            return (dist <= (r + 1e-4)).any(dim=-1)
        raise ValueError(f'unsupported radius shape {tuple(r.shape)}')
    return dist.min(dim=-1).values <= (float(radius) + 1e-4)


def _fps_torch_single(xyz, npoint):
    """Plain farthest point sampling on a single ``(M, 3)`` tensor (no CUDA ext)."""
    m = xyz.shape[0]
    npoint = int(min(npoint, m))
    idx = torch.zeros(npoint, dtype=torch.long, device=xyz.device)
    dist = torch.full((m,), 1e10, dtype=xyz.dtype, device=xyz.device)
    farthest = 0
    for i in range(npoint):
        idx[i] = farthest
        d = ((xyz - xyz[farthest]) ** 2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = int(dist.argmax())
    return idx


def _fps_batched(cand_xyz, batch_cnt, npoint):
    """FPS over a concatenation of per-sample candidate sets.

    Args:
        cand_xyz: ``(M, 3)`` concatenated candidates (grouped by sample).
        batch_cnt: ``(B,)`` number of candidates of each sample.
        npoint: ``(B,)`` number of points to sample per sample (<= batch_cnt).

    Returns:
        ``(sum(npoint),)`` long tensor of **global** indices into ``cand_xyz``.
    """
    if cand_xyz.is_cuda:
        try:
            from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as p2_stack

            sel = p2_stack.stack_farthest_point_sample(
                cand_xyz.float().contiguous(), batch_cnt.int(), npoint.int())
            return sel.long()
        except Exception:  # pragma: no cover - fall back to the pure torch version
            pass

    out, offset = [], 0
    for b in range(int(batch_cnt.numel())):
        nb = int(npoint[b])
        if nb <= 0:
            continue
        local = _fps_torch_single(cand_xyz[offset:offset + int(batch_cnt[b])], nb)
        out.append(local + offset)
        offset += int(batch_cnt[b])
    if not out:
        return torch.zeros(0, dtype=torch.long, device=cand_xyz.device)
    return torch.cat(out)


def sample_roi_points_torch(points_xyz, centers, num_samples=512, radius=0.3,
                            expand_factor=1.5, max_radius=2.0):
    """Batched GPU version of the ROI sampling (**FPS**, exactly 512 points).

    Semantics (identical to the numpy version, only the sub-sampling differs):
    the ROI is the *union* of the ``K`` balls ``B(c_k, r)`` so an overlapping
    point is kept exactly once (implicit de-duplication through the boolean
    mask / index set), the radius grows geometrically to ``max_radius`` when
    fewer than ``num_samples`` points are available, and the output is always
    exactly ``num_samples`` points per sample (with replacement as a last
    resort, flagged in ``padded``).

    Args:
        points_xyz: ``(B, N, 3)`` original cloud (same device as ``centers``).
        centers: ``(B, K, 3)`` Top-K seed coordinates.
        num_samples: points to return per sample (512).
        radius: float or ``(B,)`` tensor (per-sample adaptive radius).
        expand_factor / max_radius: geometric growth when a ROI holds
            fewer than ``num_samples`` points.

    Returns:
        dict with ``indices`` ``(B, S)`` long (indices into the original cloud,
        duplicated only when ``padded``), ``points`` ``(B, S, 3)``,
        ``num_candidates`` ``(B,)``, ``radius_used`` ``(B,)``,
        ``num_expansions`` int, ``padded`` ``(B,)`` bool, ``mask`` ``(B, N)``.
    """
    if not torch.is_tensor(points_xyz) or not torch.is_tensor(centers):
        raise TypeError('sample_roi_points_torch expects torch tensors')
    if points_xyz.dim() != 3 or points_xyz.shape[-1] != 3:
        raise ValueError(f'points_xyz must be (B, N, 3), got {tuple(points_xyz.shape)}')
    if centers.dim() != 3 or centers.shape[-1] != 3:
        raise ValueError(f'centers must be (B, K, 3), got {tuple(centers.shape)}')
    if centers.shape[0] != points_xyz.shape[0]:
        raise ValueError('batch size mismatch between points_xyz and centers')
    if centers.shape[1] == 0:
        raise ValueError('centers is empty: cannot build any ROI')
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')

    device = points_xyz.device
    batch_size, num_points = points_xyz.shape[0], points_xyz.shape[1]
    dtype = points_xyz.dtype

    # ---- radius growth until enough candidates are available --------------
    # ``radius`` is either a float (same ball for every sample) or a ``(B,)``
    # tensor (per-sample adaptive radius of the robust single ball).
    r = torch.as_tensor(radius, dtype=torch.float32, device=device).reshape(-1)
    if r.numel() == 1:
        r = r.expand(batch_size).clone()
    if r.numel() != batch_size:
        raise ValueError(f'{r.numel()} radii cannot be broadcast to a batch of '
                         f'{batch_size}')
    r_max = torch.as_tensor(max_radius, dtype=torch.float32,
                            device=device).reshape(-1)
    if r_max.numel() == 1:
        r_max = r_max.expand(batch_size).clone()
    r_max = torch.maximum(r_max, r)
    mask = _union_mask_torch(points_xyz, centers, r)
    counts = mask.sum(-1)
    num_expansions = 0
    while (not bool((counts >= num_samples).all())) and \
            bool((r < r_max - 1e-9).any()) and num_expansions < 32:
        r = torch.minimum(r * float(expand_factor), r_max)
        num_expansions += 1
        mask = _union_mask_torch(points_xyz, centers, r)
        counts = mask.sum(-1)
    radius_used = r.to(dtype)

    # ---- degenerate case: no candidate at all -> nearest cloud point ------
    empty = counts < 1
    if bool(empty.any()):
        near = torch.cdist(points_xyz[empty].float(), centers[empty].float())
        near = near.min(-1).values.argmin(-1)
        mask = mask.clone()
        mask[empty] = False
        mask[empty, near] = True
        counts = mask.sum(-1)

    # ---- gather the candidate pool and run one batched FPS ----------------
    bs_idx, pt_idx = mask.nonzero(as_tuple=True)
    cand_xyz = points_xyz[bs_idx, pt_idx].contiguous()
    npoint = counts.clamp(max=num_samples).int()
    sel = _fps_batched(cand_xyz, counts.int(), npoint)
    sel_pt = pt_idx[sel]

    # ---- pad to a fixed length (only when fewer than num_samples exist) ---
    indices = torch.zeros((batch_size, num_samples), dtype=torch.long, device=device)
    padded = torch.zeros(batch_size, dtype=torch.bool, device=device)
    offset = 0
    for b in range(batch_size):
        nb = int(npoint[b])
        cur = sel_pt[offset:offset + nb]
        offset += nb
        if nb == 0:                      # cannot happen after the degenerate fix
            cur = torch.zeros(1, dtype=torch.long, device=device)
            nb = 1
        if nb < num_samples:
            reps = (num_samples + nb - 1) // nb
            cur = cur.repeat(reps)[:num_samples]
            padded[b] = True
        indices[b] = cur[:num_samples]

    points = points_xyz.gather(
        1, indices.unsqueeze(-1).expand(batch_size, num_samples, 3))

    if points.shape[1] != num_samples or points.shape[2] != 3:
        raise RuntimeError(f'unexpected ROI tensor shape {tuple(points.shape)}')

    return {
        'indices': indices,
        'points': points.contiguous(),
        'num_candidates': counts,
        'radius_used': radius_used,
        'num_expansions': int(num_expansions),
        'padded': padded,
        'mask': mask,
    }


# --------------------------------------------------------------------------- #
# Seed reliability estimation
# --------------------------------------------------------------------------- #
#: Available ROI rankings.  ``entropy`` is the legacy multi-ball geometry (kept
#: for the visualisation / ablation), ``entropy_ball`` uses only the
#: entropy-gradient norm and ``field`` the box-proposal field; the latter two
#: share the robust single-ball geometry.
SELECTORS = ('entropy', 'entropy_ball', 'field')


def rank_normalize(x):
    """Empirical CDF of ``x`` along the last dim -> ``(0, 1]``.

    Every per-sample score enters the fusion through this transform: the raw
    ``||dH/dF||`` (and the mask logits) have a heavy, *scene dependent* tail,
    so a fixed Top-K / threshold on the raw value means a different thing on
    every scene.  The ECDF is scale free, keeps the ranking, and gives the
    fusion bounded factors that can be multiplied without weight tuning.
    """
    n = int(x.shape[-1])
    order = torch.argsort(x.float(), dim=-1)
    ranks = torch.empty_like(order)
    ranks.scatter_(-1, order, torch.arange(n, device=x.device).expand_as(order))
    return (ranks.to(torch.float32) + 0.5) / float(n)


# --------------------------------------------------------------------------- #
# Reliable ROI: weighted median centre + adaptive radius (single ball)
# --------------------------------------------------------------------------- #
def proposal_seed_field(xyz, centers, sizes, probs, top_m=8, eps=1e-3):
    """Soft-containment field of the model's own box predictions ``(B, N)``.

    The 1024 seeds are *not* a good localization prior (see the diagnostics),
    while the decoder already regresses ``Q = 256`` boxes and ranks them by
    ``P(q|m)``.  This function turns the ``M`` best text-matched boxes into a
    continuous, non-committal field over the seeds:

        kappa_im = exp(-1/2 * sum_a ((x_i,a - c_m,a) / (s_m,a / 2))^2)
        field_i  = sum_m  w_m * kappa_im / sum_m w_m ,   w_m = P(q_m | m)

    i.e. a Gaussian mixture whose components are the predicted boxes
    (Mahalanobis distance in *box units*, so a large object has a large
    support).  Taking ``M > 1`` weighted hypotheses -- instead of the single
    arg-max box -- is the reliability gate of this term: it is the EATA/SAR
    "do not trust one high-variance sample" idea applied to box hypotheses.

    Args:
        xyz: ``(B, N, 3)`` seed coordinates.
        centers: ``(B, Q, 3)`` predicted box centres (``last_center``).
        sizes: ``(B, Q, 3)`` predicted box sizes (``last_pred_size``).
        probs: ``(B, Q)`` query selection probability ``P(q|m)``.
        top_m: number of box hypotheses kept.

    Returns:
        ``(B, N)`` float32 in ``(0, 1]``.
    """
    B, N, _ = xyz.shape
    Q = probs.size(1)
    m = int(min(max(top_m, 1), Q))
    w, top_idx = probs.float().topk(m, dim=1)                     # (B, M)
    w = w / w.sum(-1, keepdim=True).clamp_min(1e-8)
    gather = top_idx.unsqueeze(-1).expand(-1, -1, 3)
    c = torch.gather(centers.float(), 1, gather)                  # (B, M, 3)
    s = torch.gather(sizes.float(), 1, gather).abs()
    half = (s * 0.5).clamp_min(eps)                               # (B, M, 3)
    diff = (xyz.float().unsqueeze(2) - c.unsqueeze(1)) / half.unsqueeze(1)
    q = (diff ** 2).sum(-1)                                       # (B, N, M)
    kappa = torch.exp(-0.5 * q)
    return (kappa * w.unsqueeze(1)).sum(-1)                       # (B, N)


def _weighted_quantile(values, weights, q):
    """Per-row weighted ``q``-quantile along the last dim -> ``(B,)``."""
    v, idx = torch.sort(values.float(), dim=-1)
    w = torch.gather(weights.float(), -1, idx)
    cw = torch.cumsum(w, dim=-1)
    cw = cw / cw[..., -1:].clamp_min(1e-8)
    pos = (cw >= float(q)).float().argmax(dim=-1)
    return torch.gather(v, -1, pos.unsqueeze(-1)).squeeze(-1)


def weighted_median(xyz, weight):
    """Coordinate-wise weighted median of ``(B, K, 3)`` points -> ``(B, 3)``.

    The median (breakdown point 50%) is what makes the ROI centre immune to a
    single wrong seed: moving one of the ``K`` selected seeds to a distractor
    cannot move the centre outside the support of the remaining ones, whereas
    the mean would be dragged by it.
    """
    return torch.stack([
        _weighted_quantile(xyz[..., a], weight, 0.5) for a in range(3)
    ], dim=-1)


def select_reliable_roi(xyz, score, k_max=32, k_min=4, nms_radius=0.5,
                        mass_frac=0.9, score_ratio=0.3, radius_q=0.9,
                        radius_margin=0.35, radius_min=0.4, radius_max=1.6):
    """Reliable seeds -> one ROI centre per sample + an adaptive radius.

    1. **Greedy NMS**: seeds are visited by decreasing ``R`` and accepted only
       if they are at least ``nms_radius`` away from every accepted seed, so
       the set ``S`` covers the reliable *region* instead of 32 copies of the
       same spot.
    2. **Adaptive stop**: stop at ``k_max``, or once ``|S| >= k_min`` and the
       accepted mass reaches ``mass_frac`` of the total mass, or when the next
       score falls below ``score_ratio * max R`` (EATA-style relative
       threshold: no absolute value is assumed, the threshold is defined by the
       sample's own distribution).
    3. **Robust centre**: coordinate-wise weighted median of ``S``.
    4. **Adaptive radius**: weighted ``radius_q`` quantile of
       ``||x_j - c||`` over ``S`` plus ``radius_margin``, clipped to
       ``[radius_min, radius_max]`` -- the extent of the *reliable seed set*
       (it saturates at ``radius_max`` whenever the score field is diffuse).

    Returns:
        dict with ``centers`` ``(B, 1, 3)``, ``radii`` ``(B,)``, ``sel_idx``
        ``(B, k_max)`` (``-1`` = padding), ``sel_w`` ``(B, k_max)``,
        ``num_sel`` ``(B,)``.
    """
    if xyz.dim() != 3 or xyz.shape[-1] != 3:
        raise ValueError(f'xyz must be (B, N, 3), got {tuple(xyz.shape)}')
    B, N, _ = xyz.shape
    k_max = int(min(k_max, N))
    xyz = xyz.float()
    score = score.float()

    centers = xyz.new_zeros(B, 3)
    radii = xyz.new_zeros(B)
    sel_idx = torch.full((B, k_max), -1, dtype=torch.long, device=xyz.device)
    sel_w = xyz.new_zeros(B, k_max)
    num_sel = torch.zeros(B, dtype=torch.long, device=xyz.device)

    for b in range(B):
        s = score[b].detach()
        order = torch.argsort(s, descending=True)
        s_max = float(s[order[0]])
        total = float(s.sum().clamp_min(1e-8))
        accepted, mass = [], 0.0
        for j in order.tolist():
            if len(accepted) >= k_max:
                break
            sj = float(s[j])
            if sj <= 0.0:
                break
            if len(accepted) >= k_min and sj < score_ratio * s_max:
                break
            if accepted:
                d = (xyz[b, accepted] - xyz[b, j]).norm(dim=-1).min()
                if float(d) <= nms_radius:
                    continue
            accepted.append(j)
            mass += sj
            if len(accepted) >= k_min and mass / total >= mass_frac:
                break
        if not accepted:
            accepted = [int(order[0])]

        idx = torch.as_tensor(accepted, dtype=torch.long, device=xyz.device)
        w = s[idx].clamp_min(1e-8)
        pts = xyz[b][idx]                                       # (K, 3)
        center = weighted_median(pts.unsqueeze(0), w.unsqueeze(0))[0]

        d = (pts - center).norm(dim=-1)
        r = float(_weighted_quantile(d.unsqueeze(0), w.unsqueeze(0), radius_q))
        r = min(max(r + float(radius_margin), float(radius_min)),
                float(radius_max))

        k = len(accepted)
        centers[b] = center
        radii[b] = r
        sel_idx[b, :k] = idx
        sel_w[b, :k] = w
        num_sel[b] = k

    return {
        'centers': centers.unsqueeze(1),          # (B, 1, 3) -> one ball
        'radii': radii,                           # (B,)
        'sel_idx': sel_idx,
        'sel_w': sel_w,
        'num_sel': num_sel,
    }
