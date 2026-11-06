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
"""

from typing import Dict, Optional

import numpy as np
import torch

try:  # scipy is already a dependency of the project (utils/box_util.py)
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


def _chunks(n, size):
    for start in range(0, n, size):
        yield start, min(start + size, n)


def roi_mask_from_centers(points_xyz, centers, radius, point_chunk=200_000):
    """Boolean mask of the points inside the union of ``K`` spheres.

    Args:
        points_xyz: ``(N, 3)`` float array (original point cloud coordinates).
        centers: ``(K, 3)`` float array (Top-K seed coordinates).
        radius: sphere radius in metres.

    Returns:
        ``(N,)`` bool array.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if centers.shape[0] == 0:
        return np.zeros(points_xyz.shape[0], dtype=bool)

    if cKDTree is not None:
        tree = cKDTree(points_xyz)
        hits = tree.query_ball_point(centers, r=float(radius))
        mask = np.zeros(points_xyz.shape[0], dtype=bool)
        for idx in hits:
            if len(idx):
                mask[np.asarray(idx, dtype=np.int64)] = True
        return mask

    # brute force fallback (chunked, avoids huge intermediates)
    mask = np.zeros(points_xyz.shape[0], dtype=bool)
    r2 = float(radius) ** 2
    for start, end in _chunks(points_xyz.shape[0], point_chunk):
        diff = points_xyz[start:end, None, :] - centers[None, :, :]
        d2 = np.einsum('nkd,nkd->nk', diff, diff)
        mask[start:end] = (d2 <= r2).any(axis=1)
    return mask


def sample_roi_points(points_xyz, centers, num_samples=512, radius=0.3,
                      seed=0, expand_factor=1.5, max_radius=2.0,
                      pad_with_replacement=True) -> Dict[str, object]:
    """Sample ``num_samples`` original points from the union of the seed balls.

    Args:
        points_xyz: ``(N, 3)`` original point cloud (never modified).
        centers: ``(K, 3)`` Top-K entropy-gradient seed coordinates.
        num_samples: number of points to return (512).
        radius: initial radius in metres (0.3).
        seed: RNG seed for the subsampling (reproducibility).
        expand_factor: radius multiplier used when too few points are inside.
        max_radius: stop growing the radius above this value.
        pad_with_replacement: if even ``max_radius`` is not enough, pad the
            sample with replacement instead of raising.

    Returns:
        dict with keys
            ``indices``       (S,) int64, original point indices (unique unless
                              ``padded``)
            ``points``        (S, 3) float32, the sampled coordinates
            ``radius_used``   float
            ``num_candidates`` int, size of the candidate pool after the last
                              radius increase
            ``num_expansions`` int
            ``padded``        bool
            ``mask``          (N,) bool, the ROI mask at ``radius_used``
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f'points_xyz must be (N, 3), got {points_xyz.shape}')
    if centers.shape[0] == 0:
        raise ValueError('centers is empty: cannot build any ROI')
    if num_samples <= 0:
        raise ValueError('num_samples must be positive')

    rng = np.random.default_rng(int(seed))

    radius_used = float(radius)
    num_expansions = 0
    mask = roi_mask_from_centers(points_xyz, centers, radius_used)
    num_candidates = int(mask.sum())

    while num_candidates < num_samples and radius_used < float(max_radius):
        radius_used = min(radius_used * float(expand_factor), float(max_radius))
        num_expansions += 1
        mask = roi_mask_from_centers(points_xyz, centers, radius_used)
        num_candidates = int(mask.sum())

    candidates = np.flatnonzero(mask)
    padded = False
    if num_candidates == 0:
        # Degenerate case (should not happen with a 50k cloud): fall back to the
        # points closest to the seed centres.
        d2 = ((points_xyz[:, None, :] - centers[None]) ** 2).sum(-1).min(1)
        candidates = np.argsort(d2, kind='stable')[:num_samples]
        padded = num_candidates == 0
    if num_candidates < num_samples:
        if not pad_with_replacement:
            raise RuntimeError(
                f'only {num_candidates} points inside the ROI (r={radius_used:.2f} m)')
        chosen = rng.choice(candidates, size=num_samples, replace=True)
        padded = True
    else:
        chosen = rng.choice(candidates, size=num_samples, replace=False)

    chosen = np.sort(chosen.astype(np.int64))
    return {
        'indices': chosen,
        'points': points_xyz[chosen].astype(np.float32),
        'radius_used': float(radius_used),
        'num_candidates': int(num_candidates),
        'num_expansions': int(num_expansions),
        'padded': bool(padded),
        'mask': mask,
    }


# --------------------------------------------------------------------------- #
# Torch / GPU version used inside the network (differentiable-free, batched)
# --------------------------------------------------------------------------- #
def _union_mask_torch(points_xyz, centers, radius):
    """``(B, N)`` bool mask of the points inside the union of ``K`` spheres."""
    d2 = torch.cdist(points_xyz.float(), centers.float())        # (B, N, K)
    return d2.min(dim=-1).values <= (float(radius) + 1e-4)


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
    """Batched GPU version of :func:`sample_roi_points` using **FPS**.

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
        radius, expand_factor, max_radius: see :func:`sample_roi_points`.

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
    r = float(radius)
    mask = _union_mask_torch(points_xyz, centers, r)
    counts = mask.sum(-1)
    num_expansions = 0
    r_max = float(max(max_radius, radius))
    while (not bool((counts >= num_samples).all())) and r < r_max - 1e-9:
        r = min(r * float(expand_factor), r_max)
        num_expansions += 1
        mask = _union_mask_torch(points_xyz, centers, r)
        counts = mask.sum(-1)
    radius_used = torch.full((batch_size,), r, dtype=dtype, device=device)

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
