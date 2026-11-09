#!/usr/bin/env python
"""Visualise the ROI candidate selection of a PVGround checkpoint.

Six panels, all produced from **real forward passes** of the checkpoint (the
script never re-samples anything: every point set it draws is read verbatim
from ``end_points``):

=====  ==================================================================
(a)    scene cloud (RGB) + GT box (green) + predicted box (red)
(b)    the 1024 FPS seeds coloured by the entropy-gradient score
       ``s_i = ||dH_sel/dF_i||_2``                        (the existing signal)
(c)    the 1024 FPS seeds coloured by the box-proposal field ``f_i``
       (the model's own boxes, soft-contained)             (the region prior)
(d)    the **multi-ball** geometry: Top-K seeds + the K spheres of radius
       ``--roi_radius`` + the points actually sampled from the union
(e)    the **robust single ball**: weighted-median centre + adaptive radius
       + the points actually sampled from it (+ GT box grown by 1 m)
(f)    the two sampled point sets side by side, with the GT box +1 m
=====  ==================================================================

Two forward passes are run per sample, both with the *same* inputs:

* ``roi_selector='entropy'`` -> the original multi-ball ROI (panel d);
* ``roi_selector='field'``   -> the recommended robust single-ball ROI
  (panels c, e, f).

Everything drawn comes from ``end_points`` (``roi_entropy_scores``,
``roi_field_rank``, ``roi_reliability``, ``roi_seed_indices``,
``roi_centers``, ``roi_radii``, ``roi_points_xyz``, ``roi_point_indices``,
``last_center``, ``last_pred_size``, ...).  The model weights, its
hyper-parameters and the point cloud are untouched, and no GT is used to
*select* anything -- the GT box is only drawn / logged for reference.

Usage
-----
conda activate 3dgrounding
python tools/visualize_roi_points.py \
    --checkpoint_path data/PV-Ground_ScanRefer_single.pth \
    --data_root data/ --output_dir visualization_results \
    --num_samples 20 --roi_points 256
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np                                        # noqa: E402
import torch                                              # noqa: E402
from torch.utils.data import DataLoader                    # noqa: E402

import matplotlib                                          # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt                            # noqa: E402
from matplotlib.lines import Line2D                        # noqa: E402

from main_utils import parse_option                        # noqa: E402
from pcdet.config import cfg as model_cfg                  # noqa: E402
from pcdet.config import cfg_from_yaml_file                # noqa: E402
from prepare_data import DataProcessor                     # noqa: E402
from train_dist_mod import TrainTester                     # noqa: E402
from utils.roi_visualization import (                      # noqa: E402
    Panel, autofit_camera, box_corners, build_score_colorizer, calibrate_tan_v,
    camera_from_points, expand_box, render_panel,
)

# --------------------------------------------------------------------------- #
# colours
# --------------------------------------------------------------------------- #
GT_GREEN = (0.05, 0.65, 0.10)
PRED_RED = (0.85, 0.10, 0.10)
BALL_ORANGE = (0.95, 0.55, 0.05)
ROI_ORANGE = (0.90, 0.45, 0.05)
BALL_BLUE = (0.05, 0.25, 0.85)
ROI_BLUE = (0.05, 0.35, 0.95)
SCENE_GREY = (0.74, 0.74, 0.74)
DEFAULT_SCENE_RGB = (0.62, 0.62, 0.62)

#: defaults of ``tools/test_scanrefer_single.sh`` + the ROI switches used here
EVAL_PRESET = {
    'model': 'PVGround',
    'num_target': 256,
    'num_decoder_layers': 6,
    'self_position_embedding': 'loc_learned',
    'self_attend': True,
    'use_color': True,
    'use_height': False,
    'use_multiview': False,
    'use_soft_token_loss': True,
    'use_contrastive_align': True,
    'detect_intermediate': True,
    'joint_det': True,
    'augment_det': True,
    'butd': False,
    'butd_gt': False,
    'butd_cls': False,
    'debug': False,
    'eval': True,
    'eval_train': False,
    'dataset': ['scanrefer'],
    'test_dataset': 'scanrefer',
    'batch_size': 1,
    # ROI branch (must stay on: it produces everything that is visualised)
    'roi': True,
    'roi_points': 256,
    'roi_topk': 32,
    'roi_radius': 0.4,
    'roi_max_radius': 2.0,
    'roi_tau': 0.0,
    'roi_selector': 'entropy',
    'roi_k_min': 4,
    'roi_nms_radius': 0.5,
    'roi_mass_frac': 0.9,
    'roi_score_ratio': 0.3,
    'roi_radius_q': 0.9,
    'roi_radius_margin': 0.35,
    'roi_radius_min': 0.4,
    'roi_radius_max': 2.0,
    'roi_topm': 4,
}

COORDINATE_SYSTEM = (
    "ScanNet axis-aligned raw coordinates in metres, identical for "
    "inputs['points'][..., 1:4], end_points['coords'], end_points['seed_xyz'], "
    "end_points['roi_points_xyz'], center_label/size_gts and last_center/"
    "last_pred_size (no centring, no scaling). Boxes are [cx, cy, cz, dx, dy, "
    "dz] with dx/dy/dz the extents along x/y/z and heading = 0 rad."
)


def expand_radii(centres, radii):
    """Broadcast the radii of a geometry to one value per centre."""
    r = np.asarray(radii, dtype=np.float64).reshape(-1)
    n = int(np.asarray(centres).shape[0])
    if r.size == n:
        return r
    if r.size == 1:
        return np.full(n, float(r[0]))
    raise ValueError(f'{r.size} radii cannot be broadcast to {n} centres')


def sphere_wireframe(center, radius, n=48):
    """Three orthogonal great circles of a sphere, as ``(3n, 3)`` points."""
    t = np.linspace(0.0, 2.0 * np.pi, int(max(n, 4)), endpoint=False)
    c, s = np.cos(t), np.sin(t)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    out = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        pts = np.zeros((t.size, 3), dtype=np.float64)
        pts[:, a], pts[:, b] = c, s
        out.append(center + float(radius) * pts)
    return np.concatenate(out, axis=0)


class _Pipeline(TrainTester):
    """``TrainTester`` without the DDP logger / tensorboard / log-dir setup."""

    def __init__(self, cfg, data_processor):    # noqa: D107 - no super().__init__
        self.model_cfg = cfg
        self.data_processor = data_processor
        self.logger = None
        self.tensorboard = None


# --------------------------------------------------------------------------- #
# command line
# --------------------------------------------------------------------------- #
def parse_cli():
    p = argparse.ArgumentParser(
        description='Visualise the ROI candidate selection of PVGround '
                    '(read-only, real forward passes).',
        allow_abbrev=False,
    )
    p.add_argument('--config', default='wandb_config.yaml')
    p.add_argument('--checkpoint_path', default='data/PV-Ground_ScanRefer_single.pth')
    p.add_argument('--data_root', default='data/')
    p.add_argument('--output_dir', default='visualization_results')
    p.add_argument('--num_samples', type=int, default=20,
                   help='number of *consecutive* test samples to visualise')
    p.add_argument('--start_index', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--panel_width', type=int, default=760)
    p.add_argument('--panel_height', type=int, default=600)
    p.add_argument('--scene_point_size', type=float, default=1.4)
    p.add_argument('--scene_max_points', type=int, default=50000,
                   help='max background points drawn per panel (overlays are '
                        'never subsampled); 0 = all points')
    p.add_argument('--cmap', default='coolwarm')
    p.add_argument('--score_norm', default='log',
                   choices=['minmax', 'robust', 'log', 'log_robust'])
    p.add_argument('--q_low', type=float, default=1.0)
    p.add_argument('--q_high', type=float, default=99.0)
    p.add_argument('--camera_dir', type=float, nargs=3,
                   default=[0.80, -1.0, 1.5])
    p.add_argument('--up', type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument('--ball_rings', type=int, default=32,
                   help='points per great circle of a drawn sphere')
    p.add_argument('--seed', type=int, default=111)
    p.add_argument('--roi_points', type=int, default=None,
                   help='ROI points per geometry (default: 256)')
    p.add_argument('--roi_topk', type=int, default=None)
    p.add_argument('--roi_radius', type=float, default=None,
                   help='fixed radius of the multi-ball geometry')
    p.add_argument('--roi_radius_max', type=float, default=None,
                   help='cap of the adaptive radius of the single-ball geometry')
    p.add_argument('--roi_topm', type=int, default=None)
    return p.parse_known_args()[0]


def build_run_args(cli):
    """``parse_option()`` namespace with the evaluation-script defaults."""
    args = parse_option()
    given = {a[2:].split('=')[0].replace('-', '_')
             for a in sys.argv[1:] if a.startswith('--')}
    for key, value in EVAL_PRESET.items():
        if key not in given:
            setattr(args, key, value)
    for key in ('roi_points', 'roi_topk', 'roi_radius', 'roi_radius_max',
                'roi_topm'):
        value = getattr(cli, key, None)
        if value is not None:
            setattr(args, key, value)
    args.data_root = cli.data_root
    args.checkpoint_path = cli.checkpoint_path
    args.num_workers = cli.num_workers
    args.eval = True
    args.eval_train = False
    return args


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def to_numpy(x, dtype=np.float32):
    if torch.is_tensor(x):
        return x.detach().to(torch.float32).cpu().numpy().astype(dtype, copy=False)
    return np.asarray(x, dtype=dtype)


def as_scalar(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return float(x.detach().float().cpu().reshape(-1)[0])
    return float(np.asarray(x).reshape(-1)[0])


def box_iou_axis_aligned(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_min, a_max = a[:3] - a[3:] / 2.0, a[:3] + a[3:] / 2.0
    b_min, b_max = b[:3] - b[3:] / 2.0, b[:3] + b[3:] / 2.0
    inter = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0.0, None)
    inter_vol = float(np.prod(inter))
    vol_a = float(np.prod(np.clip(a[3:], 0, None)))
    vol_b = float(np.prod(np.clip(b[3:], 0, None)))
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 1e-12 else 0.0


def _strip_ddp_prefix(state):
    out = {}
    for key, value in state.items():
        while key.startswith('module.'):
            key = key[len('module.'):]
        out[key] = value
    return out


def load_model_weights(args, model):
    """Load a checkpoint into the *unwrapped* model (single-process use)."""
    ckpt_path = args.checkpoint_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = _strip_ddp_prefix(ckpt['model'])
    epoch = ckpt.get('epoch')
    del ckpt
    keys = set(model.state_dict().keys())
    matched = len(keys & set(state.keys()))
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if 'position_ids' not in k]
    unexpected = [k for k in result.unexpected_keys if 'position_ids' not in k]
    print(f"=> loaded '{ckpt_path}' (epoch {epoch}): {matched} matched, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if matched < 0.9 * len(keys):
        raise RuntimeError(
            f'only {matched}/{len(keys)} parameters loaded from {ckpt_path} - '
            f'refusing to visualise a partly random network')
    return model


def pick_query(end_points, batch_data, b=0):
    """Query index of the predicted box, using the *evaluator* criterion.

    ``evaluate_one_epoch`` (and ``GroundingEvaluator.evaluate_bbox_by_pos_align``)
    scores a query with::

        prob   = softmax_l( proj_q . proj_t / 0.07 )      # over text tokens
        score  = sum_l prob[:, l] * pmap[l]                # phrase-token mass

    This is *not* the same as the query ranking of the soft-token
    classification head: using the head selects a very different (much worse)
    box, which is why the criterion is replicated here.  Only text information
    is used -- no ground truth.
    """
    proj_q = end_points['last_proj_queries'][b].float()          # (Q, 64)
    proj_t = end_points['proj_tokens'][b].float()                # (L, 64)
    prob = ((proj_q @ proj_t.t()) / 0.07).softmax(dim=-1)        # (Q, L)
    L = prob.shape[-1]

    def phrase(key):
        tensor = batch_data.get(key, None)
        if tensor is None or not torch.is_tensor(tensor):
            return None
        value = tensor[b, 0].detach().float().reshape(-1)[:256]
        if value.numel() < L:
            return None
        return value

    def term(weights):
        if weights is None:
            return prob.new_zeros(prob.shape[0])
        return (prob[:, :L] * weights[:L].unsqueeze(0)).sum(-1)

    main = phrase('positive_map')
    if main is None:
        scores = prob.max(dim=-1).values
    else:
        scores = term(main)
        for key, sign in (('modify_positive_map', 1.0), ('pron_positive_map', 1.0),
                          ('rel_positive_map', 1.0), ('other_entity_map', -1.0)):
            weights = phrase(key)
            if weights is not None and float(weights.sum()) > 0:
                scores = scores + sign * term(weights)
    return int(scores.argmax()), scores.detach().cpu().numpy()


# --------------------------------------------------------------------------- #
# forward + extraction
# --------------------------------------------------------------------------- #
def collect_geometry(model, args, pipeline, batch_data, selector):
    """Run **one** real forward pass with the given ``roi_selector``."""
    inputs = pipeline._get_inputs(batch_data)      # the forward mutates inputs
    with torch.no_grad():
        end_points = model(inputs)
    b = 0
    geo = {
        'selector': selector,
        'seed_xyz': to_numpy(end_points['seed_xyz'][b]),
        'seed_inds': to_numpy(end_points['seed_inds'][b], np.int64),
        'roi_points_xyz': to_numpy(end_points['roi_points_xyz'][b]),
        'roi_point_indices': to_numpy(end_points['roi_point_indices'][b], np.int64),
        'roi_seed_indices': to_numpy(end_points['roi_seed_indices'][b], np.int64),
        'roi_centers': to_numpy(end_points['roi_centers'][b]),
        'roi_radii': to_numpy(end_points['roi_radii'][b], np.float64).reshape(-1),
        'roi_radius_used': as_scalar(end_points['roi_radius_used'][b]),
        'roi_num_candidates': int(as_scalar(end_points['roi_num_candidates'][b])),
        'num_vis_tokens': int(end_points.get('num_vis_tokens', -1)),
        'pred_center': to_numpy(end_points['last_center'][b]),
        'pred_size': to_numpy(end_points['last_pred_size'][b]),
    }
    for key in ('roi_entropy_scores', 'roi_field_rank', 'roi_reliability'):
        if key in end_points:
            geo[key] = to_numpy(end_points[key][b])
    geo['roi_radii_per_centre'] = expand_radii(geo['roi_centers'],
                                               geo['roi_radii'])
    geo['query_index'], geo['query_scores'] = pick_query(end_points, batch_data, b)
    return geo


def collect_sample(model, args, pipeline, batch_data, dataset_index, sample_id,
                   mean_rgb=None):
    """Both geometries of one sample (multi-ball + robust single ball)."""
    original_selector = model.roi_selector
    warnings = []
    try:
        model.roi_selector = 'entropy'
        union = collect_geometry(model, args, pipeline, batch_data, 'entropy')
        model.roi_selector = 'field'
        robust = collect_geometry(model, args, pipeline, batch_data, 'field')
    finally:
        model.roi_selector = original_selector

    b = 0
    inputs = pipeline._get_inputs(batch_data)      # rebuilt: the forward mutates it
    pts_all = inputs['points']
    original_xyz = to_numpy(pts_all[:, 1:4])
    original_rgb = None
    if getattr(args, 'use_color', False) and pts_all.shape[1] >= 7:
        if mean_rgb is None:
            raise RuntimeError('dataset mean_rgb was not set')
        original_rgb = np.clip(to_numpy(pts_all[:, 4:7]) + mean_rgb[None, :], 0.0, 1.0)
    else:
        warnings.append('dataset has no RGB for this run: grey cloud is used')

    gt_center = to_numpy(batch_data['center_label'][b, 0, :3])
    gt_size = to_numpy(batch_data['size_gts'][b, 0, :3])
    gt_box = np.concatenate([gt_center, gt_size]).astype(np.float32)
    exp_center, exp_size = expand_box(gt_center, gt_size, delta=1.0)
    gt_box_exp = np.concatenate([exp_center, exp_size]).astype(np.float32)

    query_index = robust['query_index']
    pred_center = robust['pred_center'][query_index]
    pred_size = np.clip(robust['pred_size'][query_index], 1e-6, None)
    pred_box = np.concatenate([pred_center, pred_size]).astype(np.float32)

    # ---- self-checks (report only, never modify the data) ------------------
    n_points = original_xyz.shape[0]
    for tag, geo in (('multi-ball', union), ('robust-ball', robust)):
        if geo['roi_points_xyz'].shape[0] != int(model.roi_num_points):
            raise RuntimeError(f'{tag}: expected {model.roi_num_points} ROI '
                               f'points, got {geo["roi_points_xyz"].shape[0]}')
        idx = geo['roi_point_indices']
        if idx.size and 0 <= idx.min() and idx.max() < n_points:
            err = float(np.abs(original_xyz[idx] - geo['roi_points_xyz']).max())
            if err > 1e-4:
                warnings.append(f'{tag}: roi indices != cloud points ({err:.3g})')
        else:
            warnings.append(f'{tag}: roi indices out of range')
        seed_idx = geo['seed_inds']
        if seed_idx.size and 0 <= seed_idx.min() and seed_idx.max() < n_points:
            err = float(np.abs(original_xyz[seed_idx] - geo['seed_xyz']).max())
            if err > 1e-4:
                warnings.append(f'{tag}: seed_xyz != coords[seed_inds] ({err:.3g})')
        if geo['roi_radii_per_centre'].size != geo['roi_centers'].shape[0]:
            warnings.append(f'{tag}: centres / radii mismatch')

    if 'roi_entropy_scores' not in union:
        raise KeyError('roi_entropy_scores missing from the multi-ball pass')
    if 'roi_field_rank' not in robust:
        raise KeyError('roi_field_rank missing from the box-field pass')

    return {
        'sample_id': sample_id,
        'dataset_index': int(dataset_index),
        'scan_id': str(batch_data['scan_ids'][b]),
        'utterance': str(batch_data['utterances'][b]),
        'target_name': str(batch_data['target_name'][b]),
        'target_id': int(batch_data['target_id'][b]),
        'original_xyz': original_xyz.astype(np.float32),
        'original_rgb': None if original_rgb is None else original_rgb.astype(np.float32),
        'gt_box': gt_box,
        'gt_box_expanded': gt_box_exp,
        'pred_box': pred_box,
        'union': union,
        'robust': robust,
        'iou_3d': float(box_iou_axis_aligned(gt_box, pred_box)),
        'warnings': warnings,
    }


# --------------------------------------------------------------------------- #
# saving + verification
# --------------------------------------------------------------------------- #
SAVED_GEO_KEYS = (
    'seed_xyz', 'seed_inds', 'roi_points_xyz', 'roi_point_indices',
    'roi_seed_indices', 'roi_centers', 'roi_radii', 'roi_radii_per_centre',
    'roi_entropy_scores', 'roi_field_rank', 'roi_reliability',
)


def save_payload(out_dir, payload):
    sample_dir = os.path.join(out_dir, payload['sample_id'])
    os.makedirs(sample_dir, exist_ok=True)
    np.save(os.path.join(sample_dir, 'original_xyz.npy'), payload['original_xyz'])
    if payload['original_rgb'] is not None:
        np.save(os.path.join(sample_dir, 'original_rgb.npy'), payload['original_rgb'])
    for key in ('gt_box', 'gt_box_expanded', 'pred_box'):
        np.save(os.path.join(sample_dir, f'{key}.npy'), payload[key])
    for tag in ('union', 'robust'):
        geo = payload[tag]
        for key in SAVED_GEO_KEYS:
            if key not in geo:
                continue
            np.save(os.path.join(sample_dir, f'{tag}_{key}.npy'),
                    np.asarray(geo[key]))
    return sample_dir


def verify_saved(sample_dir, payload):
    """The saved point sets must equal the model output bit for bit."""
    for tag in ('union', 'robust'):
        geo = payload[tag]
        saved = np.load(os.path.join(sample_dir, f'{tag}_roi_points_xyz.npy'))
        np.testing.assert_allclose(saved, geo['roi_points_xyz'], rtol=0, atol=0)
        idx = np.load(os.path.join(sample_dir, f'{tag}_roi_point_indices.npy'))
        assert saved.shape[0] == idx.shape[0]
        np.testing.assert_allclose(payload['original_xyz'][idx], saved,
                                   rtol=0, atol=0)
        top = np.load(os.path.join(sample_dir, f'{tag}_roi_seed_indices.npy'))
        assert np.unique(top).size == top.size
        centres = np.load(os.path.join(sample_dir, f'{tag}_roi_centers.npy'))
        radii = np.load(os.path.join(sample_dir, f'{tag}_roi_radii.npy'))
        per_centre = np.load(os.path.join(
            sample_dir, f'{tag}_roi_radii_per_centre.npy'))
        assert per_centre.size == centres.shape[0]
        np.testing.assert_allclose(expand_radii(centres, radii), per_centre,
                                   rtol=0, atol=0)
    np.testing.assert_allclose(
        np.load(os.path.join(sample_dir, 'gt_box.npy')), payload['gt_box'],
        rtol=0, atol=0)


def score_stats(values):
    s = np.asarray(values, dtype=np.float64).reshape(-1)
    return {'min': float(s.min()), 'max': float(s.max()),
            'mean': float(s.mean()), 'std': float(s.std())}


def metadata(payload, model, args):
    union, robust = payload['union'], payload['robust']
    return {
        'sample_id': payload['sample_id'],
        'dataset_index': payload['dataset_index'],
        'scan_id': payload['scan_id'],
        'utterance': payload['utterance'],
        'target_name': payload['target_name'],
        'num_original_points': int(payload['original_xyz'].shape[0]),
        'num_seeds': int(union['seed_xyz'].shape[0]),
        'num_roi_points': int(model.roi_num_points),
        'roi_topk': int(model.roi_topk),
        'num_vis_tokens': {'multi_ball': union['num_vis_tokens'],
                           'robust_ball': robust['num_vis_tokens']},
        'coordinate_system': COORDINATE_SYSTEM,
        'query_index': int(union['query_index']),
        'query_selection': 'contrastive projection affinity (evaluator criterion)',
        'gt_box': payload['gt_box'].tolist(),
        'gt_box_expanded': payload['gt_box_expanded'].tolist(),
        'pred_box': payload['pred_box'].tolist(),
        'iou_3d': payload['iou_3d'],
        'entropy_score': {**score_stats(union['roi_entropy_scores']),
                          'definition': 's_i = ||dH_sel/dF_i||_2'},
        'box_field': {**score_stats(robust.get('roi_field_rank', [0.0])),
                      'definition': 'f_i = sum_m w_m exp(-0.5 * d_mahal^2), '
                                    'd_mahal in predicted-box units'},
        'geometry': {
            'multi_ball': {
                'centres': int(union['roi_centers'].shape[0]),
                'radius_cfg': float(getattr(model, 'roi_radius', -1.0)),
                'radius_used': union['roi_radius_used'],
                'num_candidates': union['roi_num_candidates'],
            },
            'robust_ball': {
                'centres': int(robust['roi_centers'].shape[0]),
                'radii': robust['roi_radii'].tolist(),
                'radius_used': robust['roi_radius_used'],
                'num_candidates': robust['roi_num_candidates'],
            },
        },
        'torch': torch.__version__,
        'warnings': payload['warnings'],
    }


# --------------------------------------------------------------------------- #
# figure (2 x 3)
# --------------------------------------------------------------------------- #
def build_figure(payload, cli, out_path):
    original_xyz = payload['original_xyz']
    n_points = original_xyz.shape[0]
    keep = np.arange(n_points)
    if 0 < int(cli.scene_max_points) < n_points:
        rng = np.random.default_rng(int(cli.seed))
        keep = np.sort(rng.choice(n_points, int(cli.scene_max_points), replace=False))
    scene_xyz = original_xyz[keep]
    if payload['original_rgb'] is not None:
        scene_rgb = np.clip(payload['original_rgb'][keep] * 1.35 + 0.02, 0.0, 1.0)
    else:
        scene_rgb = np.tile(np.asarray(DEFAULT_SCENE_RGB), (len(keep), 1))
    grey = np.tile(np.asarray(SCENE_GREY), (len(keep), 1))

    union, robust = payload['union'], payload['robust']
    seed_xyz = union['seed_xyz']
    s_scores = union['roi_entropy_scores']
    f_scores = robust['roi_field_rank']
    top_idx = union['roi_seed_indices']

    # one colour table per signal; the Top-K colours are *sliced*, never redone
    col_s = build_score_colorizer(s_scores, cmap=cli.cmap, mode=cli.score_norm,
                                  q_low=cli.q_low, q_high=cli.q_high)
    col_f = build_score_colorizer(f_scores, cmap=cli.cmap, mode='minmax',
                                  q_low=cli.q_low, q_high=cli.q_high)
    seeds_col_s = col_s.colours()
    seeds_col_f = col_f.colours()
    top_col = seeds_col_s[top_idx]

    # ---- camera: frame the scene + every box / ball that is drawn ----------
    fit = [original_xyz,
           box_corners(payload['gt_box'][:3], payload['gt_box'][3:], 0.0),
           box_corners(payload['gt_box_expanded'][:3],
                       payload['gt_box_expanded'][3:], 0.0),
           box_corners(payload['pred_box'][:3], payload['pred_box'][3:], 0.0)]
    for centre, radius in zip(union['roi_centers'],
                              union['roi_radii_per_centre']):
        fit.append(sphere_wireframe(centre, radius, n=8))
    for centre, radius in zip(robust['roi_centers'],
                              robust['roi_radii_per_centre']):
        fit.append(sphere_wireframe(centre, radius, n=8))
    fit = np.concatenate(fit, axis=0)
    camera = camera_from_points(
        fit, direction=cli.camera_dir, up=cli.up, fov=60.0,
        aspect=float(cli.panel_width) / float(cli.panel_height),
        tan_v=calibrate_tan_v(cli.panel_width, cli.panel_height, 60.0))
    camera = autofit_camera(camera, fit, cli.panel_width, cli.panel_height,
                            tan_v=calibrate_tan_v(cli.panel_width,
                                                  cli.panel_height, 60.0))

    # ---------------- panel (a): scene + boxes -----------------------------
    pa = Panel(title='(a) scene cloud + GT box / Pred box')
    pa.add_points(scene_xyz, scene_rgb, 4)
    pa.add_box(payload['gt_box'][:3], payload['gt_box'][3:], 0.0, GT_GREEN, 4.0)
    pa.add_box(payload['pred_box'][:3], payload['pred_box'][3:], 0.0, PRED_RED, 4.0)

    # ---------------- panel (b): entropy gradient --------------------------
    pb = Panel(title='(b) 1024 seeds: entropy-gradient score  s_i = ||dH_sel/dF_i||')
    pb.add_points(scene_xyz, grey, cli.scene_point_size)
    pb.add_points(seed_xyz, seeds_col_s, cli.scene_point_size * 4.5)

    # ---------------- panel (c): box-proposal field ------------------------
    pc = Panel(title='(c) 1024 seeds: box-proposal field f_i (region prior)')
    pc.add_points(scene_xyz, grey, cli.scene_point_size)
    pc.add_points(seed_xyz, seeds_col_f, cli.scene_point_size * 4.5)

    # ---------------- panel (d): multi-ball geometry -----------------------
    n_balls = union['roi_centers'].shape[0]
    pd = Panel(title='(d) Top-%d + %d balls (r=%.2f m) + sampled points'
                     % (top_idx.size, n_balls, float(model_radius(union))))
    pd.add_points(scene_xyz, grey, cli.scene_point_size)
    for centre, radius in zip(union['roi_centers'],
                              union['roi_radii_per_centre']):
        pd.add_points(sphere_wireframe(centre, radius, cli.ball_rings),
                      BALL_ORANGE, 1.3)
    pd.add_points(union['roi_points_xyz'], ROI_ORANGE, cli.scene_point_size * 3.2)
    pd.add_points(seed_xyz[top_idx], top_col, cli.scene_point_size * 8.0)
    pd.add_box(payload['gt_box'][:3], payload['gt_box'][3:], 0.0, GT_GREEN, 3.0)

    # ---------------- panel (e): robust single ball ------------------------
    radii = robust['roi_radii_per_centre']
    pe = Panel(title='(e) robust single ball (r=%.2f m) + sampled points'
                     % (float(radii[0]) if radii.size else float('nan')))
    pe.add_points(scene_xyz, grey, cli.scene_point_size)
    for centre, radius in zip(robust['roi_centers'], radii):
        pe.add_points(sphere_wireframe(centre, radius, cli.ball_rings * 2),
                      BALL_BLUE, 1.8)
        pe.add_points(np.asarray(centre).reshape(1, 3), (0.0, 0.0, 0.0),
                      cli.scene_point_size * 8.0)
    pe.add_points(robust['roi_points_xyz'], ROI_BLUE, cli.scene_point_size * 3.2)
    pe.add_box(payload['gt_box'][:3], payload['gt_box'][3:], 0.0, GT_GREEN, 3.0)
    pe.add_box(payload['gt_box_expanded'][:3], payload['gt_box_expanded'][3:],
               0.0, GT_GREEN, 2.0)

    # ---------------- panel (f): the two sampled sets ----------------------
    pf = Panel(title='(f) sampled ROI points: multi-ball vs robust ball')
    pf.add_points(scene_xyz, grey, cli.scene_point_size)
    pf.add_points(union['roi_points_xyz'], ROI_ORANGE, cli.scene_point_size * 3.2)
    pf.add_points(robust['roi_points_xyz'], ROI_BLUE, cli.scene_point_size * 3.2)
    pf.add_box(payload['gt_box_expanded'][:3], payload['gt_box_expanded'][3:],
               0.0, GT_GREEN, 2.0)

    panels = [pa, pb, pc, pd, pe, pf]
    images, backends = [], set()
    for panel in panels:
        image, backend = render_panel(panel, camera, cli.panel_width,
                                      cli.panel_height)
        images.append(image)
        backends.add(backend)

    # ---------------- matplotlib assembly (2 rows x 3 cols) ----------------
    dpi = 100
    pw, ph = cli.panel_width, cli.panel_height
    title_in, gap = 0.95, 0.10
    fig_w_in = 3.0 * pw / dpi
    fig_h_in = 2.0 * ph / dpi + title_in + gap
    panel_h_frac = (ph / dpi) / fig_h_in
    title_frac = title_in / fig_h_in
    gap_frac = gap / fig_h_in
    top_row = 1.0 - title_frac - panel_h_frac
    bot_row = top_row - gap_frac - panel_h_frac

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor('white')
    positions = []
    for row_y in (top_row, bot_row):
        for col in range(3):
            positions.append([col / 3.0, row_y, 1.0 / 3.0, panel_h_frac])
    axes = [fig.add_axes(pos) for pos in positions]

    legends = [
        [Line2D([], [], color=GT_GREEN, lw=3, label='GT box'),
         Line2D([], [], color=PRED_RED, lw=3,
                label='Pred box (query %d)' % payload['union']['query_index'])],
        [Line2D([], [], marker='o', ls='', color=col_s.cmap(0.05), ms=6,
                label='low score'),
         Line2D([], [], marker='o', ls='', color=col_s.cmap(0.95), ms=6,
                label='high score')],
        [Line2D([], [], marker='o', ls='', color=col_f.cmap(0.05), ms=6,
                label='outside the boxes'),
         Line2D([], [], marker='o', ls='', color=col_f.cmap(0.95), ms=6,
                label='inside the boxes')],
        [Line2D([], [], color=BALL_ORANGE, lw=1.5, label='%d balls' % n_balls),
         Line2D([], [], marker='o', ls='', color=ROI_ORANGE, ms=6,
                label='%d sampled pts' % union['roi_points_xyz'].shape[0]),
         Line2D([], [], marker='o', ls='', color=col_s.cmap(0.95), ms=8,
                label='Top-%d seeds' % top_idx.size)],
        [Line2D([], [], color=BALL_BLUE, lw=1.5, label='single ball'),
         Line2D([], [], marker='o', ls='', color=ROI_BLUE, ms=6,
                label='%d sampled pts' % robust['roi_points_xyz'].shape[0]),
         Line2D([], [], color=GT_GREEN, lw=2, label='GT box +1 m')],
        [Line2D([], [], marker='o', ls='', color=ROI_ORANGE, ms=6,
                label='multi-ball'),
         Line2D([], [], marker='o', ls='', color=ROI_BLUE, ms=6,
                label='robust ball'),
         Line2D([], [], color=GT_GREEN, lw=2, label='GT box +1 m')],
    ]

    for ax, image, panel, handles in zip(axes, images, panels, legends):
        ax.imshow(image, aspect='auto')
        ax.set_axis_off()
        ax.text(0.008, 0.995, panel.title, transform=ax.transAxes, va='top',
                ha='left', fontsize=11.5, color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2.5))
        ax.legend(handles=handles, loc='lower left', fontsize=8, framealpha=0.8,
                  borderpad=0.4, handlelength=1.4)

    for ax, colorizer in ((axes[1], col_s), (axes[2], col_f)):
        cax = ax.inset_axes([0.905, 0.10, 0.022, 0.42])
        cbar = fig.colorbar(colorizer.mappable, cax=cax)
        cbar.set_label(colorizer.label, fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    s_stats = score_stats(s_scores)
    fig.text(0.5, 0.996,
             f"{payload['sample_id']}  |  {payload['scan_id']}  |  "
             f"\"{payload['utterance'][:120]}\"",
             ha='center', va='top', fontsize=13, fontweight='bold')
    fig.text(0.5, 0.996 - 0.42 / fig_h_in,
             f"pred IoU={payload['iou_3d']:.3f}  |  "
             f"multi-ball: {n_balls} balls, r_used="
             f"{union['roi_radius_used']:.2f} m, cand={union['roi_num_candidates']}"
             f"  |  robust ball: r="
             f"{float(radii[0]) if radii.size else float('nan'):.2f} m, "
             f"cand={robust['roi_num_candidates']}"
             f"  |  s: mean={s_stats['mean']:.3g}, max={s_stats['max']:.3g}",
             ha='center', va='top', fontsize=9, color='0.25')

    fig.savefig(out_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return backends


def model_radius(geo):
    """Radius shown in the panel title of the multi-ball geometry."""
    radii = np.asarray(geo['roi_radii'], dtype=np.float64).reshape(-1)
    return float(radii[0]) if radii.size else float('nan')


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    cli = parse_cli()
    args = build_run_args(cli)

    random.seed(cli.seed)
    np.random.seed(cli.seed)
    torch.manual_seed(cli.seed)
    torch.cuda.manual_seed_all(cli.seed)

    if cli.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available but the ROI path needs it')
    device = torch.device(cli.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    os.makedirs(cli.output_dir, exist_ok=True)

    cfg_from_yaml_file(cli.config, model_cfg)
    pc_range = np.asarray(model_cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    num_pf = len(model_cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list)
    data_processor = DataProcessor(model_cfg.DATA_PROCESSOR, pc_range,
                                   training=False, num_point_features=num_pf)
    pipeline = _Pipeline(model_cfg, data_processor)

    model = pipeline.get_model(args)
    load_model_weights(args, model)
    model = model.to(device).eval()
    if not getattr(model, 'roi_enabled', False):
        raise RuntimeError('the model was built without the ROI branch')
    print(f"[info] ROI points={model.roi_num_points}, topk={model.roi_topk}, "
          f"multi-ball radius={model.roi_radius}, "
          f"single-ball radius_max={model.roi_radius_max}")

    _train_ds, test_ds = _Pipeline.get_datasets(args)
    mean_rgb = np.asarray(test_ds.mean_rgb, dtype=np.float32)
    print(f'[info] test set: {len(test_ds)} samples (split={test_ds.split}), '
          f'use_color={args.use_color}')

    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=cli.num_workers,
                        pin_memory=(device.type == 'cuda'), drop_last=False)

    summary = {
        'num_requested': int(cli.num_samples),
        'num_ok': 0,
        'num_failed': 0,
        'start_index': int(cli.start_index),
        'config': {
            'config_yaml': cli.config,
            'checkpoint_path': cli.checkpoint_path,
            'output_dir': cli.output_dir,
            'num_samples': cli.num_samples,
            'panel_size': [cli.panel_width, cli.panel_height],
            'cmap': cli.cmap,
            'score_norm': cli.score_norm,
            'roi': {
                'num_points': int(model.roi_num_points),
                'topk': int(model.roi_topk),
                'multi_ball_radius': float(model.roi_radius),
                'single_ball_radius_max': float(model.roi_radius_max),
                'selector_single_ball': 'field',
            },
        },
        'samples': [],
        'failures': [],
        'warnings': [],
        'backends': [],
    }

    tic = time.time()
    done = 0
    for step, batch_data in enumerate(loader):
        if step < cli.start_index:
            continue
        if done >= cli.num_samples:
            break
        sample_id = f'sample_{done + 1:06d}'
        try:
            batch_data = {k: (v.to(device, non_blocking=True)
                              if torch.is_tensor(v) else v)
                          for k, v in batch_data.items()}
            payload = collect_sample(model, args, pipeline, batch_data, step,
                                     sample_id, mean_rgb=mean_rgb)
            sample_dir = save_payload(cli.output_dir, payload)
            verify_saved(sample_dir, payload)
            backends = build_figure(payload, cli,
                                    os.path.join(sample_dir, 'visualization.png'))
            with open(os.path.join(sample_dir, 'metadata.json'), 'w') as fid:
                json.dump(metadata(payload, model, args), fid, indent=2,
                          ensure_ascii=False)
            summary['num_ok'] += 1
            summary['backends'] = sorted(set(summary['backends']) | set(backends))
            summary['samples'].append({
                'sample_id': sample_id,
                'dataset_index': step,
                'scan_id': payload['scan_id'],
                'iou_3d': payload['iou_3d'],
                'multi_ball_candidates': payload['union']['roi_num_candidates'],
                'robust_ball_candidates': payload['robust']['roi_num_candidates'],
                'robust_ball_radius': float(payload['robust']['roi_radii'][0]),
                'pred_box': payload['pred_box'].tolist(),
                'gt_box': payload['gt_box'].tolist(),
            })
            for message in payload['warnings']:
                summary['warnings'].append(f'{sample_id}: {message}')
            print(f"[{done + 1}/{cli.num_samples}] {sample_id} "
                  f"(idx={step}, {payload['scan_id']}) ok, "
                  f"IoU={payload['iou_3d']:.3f}, "
                  f"cand multi/robust="
                  f"{payload['union']['roi_num_candidates']}/"
                  f"{payload['robust']['roi_num_candidates']}"
                  + (f", warnings={payload['warnings']}" if payload['warnings'] else ''))
            done += 1
        except Exception:
            summary['num_failed'] += 1
            summary['failures'].append({'sample_id': sample_id,
                                        'dataset_index': step,
                                        'error': traceback.format_exc()})
            print(f'[error] sample {sample_id} (idx={step}) failed:')
            traceback.print_exc()
            done += 1

    summary['elapsed_sec'] = round(time.time() - tic, 2)
    ious = [s['iou_3d'] for s in summary['samples']]
    if ious:
        summary['mean_iou'] = float(np.mean(ious))
        summary['acc@0.25'] = float(np.mean([iou > 0.25 for iou in ious]))
        summary['acc@0.50'] = float(np.mean([iou > 0.50 for iou in ious]))
    with open(os.path.join(cli.output_dir, 'summary.json'), 'w') as fid:
        json.dump(summary, fid, indent=2, ensure_ascii=False)

    print(f"[done] ok={summary['num_ok']} failed={summary['num_failed']} "
          f"-> {cli.output_dir}/summary.json "
          f"(backends: {', '.join(summary['backends']) or 'n/a'})")


if __name__ == '__main__':
    main()
