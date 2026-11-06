#!/usr/bin/env python
"""Visualise the entropy-gradient guided ROI sampling of a PVGround checkpoint.

What this script does
---------------------
For the first ``--num_samples`` samples of the *test* split (original order, no
repetition) it runs the **real** ``PVGround.forward`` once and then only *reads*
the intermediate results of that forward pass:

======================  =====================================================
panel (a)               original RGB cloud + GT box (green) + Pred box (red)
panel (b)               the 1024 FPS seeds coloured by the entropy-gradient
                        score ``s_i = ||dH_sel/dF_i||_2``  (coolwarm + colorbar)
panel (c)               the Top-32 seeds (same colours as in (b)) + the 512 ROI
                        points of the model
panel (d)               only the 512 ROI points of the model + GT box grown by
                        1 m in every direction (green wireframe)
======================  =====================================================

Hard rules implemented here (see also the assertion block in
``collect_sample``):

* the 512 points of panel (d) are ``end_points['roi_points_xyz']`` **verbatim**:
  no re-sampling, no FPS, no de-duplication, no re-ordering, no sorting;
* the Top-32 seeds are ``end_points['roi_seed_indices']`` (the indices returned
  by ``torch.topk`` inside the model), never recomputed;
* the scores are ``end_points['roi_entropy_scores']``, i.e. the very tensor that
  ``_sample_roi_region`` feeds to ``torch.topk`` (a debug-only key added to
  ``models/pv_ground.py``, the computation itself is untouched);
* no ground-truth information is used to select seeds / ROI points;
* the model weights, the ROI hyper-parameters and the point cloud are unchanged.

Usage
-----
conda activate 3dgrounding
python tools/visualize_roi_points.py \
    --config wandb_config.yaml \
    --checkpoint_path data/PV-Ground_ScanRefer_single.pth \
    --data_root data/ \
    --output_dir visualization_results \
    --num_samples 100

The remaining model / dataset switches (``--use_color``, ``--detect_intermediate``,
``--joint_det``, ``--use_soft_token_loss``, ``--use_contrastive_align``,
``--self_attend``, ``--roi*`` ...) are the ones of ``main_utils.parse_option``
and default to the values used by ``tools/test_scanrefer_single.sh``; any of
them can be overridden on the command line.
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

from main_utils import BaseTrainTester, load_checkpoint, parse_option  # noqa: E402
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
ROI_BLUE = (0.05, 0.30, 0.90)
SCENE_GREY = (0.74, 0.74, 0.74)
DEFAULT_SCENE_RGB = (0.62, 0.62, 0.62)

# defaults of tools/test_scanrefer_single.sh (+ the ROI switches needed here)
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
    'lora': False,
    'init_checkpoint': None,
    # entropy-gradient guided ROI (must stay enabled: it produces the fields
    # that are visualised here)
    'roi': True,
    'roi_points': 512,
    'roi_topk': 32,
    'roi_radius': 0.3,
    'roi_max_radius': 2.0,
    'roi_expand_factor': 1.5,
    'roi_tau': 0.0,
    'roi_log_interval': 0,
}

COORDINATE_SYSTEM = (
    "ScanNet axis-aligned raw coordinates in metres, identical for "
    "inputs['points'][..., 1:4], end_points['coords'], end_points['seed_xyz'], "
    "end_points['roi_points_xyz'], center_label/size_gts and last_center/"
    "last_pred_size (no centring, no scaling). GT / Pred boxes are "
    "[cx, cy, cz, dx, dy, dz] with dx/dy/dz the extents along x/y/z and "
    "heading = 0 rad (this checkpoint has no heading head)."
)


class _Pipeline(TrainTester):
    """``TrainTester`` without the DDP logger / tensorboard / log-dir setup.

    Only the data preprocessing (``_get_inputs``) and the model factory
    (``get_model``) of the real evaluation pipeline are reused, so the inputs
    and the architecture are bit-identical to ``tools/test_scanrefer_single.sh``.
    """

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
        description='Visualise PVGround entropy-gradient ROI sampling '
                    '(read-only, uses the real forward pass).',
        allow_abbrev=False,
    )
    p.add_argument('--config', default='wandb_config.yaml',
                   help='model config yaml (pcdet format)')
    p.add_argument('--checkpoint_path',
                   default='data/PV-Ground_ScanRefer_single.pth',
                   help='trained checkpoint (full or LoRA-only)')
    p.add_argument('--data_root', default='data/',
                   help="dataset root, must contain ScanRefer/, roberta-base/, "
                        "val_v3scans.pkl, superpoints/ ...")
    p.add_argument('--output_dir', default='visualization_results')
    p.add_argument('--num_samples', type=int, default=100,
                   help='number of *consecutive* test samples to visualise')
    p.add_argument('--start_index', type=int, default=0,
                   help='index in the test set of the first sample')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--panel_width', type=int, default=900)
    p.add_argument('--panel_height', type=int, default=700)
    p.add_argument('--scene_point_size', type=float, default=1.5,
                   help='point size (px) of the background cloud')
    p.add_argument('--scene_max_points', type=int, default=50000,
                   help='max background points drawn per panel (overlays are '
                        'never subsampled); 0 = all points')
    p.add_argument('--cmap', default='coolwarm')
    p.add_argument('--score_norm', default='log',
                   choices=['minmax', 'robust', 'log', 'log_robust'],
                   help='monotone score normalisation for the colour map')
    p.add_argument('--q_low', type=float, default=1.0)
    p.add_argument('--q_high', type=float, default=99.0)
    p.add_argument('--camera_dir', type=float, nargs=3,
                   default=[0.80, -1.0, 1.5], help='view direction (shared)')
    p.add_argument('--up', type=float, nargs=3, default=[0.0, 0.0, 1.0])
    p.add_argument('--seed', type=int, default=111,
                   help='RNG seed (query generator + background subsampling). '
                        'The checkpoint draws gumbel noise in GumbelSampling '
                        'and the model uses non-deterministic CUDA reductions, '
                        'so two separate runs of the *same* sample can still '
                        'differ slightly; everything saved for one sample is '
                        'internally consistent (verified bit-exactly)')
    return p.parse_known_args()[0]


def build_run_args(cli):
    """``parse_option()`` namespace with the evaluation-script defaults."""
    args = parse_option()
    given = {a[2:].split('=')[0].replace('-', '_')
             for a in sys.argv[1:] if a.startswith('--')}
    for key, value in EVAL_PRESET.items():
        if key not in given:
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


def move_to_device(batch_data, device):
    for key in batch_data:
        if torch.is_tensor(batch_data[key]):
            batch_data[key] = batch_data[key].to(device, non_blocking=True)
    return batch_data


def score_stats(scores, top_indices):
    """Summary statistics of the raw entropy-gradient scores."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    top = np.asarray(s[top_indices], dtype=np.float64)
    return {
        'min': float(s.min()),
        'max': float(s.max()),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'top32_min': float(top.min()) if top.size else None,
    }


def box_iou_axis_aligned(a, b):
    """3D IoU of two ``[cx, cy, cz, dx, dy, dz]`` axis-aligned boxes."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_min, a_max = a[:3] - a[3:] / 2.0, a[:3] + a[3:] / 2.0
    b_min, b_max = b[:3] - b[3:] / 2.0, b[:3] + b[3:] / 2.0
    inter = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0.0, None)
    inter_vol = float(np.prod(inter))
    vol_a, vol_b = float(np.prod(np.clip(a[3:], 0, None))), float(np.prod(np.clip(b[3:], 0, None)))
    union = vol_a + vol_b - inter_vol
    return inter_vol / union if union > 1e-12 else 0.0


def _strip_ddp_prefix(state):
    """Remove the ``module.`` prefix that DDP adds to every parameter name."""
    out = {}
    for key, value in state.items():
        while key.startswith('module.'):
            key = key[len('module.'):]
        out[key] = value
    return out


def load_model_weights(args, model):
    """Load a checkpoint into the *unwrapped* model.

    ``main_utils.load_checkpoint`` is written for the DDP-wrapped model used by
    ``train_dist_mod.py``: the checkpoints are saved from
    ``DistributedDataParallel``, so every key starts with ``module.``.  Here the
    network is used directly (single process), hence the prefix is stripped and
    the same "partial loading" rules (LoRA-only checkpoints) are applied.
    """
    ckpt_path = args.checkpoint_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)
    print(f"=> loading checkpoint '{ckpt_path}'")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if ckpt.get('lora_only', False):
        base_path = getattr(args, 'init_checkpoint', None) or \
            'data/PV-Ground_ScanRefer_single.pth'
        print(f"=> LoRA-only checkpoint, restoring the frozen base weights "
              f"from '{base_path}'")
        base = torch.load(base_path, map_location='cpu', weights_only=False)
        model.load_state_dict(_strip_ddp_prefix(base['model']), strict=False)
        del base

    state = _strip_ddp_prefix(ckpt['model'])
    epoch = ckpt.get('epoch')
    del ckpt

    model_keys = set(model.state_dict().keys())
    matched = len(model_keys & set(state.keys()))
    result = model.load_state_dict(state, strict=False)
    missing = [k for k in result.missing_keys if 'position_ids' not in k]
    unexpected = [k for k in result.unexpected_keys if 'position_ids' not in k]
    print(f"=> loaded '{ckpt_path}' (epoch {epoch}): {matched} matched, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    if missing and len(missing) <= 12:
        print(f'   missing keys: {missing}')
    if unexpected and len(unexpected) <= 12:
        print(f'   unexpected keys: {unexpected}')
    if matched < 0.9 * len(model_keys):
        raise RuntimeError(
            f'only {matched}/{len(model_keys)} parameters could be loaded from '
            f'{ckpt_path} - refusing to visualise a partly random network')
    return model


def pick_query(end_points, batch_data, b=0):
    """Query index used for the predicted box (== the model's own prediction).

    Same criterion as the position-alignment branch of
    ``evaluation/grounding_evaluator.py`` (``evaluate_bbox_by_pos_align``): the
    semantic probabilities ``softmax(last_sem_cls_scores)`` are weighted with
    the (binarised) phrase maps of the first target and summed over the token
    axis; the arg-max over the 256 decoder queries gives the box.

    Only *text* information is used here - no ground-truth geometry.
    """
    sem = torch.softmax(end_points['last_sem_cls_scores'][b].float(), dim=-1)  # (Q, C)
    num_cls = sem.shape[-1]
    sem = sem.reshape(-1, num_cls)

    def phrase_map(key, binarize=False):
        tensor = batch_data.get(key, None)
        if tensor is None or not torch.is_tensor(tensor):
            return None
        pmap = tensor[b, 0].detach().float().reshape(1, -1)
        if binarize:
            pmap = (pmap > 0).float()
        return pmap[:, :num_cls] if pmap.shape[-1] >= num_cls else None

    def term(pmap):
        if pmap is None:
            return torch.zeros(sem.shape[0], dtype=sem.dtype, device=sem.device)
        return (sem[:, :pmap.shape[-1]] * pmap).sum(-1)

    main = phrase_map('positive_map', binarize=True)
    if main is None:
        scores = sem.max(-1).values
        rule = 'max_prob'
    else:
        scores = term(main)
        for key, sign in (('modify_positive_map', 1.0), ('pron_positive_map', 1.0),
                          ('rel_positive_map', 1.0), ('other_entity_map', -1.0)):
            pmap = phrase_map(key)
            if pmap is not None:
                scores = scores + sign * term(pmap)
        rule = 'pos_align'
    return int(scores.argmax()), scores.detach().cpu().numpy(), rule


# --------------------------------------------------------------------------- #
# forward + extraction (the only place that touches the model)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_forward(model, inputs):
    """One real forward pass.

    ``torch.no_grad()`` is safe here (it is exactly what the official
    evaluation loop uses): ``PVGround._selection_entropy_seed_score`` opens its
    own ``torch.enable_grad()`` block for the single ``autograd.grad`` call, so
    the entropy gradient is still computed normally.  ``torch.inference_mode()``
    would *not* be safe (its tensors cannot be used for autograd).
    """
    return model(inputs)


def collect_sample(model, args, inputs, batch_data, dataset_index, sample_id,
                   mean_rgb=None):
    """Extract every intermediate result of one real forward pass."""
    end_points = run_forward(model, inputs)
    b = 0
    warnings = []

    for key in ('roi_points_xyz', 'roi_point_indices', 'roi_seed_indices',
                'roi_entropy_scores', 'coords', 'seed_xyz', 'seed_inds'):
        if key not in end_points:
            raise KeyError(
                f'{key!r} missing from end_points - is the ROI path enabled '
                f'(args.roi={getattr(args, "roi", None)}) and was '
                f'models/pv_ground.py patched with the debug outputs?')

    # ---- scene cloud: the exact tensor the ROI sampler used ---------------
    original_xyz = to_numpy(end_points['coords'][b])                    # (Np, 3)
    pts_all = inputs['points']                                          # (Np, 4 + C)
    if pts_all.shape[0] != original_xyz.shape[0]:
        raise RuntimeError(
            f"end_points['coords'] has {original_xyz.shape[0]} points but "
            f"inputs['points'] has {pts_all.shape[0]} rows")
    num_per_sample = np.bincount(to_numpy(pts_all[:, 0], np.int64),
                                 minlength=inputs['batch_size'])
    if len(set(num_per_sample.tolist())) != 1:
        raise RuntimeError(f'ragged point counts per sample: {num_per_sample.tolist()}')

    original_rgb = None
    if getattr(args, 'use_color', False) and pts_all.shape[1] >= 7:
        if mean_rgb is None:
            raise RuntimeError('dataset mean_rgb was not set')
        # dataset stores ``scan.color - mean_rgb``; restore the [0, 1] colours
        original_rgb = np.clip(to_numpy(pts_all[:, 4:7]) + mean_rgb[None, :], 0.0, 1.0)
        if original_rgb.shape[0] != original_xyz.shape[0]:
            raise RuntimeError('RGB / xyz length mismatch')
    else:
        warnings.append('dataset has no RGB for this run: grey cloud is used')

    # ---- 1024 seeds -------------------------------------------------------
    seed_xyz = to_numpy(end_points['seed_xyz'][b])                      # (N, 3)
    seed_inds = to_numpy(end_points['seed_inds'][b], np.int64)          # (N,)
    entropy_scores = to_numpy(end_points['roi_entropy_scores'][b])      # (N,)

    # ---- Top-32 (model's own torch.topk result) ---------------------------
    top32_indices = to_numpy(end_points['roi_seed_indices'][b], np.int64)   # (K,)
    top32_xyz = seed_xyz[top32_indices]

    # ---- 512 ROI points (authoritative model output) ----------------------
    roi_points_xyz = to_numpy(end_points['roi_points_xyz'][b])          # (S, 3)
    roi_point_indices = to_numpy(end_points['roi_point_indices'][b], np.int64)
    roi_radius_used = as_scalar(end_points['roi_radius_used'][b])
    roi_num_candidates = int(as_scalar(end_points['roi_num_candidates'][b]))
    roi_padded = bool(as_scalar(end_points['roi_padded'][b])) \
        if 'roi_padded' in end_points else None

    # ---- boxes ------------------------------------------------------------
    gt_center = to_numpy(batch_data['center_label'][b, 0, :3])
    gt_size = to_numpy(batch_data['size_gts'][b, 0, :3])
    gt_box = np.concatenate([gt_center, gt_size]).astype(np.float32)
    gt_heading = 0.0                       # ScanNet GT boxes are axis aligned
    exp_center, exp_size = expand_box(gt_center, gt_size, delta=1.0)

    query_index, query_scores, query_rule = pick_query(end_points, batch_data, b)
    pred_center = to_numpy(end_points['last_center'][b, query_index])
    pred_size = np.clip(to_numpy(end_points['last_pred_size'][b, query_index]), 1e-6, None)
    pred_box = np.concatenate([pred_center, pred_size]).astype(np.float32)
    pred_heading = 0.0                     # no heading head in this checkpoint
    if not np.isfinite(pred_box).all():
        warnings.append('the predicted box contains non-finite values')
        pred_box = None

    # ------------------------------------------------------------------ #
    # consistency checks (never modify data, only report)
    # ------------------------------------------------------------------ #
    num_seeds = int(seed_xyz.shape[0])
    if seed_xyz.shape[-1] != 3 or roi_points_xyz.shape[-1] != 3 or top32_xyz.shape[-1] != 3:
        raise RuntimeError('xyz tensors must have 3 coordinates in the last dim')
    if entropy_scores.shape[0] != num_seeds:
        raise RuntimeError(f'score/seed mismatch: {entropy_scores.shape[0]} vs {num_seeds}')
    expected_roi = int(getattr(model, 'roi_num_points', roi_points_xyz.shape[0]))
    if roi_points_xyz.shape[0] != expected_roi:
        raise RuntimeError(f'expected {expected_roi} ROI points, got {roi_points_xyz.shape[0]}')
    if expected_roi != 512:
        warnings.append(f'ROI point count is {expected_roi}, not the expected 512')
    if top32_indices.size != int(getattr(model, 'roi_topk', top32_indices.size)):
        warnings.append(f'Top-K is {top32_indices.size}, not '
                        f'{getattr(model, "roi_topk", "?")}')
    if roi_points_xyz.shape[0] != roi_point_indices.shape[0]:
        raise RuntimeError('roi points / roi indices length mismatch')
    if not np.isfinite(entropy_scores).all():
        warnings.append('non-finite entropy scores')
    if float(entropy_scores.max()) <= 0.0:
        warnings.append('all entropy scores are zero (check the autograd context)')
    if roi_padded:
        warnings.append('ROI sampling padded with replacement (kept as is)')

    # seed_inds must index the same cloud as roi_point_indices
    seed_err = None
    if seed_inds.size and 0 <= seed_inds.min() and seed_inds.max() < original_xyz.shape[0]:
        seed_err = float(np.abs(original_xyz[seed_inds] - seed_xyz).max())
        if seed_err > 1e-4:
            warnings.append(f'seed_xyz != coords[seed_inds] (max err {seed_err:.4g})')
    else:
        warnings.append('seed_inds out of range of end_points["coords"]')

    roi_err = None
    if roi_point_indices.size and 0 <= roi_point_indices.min() \
            and roi_point_indices.max() < original_xyz.shape[0]:
        roi_err = float(np.abs(original_xyz[roi_point_indices] - roi_points_xyz).max())
        if roi_err > 1e-4:
            warnings.append(
                f'roi_point_indices are not indices into end_points["coords"] '
                f'(max err {roi_err:.4g})')
    else:
        warnings.append('roi_point_indices out of range of end_points["coords"]')

    # the selected seeds must really be the highest scores (report only)
    top_order_ok = None
    if 0 < top32_indices.size < num_seeds:
        rest = np.setdiff1d(np.arange(num_seeds), top32_indices)
        top_order_ok = bool(entropy_scores[top32_indices].min()
                            >= entropy_scores[rest].max() - 1e-6)
        if not top_order_ok:
            warnings.append('Top-K indices are not the K largest scores')

    payload = {
        'sample_id': sample_id,
        'dataset_index': int(dataset_index),
        'scan_id': str(batch_data['scan_ids'][b]),
        'utterance': str(batch_data['utterances'][b]),
        'target_name': str(batch_data['target_name'][b]),
        'target_id': int(batch_data['target_id'][b]),
        'original_xyz': original_xyz.astype(np.float32),
        'original_rgb': None if original_rgb is None else original_rgb.astype(np.float32),
        'seed_xyz': seed_xyz.astype(np.float32),
        'seed_inds': seed_inds,
        'entropy_scores': entropy_scores.astype(np.float32),
        'top32_indices': top32_indices,
        'top32_xyz': top32_xyz.astype(np.float32),
        'roi_points_xyz': roi_points_xyz.astype(np.float32),
        'roi_point_indices': roi_point_indices,
        'roi_radius_used': float(roi_radius_used) if roi_radius_used is not None else None,
        'roi_num_candidates': roi_num_candidates,
        'roi_padded': roi_padded,
        'gt_box': gt_box,
        'gt_heading': gt_heading,
        'gt_box_expanded': np.concatenate([exp_center, exp_size]).astype(np.float32),
        'pred_box': pred_box,
        'pred_heading': pred_heading,
        'query_index': int(query_index),
        'query_rule': query_rule,
        'query_scores': query_scores.astype(np.float32),
        'num_vis_tokens': int(end_points.get('num_vis_tokens', -1)),
        'num_per_sample': int(num_per_sample[b]),
        'iou_3d': (float(box_iou_axis_aligned(gt_box, pred_box))
                   if pred_box is not None else 0.0),
        'seed_index_max_abs_err': seed_err,
        'roi_index_max_abs_err': roi_err,
        'topk_score_order_ok': top_order_ok,
        'warnings': warnings,
    }
    return payload


# --------------------------------------------------------------------------- #
# saving + verification
# --------------------------------------------------------------------------- #
NPY_FILES = (
    'original_xyz', 'original_rgb', 'seed_xyz', 'seed_inds', 'entropy_scores',
    'top32_indices', 'top32_xyz', 'roi_points_xyz', 'roi_point_indices',
    'gt_box', 'gt_box_expanded', 'pred_box',
)


def save_payload(out_dir, payload):
    sample_dir = os.path.join(out_dir, payload['sample_id'])
    os.makedirs(sample_dir, exist_ok=True)
    for key in NPY_FILES:
        value = payload.get(key)
        if value is None:
            continue
        np.save(os.path.join(sample_dir, f'{key}.npy'), np.asarray(value))
    return sample_dir


def list_or_none(value):
    return None if value is None else np.asarray(value).tolist()


def metadata(payload, model, args):
    return {
        'sample_id': payload['sample_id'],
        'dataset_index': payload['dataset_index'],
        'scan_id': payload['scan_id'],
        'utterance': payload['utterance'],
        'target_name': payload['target_name'],
        'target_id': payload['target_id'],
        'num_original_points': int(payload['original_xyz'].shape[0]),
        'num_seeds': int(payload['seed_xyz'].shape[0]),
        'num_top_seeds': int(payload['top32_indices'].size),
        'num_roi_points': int(payload['roi_points_xyz'].shape[0]),
        'roi_radius_used': payload['roi_radius_used'],
        'roi_num_candidates': payload['roi_num_candidates'],
        'roi_padded': payload['roi_padded'],
        'roi_topk': int(getattr(model, 'roi_topk', -1)),
        'roi_num_points': int(getattr(model, 'roi_num_points', -1)),
        'roi_radius_cfg': float(getattr(model, 'roi_radius', -1.0)),
        'num_vis_tokens': payload['num_vis_tokens'],
        'coordinate_system': COORDINATE_SYSTEM,
        'roi_points_source': "end_points['roi_points_xyz']",
        'top_seed_source': "end_points['roi_seed_indices']",
        'score_source': "end_points['roi_entropy_scores'] (= roi['score'] used by torch.topk)",
        'score_definition': 's_i = ||dH_sel/dF_i||_2',
        'score_stats': score_stats(payload['entropy_scores'], payload['top32_indices']),
        'query_index': payload['query_index'],
        'query_selection': payload['query_rule'],
        'gt_box': payload['gt_box'].tolist(),
        'gt_box_expanded': payload['gt_box_expanded'].tolist(),
        'pred_box': list_or_none(payload['pred_box']),
        'box_format': '[cx, cy, cz, dx, dy, dz] (extents along x / y / z)',
        'gt_heading': payload['gt_heading'],
        'pred_heading': payload['pred_heading'],
        'gt_box_expansion_m_per_side': 1.0,
        'iou_3d': payload['iou_3d'],
        'seed_index_max_abs_err': payload['seed_index_max_abs_err'],
        'roi_index_max_abs_err': payload['roi_index_max_abs_err'],
        'topk_score_order_ok': payload['topk_score_order_ok'],
        'points_per_sample': payload['num_per_sample'],
        'model_config': {
            'num_decoder_layers': int(args.num_decoder_layers),
            'num_queries': int(args.num_target),
            'roi_tau': float(getattr(model, 'roi_tau', -1.0)),
            'roi_expand_factor': float(getattr(model, 'roi_expand_factor', -1.0)),
            'roi_max_radius': float(getattr(model, 'roi_max_radius', -1.0)),
        },
        'warnings': payload['warnings'],
    }


def verify_saved(sample_dir, payload):
    """The saved 512 points must equal the model output bit by bit."""
    saved = np.load(os.path.join(sample_dir, 'roi_points_xyz.npy'))
    np.testing.assert_allclose(saved, payload['roi_points_xyz'], rtol=0, atol=0)

    saved_top = np.load(os.path.join(sample_dir, 'top32_xyz.npy'))
    np.testing.assert_allclose(
        saved_top, payload['seed_xyz'][payload['top32_indices']], rtol=0, atol=0)

    saved_idx = np.load(os.path.join(sample_dir, 'top32_indices.npy'))
    assert np.array_equal(saved_idx, payload['top32_indices'])
    assert len(saved_idx) == payload['top32_indices'].size
    assert saved.shape == (payload['roi_points_xyz'].shape[0], 3)
    assert saved.shape[0] == payload['roi_point_indices'].shape[0]

    # shapes / index semantics
    assert payload['seed_xyz'].shape[-1] == 3
    assert payload['roi_points_xyz'].shape[-1] == 3
    assert payload['top32_xyz'].shape[-1] == 3
    assert payload['top32_xyz'].shape[0] == payload['top32_indices'].size
    assert np.unique(payload['top32_indices']).size == payload['top32_indices'].size
    np.testing.assert_allclose(
        payload['top32_xyz'], payload['seed_xyz'][payload['top32_indices']],
        rtol=0, atol=0)
    # a Top-32 seed sits exactly on the scene point with the same index
    if payload['seed_index_max_abs_err'] is not None:
        assert payload['seed_index_max_abs_err'] < 1e-4


# --------------------------------------------------------------------------- #
# figure
# --------------------------------------------------------------------------- #
def build_figure(payload, colorizer, cli, out_path):
    """Four panels -> one 2x2 PNG (rendered panel by panel)."""
    original_xyz = payload['original_xyz']
    n_points = original_xyz.shape[0]
    keep = np.arange(n_points)
    limit = int(cli.scene_max_points)
    if 0 < limit < n_points:
        rng = np.random.default_rng(int(cli.seed))
        keep = np.sort(rng.choice(n_points, limit, replace=False))

    scene_xyz = original_xyz[keep]
    if payload['original_rgb'] is not None:
        # "colours brightened" for the display only
        scene_rgb_a = np.clip(payload['original_rgb'][keep] * 1.35 + 0.02, 0.0, 1.0)
    else:
        scene_rgb_a = np.tile(np.asarray(DEFAULT_SCENE_RGB), (len(keep), 1))
    grey = np.tile(np.asarray(SCENE_GREY), (len(keep), 1))

    # the camera frames the scene *and* the boxes, so that nothing is clipped
    fit_points = [original_xyz, box_corners(payload['gt_box'][:3],
                                            payload['gt_box'][3:], payload['gt_heading']),
                  box_corners(payload['gt_box_expanded'][:3],
                              payload['gt_box_expanded'][3:], payload['gt_heading'])]
    if payload['pred_box'] is not None:
        fit_points.append(box_corners(payload['pred_box'][:3],
                                      payload['pred_box'][3:], payload['pred_heading']))
    fit_points = np.concatenate(fit_points, axis=0)
    camera = camera_from_points(
        fit_points, direction=cli.camera_dir, up=cli.up, fov=60.0,
        aspect=float(cli.panel_width) / float(cli.panel_height),
        tan_v=calibrate_tan_v(cli.panel_width, cli.panel_height, 60.0))
    # one common camera for the four panels, verified so the content fills the tile
    camera = autofit_camera(camera, fit_points, cli.panel_width, cli.panel_height,
                            tan_v=calibrate_tan_v(cli.panel_width, cli.panel_height, 60.0))
    scores = payload['entropy_scores']
    top_idx = payload['top32_indices']
    # one colour table for all 1024 seeds; the Top-32 colours are *sliced* from
    # it, so panels (b) and (c) cannot disagree
    seed_colours = colorizer.colours()
    top_colours = seed_colours[top_idx]

    # ---------------- panel (a) -------------------------------------------
    panel_a = Panel(title='(a) scene cloud (RGB) + GT box / Pred box')
    panel_a.add_points(scene_xyz, scene_rgb_a, 4)
    panel_a.add_box(payload['gt_box'][:3], payload['gt_box'][3:],
                    payload['gt_heading'], GT_GREEN, 4.0)
    if payload['pred_box'] is not None:
        panel_a.add_box(payload['pred_box'][:3], payload['pred_box'][3:],
                        payload['pred_heading'], PRED_RED, 4.0)

    # ---------------- panel (b) -------------------------------------------
    panel_b = Panel(title='(b) 1024 seeds, entropy-gradient score')
    panel_b.add_points(scene_xyz, grey, cli.scene_point_size)
    panel_b.add_points(payload['seed_xyz'], seed_colours,
                       cli.scene_point_size * 4.5)

    # ---------------- panel (c) -------------------------------------------
    panel_c = Panel(title='(c) Top-%d seeds (same colours) + 512 ROI points'
                          % top_idx.size)
    panel_c.add_points(scene_xyz, grey, cli.scene_point_size)
    panel_c.add_points(payload['roi_points_xyz'], ROI_BLUE,
                       cli.scene_point_size * 3.5)
    panel_c.add_points(payload['top32_xyz'], top_colours,
                       cli.scene_point_size * 9.0)
    panel_c.add_box(payload['gt_box'][:3], payload['gt_box'][3:],
                    payload['gt_heading'], GT_GREEN, 4.0)
    
    if payload['pred_box'] is not None:
        panel_c.add_box(payload['pred_box'][:3], payload['pred_box'][3:],
                        payload['pred_heading'], PRED_RED, 4.0)

    # ---------------- panel (d) -------------------------------------------
    panel_d = Panel(title='(d) Top-%d seeds (same colours)'
                          % top_idx.size)
    panel_d.add_points(scene_xyz, grey, cli.scene_point_size)
    panel_d.add_points(payload['top32_xyz'], top_colours,
                       cli.scene_point_size * 9.0)
  
    panel_d.add_box(payload['gt_box'][:3],
                    payload['gt_box'][3:], payload['gt_heading'],
                    GT_GREEN, 4.0)

    panels = [panel_a, panel_b, panel_c, panel_d]
    images = []
    backends = set()
    for panel in panels:
        image, backend = render_panel(panel, camera, cli.panel_width, cli.panel_height)
        images.append(image)
        backends.add(backend)

    # ---------------- matplotlib assembly --------------------------------
    dpi = 100
    pw, ph = cli.panel_width, cli.panel_height
    title_in = 0.95          # space reserved for the two header lines
    gap = 0.10               # space between the two rows (inches)
    fig_w_in = 2.0 * pw / dpi
    fig_h_in = 2.0 * ph / dpi + title_in + gap
    title_frac = title_in / fig_h_in
    panel_h_frac = (ph / dpi) / fig_h_in
    gap_frac = gap / fig_h_in
    top_row = 1.0 - title_frac - panel_h_frac
    bot_row = top_row - gap_frac - panel_h_frac

    fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=dpi)
    fig.patch.set_facecolor('white')
    positions = [[0.0, top_row, 0.5, panel_h_frac],
                 [0.5, top_row, 0.5, panel_h_frac],
                 [0.0, bot_row, 0.5, panel_h_frac],
                 [0.5, bot_row, 0.5, panel_h_frac]]
    axes = [fig.add_axes(pos) for pos in positions]

    legend_a = [Line2D([], [], color=GT_GREEN, lw=3, label='GT box')]
    if payload['pred_box'] is not None:
        legend_a.append(Line2D([], [], color=PRED_RED, lw=3,
                               label='Pred box (query %d)' % payload['query_index']))
    legends = [
        legend_a,
        [Line2D([], [], marker='o', ls='', color=colorizer.cmap(0.05), ms=6, label='1024 seeds'),
         Line2D([], [], marker='o', ls='', color=colorizer.cmap(0.95), ms=6, label='high score')],
        [Line2D([], [], marker='o', ls='', color=ROI_BLUE, ms=6, label='512 ROI points'),
         Line2D([], [], marker='o', ls='', color=colorizer.cmap(0.95), ms=8,
                label='Top-%d seeds' % top_idx.size)],
        [Line2D([], [], marker='o', ls='', color=ROI_BLUE, ms=6,
                label='512 ROI points'),
         Line2D([], [], color=GT_GREEN, lw=3, label='GT box +1 m per side')],
    ]

    for ax, image, panel, handles in zip(axes, images, panels, legends):
        ax.imshow(image, aspect='auto')
        ax.set_axis_off()
        ax.text(0.008, 0.995, panel.title, transform=ax.transAxes,
                va='top', ha='left', fontsize=13, color='black',
                bbox=dict(facecolor='white', alpha=0.75, edgecolor='none', pad=2.5))
        ax.legend(handles=handles, loc='lower left', fontsize=9, framealpha=0.8,
                  borderpad=0.4, handlelength=1.4)

    cax = axes[1].inset_axes([0.905, 0.10, 0.022, 0.42])
    cbar = fig.colorbar(colorizer.mappable, cax=cax)
    cbar.set_label(colorizer.label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    stats = score_stats(payload['entropy_scores'], payload['top32_indices'])
    fig.text(0.5, 0.996,
             f"{payload['sample_id']}  |  {payload['scan_id']}  |  "
             f"\"{payload['utterance'][:110]}\"",
             ha='center', va='top', fontsize=14, fontweight='bold')
    radius_text = ('n/a' if payload['roi_radius_used'] is None
                   else f"{payload['roi_radius_used']:.2f} m")
    fig.text(0.5, 0.996 - 0.42 / fig_h_in,
             f"query={payload['query_index']}  |  "
             f"ROI r={radius_text}, "
             f"candidates={payload['roi_num_candidates']}, "
             f"padded={payload['roi_padded']}  |  "
             f"score mean={stats['mean']:.4g}, top32 min={stats['top32_min']:.4g}, "
             f"max={stats['max']:.4g}  |  IoU={payload['iou_3d']:.3f}",
             ha='center', va='top', fontsize=10, color='0.25')

    fig.savefig(out_path, dpi=dpi, facecolor='white')
    plt.close(fig)
    return backends


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    cli = parse_cli()
    args = build_run_args(cli)

    # The query generator of the checkpoint (GumbelSampling) draws gumbel noise
    # even in eval mode, so the RNG is fixed to make the run reproducible.  This
    # does not change the model / the ROI logic in any way.
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

    # ---- config + model ------------------------------------------------
    cfg_from_yaml_file(cli.config, model_cfg)
    pc_range = np.asarray(model_cfg.DATA_CONFIG.POINT_CLOUD_RANGE)
    num_point_features = len(model_cfg.DATA_CONFIG.POINT_FEATURE_ENCODING.src_feature_list)
    data_processor = DataProcessor(model_cfg.DATA_PROCESSOR, pc_range,
                                   training=False, num_point_features=num_point_features)
    pipeline = _Pipeline(model_cfg, data_processor)

    model = pipeline.get_model(args)
    load_model_weights(args, model)
    model = model.to(device).eval()
    _STATS_MODEL = model
    if not getattr(model, 'roi_enabled', False):
        raise RuntimeError('the model was built without the ROI branch '
                           '(pass --roi or check args.roi)')
    print(f"[info] model: {type(model).__name__}, ROI points={model.roi_num_points}, "
          f"topk={model.roi_topk}, radius={model.roi_radius}, "
          f"expand_factor={model.roi_expand_factor}, max_radius={model.roi_max_radius}")

    # ---- dataset -------------------------------------------------------
    _train_ds, test_ds = _Pipeline.get_datasets(args)
    mean_rgb = np.asarray(test_ds.mean_rgb, dtype=np.float32)
    print(f'[info] test set: {len(test_ds)} samples (split={test_ds.split}), '
          f'use_color={args.use_color}')

    loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                        num_workers=cli.num_workers, pin_memory=(device.type == 'cuda'),
                        drop_last=False, persistent_workers=False)

    # ---- loop ----------------------------------------------------------
    summary = {
        'num_requested': int(cli.num_samples),
        'num_ok': 0,
        'num_failed': 0,
        'start_index': int(cli.start_index),
        'config': {
            'config_yaml': cli.config,
            'checkpoint_path': cli.checkpoint_path,
            'data_root': cli.data_root,
            'output_dir': cli.output_dir,
            'num_samples': cli.num_samples,
            'start_index': cli.start_index,
            'device': cli.device,
            'panel_size': [cli.panel_width, cli.panel_height],
            'cmap': cli.cmap,
            'score_norm': cli.score_norm,
            'q_low': cli.q_low,
            'q_high': cli.q_high,
            'camera_dir': list(cli.camera_dir),
            'up': list(cli.up),
            'scene_max_points': cli.scene_max_points,
            'roi': {
                'num_points': int(model.roi_num_points),
                'topk': int(model.roi_topk),
                'radius': float(model.roi_radius),
                'max_radius': float(model.roi_max_radius),
                'expand_factor': float(model.roi_expand_factor),
                'tau': float(model.roi_tau),
            },
            'model': args.model,
            'num_decoder_layers': int(args.num_decoder_layers),
            'num_queries': int(args.num_target),
            'use_color': bool(args.use_color),
            'detect_intermediate': bool(args.detect_intermediate),
            'joint_det': bool(args.joint_det),
            'use_soft_token_loss': bool(args.use_soft_token_loss),
            'use_contrastive_align': bool(args.use_contrastive_align),
            'self_attend': bool(args.self_attend),
            'butd': bool(args.butd),
            'torch': torch.__version__,
        },
        'samples': [],
        'failures': [],
        'warnings': [],
        'backends': [],
    }

    tic = time.time()
    done = 0
    for step, batch_data in enumerate(loader):
        dataset_index = step
        if dataset_index < cli.start_index:
            continue
        if done >= cli.num_samples:
            break

        sample_id = f'sample_{done + 1:06d}'
        try:
            batch_data = move_to_device(batch_data, device)
            inputs = pipeline._get_inputs(batch_data)
            payload = collect_sample(model, args, inputs, batch_data,
                                     dataset_index, sample_id, mean_rgb=mean_rgb)

            sample_dir = save_payload(cli.output_dir, payload)
            verify_saved(sample_dir, payload)

            colorizer = build_score_colorizer(
                payload['entropy_scores'], cmap=cli.cmap, mode=cli.score_norm,
                q_low=cli.q_low, q_high=cli.q_high)
            backends = build_figure(payload, colorizer, cli,
                                    os.path.join(sample_dir, 'visualization.png'))

            with open(os.path.join(sample_dir, 'metadata.json'), 'w') as fid:
                json.dump(metadata(payload, model, args), fid, indent=2,
                          ensure_ascii=False)

            summary['num_ok'] += 1
            summary['backends'] = sorted(set(summary['backends']) | set(backends))
            summary['samples'].append({
                'sample_id': sample_id,
                'dataset_index': dataset_index,
                'scan_id': payload['scan_id'],
                'query_index': payload['query_index'],
                'iou_3d': payload['iou_3d'],
                'roi_num_candidates': payload['roi_num_candidates'],
                'roi_radius_used': payload['roi_radius_used'],
                'roi_padded': payload['roi_padded'],
                'num_roi_points': int(payload['roi_points_xyz'].shape[0]),
                'pred_box': list_or_none(payload['pred_box']),
                'gt_box': payload['gt_box'].tolist(),
            })
            for message in payload['warnings']:
                summary['warnings'].append(f'{sample_id}: {message}')
            print(f"[{done + 1}/{cli.num_samples}] {sample_id} "
                  f"(idx={dataset_index}, {payload['scan_id']}) ok, "
                  f"IoU={payload['iou_3d']:.3f}, "
                  f"candidates={payload['roi_num_candidates']}"
                  + (f", warnings={payload['warnings']}" if payload['warnings'] else ''))
            done += 1
        except Exception:
            summary['num_failed'] += 1
            summary['failures'].append({
                'sample_id': sample_id,
                'dataset_index': dataset_index,
                'error': traceback.format_exc(),
            })
            print(f'[error] sample {sample_id} (idx={dataset_index}) failed:')
            traceback.print_exc()
            done += 1        # keep the sample numbering without gaps

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
