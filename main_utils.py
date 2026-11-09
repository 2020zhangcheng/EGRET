# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
# Parts adapted from Group-Free
# Copyright (c) 2021 Ze Liu. All Rights Reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------
"""Shared utilities for all main scripts."""

import argparse
import contextlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from models import HungarianMatcher, SetCriterion, compute_hungarian_loss
from utils import get_scheduler, setup_logger

from utils import record_tensorboard

from tqdm import tqdm

def parse_option():
    """Parse cmd arguments."""
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--num_target', type=int, default=256,
                        help='Proposal number')
    parser.add_argument('--sampling', default='kps', type=str,
                        help='Query points sampling method (kps, fps)')

    # Transformer
    parser.add_argument('--num_encoder_layers', default=3, type=int)
    parser.add_argument('--num_decoder_layers', default=6, type=int)    # 6
    parser.add_argument('--self_position_embedding', default='loc_learned',
                        type=str, help='(none, xyz_learned, loc_learned)')
    parser.add_argument('--self_attend', action='store_true')
    parser.add_argument('--model', type=str, default='BeaUTyDETR')

    # Loss
    parser.add_argument('--query_points_obj_topk', default=4, type=int)
    parser.add_argument('--use_contrastive_align', action='store_true')
    parser.add_argument('--use_soft_token_loss', action='store_true')
    parser.add_argument('--detect_intermediate', action='store_true')
    parser.add_argument('--joint_det', action='store_true')

    # Data
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size during training')
    parser.add_argument('--dataset', type=str, default=['scanrefer'],
                        nargs='+', help='list of datasets to train on')
    parser.add_argument('--test_dataset', default='scanrefer')
    parser.add_argument('--data_root', default='./')
    parser.add_argument('--use_height', action='store_true',
                        help='Use height signal in input.')
    parser.add_argument('--use_color', action='store_true',
                        help='Use RGB color in input.')     # color
    parser.add_argument('--use_multiview', action='store_true')
    parser.add_argument('--wo_obj_name', default='None')    # grounding without object name
    parser.add_argument('--butd', action='store_true')
    parser.add_argument('--butd_gt', action='store_true')
    parser.add_argument('--butd_cls', action='store_true')
    parser.add_argument('--augment_det', action='store_true')
    parser.add_argument('--num_workers', type=int, default=4)

    # Training
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=400)
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--lr_backbone", default=1e-4, type=float)
    parser.add_argument("--text_encoder_lr", default=1e-5, type=float)
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=["step", "cosine"])
    parser.add_argument('--lr_decay_epochs', type=int, default=[280, 340],
                        nargs='+', help='when to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='for step scheduler. decay rate for lr')
    parser.add_argument('--clip_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--bn_momentum', type=float, default=0.1)
    parser.add_argument('--syncbn', action='store_true')
    parser.add_argument('--warmup-epoch', type=int, default=-1)
    parser.add_argument('--warmup-multiplier', type=int, default=100)
    parser.add_argument('--frozen', action='store_true')
    parser.add_argument('--small_lr',  default=False,action='store_true')

    # Entropy-gradient guided ROI fusion (off = original 1024-seed pipeline)
    parser.add_argument('--roi', action='store_true',
                        help='enable the entropy-gradient guided ROI re-sampling')
    parser.add_argument('--roi_points', type=int, default=512,
                        help='number of ROI points sampled by FPS')
    parser.add_argument('--roi_topk', type=int, default=32,
                        help='number of Top entropy-gradient seeds (ball centres)')
    parser.add_argument('--roi_radius', type=float, default=0.3,
                        help='ball radius (m) of the ROI union')
    parser.add_argument('--roi_max_radius', type=float, default=2.0)
    parser.add_argument('--roi_tau', type=float, default=0.0,
                        help='softmax temperature over the queries (0 = auto)')

    # ---- ROI ranking rule --------------------------------------------------
    # field        : box-proposal field -> robust single ball   (recommended)
    # entropy_ball : entropy-gradient norm -> robust single ball
    # entropy      : Top-K entropy gradient -> union of K balls  (ablation /
    #                visualisation only)
    # fuse_mul / fuse_min / fuse_gate : field fused with the entropy-gradient
    #                ranking (product / AND-gate / soft gate)
    parser.add_argument('--roi_selector', type=str, default='field',
                        choices=['field', 'entropy_ball', 'entropy',
                                 'fuse_mul', 'fuse_min', 'fuse_gate'],
                        help='ROI ranking rule (baseline = do not pass --roi)')
    parser.add_argument('--roi_k_min', type=int, default=4,
                        help='minimum number of reliable seeds kept')
    parser.add_argument('--roi_nms_radius', type=float, default=0.5,
                        help='min separation (m) between two reliable seeds')
    parser.add_argument('--roi_mass_frac', type=float, default=0.9,
                        help='stop selecting seeds after this fraction of the mass')
    parser.add_argument('--roi_score_ratio', type=float, default=0.3,
                        help='relative score threshold w.r.t. the maximum')
    parser.add_argument('--roi_radius_q', type=float, default=0.9,
                        help='weighted quantile of the seed distances = support')
    parser.add_argument('--roi_radius_margin', type=float, default=0.35,
                        help='margin (m) added to the adaptive radius')
    parser.add_argument('--roi_radius_min', type=float, default=0.4)
    parser.add_argument('--roi_radius_max', type=float, default=2.0)
    parser.add_argument('--roi_topm', type=int, default=4,
                        help='number of box hypotheses fused into the proposal field')
    parser.add_argument('--roi_fuse_alpha', type=float, default=1.0,
                        help='strength of the entropy-gradient modulation in '
                             '--roi_selector fuse_gate (0 = field only)')

    # AMP / gradient accumulation / resume
    parser.add_argument('--amp', action='store_true', help='automatic mixed precision')
    parser.add_argument('--accum_steps', type=int, default=1,
                        help='gradient accumulation steps (effective batch x accum_steps)')
    parser.add_argument('--resume', action='store_true',
                        help='restore epoch / optimizer / scheduler from --checkpoint_path')

    # io
    parser.add_argument('--checkpoint_path', default=None,
                        help='Model checkpoint path')
    parser.add_argument('--log_dir', default='log',
                        help='Dump dir to save model checkpoint')
    parser.add_argument('--exp', default='exp',
                        help='exp name to save model checkpoint')
    parser.add_argument('--print_freq', type=int, default=10)  # batch-wise
    parser.add_argument('--save_freq', type=int, default=10)  # epoch-wise
    parser.add_argument('--val_freq', type=int, default=5)  # epoch-wise

    # others
    # torch.distributed.launch / torchrun pass --local-rank (hyphen);
    # keep --local_rank for backward compatibility.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1,
                        help='local rank for DistributedDataParallel')
    parser.add_argument('--ap_iou_thresholds', type=float, default=[0.25, 0.5],
                        nargs='+', help='A list of AP IoU thresholds')
    parser.add_argument("--rng_seed", type=int, default=111, help='manual seed')
    parser.add_argument("--debug", action='store_true',
                        help="try to overfit few samples")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_train', action='store_true')
    parser.add_argument('--pp_checkpoint', default=None)    # pointnet checkpoint
    parser.add_argument('--reduce_lr', action='store_true')

    args, _ = parser.parse_known_args()

    args.eval = args.eval or args.eval_train

    return args

# BRIEF load checkpoint.
def _load_state_dict(args, model, state_dict, tag=''):
    """``load_state_dict`` that also reports how many keys actually matched."""
    model_keys = set(model.state_dict().keys())
    state_keys = set(state_dict.keys())
    matched = len(model_keys & state_keys)
    missing = sorted(model_keys - state_keys)
    unexpected = sorted(state_keys - model_keys)
    missing = [k for k in missing if 'position_ids' not in k]
    unexpected = [k for k in unexpected if 'position_ids' not in k]
    model.load_state_dict(state_dict, strict=False)
    print(f"=> loaded {tag}'{args.checkpoint_path}' "
          f"({matched} matched, {len(missing)} missing, {len(unexpected)} unexpected)")
    if missing and len(missing) <= 12:
        print(f'   missing keys: {missing}')
    if unexpected and len(unexpected) <= 12:
        print(f'   unexpected keys: {unexpected}')
    return matched


def load_checkpoint(args, model, optimizer, scheduler):
    """Load from checkpoint.

    ``--resume`` additionally restores ``epoch``/``optimizer``/``scheduler``.
    """
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)

    state = checkpoint['model']
    _load_state_dict(args, model, state, tag='')

    if getattr(args, 'resume', False) and not args.eval:
        args.start_epoch = int(checkpoint.get('epoch', 0)) + 1
        if checkpoint.get('optimizer') is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scheduler') is not None and scheduler is not None:
            # keep the horizon of the *current* run (--max_epoch may have been
            # extended since the checkpoint was written)
            cur_tmax = getattr(scheduler, 'T_max', None)
            scheduler.load_state_dict(checkpoint['scheduler'])
            if cur_tmax is not None:
                scheduler.T_max = cur_tmax
            if getattr(scheduler, 'last_epoch', 0) >= getattr(scheduler, 'T_max', 1e18):
                print('   [WARN] the restored scheduler is already at the end of its '
                      'cosine horizon, the learning rate will stay at eta_min')
        print(f"=> resumed from epoch {args.start_epoch - 1} "
              f"(next epoch {args.start_epoch})")

    print("=> loaded successfully '{}' (epoch {})".format(
        args.checkpoint_path, checkpoint.get('epoch')
    ))
    del checkpoint
    torch.cuda.empty_cache()


# BRIEF save model.
def save_checkpoint(args, epoch, model, optimizer, scheduler, save_cur=False):
    """Save checkpoint if requested."""
    if save_cur or epoch % args.save_freq == 0:
        state = {
            'config': args,
            'save_path': '',
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'epoch': epoch,
        }

        spath = os.path.join(args.log_dir, f'ckpt_epoch_{epoch}.pth')
        state['save_path'] = spath
        torch.save(state, spath)
        print("Saved in {}".format(spath))
    else:
        print("not saving checkpoint")


class BaseTrainTester:
    """Basic train/test class to be inherited."""

    # logger.
    def __init__(self, args):
        """Initialize."""
        name = args.log_dir.split('/')[-1]  # log_dir: './logs/eda', name: eda
        
        # Create log dir
        args.log_dir = os.path.join(
            args.log_dir,
            ','.join(args.dataset),
            args.exp,
            f'{int(time.time())}'
        )
        os.makedirs(args.log_dir, exist_ok=True)

        # Create logger
        self.logger = setup_logger(
            output=args.log_dir, distributed_rank=dist.get_rank(),
            name=name
        )

        # tensorboard
        self.tensorboard = record_tensorboard.TensorBoard(args.log_dir, distributed_rank=dist.get_rank())

        # Save config file and initialize tb writer
        if dist.get_rank() == 0:
            path = os.path.join(args.log_dir, "config.json")
            with open(path, 'w') as f:
                json.dump(vars(args), f, indent=2)
            self.logger.info("Full config saved to {}".format(path))
            self.logger.info(str(vars(args)))

    @staticmethod
    def get_datasets(args):
        """Initialize datasets."""
        train_dataset = None
        test_dataset = None
        return train_dataset, test_dataset


    # BRIEF dataloader.
    def get_loaders(self, args):
        """Initialize data loaders."""
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            np.random.seed(np.random.get_state()[1][0] + worker_id)

        # Datasets
        train_dataset, test_dataset = self.get_datasets(args)
        
        # Samplers and loaders
        g = torch.Generator()
        g.manual_seed(0)

        if args.eval:
            train_loader = None
        else:
            train_sampler = DistributedSampler(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,      # TODO 
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                pin_memory=True,
                sampler=train_sampler,
                drop_last=True,
                generator=g
            )
        
        test_sampler = DistributedSampler(test_dataset, shuffle=False)
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker,
            pin_memory=True,
            sampler=test_sampler,
            drop_last=False,
            generator=g
        )
        return train_loader, test_loader

    @staticmethod
    def get_model(args):
        """Initialize the model."""
        return None

    @staticmethod
    def get_criterion(args):
        """Get loss criterion for training."""
        losses = ['boxes', 'labels', 'masks']
        if args.use_contrastive_align:
            losses.append('contrastive_align')
        matcher = HungarianMatcher(1, 0, 2, args.use_soft_token_loss)
        set_criterion = SetCriterion(
            matcher=matcher,
            losses=losses, eos_coef=0.1, temperature=0.07
        )
        criterion = compute_hungarian_loss

        return criterion, set_criterion

    @staticmethod
    def get_optimizer(args, model):
        """Initialize optimizer."""
        if args.frozen:
            print("-------------------------------frozen backbone parameters------------------------------------")
            for n, p in model.named_parameters():
                if "backbone_net" in n or "text_encoder" in n:
                    p.requires_grad = False
                else:
                    p.requires_grad = True
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" not in n
                        and "text_encoder" not in n
                        and p.requires_grad
                    ]
                },
                {
                    "params": [],
                    "lr": args.lr_backbone
                },
                {
                    "params": [],
                    "lr": args.text_encoder_lr
                }
            ]
        elif args.small_lr:
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "x_mask" in n or "x_query" in n or "seed_decoder" in n
                    ],
                    "lr": args.lr
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" not in n and "text_encoder" not in n 
                        and "x_mask" not in n and "x_query" not in n and "seed_decoder" not in n
                        and p.requires_grad
                    ],
                    "lr": args.lr * 0.01
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" in n and p.requires_grad
                    ],
                    "lr": args.lr_backbone * 0.01
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "text_encoder" in n and p.requires_grad
                    ],
                    "lr": args.text_encoder_lr * 0.01
                }
            ]
        else:
            param_dicts = [
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" not in n and "text_encoder" not in n
                        and p.requires_grad
                    ]
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "backbone_net" in n and p.requires_grad
                    ],
                    "lr": args.lr_backbone
                },
                {
                    "params": [
                        p for n, p in model.named_parameters()
                        if "text_encoder" in n and p.requires_grad
                    ],
                    "lr": args.text_encoder_lr
                }
            ]
        optimizer = optim.AdamW(param_dicts,
                                lr=args.lr,
                                weight_decay=args.weight_decay)
        return optimizer


    # BRIEF main training/testing
    def main(self, args):
        """Run main training/testing pipeline."""
        # Get loaders
        train_loader, test_loader = self.get_loaders(args)
        if not args.eval:
            n_data = len(train_loader.dataset)
            self.logger.info(f"length of training dataset: {n_data}")
        n_data = len(test_loader.dataset)
        self.logger.info(f"length of testing dataset: {n_data}")

        # Get model
        model = self.get_model(args)

        # Get criterion
        criterion, set_criterion = self.get_criterion(args)

        # Get optimizer
        optimizer = self.get_optimizer(args, model)

        # Get scheduler (one step per *optimizer* update, i.e. per accumulation window)
        accum_steps = max(int(getattr(args, 'accum_steps', 1)), 1)
        if not args.eval:
            scheduler = get_scheduler(
                optimizer, max(len(train_loader) // accum_steps, 1), args)
        else:
            scheduler = None
        
        # Move model to devices
        if torch.cuda.is_available():
            if args.syncbn and torch.cuda.device_count() > 1:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = model.cuda()

        # note Distributed Data-Parallel Training (DDP)
        model = DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False, find_unused_parameters=True
        )

        # Check for a checkpoint
        if args.checkpoint_path:
            assert os.path.isfile(args.checkpoint_path)
            load_checkpoint(args, model, optimizer, scheduler)
        
        # ##############################################
        # NOTE [eval-only] Just eval and end execution #
        # ##############################################
        if args.eval:
            print("Testing evaluation.....................")
            self.evaluate_one_epoch(
                args.start_epoch, test_loader,
                model, criterion, set_criterion, args
            )
            return

        # ##############################
        # NOTE Training and Validation #
        # ##############################
        # AMP scaler (no-op when --amp is not given)
        if hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('cuda', enabled=bool(getattr(args, 'amp', False)))
        else:  # pragma: no cover - older torch
            scaler = torch.cuda.amp.GradScaler(enabled=bool(getattr(args, 'amp', False)))
        if getattr(args, 'amp', False):
            self.logger.info('AMP enabled (GradScaler, fp16 autocast)')

        for epoch in range(args.start_epoch, args.max_epoch + 1):
            train_loader.sampler.set_epoch(epoch)
            tic = time.time()

            # train *
            self.train_one_epoch(
                epoch, train_loader, model,
                criterion, set_criterion,
                optimizer, scheduler, args, scaler=scaler
            )
            
            # log
            self.logger.info(
                'epoch {}, total time {:.2f}, '
                'lr_base {:.5f}, lr_pointnet {:.5f}'.format(
                    epoch, (time.time() - tic),
                    optimizer.param_groups[0]['lr'],
                    optimizer.param_groups[1]['lr']
                )
            )

            # save model and validate
            if epoch % args.val_freq == 0:
                if dist.get_rank() == 0:
                    save_checkpoint(args, epoch, model, optimizer, scheduler)
                
                # validate *
                print("Test evaluation.......")
                self.evaluate_one_epoch(
                    epoch, test_loader,
                    model, criterion, set_criterion, args
                )

        # Training is over
        save_checkpoint(args, 'last', model, optimizer, scheduler, True)
        saved_path = os.path.join(args.log_dir, 'ckpt_epoch_last.pth')
        self.logger.info("Saved in {}".format(saved_path))
        self.evaluate_one_epoch(
            args.max_epoch, test_loader,
            model, criterion, set_criterion, args
        )
        return saved_path

    @staticmethod
    def _to_gpu(data_dict):
        if torch.cuda.is_available():
            for key in data_dict:
                if isinstance(data_dict[key], torch.Tensor):
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)
        return data_dict

    @staticmethod
    def _get_inputs(batch_data):
        return {
            'point_clouds': batch_data['point_clouds'].float(),
            'text': batch_data['utterances']
        }

    @staticmethod
    def _compute_loss(end_points, criterion, set_criterion, args):
        loss, end_points = criterion(
            end_points, args.num_decoder_layers,
            set_criterion,
            query_points_obj_topk=args.query_points_obj_topk
        )
        return loss, end_points

    @staticmethod
    def _accumulate_stats(stat_dict, end_points):
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'ratio' in key:
                if key not in stat_dict:
                    stat_dict[key] = 0
                if isinstance(end_points[key], (float, int)):
                    stat_dict[key] += end_points[key]
                else:
                    stat_dict[key] += end_points[key].item()
        return stat_dict


    # BRIEF Training
    def train_one_epoch(self, epoch, train_loader, model,
                        criterion, set_criterion,
                        optimizer, scheduler, args, scaler=None):
        """
        Run a single epoch.

        Supports automatic mixed precision (``--amp``, ``scaler``) and gradient
        accumulation (``--accum_steps``).

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        stat_dict = {}  # collect statistics
        model.train()  # set model to training mode

        accum_steps = max(int(getattr(args, 'accum_steps', 1)), 1)
        use_amp = bool(getattr(args, 'amp', False)) and scaler is not None
        n_batches = len(train_loader)
        optimizer.zero_grad(set_to_none=True)

        # Loop over batches
        train_loader = tqdm(train_loader, ascii=True)
        for batch_idx, batch_data in enumerate(train_loader):
            # Move to GPU
            batch_data = self._to_gpu(batch_data)
            # get the input data: pointcloud and text
            inputs = self._get_inputs(batch_data)

            # stop DDP from all-reducing on partial accumulation windows
            is_last = (batch_idx + 1) == n_batches
            sync_grads = ((batch_idx + 1) % accum_steps == 0) or is_last
            no_sync = hasattr(model, 'no_sync') and not sync_grads
            ctx = model.no_sync() if no_sync else contextlib.nullcontext()

            with ctx:
                # note Forward pass
                with torch.amp.autocast('cuda', enabled=use_amp):
                    end_points = model(inputs)

                # note Compute loss and gradients, update parameters.
                for key in batch_data:
                    assert (key not in end_points)
                    end_points[key] = batch_data[key]
                with torch.amp.autocast('cuda', enabled=use_amp):
                    loss, end_points = self._compute_loss(
                        end_points, criterion, set_criterion, args
                    )
                # normalize so that the accumulated gradient equals the mean
                scaled_loss = loss / accum_steps
                if use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

            if sync_grads:
                if use_amp:
                    scaler.unscale_(optimizer)
                if args.clip_norm > 0:
                    grad_total_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.clip_norm
                    )
                    stat_dict['grad_norm'] = grad_total_norm

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # Accumulate statistics and print out
            stat_dict = self._accumulate_stats(stat_dict, end_points)

            # print loss
            if (batch_idx + 1) % args.print_freq == 0:
                # Terminal logs
                self.logger.info(
                    f'Train: [{epoch}][{batch_idx + 1}/{len(train_loader)}]  '  # Train: [30][2000/2432]
                )
                self.logger.info(''.join([
                    f'{key} {stat_dict[key] / (batch_idx + 1):.4f} \t'
                    for key in sorted(stat_dict.keys())
                    if 'loss' in key and 'proposal_' not in key
                    and 'last_' not in key and 'head_' not in key
                ])) # loss，loss_bbox，loss_ce，loss_sem_align，loss_giou，query_points_generation_loss

                # # reset stat_dict
                # for key in sorted(stat_dict.keys()):
                #     stat_dict[key] = 0
                
                if dist.get_rank() == 0:
                    for key in self.tensorboard.item["train_loss"]:
                        self.tensorboard.item["train_loss"][key] = stat_dict[key] / (batch_idx + 1)
                    self.tensorboard.dump_tensorboard("train_loss", (epoch-1)*len(train_loader)+batch_idx+1)

        # tensorboard
        if dist.get_rank() == 0:
            # loss
            for key in self.tensorboard.item["train_loss"]:
                self.tensorboard.item["train_loss"][key] = stat_dict[key] / len(train_loader)
            self.tensorboard.dump_tensorboard("train_loss", (epoch-1)*len(train_loader)+batch_idx+1)
            # lr
            self.tensorboard.item["train_lr"]["lr_base"] = optimizer.param_groups[0]['lr']
            self.tensorboard.item["train_lr"]["lr_pointnet"] = optimizer.param_groups[1]['lr']
            self.tensorboard.dump_tensorboard("train_lr", epoch)

    # BRIEF eval 
    @torch.no_grad()
    def _main_eval_branch(self, batch_idx, batch_data, test_loader, model,
                          stat_dict,
                          criterion, set_criterion, args):
        # Move to GPU
        batch_data = self._to_gpu(batch_data)
        inputs = self._get_inputs(batch_data)
        if "train" not in inputs:
            inputs.update({"train": False})
        else:
            inputs["train"] = False

        # STEP Forward pass
        end_points = model(inputs)

        # from thop import profile
        # macs, _ = profile(model, inputs=(inputs, ))
        # print(f"Total FLOPs: {macs} (or {macs / 1e9} GFLOPs)")

        # STEP Compute loss
        for key in batch_data:
            assert (key not in end_points)
            end_points[key] = batch_data[key]
        _, end_points = self._compute_loss(
            end_points, criterion, set_criterion, args
        )
        for key in end_points:
            if 'pred_size' in key:
                end_points[key] = torch.clamp(end_points[key], min=1e-6)

        # Accumulate statistics and print out
        stat_dict = self._accumulate_stats(stat_dict, end_points)
        if (batch_idx + 1) % args.print_freq == 0:
            self.logger.info(f'Eval: [{batch_idx + 1}/{len(test_loader)}]  ')
            self.logger.info(''.join([
                f'{key} {stat_dict[key] / (float(batch_idx + 1)):.4f} \t'
                for key in sorted(stat_dict.keys())
                if 'loss' in key and 'proposal_' not in key
                and 'last_' not in key and 'head_' not in key
            ]))
        return stat_dict, end_points

    @torch.no_grad()
    def evaluate_one_epoch(self, epoch, test_loader,
                           model, criterion, set_criterion, args):
        """
        Eval grounding after a single epoch.

        Some of the args:
            model: a nn.Module that returns end_points (dict)
            criterion: a function that returns (loss, end_points)
        """
        return None
