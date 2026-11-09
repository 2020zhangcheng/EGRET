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
import copy
import numpy as np
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
from .backbones.pointnet2_backbone import Pointnet2Backbone
from .backbones.pv_backbone import pv_backbone
from .heads.prediction_heads import (
    PointsObjClsModule, GeneralSamplingModule,
    ClsAgnosticPredictHead
)
from .heads.spatial_encoding import (
    calc_pairwise_locs, PositionEmbeddingLearned
)
from .heads.query_generation import GumbelSampling
from .heads.text_decoder import SWA, FFN
from .heads.transformer_layers import (
    BiEncoder, BiEncoderLayer, BiDecoderLayer
)
from utils.scatter_util import scatter_mean
from utils.roi_sampling import sample_roi_points_torch
import pointnet2_utils
import einops


def _fp32_ctx():
    """Autocast-disabled context.

    The custom CUDA ops of the point cloud backbone (``query_ball_point`` /
    ``grouping_operation``) only accept ``float32``, so every block that calls
    them must run outside the AMP autocast region.
    """
    return torch.amp.autocast('cuda', enabled=False)


class PVGround(nn.Module):
    """
    3D language grounder.

    Args:
        num_class (int): number of semantics classes to predict
        num_obj_class (int): number of object classes
        input_feature_dim (int): feat_dim of pointcloud (without xyz)
        num_queries (int): Number of queries generated
        num_decoder_layers (int): number of decoder layers
        self_position_embedding (str or None): how to compute pos embeddings
        contrastive_align_loss (bool): contrast queries and token features
        d_model (int): dimension of features
        butd (bool): use detected box stream
        pointnet_ckpt (str or None): path to pre-trained pp++ checkpoint
        self_attend (bool): add self-attention in encoder
    """

    def __init__(self, model_cfg, num_class=256, num_obj_class=485,
                 input_feature_dim=3,
                 num_queries=256,
                 num_decoder_layers=6, self_position_embedding='loc_learned',
                 contrastive_align_loss=True,
                 d_model=288, butd=True, pointnet_ckpt=None, data_path=None,
                 self_attend=True, roi=None):
        """Initialize layers."""
        super().__init__()

        self.num_queries = num_queries
        self.num_decoder_layers = num_decoder_layers
        self.self_position_embedding = self_position_embedding
        self.contrastive_align_loss = contrastive_align_loss
        self.butd = butd

        # ---------------- entropy-gradient guided ROI fusion ----------------
        # roi=None keeps the original 1024-seed pipeline untouched.
        self.roi_cfg = dict(roi) if roi else None
        self.roi_enabled = bool(self.roi_cfg and self.roi_cfg.get('enabled', True))
        if self.roi_enabled:
            self.roi_num_points = int(self.roi_cfg.get('num_points', 256))
            self.roi_topk = int(self.roi_cfg.get('topk', 32))
            self.roi_radius = float(self.roi_cfg.get('radius', 0.4))
            self.roi_max_radius = float(self.roi_cfg.get('max_radius', 2.0))
            self.roi_tau = float(self.roi_cfg.get('tau', 0.0))
            # ROI ranking / geometry:
            #   'field'        box-proposal field  -> robust single ball (default)
            #   'entropy_ball' entropy-gradient    -> robust single ball
            #   'entropy'      Top-K entropy grad  -> union of K balls (legacy,
            #                  kept for the visualisation / ablation only)
            self.roi_selector = str(self.roi_cfg.get('selector', 'field'))
            self.roi_k_min = int(self.roi_cfg.get('k_min', 4))
            self.roi_nms_radius = float(self.roi_cfg.get('nms_radius', 0.5))
            self.roi_mass_frac = float(self.roi_cfg.get('mass_frac', 0.9))
            self.roi_score_ratio = float(self.roi_cfg.get('score_ratio', 0.3))
            self.roi_radius_q = float(self.roi_cfg.get('radius_q', 0.9))
            self.roi_radius_margin = float(self.roi_cfg.get('radius_margin', 0.35))
            self.roi_radius_min = float(self.roi_cfg.get('radius_min', 0.4))
            self.roi_radius_max = float(self.roi_cfg.get('radius_max', 2.0))
            self.roi_topm = int(self.roi_cfg.get('topm', 4))
            # strength of the entropy-gradient modulation used by fuse_gate:
            #   R = rank(f) * (1 + alpha * rank(s)) / (1 + alpha)
            # alpha = 0 -> field only, larger -> the competition-sensitive seeds
            # inside the box region get relatively more weight.
            self.roi_fuse_alpha = float(self.roi_cfg.get('fuse_alpha', 1.0))
            # A single learned type vector separating ROI points from seeds.
            # Zero-initialised -> no behaviour change at initialisation, fully
            # trainable afterwards (it is the only *new* parameter).
            self.roi_type_embedding = nn.Parameter(torch.zeros(d_model))
#-------------------text-head decoder-------------------------------------
        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.swa_layers = nn.ModuleList([])
        self.swa_ffn_layers = nn.ModuleList([])
        self.rel_encoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 288))
        for i in range(3):
            self.swa_layers.append(SWA(d_model, nhead=8,dropout=0.2))
            self.swa_ffn_layers.append(FFN(d_model, hidden_dim=128, dropout=0.2, activation_fn='relu'))

        self.backbone_net = pv_backbone(model_cfg)

        if input_feature_dim == 3 and pointnet_ckpt is not None:
            self.backbone_net.load_state_dict(torch.load(
                pointnet_ckpt
            ), strict=False)

        # Text Encoder
        # # (1) online
        # t_type = "roberta-base"
        # NOTE (2) offline: load from the local folder.
        t_type = f'{data_path}roberta-base/'
        self.tokenizer = RobertaTokenizerFast.from_pretrained(t_type, local_files_only=True)
        self.text_encoder = RobertaModel.from_pretrained(t_type, local_files_only=True)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_projector = nn.Sequential(
            nn.Linear(self.text_encoder.config.hidden_size, d_model),
            nn.LayerNorm(d_model, eps=1e-12),
            nn.Dropout(0.1)
        )

        # Box encoder
        if self.butd:
            self.butd_class_embeddings = nn.Embedding(num_obj_class, 768)
            saved_embeddings = torch.from_numpy(np.load(
                'data/class_embeddings3d.npy', allow_pickle=True
            ))
            self.butd_class_embeddings.weight.data.copy_(saved_embeddings)
            self.butd_class_embeddings.requires_grad = False
            self.class_embeddings = nn.Linear(768, d_model - 128)
            self.box_embeddings = PositionEmbeddingLearned(6, 128)

        # Cross-encoder
        self.pos_embed = PositionEmbeddingLearned(3, d_model)
        bi_layer = BiEncoderLayer(
            d_model, dropout=0.1, activation="relu",
            n_heads=8, dim_feedforward=256,
            self_attend_lang=self_attend, self_attend_vis=self_attend,
            use_butd_enc_attn=butd
        )
        self.cross_encoder = BiEncoder(bi_layer, 3)

        # Mask Feats Generation layer
        self.x_mask = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.x_query = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, 1), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model * 2, 1),
            nn.ReLU(), 
            nn.Conv1d(d_model * 2, d_model, 1)
            )
        self.text_query_proj=nn.Sequential(
            nn.Linear(d_model,2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model,2*d_model),
            nn.ReLU(),
            nn.Linear(2*d_model,d_model)
            )
        self.super_grouper = pointnet2_utils.QueryAndGroup(radius=0.2, nsample=2, use_xyz=False, normalize_xyz=True)

        # Query initialization
        self.gumbel = GumbelSampling(288, 256)
        self.decoder_query_proj = nn.Conv1d(d_model, d_model, kernel_size=1)

        # Proposal (layer for size and center)
        self.proposal_head = ClsAgnosticPredictHead(
            num_class, 1, num_queries, d_model,
            objectness=False, heading=False,
            compute_sem_scores=True
        )

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.decoder.append(BiDecoderLayer(
                d_model, n_heads=8, dim_feedforward=256,
                dropout=0.1, activation="relu",
                self_position_embedding=self_position_embedding, butd=self.butd
            ))

        # Prediction heads
        self.prediction_heads = nn.ModuleList()
        for _ in range(self.num_decoder_layers):
            self.prediction_heads.append(ClsAgnosticPredictHead(
                num_class, 1, num_queries, d_model,
                objectness=False, heading=False,
                compute_sem_scores=True
            ))

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_image = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )
            self.contrastive_align_projection_text = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 64)
            )

        # Init
        self.init_bn_momentum()


    # BRIEF visual and text backbones.
    def _run_backbones(self, inputs):
        """Run visual and text backbones.

        Returns:
            (end_points, voxel_batch): ``voxel_batch`` is kept so that the ROI
            points can be projected into the very same voxel/BEV features.
        """
        # step 1. Visual encoder
        end_points = {}
        with _fp32_ctx():   # custom CUDA ops are float32 only
            voxel_batch = self.backbone_net(inputs)
        end_points['seed_inds'] = voxel_batch['inds']
        end_points['seed_xyz'] = voxel_batch['point_coords']
        end_points['seed_features'] = voxel_batch['point_features']

        
        # step 2. Text encoder
        tokenized = self.tokenizer(
            inputs['text'], padding="longest", return_tensors="pt", max_length=256, truncation=True
        ).to(inputs['points'].device)
        
        encoded_text = self.text_encoder(**tokenized)
        text_feats = self.text_projector(encoded_text.last_hidden_state)

        # Invert attention mask that we get from huggingface
        # because its the opposite in pytorch transformer
        text_attention_mask = tokenized.attention_mask.ne(1).bool()

        end_points['text_feats'] = text_feats
        end_points['text_attention_mask'] = text_attention_mask
        end_points['tokenized'] = tokenized
        return end_points, voxel_batch

    # BRIEF generate query.
    def _generate_queries(self, xyz, features, end_points):
        # features: [B, V, F]
        features = features.transpose(1, 2).contiguous()
        # gumbel sampling
        atten = self.gumbel(features)  # [B, num_queries, N]
        xyz_gumbel = torch.bmm(atten, xyz)  # (B, V, 3)
        features = torch.bmm(atten, features).transpose(1, 2).contiguous()  # (B, F, V)
                
        end_points['query_points_xyz'] = xyz_gumbel  # (B, V, 3)
        end_points['query_points_feature'] = features  # (B, F, V)
        return end_points
    
    # segmentation prediction
    def _seg_seeds_prediction(self, query, mask_feats, end_points, prefix=''):
        ## generate seed points masks
        pred_mask_seeds = torch.einsum('bnd,bdm->bnm', query, mask_feats)
        ## mapping seed points masks to superpoints masks
        end_points[f'{prefix}pred_mask_seeds'] = pred_mask_seeds
        return pred_mask_seeds

    def get_mask(self, query, mask_feats):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
        attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
        attn_masks = attn_masks.detach()

        return pred_masks, attn_masks

    def prediction_head(self, query, superpoint_feats):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, superpoint_feats)
        return pred_scores, pred_masks, attn_masks
    

    def avg_lang_feat(self, lang_feats, lang_masks):
        lang_len = lang_masks.sum(-1)
        lang_len = lang_len.unsqueeze(-1)
        lang_len[torch.where(lang_len == 0)] = 1
        return (lang_feats * ~lang_masks.unsqueeze(-1).expand_as(lang_feats)).sum(1) / lang_len

    # ------------------------------------------------------------------ #
    # shared building blocks (used by the 1024 seeds AND by the ROI points)
    # ------------------------------------------------------------------ #
    def _encode_vision(self, points_xyz, points_features, text_feats, text_padding_mask,
                       end_points, detected_feats=None, detected_mask=None):
        """Cross-modal encoding of an arbitrary vision sequence.

        The exact same ``cross_encoder`` weights are used for the 1024 FPS seeds
        and for the 512 ROI points, so each sequence only attends *within
        itself* (``1024x1024`` + ``512x512``) instead of a full ``1536x1536``
        global attention.

        Args:
            points_xyz: ``(B, N, 3)``
            points_features: ``(B, N, 288)``
        Returns:
            ``(B, 288, N)`` encoded vision features, ``(B, L, 288)`` text features
        """
        if points_xyz.size(1) != points_features.size(1):
            raise ValueError(
                f'xyz/feature length mismatch: {tuple(points_xyz.shape)} vs '
                f'{tuple(points_features.shape)}')
        spatial_point_xyz = calc_pairwise_locs(points_xyz)
        vis_feats, text_feats = self.cross_encoder(
            vis_feats=points_features.contiguous(),
            pos_feats=self.pos_embed(points_xyz).transpose(1, 2).contiguous(),
            padding_mask=torch.zeros(
                len(points_xyz), points_xyz.size(1)
            ).to(points_xyz.device).bool(),
            text_feats=text_feats,
            text_padding_mask=text_padding_mask,
            end_points=end_points,
            detected_feats=detected_feats,
            detected_mask=detected_mask,
            spatial_point_xyz=spatial_point_xyz
        )
        vis_feats = vis_feats.transpose(1, 2).contiguous()     # (B, 288, N)
        return vis_feats, text_feats

    def _queries_and_boxes(self, points_xyz, points_features, text_feats,
                           text_padding_mask, end_points,
                           detected_feats=None, detected_mask=None):
        """STEP 5-7 of the original pipeline (query generation -> decoder).

        Args:
            points_xyz: ``(B, N, 3)`` visual positions (N = 1024 or 1536)
            points_features: ``(B, 288, N)`` encoded visual features
        Returns:
            ``(cluster_xyz, query_last, last_prefix)``
        """
        if points_xyz.size(1) != points_features.size(-1):
            raise ValueError(
                f'xyz/feature length mismatch: {tuple(points_xyz.shape)} vs '
                f'{tuple(points_features.shape)}')

        # STEP 5. Query Points Generation
        end_points = self._generate_queries(points_xyz, points_features, end_points)
        cluster_feature = end_points['query_points_feature']    # (B, F=288, V=256)
        cluster_xyz = end_points['query_points_xyz']            # (B, V=256, 3)
        query = self.decoder_query_proj(cluster_feature)
        query = query.transpose(1, 2).contiguous()              # (B, V=256, F=288)
        # projection 288 --> 64
        if self.contrastive_align_loss:
            end_points['proposal_proj_queries'] = F.normalize(
                self.contrastive_align_projection_image(query), p=2, dim=-1
            )  # [B, 256, 64]

        # STEP 6.Proposals
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature,
            base_xyz=cluster_xyz,
            end_points=end_points,
            prefix='proposal_'
        )
        base_xyz = proposal_center.detach().clone()
        base_size = proposal_size.detach().clone()
        query_mask = None
        query_last = None
        prefix = 'last_'

        # STEP 7. Decoder
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if i == self.num_decoder_layers-1 else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError

            # step Transformer Decoder Layer
            query = self.decoder[i](
                query, points_features.transpose(1, 2).contiguous(),
                text_feats, query_pos,
                query_mask,
                text_padding_mask,
                detected_feats=(
                    detected_feats if self.butd
                    else None
                ),
                detected_mask=detected_mask if self.butd else None
            )  # (B, V, F)
            # step project
            if self.contrastive_align_loss:
                end_points[f'{prefix}proj_queries'] = F.normalize(
                    self.contrastive_align_projection_image(query), p=2, dim=-1
                )

            # step box Prediction head
            base_xyz, base_size = self.prediction_heads[i](
                query.transpose(1, 2).contiguous(),     # ([B, F=288, V=256])
                base_xyz=cluster_xyz,                   # ([B, 256, 3])
                end_points=end_points,  #
                prefix=prefix
            )
            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

            query_last = query

        return cluster_xyz, query_last, prefix

    # ------------------------------------------------------------------ #
    # entropy-gradient guided ROI
    # ------------------------------------------------------------------ #
    def _selection_entropy_seed_score(self, points_xyz, seed_features_cn,
                                      text_feats, text_padding_mask, positive_map,
                                      detected_feats=None, detected_mask=None,
                                      need_grad=True):
        """One ranking pass -> entropy gradient and everything it can give.

        ``s_i = ||dH_sel/dF_i||_2`` (the definition lives in
        ``utils/entropy_gradient.py``).  Only the *decoder* part is
        differentiated: the visual features are made a detached leaf, so no
        gradient leaks into the parameters and the graph is released right
        after the single ``autograd.grad`` call.

        The same pass is reused for additional, *free* signals (no extra
        forward):

        * ``sel_prob`` ``(B, Q)``: ``P(q|m)``, the query that owns the referred
          instance -> its feature ``query_feat`` is the model's own target
          hypothesis ``q*``;
        * ``pred_center`` / ``pred_size`` ``(B, Q, 3)``: the regressed boxes;
        * ``query_feat`` ``(B, C)``: used for the text-conditioned seed mask.

        Args:
            need_grad: when False the pass runs under ``no_grad`` and the
                backward through the decoder is skipped (``score``/``grad`` are
                then ``None``).  The ``field`` / ``proposal*`` selectors do not
                use the gradient, so they take this cheaper path; the results
                are identical because nothing else depends on it.

        Returns:
            dict with ``score`` ``(B, N)`` (or None), ``grad`` ``(B, N, C)``
            (or None), ``sel_prob`` ``(B, Q)``, ``query_feat`` ``(B, C)``.
        """
        from utils import entropy_gradient as eg

        rank_end_points = {}
        was_training = self.training
        ctx = torch.enable_grad() if need_grad else torch.no_grad()
        with ctx, torch.amp.autocast('cuda', enabled=False):
            leaf = seed_features_cn.detach().float()
            if need_grad:
                leaf.requires_grad_(True)                          # (B, C, N)
            try:
                if was_training:
                    # deterministic ranking pass: running BN stats, no dropout
                    self.eval()
                _, query_last, _ = self._queries_and_boxes(
                    points_xyz, leaf, text_feats.detach().float(), text_padding_mask,
                    rank_end_points,
                    detected_feats=None if detected_feats is None else detected_feats.detach(),
                    detected_mask=detected_mask)
            finally:
                if was_training:
                    self.train()

            grad, score, prob_qm, entropy = None, None, None, None
            if need_grad:
                if 'last_sem_cls_scores' not in rank_end_points:
                    raise RuntimeError(
                        'the ranking pass did not produce last_sem_cls_scores')
                sem_prob = torch.softmax(
                    rank_end_points['last_sem_cls_scores'].float(), dim=-1)
                score_qm = eg.selection_scores(sem_prob, positive_map)  # (B, Q, M)
                tau = eg.resolve_tau(score_qm, self.roi_tau)
                prob_qm = torch.softmax(score_qm / tau, dim=1)          # (B, Q, M)
                entropy = eg.selection_entropy_scalar(prob_qm, reduce='sum')
                grad, = torch.autograd.grad(entropy, leaf, allow_unused=False)
                # (B, C, N) -> (B, N, C): the *full* gradient is kept, its
                # direction carries the signed reliability signal.
                grad = grad.detach().transpose(1, 2).contiguous().float()
                score = grad.double().norm(dim=-1).float()              # (B, N)
                if score.shape != points_xyz.shape[:2]:
                    raise RuntimeError(
                        f'seed score/xyz mismatch: {tuple(score.shape)} vs '
                        f'{tuple(points_xyz.shape[:2])}')

        if 'last_center' not in rank_end_points or 'last_pred_size' not in rank_end_points:
            raise RuntimeError('the ranking pass did not produce last_center/last_pred_size')

        sel_prob = self._query_selection(
            rank_end_points.get('last_proj_queries'),
            text_feats.detach().float(), text_padding_mask, positive_map)
        return {
            'score': score,                # (B, N) or None (no-grad selectors)
            'grad': grad,                  # (B, N, C) or None
            'sel_prob': sel_prob,          # (B, Q) P(q|m), evaluator criterion
            'pred_center': rank_end_points['last_center'].detach().float(),
            'pred_size': rank_end_points['last_pred_size'].detach().float(),
        }

    def _query_selection(self, proj_queries, text_feats, text_padding_mask,
                         positive_map):
        """``P(q|m)`` with the *same* criterion the evaluator uses.

        ``evaluate_one_epoch`` scores a decoder query by the cosine similarity
        between its contrastive projection (``proj_queries``, 64-d) and the
        *phrase tokens* of the text, divided by ``0.07``.  Using the soft-token
        classification head instead (``last_sem_cls_scores``) gives a query
        ranking that is **not** the one used for the prediction, which is why
        the arg-max box of the head distribution is a bad ROI prior (measured:
        IoU 0.00 / 2.9 m centre error on 200 val samples, versus 0.36 / 0.55 m
        with this criterion).  Returns the softmax over the queries with the
        usual auto temperature.

        Args:
            proj_queries: ``(B, Q, 64)`` normalised query projections or None.
            text_feats: ``(B, L, C)`` text memory of the ranking pass.
            text_padding_mask: ``(B, L)`` bool (True = padded).
            positive_map: ``(B, 1, 256)`` phrase-token span or None.

        Returns:
            ``prob`` ``(B, Q)``.
        """
        if proj_queries is None:          # contrastive loss disabled -> uniform
            prob = torch.full(proj_queries.shape[:2], 1.0 / proj_queries.size(1),
                              device=text_feats.device)
            return prob

        proj_tokens = F.normalize(
            self.contrastive_align_projection_text(text_feats), p=2, dim=-1)
        sim = torch.einsum('bqc,blc->bql', proj_queries.float(), proj_tokens) / 0.07

        L = sim.size(-1)
        valid = ~text_padding_mask.bool()
        if positive_map is not None:
            pm = torch.as_tensor(np.asarray(positive_map), dtype=torch.float32,
                                 device=sim.device)
            pm = pm.reshape(pm.shape[0], -1)[:, :L]
            if pm.size(1) < L:
                pm = torch.cat([pm, pm.new_zeros(pm.size(0), L - pm.size(1))], 1)
            phrase = pm > 0
            # a sample without a usable span falls back to the whole utterance
            phrase = torch.where(phrase.any(dim=1, keepdim=True), phrase,
                                 torch.ones_like(phrase))
            valid = valid & phrase
        valid = valid.clone()
        valid[~valid.any(dim=1)] = True   # never mask everything

        affinity = sim.masked_fill(~valid.unsqueeze(1), -1e4).max(dim=-1).values
        tau = affinity.detach().std(dim=1, keepdim=True).clamp_min(1e-4)
        return torch.softmax(affinity / tau, dim=1)                     # (B, Q)

    def _sample_roi_region(self, points_xyz, seed_features_cn, cloud_xyz,
                           text_feats, text_padding_mask, positive_map,
                           detected_feats=None, detected_mask=None):
        """ROI -> FPS points, appended to the 1024 seeds.

        Three rankings are supported:

        * ``field`` (default): the box-proposal field ``f_i`` of the model's own
          predicted boxes -> robust single ball;
        * ``entropy_ball``: the entropy-gradient norm ``s_i`` -> the same robust
          single ball;
        * ``entropy``: Top-K of ``s_i`` -> union of K balls of radius
          ``roi_radius`` (legacy geometry, used by the visualisation and as the
          ablation reference).

        The robust single ball is: greedy NMS + mass stop on the ranking, the
        coordinate-wise weighted median as centre and a weighted quantile of
        the seed distances (+ margin) as radius
        (``utils.roi_sampling.select_reliable_roi``).
        """
        from utils import roi_sampling as rs

        # the entropy *gradient* is only needed by the entropy rankings: the
        # 'field' ranking only needs the boxes of the same ranking pass, so the
        # extra backward through the decoder is skipped (identical results).
        need_grad = self.roi_selector != 'field'
        ranking = self._selection_entropy_seed_score(
            points_xyz, seed_features_cn, text_feats, text_padding_mask,
            positive_map, detected_feats=detected_feats, detected_mask=detected_mask,
            need_grad=need_grad)
        score = ranking['score']

        if self.roi_selector == 'entropy':
            # ---- legacy geometry: Top-K seeds -> union of K balls ----------
            topk = int(min(self.roi_topk, points_xyz.size(1)))
            top_idx = torch.topk(score, k=topk, dim=1, largest=True,
                                 sorted=True).indices
            centers = torch.gather(
                points_xyz.detach(), 1, top_idx.unsqueeze(-1).expand(-1, -1, 3))
            roi = sample_roi_points_torch(
                cloud_xyz.detach().float(), centers.float(),
                num_samples=self.roi_num_points, radius=self.roi_radius,
                max_radius=self.roi_max_radius)
            roi = dict(roi)
            roi['top_seed_indices'] = top_idx
            roi['score'] = score
            roi['reliability'] = None
            roi['field_rank'] = None
            roi['centers'] = centers.float()
            roi['radii'] = roi['radius_used'].float()
        else:
            # ---- reliability ranking -> robust single ball ------------------
            if self.roi_selector == 'entropy_ball':
                reliability = rs.rank_normalize(score)
                field_rank = None
            else:
                field_rank = rs.proposal_seed_field(
                    points_xyz.detach(), ranking['pred_center'],
                    ranking['pred_size'], ranking['sel_prob'],
                    top_m=self.roi_topm)
                r_field = rs.rank_normalize(field_rank)
                reliability = r_field
                if self.roi_selector != 'field':
                    # fuse the region prior with the entropy-gradient ranking
                    r_grad = rs.rank_normalize(score)
                    if self.roi_selector == 'fuse_mul':
                        reliability = r_field * r_grad
                    elif self.roi_selector == 'fuse_min':
                        reliability = torch.minimum(r_field, r_grad)
                    elif self.roi_selector == 'fuse_gate':
                        a = float(self.roi_fuse_alpha)
                        reliability = r_field * (1.0 + a * r_grad) / (1.0 + a)
                    else:
                        raise ValueError(
                            f'unknown roi_selector {self.roi_selector!r}')

            geo = rs.select_reliable_roi(
                points_xyz.detach(), reliability,
                k_max=self.roi_topk, k_min=self.roi_k_min,
                nms_radius=self.roi_nms_radius, mass_frac=self.roi_mass_frac,
                score_ratio=self.roi_score_ratio, radius_q=self.roi_radius_q,
                radius_margin=self.roi_radius_margin,
                radius_min=self.roi_radius_min, radius_max=self.roi_radius_max)
            roi = sample_roi_points_torch(
                cloud_xyz.detach().float(), geo['centers'],
                num_samples=self.roi_num_points, radius=geo['radii'],
                max_radius=self.roi_max_radius)
            roi = dict(roi)
            roi['top_seed_indices'] = geo['sel_idx']
            roi['score'] = score
            roi['reliability'] = reliability
            roi['field_rank'] = field_rank
            roi['centers'] = geo['centers'].float()
            roi['radii'] = geo['radii'].float()

        if roi['points'].shape[1] != self.roi_num_points:
            raise RuntimeError(
                f'ROI sampling produced {roi["points"].shape[1]} points, '
                f'expected {self.roi_num_points}')
        return roi

    # BRIEF forward.
    def forward(self, inputs):
        """
        Forward pass.
        Args:
            inputs: dict
                {point_clouds, text}
                point_clouds (tensor): (B, Npoint, 3 + input_channels)
                text (list): ['text0', 'text1', ...], len(text) = B

                more keys if butd is enabled:
                    det_bbox_label_mask
                    det_boxes
                    det_class_ids
                optional: positive_map (B, num_obj, 256) for the ROI ranking
        Returns:
            end_points: dict
        """
        # STEP 1. vision and text encoding
        end_points, voxel_batch = self._run_backbones(inputs)
        points_xyz = end_points['seed_xyz'].contiguous()        # (B, 1024, 3)
        points_features = end_points['seed_features']           # (B, 1024, 288)
        text_feats = end_points['text_feats']                   # (B, L, 288)
        text_padding_mask = end_points['text_attention_mask']
        end_points['coords'] = inputs['points'][..., 1:4].view(inputs['batch_size'], -1, 3).contiguous()

        # STEP 2. Box encoding
        if self.butd:
            # attend on those features
            detected_mask = ~inputs['det_bbox_label_mask']

            # step box position.    det_boxes ([B, 132, 6]) -->  ([B, 128, 132])
            box_embeddings = self.box_embeddings(inputs['det_boxes'])
            # step box class        det_class_ids ([B, 132])  -->  ([B, 132, 160])
            class_embeddings = self.class_embeddings(self.butd_class_embeddings(inputs['det_class_ids']))
            # step box feature     ([B, 132, 288])
            detected_feats = torch.cat([box_embeddings, class_embeddings.transpose(1, 2)]
                                        , 1).transpose(1, 2).contiguous()
        else:
            detected_mask = None
            detected_feats = None

        # STEP 3. Cross-modality encoding of the 1024 FPS seeds (unchanged)
        text_feats_raw = text_feats
        seed_feats_raw = points_features

        seed_features, text_feats = self._encode_vision(
            points_xyz, points_features, text_feats, text_padding_mask,
            end_points, detected_feats, detected_mask)
        

        # STEP 3.5. Entropy-gradient guided ROI: 512 extra points, whose voxel
        # features are gathered with the *same* VSA layers and encoded with the
        # *same* cross-modal encoder, then concatenated along the point axis.
        if self.roi_enabled:
            positive_map = inputs.get('positive_map', None)
            if positive_map is not None:
                positive_map = positive_map[:, 0].detach().float().cpu().numpy()[:, None, :]
            cloud_xyz = end_points['coords']                    # (B, 50000, 3)
            roi = self._sample_roi_region(
                points_xyz, seed_features, cloud_xyz, text_feats,
                text_padding_mask, positive_map,
                detected_feats=detected_feats, detected_mask=detected_mask)
            roi_xyz = roi['points'].to(points_xyz.dtype)        # (B, 512, 3)

            with _fp32_ctx():   # custom CUDA ops are float32 only
                roi_feats = self.backbone_net.vsa.extract_point_features(voxel_batch, roi_xyz)
            if roi_feats.size(-1) != seed_features.size(1):
                raise RuntimeError(
                    f'ROI/seed feature dim mismatch: {roi_feats.size(-1)} vs '
                    f'{seed_features.size(1)}')
            roi_feats = roi_feats.detach() + self.roi_type_embedding   # (B, 512, 288)

            points_xyz = torch.cat([points_xyz, roi_xyz], dim=1).contiguous()
            seed_features = torch.cat([seed_feats_raw, roi_feats], dim=1)  # (B, 1024 + S, 288)

            seed_features, text_feats = self._encode_vision(
                points_xyz, seed_features, text_feats_raw, text_padding_mask,
                end_points, detected_feats, detected_mask)


            if seed_features.size(-1) != points_xyz.size(1):
                raise RuntimeError(
                    f'fusion length mismatch: {seed_features.size(-1)} vs {points_xyz.size(1)}')

            end_points['roi_points_xyz'] = roi_xyz
            end_points['roi_point_indices'] = roi['indices']
            end_points['roi_seed_indices'] = roi['top_seed_indices']
            end_points['roi_num_candidates'] = roi['num_candidates'].detach().float()
            end_points['roi_radius_used'] = roi['radius_used'].detach().float()
            # NOTE debug-only outputs (no effect on the computation / predictions):
            # ``roi['score']`` is exactly the entropy-gradient score
            # ``s_i = ||dH_sel/dF_i||_2`` produced by
            # ``_selection_entropy_seed_score`` and used by ``torch.topk``.
            if roi['score'] is not None:
                end_points['roi_entropy_scores'] = roi['score'].detach().float()
            end_points['roi_padded'] = roi['padded'].detach()
            # debug-only: the ROI geometry (ball centres / radii) actually used
            end_points['roi_centers'] = roi['centers'].detach().float()
            end_points['roi_radii'] = roi['radii'].detach().float()
            if roi['reliability'] is not None:
                end_points['roi_reliability'] = roi['reliability'].detach().float()
            if roi['field_rank'] is not None:
                end_points['roi_field_rank'] = roi['field_rank'].detach().float()
            end_points['num_vis_tokens'] = int(points_xyz.size(1))
        else:
            end_points['num_vis_tokens'] = int(points_xyz.size(1))

        end_points["text_memory"] = text_feats
        end_points['seed_features'] = seed_features          # (B, 288, 1024) or (B, 288, 1536)

        # STEP 4. text projection --> 64
        if self.contrastive_align_loss:
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(text_feats), p=2, dim=-1
            )
            end_points['proj_tokens'] = proj_tokens     # ([B, L, 64])

        # STEP 4.1 Mask Feats Generation (superpoint branch)
        mask_feats = self.x_mask(seed_features)  # [B, 288, 1536]
        superpoint = inputs['superpoint']  # [B, 50000]
        end_points['superpoints'] = superpoint
        source_xzy = inputs['points'][..., 1:4].view(inputs['batch_size'], -1, 3).contiguous()  # [B, 50000, 3]
        super_features = []
        super_xyz_list = []
        with _fp32_ctx():   # self.super_grouper uses the float32-only CUDA ops
            for bs in range(source_xzy.shape[0]):
                super_xyz = scatter_mean(source_xzy[bs], superpoint[bs], dim=0).unsqueeze(0)
                super_xyz_list.append(super_xyz)
                grouped_feature,ball_idx = self.super_grouper(points_xyz[bs].unsqueeze(0), super_xyz, mask_feats[bs].float().unsqueeze(0))  # [1, 288, super_num, nsample]
                grouped_xyz =(points_xyz[bs])[ball_idx.long().squeeze(0)].unsqueeze(0)
                super_xyz_expand=super_xyz.unsqueeze(2)
                rel_coord=grouped_xyz-super_xyz_expand
                rel_feat=(self.rel_encoder(rel_coord)).permute(0,3,1,2)
                grouped_feature=grouped_feature+rel_feat
                super_feature = F.max_pool2d(grouped_feature, kernel_size=[1, grouped_feature.size(3)]).squeeze(-1).squeeze(0)  # [288, super_num]
                super_features.append(super_feature)

        # STEP 5-7. Query generation + DETR decoder on the fused sequence
        cluster_xyz, query_last, prefix = self._queries_and_boxes(
            points_xyz, seed_features, text_feats, text_padding_mask, end_points,
            detected_feats, detected_mask)

        # step Seg Prediction head
        query_last = self.x_query(query_last.transpose(1, 2)).transpose(1, 2)
#---------------------------text decoder-----------------------------------

        text_query=text_feats
        prediction_masks = []
        sp_pred_masks = []
        adaptive_weight_lists=[]

        for bs in range(points_xyz.shape[0]):
            bs_text_query=text_query[bs].unsqueeze(0)
            _, _, attn_masks = self.prediction_head(bs_text_query, super_features[bs].unsqueeze(0).transpose(1,2))
            for i in range(3): 
                bs_text_query, _,src_weight = self.swa_layers[i]( super_features[bs].unsqueeze(0).transpose(1,2).transpose(0,1),bs_text_query.transpose(0,1), attn_mask=attn_masks)#SWA模块
                bs_text_query = self.swa_ffn_layers[i](bs_text_query)
                _, _, attn_masks = self.prediction_head(bs_text_query, super_features[bs].unsqueeze(0).transpose(1,2))
            src_weight = src_weight.softmax(1)
            src_weight = torch.where(torch.isnan(src_weight), torch.zeros_like(src_weight), src_weight)
            q_score = (src_weight*~text_padding_mask[bs].unsqueeze(-1)).sum(-1) # [B, N_q]
            q_idx = q_score[0].argmax(dim=-1, keepdim=True) 
            pred_scores, pred_masks, _ = self.prediction_head(bs_text_query[:,q_idx,:], super_features[bs].unsqueeze(0).transpose(1,2))                
            prediction_masks.append(pred_masks.expand(1,256,pred_masks.shape[-1]))
            adaptive_weight=torch.sigmoid(pred_scores.squeeze())
            adaptive_weight_lists.append(adaptive_weight)
            sp_pred_mask = self._seg_seeds_prediction(
                query_last[bs].unsqueeze(0),                                  # ([1, F=256, V=288])
                super_features[bs].unsqueeze(0),                             # ([1, F=288, V=super_num])
                end_points=end_points,  # 
                prefix=prefix
            ).squeeze(0)  
            sp_pred_masks.append(sp_pred_mask)
        end_points['sp_last_pred_masks'] = sp_pred_masks  # list  BS* [256, super_num]
        end_points['last_pred_masks'] = prediction_masks  # bs*[ 256, super_num]
        end_points['adaptive_weights']=adaptive_weight_lists
        end_points['super_xyz_list'] = super_xyz_list 

        return end_points

    def init_bn_momentum(self):
        """Initialize batch-norm momentum."""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = 0.1
