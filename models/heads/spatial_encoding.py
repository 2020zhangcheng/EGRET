"""Spatial encodings: pairwise relative locations and learned position embeddings."""

import einops
import torch
import torch.nn as nn


def calc_pairwise_locs(obj_centers,  eps=1e-10):

    pairwise_locs = einops.repeat(obj_centers, 'b l d -> b l 1 d') \
        - einops.repeat(obj_centers, 'b l d -> b 1 l d')
    pairwise_dists = torch.sqrt(torch.sum(pairwise_locs**2, 3) + eps) # (b, l, l)
    norm_pairwise_dists = pairwise_dists

    pairwise_dists_2d = torch.sqrt(torch.sum(pairwise_locs[..., :2]**2, 3)+eps)

    pairwise_locs = torch.stack(
        [norm_pairwise_dists, pairwise_locs[..., 2]/pairwise_dists, 
        pairwise_dists_2d/pairwise_dists, pairwise_locs[..., 1]/pairwise_dists_2d,
        pairwise_locs[..., 0]/pairwise_dists_2d],
        dim=3
    )
    return pairwise_locs


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        """Forward pass, xyz is (B, N, 3or6), output (B, F, N)."""
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
