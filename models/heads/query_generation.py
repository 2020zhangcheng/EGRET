"""Query point generation by gumbel-softmax sampling over seed features."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelSampling(nn.Module):
    def __init__(self, in_features, sampling_n, dropout=0.1, activation_fn=None):
        super(GumbelSampling, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=in_features, 
            num_heads=8, 
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_features)
        self.sampler = nn.Linear(in_features, sampling_n)
        
    def forward(self, x, tau=1., hard=False, mask=None):
        """
        x.shape = [bz, point_num, feature_len]
        """

        x_attn = self.attn(x, x, x)[0]
        x = x + self.dropout(x_attn)
        x = self.norm(x)

        sample_weights = self.sampler(x) # [bz, N, sampling_n]

        batch_size, num_points, sampling_n = sample_weights.shape
        attn = F.gumbel_softmax(sample_weights.transpose(1, 2).contiguous().reshape(-1, num_points), tau=tau, hard=hard)
        attn = attn.reshape(batch_size, sampling_n, num_points)
        return attn  # [batch_size, sampling_n, num_points]
