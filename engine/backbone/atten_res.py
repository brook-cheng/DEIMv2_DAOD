import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from ..deim.deim_utils import RMSNorm


class BlockAttnRes(nn.Module):
    """
    reference: Attention Residuals,https://arxiv.org/abs/2603.15031
    Block-level Attention Residuals module.

    Replaces the standard residual connection's fixed weight-1 accumulation
    with a softmax attention mechanism over block representations,
    allowing each sublayer to selectively aggregate information from
    preceding blocks with learned, input-dependent weights.

    Args:
        hidden_dim (int): The hidden dimension D of the model.
        norm_eps (float): Epsilon for RMSNorm. Default: 1e-6.
    """

    def __init__(self, hidden_dim: int, norm_eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_dim
        self.proj = nn.Linear(hidden_dim, 1, bias=False)
        self.norm = RMSNorm(hidden_dim, eps=norm_eps)

    def forward(self, blocks: List[torch.Tensor]) -> torch.Tensor:
        """
        Inter-block attention: attend over block representations sum.

        Args:
            blocks: A list of N tensors, each of shape [blocks_num,batch_nums, hiden_dem,H,W],
                    representing completed block representations from previous blocks.

        Returns:
            h: A tensor of shape [B, T, D], the attention-weighted aggregation.
        """
        B, D, H, W = blocks[0].shape
        # V: [N, B, D, H, W]->[N, B, D, H*W]->[N, B, H*W, D]
        V = torch.stack(list(blocks), dim=0)
        V = V.view(len(blocks), B, D, -1).permute(0, 1, 3, 2)

        # [num_candidates, B, T, D]
        K = self.norm(V)
        # proj.weight shape: [1, D] → squeeze to [D]
        query = self.proj.weight.squeeze(dim=0)  # [D]
        # This computes w^T · K[n,b,t,:] for each (n, b, t)
        # [N, B, T]
        logits = torch.einsum("d, n b t d -> n b t", query, K)

        # [N, B, T]
        attn_weights = F.softmax(logits, dim=0)

        # einsum: n b t, n b t d -> b t d
        # h: [B, T, D]
        h = torch.einsum("n b t, n b t d -> b t d", attn_weights, V)

        h = h.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]

        return h
