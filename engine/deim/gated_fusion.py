import torch
import torch.nn as nn


class GatedSoftmaxFusion(nn.Module):
    def __init__(self, d_model: int, n_sources: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.n_sources = n_sources
        self.weight_net = nn.Sequential(
            nn.Linear((n_sources + 1) * d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_sources),
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, srcs: list[torch.Tensor], query: torch.Tensor) -> torch.Tensor:
        assert len(srcs) == self.n_sources
        num_queries = query.shape[1]
        for i, src in enumerate(srcs):
            assert src.shape[1] == num_queries, \
                f"Source {i} has {src.shape[1]} tokens, expected {num_queries}"
        cat = torch.cat([query] + srcs, dim=-1)
        weights = torch.softmax(self.weight_net(cat), dim=-1)
        fused = torch.zeros_like(srcs[0])
        for i, src in enumerate(srcs):
            fused = fused + weights[..., i:i + 1] * src
        return self.output_proj(fused)
