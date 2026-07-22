import torch
import torch.nn.functional as F


def pairwise_cosine_matrix(tokens: torch.Tensor) -> torch.Tensor:
    tokens = F.normalize(tokens, dim=-1)
    return tokens @ tokens.transpose(-1, -2)

