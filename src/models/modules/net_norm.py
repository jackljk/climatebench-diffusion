import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn, norm=LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn, norm=LayerNorm):
        super().__init__()
        self.fn = fn
        self.norm = norm(dim)

    def forward(self, x, **kwargs):
        x = self.fn(x, **kwargs)
        return self.norm(x)
