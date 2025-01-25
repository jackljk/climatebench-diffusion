# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, trunc_normal_

from src.models.stormer.pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed
from src.utilities.utils import get_logger


log = get_logger(__name__)


class ClimaXEmbedding(nn.Module):
    def __init__(
        self,
        default_vars,
        img_size=[32, 64],
        patch_size=2,
        embed_dim=1024,
        num_heads=16,
        use_pos_emb=True,
    ):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.default_vars = tuple(default_vars)
        self.out_channels = embed_dim

        # variable tokenization: separate embedding layer for each input variable
        self.token_embeds = nn.ModuleList(
            [PatchEmbed(None, patch_size, 1, embed_dim) for i in range(len(default_vars))]
        )
        self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)

        # variable embedding to denote which variable each token belongs to
        # helps in aggregating variables
        self.channel_embed, self.channel_map = self.create_var_embedding(embed_dim)
        self._var_ids = None  # for caching

        # variable aggregation: a learnable query and a single-layer cross attention
        self.channel_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.channel_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # positional embedding
        if use_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=True)
        else:
            self.pos_embed = None
        self.initialize_weights()

    def initialize_weights(self):
        # initialize pos_emb and var_emb
        if self.pos_embed is not None:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                int(self.img_size[0] / self.patch_size),
                int(self.img_size[1] / self.patch_size),
                cls_token=False,
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        channel_embed = get_1d_sincos_pos_embed_from_grid(
            self.channel_embed.shape[-1], np.arange(len(self.default_vars))
        )
        self.channel_embed.data.copy_(torch.from_numpy(channel_embed).float().unsqueeze(0))

        # token embedding layer
        for i in range(len(self.token_embeds)):
            w = self.token_embeds[i].proj.weight.data
            trunc_normal_(w.view([w.shape[0], -1]), std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def create_var_embedding(self, dim):
        var_embed = nn.Parameter(torch.zeros(1, len(self.default_vars), dim), requires_grad=True)
        # TODO: create a mapping from var --> idx
        var_map = {}
        idx = 0
        for var in self.default_vars:
            var_map[var] = idx
            idx += 1
        return var_embed, var_map

    # @lru_cache(maxsize=None)   # Need to use torch.no_grad() to cache (not inference
    def get_var_ids(self, variables, device, recache=False):
        if not recache and self._var_ids is not None:
            # Need to return clone since otherwise receive error:
            # RuntimeError: Inference tensors cannot be saved for backward. To work around you can make a clone (...)
            return self._var_ids.clone()
        ids = np.array([self.channel_map[var] for var in variables])
        var_ids = torch.from_numpy(ids).to(device)
        self._var_ids = var_ids
        return var_ids

    # @torch.compiler.disable(recursive=True)
    def get_var_emb(self, var_emb, variables, var_ids=None):
        var_ids = self.get_var_ids(variables, var_emb.device) if var_ids is None else var_ids
        # by default, var_ids == : == range(len(variables)) == range(len(default_vars))??
        return var_emb[:, var_ids, :]

    @torch.compiler.disable(recursive=True)
    def get_var_token_emb(self, var_id, var_in_data):
        return self.token_embeds[var_id](var_in_data)

    @torch.compiler.disable(recursive=True)
    def aggregate_variables(self, x: torch.Tensor):
        """
        x: B, V, L, D
        """
        b, _, seq_len, _ = x.shape
        x = torch.einsum("bvld->blvd", x)
        x = x.flatten(0, 1)  # BxL, V, D

        var_query = self.channel_query.repeat_interleave(x.shape[0], dim=0)
        x, _ = self.channel_agg(var_query, x, x)  # BxL, D
        x = x.squeeze()

        x = x.unflatten(dim=0, sizes=(b, seq_len))  # B, L, D
        return x

    def forward(self, x: torch.Tensor, variables=None, region_info=None):
        if isinstance(variables, list):
            variables = tuple(variables)
        if variables is None:
            recache_var_ids = False
            variables = self.default_vars
        else:
            recache_var_ids = True

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device, recache=recache_var_ids)
        if x.shape[1] != len(var_ids):
            log.warning(
                f"Number of variables in data ({x.shape}[1]) does not match the number of variables ({len(var_ids)}). type(x): {type(x)}"
            )
        for i in range(len(var_ids)):
            embed_ = self.get_var_token_emb(var_id=var_ids[i], var_in_data=x[:, i : i + 1])  # B, L, D
            # Set TORCH_LOGS="+dynamo" and TORCHDYNAMO_VERBOSE=1 for more information
            if region_info is not None:
                # get the patch ids corresponding to the region
                embed_ = embed_[:, region_info["x_patch_ids"], :] if region_info["x_patch_ids"] is not None else embed_
            embeds.append(embed_)  # V * (B, L, D)
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.channel_embed, variables, var_ids=var_ids)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D
        # print(f"{x.shape=}, {self.pos_embed.shape=}, {var_embed.shape=}")
        # x.shape=([4, 75, 1920, 16]), self.pos_embed.shape=([1, 1920, 16]), var_embed.shape=([1, 75, 16])
        if self.pos_embed is None:
            pass
        elif region_info is not None:
            x = x + self.pos_embed[:, region_info["pos_emb_patch_ids"], :].unsqueeze(1)
        else:
            x = x + self.pos_embed.unsqueeze(1)

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D
        # print(f"{x.shape=}")  # x.shape=torch.Size([4, 1920, 16])

        return x


class ClimaXEmbeddingForConvNet(ClimaXEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embed_norm = torch.nn.LayerNorm(self.out_channels)

    def forward(self, x: torch.Tensor, **kwargs):
        b, c, h, w = x.shape
        x = super().forward(x, **kwargs)
        x = self.embed_norm(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x
