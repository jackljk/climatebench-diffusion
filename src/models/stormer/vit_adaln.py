# Third Party
from typing import Union

import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp, trunc_normal_

from src.models._base_model import BaseModel
from src.models.modules.attention import MemEffAttention
from src.models.stormer.climax_embedding import ClimaXEmbedding
from src.utilities.utils import exists


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, t):
        return self.mlp(t.unsqueeze(-1) if self.in_dim == 1 else t)


class Block(nn.Module):
    """
    An transformers block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, time_dim=None, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = MemEffAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 6 * hidden_size, bias=True)) if time_dim is not None else None
        )

    def forward(self, x, c):
        if self.adaLN_modulation is None:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels, time_dim=None):
        super().__init__()
        self.norm_final = nn.Identity()
        # self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_dim, 2 * hidden_size, bias=True)) if time_dim is not None else None
        )

    def forward(self, x, c):
        if self.adaLN_modulation is not None:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            x = modulate(self.norm_final(x), shift, scale)
        else:
            x = self.norm_final(x)
        x = self.linear(x)
        return x


class ViTAdaLN(BaseModel):
    def __init__(
        self,
        list_variables,
        patch_size=2,
        embed_norm=True,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        with_time_emb: Union[bool, str] = True,  # can be True or "linear" or "sinusoidal"
        outer_sample_mode: str = None,  # bilinear or nearest
        upsample_dims: tuple = None,  # (256, 256) or (128, 120) etc.
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.hparams.debug_mode:
            self.hparams.patch_size = patch_size = 4
            self.hparams.hidden_size = hidden_size = 16
            self.hparams.depth = depth = 3
            self.hparams.num_heads = num_heads = 2

        in_img_size = self.spatial_shape_in if upsample_dims is None else upsample_dims
        if in_img_size[0] % patch_size != 0:
            pad_size = patch_size - in_img_size[0] % patch_size
            in_img_size = (in_img_size[0] + pad_size, in_img_size[1])
        self.in_img_size = in_img_size
        self.list_variables = list_variables
        self.patch_size = patch_size
        self.embed_norm = embed_norm
        self.hidden_size = hidden_size

        # Upsample the input image to a size that is divisible by the patch size
        assert (upsample_dims is None and outer_sample_mode is None) or (
            upsample_dims is not None and outer_sample_mode is not None
        ), "upsample_dims and outer_sample_mode must be both None or both not None"
        # To keep spatial dimensions for uneven spatial sizes, we need to use nearest upsampling
        # and then crop the output to the desired size
        if outer_sample_mode is not None:
            # Upsample (45, 90) to be easier to divide by 2 multiple times
            # upsample_dims = (48, 96)
            self.upsampler = torch.nn.Upsample(size=tuple(upsample_dims), mode=outer_sample_mode)
        else:
            self.upsampler = None

        # embedding
        self.embedding = ClimaXEmbedding(
            default_vars=list_variables,
            img_size=in_img_size,
            patch_size=patch_size,
            embed_dim=hidden_size,
            num_heads=num_heads,
        )
        self.embed_norm_layer = nn.LayerNorm(hidden_size) if embed_norm else nn.Identity()

        # interval embedding
        self.time_emb_in_dim = 1 if with_time_emb else None
        self.time_dim = hidden_size if with_time_emb else None
        if with_time_emb in [True, "linear"]:
            self.t_embedder = TimestepEmbedder(self.time_emb_in_dim, self.time_dim)
        elif with_time_emb == "sinusoidal":
            raise NotImplementedError("Sinusoidal time embeddings are not implemented yet.")
        else:
            self.t_embedder = None

        # backbone
        self.blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, mlp_ratio=mlp_ratio, time_dim=self.time_dim) for _ in range(depth)]
        )

        # prediction layer
        self.head = FinalLayer(
            hidden_size, patch_size, self.num_output_channels, time_dim=self.time_dim
        )  # len(list_variables))
        if exists(self.upsampler):
            self.head2 = nn.Conv2d(self.num_output_channels, self.num_output_channels, kernel_size=1)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        if self.hparams.with_time_emb:
            # Initialize timestep embedding MLP:
            trunc_normal_(self.t_embedder.mlp.weight, std=0.02)

            # Zero-out adaLN modulation layers in blocks:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

            nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor, h=None, w=None):
        """
        x: (B, L, V * patch_size**2), e.g. (3, 1920, 1024)
        return imgs: (B, V, H, W)
        """
        p = self.patch_size
        v = self.num_output_channels  # was len(self.list_variables)
        h = self.in_img_size[0] // p if h is None else h // p
        w = self.in_img_size[1] // p if w is None else w // p
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, v))
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(shape=(x.shape[0], v, h * p, w * p))
        return imgs

    def forward(
        self,
        inputs,
        time=None,
        variables=None,
        region_info=None,
        condition=None,
        dynamical_condition=None,
        static_condition=None,
    ):
        x = self.concat_condition_if_needed(inputs, condition, dynamical_condition, static_condition)
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x) if exists(self.upsampler) else x
        x = self.embedding(x, variables, region_info)  # B, L, D
        x = self.embed_norm_layer(x)
        # self.log_text.info(f"Embedding shape: {x.shape}, orig_x_shape: {orig_x_shape}")

        time_interval_emb = self.t_embedder(time) if self.t_embedder is not None else None
        for block in self.blocks:
            x = block(x, time_interval_emb)
            # self.log_text.info(f"Block shape: {x.shape}")   # (B, L, D)

        x = self.head(x, time_interval_emb)  # This used to be before the unpatchify. Should be ok since linear.
        if region_info is not None:
            min_h, max_h = region_info["min_h"], region_info["max_h"]
            min_w, max_w = region_info["min_w"], region_info["max_w"]
            x = self.unpatchify(x, h=max_h - min_h + 1, w=max_w - min_w + 1)
        else:
            x = self.unpatchify(x)

        if exists(self.upsampler):  # align_corners=False, mode=bilinear
            x = nn.functional.interpolate(x, size=orig_x_shape, mode=self.hparams.outer_sample_mode)
            x = self.head2(x)

        return x


# variables = [
#     "2m_temperature",
#     "10m_u_component_of_wind",
#     "10m_v_component_of_wind",
#     "mean_sea_level_pressure",
#     "geopotential_50",
#     "geopotential_100",
#     "geopotential_150",
#     "geopotential_200",
#     "geopotential_250",
#     "geopotential_300",
#     "geopotential_400",
#     "geopotential_500",
#     "geopotential_600",
#     "geopotential_700",
#     "geopotential_850",
#     "geopotential_925",
#     "geopotential_1000",
#     "u_component_of_wind_50",
#     "u_component_of_wind_100",
#     "u_component_of_wind_150",
#     "u_component_of_wind_200",
#     "u_component_of_wind_250",
#     "u_component_of_wind_300",
#     "u_component_of_wind_400",
#     "u_component_of_wind_500",
#     "u_component_of_wind_600",
#     "u_component_of_wind_700",
#     "u_component_of_wind_850",
#     "u_component_of_wind_925",
#     "u_component_of_wind_1000",
#     "v_component_of_wind_50",
#     "v_component_of_wind_100",
#     "v_component_of_wind_150",
#     "v_component_of_wind_200",
#     "v_component_of_wind_250",
#     "v_component_of_wind_300",
#     "v_component_of_wind_400",
#     "v_component_of_wind_500",
#     "v_component_of_wind_600",
#     "v_component_of_wind_700",
#     "v_component_of_wind_850",
#     "v_component_of_wind_925",
#     "v_component_of_wind_1000",
#     "temperature_50",
#     "temperature_100",
#     "temperature_150",
#     "temperature_200",
#     "temperature_250",
#     "temperature_300",
#     "temperature_400",
#     "temperature_500",
#     "temperature_600",
#     "temperature_700",
#     "temperature_850",
#     "temperature_925",
#     "temperature_1000",
#     "specific_humidity_50",
#     "specific_humidity_100",
#     "specific_humidity_150",
#     "specific_humidity_200",
#     "specific_humidity_250",
#     "specific_humidity_300",
#     "specific_humidity_400",
#     "specific_humidity_500",
#     "specific_humidity_600",
#     "specific_humidity_700",
#     "specific_humidity_850",
#     "specific_humidity_925",
#     "specific_humidity_1000",
# ]
# from torch.utils.flop_counter import FlopCounterMode
# import numpy as np
# from stormer.utils.metrics import lat_weighted_mse
# from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# import torch.distributed as dist
# import os

# # # Mock environment variables for single-node distributed training
# # os.environ['RANK'] = '0'
# # os.environ['WORLD_SIZE'] = '1'
# # os.environ['MASTER_ADDR'] = 'localhost'
# # os.environ['MASTER_PORT'] = '12355'

# # dist.init_process_group(backend='nccl')

# device = 'cuda'
# patch_size = 8

# model = ViTAdaLN(
#     in_img_size=(721,1440),
#     list_variables=variables,
#     patch_size=patch_size,
#     embed_norm=True,
#     hidden_size=1024,
#     depth=24,
#     num_heads=16,
#     mlp_ratio=4.0,
# ).to(device).half()
# # model = FSDP(
# #     model,
# #     sharding_strategy="SHARD_GRAD_OP",
# #     # activation_checkpointing_policy={Block, ClimaXEmbedding},
# #     # auto_wrap_policy=Block
# # )

# x = torch.randn((1, 69, 721, 1440)).to(device, dtype=torch.half)
# y = torch.rand_like(x)
# pad_size = patch_size - 721 % patch_size
# padded_x = torch.nn.functional.pad(x, (0, 0, pad_size, 0), 'constant', 0)
# lat = np.random.randn(721)
# time_interval = torch.tensor([6]).to(dtype=x.dtype).to(device)

# flop_counter = FlopCounterMode(model, depth=2)
# with flop_counter:
#     # lat_weighted_mse(model(padded_x, variables, time_interval)[:, :, pad_size:], y, variables, lat)["w_mse_aggregate"].backward()
#     model(padded_x, variables, time_interval)
