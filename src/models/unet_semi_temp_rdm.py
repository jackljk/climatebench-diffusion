from functools import partial
from typing import Optional

import torch
from einops import rearrange
from torch import nn

from src.models._base_model import BaseModel
from src.models.modules.attention import Attention, LinearAttention, SpatialTransformer
from src.models.modules.misc import Residual, get_time_embedder
from src.models.modules.net_norm import PreNorm
from src.models.modules.rotary_embedding import RotaryEmbedding
from src.models.unet import Downsample, LayerNorm, ResnetBlock, Upsample
from src.models.unet3d import Attention as TimeAttention
from src.models.unet3d import RelativePositionBias
from src.utilities.utils import default, exists, raise_error_if_invalid_value


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn, **extra_kwargs):
        super().__init__()
        self.from_einops = from_einops
        # Get dims from from_einops that can be inferred from the input tensor, x
        # E.g. (b t) c h w -> None, c, h, w
        #      b c t h w -> b, c, t, h, w
        #      (b t) c (h w) -> None, c, None
        from_einops_cleaned = from_einops
        for k in extra_kwargs.keys():
            from_einops_cleaned = from_einops_cleaned.replace(f" {k}", "").replace(f"{k} ", "")
        self.from_einops_dims = []
        for dim in from_einops_cleaned.split(" "):
            if "(" not in dim and ")" not in dim:
                self.from_einops_dims.append(dim)
            else:
                self.from_einops_dims.append(None)

        self.to_einops = to_einops
        self.fn = fn
        self.extra_kwargs = extra_kwargs if len(extra_kwargs) > 0 else None

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitute_kwargs = {
            **self.extra_kwargs,
            **{dim: shape_i for dim, shape_i in zip(self.from_einops_dims, shape) if dim is not None},
        }
        # reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        try:
            x = rearrange(x, f"{self.from_einops} -> {self.to_einops}", **self.extra_kwargs)
        except ZeroDivisionError as e:
            raise ZeroDivisionError(
                f"shape: {shape}, from_einops: {self.from_einops}, to_einops: {self.to_einops}, extra_kwargs: {self.extra_kwargs}"
            ) from e

        x = self.fn(x, **kwargs)
        x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        # x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **self.extra_kwargs)
        return x


class SimpleTemporalConvBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8, dropout: float = 0.0, kernel_size=3, stride=1):
        super().__init__()
        # Kernel=3=>(3,3,3) would convolve over time, height and width (padding=(1, 1, 1))
        # Kernel=(1,3,3) would convolve only over height and width (padding=(0, 1, 1))
        # Kernel=(3,1,1) would convolve only over time (padding=(1, 0, 0))
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        padding = tuple((k // 2) for k in kernel_size)  # same padding, e.g. (1, 1, 1) for kernel_size=3

        # self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3))  # This would convolve only ove
        self.proj = nn.Conv3d(dim, dim_out, kernel_size, padding=padding, stride=stride)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class TemporalConvBlock(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, double_conv_layer: bool = True, **kwargs):
        super().__init__()
        self.double_conv_layer = double_conv_layer
        self.block1 = SimpleTemporalConvBlock(dim_in, dim_out, **kwargs)
        self.block2 = SimpleTemporalConvBlock(dim_out, dim_out, **kwargs) if double_conv_layer else nn.Identity()
        self.res_conv = nn.Conv3d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


# model
class Unet(BaseModel):
    def __init__(
        self,
        dim,
        init_dim=None,
        dim_mults=(1, 2, 4, 8),
        resnet_block_groups=8,
        with_time_emb: bool = False,
        time_dim_mult: int = 2,
        temporal_op: str = "conv_3d",  # conv_1d, conv_3d, or attn
        temporal_conv_kernel_size: int = 3,
        use_init_temporal: bool = False,
        use_rotary_emb: bool = False,
        attn_heads=8,
        attn_dim_head=32,
        temporal_dropout: float = 0.0,
        block_dropout: float = 0.0,  # for second block in resnet block
        block_dropout1: float = 0.0,  # for first block in resnet block
        attn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        double_conv_layer: bool = True,
        double_conv_temporal_layer: bool = True,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        learned_sinusoidal_dim=16,
        outer_sample_mode: str = None,  # bilinear or nearest
        upsample_dims: tuple = None,  # (256, 256) or (128, 128)
        keep_spatial_dims: bool = False,
        init_kernel_size: int = 7,
        init_padding: int = 3,
        init_stride: int = 1,
        num_conditions: int = 0,
        spatial_conditioning_mode: str = "concat",
        non_spatial_conditioning_mode: str = None,
        dim_head: int = 32,
        non_spatial_cond_hdim: int = None,
        cross_attn_dim: int = None,  # deprecated
        transformer_depth: int = 1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # determine dimensions
        assert self.num_input_channels is not None, "Please specify ``num_input_channels`` in the model config."
        assert self.num_output_channels is not None, "Please specify ``num_output_channels`` in the model config."
        assert (
            self.num_conditional_channels is not None
        ), "Please specify ``num_conditional_channels`` in the model config."
        # raise_error_if_invalid_value(conditioning_mode, ["concat", "cross_attn"], "conditioning_mode")
        input_channels = self.num_input_channels + self.num_conditional_channels
        self._input_channels = input_channels
        output_channels = self.num_output_channels or input_channels
        self.save_hyperparameters()

        if num_conditions >= 1:
            assert (
                self.num_conditional_channels > 0
            ), f"num_conditions is {num_conditions} but num_conditional_channels is {self.num_conditional_channels}"
        assert spatial_conditioning_mode == "concat", f"spatial_conditioning_mode is {spatial_conditioning_mode}"
        valid_cond_modes = [None]
        raise_error_if_invalid_value(non_spatial_conditioning_mode, valid_cond_modes, "non_spatial_conditioning_mode")
        self.non_spatial_conditioning_mode = None

        if hasattr(self, "spatial_shape") and self.spatial_shape is not None:
            b, s1, s2 = 1, *self.spatial_shape
            self.example_input_array = [
                torch.rand(b, self.num_input_channels, self.num_temporal_channels, s1, s2),
                torch.rand(b, self.num_temporal_channels) if with_time_emb else None,
                (
                    torch.rand(b, self.num_conditional_channels, self.num_temporal_channels, s1, s2)
                    if self.num_conditional_channels > 0
                    else None
                ),
            ]

        init_dim = default(init_dim, dim)

        self.init_conv = nn.Conv2d(
            input_channels,
            init_dim,
            init_kernel_size,
            padding=init_padding,
            stride=init_stride,
        )
        self.dropout_input = nn.Dropout(input_dropout)

        assert self.num_temporal_channels > 0, f"num_temporal_channels: {self.num_temporal_channels}"
        if "conv" in temporal_op:
            # temporal convolutions
            if temporal_op == "conv_1d":
                kernel_size = (temporal_conv_kernel_size, 1, 1)
            elif temporal_op == "conv_3d":
                kernel_size = (temporal_conv_kernel_size, 3, 3)
            else:
                raise ValueError(f"temporal_op: {temporal_op} is not valid")

            def temporal_conv(dim):
                return EinopsToAndFrom(
                    "(b t) c h w",
                    "b c t h w",
                    TemporalConvBlock(
                        dim,
                        dim,
                        kernel_size=kernel_size,
                        double_conv_layer=double_conv_temporal_layer,
                        groups=resnet_block_groups,
                        dropout=temporal_dropout,
                    ),
                    t=self.num_temporal_channels,
                )

            temporal_op = temporal_conv

        elif temporal_op == "attn":
            # temporal attention and its relative positional encoding
            rotary_emb = RotaryEmbedding(min(32, attn_dim_head)) if use_rotary_emb else None
            self.time_rel_pos_bias = RelativePositionBias(
                heads=attn_heads, max_distance=32
            )  # realistically will not be able to generate that many frames of video... yet

            def temporal_attn(dim):
                #         x = rearrange(x, "b c t h w -> (b t) c h w")
                return Residual(
                    PreNorm(
                        dim,
                        EinopsToAndFrom(
                            "(b t) c h w",  # "b c t h w",
                            "b (h w) t c",
                            TimeAttention(
                                dim,
                                heads=attn_heads,
                                dim_head=attn_dim_head,
                                rotary_emb=rotary_emb,
                                dropout=temporal_dropout,
                            ),
                            t=self.num_temporal_channels,
                        ),
                    )
                )

            temporal_op = temporal_attn
        else:
            raise ValueError(f"temporal_op: {temporal_op} is not valid")

        self.init_temporal_op = temporal_op(init_dim) if use_init_temporal else None

        if with_time_emb:
            pos_emb_dim = dim
            sinusoidal_embedding = "learned" if learned_sinusoidal_cond else "true"

            self.time_dim = dim * time_dim_mult
            self.time_emb_mlp = get_time_embedder(
                self.time_dim, pos_emb_dim, sinusoidal_embedding, learned_sinusoidal_dim
            )
        else:
            self.time_dim = None
            self.time_emb_mlp = None

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(
            ResnetBlock,
            groups=resnet_block_groups,
            dropout2=block_dropout,
            dropout1=block_dropout1,
            double_conv_layer=double_conv_layer,
            time_emb_dim=self.time_dim,
        )
        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)
        linear_attn_kwargs = dict(rescale="qkv", dropout=attn_dropout)
        use_spatial_transformer = self.non_spatial_conditioning_mode == "cross_attn"
        self.non_spatial_cond_hdim = None
        spatial_transformer_kwargs = dict(depth=transformer_depth, context_dim=self.non_spatial_cond_hdim)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            do_downsample = not is_last and not keep_spatial_dims
            # num_heads = dim // dim_head
            num_heads, dim_head = 4, 32

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in),
                        block_klass(dim_in, dim_in),
                        (
                            Residual(
                                PreNorm(
                                    dim_in,
                                    fn=LinearAttention(
                                        dim_in, **linear_attn_kwargs, heads=num_heads, dim_head=dim_head
                                    ),
                                    norm=LayerNorm,
                                )
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(dim_in, num_heads, dim_head, **spatial_transformer_kwargs)
                        ),
                        temporal_op(dim_in),
                        Downsample(dim_in, dim_out) if do_downsample else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        # num_heads = mid_dim // dim_head
        num_heads, dim_head = 4, 32
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        if use_spatial_transformer:
            self.mid_attn = SpatialTransformer(mid_dim, num_heads, dim_head, **spatial_transformer_kwargs)
        else:
            self.mid_attn = Residual(
                PreNorm(
                    mid_dim,
                    fn=Attention(mid_dim, dropout=attn_dropout, heads=num_heads, dim_head=dim_head),
                    norm=LayerNorm,
                )
            )
        self.mid_temporal_op = temporal_op(mid_dim)
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            do_upsample = not is_last and not keep_spatial_dims
            # num_heads = dim_out // dim_head
            num_heads, dim_head = 4, 32

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out),
                        block_klass(dim_out + dim_in, dim_out),
                        (
                            Residual(
                                PreNorm(
                                    dim_out,
                                    fn=LinearAttention(
                                        dim_out, heads=num_heads, dim_head=dim_head, **linear_attn_kwargs
                                    ),
                                    norm=LayerNorm,
                                )
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(dim_out, num_heads, dim_head, **spatial_transformer_kwargs)
                        ),
                        temporal_op(dim_out),
                        Upsample(dim_out, dim_in) if do_upsample else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = input_channels * (1 if not learned_variance else 2)
        self.out_dim = default(output_channels, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = self.get_head()

    def get_head(self):
        return nn.Conv2d(self.hparams.dim, self.out_dim, 1)

    def set_head_to_identity(self):
        self.final_conv = nn.Identity()

    def get_block(self, dim_in, dim_out, dropout: Optional[float] = None):
        return ResnetBlock(
            dim_in,
            dim_out,
            groups=self.hparams.resnet_block_groups,
            dropout1=dropout or self.hparams.block_dropout1,
            dropout2=dropout or self.hparams.block_dropout,
            time_emb_dim=self.time_dim,
        )

    def get_extra_last_block(self, dropout: Optional[float] = None):
        return self.get_block(self.hparams.dim, self.hparams.dim, dropout=dropout)

    def forward(
        self,
        inputs,
        time=None,
        condition=None,
        static_condition=None,
        return_time_emb: bool = False,
        get_intermediate_shapes: bool = False,
        **kwargs,
    ):
        x = inputs
        assert len(time.shape) == 2, f"time.shape: {time.shape}"
        b, t_dim = time.shape
        assert x.shape[0] == b, f"x.shape[0]: {x.shape[0]}, b: {b}"
        assert x.shape[2] == t_dim, f"x.shape[2]: {x.shape[2]}, t_dim: {t_dim}"
        kwargs_temp_op = dict()
        if self.hparams.temporal_op == "attn":
            time_rel_pos_bias = self.time_rel_pos_bias(t_dim, device=x.device)
            kwargs_temp_op = dict(pos_bias=time_rel_pos_bias)

        time = rearrange(time, "b t -> (b t)")
        x = rearrange(x, "b c t h w -> (b t) c h w")  # concat with cond: (b t) c h w -> (b t) (c+cond) h w
        # For time operations:        "(b t) c h w"   --->   "b (h w) t c"
        # with condition:   "(b t) (c+cond) h w"   --->   "b (h w) t (c+cond)"
        ada_ln_input = time

        # x_in = (B, C, T, H, W), t = (B, T) -> x_in = (B*T, C, H, W), t = (B*T, T)
        if self.num_conditional_channels > 0:
            if x.shape[1] == self._input_channels:
                raise ValueError(
                    f"x.shape[1]: {x.shape[1]}, self._input_channels: {self._input_channels}. please pass condition explicitly."
                )
            else:
                # exactly one of condition or static_condition should be not None
                if condition is None and static_condition is None:
                    pass
                elif condition is None:
                    assert static_condition is not None, "condition and static_condition are both None"
                    condition = static_condition
                else:
                    assert static_condition is None, "condition and static_condition are both not None"

                try:
                    condition = condition.to(dtype=x.dtype)
                    x = torch.cat((x, condition), dim=1)
                except RuntimeError as e:
                    raise RuntimeError(f"x.shape: {x.shape}, condition.shape: {condition.shape}") from e
        else:
            assert condition is None, "condition is not None but num_conditional_channels is 0"

        # Start of the model
        try:
            x = self.init_conv(x)
        except RuntimeError as e:
            raise RuntimeError(
                f"x.shape: {x.shape}, x.dtype: {x.dtype}, init_conv.weight.shape/dtype: {self.init_conv.weight.shape}/{self.init_conv.weight.dtype}"
            ) from e

        x = self.init_temporal_op(x, **kwargs_temp_op) if exists(self.init_temporal_op) else x

        r = x.clone()
        x = self.dropout_input(x)

        t = self.time_emb_mlp(ada_ln_input) if exists(self.time_emb_mlp) else None

        h = []
        for i, (block1, block2, attn, temporal_op, downsample) in enumerate(self.downs):
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            x = temporal_op(x, **kwargs_temp_op)
            h.append(x)

            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        # log.info(f'mid_attn: {x.shape}')  # e.g. [10, 256, 45, 90])
        x = self.mid_temporal_op(x, **kwargs_temp_op)
        x = self.mid_block2(x, t)
        if get_intermediate_shapes:
            return x

        for block1, block2, attn, temporal_op, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = temporal_op(x, **kwargs_temp_op)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t_dim)
        return_dict = x
        if return_time_emb:
            return return_dict, t
        return return_dict


if __name__ == "__main__":
    import time as time_clock

    temp_ops = ["conv_1d", "conv_3d", "attn", "attn"]
    for i, temp_op in enumerate(temp_ops):
        unet = Unet(
            dim=64,
            num_input_channels=3,
            num_output_channels=12,
            num_temporal_channels=6,
            spatial_shape=(64, 64),
            temporal_op=temp_op,
            use_rotary_emb=i >= 3,
        )
        x = torch.rand(10, 3, 6, 64, 64)
        time_tens = torch.rand(10, 6)
        # time it:
        t_start = time_clock.time()
        out = unet(x, time_tens)
        duration = time_clock.time() - t_start
        # log.info(f"{temp_op} - In: {x.shape}, Out: {out.shape}. Duration (s): {duration:.3f}")
