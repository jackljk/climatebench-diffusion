# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models".

This file extends the original networks_edm.py with a new DhariwalUNet3D class
that supports 3D convolutions with kernels of shape (1,3,3) where 1 is the time dimension.
"""
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn import SiLU
from torch.nn.functional import silu

from src.models._base_model import BaseModel
from src.models.networks_edm import (
    weight_init, Linear, modulate, GroupNorm, AttentionOp, 
    AttentionOpCausal, UNetBlock, PositionalEmbedding,
    FourierEmbedding
)
from src.utilities.utils import get_logger


log = get_logger(__name__)


class Conv3d(torch.nn.Module):
    """3D convolutional layer with support for upsampling and downsampling.
    
    This implementation is specifically designed to use 3D convolutions with
    a kernel shape of (1, k, k) - where 1 is the time dimension. This allows
    for efficient 3D convolutions without excessive reshaping.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=[1, 1],
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
        temporal_kernel=1,  # Kernel size in the time dimension
    ):
        assert not (up and down)
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        
        # For 3D convolution with kernel (t, h, w)
        # fan_in and fan_out calculations adjust for the 3D kernel
        init_kwargs = dict(
            mode=init_mode, 
            fan_in=in_channels * temporal_kernel * kernel * kernel, 
            fan_out=out_channels * temporal_kernel * kernel * kernel
        )
        
        self.conv_func = torch.nn.functional.conv3d
        self.tconv_func = torch.nn.functional.conv_transpose3d
        weight_shape = [out_channels, in_channels, temporal_kernel, kernel, kernel]
        
        self._singleton_dims = [1, 1, 1]
        self.weight = torch.nn.Parameter(weight_init(weight_shape, **init_kwargs) * init_weight) if kernel else None
        self.bias = (
            torch.nn.Parameter(weight_init([out_channels], **init_kwargs) * init_bias) if kernel and bias else None
        )
        
        # Create 3D resampling filter (1, k, k) from 2D filter (k, k) 
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        
        # Add singleton dimension for time
        if up or down:
            f = f.unsqueeze(2)  # Shape becomes [1, 1, 1, h, w]
        
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        w = self.weight.to(x.dtype) if self.weight is not None else None
        b = self.bias.to(x.dtype) if self.bias is not None else None
        f = self.resample_filter.to(x.dtype) if self.resample_filter is not None else None
        
        # Padding for spatial dimensions only (assuming kernel is 1 in time dimension)
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0
        
        # Define padding for 3D: (pad_w_left, pad_w_right, pad_h_left, pad_h_right, pad_d_left, pad_d_right)
        # Here we don't pad in the time dimension (dim 2)
        padding_3d = (w_pad, w_pad, w_pad, w_pad, 0, 0)

        if self.fused_resample and self.up and w is not None:
            # For upsampling, we use transposed conv in spatial dims only
            # First upsample spatially
            x = self.tconv_func(
                x,
                f.mul(4).repeat(self.in_channels, 1, 1, 1, 1),
                groups=self.in_channels,
                stride=(1, 2, 2),  # Don't upsample in time dimension
                padding=(0, f_pad, f_pad),  # No padding in time dimension
            )
            
            # Then apply normal 3D conv
            x = self.conv_func(
                x, 
                w, 
                padding=(0, max(w_pad - f_pad, 0), max(w_pad - f_pad, 0))
            )
            
        elif self.fused_resample and self.down and w is not None:
            # First apply 3D conv with padding
            x = self.conv_func(
                x, 
                w, 
                padding=(0, w_pad + f_pad, w_pad + f_pad)
            )
            
            # Then downsample spatially
            x = self.conv_func(
                x, 
                f.repeat(self.out_channels, 1, 1, 1, 1), 
                groups=self.out_channels, 
                stride=(1, 2, 2)  # Don't downsample in time dimension
            )
            
        else:
            if self.up:
                # Upsample spatially only
                x = self.tconv_func(
                    x, 
                    f.mul(4).repeat(self.in_channels, 1, 1, 1, 1), 
                    groups=self.in_channels, 
                    stride=(1, 2, 2),  # Don't upsample in time dimension
                    padding=(0, f_pad, f_pad)  # No padding in time dimension
                )
                
            if self.down:
                # Downsample spatially only
                x = self.conv_func(
                    x, 
                    f.repeat(self.in_channels, 1, 1, 1, 1), 
                    groups=self.in_channels, 
                    stride=(1, 2, 2),  # Don't downsample in time dimension
                    padding=(0, f_pad, f_pad)  # No padding in time dimension
                )
                
            if w is not None:
                # Apply normal 3D conv
                x = self.conv_func(x, w, padding=(0, w_pad, w_pad))
                
        if b is not None:
            x = x.add_(b.reshape(1, -1, *self._singleton_dims))
            
        return x


class UNetBlock3D(torch.nn.Module):
    """3D version of the UNetBlock for use with Conv3d layers."""
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        cross_attention=False,
        context_channels: int = False,
        create_context_length: Optional[int] = None,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=[1, 1],
        resample_proj=False,
        adaptive_scale=True,
        emb_init_identity=False,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
        temporal_kernel=1,  # Size of kernel in temporal dimension
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.attention = attention
        self.num_heads = (
            0 if attention is False else num_heads if num_heads is not None else out_channels // channels_per_head
        )
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            temporal_kernel=temporal_kernel,
            **init,
        )
        affine_init = init_zero if emb_init_identity else init
        self.affine = (
            Linear(in_features=emb_channels, out_features=out_channels * (2 if adaptive_scale else 1), **affine_init)
            if emb_channels is not None
            else None
        )
        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv3d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel=3, 
            temporal_kernel=temporal_kernel,
            **init_zero
        )
        self.dropout = torch.nn.Dropout(p=dropout)
        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                temporal_kernel=temporal_kernel,
                **init,
            )

        init_attn = init_attn if init_attn is not None else init
        attn_kwargs = dict(kernel=1, temporal_kernel=temporal_kernel)
        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv3d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels * 3, **init_attn)
            self.proj = Conv3d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_zero)
        if cross_attention:
            attn_kwargs_c = dict(kernel=1, temporal_kernel=1)
            self.norm3 = GroupNorm(num_channels=out_channels, eps=eps)
            self.to_q = Conv3d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_attn)
            if create_context_length is not None:
                if isinstance(create_context_length, int):
                    self.preprocess_cond = Linear(context_channels, out_channels * create_context_length, **init)
                else:
                    create_context_length = int(create_context_length.replace("non_linear", ""))
                    self.preprocess_cond = torch.nn.Sequential(
                        Linear(context_channels, out_channels * create_context_length, **init),
                        SiLU(),
                        Linear(out_channels * create_context_length, out_channels * create_context_length, **init),
                    )
                context_channels = out_channels
                self.reshape_cond = lambda c: rearrange(c, "b (c l) -> b c l", c=out_channels, l=create_context_length)
                self.cond_pos_embed = torch.nn.Parameter(torch.randn(1, out_channels, create_context_length) * 0.02)
            else:
                self.preprocess_cond = torch.nn.Identity()
                self.reshape_cond = lambda c: c.unsqueeze(-1)
                self.cond_pos_embed = None

            self.to_k = Conv3d(**attn_kwargs_c, in_channels=context_channels, out_channels=out_channels, **init_attn)
            self.to_v = Conv3d(**attn_kwargs_c, in_channels=context_channels, out_channels=out_channels, **init_attn)
            self.proj_cattn = Conv3d(**attn_kwargs, in_channels=out_channels, out_channels=out_channels, **init_zero)

    def forward(self, x, emb, shift=None, scale=None, context=None):
        orig = x
        # Apply first normalization and convolution
        x = self.conv0(silu(self.norm0(x)))

        # Apply adaptive scale if available
        if self.affine is None:
            assert emb is None
            x = silu(self.norm1(x))
        else:
            if self.adaptive_scale:
                if shift is None and scale is None:
                    params = self.affine(emb).unsqueeze(2).unsqueeze(3).to(x.dtype)
                    params = rearrange(params, "(b t) c h w -> b c t h w", b=x.shape[0])
                    scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))  # shift + (1 + scale) * norm(x)
            else:
                params = self.affine(emb).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(x.dtype)
                x = silu(self.norm1(x.add_(params)))

        # Apply second convolution with dropout
        x = self.conv1(self.dropout(x))
        
        # Apply skip connection
        x = x.add_(self.skip(orig) if self.skip is not None else orig)
        x = x * self.skip_scale

        # Apply self-attention if specified
        if self.num_heads:
            # Reshape x for attention: [B, C, T, H, W] -> [B*num_heads, C//num_heads, T*H*W]
            B, C, T, H, W = x.shape
            B2, C2 = B * self.num_heads, C // self.num_heads
            
            # Apply normalization and get q, k, v
            norm2_x = self.norm2(x)
            qkv = self.qkv(norm2_x)
            
            # Reshape for attention
            qkv = qkv.reshape(B2, C2, 3, T*H*W).unbind(2)
            q, k, v = qkv
            
            # Apply attention
            if self.attention == "causal":
                w = AttentionOpCausal.apply(q, k)
            else:
                w = AttentionOp.apply(q, k)
                
            # Apply attention results
            a = torch.einsum("nqk,nck->ncq", w, v)
            
            # Reshape back to original shape
            a = a.reshape(B, C, T, H, W)
            
            # Apply projection and skip connection
            x = self.proj(a).add_(x) * self.skip_scale

        # Apply cross-attention if applicable
        if hasattr(self, "norm3"):
            # Get query from normalized input
            norm3_x = self.norm3(x)
            q = self.to_q(norm3_x)
            
            # Reshape q for attention: [B, C, T, H, W] -> [B*num_heads, C//num_heads, T*H*W]
            B, C, T, H, W = x.shape
            B2, C2 = B * self.num_heads, C // self.num_heads
            q = q.reshape(B2, C2, T*H*W)
            
            # Process context
            context = self.reshape_cond(self.preprocess_cond(context))
            if self.cond_pos_embed is not None:
                context = context + self.cond_pos_embed
                
            # Get key and value from context
            k = self.to_k(context.unsqueeze(-1).unsqueeze(-1))
            v = self.to_v(context.unsqueeze(-1).unsqueeze(-1))
            
            # Reshape k, v for attention
            k = k.reshape(B2, C2, -1)
            v = v.reshape(B2, C2, -1)
            
            # Apply attention
            w = AttentionOp.apply(q, k)
            a = torch.einsum("nqk,nck->ncq", w, v)
            
            # Reshape attention results back to 3D
            a = a.reshape(B, C, T, H, W)
            
            # Apply projection and skip connection
            x = self.proj_cattn(a).add_(x) * self.skip_scale

        return x


class DhariwalUNet3D(BaseModel):
    """3D version of the DhariwalUNet with 3D convolutions (1, 3, 3).

    This implements the DhariwalUNet architecture but uses 3D convolutions with 
    a kernel size of (1, 3, 3) in the time dimension. This allows for efficient
    temporal processing without excessive reshaping.
    """
    is_3d = True
    def __init__(
        self,
        label_dim=0,
        augment_dim=0,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=[32, 16, 8],
        attn_levels=None,
        channels_per_head=64,
        dropout=0.10,
        label_dropout=0,
        with_time_emb: bool = True,
        emb_init_identity: bool = False,
        outer_sample_mode: str = None,
        upsample_dims: tuple = None,
        upsample_hidden_spatial_dims_auto: bool = False,
        upsample_outputs_by: int = 1,
        in_channel_cross_attn: bool = False,
        non_spatial_conditioning_mode: str = None,
        non_spatial_cond_hdim: int = None,
        create_context_length: int = None,
        null_embedding_for_non_spatial_cond: str = "zeros",
        temporal_kernel: int = 1,  # Size of kernel in temporal dimension (typically 1)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        if self.hparams.debug_mode:
            self.log_text.info("Running in debug mode with toy parameters.")
            self.hparams.channel_mult = channel_mult = (1, 2, 4)
            self.hparams.model_channels = model_channels = 32
            self.hparams.num_blocks = num_blocks = 2
            self.hparams.attn_resolutions = attn_resolutions = []
            self.hparams.attn_levels = attn_levels = [2]
            self.hparams.channels_per_head = channels_per_head = 16

        img_resolution = self.spatial_shape_out[0]
        in_channels = self.num_input_channels + self.num_conditional_channels
        out_channels = self.num_output_channels
        attn_levels = attn_levels or []

        self.label_dropout = label_dropout
        init = self.init = dict(init_mode="kaiming_uniform", init_weight=np.sqrt(1 / 3), init_bias=np.sqrt(1 / 3))
        init_zero = self.init_zero = dict(init_mode="kaiming_uniform", init_weight=0, init_bias=0)

        self.initialize_non_spatial_conditioning(
            non_spatial_conditioning_mode, non_spatial_cond_hdim, null_embedding_for_non_spatial_cond
        )
        
        # Mapping
        self.with_time_emb, self.augment_dim, self.label_dim = with_time_emb, augment_dim, label_dim
        if with_time_emb or non_spatial_conditioning_mode == "adaLN":
            emb_channels = self.emb_channels = model_channels * channel_mult_emb
            if with_time_emb:
                self.map_noise = PositionalEmbedding(num_channels=model_channels)
                map_layer_in_c = model_channels
            else:
                self.map_noise = torch.nn.Identity()
                map_layer_in_c = self.non_spatial_cond_hdim
            self.map_layer0 = Linear(in_features=map_layer_in_c, out_features=emb_channels, **init)
            self.map_layer1 = Linear(in_features=emb_channels, out_features=emb_channels, **init)
        else:
            self.map_noise = None
            emb_channels = self.emb_channels = None

        use_cross_attn = self.non_spatial_conditioning_mode == "cross_attn"
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=channels_per_head,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
            emb_init_identity=emb_init_identity,
            context_channels=self.non_spatial_cond_hdim,
            create_context_length=create_context_length,
            temporal_kernel=temporal_kernel,
        )

        if augment_dim:
            self.map_augment = Linear(in_features=augment_dim, out_features=model_channels, bias=False, **init_zero)
        else:
            self.map_augment = None

        if label_dim:
            self.map_label = Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )
        else:
            self.map_label = None

        if upsample_dims is not None:
            assert outer_sample_mode is not None, "Need to provide outer_sample_mode when upsample_dims is provided."
            self.upsample_dims = tuple(upsample_dims)
            self.hparams.outer_sample_mode = "trilinear" if outer_sample_mode == "bilinear" else outer_sample_mode
            self.upsampler = True
        else:
            if upsample_hidden_spatial_dims_auto and outer_sample_mode is None:
                self.hparams.outer_sample_mode = "trilinear" if outer_sample_mode == "bilinear" else outer_sample_mode
            self.upsampler = None
            self.upsample_dims = None

        # Encoder
        self.enc = torch.nn.ModuleDict()
        cout = in_channels
        skips = []
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level
            use_attn = level in attn_levels or res in attn_resolutions
            if use_attn:
                self.log_text.info(f"Using attention at enc. {res=}, {level=}, {mult=}, {len(channel_mult)=}")
            if level == 0:
                cin = cout
                cout = model_channels * mult
                if in_channel_cross_attn:
                    from src.models.stormer.climax_embedding import ClimaXEmbeddingForConvNet

                    ps = 2 if in_channel_cross_attn == "2x2" else 1
                    self.enc[f"{res}x{res}_var_emb_and_agg"] = ClimaXEmbeddingForConvNet(
                        default_vars=[f"var_{i}" for i in range(cin)],
                        img_size=upsample_dims or self.spatial_shape_in,
                        patch_size=ps,
                        embed_dim=cout,
                        num_heads=16,
                    )
                    log.info("Using in_channels_embedding cross attn instead of conv2d.")
                else:
                    self.enc[f"{res}x{res}_conv"] = Conv3d(in_channels=cin, out_channels=cout, kernel=3, **init)
            else:
                self.enc[f"{res}x{res}_down"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, down=True, **block_kwargs
                )
            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self.enc[f"{res}x{res}_block{idx}"] = UNetBlock3D(
                    in_channels=cin,
                    out_channels=cout,
                    attention=use_attn,
                    cross_attention=use_attn and use_cross_attn,
                    **block_kwargs,
                )

        skips += [block.out_channels for block in self.enc.values()]
        self.condition_readouts = ("mean", "sum", "max")
        self.condition_head = None
        if self.predict_non_spatial_condition:
            condition_head_in_dim = len(self.condition_readouts) * cout
            if self.non_spatial_conditioning_mode == "adaLN":
                condition_head_in_dim += self.time_dim
            # Add readout layer for condition prediction
            self.condition_head = torch.nn.Sequential(
                torch.nn.Linear(condition_head_in_dim, condition_head_in_dim),
                torch.nn.GELU(),
                torch.nn.Linear(condition_head_in_dim, self.non_spatial_cond_hdim),
            )

        # Decoder
        self.dec = torch.nn.ModuleDict()
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level
            use_attn = level in attn_levels or res in attn_resolutions
            if use_attn:
                self.log_text.info(
                    f"Using attention at dec. resolution {res}, {level=}, {mult=}, {len(channel_mult)=}"
                )
            if level == len(channel_mult) - 1:
                self.dec[f"{res}x{res}_in0"] = UNetBlock3D(
                    in_channels=cout, out_channels=cout, attention=True, cross_attention=use_cross_attn, **block_kwargs
                )
                self.dec[f"{res}x{res}_in1"] = UNetBlock3D(in_channels=cout, out_channels=cout, **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = UNetBlock3D(in_channels=cout, out_channels=cout, up=True, **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = model_channels * mult
                super_res = upsample_outputs_by > 1 and idx == num_blocks and level == 0
                self.dec[f"{res}x{res}_block{idx}"] = UNetBlock3D(
                    in_channels=cin,
                    out_channels=cout,
                    attention=use_attn,
                    cross_attention=use_attn and use_cross_attn,
                    up=super_res,
                    **block_kwargs,
                )
        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv3d(in_channels=cout, out_channels=out_channels, kernel=3, **init_zero)

    def forward(
        self,
        inputs,
        time=None,
        time_emb_2=None,
        class_labels=None,
        augment_labels=None,
        condition=None,
        dynamical_condition=None,
        condition_non_spatial=None,
        static_condition=None,
        return_time_emb: bool = False,
        skip_temporal_blocks: bool = False,
        metadata=None,
    ):
        # Reshape 5D input [B, C, T, H, W] to 5D [B, C, T, H, W] for 3D convolutions
        orig_shape = inputs.shape
        
        # Handle conditioning
        x = self.concat_condition_if_needed(inputs, condition, dynamical_condition, static_condition)
        if condition_non_spatial is not None:
            condition_non_spatial = self.preprocess_non_spatial_conditioning(condition_non_spatial)
        
        # Save original shape for later downsampling if needed
        orig_x_shape = x.shape[-2:]
        
        # Handle upsampling for 3D data - directly use interpolate on 5D tensor
        if self.upsampler:
            # For 5D tensor [B, C, T, H, W], we need to include T dimension in the size
            # Create a size tuple that keeps T dimension and changes H, W dimensions
            upsample_size = (x.shape[-3],) + self.upsample_dims  # (T, H', W')
            x = F.interpolate(
                x,
                size=upsample_size,  # Apply to all spatial dimensions (T, H, W)
                mode=self.hparams.outer_sample_mode
            )

        # Mapping
        if self.non_spatial_conditioning_mode == "adaLN":
            assert time is None, "time is not None but non_spatial_conditioning_mode is adaLN"
            assert len(condition_non_spatial.shape) == 2, f"{condition_non_spatial.shape=}"
            ada_ln_input = condition_non_spatial
        elif time is not None:
            ada_ln_input = time
        else:
            assert self.map_noise is None, f"time is None but {self.map_noise=}. {self.with_time_emb=}"

        if self.map_noise is not None:
            emb = self.map_noise(ada_ln_input)
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = self.map_layer1(emb)
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                emb = emb + self.map_label(tmp)
            emb = silu(emb)
        else:
            emb = None
        if time_emb_2 is None:
            time_emb_2 = emb

        for_cross_attn = condition_non_spatial if self.non_spatial_conditioning_mode == "cross_attn" else None

        # Process through encoder
        skips = []
        for block in self.enc.values():
            if isinstance(block, UNetBlock3D):
                x = block(x, emb, context=for_cross_attn)
            else:
                x = block(x)
            skips.append(x)

        # Condition prediction if needed
        if self.condition_head is not None:
            if self.condition_readouts == ("mean", "sum", "max"):
                cond_head_input = torch.cat(
                    [x.mean(dim=(-3, -2, -1)), x.sum(dim=(-3, -2, -1)), torch.amax(x, dim=(-3, -2, -1))], dim=1
                )
            else:
                raise ValueError(f"Unknown condition readouts: {self.condition_readouts}")
            if emb is not None:
                cond_head_input = torch.cat((cond_head_input, emb), dim=1)
            condition_pred = self.condition_head(cond_head_input)
        else:
            condition_pred = None

        # Process through decoder
        for block_name, block in self.dec.items():
            if isinstance(block, UNetBlock3D):
                if x.shape[1] != block.in_channels:
                    skip = skips.pop()
                    if self.hparams.upsample_hidden_spatial_dims_auto and (skip.shape[-2] != x.shape[-2] or skip.shape[-1] != x.shape[-1]):
                        # For 5D tensor, keep T dimension and resize H,W to match skip connection
                        x = F.interpolate(
                            x, 
                            size=(x.shape[-3], skip.shape[-2], skip.shape[-1]),
                            mode=self.hparams.outer_sample_mode
                        )
                    x = torch.cat([x, skip], dim=1)
                x = block(x, emb, context=for_cross_attn)
            else:
                x = block(x)

        # Final normalization and convolution
        if self.upsampler is not None:
            # Keep the time dimension intact, only resize spatial dimensions
            upsample_size = (x.shape[-3],) + orig_x_shape
            x = F.interpolate(x, size=upsample_size, mode=self.hparams.outer_sample_mode)

        x = self.out_conv(silu(self.out_norm(x)))
        return_dict = dict(preds=x, condition_non_spatial=condition_pred) if condition_pred is not None else x

        if return_time_emb:
            return return_dict, emb
        return return_dict


class TemporalAttentionBlock3D(torch.nn.Module):
    """Temporal attention block for 3D UNet.
    
    This block focuses attention specifically on the temporal dimension while
    operating directly on 3D tensors of shape [B, C, T, H, W].
    """
    def __init__(
        self,
        dim: int,
        num_heads=None,
        channels_per_head=None,
        eps=1e-5,
        init_zero=dict(init_weight=0),
        init_attn=None,
        causal: bool = False,
        non_linear_proj: bool = False,
        cond_dim: int = None,
        cond_type: str = "linear",
        mixing_factor: bool = False,
        pos_encoding: str = None,
        n_temporal_channels=None,
    ):
        super().__init__()
        assert (
            num_heads is not None or channels_per_head is not None
        ), "num_heads or channels_per_head must be provided"
        
        self.dim = dim
        self.num_heads = num_heads if num_heads is not None else dim // channels_per_head
        self.T = n_temporal_channels
        self.alpha = torch.nn.Parameter(torch.ones(1)) if mixing_factor else None
        self.causal = causal
        self.cond_type = cond_type
        
        # Normalization and attention layers
        self.norm = GroupNorm(num_channels=dim, eps=eps)
        
        # QKV projection - applied to all spatial locations in parallel
        # Each of q, k, v will have shape [B, C, T, H, W]
        self.qkv = Conv3d(
            in_channels=dim, 
            out_channels=dim * 3, 
            kernel=1, 
            temporal_kernel=1,
            **(init_attn or {})
        )
        
        # Output projection
        self.proj = Conv3d(
            in_channels=dim, 
            out_channels=dim, 
            kernel=1, 
            temporal_kernel=1,
            **init_zero
        )
        
        # Optional time embedding
        if cond_dim is None:
            self.affine = None
        elif cond_type == "linear":
            self.affine = Linear(cond_dim, 2 * dim, **init_zero)
        elif cond_type == "conv":
            self.affine = torch.nn.Conv1d(cond_dim, 2 * dim, kernel_size=3, padding=1)
        elif cond_type == "non-linear":
            self.emb_map = torch.nn.Linear(cond_dim, cond_dim)
            self.affine = torch.nn.Linear(cond_dim, 2 * dim, bias=True)
        elif cond_type == "non-linear-conv":
            self.emb_map = torch.nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1)
            self.affine = torch.nn.Conv1d(cond_dim, 2 * dim, kernel_size=3, padding=1)
        else:
            raise ValueError(f"Unknown cond_type: {cond_type}")
            
        # Positional encoding
        if pos_encoding == "learned":
            log.info(f"Using learned temporal positional encoding with max_frames={n_temporal_channels}")
            self.temporal_pos_encoding = torch.nn.Parameter(torch.randn(1, dim, n_temporal_channels, 1, 1) * 0.02)
        else:
            self.temporal_pos_encoding = None
            
        if causal:
            log.info("Using causal temporal attention")

    def forward(self, x, emb=None):
        # x has shape [B, C, T, H, W]
        B, C, T, H, W = x.shape
        assert T == self.T, f"Expected T={self.T}, got T={T}"
        
        # Store original input for residual connection
        skip = x
        
        # Add positional encoding if specified
        if self.temporal_pos_encoding is not None:
            x = x + self.temporal_pos_encoding
            
        # Apply normalization
        x = self.norm(x)
        
        # Get query, key, value projections
        qkv = self.qkv(x)  # [B, 3*C, T, H, W]
        
        # Reshape for attention computation
        # We want to apply attention across the temporal dimension only
        # First reshape to separate q, k, v and prepare for attention
        B_heads = B * self.num_heads
        C_per_head = C // self.num_heads
        
        # Reshape: [B, 3*C, T, H, W] -> [B*num_heads, C_per_head, 3, T, H*W]
        qkv = qkv.reshape(B, 3, self.num_heads, C_per_head, T, H*W)
        qkv = qkv.permute(0, 2, 3, 1, 4, 5).reshape(B_heads, C_per_head, 3, T, H*W)
        
        # Split into q, k, v
        q, k, v = qkv.unbind(2)  # Each has shape [B*num_heads, C_per_head, T, H*W]
        
        # For each spatial location (h, w), apply attention across time
        # Reshape to handle all spatial locations in parallel
        q = q.permute(0, 3, 1, 2).reshape(B_heads*H*W, C_per_head, T)  # [B*num_heads*H*W, C_per_head, T]
        k = k.permute(0, 3, 1, 2).reshape(B_heads*H*W, C_per_head, T)  # [B*num_heads*H*W, C_per_head, T]
        v = v.permute(0, 3, 1, 2).reshape(B_heads*H*W, C_per_head, T)  # [B*num_heads*H*W, C_per_head, T]
        
        # Apply attention - note that we're attending across the T dimension only
        if self.causal:
            # Create causal mask for temporal dimension
            causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(x.device)
            
            # Compute attention scores
            attn_scores = torch.bmm(q.transpose(1, 2), k) / (C_per_head ** 0.5)  # [B*num_heads*H*W, T, T]
            
            # Apply causal mask
            attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))
            attn_weights = F.softmax(attn_scores, dim=-1)
        else:
            # Regular attention
            attn_weights = F.softmax(torch.bmm(q.transpose(1, 2), k) / (C_per_head ** 0.5), dim=-1)  # [B*num_heads*H*W, T, T]
        
        # Apply attention weights to values
        attn_output = torch.bmm(attn_weights, v.transpose(1, 2))  # [B*num_heads*H*W, T, C_per_head]
        
        # Reshape back to original format
        attn_output = attn_output.transpose(1, 2)  # [B*num_heads*H*W, C_per_head, T]
        attn_output = attn_output.reshape(B_heads, H*W, C_per_head, T)  # [B*num_heads, H*W, C_per_head, T]
        attn_output = attn_output.permute(0, 2, 3, 1)  # [B*num_heads, C_per_head, T, H*W]
        attn_output = attn_output.reshape(B, C, T, H, W)  # [B, C, T, H, W]
        
        # Apply time embedding if provided
        if self.affine is not None:
            assert emb is not None, "Time embedding is required for affine transformation"
            # Process the embedding
            if "conv" in self.cond_type:
                emb = rearrange(emb, "(b t) c -> b c t", t=self.T)
                if hasattr(self, "emb_map"):
                    emb = silu(self.emb_map(emb))
                shift, scale = self.affine(emb).chunk(2, dim=1)
                
                # Add dimensions for spatial broadcasting
                shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
                scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
            else:
                emb = rearrange(emb, "(b t) c -> b t c", t=self.T)
                if hasattr(self, "emb_map"):
                    emb = silu(self.emb_map(emb))
                    
                shift, scale = self.affine(emb).chunk(2, dim=-1)
                shift, scale = shift.transpose(1, 2), scale.transpose(1, 2)  # [B, C, T]
                
                # Add dimensions for spatial broadcasting
                shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
                scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
                
            # Apply modulation: output = shift + (1 + scale) * output
            attn_output = torch.addcmul(shift, attn_output, scale + 1)
        
        # Apply projection and residual connection
        output = self.proj(attn_output)
        
        # Apply mixing factor if specified
        if self.alpha is not None:
            # Allow dynamic weighting of skip connection via learnable alpha
            with torch.no_grad():
                self.alpha.clamp_(0, 1)
            return self.alpha * skip + (1 - self.alpha) * output
        else:
            return skip + output


class TemporalConvBlock3D(torch.nn.Module):
    """Temporal convolution block for 3D UNet.
    
    This block applies convolutions specifically along the temporal dimension
    while operating directly on 3D tensors.
    """
    def __init__(
        self,
        dim: int,
        temporal_kernel_size: int = 3,
        eps=1e-5,
        dropout=0,
        init=dict(),
        init_zero=dict(init_weight=0),
        cond_dim: int = None,
        cond_type: str = "linear",
        mixing_factor: bool = False,
        n_temporal_channels=None,
    ):
        super().__init__()
        self.dim = dim
        self.T = n_temporal_channels
        self.alpha = torch.nn.Parameter(torch.ones(1)) if mixing_factor else None
        self.cond_type = cond_type
        
        # Ensure we have proper padding for the temporal dimension
        padding = (temporal_kernel_size - 1) // 2
        
        # Normalization
        self.norm = GroupNorm(num_channels=dim, eps=eps)
        
        # Temporal convolution - operates along temporal dimension only
        self.temporal_conv = Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel=1,  # Spatial kernel is 1x1
            temporal_kernel=temporal_kernel_size,  # Temporal kernel size
            **init
        )
        
        # Output projection
        self.proj = Conv3d(
            in_channels=dim,
            out_channels=dim,
            kernel=1,
            temporal_kernel=1,
            **init_zero
        )
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else None
        
        # Optional time embedding
        if cond_dim is None:
            self.affine = None
        elif cond_type == "linear":
            self.affine = Linear(cond_dim, 2 * dim, **init_zero)
        elif cond_type == "conv":
            self.affine = torch.nn.Conv1d(cond_dim, 2 * dim, kernel_size=3, padding=1)
        elif cond_type == "non-linear":
            self.emb_map = torch.nn.Linear(cond_dim, cond_dim)
            self.affine = torch.nn.Linear(cond_dim, 2 * dim, bias=True)
        elif cond_type == "non-linear-conv":
            self.emb_map = torch.nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1)
            self.affine = torch.nn.Conv1d(cond_dim, 2 * dim, kernel_size=3, padding=1)
        else:
            raise ValueError(f"Unknown cond_type: {cond_type}")

    def forward(self, x, emb=None):
        # x has shape [B, C, T, H, W]
        B, C, T, H, W = x.shape
        assert T == self.T, f"Expected T={self.T}, got T={T}"
        
        # Store original input for residual connection
        skip = x
        
        # Apply normalization
        x = self.norm(x)
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        
        # Apply time embedding if provided
        if self.affine is not None and emb is not None:
            # Process the embedding
            if "conv" in self.cond_type:
                emb = rearrange(emb, "(b t) c -> b c t", t=self.T)
                if hasattr(self, "emb_map"):
                    emb = silu(self.emb_map(emb))
                shift, scale = self.affine(emb).chunk(2, dim=1)
                
                # Add dimensions for spatial broadcasting
                shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
                scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
            else:
                emb = rearrange(emb, "(b t) c -> b t c", t=self.T)
                if hasattr(self, "emb_map"):
                    emb = silu(self.emb_map(emb))
                    
                shift, scale = self.affine(emb).chunk(2, dim=-1)
                shift, scale = shift.transpose(1, 2), scale.transpose(1, 2)  # [B, C, T]
                
                # Add dimensions for spatial broadcasting
                shift = shift.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
                scale = scale.unsqueeze(-1).unsqueeze(-1)  # [B, C, T, 1, 1]
                
            # Apply modulation: output = shift + (1 + scale) * output
            x = torch.addcmul(shift, x, scale + 1)
        
        # Apply non-linearity
        x = silu(x)
        
        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)
        
        # Apply projection
        x = self.proj(x)
        
        # Apply mixing factor if specified
        if self.alpha is not None:
            # Allow dynamic weighting of skip connection via learnable alpha
            with torch.no_grad():
                self.alpha.clamp_(0, 1)
            return self.alpha * skip + (1 - self.alpha) * x
        else:
            return skip + x


class DhariwalUNet3DTemporal(DhariwalUNet3D):
    """Enhanced 3D UNet with dedicated temporal attention/convolution blocks.
    
    This model builds on the DhariwalUNet3D by adding specific temporal processing
    blocks between the standard 3D convolution blocks.
    """
    def __init__(
        self,
        temporal_op: Optional[str] = "attn",  # 'attn' or 'conv'
        temporal_resolutions: Optional[List[int]] = None,  # List of resolutions to use temporal operation.
        init_temporal_op: bool = True,  # use temporal block after first conv or not
        temporal_op_before_down: Optional[bool] = None,
        final_temporal_op: bool = False,  # use temporal block after the last block or not
        temporal_pos_encoding: Optional[str] = None,  # frame position embedding.
        use_mixing_factor: bool = False,
        temporal_attn_type: str = "attn",  # or "conv"
        causal_attn: bool = False,
        temporal_kernel_size: int = 3,  # kernel size for temporal convolutions
        use_time_emb_for_temporal_layers: str = None,
        extra_time_emb_mlp: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.temporal_op = temporal_op
        assert temporal_op in ["attn", "conv"], f"Unknown temporal operation: {temporal_op}"
        assert self.num_temporal_channels is not None, f"num_temporal_channels: {self.num_temporal_channels}"
        
        # Create new ModuleDict with both original and temporal layers in correct order
        last_block = self.hparams.num_blocks - 1
        max_res = self.spatial_shape_out[0]
        kv_out_channels = None
        
        # Setup parameters for temporal blocks
        if temporal_op == "attn":
            temporal_block_cls = TemporalAttentionBlock3D
            t_block_kwargs = dict(
                channels_per_head=self.hparams.channels_per_head,
                n_temporal_channels=self.num_temporal_channels,
                pos_encoding=temporal_pos_encoding,
                cond_dim=self.emb_channels if use_time_emb_for_temporal_layers else None,
                cond_type=use_time_emb_for_temporal_layers,
                mixing_factor=use_mixing_factor,
                causal=causal_attn,
                init_zero=self.init_zero,
                init_attn=self.init,
            )
        else:  # temporal_op == "conv"
            temporal_block_cls = TemporalConvBlock3D
            t_block_kwargs = dict(
                n_temporal_channels=self.num_temporal_channels,
                temporal_kernel_size=temporal_kernel_size,
                cond_dim=self.emb_channels if use_time_emb_for_temporal_layers else None,
                cond_type=use_time_emb_for_temporal_layers,
                mixing_factor=use_mixing_factor,
                init=self.init,
                init_zero=self.init_zero,
            )
        
        self.cond_type = use_time_emb_for_temporal_layers
        
        # Extra time embedding MLP for temporal layers if specified
        self.extra_time_emb_mlp = extra_time_emb_mlp
        if extra_time_emb_mlp:
            log.info("Using extra time embedding MLP for temporal layers")
            model_channels = self.hparams.model_channels
            self.map_noise_T = PositionalEmbedding(num_channels=model_channels)
            if use_time_emb_for_temporal_layers == "linear":
                self.map_layer0_T = Linear(in_features=model_channels, out_features=self.emb_channels, **self.init)
                self.map_layer1_T = Linear(in_features=self.emb_channels, out_features=self.emb_channels, **self.init)
            elif use_time_emb_for_temporal_layers == "conv":
                log.info("Using conv for time embedding")
                self.map_layer0_T = torch.nn.Conv1d(model_channels, self.emb_channels, kernel_size=3, padding=1)
                self.map_layer1_T = torch.nn.Conv1d(self.emb_channels, self.emb_channels, kernel_size=3, padding=1)
            else:
                raise ValueError(f"Unknown time_emb_for_temporal_layers: {use_time_emb_for_temporal_layers}")
        
        # Create new encoder with temporal blocks
        new_enc = torch.nn.ModuleDict()
        for k, v in self.enc.items():
            res = int(k.split("x")[0])
            if temporal_resolutions is not None and res not in temporal_resolutions:
                new_enc[k] = v
                kv_out_channels = v.out_channels if hasattr(v, 'out_channels') else kv_out_channels
                continue

            if "_down" in k and temporal_op_before_down is True:
                # Insert temporal block before downsampling
                new_enc[f"{res}x{res}_temporal_before_down"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)

            new_enc[k] = v  # Add original layer
            kv_out_channels = v.out_channels if hasattr(v, 'out_channels') else kv_out_channels

            if "_conv" in k and init_temporal_op:
                # Insert temporal block after the first conv layer
                assert res == max_res, f"Temporal block should be added only to the first layer. {res=}, {max_res=}"
                new_enc[f"{res}x{res}_temporal_init"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)
                
            if "_down" in k and temporal_op_before_down is False:
                # Insert temporal block after downsampling
                new_enc[f"{res}x{res}_temporal_after_down"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)

            if k == f"{res}x{res}_block{last_block}":
                # Insert temporal block after the last block
                new_enc[f"{res}x{res}_temporal_last_block"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)

        # Replace the original enc with our new ordered version
        self.enc = new_enc

        # Create new decoder with temporal blocks
        new_dec = torch.nn.ModuleDict()
        for k, v in self.dec.items():
            res = int(k.split("x")[0])
            if temporal_resolutions is not None and res not in temporal_resolutions:
                new_dec[k] = v
                kv_out_channels = v.out_channels if hasattr(v, 'out_channels') else kv_out_channels
                continue

            if "_up" in k and temporal_op_before_down is True:
                # Insert temporal block before upsampling
                new_dec[f"{res}x{res}_temporal_before_up"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)

            new_dec[k] = v  # Add original layer
            kv_out_channels = v.out_channels if hasattr(v, 'out_channels') else kv_out_channels

            if "_in1" in k:
                # Insert temporal block after the first two decoder conv layers
                new_dec[f"{res}x{res}_temporal_in2"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)

            if "_up" in k and temporal_op_before_down is False:
                # Insert temporal block after upsampling
                new_dec[f"{res}x{res}_temporal_after_up"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)

            if res == max_res and not final_temporal_op:
                continue  # Skip the last block
            elif k == f"{res}x{res}_block{last_block+1}":
                # Insert temporal block after the last block  (one more block in decoder)
                new_dec[f"{res}x{res}_temporal_last_block"] = temporal_block_cls(kv_out_channels, **t_block_kwargs)

        # Replace the original dec with our new ordered version
        self.dec = new_dec

        # Log the structure
        log.info(f"enc = {' '.join([k for k in self.enc.keys()])}")
        log.info(f"dec = {' '.join([k for k in self.dec.keys()])}")
        
        # Initialize weights
        self.initialize_weights()
    
    def initialize_weights(self):
        # Initialize temporal blocks
        for m in self.enc.values():
            if isinstance(m, (TemporalAttentionBlock3D, TemporalConvBlock3D)):
                if hasattr(m, 'affine') and m.affine is not None:
                    m.affine.weight.data.zero_()
                    if hasattr(m.affine, 'bias') and m.affine.bias is not None:
                        m.affine.bias.data.zero_()
                    
        for m in self.dec.values():
            if isinstance(m, (TemporalAttentionBlock3D, TemporalConvBlock3D)):
                if hasattr(m, 'affine') and m.affine is not None:
                    m.affine.weight.data.zero_()
                    if hasattr(m.affine, 'bias') and m.affine.bias is not None:
                        m.affine.bias.data.zero_()
    
    def forward(
        self,
        inputs,
        time=None,
        time_emb_2=None,
        class_labels=None,
        augment_labels=None,
        condition=None,
        dynamical_condition=None,
        condition_non_spatial=None,
        static_condition=None,
        return_time_emb: bool = False,
        skip_temporal_blocks: bool = False,
        metadata=None,
    ):
        b, c, t = inputs.shape[:3]
        if static_condition is not None:
            # Copy the static condition to all time steps b c' h w -> (b t) c' h w
            static_condition = repeat(static_condition, "b cst h w -> b cst t h w", t=t)
        # Process extra time embedding if needed
        emb_t = None
        if self.extra_time_emb_mlp:
            # Process time separately for temporal layers
            assert time is not None, "Time is required when using extra_time_emb_mlp"
            
            # Reshape time to be per-timestep: [B*T] -> [B, T, C] -> process -> [B, C, T]
            B = inputs.shape[0] // self.num_temporal_channels
            emb_t = self.map_noise_T(time)
            
            if self.cond_type == "conv":
                emb_t = rearrange(emb_t, "(b t) c -> b c t", b=B, t=self.num_temporal_channels)
                
            emb_t = silu(self.map_layer1_T(silu(self.map_layer0_T(emb_t))))

            
        # Use the main forward method which sets up inputs, time embeddings, etc.
        # We just need to handle the temporal blocks in the encoder/decoder
        x = self.concat_condition_if_needed(inputs, condition, dynamical_condition, static_condition)
        if condition_non_spatial is not None:
            condition_non_spatial = self.preprocess_non_spatial_conditioning(condition_non_spatial)
        
        # Save original shape for later downsampling if needed
        orig_x_shape = x.shape[-2:]
        
        # Handle upsampling for 3D data - directly use interpolate on 5D tensor
        if self.upsampler:
            # For 5D tensor [B, C, T, H, W], we need to include T dimension in the size
            # Create a size tuple that keeps T dimension and changes H, W dimensions
            upsample_size = (x.shape[-3],) + self.upsample_dims  # (T, H', W')
            x = F.interpolate(
                x,
                size=upsample_size,  # Apply to all spatial dimensions (T, H, W)
                mode=self.hparams.outer_sample_mode
            )

        # Mapping for time embeddings
        if self.non_spatial_conditioning_mode == "adaLN":
            assert time is None, "time is not None but non_spatial_conditioning_mode is adaLN"
            assert len(condition_non_spatial.shape) == 2, f"{condition_non_spatial.shape=}"
            ada_ln_input = condition_non_spatial
        elif time is not None:
            ada_ln_input = time
        else:
            assert self.map_noise is None, f"time is None but {self.map_noise=}. {self.with_time_emb=}"

        if self.map_noise is not None:
            ada_ln_input = rearrange(ada_ln_input, "b t -> (b t)")
            emb = self.map_noise(ada_ln_input)
            if self.map_augment is not None and augment_labels is not None:
                emb = emb + self.map_augment(augment_labels)
            emb = silu(self.map_layer0(emb))
            emb = self.map_layer1(emb)
            if self.map_label is not None:
                tmp = class_labels
                if self.training and self.label_dropout:
                    tmp = tmp * (torch.rand([x.shape[0], 1], device=x.device) >= self.label_dropout).to(tmp.dtype)
                emb = emb + self.map_label(tmp)
            emb = silu(emb)
            # emb = rearrange(emb, "(b t) c -> b c t", b=b, t=t)
        else:
            emb = None

        for_cross_attn = condition_non_spatial if self.non_spatial_conditioning_mode == "cross_attn" else None

        # Define the types of blocks
        temporal_blocks = (TemporalAttentionBlock3D, TemporalConvBlock3D)
        
        # Process through encoder
        skips = []
        for block_name, block in self.enc.items():
            if isinstance(block, UNetBlock3D):
                x = block(x, emb, context=for_cross_attn)
                skips.append(x)
            elif isinstance(block, temporal_blocks):
                if not skip_temporal_blocks:
                    x = block(x, emb_t if emb_t is not None else emb)
            else:
                x = block(x)
                skips.append(x)

        # Process through decoder
        for block_name, block in self.dec.items():
            if isinstance(block, UNetBlock3D):
                if x.shape[1] != block.in_channels:
                    skip = skips.pop()
                    if self.hparams.upsample_hidden_spatial_dims_auto and (skip.shape[-2] != x.shape[-2] or skip.shape[-1] != x.shape[-1]):
                        # For 5D tensor, keep T dimension and resize H,W to match skip connection
                        x = F.interpolate(
                            x, 
                            size=(x.shape[-3], skip.shape[-2], skip.shape[-1]),
                            mode=self.hparams.outer_sample_mode
                        )
                    x = torch.cat([x, skip], dim=1)
                x = block(x, emb, context=for_cross_attn)
            elif isinstance(block, temporal_blocks):
                if not skip_temporal_blocks:
                    x = block(x, emb_t if emb_t is not None else emb)
            else:
                x = block(x)

        # Final normalization and convolution
        if self.upsampler is not None:
            # Keep the time dimension intact, only resize spatial dimensions
            upsample_size = (x.shape[-3],) + orig_x_shape
            x = F.interpolate(x, size=upsample_size, mode=self.hparams.outer_sample_mode)

        x = self.out_conv(silu(self.out_norm(x)))
        return_dict = x

        if return_time_emb:
            return return_dict, emb
        return return_dict