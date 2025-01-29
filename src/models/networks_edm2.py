# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Improved diffusion model architecture proposed in the paper
"Analyzing and Improving the Training Dynamics of Diffusion Models"."""

import numpy as np
import torch

from src.models._base_model import BaseModel
from src.models.modules.misc import const_like
from src.utilities.utils import get_logger


log = get_logger(__name__)

# ----------------------------------------------------------------------------
# Normalize given tensor to unit magnitude with respect to the given
# dimensions. Default = all dimensions except the first.


def normalize(x, dim=None, eps=1e-4):
    if dim is None:
        dim = list(range(1, x.ndim))
    norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
    norm = torch.add(eps, norm, alpha=np.sqrt(norm.numel() / x.numel()))
    return x / norm.to(x.dtype)


# ----------------------------------------------------------------------------
# Upsample or downsample the given tensor with the given filter,
# or keep it as is.


def resample(x, f=[1, 1], mode="keep"):
    if mode == "keep":
        return x
    f = np.float32(f)
    assert f.ndim == 1 and len(f) % 2 == 0
    pad = (len(f) - 1) // 2
    f = f / f.sum()
    f = np.outer(f, f)[np.newaxis, np.newaxis, :, :]
    f = const_like(x, f)
    c = x.shape[1]
    if mode == "down":
        return torch.nn.functional.conv2d(x, f.tile([c, 1, 1, 1]), groups=c, stride=2, padding=pad)  # (pad,)
    assert mode == "up"
    return torch.nn.functional.conv_transpose2d(
        x, (f * 4).tile([c, 1, 1, 1]), groups=c, stride=2, padding=pad
    )  # (pad,)


# ----------------------------------------------------------------------------
# Magnitude-preserving SiLU (Equation 81).


def mp_silu(x):
    return torch.nn.functional.silu(x) / 0.596


# ----------------------------------------------------------------------------
# Magnitude-preserving sum (Equation 88).


def mp_sum(a, b, t=0.5):
    return a.lerp(b, t) / np.sqrt((1 - t) ** 2 + t**2)


# ----------------------------------------------------------------------------
# Magnitude-preserving concatenation (Equation 103).


def mp_cat(a, b, dim=1, t=0.5):
    Na = a.shape[dim]
    Nb = b.shape[dim]
    C = np.sqrt((Na + Nb) / ((1 - t) ** 2 + t**2))
    wa = C / np.sqrt(Na) * (1 - t)
    wb = C / np.sqrt(Nb) * t
    return torch.cat([wa * a, wb * b], dim=dim)


# ----------------------------------------------------------------------------
# Magnitude-preserving Fourier features (Equation 75).


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return y.to(x.dtype)


# ----------------------------------------------------------------------------
# Magnitude-preserving convolution or fully-connected layer (Equation 47)
# with force weight normalization (Equation 66).


class MPConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel):
        super().__init__()
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels, *kernel))

    def forward(self, x, gain=1):
        w = self.weight.to(torch.float32)
        if self.training:
            with torch.no_grad():
                self.weight.copy_(normalize(w))  # forced weight normalization
        w = normalize(w)  # traditional weight normalization
        w = w * (gain / np.sqrt(w[0].numel()))  # magnitude-preserving scaling
        w = w.to(x.dtype)
        if w.ndim == 2:
            return x @ w.t()
        assert w.ndim == 4
        return torch.nn.functional.conv2d(x, w, padding=w.shape[-1] // 2)  # (w.shape[-1]//2,))


# ----------------------------------------------------------------------------
# U-Net encoder/decoder block with optional self-attention (Figure 21).


class Block(torch.nn.Module):
    def __init__(
        self,
        in_channels,  # Number of input channels.
        out_channels,  # Number of output channels.
        emb_channels,  # Number of embedding channels.
        emb_init_identity=False,  # Initialize the timestep adaptive LN so that it has no effect.
        flavor="enc",  # Flavor: 'enc' or 'dec'.
        resample_mode="keep",  # Resampling: 'keep', 'up', or 'down'.
        resample_filter=[1, 1],  # Resampling filter.
        attention=False,  # Include self-attention?
        channels_per_head=64,  # Number of channels per attention head.
        dropout=0,  # Dropout probability.
        res_balance=0.3,  # Balance between main branch (0) and residual branch (1).
        attn_balance=0.3,  # Balance between main branch (0) and self-attention (1).
        clip_act=None,  # Clip output activations. None = do not clip. Was 256
    ):
        super().__init__()
        self.out_channels = out_channels
        self.flavor = flavor
        self.resample_filter = resample_filter
        self.resample_mode = resample_mode
        self.num_heads = out_channels // channels_per_head if attention else 0
        self.dropout = dropout
        self.res_balance = res_balance
        self.attn_balance = attn_balance
        self.clip_act = clip_act
        assert not emb_init_identity, "Not implemented."
        self.conv_res0 = MPConv(out_channels if flavor == "enc" else in_channels, out_channels, kernel=[3, 3])
        self.emb_gain = torch.nn.Parameter(torch.zeros([])) if emb_channels is not None else None
        self.emb_linear = MPConv(emb_channels, out_channels, kernel=[]) if emb_channels is not None else None
        self.conv_res1 = MPConv(out_channels, out_channels, kernel=[3, 3])
        self.conv_skip = MPConv(in_channels, out_channels, kernel=[1, 1]) if in_channels != out_channels else None
        self.attn_qkv = MPConv(out_channels, out_channels * 3, kernel=[1, 1]) if self.num_heads != 0 else None
        self.attn_proj = MPConv(out_channels, out_channels, kernel=[1, 1]) if self.num_heads != 0 else None

    def forward(self, x, emb, shift=None, scale=None):
        # Main branch.
        x = resample(x, f=self.resample_filter, mode=self.resample_mode)
        if self.flavor == "enc":
            if self.conv_skip is not None:
                x = self.conv_skip(x)
            x = normalize(x, dim=1)  # pixel norm

        # Residual branch.
        y = self.conv_res0(mp_silu(x))
        if self.emb_linear is not None:
            c = self.emb_linear(emb, gain=self.emb_gain) + 1
            y = y * c.unsqueeze(2).unsqueeze(3).to(y.dtype)
        y = mp_silu(y)
        if self.training and self.dropout != 0:
            y = torch.nn.functional.dropout(y, p=self.dropout)
        y = self.conv_res1(y)

        # Connect the branches.
        if self.flavor == "dec" and self.conv_skip is not None:
            x = self.conv_skip(x)
        x = mp_sum(x, y, t=self.res_balance)

        # Self-attention.
        # Note: torch.nn.functional.scaled_dot_product_attention() could be used here,
        # but we haven't done sufficient testing to verify that it produces identical results.
        if self.num_heads != 0:
            y = self.attn_qkv(x)
            y = y.reshape(y.shape[0], self.num_heads, -1, 3, y.shape[2] * y.shape[3])
            q, k, v = normalize(y, dim=2).unbind(3)  # pixel norm & split
            w = torch.einsum("nhcq,nhck->nhqk", q, k / np.sqrt(q.shape[2])).softmax(dim=3)
            y = torch.einsum("nhqk,nhck->nhcq", w, v)
            y = self.attn_proj(y.reshape(*x.shape))
            x = mp_sum(x, y, t=self.attn_balance)

        # Clip activations.
        if self.clip_act is not None:
            x = x.clip_(-self.clip_act, self.clip_act)
        return x


# ----------------------------------------------------------------------------
# EDM2 U-Net model (Figure 21).


class UNet(BaseModel):
    def __init__(
        self,
        label_dim=0,  # Class label dimensionality. 0 = unconditional.
        model_channels=192,  # Base multiplier for the number of channels.
        channel_mult=[1, 2, 3, 4],  # Per-resolution multipliers for the number of channels.
        channel_mult_noise=None,  # Multiplier for noise embedding dimensionality. None = select based on channel_mult.
        channel_mult_emb=None,  # Multiplier for final embedding dimensionality. None = select based on channel_mult.
        num_blocks=3,  # Number of residual blocks per resolution.
        attn_resolutions=[32, 16, 8],  # List of resolutions with self-attention.
        attn_levels=None,  # List of levels with self-attention.
        with_time_emb: bool = True,  # Include time embedding in the output.
        emb_init_identity: bool = False,  # Initialize the timestep adaptive LN so that it has no effect.
        outer_sample_mode: str = None,  # bilinear or nearest
        upsample_dims: tuple = None,  # (256, 256) or (128, 120) etc.
        upsample_outputs_by: int = 1,
        upsample_condition_by: int = 1,
        label_balance=0.5,  # Balance between noise embedding (0) and class embedding (1).
        concat_balance=0.5,  # Balance between skip connections (0) and main path (1).
        resample_filter=[1, 1],  # Resampling filter.
        channels_per_head=64,  # Number of channels per attention head.
        dropout=0,  # Dropout probability.
        res_balance=0.3,  # Balance between main branch (0) and residual branch (1).
        attn_balance=0.3,  # Balance between main branch (0) and self-attention (1).
        non_spatial_conditioning_mode: str = None,
        non_spatial_cond_hdim: int = None,
        null_embedding_for_non_spatial_cond: str = "zeros",  # zeros, noop, learned
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

        img_resolution = self.spatial_shape_out[0]  # Image resolution at input/output.
        in_channels = self.num_input_channels + self.num_conditional_channels
        attn_levels = attn_levels or []

        self.initialize_non_spatial_conditioning(
            non_spatial_conditioning_mode, non_spatial_cond_hdim, null_embedding_for_non_spatial_cond
        )

        cblock = [model_channels * x for x in channel_mult]
        self.label_balance = label_balance
        self.concat_balance = concat_balance
        self.out_gain = torch.nn.Parameter(torch.zeros([]))

        # Embedding.
        self.with_time_emb = with_time_emb
        if with_time_emb or non_spatial_conditioning_mode == "adaLN":
            cnoise = model_channels * channel_mult_noise if channel_mult_noise is not None else cblock[0]
            emb_channels = self.emb_channels = (
                model_channels * channel_mult_emb if channel_mult_emb is not None else max(cblock)
            )
            if with_time_emb:
                self.emb_fourier = MPFourier(cnoise)
                map_layer_in_c = cnoise
            else:
                self.emb_fourier = torch.nn.Identity()
                map_layer_in_c = self.non_spatial_cond_hdim

            self.emb_noise = MPConv(map_layer_in_c, emb_channels, kernel=[])
            self.emb_label = MPConv(label_dim, emb_channels, kernel=[]) if label_dim != 0 else None
        else:
            self.emb_noise = None
            emb_channels = self.emb_channels = None

        if outer_sample_mode is not None:
            # Upsample (45, 90) -> (48, 96) to be easier to divide by 2 multiple times
            self.upsampler = torch.nn.Upsample(size=tuple(upsample_dims), mode=outer_sample_mode)
        else:
            self.upsampler = None

        self.upsample_condition = (
            torch.nn.Upsample(scale_factor=upsample_condition_by) if upsample_condition_by > 1 else torch.nn.Identity()
        )

        block_kwargs = dict(
            emb_channels=emb_channels,
            resample_filter=resample_filter,
            channels_per_head=channels_per_head,
            dropout=dropout,
            res_balance=res_balance,
            attn_balance=attn_balance,
            emb_init_identity=emb_init_identity,
        )

        # Encoder.
        self.enc = torch.nn.ModuleDict()
        cout = in_channels + 1
        for level, channels in enumerate(cblock):
            res = img_resolution >> level
            use_attn = level in attn_levels or res in attn_resolutions
            if use_attn:
                self.log_text.info(f"Using attention at enc. {res=}, {level=}, {channels=}, {len(channel_mult)=}")
            if level == 0:
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_conv"] = MPConv(cin, cout, kernel=[3, 3])
            else:
                self.enc[f"{res}x{res}_down"] = Block(cout, cout, flavor="enc", resample_mode="down", **block_kwargs)
            for idx in range(num_blocks):
                cin = cout
                cout = channels
                self.enc[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, flavor="enc", attention=use_attn, **block_kwargs
                )

        # Decoder.
        self.dec = torch.nn.ModuleDict()
        skips = [block.out_channels for block in self.enc.values()]
        for level, channels in reversed(list(enumerate(cblock))):
            res = img_resolution >> level
            use_attn = level in attn_levels or res in attn_resolutions  # Note that level is reversed here.
            if use_attn:
                self.log_text.info(f"Using attention at dec. resolution {res}, {level=}, {channels=}")
            if level == len(cblock) - 1:
                self.dec[f"{res}x{res}_in0"] = Block(cout, cout, flavor="dec", attention=True, **block_kwargs)
                self.dec[f"{res}x{res}_in1"] = Block(cout, cout, flavor="dec", **block_kwargs)
            else:
                self.dec[f"{res}x{res}_up"] = Block(cout, cout, flavor="dec", resample_mode="up", **block_kwargs)
            for idx in range(num_blocks + 1):
                cin = cout + skips.pop()
                cout = channels
                # super_res = upsample_outputs_by > 1 and idx == num_blocks and level == 0
                self.dec[f"{res}x{res}_block{idx}"] = Block(
                    cin, cout, flavor="dec", attention=use_attn, **block_kwargs
                )
        self.out_conv = MPConv(cout, self.num_output_channels, kernel=[3, 3])

    def forward(
        self,
        inputs,
        time=None,  # was called noise_labels
        time_emb_2=None,
        class_labels=None,
        augment_labels=None,
        condition=None,
        dynamical_condition=None,
        condition_non_spatial=None,
        static_condition=None,
        return_time_emb: bool = False,
    ):
        x = self.concat_condition_if_needed(inputs, condition, dynamical_condition, static_condition)
        orig_x_shape = x.shape[-2:]
        x = self.upsampler(x) if self.upsampler is not None else x

        # Embedding.
        if self.non_spatial_conditioning_mode == "adaLN":
            assert time is None, "time is not None but non_spatial_conditioning_mode is adaLN"
            # Text emb is null if all features are zero. Check for each batch element and replace with null embedding.
            # Calculate L2 norm along embedding dimension
            assert len(condition_non_spatial.shape) == 2, f"{condition_non_spatial.shape=}"
            ada_ln_input = self.preprocess_non_spatial_conditioning(condition_non_spatial)
        elif time is not None:
            ada_ln_input = time
        else:
            assert self.emb_noise is None, "time is None but emb_noise is not None"

        if self.emb_noise is not None:
            emb = self.emb_noise(self.emb_fourier(ada_ln_input))
            if self.emb_label is not None:
                emb = mp_sum(emb, self.emb_label(class_labels * np.sqrt(class_labels.shape[1])), t=self.label_balance)
            emb = mp_silu(emb)
        else:
            emb = None

        # Encoder.
        x = torch.cat([x, torch.ones_like(x[:, :1])], dim=1)
        skips = []
        for name, block in self.enc.items():
            x = block(x) if "conv" in name else block(x, emb)
            skips.append(x)

        # Decoder.
        for name, block in self.dec.items():
            if "block" in name:
                x = mp_cat(x, skips.pop(), t=self.concat_balance)
            x = block(x, emb)

        if self.upsampler is not None:
            x = torch.nn.functional.interpolate(x, size=orig_x_shape, mode=self.hparams.outer_sample_mode)
        x = self.out_conv(x, gain=self.out_gain)
        if return_time_emb:
            return x, emb
        return x


# ----------------------------------------------------------------------------
