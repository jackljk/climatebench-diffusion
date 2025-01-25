# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Literal

import torch
import torch.nn as nn

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd


# layer normalization
try:
    from apex.normalization import FusedLayerNorm
except ImportError:
    from torch.nn import LayerNorm as FusedLayerNorm  # type: ignore
from einops import rearrange

# helpers
from modulus.models.sfno.initialization import trunc_normal_
from torch.utils.checkpoint import checkpoint

from src.models._base_model import BaseModel
from src.models.modules.drop_path import DropPath
from src.models.modules.misc import get_time_embedder

# more distributed stuff
from src.models.sfno.distributed import comm
from src.models.sfno.distributed.layer_norm import DistributedInstanceNorm2d

# wrap fft, to unify interface to spectral transforms
from src.models.sfno.distributed.layers import (
    DistributedInverseRealFFT2,
    DistributedMLP,
    DistributedRealFFT2,
)

# import global convolution and non-linear spectral layers
from src.models.sfno.layers import MLP, RealFFT2, SpectralAttention2d
from src.models.sfno.s2convolutions import SpectralAttentionS2, SpectralConvS2
from src.utilities.utils import raise_error_if_invalid_value


# from src.models.module import Module
# from src.models.meta import ModelMetaData


# @dataclass
# class MetaData(ModelMetaData):
#     name: str = "SFNO"
#     # Optimization
#     jit: bool = False
#     cuda_graphs: bool = True
#     amp_cpu: bool = True
#     amp_gpu: bool = True
#     torch_fx: bool = False
#     # Inference
#     onnx: bool = False
#     # Physics informed
#     func_torch: bool = False
#     auto_grad: bool = False


class SpectralFilterLayer(nn.Module):
    """Spectral filter layer"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="block-diagonal",
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        hidden_size_factor=1,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        drop_rate=0.0,
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear" and (
            isinstance(forward_transform, th.RealSHT) or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralAttentionS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                operator_type=operator_type,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        elif filter_type == "non-linear" and (
            isinstance(forward_transform, RealFFT2) or isinstance(forward_transform, DistributedRealFFT2)
        ):
            self.filter = SpectralAttention2d(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        # spectral transform is passed to the module
        elif filter_type == "linear" and (
            isinstance(forward_transform, th.RealSHT) or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            if drop_rate > 0.0:
                print("Dropout is not used for linear filters!")
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=True,
                use_tensorly=False if factorization is None else True,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator Block"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        mlp_ratio=2.0,
        drop_rate_filter=0.0,
        drop_rate_mlp=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=(nn.LayerNorm, nn.LayerNorm),
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        use_mlp=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=0,
        time_emb_dim: int = None,
        time_scale_shift_before_filter: bool = True,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

        if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
            self.input_shape_loc = (
                forward_transform.nlat_local,
                forward_transform.nlon_local,
            )
            self.output_shape_loc = (
                inverse_transform.nlat_local,
                inverse_transform.nlon_local,
            )
        else:
            self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
            self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = norm_layer[0]()

        # time embedding
        if time_emb_dim is not None:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_emb_dim, embed_dim * 2),  # 2 for scale and shift
            )
            self.time_scale_shift_before_filter = time_scale_shift_before_filter
        else:
            self.time_mlp = None
            self.time_scale_shift_before_filter = False

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            sparsity_threshold,
            use_complex_kernels=use_complex_kernels,
            hidden_size_factor=mlp_ratio,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            drop_rate=drop_rate_filter,
        )

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

        if filter_type == "linear" or filter_type == "real linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()

        if use_mlp:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate_mlp,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()
        elif outer_skip is None:
            self.outer_skip = None
        else:
            raise NotImplementedError(f"outer_skip={outer_skip} is not implemented")

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

    def time_scale_shift(self, x, time_emb):
        assert time_emb is not None, "time_emb is None but time_scale_shift is called"
        time_emb = self.time_mlp(time_emb)
        time_emb = rearrange(time_emb, "b c -> b c 1 1")
        scale, shift = time_emb.chunk(2, dim=1)  # split into scale and shift (channel dim)
        # shapesL scale/shift: (b, 1, 1, hidden/emb/channel-dim), x: (b, h, w, hidden/emb/channel-dim)
        x = x * (scale + 1) + shift
        return x

    def forward(self, x, time_emb=None):
        x_norm = torch.zeros_like(x)
        if x_norm.shape == x.shape:
            x_norm = self.norm0(x)
        else:
            x_norm[..., : self.input_shape_loc[0], : self.input_shape_loc[1]] = self.norm0(
                x[..., : self.input_shape_loc[0], : self.input_shape_loc[1]]
            )

        if self.time_scale_shift_before_filter and self.time_mlp is not None:
            x_norm = self.time_scale_shift(x_norm, time_emb)

        x, residual = self.filter(x_norm)

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
                x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x_norm = torch.zeros_like(x)
        if x_norm.shape == x.shape:
            x_norm = self.norm1(x)
        else:
            x_norm[..., : self.output_shape_loc[0], : self.output_shape_loc[1]] = self.norm1(
                x[..., : self.output_shape_loc[0], : self.output_shape_loc[1]]
            )
        x = x_norm

        if not self.time_scale_shift_before_filter and self.time_mlp is not None:
            x = self.time_scale_shift(x, time_emb)

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if self.outer_skip is not None:
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(BaseModel):
    """
    Spherical Fourier Neural Operator Network

    Parameters
    ----------
    params : dict
        Dictionary of parameters
    spectral_transform : str, optional
        Type of spectral transformation to use, by default "sht"
    filter_type : str, optional
        Type of filter to use ('linear', 'non-linear'), by default "non-linear"
    operator_type : str, optional
        Type of operator to use ('diaginal', 'dhconv'), by default "diagonal"
    img_shape : tuple, optional
        Shape of the input channels, by default (721, 1440)
    scale_factor : int, optional
        Scale factor to use, by default 16
    in_chans : int, optional
        Number of input channels, by default 2
    out_chans : int, optional
        Number of output channels, by default 2
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of layers in the network, by default 12
    use_mlp : int, optional
        Whether to use MLP, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
    pos_embed : bool, optional
        Whether to use positional embedding, by default True
    dropout : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    num_blocks : int, optional
        Number of blocks in the network, by default 16
    sparsity_threshold : float, optional
        Threshold for sparsity, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding to apply, by default 1.0
    use_complex_kernels : bool, optional
        Whether to use complex kernels, by default True
    big_skip : bool, optional
        Whether to use big skip connections, by default True
    rank : float, optional
        Rank of the approximation, by default 1.0
    factorization : Any, optional
        Type of factorization to use, by default None
    separable : bool, optional
        Whether to use separable convolutions, by default False
    complex_network : bool, optional
        Whether to use a complex network architecture, by default True
    complex_activation : str, optional
        Type of complex activation function to use, by default "real"
    spectral_layers : int, optional
        Number of spectral layers, by default 3
    checkpointing : int, optional
        Number of checkpointing segments, by default 0

    Example:
    --------
    >>> from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet as SFNO
    >>> model = SFNO(
    ...         params={},
    ...         img_shape=(8, 16),
    ...         scale_factor=4,
    ...         in_chans=2,
    ...         out_chans=2,
    ...         embed_dim=16,
    ...         num_layers=2,
    ...         encoder_layers=1,
    ...         num_blocks=4,
    ...         spectral_layers=2,
    ...         use_mlp=True,)
    >>> model(torch.randn(1, 2, 8, 16)).shape
    torch.Size([1, 2, 8, 16])
    """

    def __init__(
        self,
        params: dict = None,
        spectral_transform: str = "sht",
        filter_type: str = "linear",
        operator_type: str = "diagonal",
        # img_shape: Tuple[int] = (721, 1440),
        scale_factor: int = 16,
        # in_chans: int = 2,
        # out_chans: int = 2,
        embed_dim: int = 256,
        num_layers: int = 12,
        use_mlp: int = True,
        mlp_ratio: int = 2.0,
        activation_function: str = "gelu",
        encoder_layers: int = 1,
        pos_embed: bool = True,
        dropout_filter: float = 0.0,
        dropout_mlp: float = 0.0,
        pos_emb_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        num_blocks: int = 16,
        sparsity_threshold: float = 0.0,
        normalization_layer: str = "instance_norm",
        hard_thresholding_fraction: float = 1.0,
        use_complex_kernels: bool = True,
        big_skip: bool = True,
        rank: float = 1.0,
        factorization: Any = None,
        separable: bool = False,
        complex_network: bool = True,
        complex_activation: str = "real",
        spectral_layers: int = 3,
        checkpointing: int = 0,
        with_time_emb: bool = False,
        time_dim_mult: int = 2,
        time_rescale: bool = False,
        time_scale_shift_before_filter: bool = True,
        non_spatial_conditioning_mode: str = None,
        data_grid: Literal["legendre-gauss", "equiangular"] = "equiangular",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if self.hparams.debug_mode:
            self.log_text.info(f"Using debug mode for SFNO.")
            embed_dim = self.hparams.embed_dim = 16
            num_layers = self.hparams.num_layers = 2
        # super(SphericalFourierNeuralOperatorNet, self).__init__(meta=MetaData())
        params = params or {}
        self.params = params
        self.spectral_transform = (
            params.spectral_transform if hasattr(params, "spectral_transform") else spectral_transform
        )
        self.filter_type = params.filter_type if hasattr(params, "filter_type") else filter_type
        self.operator_type = params.operator_type if hasattr(params, "operator_type") else operator_type
        self.img_shape = (
            (params.img_shape_x, params.img_shape_y)
            if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y")
            else self.spatial_shape_in
        )
        self.scale_factor = params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        self.in_chans = (
            params.N_in_channels
            if hasattr(params, "N_in_channels")
            else self.num_input_channels + self.num_conditional_channels
        )
        self.out_chans = params.N_out_channels if hasattr(params, "N_out_channels") else self.num_output_channels
        self.embed_dim = self.num_features = params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        self.num_layers = params.num_layers if hasattr(params, "num_layers") else num_layers
        self.num_blocks = params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        self.hard_thresholding_fraction = (
            params.hard_thresholding_fraction
            if hasattr(params, "hard_thresholding_fraction")
            else hard_thresholding_fraction
        )
        self.normalization_layer = (
            params.normalization_layer if hasattr(params, "normalization_layer") else normalization_layer
        )
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp
        self.activation_function = (
            params.activation_function if hasattr(params, "activation_function") else activation_function
        )
        self.encoder_layers = params.encoder_layers if hasattr(params, "encoder_layers") else encoder_layers
        self.pos_embed = params.pos_embed if hasattr(params, "pos_embed") else pos_embed
        self.big_skip = params.big_skip if hasattr(params, "big_skip") else big_skip
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.factorization = params.factorization if hasattr(params, "factorization") else factorization
        self.separable = params.separable if hasattr(params, "separable") else separable
        self.complex_network = params.complex_network if hasattr(params, "complex_network") else complex_network
        self.complex_activation = (
            params.complex_activation if hasattr(params, "complex_activation") else complex_activation
        )
        self.spectral_layers = params.spectral_layers if hasattr(params, "spectral_layers") else spectral_layers
        self.checkpointing = params.checkpointing if hasattr(params, "checkpointing") else checkpointing
        valid_cond_modes = ["cross_attn", "adaLN", None]
        raise_error_if_invalid_value(non_spatial_conditioning_mode, valid_cond_modes, "non_spatial_conditioning_mode")
        if non_spatial_conditioning_mode == "adaLN" and with_time_emb:
            raise ValueError("adaLN is not compatible with with_time_emb=True")
        if self.num_conditional_channels_non_spatial == 0:
            assert non_spatial_conditioning_mode is None, "non_spatial_conditioning_mode is not None"
            self.non_spatial_conditioning_mode = None
        else:
            assert non_spatial_conditioning_mode is not None, "non_spatial_conditioning_mode is None"
            self.log_text.info(f"Using non_spatial_conditioning_mode: ``{non_spatial_conditioning_mode}``")
            self.non_spatial_conditioning_mode = non_spatial_conditioning_mode
        # self.pretrain_encoding = params.pretrain_encoding if hasattr(params, "pretrain_encoding") else False

        # compute the downscaled image size
        self.h = int(self.img_shape[0] // self.scale_factor)
        self.w = int(self.img_shape[1] // self.scale_factor)

        # Compute the maximum frequencies in h and in w
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

        # determine the global padding
        img_dist_h = (self.img_shape[0] + comm.get_size("h") - 1) // comm.get_size("h")
        img_dist_w = (self.img_shape[1] + comm.get_size("w") - 1) // comm.get_size("w")
        self.padding = (
            img_dist_h * comm.get_size("h") - self.img_shape[0],
            img_dist_w * comm.get_size("w") - self.img_shape[1],
        )

        # prepare the spectral transforms
        if self.spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # parallelism
            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
                sht_handle = thd.DistributedRealSHT
                isht_handle = thd.DistributedInverseRealSHT

            # set up
            self.trans_down = sht_handle(*self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid).float()
            self.itrans_up = isht_handle(*self.img_shape, lmax=modes_lat, mmax=modes_lon, grid=data_grid).float()
            self.trans = sht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()
            self.itrans = isht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss").float()

        elif self.spectral_transform == "fft":
            fft_handle = th.RealFFT2
            ifft_handle = th.InverseRealFFT2

            # effective image size:
            self.img_shape_eff = [
                self.img_shape[0] + self.padding[0],
                self.img_shape[1] + self.padding[1],
            ]
            self.img_shape_loc = [
                self.img_shape_eff[0] // comm.get_size("h"),
                self.img_shape_eff[1] // comm.get_size("w"),
            ]

            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2

            self.trans_down = fft_handle(*self.img_shape_eff, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up = ifft_handle(*self.img_shape_eff, lmax=modes_lat, mmax=modes_lon).float()
            self.trans = fft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans = ifft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
            self.img_shape_loc = (
                self.trans_down.nlat_local,
                self.trans_down.nlon_local,
            )
            self.img_shape_eff = [
                self.trans_down.nlat_local + self.trans_down.nlatpad_local,
                self.trans_down.nlon_local + self.trans_down.nlonpad_local,
            ]
            self.h_loc = self.itrans.nlat_local
            self.w_loc = self.itrans.nlon_local
        else:
            self.img_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
            self.img_shape_eff = (self.trans_down.nlat, self.trans_down.nlon)
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

        # determine activation function
        if self.activation_function == "relu":
            self.activation_function = nn.ReLU
        elif self.activation_function == "gelu":
            self.activation_function = nn.GELU
        elif self.activation_function == "silu":
            self.activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {self.activation_function}")

        # encoder
        encoder_hidden_dim = self.embed_dim
        current_dim = self.in_chans
        encoder_modules = []
        for i in range(self.encoder_layers):
            encoder_modules.append(nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True))
            encoder_modules.append(self.activation_function())
            current_dim = encoder_hidden_dim
        encoder_modules.append(nn.Conv2d(current_dim, self.embed_dim, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)

        # dropout
        self.pos_drop = nn.Dropout(p=pos_emb_dropout) if pos_emb_dropout > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(
                nn.LayerNorm,
                normalized_shape=(self.img_shape_loc[0], self.img_shape_loc[1]),
                eps=1e-6,
            )
            norm_layer1 = partial(nn.LayerNorm, normalized_shape=(self.h_loc, self.w_loc), eps=1e-6)
        elif self.normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer0 = partial(
                    DistributedInstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                )
            else:
                norm_layer0 = partial(
                    nn.InstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                    track_running_stats=False,
                )
            norm_layer1 = norm_layer0
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
        else:
            raise NotImplementedError(f"Error, normalization {self.normalization_layer} not implemented.")
        # time embedding
        self.time_dim = None
        self.with_time_emb = with_time_emb
        if with_time_emb or self.non_spatial_conditioning_mode == "adaLN":
            if with_time_emb:
                pos_emb_dim = self.embed_dim
                sinusoidal_embedding = "true"
            else:
                pos_emb_dim = self.non_spatial_cond_hdim
                sinusoidal_embedding = None

            self.time_dim = self.embed_dim * time_dim_mult
            self.time_rescale = time_rescale
            self.min_time, self.max_time = None, None
            self.time_scaler = 1.0
            self.time_shift = 0.0
            self.time_emb_mlp = get_time_embedder(self.time_dim, pos_emb_dim, sinusoidal_embedding)
        else:
            self.time_rescale = False

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):
            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear"
            outer_skip = "identity"

            if first_layer:
                norm_layer = (norm_layer0, norm_layer1)
            elif last_layer:
                norm_layer = (norm_layer1, norm_layer0)
            else:
                norm_layer = (norm_layer1, norm_layer1)

            filter_type = self.filter_type

            operator_type = self.operator_type

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                drop_rate_filter=dropout_filter,
                drop_rate_mlp=dropout_mlp,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                sparsity_threshold=sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
                rank=self.rank,
                factorization=self.factorization,
                separable=self.separable,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                checkpointing=self.checkpointing,
                time_emb_dim=self.time_dim,
                time_scale_shift_before_filter=time_scale_shift_before_filter,
            )

            self.blocks.append(block)

        self.decoder = self.get_head()
        # learned position embedding
        if self.pos_embed:
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, self.embed_dim, self.img_shape_loc[0], self.img_shape_loc[1]))
            # self.pos_embed = nn.Parameter( torch.zeros(1, self.embed_dim, self.img_shape_eff[0], self.img_shape_eff[1]) )
            self.pos_embed.is_shared_mp = ["matmul"]
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def get_head(self):
        decoder_hidden_dim = self.embed_dim
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_modules = []
        for i in range(self.encoder_layers):
            decoder_modules.append(nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True))
            decoder_modules.append(self.activation_function())
            current_dim = decoder_hidden_dim
        decoder_modules.append(nn.Conv2d(current_dim, self.out_chans, 1, bias=False))
        decoder = nn.Sequential(*decoder_modules)
        return decoder

    def set_head_to_identity(self):
        self.decoder = nn.Identity()

    def _init_weights(self, m):
        """Helper routine for weight initialization"""
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, FusedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):  # pragma: no cover
        """Helper"""
        return {"pos_embed", "cls_token"}

    def set_min_max_time(self, min_time: float, max_time: float):
        """Use time stats to rescale time input to [0, 1000].
        For example, if min_time = 0 and max_time = 100, then time_scaler = 10.0.
        """
        self.min_time, self.max_time = min_time, max_time
        if self.time_rescale:
            self.time_scaler = 1000.0 / (max_time - min_time)
            self.time_shift = -min_time
            self.log_text.info(
                f"Time rescaling: min_time: {min_time}, max_time: {max_time}, time_scaler: {self.time_scaler}, time_shift: {self.time_shift}"
            )
        else:
            self.log_text.info(f"Time stats will be checked: min_time: {min_time}, max_time: {max_time}")

    def forward_features(self, x, time=None):
        if self.with_time_emb:
            assert (
                self.min_time is not None and self.max_time is not None
            ), "min_time and max_time must be set before using time embedding"
            assert (self.min_time <= time).all() and (
                time <= self.max_time
            ).all(), f"time must be in [{self.min_time}, {self.max_time}], but time is {time}"
            if self.time_rescale:
                time = time * self.time_scaler + self.time_shift
            t_repr = self.time_emb_mlp(time)
        else:
            t_repr = None

        for i, blk in enumerate(self.blocks):
            # if x.shape[0] == 0: raise ValueError(f'x.shape[0] == 0. x.shape: {x.shape}, block i: {i}')
            if self.checkpointing >= 3:
                x = checkpoint(blk, x, time_emb=t_repr)
            else:
                x = blk(x, time_emb=t_repr)
        return x, t_repr

    def forward(
        self,
        inputs,
        time=None,
        condition=None,
        dynamical_condition=None,
        static_condition=None,
        return_time_emb: bool = False,
        **kwargs,
    ):
        # print(f"{(inputs.shape if inputs is not None else None)}, {(condition.shape if condition is not None else None)}, {(static_condition.shape if static_condition is not None else None)}")
        x = self.concat_condition_if_needed(inputs, condition, dynamical_condition, static_condition)
        # if x.shape[0] == 0: raise ValueError(f'x.shape[0] == 0. x.shape: {x.shape}, inputs.shape: {inputs.shape}, condition.shape: {condition.shape}')
        # save big skip
        if self.big_skip:
            residual = x

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):
            # old way of treating unequally shaped weights
            if self.img_shape_loc != tuple(self.img_shape_eff):
                print(
                    f"Warning: using differently shaped position embedding {self.img_shape_loc} vs {self.img_shape_eff}, shape: {self.img_shape}, x.shape: {x.shape}, pos_embed.shape: {self.pos_embed.shape}"
                )
                xp = torch.zeros_like(x)
                xp[..., : self.img_shape_loc[0], : self.img_shape_loc[1]] = (
                    x[..., : self.img_shape_loc[0], : self.img_shape_loc[1]] + self.pos_embed
                )
                x = xp
            else:
                x = x + self.pos_embed

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        x, t_repr = self.forward_features(x, time)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x)
        else:
            x = self.decoder(x)

        if return_time_emb:
            return x, t_repr
        return x
