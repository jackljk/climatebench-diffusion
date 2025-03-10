# Copyright The Lightning team.
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
from src.evaluation.torchmetrics.image.d_lambda import SpectralDistortionIndex
from src.evaluation.torchmetrics.image.d_s import SpatialDistortionIndex
from src.evaluation.torchmetrics.image.ergas import ErrorRelativeGlobalDimensionlessSynthesis
from src.evaluation.torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from src.evaluation.torchmetrics.image.psnr import PeakSignalNoiseRatio
from src.evaluation.torchmetrics.image.psnrb import PeakSignalNoiseRatioWithBlockedEffect
from src.evaluation.torchmetrics.image.qnr import QualityWithNoReference
from src.evaluation.torchmetrics.image.rase import RelativeAverageSpectralError
from src.evaluation.torchmetrics.image.rmse_sw import RootMeanSquaredErrorUsingSlidingWindow
from src.evaluation.torchmetrics.image.sam import SpectralAngleMapper
from src.evaluation.torchmetrics.image.scc import SpatialCorrelationCoefficient
from src.evaluation.torchmetrics.image.ssim import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
)
from src.evaluation.torchmetrics.image.tv import TotalVariation
from src.evaluation.torchmetrics.image.uqi import UniversalImageQualityIndex
from src.evaluation.torchmetrics.image.vif import VisualInformationFidelity
from src.evaluation.torchmetrics.utilities.imports import (
    _TORCH_FIDELITY_AVAILABLE,
    _TORCHVISION_AVAILABLE,
)


__all__ = [
    "ErrorRelativeGlobalDimensionlessSynthesis",
    "MemorizationInformedFrechetInceptionDistance",
    "MultiScaleStructuralSimilarityIndexMeasure",
    "PeakSignalNoiseRatio",
    "PeakSignalNoiseRatioWithBlockedEffect",
    "QualityWithNoReference",
    "RelativeAverageSpectralError",
    "RootMeanSquaredErrorUsingSlidingWindow",
    "SpatialCorrelationCoefficient",
    "SpatialDistortionIndex",
    "SpectralAngleMapper",
    "SpectralDistortionIndex",
    "StructuralSimilarityIndexMeasure",
    "TotalVariation",
    "UniversalImageQualityIndex",
    "VisualInformationFidelity",
]

if _TORCH_FIDELITY_AVAILABLE:
    from src.evaluation.torchmetrics.image.fid import FrechetInceptionDistance
    from src.evaluation.torchmetrics.image.inception import InceptionScore
    from src.evaluation.torchmetrics.image.kid import KernelInceptionDistance

    __all__ += [
        "FrechetInceptionDistance",
        "InceptionScore",
        "KernelInceptionDistance",
    ]

if _TORCHVISION_AVAILABLE:
    from src.evaluation.torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    from src.evaluation.torchmetrics.image.perceptual_path_length import PerceptualPathLength

    __all__ += ["LearnedPerceptualImagePatchSimilarity", "PerceptualPathLength"]
