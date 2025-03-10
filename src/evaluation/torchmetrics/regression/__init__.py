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
from src.evaluation.torchmetrics.regression.concordance import ConcordanceCorrCoef
from src.evaluation.torchmetrics.regression.cosine_similarity import CosineSimilarity
from src.evaluation.torchmetrics.regression.crps import ContinuousRankedProbabilityScore
from src.evaluation.torchmetrics.regression.csi import CriticalSuccessIndex
from src.evaluation.torchmetrics.regression.explained_variance import ExplainedVariance
from src.evaluation.torchmetrics.regression.gradient_magnitude_percent_diff import GradientMagnitudePercentDifference
from src.evaluation.torchmetrics.regression.kendall import KendallRankCorrCoef
from src.evaluation.torchmetrics.regression.kl_divergence import KLDivergence
from src.evaluation.torchmetrics.regression.log_cosh import LogCoshError
from src.evaluation.torchmetrics.regression.log_mse import MeanSquaredLogError
from src.evaluation.torchmetrics.regression.mae import MeanAbsoluteError
from src.evaluation.torchmetrics.regression.mape import MeanAbsolutePercentageError
from src.evaluation.torchmetrics.regression.mean import Average
from src.evaluation.torchmetrics.regression.mean_error import MeanError
from src.evaluation.torchmetrics.regression.minkowski import MinkowskiDistance
from src.evaluation.torchmetrics.regression.mse import MeanSquaredError
from src.evaluation.torchmetrics.regression.nrmse import NormalizedRootMeanSquaredError
from src.evaluation.torchmetrics.regression.pearson import PearsonCorrCoef
from src.evaluation.torchmetrics.regression.r2 import R2Score
from src.evaluation.torchmetrics.regression.rse import RelativeSquaredError
from src.evaluation.torchmetrics.regression.spearman import SpearmanCorrCoef
from src.evaluation.torchmetrics.regression.spread_skill_ratio import SpreadSkillRatio
from src.evaluation.torchmetrics.regression.symmetric_mape import SymmetricMeanAbsolutePercentageError
from src.evaluation.torchmetrics.regression.tweedie_deviance import TweedieDevianceScore
from src.evaluation.torchmetrics.regression.wmape import WeightedMeanAbsolutePercentageError


__all__ = [
    "Average",
    "ConcordanceCorrCoef",
    "ContinuousRankedProbabilityScore",
    "CosineSimilarity",
    "CriticalSuccessIndex",
    "ExplainedVariance",
    "GradientMagnitudePercentDifference",
    "KLDivergence",
    "KendallRankCorrCoef",
    "LogCoshError",
    "MeanAbsoluteError",
    "MeanAbsolutePercentageError",
    "MeanError",
    "MeanSquaredError",
    "MeanSquaredLogError",
    "MinkowskiDistance",
    "NormalizedRootMeanSquaredError",
    "PearsonCorrCoef",
    "R2Score",
    "RelativeSquaredError",
    "SpearmanCorrCoef",
    "SpreadSkillRatio",
    "SymmetricMeanAbsolutePercentageError",
    "TweedieDevianceScore",
    "WeightedMeanAbsolutePercentageError",
]
