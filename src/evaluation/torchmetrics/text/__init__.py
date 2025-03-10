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
from src.evaluation.torchmetrics.text.bleu import BLEUScore
from src.evaluation.torchmetrics.text.cer import CharErrorRate
from src.evaluation.torchmetrics.text.chrf import CHRFScore
from src.evaluation.torchmetrics.text.edit import EditDistance
from src.evaluation.torchmetrics.text.eed import ExtendedEditDistance
from src.evaluation.torchmetrics.text.mer import MatchErrorRate
from src.evaluation.torchmetrics.text.perplexity import Perplexity
from src.evaluation.torchmetrics.text.rouge import ROUGEScore
from src.evaluation.torchmetrics.text.sacre_bleu import SacreBLEUScore
from src.evaluation.torchmetrics.text.squad import SQuAD
from src.evaluation.torchmetrics.text.ter import TranslationEditRate
from src.evaluation.torchmetrics.text.wer import WordErrorRate
from src.evaluation.torchmetrics.text.wil import WordInfoLost
from src.evaluation.torchmetrics.text.wip import WordInfoPreserved
from src.evaluation.torchmetrics.utilities.imports import _TRANSFORMERS_GREATER_EQUAL_4_4


__all__ = [
    "BLEUScore",
    "CHRFScore",
    "CharErrorRate",
    "EditDistance",
    "ExtendedEditDistance",
    "MatchErrorRate",
    "Perplexity",
    "ROUGEScore",
    "SQuAD",
    "SacreBLEUScore",
    "TranslationEditRate",
    "WordErrorRate",
    "WordInfoLost",
    "WordInfoPreserved",
]

if _TRANSFORMERS_GREATER_EQUAL_4_4:
    from src.evaluation.torchmetrics.text.bert import BERTScore
    from src.evaluation.torchmetrics.text.infolm import InfoLM

    __all__ += ["BERTScore", "InfoLM"]
