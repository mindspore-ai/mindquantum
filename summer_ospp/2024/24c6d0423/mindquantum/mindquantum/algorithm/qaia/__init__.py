# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Quantum Annealing-Inspired Algorithms."""

from .CAC import CAC
from .CFC import CFC
from .LQA import LQA
from .NMFA import NMFA
from .QAIA import QAIA
from .SB import ASB, BSB, DSB, BSB_INT8, BSB_HALF, DSB_INT8, DSB_HALF
from .SFC import SFC
from .SimCIM import SimCIM

__all__ = [
    "QAIA",
    "CAC",
    "CFC",
    "LQA",
    "NMFA",
    "ASB",
    "BSB",
    "DSB",
    "SFC",
    "SimCIM",
    "BSB_INT8",
    "BSB_HALF",
    "DSB_INT8",
    "DSB_HALF",
]
