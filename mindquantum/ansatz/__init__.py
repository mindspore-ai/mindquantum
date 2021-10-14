# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Implementation of different ansatz."""

from ._ansatz import Ansatz
from .max_cut import MaxCutAnsatz
from .max_2_sat import Max2SATAnsatz
from .hardware_efficient import HardwareEfficientAnsatz
from .unitary_cc import UCCAnsatz
from .qubit_ucc import QubitUCCAnsatz

__all__ = ['Ansatz', 'MaxCutAnsatz', 'Max2SATAnsatz', 'HardwareEfficientAnsatz',
           "UCCAnsatz", "QubitUCCAnsatz"]
