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
"""
Circuit.

Quantum circuit module.
"""

from .circuit import Circuit
from .circuit import pauli_word_to_circuits
from .module_circuit import UN, SwapParts, U3
from .uccsd import generate_uccsd
from .uccsd import decompose_single_term_time_evolution
from .time_evolution import TimeEvolution
from .high_level_ops import controlled
from .high_level_ops import dagger
from .high_level_ops import apply
from .high_level_ops import add_prefix
from .high_level_ops import change_param_name
from .high_level_ops import A
from .high_level_ops import AP
from .high_level_ops import C
from .high_level_ops import CPN
from .high_level_ops import D
from .state_evolution import StateEvolution
from .quantum_fourier import qft

__all__ = [
    'Circuit', 'StateEvolution', 'TimeEvolution', 'U3', 'UN', 'SwapParts',
    'qft', 'pauli_word_to_circuits', 'decompose_single_term_time_evolution',
    'generate_uccsd', 'controlled', 'dagger', 'apply', 'add_prefix',
    'change_param_name', 'C', 'D', 'A', 'AP', 'CPN'
]
