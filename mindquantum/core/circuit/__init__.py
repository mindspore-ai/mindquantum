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
from .module_circuit import U3, UN, SwapParts
from .utils import (
    AP,
    CPN,
    A,
    C,
    D,
    add_prefix,
    apply,
    as_ansatz,
    as_encoder,
    change_param_name,
    controlled,
    dagger,
    decompose_single_term_time_evolution,
    pauli_word_to_circuits,
    shift,
)

__all__ = [
    'Circuit',
    'U3',
    'UN',
    'SwapParts',
    'C',
    'A',
    'D',
    'AP',
    'CPN',
    'decompose_single_term_time_evolution',
    'pauli_word_to_circuits',
    'controlled',
    'dagger',
    'apply',
    'add_prefix',
    'change_param_name',
    'shift',
    'as_ansatz',
    'as_encoder',
]
__all__.sort()
