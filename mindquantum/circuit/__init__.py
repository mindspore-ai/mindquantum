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
from .module_circuit import UN, SwapParts
from .uccsd import generate_uccsd
from .uccsd import decompose_single_term_time_evolution


__all__ = [
    'Circuit', 'pauli_word_to_circuits', 'UN', 'SwapParts', 'generate_uccsd',
    'decompose_single_term_time_evolution'
]

__all__.sort()
