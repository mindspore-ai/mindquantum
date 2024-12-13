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
"""Error mitigation module."""
from .folding_circuit import fold_at_random
from .mitigation import zne
from .virtual_distillation import virtual_distillation
from .random_benchmarking import (
    generate_double_qubits_rb_circ,
    generate_single_qubit_rb_circ,
    query_double_qubits_clifford_elem,
    query_single_qubit_clifford_elem,
)

__all__ = [
    'fold_at_random',
    'zne',
    'virtual_distillation',
    'query_single_qubit_clifford_elem',
    'query_double_qubits_clifford_elem',
    'generate_single_qubit_rb_circ',
    'generate_double_qubits_rb_circ',
]
