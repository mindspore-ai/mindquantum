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
"""Quantum neural networks operators and cells."""

from .pqc import generate_pqc_operator, PQC
from .mindquantum_layer import MindQuantumLayer
from .evolution import generate_evolution_operator, Evolution
from .mindquantum_ansatz_only_layer import MindQuantumAnsatzOnlyLayer
from .mindquantum_ansatz_only_layer import MindQuantumAnsatzOnlyOperator

__all__ = [
    "generate_pqc_operator", "PQC", "MindQuantumLayer",
    "generate_evolution_operator", "Evolution", "MindQuantumAnsatzOnlyLayer",
    "MindQuantumAnsatzOnlyOperator"
]

__all__.sort()
