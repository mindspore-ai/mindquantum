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
MindQuantum operators library. An operator is composed of a combination of one or more basic gates.

Contains classes representing:

- Qubit operators

- Fermion operators

- TimeEvolution operator

"""

from mindquantum.third_party.interaction_operator import InteractionOperator

from ._term_value import TermValue
from .fermion_operator import FermionOperator
from .hamiltonian import Hamiltonian
from .polynomial_tensor import PolynomialTensor
from .projector import Projector
from .qubit_excitation_operator import QubitExcitationOperator
from .qubit_operator import QubitOperator
from .time_evolution import TimeEvolution
from .utils import (
    commutator,
    count_qubits,
    down_index,
    get_fermion_operator,
    ground_state_of_sum_zz,
    hermitian_conjugated,
    normal_ordered,
    number_operator,
    sz_operator,
    up_index,
)

__all__ = [
    "FermionOperator",
    "Hamiltonian",
    "PolynomialTensor",
    "Projector",
    "QubitExcitationOperator",
    "QubitOperator",
    "TimeEvolution",
    "TermValue",
    "count_qubits",
    "commutator",
    "normal_ordered",
    "get_fermion_operator",
    "number_operator",
    "hermitian_conjugated",
    "up_index",
    "down_index",
    "sz_operator",
    "ground_state_of_sum_zz",
]
__all__.append('InteractionOperator')
__all__.sort()
