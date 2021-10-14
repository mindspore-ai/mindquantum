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
"""Quantum unitary coupled cluster."""

from mindquantum.third_party.unitary_cc import uccsd_singlet_generator
from mindquantum.third_party.unitary_cc import uccsd_singlet_get_packed_amplitudes
from .qubit_hamiltonian import get_qubit_hamiltonian
from .uccsd0 import uccsd0_singlet_generator
from .quccsd import quccsd_generator

__all__ = [
    'uccsd_singlet_generator', 'uccsd_singlet_get_packed_amplitudes',
    'get_qubit_hamiltonian', 'uccsd0_singlet_generator', 'quccsd_generator'
]

__all__.sort()
