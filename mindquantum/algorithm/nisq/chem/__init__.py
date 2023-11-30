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

"""Algorithm for quantum chemistry."""

from mindquantum.third_party.unitary_cc import (
    uccsd_singlet_generator,
    uccsd_singlet_get_packed_amplitudes,
)

from .hardware_efficient_ansatz import HardwareEfficientAnsatz
from .more_hardware_efficient_ansatz import (
    ASWAP,
    PCHeaXYZ1F,
    PCHeaXYZ2F,
    RYCascade,
    RYFull,
    RYLinear,
    RYRZFull,
)
from .qubit_hamiltonian import get_qubit_hamiltonian
from .qubit_ucc_ansatz import QubitUCCAnsatz
from .quccsd import quccsd_generator
from .reference_state import get_reference_circuit
from .transform import Transform
from .uccsd import generate_uccsd
from .uccsd0 import uccsd0_singlet_generator
from .unitary_cc import UCCAnsatz

__all__ = [
    'Transform',
    'get_qubit_hamiltonian',
    'uccsd_singlet_generator',
    'uccsd_singlet_get_packed_amplitudes',
    'uccsd0_singlet_generator',
    'quccsd_generator',
    'HardwareEfficientAnsatz',
    'QubitUCCAnsatz',
    'generate_uccsd',
    'UCCAnsatz',
    'get_reference_circuit',
    'RYLinear',
    'RYFull',
    'RYCascade',
    'RYRZFull',
    'PCHeaXYZ1F',
    'PCHeaXYZ2F',
    'ASWAP',
]

__all__.sort()
