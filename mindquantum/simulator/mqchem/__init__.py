# Copyright 2025 Huawei Technologies Co., Ltd
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
MindQuantum Chemistry Simulator based on Configuration Interaction.

This module provides a specialized simulator (`MQChemSimulator`) and its
associated components (`UCCExcitationGate`, `CIHamiltonian`) for efficient
quantum chemistry simulations within a defined electron-number subspace.
It also includes a high-level utility `generate_ucc_ansatz` to streamline
the setup of VQE calculations.
"""

from .mqchem_simulator import MQChemSimulator
from .ci_hamiltonian import CIHamiltonian
from .ucc_excitation_gate import UCCExcitationGate
from .vqe_preparation import prepare_uccsd_vqe

__all__ = [
    'MQChemSimulator',
    'CIHamiltonian',
    'UCCExcitationGate',
    'prepare_uccsd_vqe',
]
