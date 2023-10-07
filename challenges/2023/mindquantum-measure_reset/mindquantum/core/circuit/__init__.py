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

from . import channel_adder
from .channel_adder import (
    BitFlipAdder,
    ChannelAdderBase,
    DepolarizingChannelAdder,
    GateSelector,
    MeasureAccepter,
    MixerAdder,
    NoiseChannelAdder,
    NoiseExcluder,
    QubitIDConstrain,
    QubitNumberConstrain,
    ReverseAdder,
    SequentialAdder,
)
from .circuit import A, Circuit, apply
from .module_circuit import UN, SwapParts
from .qfi import partial_psi_partial_psi, partial_psi_psi, qfi
from .utils import (
    AP,
    CPN,
    C,
    D,
    add_prefix,
    add_suffix,
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
    'add_suffix',
    'change_param_name',
    'shift',
    'as_ansatz',
    'qfi',
    'partial_psi_psi',
    'partial_psi_partial_psi',
    'as_encoder',
]
__all__.extend(channel_adder.__all__)
__all__.sort()
