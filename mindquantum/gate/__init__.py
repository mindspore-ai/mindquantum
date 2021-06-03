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
Gate.

Gate provides different quantum gate.
"""

from .basic import BasicGate
from .basic import IntrinsicOneParaGate
from .basic import NoneParameterGate
from .basic import ParameterGate
from .basicgate import IGate
from .basicgate import XGate
from .basicgate import YGate
from .basicgate import ZGate
from .basicgate import HGate
from .basicgate import SWAPGate
from .basicgate import CNOTGate
from .basicgate import H
from .basicgate import CNOT
from .basicgate import X
from .basicgate import Y
from .basicgate import Z
from .basicgate import I
from .basicgate import S
from .basicgate import Power
from .basicgate import SWAP
from .basicgate import UnivMathGate
from .basicgate import RX
from .basicgate import RY
from .basicgate import RZ
from .basicgate import PhaseShift
from .basicgate import XX
from .basicgate import YY
from .basicgate import ZZ
from .hamiltonian import Hamiltonian
from .projector import Projector


__all__ = [
    'BasicGate', 'IntrinsicOneParaGate', 'NoneParameterGate', 'ParameterGate',
    'H', 'CNOT', 'X', 'Y', 'Z', 'I', 'S', 'Power', 'SWAP', 'UnivMathGate',
    'RX', 'RY', 'RZ', 'PhaseShift', 'XX', 'YY', 'ZZ', 'IGate', 'XGate',
    'YGate', 'ZGate', 'HGate', 'SWAPGate', 'CNOTGate', 'Hamiltonian',
    'Projector'
]
