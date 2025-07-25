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
"""Gate module that provides different quantum gate."""

from .basic import (
    HERMITIAN_PROPERTIES,
    BasicGate,
    NoiseGate,
    NoneParameterGate,
    ParameterGate,
    QuantumGate,
)
from .basicgate import (
    BARRIER,
    CNOT,
    ISWAP,
    RX,
    RY,
    RZ,
    SWAP,
    SX,
    U3,
    BarrierGate,
    CNOTGate,
    FSim,
    Givens,
    GlobalPhase,
    GroupedPauli,
    H,
    HGate,
    I,
    IGate,
    ISWAPGate,
    PhaseShift,
    Power,
    Rn,
    RotPauliString,
    Rxx,
    Rxy,
    Rxz,
    Ryy,
    Ryz,
    Rzz,
    S,
    SGate,
    SWAPalpha,
    SWAPGate,
    SXGate,
    T,
    TGate,
    UnivMathGate,
    X,
    XGate,
    Y,
    YGate,
    Z,
    ZGate,
    gene_univ_parameterized_gate,
    gene_univ_two_params_gate,
)
from .channel import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    BitPhaseFlipChannel,
    DepolarizingChannel,
    GroupedPauliChannel,
    KrausChannel,
    PauliChannel,
    PhaseDampingChannel,
    PhaseFlipChannel,
    ThermalRelaxationChannel,
)
from .measurement import Measure, MeasureResult

__all__ = [
    "BasicGate",
    "QuantumGate",
    "NoiseGate",
    "NoneParameterGate",
    "ParameterGate",
    "HERMITIAN_PROPERTIES",
    "BarrierGate",
    "CNOTGate",
    "HGate",
    "IGate",
    "XGate",
    "YGate",
    "ZGate",
    "gene_univ_parameterized_gate",
    "gene_univ_two_params_gate",
    "UnivMathGate",
    "SWAPGate",
    "ISWAPGate",
    "SWAPalpha",
    "Givens",
    "RX",
    "RY",
    "RZ",
    "PhaseShift",
    "SGate",
    "TGate",
    "Rxx",
    "Ryy",
    "Rzz",
    "Rxy",
    "Rxz",
    "Ryz",
    "Rn",
    "Power",
    "I",
    "X",
    "Y",
    "Z",
    "H",
    "S",
    "T",
    "SX",
    "SXGate",
    "SWAP",
    "ISWAP",
    "CNOT",
    "BARRIER",
    "Measure",
    "MeasureResult",
    "PauliChannel",
    "BitFlipChannel",
    "PhaseFlipChannel",
    "BitPhaseFlipChannel",
    "DepolarizingChannel",
    "GlobalPhase",
    "AmplitudeDampingChannel",
    "PhaseDampingChannel",
    "KrausChannel",
    "GroupedPauliChannel",
    "ThermalRelaxationChannel",
    "U3",
    "FSim",
    "GroupedPauli",
    "RotPauliString",
]

__all__.sort()
