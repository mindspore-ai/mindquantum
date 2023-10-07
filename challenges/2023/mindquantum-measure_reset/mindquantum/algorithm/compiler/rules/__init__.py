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
"""Compiler rules."""
from .basic_decompose import BasicDecompose
from .basic_rule import (
    BasicCompilerRule,
    KroneckerSeqCompiler,
    SequentialCompiler,
    compile_circuit,
)
from .compiler_logger import CompileLog, LogIndentation
from .device_based import CZBasedChipCompiler
from .gate_replacer import CXToCZ, CZToCX, GateReplacer
from .neighbor_canceler import FullyNeighborCanceler, SimpleNeighborCanceler

__all__ = [
    'BasicCompilerRule',
    'KroneckerSeqCompiler',
    'SequentialCompiler',
    'BasicDecompose',
    'CZBasedChipCompiler',
    'CXToCZ',
    'CZToCX',
    'GateReplacer',
    'FullyNeighborCanceler',
    'SimpleNeighborCanceler',
    'compile_circuit',
]
