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
"""Device based compiler rules."""

from .basic_decompose import BasicDecompose
from .basic_rule import SequentialCompiler
from .gate_replacer import CXToCZ
from .neighbor_canceler import FullyNeighborCanceler


class CZBasedChipCompiler(SequentialCompiler):
    """
    A compiler that suitable for chip that use cz gate.

    Args:
        log_level (int): log level to display message. For more explanation of log level,
            please refers to :class:`~.algorithm.compiler.BasicCompilerRule`. Default: ``0``.
    """

    def __init__(self, log_level=0):
        """Initialize a CZBasedChipCompiler."""
        super().__init__([BasicDecompose(), CXToCZ(), FullyNeighborCanceler()], "CZBasedChipCompiler", log_level)
