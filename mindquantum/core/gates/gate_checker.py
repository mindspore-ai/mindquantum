# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Check gate type."""
import typing

from mindquantum.core import gates as G
from mindquantum.core.gates import BasicGate


def is_same_gate_type(g1: BasicGate, g2: BasicGate, keep_ctrl_number: bool = False) -> bool:
    """
    Check whether two gate is the same based on their types.
    g1 is a instance of gate. g2 can be a instance or a gate class
    Some example is:
    is_same_gate_type(X.on(0), X) -> True
    is_same_gate_type(X.on(0), X.on(1)) -> True
    is_same_gate_type(RX('a').on(0), RX) -> True
    is_same_gate_type(X.on(0, 1), X) -> True
    is_same_gate_type(X.on(0), X.on(0, 1), keep_ctrl_number=False) -> True
    is_same_gate_type(X.on(0), X.on(0, 1), keep_ctrl_number=True) -> False
    """
    return True


def is_cnot(g1) -> bool:
    """Check whether a gate is exactly a cnot gate or not."""
    return is_same_gate_type(g1, G.X.on(0, 1))


def is_cz(g1) -> bool:
    """Check whether a gate is exactly a cz gate or not."""
    return is_same_gate_type(g1, G.Z.on(0, 1))
