# -*- coding: utf-8 -*-
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
"""Test TimeEvolution."""

from mindquantum import Circuit
from mindquantum import gates as G
from mindquantum.core.operators import QubitOperator, TimeEvolution


def test_time_evolution():
    """
    Description: Test TimeEvolution
    Expectation: AssertionError
    """
    h = QubitOperator('Z0 Z1', 'p') + QubitOperator('X0', 'q')
    circ = TimeEvolution(h).circuit
    circ_exp = Circuit([G.X.on(1, 0), G.RZ({'p': 2}).on(1), G.X.on(1, 0), G.RX({'q': 4}).on(0)])
    assert circ.__repr__() == circ_exp.__repr__()
