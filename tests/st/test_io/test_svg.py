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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test svg."""

from mindquantum.algorithm.library import qft
from mindquantum.core.gates import RX, BarrierGate
from mindquantum.simulator import Simulator


def test_measure_svg():
    """
    Description: Test measure result svg.
    Expectation: success.
    """
    circ = qft(range(3)).measure_all()
    sim = Simulator('mqvector', 3)
    res = sim.sampling(circ, shots=100, seed=42)
    text = res.svg()._repr_svg_().split('bar')  # pylint: disable=protected-access
    text = "bar".join([text[0]] + ['"'.join(i.split('"')[1:]) for i in text[1:]])
    len_text_exp = 9258
    assert len(text) == len_text_exp


def test_circuit_svg():
    """
    Description: Test svg default style.
    Expectation: success.
    """
    # pylint: disable=protected-access
    text = (qft(range(3)) + RX({'a': 1.2}).on(1) + BarrierGate()).measure_all().svg()._repr_svg_().strip()
    assert len(text) in (7079, 7078, 7130, 7033, 6941, 6940)
