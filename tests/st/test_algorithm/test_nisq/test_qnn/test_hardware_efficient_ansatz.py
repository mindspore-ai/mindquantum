# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test hardware efficient ansatz."""

import numpy as np

from mindquantum.algorithm import nisq
from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import PRGenerator


def test_ry_linear():
    """
    Description: test RYLinear.
    Expectation: success
    """
    circ1 = nisq.RYLinear(3, 2, prefix='a').circuit
    circ2 = Circuit()
    circ2.ry('a_p0', 0).ry('a_p1', 1).ry('a_p2', 2)
    circ2.x(1, 0).x(2, 1).barrier(False)
    circ2.ry('a_p3', 0).ry('a_p4', 1).ry('a_p5', 2)
    circ2.x(1, 0).x(2, 1).barrier(False)
    circ2.ry('a_p6', 0).ry('a_p7', 1).ry('a_p8', 2)
    assert circ1 == circ2


def test_ry_full():
    """
    Description: test RYLinear.
    Expectation: success
    """
    circ1 = nisq.RYFull(3, 2, prefix='a').circuit
    circ2 = Circuit()
    circ2.ry('a_p0', 0).ry('a_p1', 1).ry('a_p2', 2)
    circ2.x(1, 0).x(2, 0).x(2, 1).barrier(False)
    circ2.ry('a_p3', 0).ry('a_p4', 1).ry('a_p5', 2)
    circ2.x(1, 0).x(2, 0).x(2, 1).barrier(False)
    circ2.ry('a_p6', 0).ry('a_p7', 1).ry('a_p8', 2)
    assert circ1 == circ2


def test_ry_rz_full():
    """
    Description: test RYRZFull.
    Expectation: success
    """
    circ1 = nisq.RYRZFull(3, 1, prefix='a').circuit
    circ2 = Circuit()
    circ2.ry('a_p0', 0).ry('a_p1', 1).ry('a_p2', 2)
    circ2.rz('a_p3', 0).rz('a_p4', 1).rz('a_p5', 2)
    circ2.x(1, 0).x(2, 0).x(2, 1).barrier(False)
    circ2.ry('a_p6', 0).ry('a_p7', 1).ry('a_p8', 2)
    circ2.rz('a_p9', 0).rz('a_p10', 1).rz('a_p11', 2)
    assert circ1 == circ2


def test_ry_cascade():
    """
    Description: test RYCascade.
    Expectation: success
    """
    circ1 = nisq.RYCascade(3, 1, prefix='a').circuit
    circ2 = Circuit()
    circ2.ry('a_p0', 0).ry('a_p1', 1).ry('a_p2', 2)
    circ2.x(1, 0).x(2, 1).barrier(False)
    circ2.ry('a_p3', 0).ry('a_p4', 1).ry('a_p5', 2)
    circ2.x(2, 1).x(1, 0).barrier(False)
    circ2.ry('a_p6', 0).ry('a_p7', 1).ry('a_p8', 2)
    assert circ1 == circ2


def test_aswap():
    """
    Description: test ASWAP.
    Expectation: success
    """
    circ1 = nisq.ASWAP(3, 1, prefix='a').circuit
    circ2 = Circuit().x(0, 1)
    ansatz_1 = Circuit().rz('a_p0', 1).rz(np.pi, 1).ry('a_p1', 1).ry(np.pi / 2, 1)
    circ2 += ansatz_1
    circ2.x(1, 0)
    circ2 += ansatz_1.hermitian()
    circ2.x(0, 1).x(1, 2)
    ansatz_2 = Circuit().rz('a_p2', 2).rz(np.pi, 2).ry('a_p3', 2).ry(np.pi / 2, 2)
    circ2 += ansatz_2
    circ2.x(2, 1)
    circ2 += ansatz_2.hermitian()
    circ2.x(1, 2)
    assert circ1 == circ2


def test_pchea_xyz_1f():
    """
    Description: test PCHeaXYZ1F.
    Expectation: success
    """
    prg = PRGenerator()
    circ1 = nisq.PCHeaXYZ1F(3, 1).circuit
    circ2 = Circuit()
    ansatz_1 = Circuit().rx(prg.new(), 0).rx(prg.new(), 1).rx(prg.new(), 2)
    ansatz_1.ry(prg.new(), 0).ry(prg.new(), 1).ry(prg.new(), 2)
    parameter_6 = prg.new()
    parameter_7 = prg.new()
    ansatz_2 = Circuit().ry(parameter_6 * -0.5, 1).fsim(parameter_7, parameter_6, [0, 1]).ry(parameter_6 * 0.5, 1)
    parameter_8 = prg.new()
    parameter_9 = prg.new()
    ansatz_3 = Circuit().ry(parameter_8 * -0.5, 2).fsim(parameter_9, parameter_8, [1, 2]).ry(parameter_8 * 0.5, 2)
    circ2 += ansatz_1
    circ2.barrier(False)
    circ2 += ansatz_2
    circ2.barrier(False)
    circ2 += ansatz_3
    circ2.barrier(False)
    circ2.rz(prg.new(), 2)
    circ2.barrier(False)
    circ2 += ansatz_3.hermitian()
    circ2.barrier(False)
    circ2 += ansatz_2.hermitian()
    circ2.barrier(False)
    circ2 += ansatz_1.hermitian()
    assert circ1 == circ2


def test_pchea_xyz_2f():
    """
    Description: test PCHeaXYZ2F.
    Expectation: success
    """
    prg = PRGenerator()
    circ1 = nisq.PCHeaXYZ2F(3, 1).circuit
    circ2 = Circuit()
    ansatz_1 = Circuit().rx(prg.new(), 0).rx(prg.new(), 1).rx(prg.new(), 2)
    ansatz_1.ry(prg.new(), 0).ry(prg.new(), 1).ry(prg.new(), 2)
    parameter_6 = prg.new()
    parameter_7 = prg.new()
    ansatz_2 = Circuit().ry(parameter_6 * -0.5, 1).fsim(parameter_7, parameter_6, [0, 1]).ry(parameter_6 * 0.5, 1)
    parameter_8 = prg.new()
    parameter_9 = prg.new()
    ansatz_3 = Circuit().ry(parameter_8 * -0.5, 2).fsim(parameter_9, parameter_8, [1, 2]).ry(parameter_8 * 0.5, 2)
    circ2 += ansatz_1
    circ2.barrier(False)
    circ2 += ansatz_2
    circ2.barrier(False)
    circ2 += ansatz_3
    circ2.barrier(False)
    circ2.rz(prg.new(), 0).rz(prg.new(), 1).rz(prg.new(), 2)
    circ2.barrier(False)
    circ2 += ansatz_3.hermitian()
    circ2.barrier(False)
    circ2 += ansatz_2.hermitian()
    circ2.barrier(False)
    circ2 += ansatz_1.hermitian()
    assert circ1 == circ2
