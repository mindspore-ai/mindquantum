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
"""Test qcis."""
import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.io import QCIS

def test_qcis():
    """
    test
    Description:
    Expectation:
    """
    circ0 = Circuit()
    circ0.x(0).z(1,0).rx({"a":-2*np.sqrt(2)}, 0).sx(0).barrier()
    circ0.ry(ParameterResolver(data={'theta':-np.pi}, const=np.pi), 1)
    string = QCIS().to_string(circ0)
    circ1 = QCIS().from_string(string)
    assert circ1 == circ0
    assert circ1.x(0) != circ0

    circ2 = Circuit().rx({"a":-2*np.sqrt(2)}, 0)
    circ2.ry(ParameterResolver(data={'theta':-np.pi}, const=np.pi/2), 0).rz(0., 0)
    string = QCIS().to_string(circ2, parametric=False)
    circ3 = Circuit().from_qcis(string)
    circ4 = Circuit().ry(np.pi/2, 0)
    assert circ3 == circ4
