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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test openqasm."""

from mindquantum.core import Circuit
from mindquantum.io import OpenQASM

import numpy as np


def test_openqasm():
    """
    test openqasm
    Description: test openqasm api
    Expectation: success
    """
    cir = Circuit().h(0).x(1).rz(0.1, 0, 1)
    test_openqasms = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\nx q[1];\ncrz(0.1) q[1],q[0];'
    openqasm = OpenQASM().to_string(cir)
    assert len(openqasm) == 82
    assert openqasm[63:] == 'crz(0.1) q[1],q[0];'
    assert openqasm == test_openqasms
    cir = Circuit().rz(0.1, 0, 1)
    test_openqasms = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncrz(0.1) q[1],q[0];'
    test_cir = OpenQASM().from_string(test_openqasms)
    assert np.allclose(test_cir.matrix(), cir.matrix())
