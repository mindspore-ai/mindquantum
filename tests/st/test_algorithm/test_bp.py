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
"""Test barren plateau."""
import numpy as np
import pytest

from mindquantum.algorithm.nisq import ansatz_variance
from mindquantum.algorithm.nisq.chem import HardwareEfficientAnsatz
from mindquantum.core.gates import RY, RZ, Z
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.simulator.available_simulator import SUPPORTED_SIMULATOR


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize("config", list(filter(lambda x: x != 'stabilizer', SUPPORTED_SIMULATOR)))
def test_barren_plateau(config):
    """
    Description: Test barren plateau
    Expectation: success
    """
    simulator, dtype = config
    if simulator != 'mqvector':
        return
    np.random.seed(42)
    ham = Hamiltonian(QubitOperator('Z0 Z1'), dtype=dtype)
    q_list = [4, 6, 8, 10]
    varances = []
    for qubit in q_list:
        circ = HardwareEfficientAnsatz(qubit, [RY, RZ], Z, depth=50).circuit
        varances.append(ansatz_variance(circ, ham, circ.params_name[0], sim=Simulator(simulator, qubit, dtype=dtype)))
    vars_expect = [
        0.03366677155540075,
        0.007958129595835611,
        0.0014247908876269244,
        0.0006696567877430079,
    ]
    assert np.allclose(varances, vars_expect)
