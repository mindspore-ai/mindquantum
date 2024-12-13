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
"""Test virtual distillation."""

import numpy as np

from mindquantum.algorithm.error_mitigation import virtual_distillation
from mindquantum.simulator import Simulator
from mindquantum.utils import random_circuit
from mindquantum import Hamiltonian, QubitOperator


def test_virtual_distillation():
    """
    Description: Test virtual distillation
    Expectation: success
    """

    circ = random_circuit(4, 100)

    def executor(test_circ):
        sim = Simulator("mqvector", 2 * circ.n_qubits)
        res_dict = sim.sampling(test_circ, shots=1000000).data
        return res_dict

    result = virtual_distillation(circ, executor)
    sim2 = Simulator("mqvector", 4)
    hams = [Hamiltonian(QubitOperator(f"Z{i}")) for i in range(4)]
    exp = []
    for i in range(4):
        exp.append(sim2.get_expectation(hams[i], circ))
    assert np.allclose(result, exp, atol=0.1)
