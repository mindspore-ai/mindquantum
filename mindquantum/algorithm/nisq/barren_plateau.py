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
"""Barren plateau related module."""
import typing
import numpy as np
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
from mindquantum.core.operators import Hamiltonian


def ansatz_variance(ansatz: Circuit,
                    ham: Hamiltonian,
                    focus: str,
                    var_range: typing.Tuple[float, float],
                    other_var: np.array,
                    shots: int = 4000,
                    sim: typing.Union[Simulator, str] = 'mqvector'):
    """
    Calculate the variance of the gradient of certain parameters of parameterized quantum circuit.

    Args:
        ansatz (Circuit): The given parameterized quantum circuit.
        ham (Hamiltonian): The objective observable.
        focus (str): Which parameters you want to check.
        var_range (Tuple[float, float]): The random range for focusing parameter.
        other_var: (numpy.array): The value of other parameters.
        shots (int): Shots number when getting variance.
        sim (Union[Simulator, str]): Simulator you want to use.
    """
    params_name = ansatz.params_name
    if focus not in params_name:
        raise ValueError(f"Parameter {focus} is not in given ansatz.")
    if len(var_range) != 2:
        raise ValueError("var_range should have two float.")
    if len(other_var) != len(params_name) - 1:
        raise ValueError(f"other_var should have length of {len(params_name)-1}, but get {len(other_var)}")
    params_name.remove(focus)
    encoder = ansatz.apply_value(dict(zip(params_name, other_var))).as_encoder(False)
    if not isinstance(sim, Simulator):
        sim = Simulator(sim, encoder.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, encoder)
    p0 = np.random.uniform(var_range[0], var_range[1], size=(shots, 1))
    f, g = grad_ops(p0)
    return np.var(g)



from mindquantum import *

def ansatz(n_qubits, depth=1):
    ansatz = UN(RY(np.pi/4), n_qubits)
    entangle = Circuit()
    for i in range(0, n_qubits, 2):
        j = i+1
        if j<n_qubits:
            entangle += Z.on(i, j)
    for i in range(1, n_qubits, 2):
        j = i + 1
        if j < n_qubits:
            entangle += Z.on(i, j)
    for i in range(depth):
        for j in range(n_qubits):
            ansatz += np.random.choice([RX, RY, RZ])(f"p_{i}_{j}").on(j)
        ansatz += entangle
    return ansatz

ham = Hamiltonian(QubitOperator('Z0 Z1'))
vars = {}
for n_qubit in range(3 ,15):
    ans = ansatz(n_qubit, 2)
    vars[n_qubit] = ansatz_variance(ans, ham, 'p_0_0', (0, 2*np.pi), np.random.uniform(0, 2*np.pi,len(ans.params_name)-1), sim = 'mqvector_gpu')
    print(f"{n_qubit}: {vars[n_qubit]}")
