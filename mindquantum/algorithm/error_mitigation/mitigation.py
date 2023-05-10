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
"""Error mitigation algorithm."""
import typing

import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator

from .folding_circuit import fold_at_random


# pylint: disable=too-many-arguments,too-many-locals
def zne(
    circuit: Circuit, observable: Hamiltonian, sim: Simulator, scaling: typing.List[float], order, method="R", a=0
) -> float:
    """
    Zero noise extrapolation.

    Args:
        circuit (Circuit): The noise quantum circuit.
        observable (Hamiltonian): The observable.
        sim (Simulator): The simulator you want to evaluate the circuit.
        scaling (List[float]): The scaling factor to folding circuit.
        order (float): Order of extrapolation for polynomial.
        method (str): Extrapolation method, could be ``'R'`` (Richardson), ``'P'`` (polynomial) and
            ``'PE``' (poly exponential). Default: ``'R'``.
        a (float): Poly exponential extrapolation factor. Default: ``0``.
    """
    y = []
    mitigated = 0
    for factor in scaling:
        expectation = sim.get_expectation(observable, fold_at_random(circuit, factor))
        y.append(expectation)

    if method == "R":
        for k, y_k in enumerate(y):
            product = 1
            for i in range(0, len(y)):
                if k != i:
                    product = product * (scaling[i] / (scaling[i] - scaling[k]))
            mitigated = mitigated + y_k * product
    elif method == "P":
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
    elif method == "PE":
        y = y - a
        y = np.log(y)
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
    else:
        print("Provide a valid extrapolation scheme R, PE, P")

    return mitigated
