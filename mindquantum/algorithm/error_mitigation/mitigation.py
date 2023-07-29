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
"""Error mitigation algorithm."""
import typing

import numpy as np

from mindquantum.core.circuit import Circuit

from .folding_circuit import fold_at_random


# pylint: disable=too-many-arguments,too-many-locals
def zne(
    circuit: Circuit,
    executor: typing.Callable[[Circuit], float],
    scaling: typing.List[float] = None,
    order=None,
    method="R",
    a=0,
    args=None,
) -> float:
    """
    Zero noise extrapolation.

    Args:
        circuit (:class:`~.core.circuit.Circuit`): A quantum circuit.
        executor (Callable[[:class:`~.core.circuit.Circuit`], float]): A callable method that can evaluate a
            quantum circuit and return some value.
        scaling (List[float]): The scaling factor to folding circuit. If ``None``, it will be ``[1.0, 2.0, 3.0]``.
            Default: ``None``.
        order (float): Order of extrapolation for polynomial. Default: ``None``.
        method (str): Extrapolation method, could be ``'R'`` (Richardson), ``'P'`` (polynomial) and
            ``'PE``' (poly exponential). Default: ``'R'``.
        a (float): Poly exponential extrapolation factor. Default: ``0``.
        args (Tuple): The other arguments for executor except first one.
    """
    y = []
    mitigated = 0
    if scaling is None:
        scaling = [1.0, 2.0, 3.0]
    for factor in scaling:
        expectation = executor(fold_at_random(circuit, factor), *args)
        y.append(expectation)
    if method == "R":
        for k, y_k in enumerate(y):
            product = 1
            for i in range(0, len(y)):
                if k != i:
                    try:
                        product = product * (scaling[i] / (scaling[i] - scaling[k]))
                    except ZeroDivisionError as exc:
                        raise ZeroDivisionError(f"Error scaling: {scaling}") from exc
            mitigated = mitigated + y_k * product
        return mitigated
    if order is None:
        raise ValueError("For polynomial and poly exponential, order cannot be None.")
    if method == "P":
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
        return mitigated
    if method == "PE":
        y = y - a
        y = np.log(y)
        z = np.polyfit(scaling, y, (order - 1))
        f = np.poly1d(z)
        mitigated = f(0)
        mitigated = a + np.exp(mitigated)
    else:
        print("Provide a valid extrapolation scheme R, PE, P")

    return mitigated
