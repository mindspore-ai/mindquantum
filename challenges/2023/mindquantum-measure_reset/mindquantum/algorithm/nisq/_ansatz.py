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
"""Base class of ansatz."""

import inspect
import typing
from abc import abstractmethod

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import NoneParameterGate, ParameterGate
from mindquantum.core.parameterresolver import PRGenerator


class Ansatz:  # pylint: disable=too-few-public-methods
    """
    Base class for Ansatz.

    Args:
        name (str): The name of this ansatz.
        n_qubits (int): How many qubits this ansatz act on.
    """

    def __init__(self, name, n_qubits, *args, **kwargs):
        """Initialize an Ansatz object."""
        self.name = name
        self.n_qubits = n_qubits
        self._circuit = Circuit()
        self._implement(*args, **kwargs)

    @abstractmethod
    def _implement(self, *args, **kwargs):
        """Implement of ansatz."""

    @property
    def circuit(self) -> Circuit:
        """
        Get the quantum circuit of this ansatz.

        Returns:
            Circuit, the quantum circuit of this ansatz.
        """
        return self._circuit


def single_qubit_gate_layer(
    gate: typing.Union[ParameterGate, NoneParameterGate], n_qubits: int, stop: int = None, pr_gen: PRGenerator = None
):
    """
    Generate a single qubit gate layer.

    Args:
        gate (Union[:class:`~.core.gates.ParameterGate`, :class:`~.core.gates.NoneParameterGate`]): A
            quantum gate, can be a class or a instance.
        n_qubits (int): Number qubits of ansatz. If `stop` is not ``None``, then `n_qubits` would be the start qubit.
        stop (int): The stop qubit. If ``None``, `n_qubits` would be the stop qubit. Default: ``None``.
        pr_gen (:class:`~.core.parameterresolver.PRGenerator`): Object that generate parameters. If given `gate` is
            a sub class of ParameterGate, then `pr_gen` cannot be ``None``. Default: ``None``.
    """
    circ = Circuit()
    q_range = range(n_qubits)
    if stop is not None:
        q_range = range(n_qubits, stop)
    for i in q_range:
        if inspect.isclass(gate):
            if issubclass(gate, ParameterGate):
                circ += gate(pr_gen.new()).on(i)
            elif issubclass(gate, NoneParameterGate):
                circ += gate().on(i)
        else:
            circ += gate.on(i)
    return circ
