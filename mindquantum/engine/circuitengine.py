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
"""Simple engine to generate parameterized quantum circuit."""

from mindquantum.circuit import Circuit


class BasicQubit:
    """
    A quantum qubit.

    Args:
        qubit_id (int): The id of this quantum qubit.
        circuit (Circuit): The quantum circuit that this qubit belongs to. Default: None.
    """
    def __init__(self, qubit_id, circuit=None):
        if not isinstance(qubit_id, int):
            raise TypeError("qubit_id should be int, but get {}!".format(
                type(qubit_id)))
        self.qubit_id = qubit_id
        if circuit is None:
            self.circuit_ = Circuit()
        elif isinstance(circuit, Circuit):
            self.circuit_ = circuit
        else:
            raise TypeError(
                "circuit should be a quantum circuit, but get {}!".format(
                    type(circuit)))

    def __str__(self):
        return 'qubit_{}'.format(self.qubit_id)

    def __repr__(self):
        return self.__str__()

    @property
    def circuit(self):
        """
        Get the quantum circuit that this qubit belongs to.
        """
        return self.circuit_


class CircuitEngine:
    """
    A simple circuit engine that allows you to generate quantum circuit as
    projectq style.

    Note:
        For more usage, please refers to :class:`CircuitEngine.generator`
    """
    def __init__(self):
        self.current_id = -1
        self.qubits = []
        self.circuit_ = Circuit()

    def allocate_qubit(self):
        """
        Allocate a quantum qubit.
        """
        self.current_id += 1
        self.qubits.append(BasicQubit(self.current_id, self.circuit_))
        return [self.qubits[-1]]

    def allocate_qureg(self, n):
        """
        Allocate a quantum register.

        Args:
            n (int): Number of quantum qubits.
        """
        for _ in range(n):
            self.current_id += 1
            self.qubits.append(BasicQubit(self.current_id, self.circuit_))
        return self.qubits[-n:]

    @property
    def circuit(self):
        """
        Get the quantum circuit that construct by this engin.
        """
        return self.circuit_

    @staticmethod
    def generator(n_qubits, *args, **kwds):
        """Quantum circuit register.

        Args:
            n_qubits (int): qubit number of quantum circuit.

        Examples:
            >>> import mindquantum.gate as G
            >>> from mindquantum.engine import circuit_generator
            >>> @circuit_generator(2,prefix='p')
            >>> def ansatz(qubits, prefix):
            >>>     G.X | (qubits[0], qubits[1])
            >>>     G.RX(prefix+'_0') | qubits[1]
            >>> print(ansatz)
            X(1 <-: 0)
            RX(p_0|1)
            >>> print(type(ansatz))
            <class 'mindquantum.circuit.circuit.Circuit'>
        """
        def deco(fn):
            eng = CircuitEngine()
            qubits = eng.allocate_qureg(n_qubits)
            fn(qubits, *args, **kwds)
            return eng.circuit

        return deco


def circuit_generator(n_qubits, *args, **kwds):
    """
        Generate quantum circuit as projectq style.

        Args:
            n_qubits (int): qubit number of quantum circuit.

        Note:
            More information please refers to :class:`CircuitEngine.generator`.
    """
    return CircuitEngine().generator(n_qubits, *args, **kwds)
