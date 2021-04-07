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
"""Circuit module."""

from collections.abc import Iterable
from typing import List
import numpy as np
from projectq.ops import QubitOperator as pq_operator
from openfermion.ops import QubitOperator as of_operator
from mindquantum.gate import BasicGate
from mindquantum.gate import I
from mindquantum.gate import X
from mindquantum.gate import Y
from mindquantum.gate import Z
from mindquantum.gate import Hamiltonian
from mindquantum.gate.basic import _check_gate_type
from mindquantum.utils import bprint
from mindquantum.parameterresolver import ParameterResolver as PR

GateSeq = List[BasicGate]


def _two_dim_array_to_list(data):
    """
    Convert a two dimension array to a list of string.
    """
    if len(data.shape) != 2:
        raise ValueError(
            "data need two dimensions, but get {} dimensions".format(
                len(data.shape)))
    out_real = []
    out_imag = []
    for i in data:
        out_real.append([])
        out_imag.append([])
        for j in i:
            out_real[-1].append(str(float(np.real(j))))
            out_imag[-1].append(str(float(np.imag(j))))
    return [out_real, out_imag]


class Circuit(list):
    """
    The quantum circuit module.

    A quantum circuit contains one or more quantum gates, and can be evaluated
    in a quantum simulator. You can build a quantum circuit very easy by add
    a quantum gate or another circuit.

    Args:
        gates (BasicGate, list[BasicGate]): You can
            initialize the quantum circuit by a single quantum gate or a
            list of gates.


    Examples:
        >>> circuit1 = Circuit()
        >>> circuit1 += RX('a').on(0)
        >>> circuit1 *= 2
        >>> print(circuit1)
        RX(a|0)
        RX(a|0)
        >>> circuit2 = Circuit([X.on(0,1)])
        >>> circuit3= circuit1 + circuit2
        >>> assert len(circuit3) == 3
        >>> circuit3.summary()
        =======Circuit Summary=======
        |Total number of gates  : 3.|
        |Parameter gates        : 2.|
        |with 1 parameters are  : a.|
        |Number qubit of circuit: 2 |
        =============================
    """
    def __init__(self, gates=None):
        list.__init__([])
        self.all_qubits = set()
        self.all_paras = set()
        self.n_qubits = -1
        if gates is not None:
            if isinstance(gates, Iterable):
                for gate in gates:
                    _check_gate_type(gate)
                self.extend(gates)
            else:
                _check_gate_type(gates)
                self.append(gates)

    def __add__(self, gates):
        return Circuit(super().__add__(Circuit(gates)))

    def __radd__(self, gates):
        return Circuit(gates) + self

    def __iadd__(self, gates):
        if isinstance(gates, BasicGate):
            self.append(gates)
        elif isinstance(gates, Circuit):
            self.extend(gates)
        else:
            raise TypeError(
                "Require a quantum gate or a quantum circuit, but get {}.".
                format(type(gates)))
        return self

    def __mul__(self, num):
        return Circuit(super().__mul__(num))

    def __rmul__(self, num):
        return self.__mul__(num)

    def __setitem__(self, k, v):
        _check_gate_type(v)
        super().__setitem__(k, v)

    def __getitem__(self, sliced):
        if isinstance(sliced, int):
            return super().__getitem__(sliced)
        return Circuit(super().__getitem__(sliced))

    def insert(self, index, gates):
        """
        Insert a quantum gate or quantum circuit in index.

        Args:
            index (int): Index to set gate.
            gates (Union[BasicGate, list[BasicGate]]): Gates you need to insert.
        """
        if isinstance(gates, BasicGate):
            super().insert(index, gates)
        elif isinstance(gates, Iterable):
            for gate in gates[::-1]:
                _check_gate_type(gate)
                self.insert(index, gate)
        else:
            raise TypeError("Unsupported type for quantum gate: {}".format(
                type(gates)))

    def no_grad(self):
        """
        Set all parameterized gate in this quantum circuit not require grad.
        """
        for gate in self:
            gate.no_grad()

    def requires_grad(self):
        """
        Set all parameterized gates in this quantum circuit require grad.
        """
        for gate in self:
            gate.requires_grad()

    def __str__(self):
        return '\n'.join(repr(i) for i in self)

    def __repr__(self):
        return self.__str__()

    def summary(self, show=True):
        """
        Print the information about current circuit, including block number,
        gate number, non-parameterized gate number, parameterized gate number
        and the total parameters.

        Examples:
            >>> circuit = Circuit([RX('a').on(1), H.on(1), RX('b').on(0)])
            >>> circuit.summary()
            ========Circuit Summary========
            |Total number of gates:3.     |
            |Blocks               :1.     |
            |Non parameter gates  :1.     |
            |Parameter gates      :2.     |
            |with 2 parameters are: b, a. |
            ===============================
        """
        self.num_non_para_gate = 0
        self.num_para_gate = 0
        self.all_paras = set()
        self.all_qubits = set()
        for gate in self:
            if gate.isparameter:
                self.num_para_gate += 1
                self.all_paras.update(gate.coeff)
            else:
                self.num_non_para_gate += 1
            self.all_qubits.update(gate.obj_qubits)
            self.all_qubits.update(gate.ctrl_qubits)
        self.n_qubits = max(self.all_qubits) + 1
        if show:
            info = bprint([
                'Total number of gates: {}.'.format(self.num_para_gate +
                                                    self.num_non_para_gate),
                'Parameter gates: {}.'.format(self.num_para_gate),
                'with {} parameters are: {}{}'.format(
                    len(self.all_paras), ', '.join(list(self.all_paras)[:10]),
                    ('.' if len(self.all_paras) <= 10 else '...')),
                'Number qubit of circuit: {}'.format(self.n_qubits)
            ],
                          title='Circuit Summary')
            for i in info:
                print(i)

    @property
    def hermitian(self):
        """
        Get the hermitian of this quantum circuit.

        Examples:
            >>> circ = Circuit(RX({'a': 0.2}).on(0))
            >>> herm_circ = circ.hermitian
            >>> herm_circ[0].coeff
            {'a': -0.2}
        """
        return Circuit([gate.hermitian() for gate in self[::-1]])

    def parameter_resolver(self):
        """
        Get the parameter resolver of the whole circuit.

        Note:
            This parameter resolver only tells you what are the parameters of
            this quantum circuit, and which part of parameters need grad, since
            the same parameter can be in different gate, and the coefficient
            can be different. The detail parameter resolver that shows the
            coefficient is in each gate of the circuit.

        Returns:
            ParameterResolver, the parameter resolver of the whole circuit.
        """
        pr = PR()
        for gate in self:
            if gate.isparameter:
                pr.update(gate.coeff)
        pr *= 0
        return pr

    @property
    def para_name(self):
        """
        Get the parameter name of this circuit.

        Returns:
            list[str], a list that contains the parameter name.

        Examples:
            >>> from mindquantum.gate import RX
            >>> from mindquantum.circuit import Circuit
            >>> circuit = Circuit(RX({'a': 1, 'b': 2}).on(0))
            >>> circuit.para_name
            ['a', 'b']
        """
        return self.parameter_resolver().para_name

    def apply_value(self, pr):
        """
        Convert this circuit to a non parameterized circuit with parameter you input.

        Args:
            pr (Union[dict, ParameterResolver]): parameters you want to apply into this circuit.

        Returns:
            Circuit, a non parameterized circuit.

        Examples:
            >>> from mindquantum.gate import X, RX
            >>> from mindquantum.circuit import Circuit
            >>> circuit = Circuit()
            >>> circuit += X.on(0)
            >>> circuit += RX({'a': 2}).on(0)
            >>> circuit = circuit.apply_value({'a': 1.5})
            >>> circuit
            X(0)
            RX(3.0|0)
        """
        circuit = Circuit()
        for gate in self:
            if not gate.isparameter:
                circuit += gate
            else:
                circuit += gate.__class__(gate.coeff.combination(pr)).on(
                    gate.obj_qubits, gate.ctrl_qubits)
        return circuit

    def mindspore_data(self):
        """
        Serialize the circuit. The result can be used by QNN operators.
        """
        m_data = {
            'gate_names': [],
            'gate_matrix': [],
            'gate_obj_qubits': [],
            'gate_ctrl_qubits': [],
            'gate_params_names': [],
            'gate_coeff': [],
            'gate_requires_grad': []
        }
        for gate in self:
            if gate.isparameter:
                m_data['gate_names'].append(gate.name)
                m_data['gate_matrix'].append([[["0.0", "0.0"], ["0.0", "0.0"]],
                                              [["0.0", "0.0"], ["0.0",
                                                                "0.0"]]])
                pr_data = gate.coeff.mindspore_data()
            else:
                m_data['gate_names'].append("npg")
                m_data['gate_matrix'].append(
                    _two_dim_array_to_list(gate.matrix()))
                pr_data = PR().mindspore_data()
            m_data['gate_params_names'].append(pr_data['gate_params_names'])
            m_data['gate_coeff'].append(pr_data['gate_coeff'])
            m_data['gate_requires_grad'].append(pr_data['gate_requires_grad'])
            m_data['gate_obj_qubits'].append(gate.obj_qubits)
            m_data['gate_ctrl_qubits'].append(gate.ctrl_qubits)
        return m_data


def pauli_word_to_circuits(qubitops):
    """
    Convert a single pauli word qubit operator to a quantum circuit.

    Args:
        qubitops (QubitOperator, Hamiltonian): The single pauli word qubit operator.

    Returns:
        Circuit, a quantum circuit.

    Examples:
        >>> from openfermion.ops import QubitOperator
        >>> qubitops = QubitOperator('X0 Y1')
        >>> pauli_word_to_circuits(qubitops)
        X(0)
        Y(1)
    """
    if not isinstance(qubitops, (pq_operator, of_operator, Hamiltonian)):
        raise TypeError(
            "Require a QubitOperator or a Hamiltonian, but get {}!".format(
                type(qubitops)))
    if isinstance(qubitops, Hamiltonian):
        qubitops = qubitops.hamiltonian
    if len(qubitops.terms) > 1:
        raise Exception("Onle work for QubitOperator with single pauliword!")
    gate_map = {'X': X, 'Y': Y, 'Z': Z}
    for ops in qubitops.terms.keys():
        circ = Circuit()
        if ops:
            for ind, single_op in ops:
                circ += gate_map[single_op].on(ind)
        else:
            circ += I.on(0)
    return circ
