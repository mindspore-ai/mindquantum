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
import copy
import numpy as np
from projectq.ops import QubitOperator as pq_operator
from openfermion.ops import QubitOperator as of_operator
from mindquantum.ops import QubitOperator as hiq_operator
from mindquantum.gate import BasicGate
from mindquantum.gate import I
from mindquantum.gate import X
from mindquantum.gate import Y
from mindquantum.gate import Z
from mindquantum.gate import Hamiltonian
import mindquantum.gate as G
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


class CollectionMap:
    """A collection container."""
    def __init__(self):
        self.map = {}

    def __str__(self):
        return self.map.__str__()

    def __repr__(self):
        return self.map.__repr__()

    def collect(self, keys):
        """collect items"""
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            if k not in self.map:
                self.map[k] = 1
            else:
                self.map[k] += 1

    def delete(self, keys):
        """delete items"""
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            if k in self.map:
                if self.map[k] == 1:
                    self.map.pop(k)
                else:
                    self.map[k] -= 1

    def num(self, k):
        """items count number"""
        if k not in self.map:
            return 0
        return self.map[k]

    def keys(self):
        """All items list"""
        return list(self.map.keys())

    @property
    def size(self):
        """number of items"""
        return len(self.map)

    def __len__(self):
        return self.size

    def merge(self, other):
        """merge with other collection container"""
        for k, v in other.map.items():
            if k in self.map:
                self.map[k] += v
            else:
                self.map[k] = v

    def unmerge(self, other):
        """delete with other collection container"""
        for k, v in other.map.items():
            if k in self.map:
                if self.map[k] <= v:
                    self.map.pop(k)
                else:
                    self.map[k] -= v

    def __copy__(self):
        """copy this container"""
        out = CollectionMap()
        out.merge(self)
        return out

    def __deepcopy__(self, memo):
        """deepcopy this container"""
        out = CollectionMap()
        out.merge(self)
        return out


class Circuit(list):
    """
    The quantum circuit module.

    A quantum circuit contains one or more quantum gates, and can be evaluated
    in a quantum simulator. You can build a quantum circuit very easy by add
    a quantum gate or another circuit.

    Args:
        gates (BasicGate, list[BasicGate]): You can
            initialize the quantum circuit by a single quantum gate or a
            list of gates. gates: None.


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
        self.all_qubits = CollectionMap()
        self.all_paras = CollectionMap()
        if gates is not None:
            if isinstance(gates, Iterable):
                self.extend(gates)
            else:
                self.append(gates)

    def append(self, gate):
        """Append a gate."""
        _check_gate_type(gate)
        super().append(gate)
        self.all_qubits.collect(gate.obj_qubits)
        self.all_qubits.collect(gate.ctrl_qubits)
        if gate.isparameter:
            self.all_paras.collect(list(gate.coeff.keys()))

    def extend(self, gates):
        """Extend a circuit."""
        if isinstance(gates, Circuit):
            super().extend(gates)
            self.all_qubits.merge(gates.all_qubits)
            self.all_paras.merge(gates.all_paras)
        else:
            for gate in gates:
                self.append(gate)

    def __add__(self, gates):
        out = Circuit()
        out.extend(self)
        if isinstance(gates, BasicGate):
            out.append(gates)
        else:
            out.extend(gates)
        return out

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
        if not isinstance(num, int):
            raise TypeError(
                f'{type(num)} object cannot be interpreted as an integer')
        out = Circuit()
        for _ in range(num):
            out += copy.deepcopy(self)
        return out

    def __deepcopy__(self, memo):
        res = Circuit()
        for gate in self:
            res.append(copy.deepcopy(gate))
        return res

    def __copy__(self):
        res = Circuit()
        for gate in self:
            res.append(copy.deepcopy(gate))
        return res

    def __rmul__(self, num):
        return self.__mul__(num)

    def __setitem__(self, k, v):
        _check_gate_type(v)
        old_v = self[k]
        self.all_qubits.delete(old_v.obj_qubits)
        self.all_qubits.delete(old_v.ctrl_qubits)
        if old_v.isparameter:
            self.all_paras.delete(list(old_v.coeff.keys()))
        super().__setitem__(k, v)
        self.all_qubits.collect(v.obj_qubits)
        self.all_qubits.collect(v.ctrl_qubits)
        if v.isparameter:
            self.all_paras.collect(list(v.coeff.keys()))

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
            self.all_qubits.collect(gates.obj_qubits)
            self.all_qubits.collect(gates.ctrl_qubits)
            if gates.isparameter:
                self.all_paras.collect(list(gates.coeff.keys()))
        elif isinstance(gates, Iterable):
            for gate in gates[::-1]:
                _check_gate_type(gate)
                self.insert(index, gate)
                self.all_qubits.collect(gate.obj_qubits)
                self.all_qubits.collect(gate.ctrl_qubits)
                if gate.isparameter:
                    self.all_paras.collect(list(gate.coeff.keys()))
        else:
            raise TypeError("Unsupported type for quantum gate: {}".format(
                type(gates)))

    def no_grad(self):
        """
        Set all parameterized gate in this quantum circuit not require grad.
        """
        for gate in self:
            gate.no_grad()
        return self

    def requires_grad(self):
        """
        Set all parameterized gates in this quantum circuit require grad.
        """
        for gate in self:
            gate.requires_grad()
        return self

    def __str__(self):
        return '\n'.join(repr(i) for i in self)

    def __repr__(self):
        return self.__str__()

    @property
    def n_qubits(self):
        return max(self.all_qubits.keys()) + 1

    def summary(self, show=True):
        """
        Print the information about current circuit, including block number,
        gate number, non-parameterized gate number, parameterized gate number
        and the total parameters.

        Args:
            show (bool): whether to show the information. Default: True.

        Examples:
            >>> from mindquantum import Circuit, RX, H
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
        for gate in self:
            if gate.isparameter:
                self.num_para_gate += 1
            else:
                self.num_non_para_gate += 1
        if show:
            info = bprint([
                'Total number of gates: {}.'.format(self.num_para_gate +
                                                    self.num_non_para_gate),
                'Parameter gates: {}.'.format(self.num_para_gate),
                'with {} parameters are: {}{}'.format(
                    len(self.all_paras), ', '.join(self.all_paras.keys()[:10]),
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
        pr = PR(self.all_paras.map)
        pr *= 0
        return pr

    @property
    def para_name(self):
        """
        Get the parameter name of this circuit.

        Returns:
            list, a list that contains the parameter name.

        Examples:
            >>> from mindquantum.gate import RX
            >>> from mindquantum.circuit import Circuit
            >>> circuit = Circuit(RX({'a': 1, 'b': 2}).on(0))
            >>> circuit.para_name
            ['a', 'b']
        """
        return list(self.all_paras.keys())

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
                if set(gate.coeff.para_name).issubset(pr):
                    coeff = gate.coeff.combination(pr)
                else:
                    coeff = 1 * gate.coeff
                circuit += gate.__class__(coeff).on(gate.obj_qubits,
                                                    gate.ctrl_qubits)
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

    def h(self, obj_qubits, ctrl_qubits=None):
        """Add a hadamard gate."""
        self.append(G.H.on(obj_qubits, ctrl_qubits))
        return self

    def x(self, obj_qubits, ctrl_qubits=None):
        """Add a X gate."""
        self.append(G.X.on(obj_qubits, ctrl_qubits))
        return self

    def y(self, obj_qubits, ctrl_qubits=None):
        """Add a Y gate."""
        self.append(G.Y.on(obj_qubits, ctrl_qubits))
        return self

    def z(self, obj_qubits, ctrl_qubits=None):
        """Add a Z gate."""
        self.append(G.Z.on(obj_qubits, ctrl_qubits))
        return self

    def s(self, obj_qubits, ctrl_qubits=None):
        """Add a S gate."""
        self.append(G.S.on(obj_qubits, ctrl_qubits))
        return self

    def swap(self, obj_qubits, ctrl_qubits=None):
        """Add a SWAP gate."""
        self.append(G.SWAP.on(obj_qubits, ctrl_qubits))
        return self

    def rx(self, para, obj_qubits, ctrl_qubits=None):
        """Add a RX gate."""
        self.append(G.RX(para).on(obj_qubits, ctrl_qubits))
        return self

    def ry(self, para, obj_qubits, ctrl_qubits=None):
        """Add a RY gate."""
        self.append(G.RY(para).on(obj_qubits, ctrl_qubits))
        return self

    def rz(self, para, obj_qubits, ctrl_qubits=None):
        """Add a RZ gate."""
        self.append(G.RZ(para).on(obj_qubits, ctrl_qubits))
        return self

    def phase_shift(self, para, obj_qubits, ctrl_qubits=None):
        """Add a Phase Shift gate."""
        self.append(G.PhaseShift(para).on(obj_qubits, ctrl_qubits))
        return self

    def xx(self, para, obj_qubits, ctrl_qubits=None):
        """Add a XX gate."""
        self.append(G.XX(para).on(obj_qubits, ctrl_qubits))
        return self

    def yy(self, para, obj_qubits, ctrl_qubits=None):
        """Add a YY gate."""
        self.append(G.YY(para).on(obj_qubits, ctrl_qubits))
        return self

    def zz(self, para, obj_qubits, ctrl_qubits=None):
        """Add a ZZ gate."""
        self.append(G.ZZ(para).on(obj_qubits, ctrl_qubits))
        return self


def pauli_word_to_circuits(qubitops):
    """
    Convert a single pauli word qubit operator to a quantum circuit.

    Args:
        qubitops (QubitOperator, Hamiltonian): The single pauli word qubit operator.

    Returns:
        Circuit, a quantum circuit.

    Examples:
        >>> from mindquantum.ops import QubitOperator
        >>> qubitops = QubitOperator('X0 Y1')
        >>> pauli_word_to_circuits(qubitops)
        X(0)
        Y(1)
    """
    if not isinstance(qubitops,
                      (pq_operator, of_operator, hiq_operator, Hamiltonian)):
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
