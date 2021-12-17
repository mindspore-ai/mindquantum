# -*- coding: utf-8 -*-
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
from rich.console import Console
import mindquantum.core.gates as G
from mindquantum.core.gates.basic import _check_gate_type
from mindquantum.core.parameterresolver import ParameterResolver as PR
from mindquantum.core.parameterresolver.parameterresolver import ParameterResolver
from mindquantum.io import bprint
from mindquantum.io.display import brick_model
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.utils.type_value_check import _check_and_generate_pr_type
from .utils import apply

GateSeq = List[G.BasicGate]


def _two_dim_array_to_list(data):
    """
    Convert a two dimension array to a list of string.
    """
    if len(data.shape) != 2:
        raise ValueError("data need two dimensions, but get {} dimensions".format(len(data.shape)))
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

    def collect_only_one(self, keys, raise_msg):
        """collect item only single time, otherwise raise error"""
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            if k in self.map:
                raise ValueError(raise_msg)
            self.map[k] = 1

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

    def merge_only_one(self, other, raise_msg):
        """merge with other collection container"""
        for k, _ in other.map.items():
            if k in self.map:
                raise ValueError(raise_msg)
            self.map[k] = 1

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
        >>> from mindquantum import Circuit, RX, X
        >>> circuit1 = Circuit()
        >>> circuit1 += RX('a').on(0)
        >>> circuit1 *= 2
        >>> circuit1
        q0: ──RX(a)────RX(a)──
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
        >>> circuit3
        q0: ──RX(a)────RX(a)────X──
                                │
        q1: ────────────────────●──
    """
    def __init__(self, gates=None):
        list.__init__([])
        self.all_qubits = CollectionMap()
        self.all_paras = CollectionMap()
        self.all_measures = CollectionMap()
        if gates is not None:
            if isinstance(gates, Iterable):
                self.extend(gates)
            else:
                self.append(gates)
        self.has_cpp_obj = False

    def append(self, gate):
        """
        Append a gate.

        Args:
            gate (BasicGate): The gate you want to append.
        """
        _check_gate_type(gate)
        if isinstance(gate, G.Measure):
            self.all_measures.collect_only_one(gate, f'measure key {gate.key} already exist.')
        self.all_qubits.collect(gate.obj_qubits)
        self.all_qubits.collect(gate.ctrl_qubits)
        if gate.parameterized:
            self.all_paras.collect(list(gate.coeff.keys()))
        super().append(gate)
        self.has_cpp_obj = False

    def extend(self, gates):
        """
        Extend a circuit.

        Args:
            gates (Union[Circuit, list[BasicGate]]), A `Circuit` or a list of `BasicGate` you want to extend.
        """
        if isinstance(gates, Circuit):
            self.all_measures.merge_only_one(gates.all_measures, "Measure already exist.")
            self.all_qubits.merge(gates.all_qubits)
            self.all_paras.merge(gates.all_paras)
            super().extend(gates)
        else:
            for gate in gates:
                self.append(gate)
        self.has_cpp_obj = False

    def __add__(self, gates):
        out = Circuit()
        out.extend(self)
        if isinstance(gates, G.BasicGate):
            out.append(gates)
        else:
            out.extend(gates)
        return out

    def __radd__(self, gates):
        return Circuit(gates) + self

    def __iadd__(self, gates):
        if isinstance(gates, G.BasicGate):
            self.append(gates)
        elif isinstance(gates, Circuit):
            self.extend(gates)
        else:
            raise TypeError("Require a quantum gate or a quantum circuit, but get {}.".format(type(gates)))
        return self

    def __mul__(self, num):
        if not isinstance(num, int):
            raise TypeError(f'{type(num)} object cannot be interpreted as an integer')
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
        if old_v.parameterized:
            self.all_paras.delete(list(old_v.coeff.keys()))
        if isinstance(old_v, G.Measure):
            self.all_measures.delete(old_v)
        super().__setitem__(k, v)
        self.all_qubits.collect(v.obj_qubits)
        self.all_qubits.collect(v.ctrl_qubits)
        if v.parameterized:
            self.all_paras.collect(list(v.coeff.keys()))
        if isinstance(v, G.Measure):
            self.all_measures.collect_only_one(v, f'measure key {v.key} already exist.')
        self.has_cpp_obj = False

    def __getitem__(self, sliced):
        if isinstance(sliced, int):
            return super().__getitem__(sliced)
        return Circuit(super().__getitem__(sliced))

    @property
    def has_measure_gate(self):
        return self.all_measures.size != 0

    @property
    def parameterized(self):
        return self.all_paras.size != 0

    def insert(self, index, gates):
        """
        Insert a quantum gate or quantum circuit in index.

        Args:
            index (int): Index to set gate.
            gates (Union[BasicGate, list[BasicGate]]): Gates you need to insert.
        """
        if isinstance(gates, G.BasicGate):
            super().insert(index, gates)
            self.all_qubits.collect(gates.obj_qubits)
            self.all_qubits.collect(gates.ctrl_qubits)
            if gates.parameterized:
                self.all_paras.collect(list(gates.coeff.keys()))
            if isinstance(gates, G.Measure):
                self.all_measures.collect_only_one(gates, f'measure key {gates.key} already exist.')
        elif isinstance(gates, Iterable):
            for gate in gates[::-1]:
                _check_gate_type(gate)
                self.insert(index, gate)
                self.all_qubits.collect(gate.obj_qubits)
                self.all_qubits.collect(gate.ctrl_qubits)
                if gate.parameterized:
                    self.all_paras.collect(list(gate.coeff.keys()))
                if isinstance(gate, G.Measure):
                    self.all_measures.collect_only_one(gate, f'measure key {gate.key} already exist.')
        else:
            raise TypeError("Unsupported type for quantum gate: {}".format(type(gates)))
        self.has_cpp_obj = False

    def no_grad(self):
        """
        Set all parameterized gate in this quantum circuit not require grad.
        """
        for gate in self:
            gate.no_grad()
        self.has_cpp_obj = False
        return self

    def requires_grad(self):
        """
        Set all parameterized gates in this quantum circuit require grad.
        """
        for gate in self:
            gate.requires_grad()
        self.has_cpp_obj = False
        return self

    def compress(self):
        r"""
        Remove all unused qubits, and map qubits to `range(n_qubits)`.

        Examples:
            >>> from mindquantum import qft
            >>> qft([0, 2, 4])
            q0: ──H────PS(π/2)────PS(π/4)─────────────────────────@──
                          │          │                            │
            q2: ──────────●──────────┼───────H────PS(π/2)─────────┼──
                                     │               │            │
            q4: ─────────────────────●───────────────●───────H────@──
            >>> qft([0, 2, 4]).compress()
            q0: ──H────PS(π/2)────PS(π/4)─────────────────────────@──
                          │          │                            │
            q1: ──────────●──────────┼───────H────PS(π/2)─────────┼──
                                     │               │            │
            q2: ─────────────────────●───────────────●───────H────@──
        """
        circ = apply(self, list(range(len(self.all_qubits))))
        return circ

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        from mindquantum.io.display._config import _CIRCUIT_STYLE
        circ = self.compress()
        s = brick_model(circ, sorted(self.all_qubits.map))
        s = '\n'.join(s)
        console = Console(record=True)
        if not console.is_jupyter:
            console.width = len(s)
            with console.capture() as capture:
                console.print(s, style=_CIRCUIT_STYLE['style'], width=len(s))
            s = capture.get()
        return s

    def _repr_html_(self):
        """repr for jupyter nontebook"""
        from mindquantum.io.display._config import CIRCUIT_HTML_FORMAT
        from mindquantum.io.display._config import _CIRCUIT_STYLE
        console = Console(record=True)
        circ = self.compress()
        s = brick_model(circ, sorted(self.all_qubits.map))
        s = '\n'.join(s)
        console.width = len(s)
        with console.capture() as _:
            console.print(s, style=_CIRCUIT_STYLE['style'], width=len(s))
        s = console.export_html(code_format=CIRCUIT_HTML_FORMAT, inline_styles=True)
        return '\n'.join(s.split('\n')[1:])

    @property
    def n_qubits(self):
        if self.all_qubits:
            return max(self.all_qubits.keys()) + 1
        return 0

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
            if gate.parameterized:
                self.num_para_gate += 1
            else:
                self.num_non_para_gate += 1
        if show:
            info = bprint([
                'Total number of gates: {}.'.format(self.num_para_gate + self.num_non_para_gate),
                'Parameter gates: {}.'.format(self.num_para_gate), 'with {} parameters are: {}{}'.format(
                    len(self.all_paras), ', '.join(self.all_paras.keys()[:10]),
                    ('.' if len(self.all_paras) <= 10 else '...')), 'Number qubit of circuit: {}'.format(self.n_qubits)
            ],
                          title='Circuit Summary')
            for i in info:
                print(i)

    def hermitian(self):
        """
        Get the hermitian of this quantum circuit.

        Examples:
            >>> circ = Circuit(RX({'a': 0.2}).on(0))
            >>> herm_circ = circ.hermitian()
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
    def params_name(self):
        """
        Get the parameter name of this circuit.

        Returns:
            list, a list that contains the parameter name.

        Examples:
            >>> from mindquantum.core.gates import RX
            >>> from mindquantum.core.circuit import Circuit
            >>> circuit = Circuit(RX({'a': 1, 'b': 2}).on(0))
            >>> circuit.params_name
            ['a', 'b']
        """
        return list(self.all_paras.keys())

    def matrix(self, pr=None, big_end=False, backend='projectq'):
        """
        Get the matrix of this circuit.

        Args:
            pr (ParameterResolver, dict, numpy.ndarray, list, numbers.Number): The parameter
                resolver for parameterized quantum circuit. Default: None.
            big_end (bool): The low index qubit is place in the end or not. Default: False.
            backend (str): The backend to do simulation. Default: 'projectq'.

        Examples:
            >>> from mindquantum.core import Circuit
            >>> circuit = Circuit().rx('a',0).h(0)
            >>> circuit.matrix({'a': 1.0})
            array([[ 0.62054458-0.33900505j,  0.62054458-0.33900505j],
                [ 0.62054458+0.33900505j, -0.62054458-0.33900505j]])

        Returns:
            numpy.ndarray, two dimensional complex matrix of this circuit.
        """
        _check_input_type('big_end', bool, big_end)
        if big_end:
            circ = apply(self, list(range(self.n_qubits))[::-1])
        else:
            circ = self
        if pr is None:
            pr = ParameterResolver()
        pr = _check_and_generate_pr_type(pr, self.params_name)
        if self.has_measure_gate:
            raise ValueError("This circuit cannot have measurement gate.")
        from mindquantum.simulator import Simulator
        sim = Simulator(backend, self.n_qubits)
        m = np.array(sim.sim.get_circuit_matrix(circ.get_cpp_obj(), pr.get_cpp_obj())).T
        return m

    def apply_value(self, pr):
        """
        Convert this circuit to a non parameterized circuit with parameter you input.

        Args:
            pr (Union[dict, ParameterResolver]): parameters you want to apply into this circuit.

        Returns:
            Circuit, a non parameterized circuit.

        Examples:
            >>> from mindquantum.core.gates import X, RX
            >>> from mindquantum.core.circuit import Circuit
            >>> circuit = Circuit()
            >>> circuit += X.on(0)
            >>> circuit += RX({'a': 2}).on(0)
            >>> circuit = circuit.apply_value({'a': 1.5})
            >>> circuit
            q0: ──X────RX(3)──
        """
        circuit = Circuit()
        for gate in self:
            if not gate.parameterized:
                circuit += gate
            else:
                if set(gate.coeff.params_name).issubset(pr):
                    coeff = gate.coeff.combination(pr)
                else:
                    coeff = 1 * gate.coeff
                circuit += gate.__class__(coeff).on(gate.obj_qubits, gate.ctrl_qubits)
        return circuit

    def remove_barrier(self):
        """Remove all barrier gates"""
        circ = Circuit()
        for g in self:
            if not isinstance(g, G.BarrierGate):
                circ += g
        return circ

    def remove_measure(self):
        """Remove all measure gate."""
        circ = Circuit()
        for g in self:
            if not isinstance(g, G.Measure):
                circ += g
        return circ

    def remove_measure_on_qubits(self, qubits):
        """
        Remove all measure gate on some certain qubits.

        Args:
            qubit (Union[int, list[int]]): The qubits you want to remove measure.

        Examples:
            >>> from mindquantum import UN, H, Measure
            >>> circ = UN(H, 3).x(0, 1).x(1, 2).measure_all()
            >>> circ += H.on(0)
            >>> circ += Measure('q0_1').on(0)
            >>> circ.remove_measure_on_qubits(0)
            q0: ──H────X────H───────────
                       │
            q1: ──H────●────X────M(q1)──
                            │
            q2: ──H─────────●────M(q2)──
        """
        if not isinstance(qubits, list):
            qubits = [qubits]
        circ = Circuit()
        for gate in self:
            if isinstance(gate, G.Measure) and gate.obj_qubits[0] in qubits:
                continue
            circ += gate
        return circ

    def get_cpp_obj(self, hermitian=False):
        """
        Get cpp obj of circuit.

        Args:
            hermitian (bool): Whether to get cpp object of this circuit in hermitian version. Default: False.
        """
        if not self.has_cpp_obj:
            self.has_cpp_obj = True
            self.cpp_obj = [i.get_cpp_obj() for i in self if not isinstance(i, G.BarrierGate)]
            self.herm_cpp_obj = [i.get_cpp_obj() for i in self.hermitian() if not isinstance(i, G.BarrierGate)]

        if hasattr(self, 'cpp_obj') and hasattr(self, 'herm_cpp_obj'):
            if hermitian:
                return self.herm_cpp_obj
            return self.cpp_obj
        raise ValueError("Circuit does not generate cpp obj yet.")

    def h(self, obj_qubits, ctrl_qubits=None):
        """
        Add a hadamard gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `H` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `H` gate. Default: None.
        """
        self.append(G.H.on(obj_qubits, ctrl_qubits))
        return self

    def x(self, obj_qubits, ctrl_qubits=None):
        """
        Add a X gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `X` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `X` gate. Default: None.
        """
        self.append(G.X.on(obj_qubits, ctrl_qubits))
        return self

    def y(self, obj_qubits, ctrl_qubits=None):
        """
        Add a Y gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `Y` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Y` gate. Default: None.
        """
        self.append(G.Y.on(obj_qubits, ctrl_qubits))
        return self

    def z(self, obj_qubits, ctrl_qubits=None):
        """
        Add a Z gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `Z` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Z` gate. Default: None.
        """
        self.append(G.Z.on(obj_qubits, ctrl_qubits))
        return self

    def s(self, obj_qubits, ctrl_qubits=None):
        """
        Add a S gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `S` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `S` gate. Default: None.
        """
        self.append(G.S.on(obj_qubits, ctrl_qubits))
        return self

    def swap(self, obj_qubits, ctrl_qubits=None):
        """
        Add a SWAP gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `SWAP` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `SWAP` gate. Default: None.
        """
        self.append(G.SWAP.on(obj_qubits, ctrl_qubits))
        return self

    def rx(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a RX gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `RX` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `RX` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `RX` gate. Default: None.
        """
        self.append(G.RX(para).on(obj_qubits, ctrl_qubits))
        return self

    def ry(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a RY gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `RY` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `RY` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `RY` gate. Default: None.
        """
        self.append(G.RY(para).on(obj_qubits, ctrl_qubits))
        return self

    def rz(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a RZ gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `RZ` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `RZ` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `RZ` gate. Default: None.
        """
        self.append(G.RZ(para).on(obj_qubits, ctrl_qubits))
        return self

    def phase_shift(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Phase Shift gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `PhaseShift` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `PhaseShift` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `PhaseShift` gate. Default: None."""
        self.append(G.PhaseShift(para).on(obj_qubits, ctrl_qubits))
        return self

    def xx(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a XX gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `XX` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `XX` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `XX` gate. Default: None.
        """
        self.append(G.XX(para).on(obj_qubits, ctrl_qubits))
        return self

    def yy(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a YY gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `YY` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `YY` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `YY` gate. Default: None.
        """
        self.append(G.YY(para).on(obj_qubits, ctrl_qubits))
        return self

    def zz(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a ZZ gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `ZZ` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `ZZ` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `ZZ` gate. Default: None.
        """
        self.append(G.ZZ(para).on(obj_qubits, ctrl_qubits))
        return self

    def measure(self, key, obj_qubit=None):
        """
        Add a measure gate.

        Args:
            key (Union[int, str]): If `obj_qubit` is None, then `key` should be a int and means which qubit to measure,
                otherwise, `key` should be a str and means the name of this measure gate.
            obj_qubit (int): Which qubit to measure. Default: None.
        """
        if obj_qubit is None:
            self.append(G.Measure().on(key))
        else:
            self.append(G.Measure(key).on(obj_qubit))
        return self

    def measure_all(self, subfix=None):
        """
        Measure all qubits

        Args:
            subfix (str): The subfix string you want to add to the name of measure gate.
        """
        for i in range(self.n_qubits):
            s = f"q{i}" if subfix is None else f"q{i}_{subfix}"
            self += G.Measure(s).on(i)
        return self

    def barrier(self, show=True):
        """
        Add a barrier.

        Args:
            show (bool): Whether show barrier or not. Default: True.
        """
        _check_input_type('show', bool, show)
        self.append(G.BarrierGate(show))
        return self

    def un(self, gate, maps_obj, maps_ctrl=None):
        """
        Map a quantum gate to different objective qubits and control qubits.
        Please refers to UN.

        Args:
            gate (BasicBage): The BasicGate you want to map.
            map_obj (Union[int, list[int]]): object qubits.
            maps_ctrl (Union[int, list[int]]): control qubits. Default: None.
        """
        from mindquantum import UN
        self += UN(gate, maps_obj, maps_ctrl)
        return self

    def get_qs(self, backend='projectq', pr=None, ket=False, seed=42):
        """
        Get the final quantum state of this circuit.

        Args:
            backend (str): Which backend you want to use. Default: 'projectq'.
            pr (Union[numbers.Number, ParameterResolver, dict, numpy.ndarray]): The parameter of this circuit,
                if this circuit is parameterized. Default: None.
            ket (str): Whether to return the quantum state in ket format. Default: False.
            seed (int): The random seed of simulator.
        """
        from mindquantum import Simulator
        sim = Simulator(backend, self.n_qubits, seed)
        sim.apply_circuit(self, pr)
        return sim.get_qs(ket)


__all__ = ['Circuit']
