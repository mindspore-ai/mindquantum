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

# pylint: disable=too-many-lines,too-many-arguments
"""Circuit module."""

import copy
import os
from collections.abc import Iterable
from types import FunctionType, MethodType
from typing import List, Optional

import numpy as np
from rich.box import ROUNDED
from rich.console import Console
from rich.table import Table
from rich.text import Text

from mindquantum.utils.quantifiers import quantifier_selector
from mindquantum.utils.type_value_check import (
    _check_and_generate_pr_type,
    _check_gate_has_obj,
    _check_gate_type,
    _check_input_type,
)

from .. import gates as mq_gates
from ..gates.basic import BasicGate, ParameterGate
from ..parameterresolver import ParameterResolver


def _apply_circuit(circ, qubits):
    """Apply a circuit to other different qubits."""
    old_qubits_set = set()
    for gate in circ:
        old_qubits_set.update(gate.obj_qubits)
        old_qubits_set.update(gate.ctrl_qubits)
    old_qubits = list(old_qubits_set)
    old_qubits.sort()
    if len(old_qubits) != len(qubits):
        raise ValueError(f"Can not apply a {len(old_qubits)} qubits unit to {len(qubits)} qubits circuit.")
    qubits_map = dict(zip(old_qubits, qubits))
    out = Circuit()
    for gate in circ:
        gate = copy.deepcopy(gate)
        gate.obj_qubits = [qubits_map[i] for i in gate.obj_qubits]
        gate.ctrl_qubits = [qubits_map[i] for i in gate.ctrl_qubits]
        out += gate
    return out


def apply(circuit_fn, qubits):
    """
    Apply a quantum circuit or a quantum operator (a function that can generate a quantum circuit) to different qubits.

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit, or a function that can generate a
            quantum circuit.
        qubits (list[int]): The new qubits that you want to apply.

    Returns:
        Circuit or a function that can generate a Circuit.

    Raises:
        TypeError: If `qubits` is not a list.
        ValueError: If any element of `qubits` is negative.
        TypeError: If `circuit_fn` is not Circuit or can not return a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import apply
        >>> u1 = qft([0, 1])
        >>> u2 = apply(u1, [1, 0])
        >>> u3 = apply(qft, [1, 0])
        >>> u3 = u3([0, 1])
        >>> u2
                                ┏━━━┓
        q0: ─────────────■──────┨ H ┠─╳───
                         ┃      ┗━━━┛ ┃
              ┏━━━┓ ┏━━━━┻━━━━┓       ┃
        q1: ──┨ H ┠─┨ PS(π/2) ┠───────╳───
              ┗━━━┛ ┗━━━━━━━━━┛
        >>> u3
                                ┏━━━┓
        q0: ─────────────■──────┨ H ┠─╳───
                         ┃      ┗━━━┛ ┃
              ┏━━━┓ ┏━━━━┻━━━━┓       ┃
        q1: ──┨ H ┠─┨ PS(π/2) ┠───────╳───
              ┗━━━┛ ┗━━━━━━━━━┛
    """
    if not isinstance(qubits, list):
        raise TypeError(f"qubits need a list, but get {type(qubits)}!")
    if len(qubits) > 1:
        for index, qubit in enumerate(qubits[1:]):
            if qubit < 0 or qubits[index] < 0:
                raise ValueError("Qubit index can not negative!")
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return apply(circ, qubits)
            return _apply_circuit(circ, qubits)

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return _apply_circuit(circuit_fn, qubits)
    raise TypeError("circuit_fn need a circuit or a function that can generate a circuit.")


GateSeq = List[mq_gates.BasicGate]


def _two_dim_array_to_list(data):
    """Convert a two dimension array to a list of string."""
    if len(data.shape) != 2:
        raise ValueError(f"data need two dimensions, but get {len(data.shape)} dimensions")
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
        """Initialize a CollectionMap object."""
        self.map = {}

    def __str__(self):
        """Return a string representation of the object."""
        return self.map.__str__()

    def __repr__(self):
        """Return a string representation of the object."""
        return self.map.__repr__()

    def collect(self, keys):
        """Collect items."""
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            if k not in self.map:
                self.map[k] = 1
            else:
                self.map[k] += 1

    def collect_only_one(self, keys, raise_msg):
        """Collect item only single time, otherwise raise error."""
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            if k in self.map:
                raise ValueError(raise_msg)
            self.map[k] = 1

    def delete(self, keys):
        """Delete items."""
        if not isinstance(keys, list):
            keys = [keys]
        for k in keys:
            if k in self.map:
                if self.map[k] == 1:
                    self.map.pop(k)
                else:
                    self.map[k] -= 1

    def num(self, k):
        """Items count number."""
        if k not in self.map:
            return 0
        return self.map[k]

    def keys(self):
        """Return the list of all items list."""
        return list(self.map.keys())

    @property
    def size(self):
        """Return the number of items in the container."""
        return len(self.map)

    def __len__(self):
        """Size of the container."""
        return self.size

    def merge(self, other):
        """Merge with other collection container."""
        for k, v in other.map.items():
            if k in self.map:
                self.map[k] += v
            else:
                self.map[k] = v

    def merge_only_one(self, other, raise_msg):
        """Merge with other collection container."""
        for k, _ in other.map.items():
            if k in self.map:
                raise ValueError(raise_msg)
            self.map[k] = 1

    def unmerge(self, other):
        """Delete with other collection container."""
        for k, v in other.map.items():
            if k in self.map:
                if self.map[k] <= v:
                    self.map.pop(k)
                else:
                    self.map[k] -= v

    def __copy__(self):
        """Copy this container."""
        out = CollectionMap()
        out.merge(self)
        return out

    def __deepcopy__(self, memo):
        """Deepcopy this container."""
        out = CollectionMap()
        out.merge(self)
        return out


class Circuit(list):  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """
    The quantum circuit module.

    A quantum circuit contains one or more quantum gates, and can be evaluated
    in a quantum simulator. You can build a quantum circuit very easy by add
    a quantum gate or another circuit.

    Args:
        gates (BasicGate, list[BasicGate]): You can
            initialize the quantum circuit by a single quantum gate or a
            list of gates. Default: ``None``.


    Examples:
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import RX, X
        >>> circuit1 = Circuit()
        >>> circuit1 += RX('a').on(0)
        >>> circuit1 *= 2
        >>> circuit1
              ┏━━━━━━━┓ ┏━━━━━━━┓
        q0: ──┨ RX(a) ┠─┨ RX(a) ┠───
              ┗━━━━━━━┛ ┗━━━━━━━┛
        >>> circuit2 = Circuit([X.on(0,1)])
        >>> circuit3= circuit1 + circuit2
        >>> assert len(circuit3) == 3
        >>> circuit3.summary()
                Circuit Summary
        ╭──────────────────────┬───────╮
        │ Info                 │ value │
        ├──────────────────────┼───────┤
        │ Number of qubit      │ 2     │
        ├──────────────────────┼───────┤
        │ Total number of gate │ 3     │
        │ Barrier              │ 0     │
        │ Noise Channel        │ 0     │
        │ Measurement          │ 0     │
        ├──────────────────────┼───────┤
        │ Parameter gate       │ 2     │
        │ 1 ansatz parameter   │ a     │
        ╰──────────────────────┴───────╯
        >>> circuit3
              ┏━━━━━━━┓ ┏━━━━━━━┓ ┏━━━┓
        q0: ──┨ RX(a) ┠─┨ RX(a) ┠─┨╺╋╸┠───
              ┗━━━━━━━┛ ┗━━━━━━━┛ ┗━┳━┛
                                    ┃
        q1: ────────────────────────■─────
        >>> Circuit.display_detail(False)
        >>> circuit3
             ┏━━━━┓┏━━━━┓┏━━━┓
        q0: ─┨ RX ┠┨ RX ┠┨╺╋╸┠───
             ┗━━━━┛┗━━━━┛┗━┳━┛
                           ┃
        q1: ───────────────■─────
    """

    # pylint: disable=invalid-name

    def __init__(self, gates=None):
        """Initialize a Circuit object."""
        list.__init__([])
        self.all_qubits = CollectionMap()
        self.all_paras = CollectionMap()
        self.all_measures = CollectionMap()
        self.all_noises = CollectionMap()
        self.all_encoder = CollectionMap()
        self.all_ansatz = CollectionMap()
        if gates is not None:
            if isinstance(gates, Iterable):
                self.extend(gates)
            else:
                self.append(gates)
        self.has_cpp_obj = False
        self.cpp_obj = None
        self.herm_cpp_obj = None

    def _collect_parameterized_gate(self, gate: ParameterGate):
        """Collect parameterized gate information."""
        keys, ansatz_params, encoder_params = gate.__params_prop__()
        self.all_paras.collect(keys)
        gate_ansatz = list(ansatz_params)
        gate_encoder = list(encoder_params)
        for k in gate_ansatz:
            if k in self.all_encoder.map:
                raise RuntimeError(f"Parameter '{k}' already set to encoder parameter.")
        for k in gate_encoder:
            if k in self.all_ansatz.map:
                raise RuntimeError(f"Parameter '{k}' already set to ansatz parameters.")
        self.all_ansatz.collect(gate_ansatz)
        self.all_encoder.collect(gate_encoder)

    def append(self, gate):
        """
        Append a gate.

        Args:
            gate (BasicGate): The gate you want to append.
        """
        _check_gate_type(gate)
        _check_gate_has_obj(gate)
        if isinstance(gate, mq_gates.Measure):
            self.all_measures.collect_only_one(gate, f'measure key {gate.key} already exist.')
        if isinstance(gate, mq_gates.NoiseGate):
            self.all_noises.collect(gate.name)
        self.all_qubits.collect(gate.obj_qubits)
        self.all_qubits.collect(gate.ctrl_qubits)
        if gate.parameterized:
            self._collect_parameterized_gate(gate)
        super().append(gate)
        self.has_cpp_obj = False

    def extend(self, gates):
        """
        Extend a circuit.

        Args:
            gates (Union[Circuit, list[BasicGate]]): A `Circuit` or a list of `BasicGate` you want to extend.
        """
        if isinstance(gates, Circuit):
            self.all_measures.merge_only_one(gates.all_measures, "Measure already exist.")
            self.all_qubits.merge(gates.all_qubits)
            self.all_paras.merge(gates.all_paras)
            self.all_noises.merge(gates.all_noises)
            conflict_params = set(self.all_encoder.keys()) & set(gates.all_ansatz.keys())
            if conflict_params:
                raise RuntimeError(
                    f"Parameters {conflict_params} can not be both encoder parameters and ansatz parameters."
                )
            conflict_params = set(self.all_ansatz.keys()) & set(gates.all_encoder.keys())
            if conflict_params:
                raise RuntimeError(
                    f"Parameters {conflict_params} can not be both encoder parameters and ansatz parameters."
                )
            self.all_encoder.merge(gates.all_encoder)
            self.all_ansatz.merge(gates.all_ansatz)
            super().extend(gates)
        else:
            for gate in gates:
                self.append(gate)
        self.has_cpp_obj = False

    def copy(self):
        """Return a shallow copy of the circuit."""
        return copy.copy(self)

    def __add__(self, gates):
        """Addition operator."""
        out = Circuit()
        out.extend(self)
        if isinstance(gates, mq_gates.BasicGate):
            out.append(gates)
        else:
            out.extend(gates)
        return out

    def __radd__(self, gates):
        """Right-addition operator."""
        if isinstance(gates, int) and gates == 0:
            return self
        return Circuit(gates) + self

    def __iadd__(self, gates):
        """In-place addition operator."""
        if isinstance(gates, mq_gates.BasicGate):
            self.append(gates)
        elif isinstance(gates, Circuit):
            self.extend(gates)
        else:
            raise TypeError(f"Require a quantum gate or a quantum circuit, but get {type(gates)}.")
        return self

    def __mul__(self, num):
        """Repeat the circuit."""
        if not isinstance(num, int):
            raise TypeError(f'{type(num)} object cannot be interpreted as an integer')
        out = Circuit()
        for _ in range(num):
            out += copy.deepcopy(self)
        return out

    def __deepcopy__(self, memo):
        """Deep-copy operator."""
        res = Circuit()
        for gate in self:
            res.append(copy.deepcopy(gate))
        return res

    def __copy__(self):
        """Copy operator."""
        res = Circuit()
        for gate in self:
            res.append(copy.deepcopy(gate))
        return res

    def __rmul__(self, num):
        """Repeat the circuit."""
        return self.__mul__(num)

    def __setitem__(self, k, v):
        """Implement the dictionary-like [] operator (write)."""
        _check_gate_type(v)
        _check_gate_has_obj(v)
        old_v = self[k]
        self.all_qubits.delete(old_v.obj_qubits)
        self.all_qubits.delete(old_v.ctrl_qubits)
        if old_v.parameterized:
            for coeff in old_v.get_parameters():
                self.all_paras.delete(list(coeff.keys()))
                self.all_ansatz.delete(list(coeff.ansatz_parameters))
                self.all_encoder.delete(list(coeff.encoder_parameters))
        if isinstance(old_v, mq_gates.Measure):
            self.all_measures.delete(old_v)
        if isinstance(old_v, mq_gates.NoiseGate):
            self.all_noises.delete(old_v.name)
        super().__setitem__(k, v)
        self.all_qubits.collect(v.obj_qubits)
        self.all_qubits.collect(v.ctrl_qubits)
        if v.parameterized:
            self._collect_parameterized_gate(v)
        if isinstance(v, mq_gates.Measure):
            self.all_measures.collect_only_one(v, f'measure key {v.key} already exist.')
        if isinstance(v, mq_gates.NoiseGate):
            self.all_noises.collect(v.name)
        self.has_cpp_obj = False

    def __getitem__(self, sliced):
        """Implement the dictionary-like [] operator (read)."""
        if isinstance(sliced, int):
            return super().__getitem__(sliced)
        return Circuit(super().__getitem__(sliced))

    def __iter__(self) -> BasicGate:
        """Iterate quantum circuit."""
        yield from super().__iter__()

    def __reduce__(self):
        """Reduce fun for pickle."""
        return (Circuit, (list(self),))

    @property
    def has_measure_gate(self):
        """
        To check whether this circuit has measure gate.

        Returns:
            bool, whether this circuit has measure gate.
        """
        return self.all_measures.size != 0

    @property
    def parameterized(self):
        """
        To check whether this circuit is a parameterized quantum circuit.

        Returns:
            bool, whether this circuit is a parameterized quantum circuit.
        """
        return self.all_paras.size != 0

    @property
    def is_noise_circuit(self):
        """
        To check whether this circuit has noise channel.

        Returns:
            bool, whether this circuit has noise channel.
        """
        return self.all_noises.size != 0

    def insert(self, index, gates):
        """
        Insert a quantum gate or quantum circuit in index.

        Args:
            index (int): Index to set gate.
            gates (Union[BasicGate, list[BasicGate]]): Gates you need to insert.
        """
        if isinstance(gates, mq_gates.BasicGate):
            _check_gate_has_obj(gates)
            _check_gate_type(gates)
            super().insert(index, gates)
            self.all_qubits.collect(gates.obj_qubits)
            self.all_qubits.collect(gates.ctrl_qubits)
            if gates.parameterized:
                self._collect_parameterized_gate(gates)
            if isinstance(gates, mq_gates.Measure):
                self.all_measures.collect_only_one(gates, f'measure key {gates.key} already exist.')
        elif isinstance(gates, Iterable):
            for gate in gates[::-1] if index > 0 else gates:
                self.insert(index, gate)
                self.all_qubits.collect(gate.obj_qubits)
                self.all_qubits.collect(gate.ctrl_qubits)
                if gate.parameterized:
                    self._collect_parameterized_gate(gate)
                if isinstance(gate, mq_gates.Measure):
                    self.all_measures.collect_only_one(gate, f'measure key {gate.key} already exist.')
        else:
            raise TypeError(f"Unsupported type for quantum gate: {type(gates)}")
        self.has_cpp_obj = False

    def no_grad(self):
        """Set all parameterized gate in this quantum circuit not require grad."""
        for gate in self:
            gate.no_grad()
        self.has_cpp_obj = False
        return self

    def requires_grad(self):
        """Set all parameterized gates in this quantum circuit require grad."""
        for gate in self:
            gate.requires_grad()
        self.has_cpp_obj = False
        return self

    def compress(self):
        r"""
        Remove all unused qubits, and map qubits to `range(n_qubits)`.

        Examples:
            >>> from mindquantum.algorithm.library import qft
            >>> qft([0, 2, 4])
                  ┏━━━┓ ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
            q0: ──┨ H ┠─┨ PS(π/2) ┠─┨ PS(π/4) ┠─────────────────────────╳───
                  ┗━━━┛ ┗━━━━┳━━━━┛ ┗━━━━┳━━━━┛                         ┃
                             ┃           ┃      ┏━━━┓ ┏━━━━━━━━━┓       ┃
            q2: ─────────────■───────────╂──────┨ H ┠─┨ PS(π/2) ┠───────┃───
                                         ┃      ┗━━━┛ ┗━━━━┳━━━━┛       ┃
                                         ┃                 ┃      ┏━━━┓ ┃
            q4: ─────────────────────────■─────────────────■──────┨ H ┠─╳───
                                                                  ┗━━━┛
            >>> qft([0, 2, 4]).compress()
                  ┏━━━┓ ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
            q0: ──┨ H ┠─┨ PS(π/2) ┠─┨ PS(π/4) ┠─────────────────────────╳───
                  ┗━━━┛ ┗━━━━┳━━━━┛ ┗━━━━┳━━━━┛                         ┃
                             ┃           ┃      ┏━━━┓ ┏━━━━━━━━━┓       ┃
            q1: ─────────────■───────────╂──────┨ H ┠─┨ PS(π/2) ┠───────┃───
                                         ┃      ┗━━━┛ ┗━━━━┳━━━━┛       ┃
                                         ┃                 ┃      ┏━━━┓ ┃
            q2: ─────────────────────────■─────────────────■──────┨ H ┠─╳───
                                                                  ┗━━━┛
        """
        return apply(self, list(range(len(self.all_qubits))))

    def __str__(self):
        """Return a string representation of the object."""
        return self.__repr__()

    def __repr__(self):
        """Return a string representation of the object."""
        # pylint: disable=import-outside-toplevel
        from mindquantum.io.display._config import _text_circ_config
        from mindquantum.io.display.circuit_text_drawer import rich_circuit
        from mindquantum.io.display.circuit_text_drawer_helper import Monitor

        if not self:
            return ""
        console = Console(record=True)
        circ = self.compress() if _text_circ_config.compress_unuse_qubit else self
        rich_circ = rich_circuit(
            circ,
            console.width,
            style=_text_circ_config,
            qubit_map=dict(enumerate(sorted(self.all_qubits.map.keys()))),
        )
        if console.is_jupyter:
            rich_circ.disable_rich()
        string = Monitor(rich_circ).get_str()
        if not console.is_jupyter:
            with console.capture() as capture:
                console.print(string, width=len(string))
            return capture.get()
        return string

    def _repr_html_(self):
        """Repr for jupyter notebook."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.io.display._config import (
            CIRCUIT_HTML_FORMAT,
            _text_circ_config,
        )
        from mindquantum.io.display.circuit_text_drawer import rich_circuit
        from mindquantum.io.display.circuit_text_drawer_helper import Monitor

        if not self:
            return ""
        console = Console(record=True)
        circ = self.compress() if _text_circ_config.compress_unuse_qubit else self
        string = Monitor(
            rich_circuit(
                circ,
                console.width,
                style=_text_circ_config,
                qubit_map=dict(enumerate(sorted(self.all_qubits.map.keys()))),
            )
        ).get_str()
        with console.capture() as _:
            console.print(string, width=len(string))
        string = console.export_html(code_format=CIRCUIT_HTML_FORMAT, inline_styles=True)
        return '\n'.join(string.split('\n')[1:])

    @property
    def n_qubits(self):
        """Get the total number of qubits used."""
        if self.all_qubits:
            return max(self.all_qubits.keys()) + 1
        return 0

    def summary(self, show=True):
        r"""
        Print a summary of the current circuit.

        Print the information about current circuit, including block number,
        gate number, non-parameterized gate number, parameterized gate number
        and the total parameters.

        Args:
            show (bool): whether to show the information. Default: ``True``.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.gates import RX, H
            >>> circuit = Circuit([RX('a').on(1), H.on(1), RX('b').on(0)])
            >>> circuit.summary()
                    Circuit Summary
            ╭──────────────────────┬───────╮
            │ Info                 │ value │
            ├──────────────────────┼───────┤
            │ Number of qubit      │ 2     │
            ├──────────────────────┼───────┤
            │ Total number of gate │ 3     │
            │ Barrier              │ 0     │
            │ Noise Channel        │ 0     │
            │ Measurement          │ 0     │
            ├──────────────────────┼───────┤
            │ Parameter gate       │ 2     │
            │ 2 ansatz parameters  │ a, b  │
            ╰──────────────────────┴───────╯
        """
        _check_input_type('show', bool, show)
        num_non_para_gate = 0
        num_para_gate = 0
        barrier = 0
        noise_channel = 0
        measure_gate = 0
        for gate in self:
            if isinstance(gate, mq_gates.BarrierGate):
                barrier += 1
                continue
            if isinstance(gate, mq_gates.NoiseGate):
                noise_channel += 1
                continue
            if isinstance(gate, mq_gates.Measure):
                measure_gate += 1
                continue
            if gate.parameterized:
                num_para_gate += 1
            else:
                num_non_para_gate += 1

        title = Text("Circuit Summary", style="bold #ff0000")
        table = Table(title=title, box=ROUNDED)
        table.add_column("[#3b3b95]Info[/]")
        table.add_column("[#3b3b95]value[/]")
        table.add_row("[bold]Number of qubit[/]", f"{self.n_qubits}", end_section=True)
        table.add_row("[bold]Total number of gate[/]", f"{num_para_gate + num_non_para_gate}")
        table.add_row("Barrier", f"{barrier}")
        table.add_row("Noise Channel", f"{noise_channel}")
        table.add_row("Measurement", f"{measure_gate}", end_section=True)

        def show_params(table: Table, name: str, params, n_limit=10, color='#000000'):
            if not params:
                return
            n_p = len(params)
            row_name = f"{quantifier_selector(n_p, f'{name} parameter', f'{name} parameters')}"
            params_name = ', '.join(params[:n_limit])
            if n_p > n_limit:
                params_name += '...'
            table.add_row(row_name, f'[{color}]{params_name}[/]')

        table.add_row("[bold]Parameter gate[/]", f"{num_para_gate}")
        if self.params_name:
            show_params(table, "encoder", self.encoder_params_name, color='#FFC857')
            show_params(table, "ansatz", self.ansatz_params_name, color='#48C9B0')

        if show:
            console = Console()
            console.print(table)

    def hermitian(self):
        """
        Get the hermitian of this quantum circuit.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.gates import RX
            >>> circ = Circuit(RX({'a': 0.2}).on(0))
            >>> herm_circ = circ.hermitian()
            >>> print(herm_circ)
                  ┏━━━━━━━━━━━━┓
            q0: ──┨ RX(-1/5*a) ┠───
                  ┗━━━━━━━━━━━━┛
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
        pr = ParameterResolver(self.all_paras.map)
        for k in pr.keys():
            pr[k] = 1
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

    @property
    def encoder_params_name(self):
        """
        Get the encoder parameter name of this circuit.

        Returns:
            list, a list that contains the parameter name that works as encoder.

        Examples:
            >>> from mindquantum.core.gates import RX, RY
            >>> from mindquantum.core.circuit import Circuit
            >>> circuit = Circuit(RX({'a': 1, 'b': 2}).on(0)).as_encoder()
            >>> circuit += Circuit(RY('c').on(0)).as_ansatz()
            >>> circuit.encoder_params_name
            ['a', 'b']
        """
        return list(self.all_encoder.keys())

    @property
    def ansatz_params_name(self):
        """
        Get the ansatz parameter name of this circuit.

        Returns:
            list, a list that contains the parameter name that works as ansatz.

        Examples:
            >>> from mindquantum.core.gates import RX, RY
            >>> from mindquantum.core.circuit import Circuit
            >>> circuit = Circuit(RX({'a': 1, 'b': 2}).on(0)).as_encoder()
            >>> circuit += Circuit(RY('c').on(0)).as_ansatz()
            >>> circuit.ansatz_params_name
            ['c']
        """
        return list(self.all_ansatz.keys())

    @property
    def is_measure_end(self):
        """
        Check whether each qubit has a measurement as its last operation.

        Check whether the circuit is end with measurement gate that there is at most one measurement
        gate that act on each qubit, and this measurement gate should be at end of gate serial of
        this qubit.

        Returns:
            bool, whether the circuit is end with measurement.
        """
        circ = self.remove_barrier()
        high = [0 for i in range(self.n_qubits)]
        for gate in circ:
            for idx in set(gate.obj_qubits + gate.ctrl_qubits):
                high[idx] += 1
            if isinstance(gate, mq_gates.Measure):
                m_idx = gate.obj_qubits[0]
                if high[m_idx] != self.all_qubits.map[m_idx]:
                    return False
        return True

    def matrix(self, pr=None, big_end=False, backend='mqvector', seed=None, dtype=None):
        """
        Get the matrix of this circuit.

        Args:
            pr (ParameterResolver, dict, numpy.ndarray, list, numbers.Number): The parameter
                resolver for parameterized quantum circuit. Default: None.
            big_end (bool): The low index qubit is place in the end or not. Default: False.
            backend (str): The backend to do simulation. Default: 'mqvector'.
            seed (int): The random to generate circuit matrix, if the circuit has noise channel.
            dtype (mindquantum.dtype): data type of simulator. Default: 'None'.

        Returns:
            numpy.ndarray, two dimensional complex matrix of this circuit.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> circuit = Circuit().rx('a',0).h(0)
            >>> circuit.matrix({'a': 1.0})
            array([[ 0.62054458-0.33900505j,  0.62054458-0.33900505j],
                   [ 0.62054458+0.33900505j, -0.62054458-0.33900505j]])
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
        if self.is_noise_circuit:
            raise ValueError("This circuit cannot have noise channel.")
        if backend.startswith('mqmatrix'):
            raise ValueError("mqmatrix backend not support to get circuit matrix.")
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.simulator import Simulator

        sim = Simulator(backend, self.n_qubits, seed=seed, dtype=dtype)
        return np.array(sim.backend.get_circuit_matrix(circ, pr)).T

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
                  ┏━━━┓ ┏━━━━━━━┓
            q0: ──┨╺╋╸┠─┨ RX(3) ┠───
                  ┗━━━┛ ┗━━━━━━━┛
        """
        circuit = Circuit()
        for gate in self:
            if not gate.parameterized:
                circuit += gate
            else:
                coeffs = []
                for coeff in gate.get_parameters():
                    coeff = coeff * 1
                    for k, v in dict(coeff).items():
                        if k in pr:
                            coeff.const += pr[k] * v
                            coeff.pop(k)
                    coeffs.append(coeff)
                circuit += gate.__class__(*coeffs).on(gate.obj_qubits, gate.ctrl_qubits)
        return circuit

    @staticmethod
    def display_detail(state: bool):
        """
        Whether to display the detail of circuit.

        Args:
            state (bool): The state of whether to display the detail of circuit.

        Examples:
            >>> from mindquantum import Circuit
            >>> circ = Circuit().rx('a', 0).ry(1.2, 0)
            >>> circ
                  ┏━━━━━━━┓ ┏━━━━━━━━━┓
            q0: ──┨ RX(a) ┠─┨ RY(6/5) ┠───
                  ┗━━━━━━━┛ ┗━━━━━━━━━┛
            >>> Circuit.display_detail(False)
            >>> circ
                 ┏━━━━┓┏━━━━┓
            q0: ─┨ RX ┠┨ RY ┠───
                 ┗━━━━┛┗━━━━┛
        """
        # pylint: disable=import-outside-toplevel
        from mindquantum.io.display._config import _text_circ_config

        _text_circ_config.simple_mode = not state

    def remove_barrier(self):
        """Remove all barrier gates."""
        circ = Circuit()
        for gate in self:
            if not isinstance(gate, mq_gates.BarrierGate):
                circ += gate
        return circ

    def remove_measure(self):
        """Remove all measure gate."""
        circ = Circuit()
        for gate in self:
            if not isinstance(gate, mq_gates.Measure):
                circ += gate
        return circ

    def remove_measure_on_qubits(self, qubits):
        """
        Remove all measure gate on some certain qubits.

        Args:
            qubit (Union[int, list[int]]): The qubits you want to remove measure.

        Examples:
            >>> from mindquantum.core.circuit import UN
            >>> from mindquantum.core.gates import H, Measure
            >>> circ = UN(H, 3).x(0, 1).x(1, 2).measure_all()
            >>> circ += H.on(0)
            >>> circ += Measure('q0_1').on(0)
            >>> circ.remove_measure_on_qubits(0)
                  ┏━━━┓ ┏━━━┓ ┏━━━┓
            q0: ──┨ H ┠─┨╺╋╸┠─┨ H ┠────────────
                  ┗━━━┛ ┗━┳━┛ ┗━━━┛
                  ┏━━━┓   ┃   ┏━━━┓ ┍━━━━━━┑
            q1: ──┨ H ┠───■───┨╺╋╸┠─┤ M q1 ├───
                  ┗━━━┛       ┗━┳━┛ ┕━━━━━━┙
                  ┏━━━┓         ┃   ┍━━━━━━┑
            q2: ──┨ H ┠─────────■───┤ M q2 ├───
                  ┗━━━┛             ┕━━━━━━┙
        """
        if not isinstance(qubits, list):
            qubits = [qubits]
        circ = Circuit()
        for gate in self:
            if isinstance(gate, mq_gates.Measure) and gate.obj_qubits[0] in qubits:
                continue
            circ += gate
        return circ

    def get_cpp_obj(self, hermitian=False):
        """
        Get cpp obj of circuit.

        Args:
            hermitian (bool): Whether to get cpp object of this circuit in hermitian version. Default: ``False``.
        """
        if not self.has_cpp_obj:
            self.has_cpp_obj = True
            self.cpp_obj = [i.get_cpp_obj() for i in self if not isinstance(i, mq_gates.BarrierGate)]
            self.herm_cpp_obj = [i.get_cpp_obj() for i in self.hermitian() if not isinstance(i, mq_gates.BarrierGate)]

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
            ctrl_qubits (Union[int, list[int]]): the control qubits of `H` gate. Default: ``None``.
        """
        self.append(mq_gates.H.on(obj_qubits, ctrl_qubits))
        return self

    def x(self, obj_qubits, ctrl_qubits=None):
        """
        Add a X gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `X` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `X` gate. Default: ``None``.
        """
        self.append(mq_gates.X.on(obj_qubits, ctrl_qubits))
        return self

    def y(self, obj_qubits, ctrl_qubits=None):
        """
        Add a Y gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `Y` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Y` gate. Default: ``None``.
        """
        self.append(mq_gates.Y.on(obj_qubits, ctrl_qubits))
        return self

    def z(self, obj_qubits, ctrl_qubits=None):
        """
        Add a Z gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `Z` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Z` gate. Default: ``None``.
        """
        self.append(mq_gates.Z.on(obj_qubits, ctrl_qubits))
        return self

    def s(self, obj_qubits, ctrl_qubits=None, hermitian=False):
        """
        Add a S gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `S` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `S` gate. Default: ``None``.
            hermitian (bool): Whether use the hermitian conjugated version. Default: ``False``.
        """
        gate = mq_gates.S.on(obj_qubits, ctrl_qubits)
        if hermitian:
            gate = gate.hermitian()
        self.append(gate)
        return self

    def t(self, obj_qubits, ctrl_qubits=None, hermitian=False):
        """
        Add a T gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `T` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `T` gate. Default: ``None``.
            hermitian (bool): Whether use the hermitian conjugated version. Default: ``False``.
        """
        gate = mq_gates.T.on(obj_qubits, ctrl_qubits)
        if hermitian:
            gate = gate.hermitian()
        self.append(gate)
        return self

    def to_openqasm(self, file_name: Optional[str] = None, version: str = '2.0') -> str:
        """
        Convert a MindQuantum circuit to OpenQASM format string or file.

        Args:
            file_name (str): File name if you want to save OpenQASM. If it is ``None``, we
                will return the string. Otherwise, we will save to given file.
                Default: ``None``.
            version (str): The OpenQASM version you want to use. Default: ``'2.0'``.
        """
        # pylint: disable=import-outside-toplevel
        from mindquantum.io.qasm import OpenQASM

        if file_name is None:
            return OpenQASM().to_string(self, version=version)
        OpenQASM().to_file(file_name, self, version=version)
        return ""

    def to_hiqasm(self, file_name: Optional[str] = None, version: str = '0.1') -> str:
        """
        Convert a MindQuantum circuit to HiQASM format string or file.

        Args:
            file_name (str): File name if you want to save HiQASM. If it is ``None``, we
                will return the string. Otherwise, we will save to given file.
                Default: ``None``.
            version (str): The HiQASM version you want to use. Default: ``'0.1'``.
        """
        # pylint: disable=import-outside-toplevel
        from mindquantum.io.qasm import HiQASM

        if file_name is None:
            return HiQASM().to_string(self, version=version)
        HiQASM().to_file(file_name, self, version=version)
        return ""

    def sx(self, obj_qubits, ctrl_qubits=None, hermitian=False):
        """
        Add a SX gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `SX` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `SX` gate. Default: ``None``.
            hermitian (bool): Whether use the hermitian conjugated version. Default: ``False``.
        """
        gate = mq_gates.SX.on(obj_qubits, ctrl_qubits)
        if hermitian:
            gate = gate.hermitian()
        self.append(gate)
        return self

    def swap(self, obj_qubits, ctrl_qubits=None):
        """
        Add a SWAP gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `SWAP` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `SWAP` gate. Default: ``None``.
        """
        self.append(mq_gates.SWAP.on(obj_qubits, ctrl_qubits))
        return self

    def iswap(self, obj_qubits, ctrl_qubits=None):
        """
        Add a ISWAP gate.

        Args:
            obj_qubits (Union[int, list[int]]): The object qubits of `ISWAP` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `ISWAP` gate. Default: ``None``.
        """
        self.append(mq_gates.ISWAP.on(obj_qubits, ctrl_qubits))
        return self

    def swap_alpha(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a SWAPalpha gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `SWAPalpha` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `SWAPalpha` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `SWAPalpha` gate. Default: ``None``.
        """
        self.append(mq_gates.SWAPalpha(para).on(obj_qubits, ctrl_qubits))
        return self

    def rx(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a RX gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `RX` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `RX` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `RX` gate. Default: ``None``.
        """
        self.append(mq_gates.RX(para).on(obj_qubits, ctrl_qubits))
        return self

    def ry(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a RY gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `RY` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `RY` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `RY` gate. Default: ``None``.
        """
        self.append(mq_gates.RY(para).on(obj_qubits, ctrl_qubits))
        return self

    def rz(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a RZ gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `RZ` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `RZ` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `RZ` gate. Default: ``None``.
        """
        self.append(mq_gates.RZ(para).on(obj_qubits, ctrl_qubits))
        return self

    def phase_shift(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Phase Shift gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `PhaseShift` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `PhaseShift` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `PhaseShift` gate. Default: ``None``.
        """
        self.append(mq_gates.PhaseShift(para).on(obj_qubits, ctrl_qubits))
        return self

    def global_phase(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Global Phase gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `GlobalPhase` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `GlobalPhase` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `GlobalPhase` gate. Default: ``None``.
        """
        self.append(mq_gates.GlobalPhase(para).on(obj_qubits, ctrl_qubits))
        return self

    def givens(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Givens rotation gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `Givens` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `Givens` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Givens` gate. Default: ``None``.
        """
        self.append(mq_gates.Givens(para).on(obj_qubits, ctrl_qubits))
        return self

    def u3(self, theta, phi, lamda, obj_qubits, ctrl_qubits=None):
        """
        Add a U3 gate.

        Args:
            theta (Union[dict, ParameterResolver]): First parameter for `U3` gate.
            phi (Union[dict, ParameterResolver]): Second parameter for `U3` gate.
            lamda (Union[dict, ParameterResolver]): Third parameter for `U3` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `U3` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `U3` gate. Default: ``None``.
        """
        self.append(mq_gates.U3(theta, phi, lamda).on(obj_qubits, ctrl_qubits))
        return self

    def rxx(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Rxx gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `Rxx` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `Rxx` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Rxx` gate. Default: ``None``.
        """
        self.append(mq_gates.Rxx(para).on(obj_qubits, ctrl_qubits))
        return self

    def rxy(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Rxy gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `Rxy` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `Rxy` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Rxy` gate. Default: ``None``.
        """
        self.append(mq_gates.Rxy(para).on(obj_qubits, ctrl_qubits))
        return self

    def rxz(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Rxz gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `Rxz` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `Rxz` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Rxz` gate. Default: ``None``.
        """
        self.append(mq_gates.Rxz(para).on(obj_qubits, ctrl_qubits))
        return self

    def ryy(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Ryy gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `Ryy` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `Ryy` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Ryy` gate. Default: ``None``.
        """
        self.append(mq_gates.Ryy(para).on(obj_qubits, ctrl_qubits))
        return self

    def ryz(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Ryz gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `Ryz` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `Ryz` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Ryz` gate. Default: ``None``.
        """
        self.append(mq_gates.Ryz(para).on(obj_qubits, ctrl_qubits))
        return self

    def rzz(self, para, obj_qubits, ctrl_qubits=None):
        """
        Add a Rzz gate.

        Args:
            para (Union[dict, ParameterResolver]): The parameter for `Rzz` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `Rzz` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `Rzz` gate. Default: ``None``.
        """
        self.append(mq_gates.Rzz(para).on(obj_qubits, ctrl_qubits))
        return self

    @staticmethod
    def from_openqasm(openqasm_str: str):
        """
        Convert an OpenQASM string or an OpenQASM file to MindQuantum circuit.

        Args:
            openqasm_str (str): String format of OpenQASM or an OpenQASM file name.

        Returns:
            :class:`~.core.circuit.Circuit`, The MindQuantum circuit converted from OpenQASM.
        """
        # pylint: disable=import-outside-toplevel
        from mindquantum.io.qasm import OpenQASM

        if os.path.exists(openqasm_str):
            return OpenQASM().from_file(openqasm_str)
        return OpenQASM().from_string(openqasm_str)

    @staticmethod
    def from_hiqasm(hiqasm_str: str):
        """
        Convert a HiQASM string or a HiQASM file to MindQuantum circuit.

        Args:
            hiqasm_str (str): String format of HiQASM or a HiQASM file name.

        Returns:
            :class:`~.core.circuit.Circuit`, The MindQuantum circuit converted from HiQASM.
        """
        # pylint: disable=import-outside-toplevel
        from mindquantum.io.qasm import HiQASM

        if os.path.exists(hiqasm_str):
            return HiQASM().from_file(hiqasm_str)
        return HiQASM().from_string(hiqasm_str)

    def fsim(self, theta, phi, obj_qubits, ctrl_qubits=None):
        """
        Add a FSim gate.

        Args:
            theta (Union[dict, ParameterResolver]): First parameter for `FSim` gate.
            phi (Union[dict, ParameterResolver]): Second parameter for `FSim` gate.
            obj_qubits (Union[int, list[int]]): The object qubits of `FSim` gate.
            ctrl_qubits (Union[int, list[int]]): the control qubits of `FSim` gate. Default: ``None``.
        """
        self.append(mq_gates.FSim(theta, phi).on(obj_qubits, ctrl_qubits))
        return self

    def measure(self, key, obj_qubit=None, reset_to=None):
        """
        Add a measure gate.

        Args:
            key (Union[int, str]): If `obj_qubit` is ``None``, then `key` should be a int and means which
                qubit to measure, otherwise, `key` should be a str and means the name of this measure gate.
            obj_qubit (int): Which qubit to measure. Default: ``None``.
            reset_to (Union[int, None]): Reset the qubit to 0 state or 1 state. If ``None``, do not reset.
                Default: ``None``.
        """
        if obj_qubit is None:
            self.append(mq_gates.Measure(reset_to=reset_to).on(key))
        else:
            self.append(mq_gates.Measure(key, reset_to=reset_to).on(obj_qubit))
        return self

    def measure_all(self, suffix=None, up_to: int = -1):
        """
        Measure all qubits.

        Args:
            suffix (str): The suffix string you want to add to the name of measure gate.
            up_to (int): The maximum qubit you want to measure. If this value is less than the qubit number of
                this quantum circuit, the circuit qubit number will be used. Default: ``-1``.
        """
        for i in range(max(self.n_qubits, up_to)):
            string = f"q{i}" if suffix is None else f"q{i}_{suffix}"
            self += mq_gates.Measure(string).on(i)
        return self

    def barrier(self, show=True):
        """
        Add a barrier.

        Args:
            show (bool): Whether show barrier or not. Default: True.
        """
        _check_input_type('show', bool, show)
        self.append(mq_gates.BarrierGate(show))
        return self

    def un(self, gate, maps_obj, maps_ctrl=None):
        """
        Map a quantum gate to different objective qubits and control qubits.

        Please refer to UN.

        Args:
            gate (BasicGate): The BasicGate you want to map.
            maps_obj (Union[int, list[int]]): object qubits.
            maps_ctrl (Union[int, list[int]]): control qubits. Default: ``None``.
        """
        from mindquantum import UN  # pylint: disable=import-outside-toplevel

        self += UN(gate, maps_obj, maps_ctrl)
        return self

    def get_qs(self, backend='mqvector', pr=None, ket=False, seed=None, dtype=None):
        """
        Get the final quantum state of this circuit.

        Args:
            backend (str): Which backend you want to use. Default: ``'mqvector'``.
            pr (Union[numbers.Number, ParameterResolver, dict, numpy.ndarray]): The parameter of this circuit,
                if this circuit is parameterized. Default: ``None``.
            ket (str): Whether to return the quantum state in ket format. Default: ``False``.
            seed (int): The random seed of simulator. Default: ``None``
            dtype (mindquantum.dtype): The data type of simulator.
        """
        from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Simulator,
        )

        sim = Simulator(backend, self.n_qubits, seed=seed, dtype=dtype)
        sim.apply_circuit(self, pr)
        return sim.get_qs(ket)

    def reverse_qubits(self):
        """
        Flip the circuit to big endian.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> circ = Circuit().h(0).x(2, 0).y(3).x(3, 2)
            >>> circ
                  ┏━━━┓
            q0: ──┨ H ┠───■───────────
                  ┗━━━┛   ┃
                        ┏━┻━┓
            q2: ────────┨╺╋╸┠───■─────
                        ┗━━━┛   ┃
                  ┏━━━┓       ┏━┻━┓
            q3: ──┨ Y ┠───────┨╺╋╸┠───
                  ┗━━━┛       ┗━━━┛
            >>> circ.reverse_qubits()
                  ┏━━━┓       ┏━━━┓
            q0: ──┨ Y ┠───────┨╺╋╸┠───
                  ┗━━━┛       ┗━┳━┛
                        ┏━━━┓   ┃
            q1: ────────┨╺╋╸┠───■─────
                        ┗━┳━┛
                  ┏━━━┓   ┃
            q3: ──┨ H ┠───■───────────
                  ┗━━━┛
        """
        return apply(self, [self.n_qubits - 1 - i for i in self.all_qubits.keys()])

    def svg(self, style=None, width=None, scale=None):
        """
        Display current quantum circuit into SVG picture in jupyter notebook.

        Args:
            style (dict, str): the style to set svg circuit. Currently, we support
                ``'official'``, ``'light'`` and ``'dark'``. Default: ``None``.
            width (int, float): the max width of circuit. Default: ``None``.
            scale (Union[float, None]): Scale factor for scaling svg. If ``None``, do
                not scale. Default: ``None``.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.io.display._config import (
            _svg_config_dark,
            _svg_config_light,
            _svg_config_official,
        )
        from mindquantum.io.display.circuit_svg_drawer import SVGCircuit

        if width is None:
            width = np.inf
        if not isinstance(width, (int, float)):
            raise TypeError(f"width requires a int or a float, but get {type(width)}")
        if width < 250:
            raise ValueError("Windows too small to display svg circuit, width should be more than 250.")
        supported_style = {
            'official': _svg_config_official,
            'dark': _svg_config_dark,
            'light': _svg_config_light,
        }
        if style is None:
            style = _svg_config_official
        _check_input_type("style", (dict, str), style)
        if isinstance(style, str):
            if style not in supported_style:
                raise ValueError(f"Style not found, currently we support {list(supported_style.keys())}")
            style = supported_style[style]
        svg = SVGCircuit(self, style, width)
        if scale is not None:
            svg.scale(scale)
        return svg

    def remove_noise(self):
        """Remove all noise gate."""
        circ = Circuit()
        for gate in self:
            if not isinstance(gate, mq_gates.NoiseGate):
                circ += gate
        return circ

    def with_noise(self, noise_gate=mq_gates.AmplitudeDampingChannel(0.001), also_ctrl=False):
        """
        Apply noises on each gate.

        Args:
            noise_gate (NoiseGate): The NoiseGate you want to apply. Default: ``AmplitudeDampingChannel(0.001)``.
            also_ctrl (bool): Whether add NoiseGate on control qubits. Default: ``False``.
        """
        circ = Circuit()
        for gate in self:
            circ += gate
            if not isinstance(gate, (mq_gates.Measure, mq_gates.NoiseGate)):
                for i in gate.obj_qubits:
                    circ += noise_gate.on(i)
                if also_ctrl:
                    for i in gate.ctrl_qubits:
                        circ += noise_gate.on(i)
        return circ

    def as_encoder(self, inplace=True):
        """
        To set this circuit to encoder.

        Args:
            inplace (bool): Whether to set inplace. Defaults: ``True``.
        """
        _check_input_type("inplace", bool, inplace)
        if inplace:
            circ = self
        else:
            circ = self * 1
        for gate in circ:
            if gate.parameterized:
                for coeff in gate.get_parameters():
                    coeff.as_encoder()
        circ.all_encoder.merge(circ.all_ansatz)
        circ.all_ansatz.map = {}
        circ.has_cpp_obj = False
        return circ

    def as_ansatz(self, inplace=True):
        """
        To set this circuit to ansatz or not.

        Args:
            inplace (bool): Whether to set inplace. Defaults: ``True``.
        """
        _check_input_type("inplace", bool, inplace)
        if inplace:
            circ = self
        else:
            circ = self * 1
        gate: ParameterGate
        for gate in circ:
            if gate.parameterized:
                for coeff in gate.get_parameters():
                    coeff.as_ansatz()
        circ.all_ansatz.merge(circ.all_encoder)
        circ.all_encoder.map = {}
        circ.has_cpp_obj = False
        return circ


A = apply

__all__ = ['Circuit', 'A']
