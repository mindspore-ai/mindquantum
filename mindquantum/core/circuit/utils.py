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
"""Tools for MindQuantum eDSL"""
from types import FunctionType, MethodType
import copy
import numpy as np
from projectq.ops import QubitOperator as pq_operator
from openfermion.ops import QubitOperator as of_operator

from mindquantum.core.parameterresolver.parameterresolver import ParameterResolver


def decompose_single_term_time_evolution(term, para):
    """
    Decompose a time evolution gate into basic quantum gates.

    This function only work for the hamiltonian with only single pauli word.
    For example, exp(-i * t * ham), ham can only be a single pauli word, such
    as ham = X0 x Y1 x Z2, and at this time, term will be
    ((0, 'X'), (1, 'Y'), (2, 'Z')). When the evolution time is expressd as
    t = a*x + b*y, para would be {'x':a, 'y':b}.

    Args:
        term (tuple, QubitOperator): the hamiltonian term of just the
            evolution qubit operator.
        para (Union[dict, numbers.Number]): the parameters of evolution operator.

    Returns:
        Circuit, a quantum circuit.

    Raises:
        ValueError: If term has more than one pauli string.
        TypeError: If term is not map.

    Example:
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum.core.circuit import decompose_single_term_time_evolution
        >>> ham = QubitOperator('X0 Y1')
        >>> circuit = decompose_single_term_time_evolution(ham, {'a':1})
        >>> print(circuit)
        q0: ─────H───────●───────────────●───────H──────
                         │               │
        q1: ──RX(π/2)────X────RZ(2*a)────X────RX(7π/2)──
    """
    from mindquantum import gates as G
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.parameterresolver import ParameterResolver as PR
    from mindquantum.utils.type_value_check import _num_type
    if not isinstance(term, tuple):
        try:
            if len(term.terms) != 1:
                raise ValueError("Only work for single term time \
                    evolution operator, but get {}".format(len(term)))
            term = list(term.terms.keys())[0]
        except TypeError:
            raise TypeError("Not supported type:{}".format(type(term)))
    if not isinstance(para, _num_type):
        if not isinstance(para, (dict, ParameterResolver)):
            raise TypeError(f'para requiers a number or a dict or a ParameterResolver, but get {type(para)}')
        para = ParameterResolver(para)

    out = []
    term = sorted(term)
    rxs = []
    if len(term) == 1:  # single pauli operator
        if term[0][1] == 'X':
            out.append(G.RX(para * 2).on(term[0][0]))
        elif term[0][1] == 'Y':
            out.append(G.RY(para * 2).on(term[0][0]))
        else:
            out.append(G.RZ(para * 2).on(term[0][0]))
    else:
        for index, action in term:
            if action == 'X':
                out.append(G.H.on(index))
            elif action == 'Y':
                rxs.append(len(out))
                out.append(G.RX(np.pi / 2).on(index))

        out.append(G.BarrierGate(False))
        for i in range(len(term) - 1):
            out.append(G.X.on(term[i + 1][0], term[i][0]))
        out.append(G.BarrierGate(False))
        if isinstance(para, (dict, PR)):
            out.append(G.RZ({i: 2 * j for i, j in para.items()}).on(term[-1][0]))
        else:
            out.append(G.RZ(2 * para).on(term[-1][0]))
        for i in range(len(out) - 1)[::-1]:
            if i in rxs:
                out.append(G.RX(np.pi * 3.5).on(out[i].obj_qubits))
            else:
                out.append(out[i])
    return Circuit(out)


def pauli_word_to_circuits(qubitops):
    """
    Convert a single pauli word qubit operator to a quantum circuit.

    Args:
        qubitops (QubitOperator, Hamiltonian): The single pauli word qubit operator.

    Returns:
        Circuit, a quantum circuit.

    Raises:
        TypeError: If qubitops is not a QubitOperator or a Hamiltonian.
        ValueError: If qubitops is Hamiltonian but not in origin mode.
        ValueError: If qubitops has more than one pauliwords.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum.core.circuit import pauli_word_to_circuits
        >>> qubitops = QubitOperator('X0 Y1')
        >>> pauli_word_to_circuits(qubitops)
        q0: ──X──

        q1: ──Y──
    """
    from mindquantum import gates as G
    from mindquantum import operators as ops
    from mindquantum.core import Circuit
    allow_ops = (pq_operator, of_operator, ops.QubitOperator, ops.Hamiltonian)
    if not isinstance(qubitops, allow_ops):
        raise TypeError("qubitops require a QubitOperator or a Hamiltonian, but get {}!".format(type(qubitops)))
    if isinstance(qubitops, ops.Hamiltonian):
        if qubitops.how_to != 0:
            raise ValueError("Hamiltonian should be in origin mode.")
        qubitops = qubitops.hamiltonian
    if len(qubitops.terms) > 1:
        raise ValueError("Onle work for QubitOperator with single pauliword!")
    gate_map = {'X': G.X, 'Y': G.Y, 'Z': G.Z}
    for ops in qubitops.terms.keys():
        circ = Circuit()
        if ops:
            for ind, single_op in ops:
                circ += gate_map[single_op].on(ind)
        else:
            circ += G.I.on(0)
    return circ


def _add_ctrl_qubits(circ, ctrl_qubits):
    """Add control qubits on a circuit."""
    from mindquantum.core import Circuit
    from mindquantum import gates as G
    if not isinstance(ctrl_qubits, (int, list)):
        raise TypeError("Require a int or a list of int for ctrl_qubits, but get {}!".format(type(ctrl_qubits)))
    if isinstance(ctrl_qubits, int):
        ctrl_qubits = [ctrl_qubits]
    for q in ctrl_qubits:
        if q < 0:
            raise ValueError("ctrl_qubits should not be negative value, but get {}!".format(q))
    circ_out = Circuit()
    ctrl_qubits = set(ctrl_qubits)
    for gate in circ:
        intersection = ctrl_qubits.intersection(set(gate.obj_qubits))
        if intersection:
            raise ValueError(
                f"Qubit {intersection} in ctrl_qubits {ctrl_qubits} already used n obj_qubits of gate {gate}")
        curr_ctrl = set(gate.ctrl_qubits)
        curr_ctrl = list(curr_ctrl.union(ctrl_qubits))
        curr_ctrl.sort()
        new_gate = copy.deepcopy(gate)
        if not isinstance(gate, (G.Measure, G.BarrierGate)):
            new_gate.ctrl_qubits = curr_ctrl
        new_gate.generate_description()
        circ_out += new_gate
    return circ_out


def controlled(circuit_fn):
    """
    Add control qubits on a quantum circuit or a quantum operator (a function
    that can generate a quantum circuit)

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.

    Raises:
        TypeError: circuit_fn is not a Circuit or can not return a Circuit.

    Returns:
        function that can generate a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import controlled
        >>> u1 = qft([0, 1])
        >>> u2 = controlled(u1)
        >>> u3 = controlled(qft)
        >>> u3 = u3(2, [0, 1])
        >>> u2(2)
        q0: ──H────PS(π/2)─────────@──
              │       │            │
        q1: ──┼───────●───────H────@──
              │       │       │    │
        q2: ──●───────●───────●────●──
        >>> u3
        q0: ──H────PS(π/2)─────────@──
              │       │            │
        q1: ──┼───────●───────H────@──
              │       │       │    │
        q2: ──●───────●───────●────●──
    """
    from mindquantum.core import Circuit
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(ctrl_qubits, *arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return controlled(circ)
            return _add_ctrl_qubits(circ, ctrl_qubits)

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return lambda ctrl_qubits: _add_ctrl_qubits(circuit_fn, ctrl_qubits)
    raise TypeError("Input need a circuit or a function that can generate a circuit.")


def dagger(circuit_fn):
    """
    Get the hermitian dagger of a quantum circuit or a quantum operator (a function
    that can generate a quantum circuit)

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.

    Raises:
        TypeError: If circuit_fn is not a Circuit or can not return a Circuit.

    Returns:
        Circuit or a function that can generate Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import dagger
        >>> u1 = qft([0, 1])
        >>> u2 = dagger(u1)
        >>> u3 = dagger(qft)
        >>> u3 = u3([0, 1])
        >>> u2
        q0: ──@─────────PS(-π/2)────H──
              │            │
        q1: ──@────H───────●───────────
        >>> u3
        q0: ──@─────────PS(-π/2)────H──
              │            │
        q1: ──@────H───────●───────────
    """
    from mindquantum.core import Circuit
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return dagger(circ)
            return circ.hermitian()

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return circuit_fn.hermitian()
    raise TypeError("circuit_fn need a circuit or a function that can generate a circuit.")


def _apply_circuit(circ, qubits):
    """Apply a circuit to other different qubits."""
    from mindquantum.core import Circuit
    old_qubits = set([])
    for g in circ:
        old_qubits.update(g.obj_qubits)
        old_qubits.update(g.ctrl_qubits)
    old_qubits = list(old_qubits)
    old_qubits.sort()
    if len(old_qubits) != len(qubits):
        raise ValueError(f"Can not apply a {len(old_qubits)} qubits unit to {len(qubits)} qubits circuit.")
    qubits_map = dict(zip(old_qubits, qubits))
    out = Circuit()
    for g in circ:
        g = copy.deepcopy(g)
        g.obj_qubits = [qubits_map[i] for i in g.obj_qubits]
        g.ctrl_qubits = [qubits_map[i] for i in g.ctrl_qubits]
        g.generate_description()
        out += g
    return out


def apply(circuit_fn, qubits):
    """
    Apply a quantum circuit or a quantum operator (a function
    that can generate a quantum circuit) to different qubits.

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.
        qubits (list[int]): The new qubits that you want to apply.

    Raises:
        TypeError: If qubits is not a list.
        ValueError: If any element of qubits is negative.
        TypeError: If circuit_fn is not Circuit or can not return a Circuit.

    Returns:
        Circuit or a function that can generate a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import apply
        >>> u1 = qft([0, 1])
        >>> u2 = apply(u1, [1, 0])
        >>> u3 = apply(qft, [1, 0])
        >>> u3 = u3([0, 1])
        >>> u2
        q0: ──────────●───────H────@──
                      │            │
        q1: ──H────PS(π/2)─────────@──
        >>> u3
        q0: ──────────●───────H────@──
                      │            │
        q1: ──H────PS(π/2)─────────@──
    """
    from mindquantum.core import Circuit
    if not isinstance(qubits, list):
        raise TypeError(f"qubits need a list, but get {type(qubits)}!")
    if len(qubits) > 1:
        for index, q in enumerate(qubits[1:]):
            if q < 0 or qubits[index] < 0:
                raise ValueError(f"Qubit index can not negative!")
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


def _add_prefix(circ, prefix):
    """Add prefix to every parameters in circuit."""
    from mindquantum.core import Circuit
    from mindquantum.core import ParameterResolver as PR
    out = Circuit()
    for g in circ:
        g = copy.deepcopy(g)
        if g.parameterized:
            pr = PR()
            for k, v in g.coeff.items():
                pr[f'{prefix}_{k}'] = v
            g.coeff = pr
            g.generate_description()
        out += g
    return out


def add_prefix(circuit_fn, prefix):
    """
    Add a prefix on the parameter of a parameterized quantum circuit or a parameterized
    quantum operator (a function that can generate a parameterized quantum circuit).

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.
        prefix (str): The prefix you want to add to every parameters.

    Raises:
        TypeError: If prefix is not a string.
        TypeError: circuit_fn is not a Circuit or can not return a Circuit.

    Returns:
        Circuit or a function that can generate a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import add_prefix
        >>> from mindquantum import RX, H, Circuit
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = add_prefix(u1, 'ansatz')
        >>> u3 = add_prefix(u, 'ansatz')
        >>> u3 = u3(0)
        >>> u2
        q0: ──H────RX(ansatz_a)──
        >>> u3
        q0: ──H────RX(ansatz_a)──
    """
    from mindquantum.core import Circuit
    if not isinstance(prefix, str):
        raise TypeError(f"prefix need string, but get {type(prefix)}")
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return add_prefix(circ, prefix)
            return _add_prefix(circ, prefix)

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return _add_prefix(circuit_fn, prefix)
    raise TypeError("circuit_fn need a circuit or a function that can generate a circuit.")


def _change_param_name(circ, name_map):
    """Change the parameter of circuit according to the name map."""
    from mindquantum.core import Circuit
    from mindquantum.core import ParameterResolver as PR
    out = Circuit()
    for g in circ:
        g = copy.deepcopy(g)
        if g.parameterized:
            pr = PR()
            for k, v in g.coeff.items():
                if k not in name_map:
                    raise KeyError(f"Original parameter {k} not in name_map!")
                pr[name_map[k]] = v
            g.coeff = pr
            g.generate_description()
        out += g
    return out


def change_param_name(circuit_fn, name_map):
    """
    Change the parameter name of a parameterized quantum circuit or a parameterized
    quantum operator (a function that can generate a parameterized quantum circuit).

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.
        name_map (dict): The parameter name mapping dict.

    Raises:
        TypeError: If name_map is not a map.
        TypeError: If key of name_map is not string.
        TypeError: If value of name_map is not string.
        TypeError: If circuit_fn is not a Circuit or can not return a Circuit.

    Returns:
        Circuit or a function that can generate a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import change_param_name
        >>> from mindquantum import RX, H, Circuit
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = change_param_name(u1, {'a': 'b'})
        >>> u3 = change_param_name(u, {'a': 'b'})
        >>> u3 = u3(0)
        >>> u2
        q0: ──H────RX(b)──
        >>> u3
        q0: ──H────RX(b)──
    """
    from mindquantum.core import Circuit
    if not isinstance(name_map, dict):
        raise TypeError(f"name_map need map, but get {type(name_map)}")
    for k, v in name_map.items():
        if not isinstance(k, str):
            raise TypeError(f"key of name_map need a string, but get {k}, which is {type(k)}")
        if not isinstance(v, str):
            raise TypeError(f"value of name_map need a string, but get {v}, which is {type(v)}")
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return change_param_name(circ, name_map)
            return _change_param_name(circ, name_map)

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return _change_param_name(circuit_fn, name_map)
    raise TypeError("circuit_fn need a circuit or a function that can generate a circuit.")


C = controlled
D = dagger
A = apply
AP = add_prefix
CPN = change_param_name
