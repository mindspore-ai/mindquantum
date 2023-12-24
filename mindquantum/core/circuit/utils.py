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
"""Tools for MindQuantum eDSL."""

import copy
from types import FunctionType, MethodType

import numpy as np

from mindquantum.core.gates.basic import ParameterGate
from mindquantum.utils.type_value_check import _check_input_type

from ..parameterresolver.parameterresolver import ParameterResolver
from .circuit import Circuit, apply


def decompose_single_term_time_evolution(term, para):  # pylint: disable=too-many-branches
    r"""
    Decompose a time evolution gate into basic quantum gates.

    This function only works for the hamiltonian with only single pauli word.
    For example, :math:`exp^{-it\text{ham}}`, :math:`\text{ham}` can only be a single pauli word, such
    as :math:`\text{ham}=X_0 Y_1 Z_2`, and at this time, term will be
    ((0, 'X'), (1, 'Y'), (2, 'Z')). When the evolution time is expressed as
    :math:`t=ax+by`, para would be {'x':a, 'y':b}.

    Args:
        term (tuple, QubitOperator): the hamiltonian term of just the
            evolution qubit operator.
        para (Union[dict, numbers.Number]): the parameters of evolution operator.

    Returns:
        Circuit, a quantum circuit.

    Raises:
        ValueError: If `term` has more than one pauli string.
        TypeError: If `term` is not map.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum.core.circuit import decompose_single_term_time_evolution
        >>> ham = QubitOperator('X0 Y1')
        >>> circuit = decompose_single_term_time_evolution(ham, {'a':1})
        >>> print(circuit)
              ┏━━━┓                               ┏━━━┓
        q0: ──┨ H ┠─────────■─────────────────■───┨ H ┠──────────
              ┗━━━┛         ┃                 ┃   ┗━━━┛
              ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━┓ ┏━┻━┓ ┏━━━━━━━━━━┓
        q1: ──┨ RX(π/2) ┠─┨╺╋╸┠─┨ RZ(2*a) ┠─┨╺╋╸┠─┨ RX(7π/2) ┠───
              ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━┛ ┗━━━┛ ┗━━━━━━━━━━┛
    """
    # pylint: disable=import-outside-toplevel,cyclic-import
    from mindquantum.core import gates
    from mindquantum.utils.type_value_check import _num_type

    if not isinstance(term, tuple):
        try:
            if len(term.terms) != 1:
                raise ValueError(f"Only work for single term time evolution operator, but get {len(term)}")
            term = list(term.terms.keys())[0]
        except TypeError as exc:
            raise TypeError(f"Not supported type:{type(term)}") from exc
    if not isinstance(para, _num_type):
        if not isinstance(para, (dict, ParameterResolver)):
            raise TypeError(f'para requires a number or a dict or a ParameterResolver, but get {type(para)}')
        para = ParameterResolver(para)
    para = 2 * ParameterResolver(para)
    out = []
    term = sorted(term)
    rxs = []
    if not term:
        raise ValueError("Get constant hamiltonian, please use GlobalPhase gate and give the obj_qubit by yourself.")
    if len(term) == 1:  # single pauli operator
        if term[0][1] == 'X':
            out.append(gates.RX(para).on(term[0][0]))
        elif term[0][1] == 'Y':
            out.append(gates.RY(para).on(term[0][0]))
        else:
            out.append(gates.RZ(para).on(term[0][0]))
    else:
        for index, action in term:
            if action == 'X':
                out.append(gates.H.on(index))
            elif action == 'Y':
                rxs.append(len(out))
                out.append(gates.RX(np.pi / 2).on(index))

        out.append(gates.BarrierGate(False))
        for i in range(len(term) - 1):
            out.append(gates.X.on(term[i + 1][0], term[i][0]))
        out.append(gates.BarrierGate(False))
        out.append(gates.RZ(para).on(term[-1][0]))
        for i in range(len(out) - 1)[::-1]:
            if i in rxs:
                out.append(gates.RX(np.pi * 3.5).on(out[i].obj_qubits))
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
        ValueError: If qubitops has more than one pauli words.

    Examples:
        >>> from mindquantum.core import X
        >>> from mindquantum.core.operators import QubitOperator
        >>> from mindquantum.core.circuit import pauli_word_to_circuits
        >>> qubitops = QubitOperator('X0 Y1')
        >>> pauli_word_to_circuits(qubitops) + X(1, 0)
              ┏━━━┓
        q0: ──┨╺╋╸┠───■─────
              ┗━━━┛   ┃
              ┏━━━┓ ┏━┻━┓
        q1: ──┨ Y ┠─┨╺╋╸┠───
              ┗━━━┛ ┗━━━┛
    """
    # pylint: disable=import-outside-toplevel
    from mindquantum import operators as ops
    from mindquantum.core import gates

    allow_ops = (ops.QubitOperator, ops.Hamiltonian)
    if not isinstance(qubitops, allow_ops):
        raise TypeError(f"qubitops require a QubitOperator or a Hamiltonian, but get {type(qubitops)}!")
    if isinstance(qubitops, ops.Hamiltonian):
        if qubitops.how_to != 0:
            raise ValueError("Hamiltonian should be in origin mode.")
        qubitops = qubitops.hamiltonian
    if len(qubitops.terms) > 1:
        raise ValueError("Only work for QubitOperator with single pauli word!")
    gate_map = {'X': gates.X, 'Y': gates.Y, 'Z': gates.Z}
    for operator in qubitops.terms.keys():
        circ = Circuit()
        if operator:
            for ind, single_op in operator:
                circ += gate_map.get(single_op).on(ind)
        else:
            circ += gates.I.on(0)
    return circ


def _add_ctrl_qubits(circ, ctrl_qubits):
    """Add control qubits on a circuit."""
    # pylint: disable=import-outside-toplevel,cyclic-import
    from mindquantum.core import gates

    if not isinstance(ctrl_qubits, (int, list)):
        raise TypeError(f"Require a int or a list of int for ctrl_qubits, but get {type(ctrl_qubits)}!")
    if isinstance(ctrl_qubits, int):
        ctrl_qubits = [ctrl_qubits]
    for qubit in ctrl_qubits:
        if qubit < 0:
            raise ValueError(f"ctrl_qubits should not be negative value, but get {qubit}!")
    circ_out = Circuit()
    ctrl_qubits_set = set(ctrl_qubits)
    for gate in circ:
        intersection = ctrl_qubits_set.intersection(set(gate.obj_qubits))
        if intersection:
            raise ValueError(
                f"Qubit {intersection} in ctrl_qubits {ctrl_qubits_set} already used in obj_qubits of gate {gate}"
            )
        curr_ctrl = set(gate.ctrl_qubits)
        curr_ctrl = list(curr_ctrl.union(ctrl_qubits_set))
        curr_ctrl.sort()
        new_gate = copy.deepcopy(gate)
        if not isinstance(gate, (gates.Measure, gates.BarrierGate)):
            new_gate.ctrl_qubits = curr_ctrl
        circ_out += new_gate
    return circ_out


def controlled(circuit_fn):
    """
    Add control qubits on a quantum circuit or a quantum operator.

    (a function that can generate a quantum circuit)

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit, or a function that can generate a
            quantum circuit.

    Returns:
        function that can generate a Circuit.

    Raises:
        TypeError: `circuit_fn` is not a Circuit or can not return a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import controlled
        >>> u1 = qft([0, 1])
        >>> u2 = controlled(u1)
        >>> u3 = controlled(qft)
        >>> u3 = u3(2, [0, 1])
        >>> u2(2)
              ┏━━━┓ ┏━━━━━━━━━┓
        q0: ──┨ H ┠─┨ PS(π/2) ┠───────╳───
              ┗━┳━┛ ┗━━━━┳━━━━┛       ┃
                ┃        ┃      ┏━━━┓ ┃
        q1: ────╂────────■──────┨ H ┠─╳───
                ┃        ┃      ┗━┳━┛ ┃
                ┃        ┃        ┃   ┃
        q2: ────■────────■────────■───■───
        >>> u3
              ┏━━━┓ ┏━━━━━━━━━┓
        q0: ──┨ H ┠─┨ PS(π/2) ┠───────╳───
              ┗━┳━┛ ┗━━━━┳━━━━┛       ┃
                ┃        ┃      ┏━━━┓ ┃
        q1: ────╂────────■──────┨ H ┠─╳───
                ┃        ┃      ┗━┳━┛ ┃
                ┃        ┃        ┃   ┃
        q2: ────■────────■────────■───■───
    """
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
    Get the hermitian dagger of a quantum circuit or a quantum operator.

    (a function that can generate a quantum circuit)

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit, or a function that can generate a
            quantum circuit.

    Returns:
        Circuit or a function that can generate Circuit.

    Raises:
        TypeError: If `circuit_fn` is not a Circuit or can not return a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import dagger
        >>> u1 = qft([0, 1])
        >>> u2 = dagger(u1)
        >>> u3 = dagger(qft)
        >>> u3 = u3([0, 1])
        >>> u2
                      ┏━━━━━━━━━━┓ ┏━━━┓
        q0: ──╳───────┨ PS(-π/2) ┠─┨ H ┠───
              ┃       ┗━━━━━┳━━━━┛ ┗━━━┛
              ┃ ┏━━━┓       ┃
        q1: ──╳─┨ H ┠───────■──────────────
                ┗━━━┛
        >>> u3
                      ┏━━━━━━━━━━┓ ┏━━━┓
        q0: ──╳───────┨ PS(-π/2) ┠─┨ H ┠───
              ┃       ┗━━━━━┳━━━━┛ ┗━━━┛
              ┃ ┏━━━┓       ┃
        q1: ──╳─┨ H ┠───────■──────────────
                ┗━━━┛
    """
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


def _add_prefix_or_suffix(circ: Circuit, fix: str, is_prefix: bool):
    """Add prefix to every parameters in circuit."""
    out = Circuit()
    gate: ParameterGate
    for gate in circ:
        if gate.parameterized:
            new_prs = []
            for coeff in gate.get_parameters():
                origin_encoder = coeff.encoder_parameters
                origin_req_grad = coeff.requires_grad_parameters
                pr = ParameterResolver(coeff.const)
                for k, v in dict(coeff).items():
                    new_name = ''
                    if is_prefix:
                        new_name = f'{fix}_{k}'
                    else:
                        new_name = f'{k}_{fix}'
                    pr[new_name] = v
                    if k in origin_encoder:
                        pr.encoder_part(new_name)
                    if k in origin_req_grad:
                        pr.requires_grad_part(new_name)
                new_prs.append(pr)
            out += gate(*new_prs).on(gate.obj_qubits, gate.ctrl_qubits)
        else:
            out += copy.deepcopy(gate)
    return out


def add_prefix_or_suffix(circuit_fn, fix: str, is_prefix: bool):
    """Add prefix or suffix of the parameters of quantum circuit."""
    if not isinstance(fix, str):
        raise TypeError(f"prefix or suffix need string, but get {type(fix)}")
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return add_prefix_or_suffix(circ, fix, is_prefix)
            return _add_prefix_or_suffix(circ, fix, is_prefix)

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return _add_prefix_or_suffix(circuit_fn, fix, is_prefix)
    raise TypeError("circuit_fn need a circuit or a function that can generate a circuit.")


def add_prefix(circuit_fn, prefix: str):
    """
    Add a prefix on the parameter of a parameterized quantum circuit or a parameterized quantum operator.

    (a function that can generate a parameterized quantum circuit).

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.
        prefix (str): The prefix you want to add to every parameters.

    Returns:
        Circuit or a function that can generate a Circuit.

    Raises:
        TypeError: If `prefix` is not a string.
        TypeError: `circuit_fn` is not a Circuit or can not return a Circuit.

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
              ┏━━━┓ ┏━━━━━━━━━━━━━━┓
        q0: ──┨ H ┠─┨ RX(ansatz_a) ┠───
              ┗━━━┛ ┗━━━━━━━━━━━━━━┛
        >>> u3
              ┏━━━┓ ┏━━━━━━━━━━━━━━┓
        q0: ──┨ H ┠─┨ RX(ansatz_a) ┠───
              ┗━━━┛ ┗━━━━━━━━━━━━━━┛
    """
    return add_prefix_or_suffix(circuit_fn, prefix, True)


def add_suffix(circuit_fn, suffix: str):
    """
    Add a suffix on the parameter of a parameterized quantum circuit or a parameterized quantum operator.

    (a function that can generate a parameterized quantum circuit).

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.
        suffix (str): The suffix you want to add to every parameters.

    Returns:
        Circuit or a function that can generate a Circuit.

    Raises:
        TypeError: If suffix is not a string.
        TypeError: circuit_fn is not a Circuit or can not return a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import add_suffix
        >>> from mindquantum import RX, H, Circuit
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = add_suffix(u1, '1')
        >>> u3 = add_suffix(u, '1')
        >>> u3 = u3(0)
        >>> u2
              ┏━━━┓ ┏━━━━━━━━━┓
        q0: ──┨ H ┠─┨ RX(a_1) ┠───
              ┗━━━┛ ┗━━━━━━━━━┛
        >>> u3
              ┏━━━┓ ┏━━━━━━━━━┓
        q0: ──┨ H ┠─┨ RX(a_1) ┠───
              ┗━━━┛ ┗━━━━━━━━━┛
    """
    return add_prefix_or_suffix(circuit_fn, suffix, False)


def shift(circ, inc):
    """
    Shift the qubit range of the given circuit.

    Args:
        circ (circuit): The circuit that you want to do shift operator.
        inc (int): The qubit distance you want to shift.

    Examples:
        >>> from mindquantum.core.circuit import shift, Circuit
        >>> circ = Circuit().x(1, 0)
        >>> circ
        q0: ────■─────
                ┃
              ┏━┻━┓
        q1: ──┨╺╋╸┠───
              ┗━━━┛
        >>> shift(circ, 1)
        q1: ────■─────
                ┃
              ┏━┻━┓
        q2: ──┨╺╋╸┠───
              ┗━━━┛

    Returns:
        Circuit, the shifted circuit.
    """
    _check_input_type('circ', Circuit, circ)
    _check_input_type('p', int, inc)
    return apply(circ, [i + inc for i in sorted(circ.all_qubits.keys())])


def _change_param_name(circ, name_map):
    """Change the parameter of circuit according to the name map."""
    out = Circuit()
    gate: ParameterGate
    for gate in circ:
        if gate.parameterized:
            new_prs = []
            for coeff in gate.get_parameters():
                origin_encoder = coeff.encoder_parameters
                origin_req_grad = coeff.requires_grad_parameters
                pr = ParameterResolver(coeff.const)
                for k, v in dict(coeff).items():
                    if k not in name_map:
                        raise KeyError(f"Original parameter {k} not in name_map!")
                    pr[name_map[k]] = v
                    if k in origin_encoder:
                        pr.encoder_part(name_map[k])
                    if k in origin_req_grad:
                        pr.requires_grad_part(name_map[k])
                new_prs.append(pr)
            out += gate(*new_prs).on(gate.obj_qubits, gate.ctrl_qubits)
        else:
            out += copy.deepcopy(gate)
    return out


def change_param_name(circuit_fn, name_map):
    """
    Change the parameter name of a parameterized quantum circuit or a parameterized quantum operator.

    (a function that can generate a parameterized quantum circuit).

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.
        name_map (dict): The parameter name mapping dict.

    Returns:
        Circuit or a function that can generate a Circuit.

    Raises:
        TypeError: If `name_map` is not a map.
        TypeError: If key of `name_map` is not string.
        TypeError: If value of `name_map` is not string.
        TypeError: If `circuit_fn` is not a Circuit or can not return a Circuit.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.core.circuit import change_param_name, Circuit
        >>> from mindquantum.core.gates import RX, H
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = change_param_name(u1, {'a': 'b'})
        >>> u3 = change_param_name(u, {'a': 'b'})
        >>> u3 = u3(0)
        >>> u2
              ┏━━━┓ ┏━━━━━━━┓
        q0: ──┨ H ┠─┨ RX(b) ┠───
              ┗━━━┛ ┗━━━━━━━┛
        >>> u3
              ┏━━━┓ ┏━━━━━━━┓
        q0: ──┨ H ┠─┨ RX(b) ┠───
              ┗━━━┛ ┗━━━━━━━┛
    """
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


def as_encoder(circuit_fn):
    """
    Conversion decorator of a circuit to an encoder circuit.

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A Circuit or a callable function that can return
            a Circuit.

    Returns:
        Function, if `circuit_fn` is a callable function that will return a Circuit.
        Circuit, if `circuit_fn` is already a Circuit.

    Examples:
        >>> from mindquantum.core.circuit import as_encoder, Circuit
        >>> from mindquantum.core.gates import RX
        >>> @as_encoder
        ... def create_circuit():
        ...     circ = Circuit()
        ...     circ += RX('a').on(0)
        ...     return circ
        >>> circ = create_circuit()
        >>> circ.encoder_params_name
        ['a']
        >>> circ.as_ansatz()
        >>> circ.encoder_params_name
        []
        >>> circ = as_encoder(circ)
        >>> circ.encoder_params_name
        ['a']
    """
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def gene_circ(*args, **kwargs):
            circ = circuit_fn(*args, **kwargs)
            if not isinstance(circ, Circuit):
                raise ValueError(f"The callable circuit_fn should return a Circuit, but get {type(circ)}")
            circ.as_encoder()
            return circ

        return gene_circ

    if isinstance(circuit_fn, Circuit):
        new_circ = copy.deepcopy(circuit_fn)
        return new_circ.as_encoder()
    raise TypeError(f"circuit_fn need a circuit or a function that can generate a circuit, but get {type(circuit_fn)}.")


def as_ansatz(circuit_fn):
    """
    Conversion decorator of a circuit to an ansatz circuit.

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A Circuit or a callable function that can return
            a Circuit.

    Returns:
        Function, if `circuit_fn` is a callable function that will return a Circuit.
        Circuit, if `circuit_fn` is already a Circuit.

    Examples:
        >>> from mindquantum.core.circuit import as_ansatz, Circuit
        >>> from mindquantum.core.gates import RX
        >>> @as_ansatz
        ... def create_circuit():
        ...     circ = Circuit()
        ...     circ += RX('a').on(0)
        ...     return circ
        >>> circ = create_circuit()
        >>> circ.ansatz_params_name
        ['a']
        >>> circ.as_encoder()
        >>> circ.ansatz_params_name
        []
        >>> circ = as_ansatz(circ)
        >>> circ.ansatz_params_name
        ['a']
    """
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def gene_circ(*args, **kwargs):
            circ = circuit_fn(*args, **kwargs)
            if not isinstance(circ, Circuit):
                raise ValueError(f"The callable circuit_fn should return a Circuit, but get {type(circ)}")
            circ.as_ansatz()
            return circ

        return gene_circ

    if isinstance(circuit_fn, Circuit):
        new_circ = copy.deepcopy(circuit_fn)
        return new_circ.as_ansatz()
    raise TypeError(f"circuit_fn need a circuit or a function that can generate a circuit, but get {type(circuit_fn)}.")


C = controlled
D = dagger
AP = add_prefix
CPN = change_param_name
