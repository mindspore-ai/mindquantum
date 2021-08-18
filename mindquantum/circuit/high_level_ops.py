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
"""High level circuit operators."""

from types import FunctionType, MethodType
import copy

from mindquantum.circuit import Circuit
from mindquantum.parameterresolver import ParameterResolver as PR


def _add_ctrl_qubits(circ, ctrl_qubits):
    """Add control qubits on a circuit."""
    if not isinstance(ctrl_qubits, (int, list)):
        raise TypeError(
            "Require a int or a list of int for ctrl_qubits, but get {}!".
            format(type(ctrl_qubits)))
    if isinstance(ctrl_qubits, int):
        ctrl_qubits = [ctrl_qubits]
    for q in ctrl_qubits:
        if q < 0:
            raise ValueError(
                "ctrl_qubits should not be negative value, but get {}!".format(
                    q))
    circ_out = Circuit()
    ctrl_qubits = set(ctrl_qubits)
    for gate in circ:
        intersection = ctrl_qubits.intersection(set(gate.obj_qubits))
        if intersection:
            raise ValueError(
                f"Qubit {intersection} in ctrl_qubits {ctrl_qubits} already used n obj_qubits of gate {gate}"
            )
        curr_ctrl = set(gate.ctrl_qubits)
        curr_ctrl = list(curr_ctrl.union(ctrl_qubits))
        curr_ctrl.sort()
        new_gate = copy.deepcopy(gate)
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

    Examples:
        >>> from mindquantum.circuit import qft, controlled
        >>> u1 = qft([0, 1])
        >>> u2 = controlled(u1)
        >>> u3 = controlled(qft)
        >>> u3 = u3(2, [0, 1])
        >>> u2(2)
        H(0 <-: 2)
        PS(1.571|0 <-: 1 2)
        H(1 <-: 2)
        SWAP(0 1 <-: 2)
        >>> u3
        H(0 <-: 2)
        PS(1.571|0 <-: 1 2)
        H(1 <-: 2)
        SWAP(0 1 <-: 2)
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
    raise TypeError(
        "Input need a circuit or a function that can generate a circuit.")


def dagger(circuit_fn):
    """
    Get the hermitian dagger of a quantum circuit or a quantum operator (a function
    that can generate a quantum circuit)

    Args:
        circuit_fn (Union[Circuit, FunctionType, MethodType]): A quantum circuit,
            or a function that can generate a quantum circuit.

    Examples:
        >>> from mindquantum.circuit import qft, dagger
        >>> u1 = qft([0, 1])
        >>> u2 = dagger(u1)
        >>> u3 = dagger(qft)
        >>> u3 = u3([0, 1])
        >>> u2
        SWAP(0 1)
        H(1)
        PS(-1.571|0 <-: 1)
        H(0)
        >>> u3
        SWAP(0 1)
        H(1)
        PS(-1.571|0 <-: 1)
        H(0)
    """
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return dagger(circ)
            return circ.hermitian

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return circuit_fn.hermitian
    raise TypeError(
        "Input need a circuit or a function that can generate a circuit.")


def _apply_circuit(circ, qubits):
    """Apply a circuit to other different qubits."""
    old_qubits = set([])
    for g in circ:
        old_qubits.update(g.obj_qubits)
        old_qubits.update(g.ctrl_qubits)
    old_qubits = list(old_qubits)
    old_qubits.sort()
    if len(old_qubits) != len(qubits):
        raise ValueError(
            f"Can not apply a {len(old_qubits)} qubits unit to {len(qubits)} qubits circuit."
        )
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

    Examples:
        >>> from mindquantum.circuit import qft, apply
        >>> u1 = qft([0, 1])
        >>> u2 = apply(u1, [1, 2])
        >>> u3 = apply(qft, [1, 2])
        >>> u3 = u3([0, 1])
        >>> u2
        H(1)
        PS(1.571|1 <-: 2)
        H(2)
        SWAP(1 2)
        >>> u3
        H(1)
        PS(1.571|1 <-: 2)
        H(2)
        SWAP(1 2)
    """
    if not isinstance(qubits, list):
        raise TypeError(f"New qubits need a list, but get {type(qubits)}!")
    if len(qubits) > 1:
        for index, q in enumerate(qubits[1:]):
            if q < 0 or qubits[index] < 0:
                raise ValueError(f"Qubit index can not negative!")
            if q <= qubits[index]:
                raise ValueError(f"Qubits should be in ascending order!")
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return apply(circ, qubits)
            return _apply_circuit(circ, qubits)

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return _apply_circuit(circuit_fn, qubits)
    raise TypeError(
        "Input need a circuit or a function that can generate a circuit.")


def _add_prefix(circ, prefix):
    """Add prefix to every parameters in circuit."""
    out = Circuit()
    for g in circ:
        g = copy.deepcopy(g)
        if g.isparameter:
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

    Examples:
        >>> from mindquantum.circuit import qft, add_prefix
        >>> from mindquantum import RX, H, Circuit
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = add_prefix(u1, 'ansatz')
        >>> u3 = add_prefix(u, 'ansatz')
        >>> u3 = u3(0)
        >>> u2
        H(0)
        RX(ansatz_a|0)
        >>> u3
        H(0)
        RX(ansatz_a|0)
    """
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
    raise TypeError(
        "Input need a circuit or a function that can generate a circuit.")


def _change_param_name(circ, name_map):
    """Change the parameter of circuit according to the name map."""
    out = Circuit()
    for g in circ:
        g = copy.deepcopy(g)
        if g.isparameter:
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

    Examples:
        >>> from mindquantum.circuit import qft, change_param_name
        >>> from mindquantum import RX, H, Circuit
        >>> u = lambda qubit: Circuit([H.on(0), RX('a').on(qubit)])
        >>> u1 = u(0)
        >>> u2 = change_param_name(u1, {'a': 'b'})
        >>> u3 = change_param_name(u, {'a': 'b'})
        >>> u3 = u3(0)
        >>> u2
        H(0)
        RX(b|0)
        >>> u3
        H(0)
        RX(b|0)
    """
    if not isinstance(name_map, dict):
        raise TypeError(
            f"Parameters name map need map, but get {type(name_map)}")
    for k, v in name_map.items():
        if not isinstance(k, str):
            raise ValueError(
                f"Parameter need a string, but get {k}, which is {type(k)}")
        if not isinstance(v, str):
            raise ValueError(
                f"Parameter need a string, but get {v}, which is {type(v)}")
    if isinstance(circuit_fn, (FunctionType, MethodType)):

        def wrapper(*arg, **keywords):
            circ = circuit_fn(*arg, **keywords)
            if not isinstance(circ, Circuit):
                return change_param_name(circ, name_map)
            return _change_param_name(circ, name_map)

        return wrapper
    if isinstance(circuit_fn, Circuit):
        return _change_param_name(circuit_fn, name_map)
    raise TypeError(
        "Input need a circuit or a function that can generate a circuit.")


C = controlled
D = dagger
A = apply
AP = add_prefix
CPN = change_param_name
