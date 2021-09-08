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
"""Module circuit"""

from collections.abc import Iterable
import numpy as np
from mindquantum.gate import ParameterGate
from mindquantum.gate import SWAP
from mindquantum.gate.basic import _check_gate_type
from .circuit import Circuit


def _is_parameterized_gate_class(gate_class):
    if not hasattr(gate_class, 'isparameter') and issubclass(
            gate_class, ParameterGate):
        return True
    return False


class UN(Circuit):
    """
    Map a quantum gate to different objective qubits and control qubits.

    Args:
        gate (BasicGate): A quantum gate.
        maps_obj (Union[int, list[int]]): Objective qubits. If a int is given and maps_obj is None,
            then the gate will act on qubit from 0 to this int.
        coeff (Union[str, List[str], numbers.Number, List[numbers.Number]]): The parameters for
            gate (if it is a parameterized gate).
        maps_ctrl (Union[int, list[int]]): Control qubits. Default: None.

    Returns:
        Circuit, Return a quantum circuit.

    Examples:
        >>> from mindquantum import UN, RX, H, X, SWAP
        >>> circuit1 = UN(X, maps_obj = [0, 1], maps_ctrl = [2, 3])
        >>> print(circuit1)
        X(0 <-: 2)
        X(1 <-: 3)
        >>> circuit2 = UN(SWAP, maps_obj =[[0, 1], [2, 3]])
        >>> print(circuit2)
        SWAP(0 1)
        SWAP(2 3)
        >>> circuit3 = UN(H, 3)
        >>> print(circuit3)
        H(0)
        H(1)
        H(2)
        >>> circuit4 = UN(RX, 2, 'a')
        >>> print(circuit4)
        RX(a_0|0)
        RX(a_1|1)
        >>> circuit5 = UN(RX, 2, ['a', 'b'])
        >>> print(circuit5)
        RX(a|0)
        RX(b|1)
        >>> circuit6 = UN(RX('a'), [0, 3])
        >>> print(circuit6)
        RX(a|0)
        RX(a|3)
    """
    def __init__(self, gate, maps_obj, coeff=None, maps_ctrl=None):
        self._check_gate_and_coeff(gate, coeff)
        objs, ctrls = self._get_objs_ctrls(maps_obj, maps_ctrl)
        if isinstance(coeff, (str, int, float)):
            if isinstance(coeff, str):
                coeff = [f'{coeff}_{i}' for i, _ in enumerate(objs)]
            else:
                coeff = [coeff for i in objs]
        elif isinstance(coeff, Iterable):
            if len(coeff) != len(objs):
                raise ValueError(
                    f"coeff size of correct, need {len(objs)}, but get {len(coeff)}"
                )
        if _is_parameterized_gate_class(gate):
            gates = [gate(i).on(j, k) for i, j, k in zip(coeff, objs, ctrls)]
        else:
            gates = [gate.on(i, j) for i, j in zip(objs, ctrls)]
        Circuit.__init__(self, gates)

    def _check_gate_and_coeff(self, gate, coeff):
        _check_gate_type(gate)
        if _is_parameterized_gate_class(gate) and coeff is None:
            raise ValueError(
                "Given a parameterized gate without coeff specified")
        if not _is_parameterized_gate_class(gate) and not coeff is None:
            raise ValueError("Non parameterized gate do not need coeff")

    def _get_objs_ctrls(self, maps_obj, maps_ctrl):
        """get objs and ctrls"""
        if isinstance(maps_obj, int):
            if isinstance(maps_ctrl, int):
                raise ValueError(
                    "You do not need UN for both obj and ctrl qubit are single value"
                )
            objs = range(maps_obj)
            ctrls = [None for i in objs]
        elif isinstance(maps_obj, Iterable):
            objs = [i for i in maps_obj]
            if maps_ctrl is None:
                ctrls = [None for i in objs]
            elif isinstance(maps_ctrl, Iterable):
                if len(maps_ctrl) != len(maps_obj):
                    raise ValueError(
                        "size of obj qubits and ctrl qubits not match")
                ctrls = [i for i in maps_ctrl]
            else:
                raise ValueError(
                    f"maps_ctrl need a Iterable type, but get {type(maps_ctrl)}"
                )
        else:
            raise ValueError(
                f"maps_obj need a Iterable or a int, but get {type(maps_obj)}")
        return objs, ctrls


class SwapParts(Circuit):
    """
    Swap two different part of quantum circuit, with or without control qubits.

    Args:
        a (Iterable): The first part you need to swap.
        b (Iterable): The second part you need to swap.
        maps_ctrl (int, Iterable): Control the swap by a single qubit or by
            different qubits or just no control qubit. Default: None.

    Examples:
        >>> from mindquantum import SwapParts
        >>> SwapParts([1, 2], [3, 4], 0)
        SWAP(1 3 <-: 0)
        SWAP(2 4 <-: 0)
    """
    def __init__(self, a: Iterable, b: Iterable, maps_ctrl=None):
        if not isinstance(a, Iterable) or not isinstance(b, Iterable):
            raise Exception("Swap part should be iterable!")
        maps = [[a[i], b[i]] for i in range(len(a))]
        if isinstance(maps_ctrl, int):
            maps_ctrl = [maps_ctrl for _ in maps]
        Circuit.__init__(self, UN(SWAP, maps, maps_ctrl=maps_ctrl))


class U3(Circuit):
    """
    This circuit represent arbitrary single qubit gate.

    Args:
        a (Union[numbers.Number, dict, ParameterResolver]): First parameter for U3 circuit.
        b (Union[numbers.Number, dict, ParameterResolver]): Second parameter for U3 circuit.
        c (Union[numbers.Number, dict, ParameterResolver]): Third parameter for U3 circuit.
        obj_qubit (int): Which qubit the U3 circuit will act on. Default: None.

    Examples:
        >>> from mindquantum import U3
        >>> U3('a','b','c')
        RZ(a|0)
        RX(-1.571|0)
        RZ(b|0)
        RX(1.571|0)
        RZ(c|0)
    """
    def __init__(self, a, b, c, obj_qubit=None):
        if obj_qubit is None:
            obj_qubit = 0
        circ = Circuit()
        circ.rz(a, obj_qubit)
        circ.rx(-np.pi / 2, obj_qubit)
        circ.rz(b, obj_qubit)
        circ.rx(np.pi / 2, obj_qubit)
        circ.rz(c, obj_qubit)
        Circuit.__init__(self, circ)
