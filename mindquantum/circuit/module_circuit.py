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
from mindquantum.gate import BasicGate
from mindquantum.gate import SWAP
from mindquantum.gate.basic import _check_gate_type
from .circuit import Circuit


class UN(Circuit):
    """
    Map a quantum gate to different objective qubits and control qubits.

    Args:
        gate (BasicGate): A quantum gate.
        maps_obj (Union[int, list[int]]): Objective qubits.
        maps_ctrl (Union[int, list[int]]): Control qubits.

    Returns:
        Circuit, Return a quantum circuit.

    Examples:
        >>> circuit1 = UN(X, maps_obj = [0, 1], maps_ctrl = [2, 3])
        >>> print(circuit1)
        X(0 <-: 2)
        X(1 <-: 3)
        >>> circuit2 = UN(SWAP, maps_obj =[[0, 1], [2, 3]])
        >>> print(circuit2)
        SWAP(0 1)
        SWAP(2 3)
    """
    def __init__(self, gate: BasicGate, maps_obj, maps_ctrl=None):
        _check_gate_type(gate)
        if isinstance(maps_obj, Iterable):
            if maps_ctrl is None:
                gates = [gate.on(i) for i in maps_obj]
            else:
                if isinstance(maps_ctrl, Iterable):
                    gates = [
                        gate.on(maps_obj[i], maps_ctrl[i])
                        for i in range(len(maps_obj))
                    ]
                else:
                    gates = [gate.on(i, maps_ctrl) for i in maps_obj]
        else:
            if maps_ctrl is None:
                gates = [gate.on(i) for i in range(maps_obj)]
            else:
                if isinstance(maps_ctrl, Iterable):
                    gates = [gate.on(maps_obj, i) for i in maps_ctrl]
                else:

                    gates = [gate.on(maps_obj, maps_ctrl)]
        Circuit.__init__(self, gates)


class SwapParts(Circuit):
    """
    Swap two different part of quantum circuit, with or without control qubits.

    Args:
        a (Iterable): The first part you need to swap.
        b (Iterable): The second part you need to swap.
        maps_ctrl (int, Iterable): Control the swap by a single qubit or by
            different qubits or just no control qubit. Default: None.

    Examples:
        >>> SwapParts([1, 2], [3, 4], 0)
        SWAP(1 3 <-: 0)
        SWAP(2 4 <-: 0)
    """
    def __init__(self, a: Iterable, b: Iterable, maps_ctrl=None):
        if not isinstance(a, Iterable) or not isinstance(b, Iterable):
            raise Exception("Swap part should be iterable!")
        maps = [[a[i], b[i]] for i in range(len(a))]
        Circuit.__init__(self, UN(SWAP, maps, maps_ctrl))


class U3(Circuit):
    """
    This circuit represent arbitrary single qubit gate.

    Args:
        a (Union[number.Numbers, dict, ParameterResoler]): First parameter for U3 circuit.
        b (Union[number.Numbers, dict, ParameterResoler]): Second parameter for U3 circuit.
        c (Union[number.Numbers, dict, ParameterResoler]): Third parameter for U3 circuit.
        obj_qubit (int): Which qubit the U3 circuit will act on.

    Examples:
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
