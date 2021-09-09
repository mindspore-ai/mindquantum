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
"""Hardware efficient ansatz."""

import itertools
import numpy as np
from mindquantum.gate import BasicGate, X
from mindquantum.circuit import Circuit, AP, A
from ._ansatz import Ansatz


def _check_single_rot_gate_seq(single_rot_gate_seq):
    """check single rotation gate seq"""
    if not isinstance(single_rot_gate_seq, list):
        raise TypeError(
            f"single_rot_gate_seq requires a list, but get {type(single_rot_gate_seq)}"
        )
    for gate in single_rot_gate_seq:
        if not issubclass(gate, BasicGate):
            raise ValueError(
                f"single rotation gate require a parameterized gate, but get {gate}"
            )
        gate_shape = gate(0).matrix().shape
        if gate_shape[0] != 2 or gate_shape[1] != 2:
            raise ValueError(
                f"single rotation gate should be a one qubit gate.")


class HardwareEfficientAnsatz(Ansatz):
    r"""
    Hardware efficient ansatz is a kind of ansatz that can be easily implement on quantum chip.

    The hardware efficient is constructed by a layer of single qubit rotation gate and a layer
    of two qubits entanglement gate. The single qubit rotation gate layer is constructed by one
    or several rotation gate that act on every qubit. The two qubits entanglement gate layer is
    constructed by CNOT, CZ, XX, YY, ZZ, etc. acting on entangle_mapping. For more detail, please
    refers https://www.nature.com/articles/nature23879.

    Args:
        n_qubits (int): number of qubit that this ansatz act on.
        single_rot_gate_seq (list[BasicGate]): A list of parameterized rotation gate that act on
            each qubit.
        entangle_gate (BasicGate): The non parameterized entanglement gate. If it is a single qubit
            gate, than the control version will be used. Default: XGate.
        entangle_mapping (Union[str, list[tuple[int]]]): The entanglement mapping of entanglement gate.
            'linear' means the entanglement gate will be act on every neighboring qubits. 'all' means
            the entanglemtn gate will be act on any two qbuits. Besides, you can specific which two
            qubits you want to do entanglement by setting the entangle_mapping to a list of two qubits
            tuple. Default: "linear".
        depth (int): Repeat the entanglement gate layer and single_rot_gate_seq in depth times. Default: 1.

    Examples:
        >>> from mindquantum.ansatz import HardwareEfficientAnsatz
        >>> from mindquantum import RY, RZ, Z
        >>> hea = HardwareEfficientAnsatz(3, [RY, RZ], Z, [(1, 0), (2, 0)])
        >>> hea.circuit
        RY(d0_n0_0|0)
        RZ(d0_n0_1|0)
        RY(d0_n1_0|1)
        RZ(d0_n1_1|1)
        RY(d0_n2_0|2)
        RZ(d0_n2_1|2)
        Z(1 <-: 0)
        Z(2 <-: 0)
        RY(d1_n0_0|0)
        RZ(d1_n0_1|0)
        RY(d1_n1_0|1)
        RZ(d1_n1_1|1)
        RY(d1_n2_0|2)
        RZ(d1_n2_1|2)
    """
    def __init__(self,
                 n_qubits,
                 single_rot_gate_seq,
                 entangle_gate=X,
                 entangle_mapping='linear',
                 depth=1):
        _check_single_rot_gate_seq(single_rot_gate_seq)
        if not isinstance(depth, int) or depth <= 0:
            raise ValueError(f"depth requires a positive int, but get {depth}")
        if not isinstance(entangle_gate,
                          BasicGate) or entangle_gate.isparameter:
            raise ValueError(
                f"entangle gate requires a non parameterized gate, but get {entangle_gate}"
            )
        super().__init__('Hardware Efficient', n_qubits, single_rot_gate_seq,
                         entangle_gate, entangle_mapping, depth)

    def _implement(self, single_rot_gate_seq, entangle_gate, entangle_mapping,
                   depth):
        """Implement of hardware efficient ansatz"""
        entangle_mapping = self._get_entangle_mapping(entangle_mapping)
        circ = Circuit()
        for d in range(depth):
            circ += AP(self._build_single_rot(single_rot_gate_seq), f'd{d}')
            circ += self._build_entangle(entangle_gate, entangle_mapping)
        circ += AP(self._build_single_rot(single_rot_gate_seq), f'd{d+1}')
        self._circuit = circ

    def _get_entangle_mapping(self, entangle_mapping):
        """Get entanglement mapping"""
        if isinstance(entangle_mapping, str):
            if entangle_mapping == 'all':
                return list(itertools.combinations(range(self.n_qubits), 2))
            if entangle_mapping == 'linear':
                res = []
                for i in range(self.n_qubits - 1):
                    res.append((i, i + 1))
                return res
            raise ValueError("entangle_mapping can only be 'all', 'linear', \
or a list of tuple of the qubits that the entanglement gate act on.")
        if isinstance(entangle_mapping, list):
            for i in entangle_mapping:
                if isinstance(i, tuple):
                    if len(i) != 2:
                        raise ValueError(
                            f"entanglement only act on two qubits, but get {i}"
                        )
                else:
                    raise TypeError(
                        f"Element of entangle_mapping need a tuple, but get {type(i)}"
                    )
            entangle_mapping = [(j, i) for i, j in entangle_mapping]
        else:
            raise ValueError("entangle_mapping can only be 'all', 'linear', \
or a list of tuple of the qubits that the entanglement gate act on.")
        return entangle_mapping

    def _build_entangle(self, entangle_gate, entangle_mapping):
        """build entanglement layer"""
        gate_qubit = int(np.log2(entangle_gate.matrix().shape[0]))
        circ = Circuit()
        for qs in entangle_mapping:
            if gate_qubit == 1:
                circ += entangle_gate.on(qs[1], qs[0])
            elif gate_qubit == 2:
                circ += entangle_gate.on(qs)
            else:
                raise ValueError(
                    f"Entangle gate can only be a controlled single qubit gate \
or two qubits gate, but get {gate_qubit} qubits gate.")
        return circ

    def _build_single_rot(self, single_rot_gate_seq):
        """build single rotation layer"""
        circ = Circuit()
        single_circ = Circuit()
        for index, gate in enumerate(single_rot_gate_seq):
            single_circ += gate(f'{index}').on(0)
        for n in range(self.n_qubits):
            circ += A(AP(single_circ, f'n{n}'), [n])
        return circ
