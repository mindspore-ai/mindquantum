# Copyright 2023 Huawei Technologies Co., Ltd
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
"""randomized benchmarking module."""
# pylint:disable=no-member
import numpy as np

from mindquantum import _mq_vector
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator, decompose_stabilizer
from mindquantum.simulator.stabilizer import Stabilizer
from mindquantum.utils.type_value_check import (
    _check_int_type,
    _check_seed,
    _check_value_should_not_less,
)


def query_single_qubit_clifford_elem(idx: int) -> Simulator:
    """
    Query a element of single qubit clifford group.

    The size of single qubit clifford group is 24.

    Args:
        idx (int): The index of clifford element in single qubit clifford group.

    Returns:
        :class:`~.simulator.Simulator`, a stabilizer with tableau be the queried clifford element.

    Examples:
        >>> from mindquantum.algorithm.error_mitigation import query_single_qubit_clifford_elem
        >>> from mindquantum.simulator import decompose_stabilizer
        >>> elem = query_single_qubit_clifford_elem(12)
        >>> decompose_stabilizer(elem)
              ┏━━━┓ ┏━━━┓ ┏━━━━┓
        q0: ──┨╺╋╸┠─┨ H ┠─┨ S† ┠───
              ┗━━━┛ ┗━━━┛ ┗━━━━┛
    """
    _check_int_type('idx', idx)
    _check_value_should_not_less("idx", 0, idx)
    clifford = Stabilizer(_mq_vector.stabilizer.query_single_qubit_clifford_elem(idx), n_qubits=1, internal=True)
    sim = Simulator('stabilizer', 1)
    sim.backend = clifford
    return sim


def query_double_qubits_clifford_elem(idx: int) -> Simulator:
    """
    Query a element of double qubits clifford group.

    The size of double qubits clifford group is 11520.

    Args:
        idx (int): The index of clifford element in double qubits clifford group.

    Returns:
        :class:`~.simulator.Simulator`, a stabilizer with tableau be the queried clifford element.

    Examples:
        >>> from mindquantum.algorithm.error_mitigation import query_double_qubits_clifford_elem
        >>> from mindquantum.simulator import decompose_stabilizer
        >>> elem = query_double_qubits_clifford_elem(11111)
        >>> decompose_stabilizer(elem)
              ┏━━━┓ ┏━━━━┓       ┏━━━┓       ┏━━━┓
        q0: ──┨╺╋╸┠─┨ S† ┠───────┨╺╋╸┠───■───┨╺╋╸┠───
              ┗━━━┛ ┗━━━━┛       ┗━┳━┛   ┃   ┗━┳━┛
              ┏━━━┓ ┏━━━━┓ ┏━━━┓   ┃   ┏━┻━┓   ┃
        q1: ──┨ H ┠─┨ S† ┠─┨ H ┠───■───┨╺╋╸┠───■─────
              ┗━━━┛ ┗━━━━┛ ┗━━━┛       ┗━━━┛
    """
    _check_int_type('idx', idx)
    _check_value_should_not_less("idx", 0, idx)
    clifford = Stabilizer(_mq_vector.stabilizer.query_double_qubits_clifford_elem(idx), n_qubits=2, internal=True)
    sim = Simulator('stabilizer', 2)
    sim.backend = clifford
    return sim


def generate_single_qubit_rb_circ(length: int, seed: int = None) -> Circuit:
    """
    Generate a single qubit randomized benchmarking circuit.

    Args:
        length (int): The length of total clifford elements.
        seed (int): The random seed to generate benchmarking circuit. If ``None``, a random seed will be used.
            Default: ``None``.

    Returns:
        :class:`~.core.circuit.Circuit`, the single qubit randomized benchmarking circuit, the quantum
        state of this circuit is zero state.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.error_mitigation import generate_single_qubit_rb_circ
        >>> circ = generate_single_qubit_rb_circ(5, 42)
        >>> circ
              ┏━━━┓ ┏━━━┓   ┏━━━┓   ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓   ┏━━━┓ ┏━━━┓
        q0: ──┨ S ┠─┨ Z ┠─▓─┨ H ┠─▓─┨ H ┠─┨ S ┠─┨ H ┠─┨ Z ┠─┨╺╋╸┠─▓─┨ S ┠─┨ H ┠─↯─
              ┗━━━┛ ┗━━━┛   ┗━━━┛   ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛   ┗━━━┛ ┗━━━┛
              ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓   ┏━━━┓ ┏━━━┓
        q0: ──┨ S ┠─┨ H ┠─┨ Z ┠─┨╺╋╸┠─▓─┨ S ┠─┨ Z ┠─▓───
              ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛   ┗━━━┛ ┗━━━┛
        >>> np.abs(circ.get_qs())
        array([1., 0.])
    """
    _check_int_type('length', length)
    _check_value_should_not_less('length', 0, length)
    if seed is None:
        seed = int(np.random.randint(1, 2 << 20))
    _check_seed(seed)

    stabs = _mq_vector.stabilizer.generate_single_qubit_rb_circ(length, seed)
    circ = Circuit()
    for stab in stabs:
        circ.barrier()
        circ += decompose_stabilizer(Stabilizer(stab, n_qubits=1, internal=True))
    return circ.hermitian()


def generate_double_qubits_rb_circ(length: int, seed: int = None) -> Circuit:
    """
    Generate a double qubits randomized benchmarking circuit.

    Args:
        length (int): The length of total clifford elements.
        seed (int): The random seed to generate benchmarking circuit. If ``None``, a random seed will be used.
            Default: ``None``.

    Returns:
        :class:`~.core.circuit.Circuit`, the double qubit randomized benchmarking circuit, the quantum state of
        this circuit is zero state.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.error_mitigation import generate_double_qubits_rb_circ
        >>> circ = generate_double_qubits_rb_circ(3, 42)
        >>> circ
              ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓         ┏━━━┓ ┏━━━┓
        q0: ──┨╺╋╸┠─┨ S ┠─┨╺╋╸┠─┨ H ┠─┨ S ┠─┨ H ┠─┨ Z ┠─┨╺╋╸┠─▓───■───┨╺╋╸┠─┨ H ┠─↯─
              ┗━┳━┛ ┗━━━┛ ┗━┳━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ▓   ┃   ┗━┳━┛ ┗━━━┛
                ┃           ┃   ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓       ▓ ┏━┻━┓   ┃
        q1: ────■───────────■───┨ S ┠─┨ H ┠─┨ Z ┠─┨╺╋╸┠───────▓─┨╺╋╸┠───■─────────↯─
                                ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛         ┗━━━┛
                    ┏━━━┓                           ┏━━━┓       ┏━━━┓ ┏━━━┓
        q0: ────■───┨ H ┠─────────────────────────▓─┨╺╋╸┠───■───┨ S ┠─┨ H ┠───■───↯─
                ┃   ┗━━━┛                         ▓ ┗━┳━┛   ┃   ┗━━━┛ ┗━━━┛   ┃
              ┏━┻━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ▓   ┃   ┏━┻━┓             ┏━┻━┓
        q1: ──┨╺╋╸┠─┨ H ┠─┨ S ┠─┨ H ┠─┨ Z ┠─┨╺╋╸┠─▓───■───┨╺╋╸┠─────────────┨╺╋╸┠─↯─
              ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛         ┗━━━┛             ┗━━━┛
              ┏━━━┓ ┏━━━┓
        q0: ──┨ H ┠─┨ Z ┠───────────────────▓───
              ┗━━━┛ ┗━━━┛                   ▓
              ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ┏━━━┓ ▓
        q1: ──┨ S ┠─┨ H ┠─┨ S ┠─┨ H ┠─┨ Z ┠─▓───
              ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛ ┗━━━┛
        >>> np.abs(circ.get_qs())
        array([1., 0., 0., 0.])
    """
    _check_int_type('length', length)
    _check_value_should_not_less('length', 0, length)
    if seed is None:
        seed = int(np.random.randint(1, 2 << 20))
    _check_seed(seed)

    stabs = _mq_vector.stabilizer.generate_double_qubits_rb_circ(length, seed)
    circ = Circuit()
    for stab in stabs:
        circ.barrier()
        circ += decompose_stabilizer(Stabilizer(stab, n_qubits=2, internal=True))
    return circ.hermitian()
