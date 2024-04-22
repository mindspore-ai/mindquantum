# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Mindquantum Clifford Stabilizer Simulator."""
# pylint:disable=abstract-method,keyword-arg-before-vararg,no-member,unused-argument
# pylint:disable=redefined-outer-name
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from mindquantum import _mq_vector
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import MeasureResult, S, X
from mindquantum.simulator.backend_base import BackendBase
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_seed,
    _check_value_should_not_less,
)

if TYPE_CHECKING:
    from mindquantum.simulator import Simulator


class Stabilizer(BackendBase):
    """Stabilizer simulator."""

    def __init__(self, name: str, n_qubits, seed=None, *args, **kwargs):
        """Initialize a stabilizer simulator."""
        super().__init__(name, n_qubits, seed)
        if kwargs.get('internal', False):
            self.sim = name
            self.name = 'stabilizer'
        else:
            self.sim = _mq_vector.stabilizer.StabilizerTableau(n_qubits, seed)

    def __str__(self):
        """Return a string representation of the object."""
        return f"stabilizer simulator with {self.n_qubits} qubits.\nCurrent tableau:\n{self.sim.tableau_to_string()}"

    def __repr__(self):
        """Return a string representation of the object."""
        return self.__str__()

    # pylint: disable=arguments-differ
    def apply_circuit(self, circuit: Circuit, *args, **kwargs):
        """Apply a quantum circuit."""
        _check_input_type('circuit', Circuit, circuit)
        if self.n_qubits < circuit.n_qubits:
            raise ValueError(f"Circuit has {circuit.n_qubits} qubits, which is more than simulator qubits.")
        res = self.sim.apply_circuit(circuit.get_cpp_obj())
        if res:
            out = MeasureResult()
            out.add_measure(circuit.all_measures.keys())
            out.collect_data([[res[i] for i in out.keys_map]])
            return out
        return None

    def copy(self) -> Stabilizer:
        """Copy a simulator."""
        sim = Stabilizer(self.name, self.n_qubits, self.seed)
        sim.sim = self.sim.copy()
        return sim

    def __eq__(self, other: Stabilizer) -> bool:
        """Check whether two stabilizers are equal or not."""
        _check_input_type('other', Stabilizer, other)
        return _mq_vector.stabilizer.StabilizerTableau.__eq__(self.sim, other.sim)

    # pylint: disable=arguments-differ
    def get_qs(self, *args, **kwargs):
        """Return the tableau of stabilizer."""
        return np.array(self.sim.tableau_to_vector())

    # pylint: disable=arguments-differ
    def sampling(self, circuit: Circuit, shots: int = 1, seed: int = None, *args, **kwargs):
        """Sample the quantum state."""
        if not circuit.all_measures.map:
            raise ValueError("circuit must have at least one measurement gate.")
        _check_input_type("circuit", Circuit, circuit)
        if self.n_qubits < circuit.n_qubits:
            raise ValueError(f"Circuit has {circuit.n_qubits} qubits, which is more than simulator qubits.")
        _check_int_type("sampling shots", shots)
        _check_value_should_not_less("sampling shots", 1, shots)
        if seed is None:
            seed = int(np.random.randint(1, 2 << 20))
        else:
            _check_seed(seed)
        res = MeasureResult()
        res.add_measure(circuit.all_measures.keys())
        if circuit.is_measure_end and not circuit.is_noise_circuit:
            sampler = self.sim.sampling_measure_ending_without_noise
        else:
            sampler = self.sim.sampling
        samples = np.array(sampler(circuit.get_cpp_obj(), shots, res.keys_map, seed)).reshape((shots, -1))
        res.collect_data(samples)
        return res


def decompose_stabilizer(sim: Simulator | Stabilizer) -> Circuit:
    """
    Decompose a stabilizer into clifford quantum circuit.

    Args:
        sim (Simulator): A stabilizer simulator.

    Examples:
        >>> from mindquantum.simulator import Simulator, decompose_stabilizer
        >>> from mindquantum.core.circuit import Circuit
        >>> stabilizer = Simulator('stabilizer', 2)
        >>> stabilizer.apply_circuit(Circuit().h(0).x(1, 0))
        >>> decompose_stabilizer(stabilizer)
              ┏━━━┓
        q0: ──┨ H ┠───■─────
              ┗━━━┛   ┃
                    ┏━┻━┓
        q1: ────────┨╺╋╸┠───
                    ┗━━━┛
        >>> from mindquantum.algorithm.error_mitigation import query_single_qubit_clifford_elem
        >>> decompose_stabilizer(query_single_qubit_clifford_elem(10))
              ┏━━━┓ ┏━━━┓ ┏━━━━┓
        q0: ──┨╺╋╸┠─┨ Z ┠─┨ S† ┠───
              ┗━━━┛ ┗━━━┛ ┗━━━━┛
    """
    # pylint: disable=import-outside-toplevel
    from mindquantum.simulator import Simulator

    if isinstance(sim, Simulator):
        if sim.backend.name != 'stabilizer':
            raise TypeError(f"Input simulator should be a stabilizer simulator, but get {sim.backend.name}.")
        sim = sim.backend
    elif not isinstance(sim, Stabilizer):
        raise TypeError(f'sim require a type of Simulator or Stabilizer, but get {type(sim)}.')
    circ = Circuit()
    decomposed = sim.sim.decompose()
    for g in decomposed:
        g_id = str(g.get_id())
        obj = g.get_obj_qubits()
        ctrl = g.get_ctrl_qubits()
        if g_id in 'HXYZS':
            getattr(circ, g_id.lower())(obj, ctrl)
        elif g_id == 'Sdag':
            circ += S.on(obj, ctrl).hermitian()
        elif g_id == 'CNOT':
            circ += X.on(obj[0], obj[1])
        else:
            raise RuntimeError(f"Unknown gate id: {g_id}")
    return circ


def get_tableau_string(sim: Simulator | Stabilizer) -> str:
    """
    Get the string expression of a stabilizer tableau.

    Args:
        sim (Simulator): A stabilizer simulator.

    Examples:
        >>> from mindquantum.simulator import Simulator, get_tableau_string
        >>> from mindquantum.core.circuit import Circuit
        >>> stabilizer = Simulator('stabilizer', 2)
        >>> stabilizer.apply_circuit(Circuit().h(0).x(1, 0))
        >>> print(get_tableau_string(stabilizer))
        0 0 | 1 0 | 0
        0 1 | 0 0 | 0
        -------------
        1 1 | 0 0 | 0
        0 0 | 1 1 | 0
    """
    # pylint: disable=import-outside-toplevel
    from mindquantum.simulator import Simulator

    if isinstance(sim, Simulator):
        if sim.backend.name != 'stabilizer':
            raise TypeError(f"Input simulator should be a stabilizer simulator, but get {sim.backend.name}.")
        sim = sim.backend
    elif not isinstance(sim, Stabilizer):
        raise TypeError(f'sim require a type of Simulator or Stabilizer, but get {type(sim)}.')
    return sim.sim.tableau_to_string()


def get_stabilizer_string(sim: Simulator | Stabilizer) -> str:
    """
    Get the string expression of a stabilizer.

    Args:
        sim (Simulator): A stabilizer simulator.

    Examples:
        >>> from mindquantum.simulator import Simulator, get_stabilizer_string
        >>> from mindquantum.core.circuit import Circuit
        >>> stabilizer = Simulator('stabilizer', 2)
        >>> stabilizer.apply_circuit(Circuit().h(0).x(1, 0))
        >>> print(get_stabilizer_string(stabilizer))
        destabilizer:
        +IZ
        +XI
        stabilizer:
        +XX
        +ZZ
    """
    # pylint: disable=import-outside-toplevel
    from mindquantum.simulator import Simulator

    if isinstance(sim, Simulator):
        if sim.backend.name != 'stabilizer':
            raise TypeError(f"Input simulator should be a stabilizer simulator, but get {sim.backend.name}.")
        sim = sim.backend
    elif not isinstance(sim, Stabilizer):
        raise TypeError(f'sim require a type of Simulator or Stabilizer, but get {type(sim)}.')
    return sim.sim.stabilizer_to_string()
