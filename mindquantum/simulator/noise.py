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
"""Noise simulator."""
from typing import Dict, Union

import numpy as np

from mindquantum.core.circuit import Circuit
from mindquantum.core.circuit.channel_adder import ChannelAdderBase
from mindquantum.core.gates import BasicGate
from mindquantum.core.operators import Hamiltonian
from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.simulator.backend_base import BackendBase


# pylint: disable=abstract-method,super-init-not-called,too-many-arguments
class NoiseBackendImpl(BackendBase):
    """Add noise based on channel adder."""

    def __init__(self, base_sim: str, n_qubits: int, adder: ChannelAdderBase, seed: int = None, dtype=None):
        """Initialize a noise backend."""
        # pylint: disable=import-outside-toplevel
        from mindquantum.simulator import Simulator

        self.base_sim = Simulator(base_sim, n_qubits, seed=seed)
        self.adder: ChannelAdderBase = adder
        self.name = "NoiseBackend"
        self.n_qubits = n_qubits

    def apply_circuit(self, circuit: Circuit, pr: Union[Dict, ParameterResolver] = None):
        """Apply a quantum circuit."""
        return self.base_sim.apply_circuit(self.adder(circuit), pr)

    def apply_gate(self, gate: BasicGate, pr: Union[Dict, ParameterResolver] = None, diff: bool = False):
        """Apply a quantum gate."""
        if diff:
            raise ValueError("For noise simulator, you cannot set diff to True.")
        # pylint: disable=too-many-function-args
        return self.base_sim.apply_circuit(self.adder(Circuit(gate)), pr, diff)

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """Apply a hamiltonian."""
        return self.base_sim.apply_hamiltonian(hamiltonian)

    def get_expectation(self, hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None) -> np.ndarray:
        """Get expectation of a hamiltonian."""
        if circ_right is not None:
            circ_right = self.adder(circ_right)
        if circ_left is not None:
            circ_left = self.adder(circ_left)
        return self.base_sim.get_expectation(hamiltonian, circ_right, circ_left, simulator_left, pr)

    def get_qs(self, ket=False):
        """Get quantum state."""
        return self.base_sim.get_qs(ket)

    def reset(self):
        """Reset mindquantum simulator to quantum zero state."""
        self.base_sim.reset()

    def sampling(self, circuit: Circuit, pr: Union[Dict, ParameterResolver] = None, shots: int = 1, seed: int = None):
        """Sample the quantum state."""
        return self.base_sim.sampling(self.adder(circuit), pr, shots, seed)

    def transform_circ(self, circuit: Circuit) -> Circuit:
        """Transform a noiseless circuit to a noise circuit based on this noise backend."""
        return self.adder(circuit)


class NoiseBackend(NoiseBackendImpl):
    """
    Noise simulator backend based on channel adder.

    Args:
        base_sim (str): The simulator that supported by MindQuantum.
        n_qubits (int): The qubit number of this noise simulator.
        adder (:class:`~.core.circuit.ChannelAdderBase`): A channel adder that can transform a circuit
            to noise circuit.
        seed (int): A random seed. Default: ``None``.
        dtype (mindquantum.dtype): The data type of simulator. If ``None``, it will be mindquantum.complex128.
            Default: ``None``.

    Examples:
        >>> from mindquantum.simulator import NoiseBackend, Simulator
        >>> from mindquantum.core.circuit import Circuit, MeasureAccepter, MixerAdder, BitFlipAdder
        >>> circ = Circuit().h(0).x(1, 0).measure_all()
        >>> circ
              ┏━━━┓       ┍━━━━━━┑
        q0: ──┨ H ┠───■───┤ M q0 ├───
              ┗━━━┛   ┃   ┕━━━━━━┙
                    ┏━┻━┓ ┍━━━━━━┑
        q1: ────────┨╺╋╸┠─┤ M q1 ├───
                    ┗━━━┛ ┕━━━━━━┙
        >>> adder = MixerAdder([
        ...     MeasureAccepter(),
        ...     BitFlipAdder(0.2),
        ... ], add_after=False)
        >>> adder(circ)
              ┏━━━┓       ╔════════════╗ ┍━━━━━━┑
        q0: ──┨ H ┠───■───╢ BFC(p=1/5) ╟─┤ M q0 ├───
              ┗━━━┛   ┃   ╚════════════╝ ┕━━━━━━┙
                    ┏━┻━┓ ╔════════════╗ ┍━━━━━━┑
        q1: ────────┨╺╋╸┠─╢ BFC(p=1/5) ╟─┤ M q1 ├───
                    ┗━━━┛ ╚════════════╝ ┕━━━━━━┙
        >>> noise_sim = Simulator(NoiseBackend('mqvector', 2, adder=adder))
        >>> noise_sim.sampling(circ,seed=42, shots=5000)
        shots: 5000
        Keys: q1 q0│0.00   0.085        0.17       0.255        0.34       0.425
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 01│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 10│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
                 11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
        {'00': 1701, '01': 796, '10': 804, '11': 1699}
    """
