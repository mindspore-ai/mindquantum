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

# pylint: disable=abstract-method,no-member
"""Quantum channel."""

from typing import Iterable

import numpy as np

from mindquantum import mqbackend as mb
from mindquantum.utils.f import _check_num_array

from .basic import BasicGate, NoiseGate, NonHermitianGate, SelfHermitianGate


class PauliChannel(NoiseGate, SelfHermitianGate):
    r"""
    A pauli channel.

    Pauli channel express error that randomly applies an additional :math:`X`, :math:`Y` or :math:`Z` gate
    on qubits with different probabilities :math:`P_x`, :math:`P_y` and :math:`P_z`,
    or do noting (applies :math:`I` gate) with probability :math:`1-P_x-P_y-P_z`.

    Pauli channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P_x - P_y - P_z)\rho + P_x X \rho X + P_y Y \rho Y + P_z Z \rho Z

    where :math:`\rho` is quantum state as density matrix type;
    :math:`P_x`, :math:`P_y` and :math:`P_z` is the probability of applying
    an additional :math:`X`, :math:`Y` and :math:`Z` gate.

    Args:
        px (int, float): probability of applying X gate.
        py (int, float): probability of applying Y gate.
        pz (int, float): probability of applying Z gate.

    Examples:
        >>> from mindquantum.core.gates import PauliChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += PauliChannel(0.8, 0.1, 0.1).on(0)
        >>> circ.measure_all()
        >>> print(circ)
        q0: ──PC────M(q0)──
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('mqvector', 1)
        >>> sim.sampling(circ, shots=1000, seed=42)
        shots: 1000
        Keys: q0│0.00     0.2         0.4         0.6         0.8         1.0
        ────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
               0│▒▒▒▒▒▒▒
                │
               1│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                │
        {'0': 101, '1': 899}
    """

    def __init__(self, px: float, py: float, pz: float, **kwargs):
        """Initialize a PauliChannel object."""
        # pylint: disable=invalid-name
        if 'name' not in kwargs:
            kwargs['name'] = 'PC'
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        SelfHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(px, (int, float)):
            raise TypeError(f"Unsupported type for px, get {type(px)}.")
        if not isinstance(py, (int, float)):
            raise TypeError(f"Unsupported type for py, get {type(py)}.")
        if not isinstance(pz, (int, float)):
            raise TypeError(f"Unsupported type for pz, get {type(pz)}.")
        if 0 <= px + py + pz <= 1:
            self.px = float(px)
            self.py = float(py)
            self.pz = float(pz)
        else:
            raise ValueError("Required total probability P = px + py + pz ∈ [0,1].")

    def __extra_prop__(self):
        """Extra prop magic method."""
        return {'px': self.px, 'py': self.py, 'pz': self.pz}

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.PauliChannel(self.px, self.py, self.pz, self.obj_qubits, self.ctrl_qubits)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None

    def __eq__(self, other):
        """Equality comparison operator."""
        if isinstance(other, PauliChannel):
            if BasicGate.__eq__(self, other) and self.px == other.px and self.py == other.py and self.pz == other.pz:
                return True
        return False


class BitFlipChannel(PauliChannel):
    r"""
    A bit flip channel.

    Bit flip channel express error that randomly flip the qubit (applies :math:`X` gate)
    with probability :math:`P`, or do noting (applies :math:`I` gate) with probability :math:`1-P`.

    Bit flip channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P X \rho X

    where :math:`\rho` is quantum state as density matrix type; :math:`P` is
    the probability of applying an additional :math:`X` gate.

    Args:
        p (int, float): probability of occurred error.

    Examples:
        >>> from mindquantum.core.gates import BitFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += BitFlipChannel(0.02).on(0)
        >>> print(circ)
        q0: ──BF(0.02)──
    """

    # pylint: disable=invalid-name

    def __init__(self, p: float, **kwargs):
        """Initialize a BitFlipChannel object."""
        kwargs['name'] = 'BFC'
        kwargs['n_qubits'] = 1
        kwargs['px'] = p
        kwargs['py'] = 0
        kwargs['pz'] = 0
        PauliChannel.__init__(self, **kwargs)
        self.p = p

    def __extra_prop__(self):
        """Extra prop magic method."""
        prop = super().__extra_prop__()
        prop['p'] = self.p
        return prop

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"BF({self.p})"


class PhaseFlipChannel(PauliChannel):
    r"""
    A phase flip channel.

    Phase flip channel express error that randomly flip the phase of qubit (applies Z gate)
    with probability :math:`P`, or do noting (applies I gate) with probability :math:`1-P`.

    Phase flip channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Z \rho Z

    where :math:`\rho` is quantum state as density matrix type; :math:`P` is the
    probability of applying an additional Z gate.

    Args:
        p (int, float): probability of occurred error.

    Examples:
        >>> from mindquantum.core.gates import PhaseFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += PhaseFlipChannel(0.02).on(0)
        >>> print(circ)
        q0: ──PF(0.02)──
    """

    # pylint: disable=invalid-name

    def __init__(self, p: float, **kwargs):
        """Initialize a PhaseFlipChannel object."""
        kwargs['name'] = 'PFC'
        kwargs['n_qubits'] = 1
        kwargs['px'] = 0
        kwargs['py'] = 0
        kwargs['pz'] = p
        PauliChannel.__init__(self, **kwargs)
        self.p = p

    def __extra_prop__(self):
        """Extra prop magic method."""
        prop = super().__extra_prop__()
        prop['p'] = self.p
        return prop

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"PF({self.p})"


class BitPhaseFlipChannel(PauliChannel):
    r"""
    A bit&phase flip channel.

    Bit phase flip channel express error that randomly flip both the state and phase
    of qubit (applies :math:`Y` gate) with probability :math:`P`, or do noting (applies :math:`I` gate)
    with probability :math:`1-P`.

    Bit phase flip channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Y \rho Y

    where :math:`\rho` is quantum state as density matrix type; :math:`P` is the
    probability of applying an additional :math:`Y` gate.

    Args:
        p (int, float): probability of occurred error.

    Examples:
        >>> from mindquantum.core.gates import BitPhaseFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += BitPhaseFlipChannel(0.02).on(0)
        >>> print(circ)
        q0: ──BPF(0.02)──
    """

    # pylint: disable=invalid-name

    def __init__(self, p: float, **kwargs):
        """Initialize a BitPhaseFlipChannel object."""
        kwargs['name'] = 'BPFC'
        kwargs['n_qubits'] = 1
        kwargs['px'] = 0
        kwargs['py'] = p
        kwargs['pz'] = 0
        PauliChannel.__init__(self, **kwargs)
        self.p = p

    def __extra_prop__(self):
        """Extra prop magic method."""
        prop = super().__extra_prop__()
        prop['p'] = self.p
        return prop

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"BPF({self.p})"


class DepolarizingChannel(NoiseGate, SelfHermitianGate):
    r"""
    A depolarizing channel.

    Depolarizing channel express errors that have probability :math:`P` to turn qubit's quantum state into
    maximally mixed state, by randomly applying one of the pauli gate(I,X,Y,Z) with same probability :math:`P/4`.
    And it has probability :math:`1-P` to change nothing.

    In one qubit case, depolarizing channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P/4( I \rho I + X \rho X + Y \rho Y + Z \rho Z)

    where :math:`\rho` is quantum state as density matrix type; :math:`P` is the probability of occurred the
    depolarizing error.

    This channel supports many object qubits. In :math:`N` qubit case, depolarizing channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + \frac{P}{4^N} \sum_j U_j \rho U_j

    where :math:`N` is the number of object qubits;
    :math:`U_j \in \left\{ I, X, Y, Z \right\} ^{\otimes N}` is many qubit pauli operator.

    Args:
        p (int, float): probability of occurred error.

    Examples:
        >>> from mindquantum.core.gates import DepolarizingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += DepolarizingChannel(0.02).on(0)
        >>> circ += DepolarizingChannel(0.01).on([1, 2])
        >>> print(circ)
        q0: ──Dep(0.02)────Dep(0.01)──
                               │
        q1: ───────────────Dep(0.01)──
    """

    # pylint: disable=invalid-name

    def __init__(self, p: float, **kwargs):
        """Initialize a DepolarizingChannel object."""
        kwargs['name'] = 'DC'
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        SelfHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(p, (int, float)):
            raise TypeError(f"Unsupported type for probability p, get {type(p)}.")
        if 0 <= p <= 1:
            self.p = p
        else:
            raise ValueError(f"Required probability p ∈ [0,1], but get p = {p}.")

    def on(self, obj_qubits, ctrl_qubits=None):
        """
        Define which qubit the gate act on.

        Args:
            obj_qubits (int, list[int]): Specific which qubits the gate act on.
            ctrl_qubits (int, list[int]): Control qubit for quantum channel should always be ``None``.
        """
        if isinstance(obj_qubits, Iterable):
            self.n_qubits = len(obj_qubits)
        return super().on(obj_qubits, ctrl_qubits)

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.DepolarizingChannel(self.p, self.obj_qubits, self.ctrl_qubits)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None

    def __eq__(self, other):
        """Equality comparison operator."""
        return super().__eq__(other) and self.p == other.p

    def __extra_prop__(self):
        """Extra prop magic method."""
        return {'p': self.p}

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"Dep({self.p})"


class AmplitudeDampingChannel(NoiseGate, NonHermitianGate):
    r"""
    Amplitude damping channel express error that qubit is affected by the energy dissipation.

    Amplitude damping channel applies noise as:

    .. math::

        \begin{gather*}
        \epsilon(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger
        \\
        \text{where}\ {E_0}=\begin{bmatrix}1&0\\
                0&\sqrt{1-\gamma}\end{bmatrix},
            \ {E_1}=\begin{bmatrix}0&\sqrt{\gamma}\\
                0&0\end{bmatrix}
        \end{gather*}

    where :math:`\rho` is quantum state as density matrix type;
    :math:`\gamma` is the coefficient of energy dissipation.

    Args:
        gamma (int, float): damping coefficient.

    Examples:
        >>> from mindquantum.core.gates import AmplitudeDampingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += AmplitudeDampingChannel(0.02).on(0)
        >>> print(circ)
        q0: ──AD(0.02)──
    """

    def __init__(self, gamma: float, **kwargs):
        """Initialize an AmplitudeDampingChannel object."""
        kwargs['name'] = 'ADC'
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        NonHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"Unsupported type for gamma, get {type(gamma)}.")
        if 0 <= gamma <= 1:
            self.gamma = gamma
        else:
            raise ValueError("Required damping coefficient gamma ∈ [0,1].")

    def __eq__(self, other):
        """Equality comparison operator."""
        return BasicGate.__eq__(self, other) and self.gamma == other.gamma

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"AD({self.gamma})"

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.AmplitudeDampingChannel(self.hermitianed, self.gamma, self.obj_qubits, self.ctrl_qubits)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


class PhaseDampingChannel(NoiseGate, NonHermitianGate):
    r"""
    A phase damping channel.

    Phase damping channel express error that qubit loses quantum information without exchanging energy with environment.

    Phase damping channel applies noise as:

    .. math::

        \begin{gather*}
        \epsilon(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger
        \\
        \text{where}\ {E_0}=\begin{bmatrix}1&0\\
                0&\sqrt{1-\gamma}\end{bmatrix},
            \ {E_1}=\begin{bmatrix}0&0\\
                0&\sqrt{\gamma}\end{bmatrix}
        \end{gather*}

    where :math:`\rho` is quantum state as density matrix type;
    :math:`\gamma` is the coefficient of quantum information loss.

    Args:
        gamma (int, float): damping coefficient.

    Examples:
        >>> from mindquantum.core.gates import PhaseDampingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += PhaseDampingChannel(0.02).on(0)
        >>> print(circ)
        q0: ──PD(0.02)──
    """

    def __init__(self, gamma: float, **kwargs):
        """Initialize a PhaseDampingChannel object."""
        kwargs['name'] = 'PDC'
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        NonHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"Unsupported type for gamma, get {type(gamma)}.")
        if 0 <= gamma <= 1:
            self.gamma = gamma
        else:
            raise ValueError("Required damping coefficient gamma ∈ [0,1].")

    def __eq__(self, other):
        """Equality comparison operator."""
        return super().__eq__(other) and self.gamma == other.gamma

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"PD({self.gamma})"

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.PhaseDampingChannel(self.gamma, self.obj_qubits, self.ctrl_qubits)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None


class KrausChannel(NoiseGate, NonHermitianGate):
    r"""
    A kraus channel.

    Kraus channel accepts two or more 2x2 matrices as Kraus operator to construct
    custom (single-qubit) noise in quantum circuit.

    Kraus channel applies noise as:

    .. math::

        \epsilon(\rho) = \sum_{k=0}^{m-1} E_k \rho E_k^\dagger

    where :math:`\rho` is quantum state as density matrix type; {:math:`E_k`} is Kraus operator,
    and it should satisfy the completeness condition: :math:`\sum_k E_k^\dagger E_k = I`.

    Args:
        name (str): the name of this custom noise channel.
        kraus_op (list, np.ndarray): Kraus operator, with two or more 2x2 matrices packaged as a list.

    Examples:
        >>> from mindquantum.core.gates import KrausChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> from cmath import sqrt
        >>> gamma = 0.5
        >>> kmat0 = [[1, 0], [0, sqrt(1 - gamma)]]
        >>> kmat1 = [[0, sqrt(gamma)], [0, 0]]
        >>> amplitude_damping = KrausChannel('damping', [kmat0, kmat1])
        >>> circ = Circuit()
        >>> circ += amplitude_damping.on(0)
        >>> print(circ)
        q0: ──damping──
    """

    def __init__(self, name: str, kraus_op, **kwargs):
        """Initialize an KrausChannel object."""
        _check_num_array(kraus_op, name)
        if not isinstance(kraus_op, np.ndarray):
            kraus_op = np.array(kraus_op)
        for mat in kraus_op:
            if len(mat.shape) != 2:
                raise ValueError(f"matrix_value require shape of 2, but get shape of {mat.shape}")
            if mat.shape[0] != mat.shape[1]:
                raise ValueError(f"matrix_value need a square matrix, but get shape {mat.shape}")
            if mat.shape[0] != 2:
                raise ValueError(f"Dimension of matrix_value need should be 2, but get {mat.shape[0]}")
        sum_of_mat = np.zeros((2, 2), 'complex128')
        for mat in kraus_op:
            sum_of_mat += np.dot(mat.T.conj(), mat)
        if not np.allclose(sum_of_mat, [[1, 0], [0, 1]]):
            raise ValueError(f"kraus_op need to satisfy the completeness condition, but get {sum_of_mat}")
        kwargs['name'] = name
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        NonHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        self.kraus_op = kraus_op

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.KrausChannel(self.kraus_op, self.obj_qubits, self.ctrl_qubits)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return self.name
