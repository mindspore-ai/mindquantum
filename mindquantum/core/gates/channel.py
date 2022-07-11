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

# pylint: disable=abstract-method

"""Quantum channel."""

from mindquantum import mqbackend as mb

from .basic import BasicGate, NoiseGate, SelfHermitianGate


class PauliChannel(NoiseGate, SelfHermitianGate):
    r"""
    Quantum channel that express the incoherent noise in quantum computing.

    Pauli channel express error that randomly applies an additional X, Y or Z gate
    on qubits with different probabilities Px, Py and Pz, or do noting (applies I gate)
    with probability P = (1 - Px - Py - Pz).

    Pauli channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P_x - P_y - P_z)\rho + P_x X \rho X + P_y Y \rho Y + P_z Z \rho Z

    where ρ is quantum state as density matrix type;
    Px, Py and Pz is the probability of applying an additional X, Y and Z gate.

    Args:
        px (int, float): probability of applying X gate.
        py (int, float): probability of applying Y gate.
        pz (int, float): probability of applying Z gate.

    Examples:
        >>> from mindquantum.core.gates import PauliChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += PauliChannel(0.8, 0.1, 0.1).on(0)
        >>> circ += PauliChannel(0, 0.05, 0.9).on(1, 0)
        >>> circ.measure_all()
        >>> print(circ)
        q0: ──PC────●─────M(q0)──
                    │
        q1: ────────PC────M(q1)──
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('projectq', 2)
        >>> sim.sampling(circ, shots=1000, seed=42)
        shots: 1000
        Keys: q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 00│▒▒▒▒▒▒▒
                   │
                 01│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 11│▒▒▒
                   │
        {'00': 101, '01': 862, '11': 37}
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
            self.px = px
            self.py = py
            self.pz = pz
        else:
            raise ValueError("Required total probability P = px + py + pz ∈ [0,1].")

    def __extra_prop__(self):
        """Extra prop magic method."""
        return {'px': self.px, 'py': self.py, 'pz': self.pz}

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        cpp_gate = mb.basic_gate('PL', True, self.px, self.py, self.pz)
        cpp_gate.obj_qubits = self.obj_qubits
        cpp_gate.ctrl_qubits = self.ctrl_qubits
        return cpp_gate

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
    Quantum channel that express the incoherent noise in quantum computing.

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
        >>> circ += BitFlipChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──BF(0.02)───────●──────
                             │
        q1: ──────────────BF(0.01)──
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
    Quantum channel that express the incoherent noise in quantum computing.

    Phase flip channel express error that randomly flip the phase of qubit (applies Z gate)
    with probability P, or do noting (applies I gate) with probability 1 - P.

    Phase flip channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Z \rho Z

    where ρ is quantum state as density matrix type; P is the probability of applying an additional Z gate.

    Args:
        p (int, float): probability of occurred error.

    Examples:
        >>> from mindquantum.core.gates import PhaseFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += PhaseFlipChannel(0.02).on(0)
        >>> circ += PhaseFlipChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──PF(0.02)───────●──────
                             │
        q1: ──────────────PF(0.01)──
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
    Quantum channel that express the incoherent noise in quantum computing.

    Bit phase flip channel express error that randomly flip both the state and phase
    of qubit (applies Y gate) with probability P, or do noting (applies I gate)
    with probability 1 - P.

    Bit phase flip channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P Y \rho Y

    where ρ is quantum state as density matrix type; P is the probability of applying an additional Y gate.

    Args:
        p (int, float): probability of occurred error.

    Examples:
        >>> from mindquantum.core.gates import BitPhaseFlipChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += BitPhaseFlipChannel(0.02).on(0)
        >>> circ += BitPhaseFlipChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──BPF(0.02)────────●──────
                               │
        q1: ───────────────BPF(0.01)──
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


class DepolarizingChannel(PauliChannel):
    r"""
    Quantum channel that express the incoherent noise in quantum computing.

    Depolarizing channel express errors that have probability P to turn qubit's quantum state into
    maximally mixed state, by randomly applying one of the pauli gate(X,Y,Z) with same probability P/3.
    And it has probability 1 - P to change nothing (applies I gate).

    Depolarizing channel applies noise as:

    .. math::

        \epsilon(\rho) = (1 - P)\rho + P/3( X \rho X + Y \rho Y + Z \rho Z)

    where ρ is quantum state as density matrix type; P is the probability of occurred the depolarizing error.

    Args:
        p (int, float): probability of occurred error.

    Examples:
        >>> from mindquantum.core.gates import DepolarizingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += DepolarizingChannel(0.02).on(0)
        >>> circ += DepolarizingChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──Dep(0.02)────────●──────
                               │
        q1: ───────────────Dep(0.01)──
    """

    # pylint: disable=invalid-name

    def __init__(self, p: float, **kwargs):
        """Initialize a DepolarizingChannel object."""
        kwargs['name'] = 'DC'
        kwargs['n_qubits'] = 1
        kwargs['px'] = p / 3
        kwargs['py'] = p / 3
        kwargs['pz'] = p / 3
        PauliChannel.__init__(self, **kwargs)
        self.p = p

    def __extra_prop__(self):
        """Extra prop magic method."""
        prop = super().__extra_prop__()
        prop['p'] = self.p
        return prop

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"Dep({self.p})"


class AmplitudeDampingChannel(NoiseGate, SelfHermitianGate):
    r"""
    Quantum channel that express the incoherent noise in quantum computing.

    Amplitude damping channel express error that qubit is affected by the energy dissipation.

    Amplitude damping channel applies noise as:

    .. math::

        \epsilon(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger

        where\ {E_0}=\begin{bmatrix}1&0\\
                0&\sqrt{1-\gamma}\end{bmatrix},
            \ {E_1}=\begin{bmatrix}0&\sqrt{\gamma}\\
                0&0\end{bmatrix}

    where :math:`\rho` is quantum state as density matrix type;
    :math:`\gamma` is the coefficient of energy dissipation.

    Args:
        gamma (int, float): damping coefficient.

    Examples:
        >>> from mindquantum.core.gates import AmplitudeDampingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += AmplitudeDampingChannel(0.02).on(0)
        >>> circ += AmplitudeDampingChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──AD(0.02)───────●──────
                             │
        q1: ──────────────AD(0.01)──
    """

    def __init__(self, gamma: float, **kwargs):
        """Initialize an AmplitudeDampingChannel object."""
        kwargs['name'] = 'ADC'
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        SelfHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"Unsupported type for gamma, get {type(gamma)}.")
        if 0 <= gamma <= 1:
            self.gamma = gamma
        else:
            raise ValueError("Required damping coefficient gamma ∈ [0,1].")

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        cpp_gate = mb.basic_gate('ADC', True, self.gamma)
        cpp_gate.obj_qubits = self.obj_qubits
        cpp_gate.ctrl_qubits = self.ctrl_qubits
        return cpp_gate

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None

    def __eq__(self, other):
        """Equality comparison operator."""
        return BasicGate.__eq__(self, other) and self.gamma == other.gamma

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"AD({self.gamma})"


class PhaseDampingChannel(NoiseGate, SelfHermitianGate):
    r"""
    Quantum channel that express the incoherent noise in quantum computing.

    Phase damping channel express error that qubit loses quantum information without exchanging energy with environment.

    Phase damping channel applies noise as:

    .. math::

        \epsilon(\rho) = E_0 \rho E_0^\dagger + E_1 \rho E_1^\dagger

        where\ {E_0}=\begin{bmatrix}1&0\\
                0&\sqrt{1-\gamma}\end{bmatrix},
            \ {E_1}=\begin{bmatrix}0&0\\
                0&\sqrt{\gamma}\end{bmatrix}

    where ρ is quantum state as density matrix type;
    gamma is the coefficient of quantum information loss.

    Args:
        gamma (int, float): damping coefficient.

    Examples:
        >>> from mindquantum.core.gates import PhaseDampingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += PhaseDampingChannel(0.02).on(0)
        >>> circ += PhaseDampingChannel(0.01).on(1, 0)
        >>> print(circ)
        q0: ──PD(0.02)───────●──────
                             │
        q1: ──────────────PD(0.01)──
    """

    def __init__(self, gamma: float, **kwargs):
        """Initialize a PhaseDampingChannel object."""
        kwargs['name'] = 'PDC'
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        SelfHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"Unsupported type for gamma, get {type(gamma)}.")
        if 0 <= gamma <= 1:
            self.gamma = gamma
        else:
            raise ValueError("Required damping coefficient gamma ∈ [0,1].")

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        cpp_gate = mb.basic_gate('PDC', True, self.gamma)
        cpp_gate.obj_qubits = self.obj_qubits
        cpp_gate.ctrl_qubits = self.ctrl_qubits
        return cpp_gate

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None

    def __eq__(self, other):
        """Equality comparison operator."""
        return super().__eq__(other) and self.gamma == other.gamma

    def __str_in_circ__(self):
        """Return a string representation of the object in a quantum circuit."""
        return f"PD({self.gamma})"
