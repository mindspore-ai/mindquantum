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

from itertools import product
from math import exp, sqrt

import numpy as np
import numpy.typing as npt

from mindquantum import mqbackend as mb
from mindquantum.utils.f import _check_input_type, _check_num_array
from mindquantum.utils.string_utils import string_expression

from .basic import BasicGate, NoiseGate, NonHermitianGate, SelfHermitianGate


class GroupedPauliChannel(NoiseGate, SelfHermitianGate):
    r"""
    A group of pauli channels.

    This quantum channel is equivalent to a list of pauli channels, but will evaluate much
    faster than apply them one by one. For detail of pauli channel, please refers to
    :class:`~.core.gates.PauliChannel`.

    .. math::

        \epsilon(\rho) = \otimes_i \epsilon_\text{pauli}^i(\rho)

    Args:
        probs (numpy.ndarray): The error probabilities of pauli channel. It has dimension of
            `(n, 3)`, where the first dimension `n` is the qubit number or the number of pauli channels
            represented by this quantum channel . The second dimension `3` represents the probability of
            applying :math:`X`, :math:`Y` or :math:`Z` gate.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core.gates import GroupedPauliChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.simulator import Simulator
        >>> probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.3, 0.0]])
        >>> circ = Circuit([GroupedPauliChannel(probs).on([0, 1])]).measure_all()
        >>> circ
              ╭ ╔═══Grouped Pauli Channel     ╮ ┍━━━━━━┑
        q0: ──┤─╢ PC(px=1, py=0, pz=0) ╟──────├─┤ M q0 ├───
              │ ╚══════════════════════╝      │ ┕━━━━━━┙
              │ ╔═════════════════════════╗   │ ┍━━━━━━┑
        q1: ──┤─╢ PC(px=0, py=3/10, pz=0) ╟───├─┤ M q1 ├───
              ╰ ╚═════════════════════════╝   ╯ ┕━━━━━━┙
        >>> Simulator('mqvector', circ.n_qubits).sampling(circ, shots=1000, seed=42)
        shots: 1000
        Keys: q1 q0│0.00   0.177       0.355       0.532        0.71       0.887
        ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                 01│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                   │
                 11│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                   │
        {'01': 710, '11': 290}
    """

    def __init__(self, probs: npt.NDArray[np.float64], **kwargs):
        """Initialize a GroupedPauliChannel object."""
        if 'name' not in kwargs:
            kwargs['name'] = 'GPC'
        kwargs['n_qubits'] = probs.shape[0]
        _check_input_type("probs", np.ndarray, probs)
        if not np.issubdtype(probs.dtype, np.floating):
            raise TypeError(f"probs requires a real number type, but get {probs.dtype}")
        if np.any(probs < 0.0):
            raise ValueError("probs cannot be negative value.")
        shape = probs.shape
        if len(shape) != 2:
            raise ValueError(f"probs requires a two dimension array, but get dimension with {shape}")
        if shape[1] != 3:
            raise ValueError(f"The second dimension of probs should be two, but get {shape[1]}")
        wrong_channel = np.where(probs.sum(axis=1) > 1)[0]
        if wrong_channel.shape[0]:
            raise ValueError(f"Wrong probs for {wrong_channel[0]}-pauli channel: {probs[wrong_channel[0]]}")
        self.probs = probs
        NoiseGate.__init__(self, **kwargs)
        SelfHermitianGate.__init__(self, **kwargs)

    def __eq__(self, other):
        """Equality comparison operator."""
        if isinstance(other, GroupedPauliChannel):
            return BasicGate.__eq__(self, other) and np.allclose(self.probs, other.probs, atol=1e-8)
        return False

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.GroupedPauliChannel(self.probs, self.obj_qubits, self.ctrl_qubits)

    def matrix(self):  # pylint: disable=arguments-differ
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of every pauli channel.
        """
        mat = []
        for prob in self.probs:
            px, py, pz = prob
            mat_i = sqrt(1 - px - py - pz) * np.array([[1, 0], [0, 1]])
            mat_x = sqrt(px) * np.array([[0, 1], [1, 0]])
            mat_y = sqrt(py) * np.array([[0, -1j], [1j, 0]])
            mat_z = sqrt(pz) * np.array([[1, 0], [0, -1]])
            mat.append([mat_i, mat_x, mat_y, mat_z])
        return mat


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
              ╔══════════════════════════════╗ ┍━━━━━━┑
        q0: ──╢ PC(px=4/5, py=1/10, pz=1/10) ╟─┤ M q0 ├───
              ╚══════════════════════════════╝ ┕━━━━━━┙
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
        if np.any(np.array([px, py, pz]) < 0.0):
            raise ValueError("Probability cannot be negative value.")
        if 0 <= px + py + pz <= 1:
            self.px = float(px)
            self.py = float(py)
            self.pz = float(pz)
        else:
            raise ValueError("Required total probability P = px + py + pz ∈ [0,1].")

    def __extra_prop__(self):
        """Extra prop magic method."""
        return {'px': self.px, 'py': self.py, 'pz': self.pz}

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f'px={string_expression(self.px)}, py={string_expression(self.py)}, pz={string_expression(self.pz)}'

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

    def matrix(self):  # pylint: disable=arguments-differ
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        mat_i = sqrt(1 - self.px - self.py - self.pz) * np.array([[1, 0], [0, 1]])
        mat_x = sqrt(self.px) * np.array([[0, 1], [1, 0]])
        mat_y = sqrt(self.py) * np.array([[0, -1j], [1j, 0]])
        mat_z = sqrt(self.pz) * np.array([[1, 0], [0, -1]])
        return [mat_i, mat_x, mat_y, mat_z]


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
              ╔═════════════╗
        q0: ──╢ BFC(p=1/50) ╟───
              ╚═════════════╝
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

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f'p={string_expression(self.p)}'

    def matrix(self):
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        mat_i = sqrt(1 - self.p) * np.array([[1, 0], [0, 1]])
        mat_x = sqrt(self.p) * np.array([[0, 1], [1, 0]])
        return [mat_i, mat_x]


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
              ╔═════════════╗
        q0: ──╢ PFC(p=1/50) ╟───
              ╚═════════════╝
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

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f'p={string_expression(self.p)}'

    def matrix(self):
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        mat_i = sqrt(1 - self.p) * np.array([[1, 0], [0, 1]])
        mat_z = sqrt(self.p) * np.array([[1, 0], [0, -1]])
        return [mat_i, mat_z]


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
              ╔══════════════╗
        q0: ──╢ BPFC(p=1/50) ╟───
              ╚══════════════╝
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

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f'p={string_expression(self.p)}'

    def matrix(self):
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        mat_i = sqrt(1 - self.p) * np.array([[1, 0], [0, 1]])
        mat_y = sqrt(self.p) * np.array([[0, -1j], [1j, 0]])
        return [mat_i, mat_y]


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

    * For :math:`0 \le P \le 1` case, this channel is a depolarizing channel, and it becomes a completely
      depolarizing channel when :math:`P = 1`.
    * However, :math:`1 < P \le 4^N / (4^N - 1)` is also an available case, but not a depolarizing channel
      any more. When :math:`P = 4^N / (4^N - 1)` it becomes a uniform Pauli error channel:
      :math:`E(\rho) = \sum_j V_j \rho V_j / (4^n - 1)`, where :math:`V_j = U_j \setminus I^{\otimes N}`.

    Args:
        p (int, float): probability of occurred depolarizing error.
        n_qubits (int): qubit number of this depolarizing channel. Default: ``1``.

    Examples:
        >>> from mindquantum.core.gates import DepolarizingChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit()
        >>> circ += DepolarizingChannel(0.02).on(0)
        >>> circ += DepolarizingChannel(0.01, 2).on([0, 1])
        >>> print(circ)
              ╔════════════╗ ╔═════════════╗
        q0: ──╢ DC(p=1/50) ╟─╢             ╟───
              ╚════════════╝ ║             ║
                             ║ DC(p=1/100) ║
        q1: ─────────────────╢             ╟───
                             ╚═════════════╝
    """

    # pylint: disable=invalid-name

    def __init__(self, p: float, n_qubits: int = 1, **kwargs):
        """Initialize a DepolarizingChannel object."""
        kwargs['name'] = 'DC'
        kwargs['n_qubits'] = n_qubits
        NoiseGate.__init__(self, **kwargs)
        SelfHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(p, (int, float)):
            raise TypeError(f"Unsupported type for argument p, get {type(p)}.")
        self.p = p
        if not 0 <= self.p <= 4**self.n_qubits / (4**self.n_qubits - 1):
            raise ValueError(
                f"Required argument p ∈ [0, {4**self.n_qubits}/{4**self.n_qubits - 1}], but get p = {self.p}."
            )

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.DepolarizingChannel(self.p, self.obj_qubits, self.ctrl_qubits)

    def __eq__(self, other):
        """Equality comparison operator."""
        return super().__eq__(other) and self.p == other.p

    def __extra_prop__(self):
        """Extra prop magic method."""
        return {'p': self.p}

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f'p={string_expression(self.p)}'

    def matrix(self):  # pylint: disable=arguments-differ
        r"""
        Kraus operator of the quantum channel.

        Returns:
            list, the kraus matrix of this operator, while order of output is in dictionary
                order of :math:`\left\{ I, X, Y, Z \right\} ^{\otimes N}`.
        """
        # pylint: disable=import-outside-toplevel
        from mindquantum.core.operators import QubitOperator

        p = np.ones(4**self.n_qubits) * sqrt(self.p) / 2**self.n_qubits
        p[0] = sqrt(1 - self.p + self.p / 4**self.n_qubits)
        out = []
        for i_pauli, pauli_tuple in enumerate(product(['I', 'X', 'Y', 'Z'], repeat=self.n_qubits)):
            pauli_string = ' '.join(f"{pauli}{idx}" for idx, pauli in enumerate(pauli_tuple) if pauli != 'I')
            m = QubitOperator(pauli_string).matrix(self.n_qubits) * p[i_pauli]
            out.append(m.toarray())
        return out


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
              ╔═════════════╗
        q0: ──╢ ADC(γ=1/50) ╟───
              ╚═════════════╝
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

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f'γ={string_expression(self.gamma)}'

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.AmplitudeDampingChannel(self.hermitianed, self.gamma, self.obj_qubits, self.ctrl_qubits)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None

    def matrix(self):  # pylint: disable=arguments-differ
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        mat_0 = np.array([[1, 0], [0, sqrt(1 - self.gamma)]])
        mat_1 = np.array([[0, sqrt(self.gamma)], [0, 0]])
        if self.hermitianed:
            return np.array([mat_0, mat_1.T])
        return [mat_0, mat_1]


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
              ╔═════════════╗
        q0: ──╢ PDC(γ=1/50) ╟───
              ╚═════════════╝
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

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f'γ={string_expression(self.gamma)}'

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.PhaseDampingChannel(self.gamma, self.obj_qubits, self.ctrl_qubits)

    def define_projectq_gate(self):
        """Define the corresponded projectq gate."""
        self.projectq_gate = None

    def matrix(self):  # pylint: disable=arguments-differ
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        mat_0 = np.array([[1, 0], [0, sqrt(1 - self.gamma)]])
        mat_1 = np.array([[0, 0], [0, sqrt(self.gamma)]])
        return [mat_0, mat_1]


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
              ╔═════════╗
        q0: ──╢ damping ╟───
              ╚═════════╝
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
        if not np.allclose(sum_of_mat, [[1, 0], [0, 1]], atol=1e-6):
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

    def matrix(self):  # pylint: disable=arguments-differ
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        return list(self.kraus_op)


class ThermalRelaxationChannel(NoiseGate, NonHermitianGate):
    r"""
    Thermal relaxation channel.

    The thermal relaxation channel describes the thermal decoherence and dephasing of qubit
    when a quantum gate is applied, and is determined by T1, T2 and gate time.

    The Choi-matrix representation of this channel is as below:

    .. math::

        \begin{gather*}
            \epsilon(\rho) = \text{tr}_1 \left[ \Lambda \left( \rho^T \otimes I \right) \right],
            \Lambda=\begin{pmatrix}
                \epsilon_{T_1} & 0 & 0 & \epsilon_{T_2} \\
                0 & 1-\epsilon_{T_1} & 0 & 0            \\
                0 & 0 & 0 & 0                           \\
                \epsilon_{T_2} & 0 & 0 & 1
            \end{pmatrix}
            \\
            \text{where}\ \epsilon_{T_1}=e^{-T_g/T_1}, \epsilon_{T_2}=e^{-T_g/T_2}
        \end{gather*}

    Where :math:`\rho` is quantum state as density matrix type; :math:`\Lambda` is Choi matrix,
    :math:`T_1` is thermal relaxation time of qubit, :math:`T_2` is dephasing time of qubit,
    :math:`T_g` is gate time.

    Args:
        t1 (int, float): T1 of the qubit.
        t2 (int, float): T2 of the qubit.
        gate_time (int, float): time of the quantum gate.

    Examples:
        >>> from mindquantum.core.gates import ThermalRelaxationChannel
        >>> from mindquantum.core.circuit import Circuit
        >>> t1 = 100000
        >>> t2 = 50000
        >>> gate_time = 35
        >>> circ = Circuit()
        >>> circ += ThermalRelaxationChannel(t1, t2, gate_time).on(0)
        >>> print(circ)
              ╔═══════════════════════════════╗
        q0: ──╢ TRC(t1=100000,t2=50000,tg=35) ╟───
              ╚═══════════════════════════════╝
    """

    def __init__(self, t1: float, t2: float, gate_time: float, **kwargs):
        """Initialize a ThermalRelaxationChannel object."""
        kwargs['name'] = 'TRC'
        kwargs['n_qubits'] = 1
        NoiseGate.__init__(self, **kwargs)
        NonHermitianGate.__init__(self, **kwargs)
        self.projectq_gate = None
        if not isinstance(t1, (int, float)):
            raise TypeError(f"Unsupported type for t1, get {type(t1)}.")
        if not isinstance(t2, (int, float)):
            raise TypeError(f"Unsupported type for t2, get {type(t2)}.")
        if not isinstance(gate_time, (int, float)):
            raise TypeError(f"Unsupported type for gate_time, get {type(gate_time)}.")
        if (t1 <= 0) or (t2 <= 0):
            raise ValueError(f"T1 and T2 must be positive, but get T1={t1}, T2={t2}")
        if gate_time < 0:
            raise ValueError(f"gate time cannot be negative, but get {gate_time}")
        if t2 >= 2 * t1:
            raise ValueError("(T2 >= 2 * T1) is invalid case for thermal relaxation channel.")
        self.t1 = t1
        self.t2 = t2
        self.gate_time = gate_time

    def get_cpp_obj(self):
        """Get underlying C++ object."""
        return mb.gate.ThermalRelaxationChannel(self.t1, self.t2, self.gate_time, self.obj_qubits, self.ctrl_qubits)

    def __type_specific_str__(self):
        """Return a string representation of the object."""
        return f't1={string_expression(self.t1)},t2={string_expression(self.t2)},tg={string_expression(self.gate_time)}'

    def matrix(self):  # pylint: disable=arguments-differ,too-many-locals
        """
        Kraus operator of the quantum channel.

        Returns:
            list, contains all Kraus operators of this quantum channel.
        """
        e1 = exp(-self.gate_time / self.t1)
        e2 = exp(-self.gate_time / self.t2)
        p_reset = 1 - e1
        if self.t1 >= self.t2:
            pz = e1 * (1 - e2 / e1) / 2
            ki = np.array([[sqrt(1 - pz - p_reset), 0], [0, sqrt(1 - pz - p_reset)]])
            kz = np.array([[sqrt(pz), 0], [0, -sqrt(pz)]])
            k_reset00 = np.array([[sqrt(p_reset), 0], [0, 0]])
            k_reset01 = np.array([[0, sqrt(p_reset)], [0, 0]])
            return [ki, kz, k_reset00, k_reset01]
        if 2 * self.t1 > self.t2:
            eigenvalue0 = (2 - p_reset + sqrt(p_reset**2 + 4 * e2**2)) / 2
            eigenvalue1 = (2 - p_reset - sqrt(p_reset**2 + 4 * e2**2)) / 2
            eigen_vector0 = (eigenvalue0 - e1) / e2
            eigen_vector1 = (eigenvalue1 - e1) / e2
            k0 = np.array([[eigen_vector0, 0], [0, 1]]) * sqrt(eigenvalue0 / (eigen_vector0**2 + 1))
            k1 = np.array([[eigen_vector1, 0], [0, 1]]) * sqrt(eigenvalue1 / (eigen_vector1**2 + 1))
            k2 = np.array([[0, sqrt(p_reset)], [0, 0]])
            return [k0, k1, k2]
        raise ValueError("(T2 >= 2 * T1) is invalid case for thermal relaxation channel.")
