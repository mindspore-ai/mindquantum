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
"""Simulator."""
import numpy as np

from mindquantum.dtype import to_mq_type

from ..core.operators import Hamiltonian
from ..utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_seed,
    _check_value_should_not_less,
)
from .available_simulator import SUPPORTED_SIMULATOR
from .backend_base import BackendBase
from .mq_blas import MQBlas
from .mqsim import MQSim


def get_supported_simulator():
    """
    Get simulator name that supported by MindQuantum.

    Returns:
        list, The supported simulator list.
    """
    return list(SUPPORTED_SIMULATOR)


class Simulator:
    """
    Quantum simulator that simulate quantum circuit.

    Args:
        backend (str): which backend you want. The supported backend can be found
            in SUPPORTED_SIMULATOR
        n_qubits (int): number of quantum simulator.
        seed (int): the random seed for this simulator, if ``None``, seed will generate
            by `numpy.random.randint`. Default: ``None``.
        dtype (mindquantum.dtype): the data type of simulator. Default: ``None``.

    Raises:
        TypeError: if `backend` is not str.
        TypeError: if `n_qubits` is not int.
        TypeError: if `seed` is not int.
        ValueError: if `backend` is not supported.
        ValueError: if `n_qubits` is negative.
        ValueError: if `seed` is less than 0 or great than 2**23 - 1.

    Examples:
        >>> from mindquantum.algorithm.library import qft
        >>> from mindquantum.simulator import Simulator
        >>> sim = Simulator('mqvector', 2)
        >>> sim.apply_circuit(qft(range(2)))
        >>> sim.get_qs()
        array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
    """

    def __init__(self, backend, n_qubits, *args, seed=None, dtype=None, **kwargs):
        """Initialize a Simulator object."""
        if isinstance(backend, BackendBase):
            self.backend = backend
        else:
            _check_input_type('backend', str, backend)
            _check_int_type('n_qubits', n_qubits)
            _check_value_should_not_less('n_qubits', 0, n_qubits)
            if seed is None:
                seed = np.random.randint(1, 2**23)
            _check_seed(seed)
            self.backend = SUPPORTED_SIMULATOR.py_class(backend)(backend, n_qubits, seed, dtype, *args, **kwargs)

    def astype(self, dtype, seed=None):
        """
        Convert simulator to other data type.

        Note:
            The quantum state will copied from origin simulator.

        Args:
            dtype (mindquantum.dtype): the data type of new simulator.
            seed (int): the seed of new simulator. Default: ``None``.
        """
        if seed is None:
            seed = np.random.randint(1, 2**23)
        _check_seed(seed)
        return Simulator(self.backend.astype(to_mq_type(dtype), seed=seed), self.n_qubits)

    @property
    def dtype(self):
        """Get data type of simulator."""
        return self.backend.dtype

    @property
    def n_qubits(self):
        """
        Get simulator qubit.

        Returns:
            int, the qubit number of simulator.
        """
        return self.backend.n_qubits

    def copy(self):
        """
        Copy this simulator.

        Returns:
            Simulator, a copy version of this simulator.

        Examples:
            >>> from mindquantum.core.gates import RX
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('mqvector', 1)
            >>> sim.apply_gate(RX(1).on(0))
            >>> sim2 = sim.copy()
            >>> sim2.apply_gate(RX(-1).on(0))
            >>> sim2
            mqvector simulator with 1 qubit (little endian).
            Current quantum state:
            1¦0⟩
        """
        return self.__class__(self.backend.copy(), None)

    def __str__(self):
        """Return a string representation of the object."""
        return self.backend.__str__()

    def __repr__(self):
        """Return a string representation of the object."""
        return self.backend.__repr__()

    def reset(self):
        """
        Reset simulator to zero state.

        Examples:
            >>> from mindquantum.algorithm.library import qft
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('mqvector', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.reset()
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
        """
        self.backend.reset()

    def apply_gate(self, gate, pr=None, diff=False):
        """
        Apply a gate on this simulator, can be a quantum gate or a measurement operator.

        Args:
            gate (BasicGate): The gate you want to apply.
            pr (Union[numbers.Number, numpy.ndarray, ParameterResolver, list]): The
                parameter for parameterized gate. Default: ``None``.
            diff (bool): Whether to apply the derivative gate on this simulator. Default: ``False``.

        Returns:
            int or None, if the gate if a measure gate, then return a collapsed state, Otherwise
            return None.

        Raises:
            TypeError: if `gate` is not a BasicGate.
            ValueError: if any qubit of `gate` is higher than simulator qubits.
            ValueError: if `gate` is parameterized, but no parameter supplied.
            TypeError: the `pr` is not a ParameterResolver if `gate` is parameterized.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.gates import RY, Measure
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('mqvector', 1)
            >>> sim.apply_gate(RY('a').on(0), np.pi/2)
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
            >>> sim.apply_gate(Measure().on(0))
            1
            >>> sim.get_qs()
            array([0.+0.j, 1.+0.j])
        """
        return self.backend.apply_gate(gate, pr, diff)

    def apply_circuit(self, circuit, pr=None):
        """
        Apply a circuit on this simulator.

        Args:
            circuit (Circuit): The quantum circuit you want to apply on this simulator.
            pr (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): The
                parameter resolver for this circuit. If the circuit is not parameterized,
                this arg should be ``None``. Default: ``None``.

        Returns:
            MeasureResult or None, if the circuit has measure gate, then return a MeasureResult,
            otherwise return None.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.gates import H
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('mqvector', 2)
            >>> sim.apply_circuit(Circuit().un(H, 2))
            >>> sim.apply_circuit(Circuit().ry('a', 0).ry('b', 1), np.array([1.1, 2.2]))
            >>> sim
            mqvector simulator with 2 qubits  (little endian).
            Current quantum state:
            -0.0721702531972066¦00⟩
            -0.30090405886869676¦01⟩
            0.22178317006196263¦10⟩
            0.9246947752567126¦11⟩
            >>> sim.apply_circuit(Circuit().measure(0).measure(1))
            shots: 1
            Keys: q1 q0│0.00     0.2         0.4         0.6         0.8         1.0
            ───────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                     11│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                       │
            {'11': 1}
        """
        return self.backend.apply_circuit(circuit, pr)

    def sampling(self, circuit, pr=None, shots=1, seed=None):
        """
        Sample the measure qubit in circuit.

        Sampling does not change the origin quantum state of this simulator.

        Args:
            circuit (Circuit): The circuit that you want to evolution and do sampling.
            pr (Union[None, dict, ParameterResolver]): The parameter
                resolver for this circuit, if this circuit is a parameterized circuit.
                Default: ``None``.
            shots (int): How many shots you want to sampling this circuit. Default: ``1``.
            seed (int): Random seed for random sampling. If ``None``, seed will be a random
                int number. Default: ``None``.

        Returns:
            MeasureResult, the measure result of sampling.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.gates import Measure
            >>> from mindquantum.simulator import Simulator
            >>> circ = Circuit().ry('a', 0).ry('b', 1)
            >>> circ += Measure('q0_0').on(0)
            >>> circ += Measure('q0_1').on(0)
            >>> circ += Measure('q1').on(1)
            >>> sim = Simulator('mqvector', circ.n_qubits)
            >>> res = sim.sampling(circ, {'a': 1.1, 'b': 2.2}, shots=100, seed=42)
            >>> res
            shots: 100
            Keys: q1 q0_1 q0_0│0.00   0.122       0.245       0.367        0.49       0.612
            ──────────────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
                           000│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                              │
                           011│▒▒▒▒▒▒▒▒▒
                              │
                           100│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                              │
                           111│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                              │
            {'000': 18, '011': 9, '100': 49, '111': 24}
        """
        return self.backend.sampling(circuit, pr, shots, seed)

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        """
        Apply hamiltonian to a simulator, this hamiltonian can be hermitian or non hermitian.

        Note:
            The quantum state may be not a normalized quantum state after apply hamiltonian.

        Args:
            hamiltonian (Hamiltonian): the hamiltonian you want to apply.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
            >>> from mindquantum.simulator import Simulator
            >>> import scipy.sparse as sp
            >>> sim = Simulator('mqvector', 1)
            >>> sim.apply_circuit(Circuit().h(0))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
            >>> ham1 = Hamiltonian(QubitOperator('Z0'))
            >>> sim.apply_hamiltonian(ham1)
            >>> sim.get_qs()
            array([ 0.70710678+0.j, -0.70710678+0.j])
            >>> sim.reset()
            >>> ham2 = Hamiltonian(sp.csr_matrix([[1, 2], [3, 4]]))
            >>> sim.apply_hamiltonian(ham2)
            >>> sim.get_qs()
            array([1.+0.j, 3.+0.j])
        """
        self.backend.apply_hamiltonian(hamiltonian)

    # pylint: disable=too-many-arguments
    def get_expectation(self, hamiltonian, circ_right=None, circ_left=None, simulator_left=None, pr=None):
        r"""
        Get expectation of the given hamiltonian. The hamiltonian could be non hermitian.

        This method is designed to calculate the expectation shown as below.

        .. math::

            E = \left<\varphi\right|U_l^\dagger H U_r \left|\psi\right>

        where :math:`U_l` is circ_left, :math:`U_r` is circ_right, :math:`H` is hams
        and :math:`\left|\psi\right>` is the current quantum state of this simulator,
        and :math:`\left|\varphi\right>` is the quantum state of `simulator_left`.

        Args:
            hamiltonian (Hamiltonian): The hamiltonian you want to get expectation.
            circ_right (Circuit): The :math:`U_r` circuit described above. If it is ``None``,
                we will use empty circuit. Default: ``None``.
            circ_left (Circuit): The :math:`U_l` circuit described above. If it is ``None``,
                then it will be the same as ``circ_right``. Default: ``None``.
            simulator_left (Simulator): The simulator that contains :math:`\left|\varphi\right>`. If
                ``None``, then :math:`\left|\varphi\right>` is assumed to be equals to :math:`\left|\psi\right>`.
                Default: ``None``.
            pr (Union[Dict[str, numbers.Number], ParameterResolver]): the
                variable value of circuit. Default: ``None``.

        Returns:
            numbers.Number, the expectation value.

        Examples:
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('mqvector', 1)
            >>> sim.apply_circuit(Circuit().ry(1.2, 0))
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> sim.get_expectation(ham)
            (0.36235775447667357+0j)
            >>> sim.get_expectation(ham, Circuit().rx('a', 0), Circuit().ry(2.3, 0), pr={'a': 2.4})
            (-0.25463350745693886+0.8507316752782879j)
            >>> sim1, sim2 = Simulator('mqvector', 1), Simulator('mqvector', 1)
            >>> sim1.apply_circuit(Circuit().ry(1.2, 0).rx(2.4, 0))
            >>> sim2.apply_circuit(Circuit().ry(1.2, 0).ry(2.3, 0))
            >>> sim1.apply_hamiltonian(ham)
            >>> from mindquantum.simulator import inner_product
            >>> inner_product(sim2, sim1)
            (-0.25463350745693886+0.8507316752782879j)
        """
        return self.backend.get_expectation(hamiltonian, circ_right, circ_left, simulator_left, pr)

    def set_threads_number(self, number):
        """
        Set maximum number of threads.

        Args:
            number (int): The thread number the simulator will use for thread pool.
        """
        return self.backend.set_threads_number(number)

    def get_qs(self, ket=False):
        """
        Get current quantum state of this simulator.

        Args:
            ket (bool): Whether to return the quantum state in ket format or not.
                Default: ``False``.

        Returns:
            numpy.ndarray, the current quantum state.

        Examples:
            >>> from mindquantum.algorithm.library import qft
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('mqvector', 2)
            >>> sim.apply_circuit(qft(range(2)))
            >>> sim.get_qs()
            array([0.5+0.j, 0.5+0.j, 0.5+0.j, 0.5+0.j])
        """
        return self.backend.get_qs(ket)

    def set_qs(self, quantum_state):
        """
        Set quantum state for this simulation.

        Args:
            quantum_state (numpy.ndarray): the quantum state that you want.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.simulator import Simulator
            >>> sim = Simulator('mqvector', 1)
            >>> sim.get_qs()
            array([1.+0.j, 0.+0.j])
            >>> sim.set_qs(np.array([1, 1]))
            >>> sim.get_qs()
            array([0.70710678+0.j, 0.70710678+0.j])
        """
        self.backend.set_qs(quantum_state)

    # pylint: disable=too-many-arguments
    def get_expectation_with_grad(
        self,
        hams,
        circ_right,
        circ_left=None,
        simulator_left=None,
        parallel_worker=None,
    ):
        r"""
        Get a function that return the forward value and gradient w.r.t circuit parameters.

        This method is designed to calculate the expectation and its gradient shown as below.

        .. math::

            E = \left<\varphi\right|U_l^\dagger H U_r \left|\psi\right>

        where :math:`U_l` is circ_left, :math:`U_r` is circ_right, :math:`H` is hams
        and :math:`\left|\psi\right>` is the current quantum state of this simulator,
        and :math:`\left|\varphi\right>` is the quantum state of `simulator_left`.

        Args:
            hams (Hamiltonian): The hamiltonian that need to get expectation.
            circ_right (Circuit): The :math:`U_r` circuit described above.
            circ_left (Circuit): The :math:`U_l` circuit described above. By default, this circuit
                will be ``none``, and in this situation, :math:`U_l` will be equals to
                :math:`U_r`. Default: ``None``.
            simulator_left (Simulator): The simulator that contains :math:`\left|\varphi\right>`. If
                ``None``, then :math:`\left|\varphi\right>` is assumed to be equals to :math:`\left|\psi\right>`.
                Default: ``None``.
            parallel_worker (int): The parallel worker numbers. The parallel workers can handle
                batch in parallel threads. Default: No``ne.

        Returns:
            GradOpsWrapper, a grad ops wrapper than contains information to generate this grad ops.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.core.operators import QubitOperator, Hamiltonian
            >>> from mindquantum.simulator import Simulator
            >>> circ = Circuit().ry('a', 0)
            >>> ham = Hamiltonian(QubitOperator('Z0'))
            >>> sim = Simulator('mqvector', 1)
            >>> grad_ops = sim.get_expectation_with_grad(ham, circ)
            >>> grad_ops(np.array([1.0]))
            (array([[0.54030231+0.j]]), array([[[-0.84147098+0.j]]]))
            >>> sim1 = Simulator('mqvector', 1)
            >>> prep_circ = Circuit().h(0)
            >>> ansatz = Circuit().ry('a', 0).rz('b', 0).ry('c', 0)
            >>> sim1.apply_circuit(prep_circ)
            >>> sim2 = Simulator('mqvector', 1)
            >>> ham = Hamiltonian(QubitOperator(""))
            >>> grad_ops = sim2.get_expectation_with_grad(ham, ansatz, Circuit(), simulator_left=sim1)
            >>> f, g = grad_ops(np.array([7.902762e-01, 2.139225e-04, 7.795934e-01]))
            >>> f
            array([[0.99999989-7.52279618e-05j]])
        """
        if self.backend.name == "mqmatrix":
            if circ_left is not None:
                raise ValueError("Density matrix simulator doesn't support circ_left.")
            if simulator_left is not None:
                raise ValueError("Density matrix simulator doesn't support simulator_left.")
        return self.backend.get_expectation_with_grad(
            hams,
            circ_right,
            circ_left,
            (simulator_left.backend if simulator_left is not None else None),
            parallel_worker,
        )


def inner_product(bra_simulator: Simulator, ket_simulator: Simulator):
    """
    Calculate the inner product of two state that in the given simulator.

    Args:
        bra_simulator (Simulator): The simulator that serve as bra state.
        ket_simulator (Simulator): The simulator that serve as ket state.

    Returns:
        numbers.Number, the inner product of two quantum state.

    Examples:
        >>> from mindquantum.core.gates import RX, RY
        >>> from mindquantum.simulator import inner_product, Simulator
        >>> bra_simulator = Simulator('mqvector', 1)
        >>> bra_simulator.apply_gate(RY(1.2).on(0))
        >>> ket_simulator = Simulator('mqvector', 1)
        >>> ket_simulator.apply_gate(RX(2.3).on(0))
        >>> inner_product(bra_simulator, ket_simulator)
        (0.33713923320500694-0.5153852888544989j)
    """
    _check_input_type('bra_simulator', Simulator, bra_simulator)
    _check_input_type('ket_simulator', Simulator, ket_simulator)
    if bra_simulator.n_qubits != ket_simulator.n_qubits:
        raise ValueError(
            "Two simulator should have same quantum state, "
            f"but get {bra_simulator.n_qubits} and {ket_simulator.n_qubits}."
        )
    if bra_simulator.backend.name != ket_simulator.backend.name:
        raise ValueError("The backend of two simulator should be same.")
    if bra_simulator.dtype != ket_simulator.dtype:
        raise TypeError(
            f"Data type of two simulator are different: bra_simulator is {bra_simulator.dtype}, "
            f"while ket_simulator is {ket_simulator.dtype}."
        )
    if isinstance(bra_simulator.backend, MQSim):
        return MQBlas.inner_product(bra_simulator.backend, ket_simulator.backend)
    raise NotImplementedError(f"inner_product for backend {bra_simulator.backend} not implement.")


__all__ = ['Simulator', 'get_supported_simulator', 'inner_product']
