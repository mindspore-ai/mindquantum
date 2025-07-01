# Copyright 2025 Huawei Technologies Co., Ltd
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
"""MQ Chemistry Simulator."""
# pylint: disable=c-extension-no-member

from typing import Callable, Iterable, Union

import numpy as np

from ...utils.string_utils import ket_string
from ...utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
    _check_value_should_between_close_set,
    _check_seed,
    _check_number_type,
    _check_and_generate_pr_type,
)
from ...core.circuit import Circuit
from .ucc_excitation_gate import UCCExcitationGate
from .ci_hamiltonian import CIHamiltonian
from ...core.parameterresolver import ParameterResolver
from ... import _mq_chem


class MQChemSimulator:
    """
    A simulator for quantum chemistry based on the Configuration Interaction (CI) method.

    This simulator is optimized for chemistry problems by working within a CI vector space,
    which is a subspace of the full Hilbert space defined by a fixed number of electrons.
    This approach is significantly more memory-efficient than using a full state-vector
    simulator for typical chemistry simulations.

    The simulator is designed to work with
    :class:`~.simulator.mqchem.UCCExcitationGate` and
    :class:`~.simulator.mqchem.CIHamiltonian`. It provides methods to apply UCC circuits,
    calculate Hamiltonian expectation values, and compute gradients required for
    variational quantum algorithms like VQE.

    By default, the simulator is initialized in the Hartree-Fock state, which serves as
    the typical reference state for quantum chemistry calculations.

    Args:
        n_qubits (int): The total number of qubits (spin-orbitals) in the system.
        n_electrons (int): The number of electrons, which defines the dimension of the CI space.
        seed (int): The random seed for this simulator. If ``None``, a random seed will be
            generated. Default: ``None``.
        dtype (str): The data type for simulation, either ``"float"`` or ``"double"``.
            Default: ``"double"``.

    Examples:
        >>> from mindquantum.simulator import mqchem
        >>> sim = mqchem.MQChemSimulator(4, 2, seed=42)
        >>> sim.n_qubits
        4
        >>> sim.n_electrons
        2
        >>> # The simulator starts in the Hartree-Fock state |0011>
        >>> print(sim.get_qs(ket=True))
        1¦0011⟩
    """

    def __init__(self, n_qubits, n_electrons, seed=None, dtype="double"):
        """Initialize a MQChemSimulator."""
        _check_int_type("n_qubits", n_qubits)
        _check_value_should_not_less("n_qubits", 1, n_qubits)
        _check_int_type("n_electrons", n_electrons)
        _check_value_should_not_less("n_electrons", 0, n_electrons)
        _check_value_should_between_close_set("n_electrons", 0, n_qubits, n_electrons)
        if seed is None:
            seed = np.random.randint(1, 2**23)
        else:
            _check_seed(seed)
        _check_input_type("dtype", str, dtype)
        if dtype == "float":
            mod = _mq_chem.float
        elif dtype == "double":
            mod = _mq_chem.double
        else:
            raise ValueError(f"Unsupported dtype '{dtype}', use 'float' or 'double'.")
        self._sim = mod.mqchem(n_qubits, n_electrons, seed)
        self.n_qubits = n_qubits
        self.n_electrons = n_electrons
        self._backend = mod

    def reset(self):
        """
        Reset the simulator's state to the Hartree-Fock (HF) state.

        The Hartree-Fock state is the ground state of a non-interacting fermionic system,
        where the `n_electrons` lowest-energy spin-orbitals are occupied. In the
        computational basis, this corresponds to the state :math:`|11...100...0⟩`.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> from mindquantum.simulator import mqchem
            >>> sim = mqchem.MQChemSimulator(4, 2)
            >>> t = FermionOperator('2^ 0', 0.1)
            >>> ucc_gate = mqchem.UCCExcitationGate(t)
            >>> sim.apply_gate(ucc_gate) # State is no longer HF state
            >>> sim.reset()
            >>> print(sim.get_qs(ket=True))
            1¦0011⟩
        """
        self._sim.reset()

    def get_qs(self, ket: bool = False):
        r"""
        Get the current quantum state of the simulator.

        While the simulator internally stores the state as a compact CI vector, this method
        returns the state as a dense state vector in the full :math:`2^{n_{qubits}}`
        dimensional computational basis.

        Args:
            ket (bool): If ``True``, returns the quantum state in Dirac (ket) notation as a string.
                If ``False``, returns the state as a NumPy array. Default: ``False``.

        Returns:
            Union[numpy.ndarray, str]: The quantum state vector as a NumPy array or a string
            in ket notation.

        Raises:
            TypeError: If `ket` is not a boolean.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.simulator import mqchem
            >>> sim = mqchem.MQChemSimulator(4, 2)
            >>> # Get state vector (Hartree-Fock |0011>)
            >>> np.round(sim.get_qs(), 2)
            array([0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j,
                   0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
            >>> # Get state in ket string format
            >>> print(sim.get_qs(ket=True))
            1¦0011⟩
        """
        if not isinstance(ket, bool):
            raise TypeError(f"ket requires a bool, but get {type(ket)}")
        state = self._sim.get_state_vector(self.n_qubits)
        if ket:
            return '\n'.join(ket_string(state))
        return state

    def set_qs(self, qs_rep):
        """
        Set the CI vector from a sparse representation.

        Args:
            qs_rep (List[Tuple[int, complex]]): A list of tuples, where each tuple
                `(mask, amplitude)` defines the amplitude of a basis state. `mask` is an
                integer representing the computational basis state (Slater determinant),
                and `amplitude` is its corresponding complex amplitude. All basis states
                in `qs_rep` must have a population count equal to `n_electrons`.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.simulator import mqchem
            >>> sim = mqchem.MQChemSimulator(4, 2)
            >>> # Prepare a state that is a superposition of |0101> and |1001>
            >>> # Note: masks 5 (0101) and 9 (1001) both have 2 electrons.
            >>> new_state_rep = [(5, 1/np.sqrt(2)), (9, 1/np.sqrt(2))]
            >>> sim.set_qs(new_state_rep)
            >>> print(sim.get_qs(ket=True))
            √2/2¦0101⟩
            √2/2¦1001⟩
        """
        _check_input_type("qs_rep", (list, tuple), qs_rep)
        for mask, amp in qs_rep:
            _check_int_type("mask", mask)
            _check_value_should_not_less("mask", 0, mask)
            _check_value_should_between_close_set("mask", 0, (1 << self.n_qubits) - 1, mask)
            _check_number_type("amplitude", amp)
            if bin(mask).count("1") != self.n_electrons:
                raise ValueError(f"mask {mask} has {bin(mask).count('1')} electrons, expected {self.n_electrons}")
        self._sim.set_qs(qs_rep)

    def apply_circuit(
        self,
        circuit: Union[Circuit, Iterable[UCCExcitationGate]],
        pr: ParameterResolver = None,
    ) -> None:
        r"""
        Apply a quantum circuit to the current simulator state.

        Note:
            Only :class:`~.simulator.mqchem.UCCExcitationGate` instances within the circuit
            will be applied; all other gate types are ignored.

        Args:
            circuit (Union[Circuit, Iterable[UCCExcitationGate]]): The quantum circuit or an
                iterable of UCC gates to apply.
            pr (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): A parameter
                resolver to substitute parameter values. If ``None``, the parameters defined
                in the gates are used directly. Default: ``None``.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.operators import FermionOperator
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.simulator import mqchem
            >>> sim = mqchem.MQChemSimulator(4, 2)
            >>> t1 = FermionOperator('2^ 0', 'a')
            >>> t2 = FermionOperator('3^ 1', 'b')
            >>> circ = Circuit([mqchem.UCCExcitationGate(t1), mqchem.UCCExcitationGate(t2)])
            >>> sim.apply_circuit(circ, {'a': 0.1, 'b': 0.2})
            >>> print(sim.get_qs(ket=True))
            0.97517033¦0011⟩
            -0.0978434¦0110⟩
            0.19767681¦1001⟩
            0.01983384¦1100⟩
        """
        _check_input_type("circuit", (Circuit, Iterable), circuit)
        circuit_ = Circuit()
        if isinstance(circuit, Iterable):
            for gate in circuit:
                circuit_ += gate
        else:
            circuit_ = circuit
        gates = getattr(circuit_, 'gates', circuit_)
        cpp_circ = [
            gate.get_cpp_obj(self.n_qubits, self.n_electrons, self._backend)
            for gate in gates
            if isinstance(gate, UCCExcitationGate)
        ]
        if pr is None:
            # No external parameter mapping: C++ uses gate.coeff for constants or variables
            self._sim.apply_circuit(cpp_circ)
        else:
            pr = _check_and_generate_pr_type(pr, circuit_.params_name)
            self._sim.apply_circuit(cpp_circ, pr)

    def apply_gate(self, gate: UCCExcitationGate, pr: ParameterResolver = None) -> None:
        r"""
        Apply a single UCC excitation gate to the current simulator state.

        Args:
            gate (UCCExcitationGate): The UCC excitation gate to apply.
            pr (Union[ParameterResolver, dict, numpy.ndarray, list, numbers.Number]): A parameter
                resolver to substitute parameter values. If ``None``, the parameter defined
                in the gate is used directly. Default: ``None``.

        Raises:
            TypeError: If `gate` is not a :class:`~.simulator.mqchem.UCCExcitationGate`.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.operators import FermionOperator
            >>> from mindquantum.simulator import mqchem
            >>> sim = mqchem.MQChemSimulator(4, 2)
            >>> t = FermionOperator('2^ 0', 'theta')
            >>> gate = mqchem.UCCExcitationGate(t)
            >>> # Apply gate with theta = 0.1
            >>> sim.apply_gate(gate, {'theta': 0.1})
            >>> print(sim.get_qs(ket=True))
            0.99500417¦0011⟩
            -0.09983342¦0110⟩
        """
        _check_input_type("gate", UCCExcitationGate, gate)
        cpp_gate = gate.get_cpp_obj(self.n_qubits, self.n_electrons, self._backend)
        if pr is None:
            self._sim.apply_single_ucc_gate(cpp_gate)
        else:
            pr = _check_and_generate_pr_type(pr, gate.coeff.params_name)
            self._sim.apply_single_ucc_gate(cpp_gate, pr)

    def get_expectation(self, ham):
        r"""
        Compute the expectation value of a Hamiltonian with respect to the current state.

        Calculates :math:`\langle\psi|H|\psi\rangle`, where :math:`|\psi\rangle` is the
        current CI state vector and :math:`H` is a CI Hamiltonian.

        Args:
            ham (CIHamiltonian): The Hamiltonian for which to compute the expectation.

        Returns:
            float: The real-valued expectation value.

        Raises:
            TypeError: If `ham` is not a :class:`~.simulator.mqchem.CIHamiltonian`.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> from mindquantum.simulator import mqchem
            >>> sim = mqchem.MQChemSimulator(4, 2)
            >>> # Hartree-Fock state is |0011>
            >>> ham_op = FermionOperator('0^ 0') + FermionOperator('1^ 1')
            >>> ham = mqchem.CIHamiltonian(ham_op)
            >>> # Expectation of <HF| (n_0 + n_1) |HF> should be 1+1=2
            >>> sim.get_expectation(ham)
            2.0
        """
        _check_input_type("ham", CIHamiltonian, ham)
        return self._sim.get_expectation(ham.get_cpp_obj(self._backend, self.n_qubits, self.n_electrons))

    def get_expectation_with_grad(
        self,
        ham: CIHamiltonian,
        circuit: Union[Circuit, Iterable[UCCExcitationGate]],
    ) -> Callable:
        r"""
        Generate a function to compute the expectation value and its gradient.

        This method implements the adjoint differentiation method to calculate the gradient
        of the expectation value :math:`\langle\psi(\theta)|H|\psi(\theta)\rangle` with
        respect to the parameters :math:`\theta` of a UCC ansatz circuit.
        The state is prepared as :math:`|\psi(\theta)\rangle = U(\theta)|\psi_0\rangle`,
        where :math:`|\psi_0\rangle` is the current state of the simulator.

        Args:
            ham (CIHamiltonian): The Hamiltonian :math:`H`.
            circuit (Union[Circuit, Iterable[UCCExcitationGate]]): The parameterized UCC circuit
                :math:`U(\theta)`. The circuit must have parameters for gradient calculation.

        Returns:
            Callable: A function that accepts a NumPy array of parameter values `x` and
            returns a tuple `(expectation, gradient)`. `expectation` is the float
            expectation value, and `gradient` is a NumPy array containing the derivatives
            with respect to each parameter in `x`. The order of parameters is determined
            by `circuit.params_name`.

        Raises:
            TypeError: If `ham` is not a :class:`~.simulator.mqchem.CIHamiltonian`.

        Examples:
            >>> import numpy as np
            >>> from mindquantum.core.operators import FermionOperator
            >>> from mindquantum.core.circuit import Circuit
            >>> from mindquantum.simulator import mqchem
            >>>
            >>> # Prepare simulator, Hamiltonian, and ansatz
            >>> sim = mqchem.MQChemSimulator(4, 2)
            >>> ham_op = FermionOperator('0^ 0', 1) + FermionOperator('1^ 1', 1)
            >>> ham = mqchem.CIHamiltonian(ham_op)
            >>> t = FermionOperator('2^ 0', 'theta')
            >>> ansatz = Circuit(mqchem.UCCExcitationGate(t))
            >>>
            >>> # Get the gradient function
            >>> grad_fn = sim.get_expectation_with_grad(ham, ansatz)
            >>>
            >>> # Calculate expectation and gradient for theta = 0.1
            >>> params = np.array([0.1])
            >>> expectation, gradient = grad_fn(params)
            >>> print(f"Expectation: {expectation:.5f}")
            Expectation: 1.99003+0.00000j
            >>> print(f"Gradient: {gradient[0]:.5f}")
            Gradient: -0.19867+0.00000j
        """
        _check_input_type("ham", CIHamiltonian, ham)
        cpp_ham = ham.get_cpp_obj(self._backend, self.n_qubits, self.n_electrons)

        _check_input_type("circuit", (Circuit, Iterable), circuit)
        gates = getattr(circuit, 'gates', circuit)
        _check_input_type("circuit.gates", Iterable, gates)
        circ_cpp = [
            gate.get_cpp_obj(self.n_qubits, self.n_electrons, self._backend)
            for gate in gates
            if isinstance(gate, UCCExcitationGate)
        ]

        def grad_ops(x):
            x_arr = np.array(x)
            if x_arr.ndim != 1:
                raise ValueError(f"Expected 1D array for ansatz parameters, got shape {x_arr.shape}")
            ans_data = x_arr.tolist()
            enc_data = [[]]
            enc_name = []
            batch_threads = 1
            mea_threads = 1
            f_and_g = np.array(
                self._sim.get_expectation_with_grad_multi_multi(
                    cpp_ham, circ_cpp, enc_data, ans_data, enc_name, circuit.params_name, batch_threads, mea_threads
                )
            )
            f = f_and_g[0, 0, 0]
            g = f_and_g[0, 0, 1:]
            return f.real, g.real

        return grad_ops
