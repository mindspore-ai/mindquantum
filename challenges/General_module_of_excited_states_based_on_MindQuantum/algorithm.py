from mindquantum.simulator import Simulator
from mindquantum.core import Hamiltonian, QubitOperator, Circuit, X
import numpy as np

class BaseVQEClass:
    """Base class for VQE algorithms."""
    def _initialize_simulator(self):
        """Initialize the simulator if not already done."""
        if not hasattr(self, "sim") or self.sim is None:
            self.sim = Simulator(self.backend, self.n_qubits)

    def _initialize_energy_evaluator(self):
        """Initialize the energy evaluator if not already done."""
        self._initialize_simulator()
        if not hasattr(self, "energy_op") or self.energy_op is None:
            self.energy_op = self.sim.get_expectation_with_grad(
                Hamiltonian(self.ham), self.ansatz)
            
    def _generate_operators(self):
        """This function generate the operators for the objective function and its gradient.
        """
        if not hasattr(self, "ops") or self.ops is None:
            self.ops = []
            self.ops.append(Hamiltonian(self.ham))

    def _initialize_grad_evaluator(self):
        """Initialize the gradient evaluator if not already done."""
        self._initialize_simulator()
        self._generate_operators()

        if not hasattr(self, "grad_op") or self.grad_op is None:
            self.grad_op = self.sim.get_expectation_with_grad(
                self.ops, self.ansatz)
            
    def energy(self, params):
        """Evaluate the energy of the current ansatz."""
        self._initialize_energy_evaluator()
        f, _ = self.energy_op(params)
        return f[0, 0].real


class Folded_spectrum(BaseVQEClass):

    def __init__(self, n_qubits, ansatz, ham, target_energy, backend='mqvector'):
        """This is a implementation of folded spectrum method for VQE.

        Args:
            n_qubits: the number of qubits (the target system size)
            ansatz: the ansatz circuit
            ham: the hamiltonian of the target system
            target_energy: the approximate energy of around the eigenenergy of the target eigenstate
        """
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.ham = ham
        self.target_energy = target_energy
        self.backend = backend

    def evaluate(self, params):
        """This function evaluate the objective function and its gradient.

        Args:
            params: the parameters of the ansatz circuit

        Returns:
            the value of the objective function and its gradient
        """
        self._initialize_simulator()
        self._generate_operators()
        self._initialize_grad_evaluator()
            
        f, g = self.grad_op(params)
        f1, f2 = f[0, 0].real, f[0, 1].real
        g1, g2 = np.array(g[0, 0, :].real), np.array(g[0, 1, :].real)

        fval = f1 + f2 + self.target_energy**2
        gval = g1 + g2
        return fval, gval

    def _generate_operators(self):
        self.ops = []
        self.ops.append(Hamiltonian(self.ham**2))
        self.ops.append(Hamiltonian(-2 * self.target_energy * self.ham))


class Orthogonally_constrained(BaseVQEClass):

    def __init__(self, n_qubits, ham, backend='mqvector'):
        """This is a implementation of orthogonally constrained method for VQE.

        Args:
            n_qubits: the number of qubits (the target system size)
            ham (_type_): the hamiltonian of the target system
        """
        self.n_qubits = n_qubits
        self.ham = ham
        self.backend = backend

        self.betas = []
        self.overlap_op = []
        self.converged_circs = []

        self.ansatz = None

    def evaluate(self, params):
        """This function evaluate the objective function and its gradient.

        Args:
            params: the parameters of the ansatz circuit

        Returns:
            the value of the objective function and its gradient
        """
        self._initialize_simulator()
        self._generate_operators()
        self._initialize_grad_evaluator()

        f, g = self.grad_op(params)
        gval = np.array(g[0, 0, :].real)
        fval = f[0, 0].real

        for i, op in enumerate(self.overlap_op):
            ff, gg = op(params)
            f1 = ff[0, 0]
            g1 = gg[0, 0, :]

            fval += self.betas[i] * (np.abs(f1)**2)
            gval += self.betas[i] * (f1 * np.conj(np.array(g1)) +
                                     np.conj(f1) * np.array(g1)).real
        return fval, gval

    def initial_next_eigenstate(self, new_ansatz, beta, converged_params):
        """This function update the ansatz circuit and the overlap operators. It is used for generating the next eigenstate.

        Args:
            new_ansatz: the new ansatz circuit
            beta: the weight of new overlap penalty term
            converged_params: the parameters of the converged ansatz circuit of the last eigenstate
        """
        converged_circs = None
        if self.ansatz is not None:
            self.betas.append(beta)
            converged_circs = self.ansatz.apply_value(
                dict(zip(self.ansatz.params_name, converged_params)))

        self.ansatz = new_ansatz
        self.energy_op = None
        self.grad_op = None

        if converged_circs is not None:
            op = self.sim.get_expectation_with_grad(
                Hamiltonian(QubitOperator("")), self.ansatz, converged_circs)
            self.overlap_op.append(op)



class Subspace_expansion(BaseVQEClass):

    def __init__(self, n_qubits, ansatz, ham, excitation_ops, backend='mqvector'):
        """This is a implementation of subspace expansion method for VQE.

        Args:
            n_qubits: the number of qubits (the target system size)
            ansatz: the ansatz circuit
            ham: the hamiltonian of the target system
            excitation_ops: the excitation operators to expand the subspace
        """
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.ham = ham
        self.excitation_ops = excitation_ops
        self.backend = backend

    def evaluate(self, params):
        """This function evaluate the objective function and its gradient.

        Args:
            params: the parameters of the ansatz circuit

        Returns:
            the value of the objective function and its gradient
        """
        self._initialize_simulator()
        self._generate_operators()
        self._initialize_grad_evaluator()
        f, g = self.grad_op(params)
        gval = np.array(g[0, 0, :].real)
        fval = f[0, 0].real

        return fval, gval

    def evaluate_matrics(self, converged_params):
        """This function evaluate the matrix elements of the subspace Hamiltonian and the overlap matrix.

        Args:
            converged_params: the parameters of the converged ansatz circuit of the ground state

        Returns:
            the subspace Hamiltonian matrix and the overlap matrix
        """
        self._initialize_simulator()
        converged_circs = self.ansatz.apply_value(
            dict(zip(self.ansatz.params_name, converged_params)))
        
        ops_number = len(self.excitation_ops)
        self.ham_sub = np.zeros((ops_number, ops_number))
        self.s_mat = np.zeros((ops_number, ops_number))

        self.sim.reset()
        self.sim.apply_circuit(converged_circs)

        for i in range(ops_number):
            for j in range(ops_number):
                hij = self.sim.get_expectation(
                    Hamiltonian(self.excitation_ops[i].hermitian() * self.ham *
                                self.excitation_ops[j]))
                sij = self.sim.get_expectation(
                    Hamiltonian(self.excitation_ops[i].hermitian() *
                                self.excitation_ops[j]))
                self.ham_sub[i][j] = hij.real
                self.s_mat[i][j] = sij.real
        self.sim.reset()
        return self.ham_sub, self.s_mat


class Base_subspace_search(BaseVQEClass):
    def __init__(self, n_qubits, init_circs, ansatz, ham, k, backend='mqvector'):
        """This is a base class of subspace search based methods.

        Args:
            n_qubits: the number of qubits (the target system size)
            init_circs: the initial circuits to generate the orthogonal initial states
            ansatz: the ansatz circuit
            ham: the hamiltonian of the target system
            k: the number of excited states to be generated
        """
        self.n_qubits = n_qubits
        self.ansatz = ansatz
        self.ham = ham
        self.k = k
        self.init_circs = init_circs
        self.backend = backend

    def _initialize_grad_evaluator(self):
        """This function generate the operators for evaluators of the objective function and its gradient.
        """
        self._initialize_simulator()
        self.grad_ops = []
        for i in range(self.k):
            self.grad_ops.append(
                self.sim.get_expectation_with_grad(
                    Hamiltonian(self.ham), self.init_circs[i] + self.ansatz))

    def energy(self, params):
        """This function evaluate the average energy of each generated states with respect to the target Hamiltonian.

        Args:
            params: the parameters of the ansatz circuit

        Returns:
            the value of the average energy
        """
        self._initialize_simulator()
        energies = []
        for i in range(self.k):
            f, _ = self.grad_ops[i](params)
            energies.append(f[0, 0].real)
        return energies

class Subspace_search(Base_subspace_search):
    """This is a implementation of subspace search method for VQE."""

    def evaluate(self, params):
        """This function evaluate the objective function and its gradient.

        Args:
            params: the parameters of the ansatz circuit

        Returns:
            the value of the objective function and its gradient
        """
        self._initialize_simulator()
        self._initialize_grad_evaluator()

        fval = 0
        gval = 0
        for i in range(self.k):
            f, g = self.grad_ops[i](params)
            g1 = np.array(g[0, 0, :].real)
            f1 = f[0, 0].real
            fval += (self.k - i) * f1
            gval += (self.k - i) * g1
        return fval, gval



class Multistate_constracted(Base_subspace_search):
    """This is a implementation of multistate constracted method for VQE."""

    def evaluate(self, params):
        """This function evaluate the objective function and its gradient.

        Args:
            params: the parameters of the ansatz circuit

        Returns:
            the value of the objective function and its gradient
        """
        self._initialize_simulator()
        self._initialize_grad_evaluator()

        fval = 0
        gval = 0
        for i in range(self.k):
            f, g = self.grad_ops[i](params)
            g1 = np.array(g[0, 0, :].real)
            f1 = f[0, 0].real
            fval += f1
            gval += g1
        return fval, gval

    def evaluate_matrics(self, converged_params):
        """This function evaluate the matrix elements of the subspace Hamiltonian (in this case is the overlap matrix between different generated states).

        Args:
            converged_params: the parameters of the converged ansatz circuit

        Returns:
            the subspace Hamiltonian matrix
        """
        self._initialize_simulator()

        converged_circs = self.ansatz.apply_value(
            dict(zip(self.ansatz.params_name, converged_params)))

        evaluators = np.zeros((self.k, self.k), dtype=object)
        for i in range(self.k):
            for j in range(self.k):
                evaluators[i][j] = self.sim.get_expectation_with_grad(
                    Hamiltonian(self.ham),
                    self.init_circs[i] + converged_circs,
                    self.init_circs[j] + converged_circs)

        self.ham_sub = np.zeros((self.k, self.k))

        for i in range(self.k):
            for j in range(self.k):
                hij, _ = evaluators[i][j](np.array([]))
                self.ham_sub[i][j] = hij[0, 0].real
        return self.ham_sub
    

class Orthogonal_state_reduction(BaseVQEClass):

    def __init__(self, n_qubits, init_state_bitstring, ham):
        """This is a implementation of orthogonal state reduction method for VQE.

        Args:
            n_qubits: the number of qubits (the target system size)
            init_state_bitstring: the bitstring of the initial product state
            ham: the hamiltonian of the target system
        """
        self.n_qubits = n_qubits
        self.ham = ham
        self.init_state_bitstring = init_state_bitstring
        self.converged_circs = []

        self.ansatz = None
        self.variational_block = None

    def evaluate(self, params):
        """This function evaluate the objective function and its gradient.

        Args:
            params: the parameters of the ansatz circuit

        Returns:
            the value of the objective function and its gradient
        """
        self._initialize_simulator()
        self._generate_operators()
        self._initialize_grad_evaluator()

        f, g = self.grad_op(params)
        g1 = np.array(g[0, 0, :].real)
        f1 = f[0, 0].real
        return f1, g1
    
    
    def initial_next_eigenstate(self, new_ansatz, converged_params):
        """This function update the ansatz circuit and the corresponding gradient evaluator. It is used for generating the next eigenstate.

        Args:
            new_ansatz: the ansatz circuit for the next eigenstate
            converged_params: the parameters of the converged ansatz circuit of the last eigenstate
        """
        converged_circ = None
        if self.variational_block is not None:
            converged_circ = self.variational_block.apply_value(
                dict(zip(self.variational_block.params_name, converged_params)))
            self.converged_circs.append(converged_circ)
        
        self.variational_block = new_ansatz
        self.ansatz = self._initial_circuit(self.init_state_bitstring)
        self.ansatz += new_ansatz

        k = len(self.converged_circs)
        for i in range(k):
            self.ansatz += self._multi_controlled_x(self.init_state_bitstring, self.n_qubits + i)
            self.ansatz += self.converged_circs[-i-1]

        self.sim = Simulator('mqvector', self.n_qubits + k)
        
        new_qubit_op = self.ham
        for i in range(k):
            new_op = (QubitOperator(f'I{self.n_qubits + i}') + QubitOperator(f'Z{self.n_qubits + i}')) / 2
            new_qubit_op *= new_op
        new_qubit_op = Hamiltonian(new_qubit_op)
        self.grad_op = self.sim.get_expectation_with_grad(new_qubit_op, self.ansatz)
        self.energy_op = None

            
    def _initial_circuit(self, bitstring):
        """This function generate the initialization circuit according to the given bitstring respresenting the initial product state.

        Args:
            bitstring: the bitstring of the initial product state

        Returns:
            the initialization circuit
        """
        circ = Circuit()
        for i, c in enumerate(bitstring):
            if c == '1':
                circ += X.on(i)
        return circ
    
    def _multi_controlled_x(self, bitstring, obj_qubit):
        """This function generate the multi-controlled X gate according to the given bitstring and the target qubit.

        Args:
            bitstring: the bitstring of the initial product state
            obj_qubit: the target qubit, in which the X is operated on

        Returns:
            the multi-controlled X gate circuit
        """
        circ = Circuit()
        for i, c in enumerate(bitstring):
            if c == '0':
                circ += X.on(i)
        circ += X.on(obj_qubit, list(range(len(bitstring))))
        for i, c in enumerate(bitstring):
            if c == '0':
                circ += X.on(i)
        return circ
    