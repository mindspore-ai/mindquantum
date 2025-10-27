import numpy as np
import matplotlib.pyplot as plt
import logging
import dimod
import neal
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import RX, RZ, Rzz, H
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.core import ParameterResolver
from scipy.optimize import minimize

# Set NumPy printing options for better array readability
np.set_printoptions(linewidth=100)

# ---------------- Hill Climbing Algorithm ----------------
def hill_climbing(problem, x0, max_iter=50):
    """
    Simple Hill Climbing algorithm: iteratively flips one bit and accepts the new solution
    if it improves the objective function.

    Args:
        problem: The objective function to minimize. Takes a numpy array as input and returns a scalar.
        x0: Initial solution (numpy array of 0/1).
        max_iter: Maximum number of iterations.

    Returns:
        history: List of objective function values at each iteration.
        best_x: The best solution found.
        best_value: The objective function value of the best solution.
    """
    x = x0.copy()
    best_value = problem(x)
    best_x = x.copy()
    history = [best_value]

    for _ in range(max_iter):
        improved = False
        for i in range(len(x)):
            # Try flipping the i-th bit
            candidate = x.copy()
            candidate[i] = 1 - candidate[i]
            value = problem(candidate)
            # Accept the new solution if it improves the objective function
            if value < best_value:
                best_value = value
                best_x = candidate.copy()
                x = candidate
                improved = True
        history.append(best_value)
        # Stop if no improvement is possible
        if not improved:
            break
    return history, best_x, best_value


# ---------------- QUCOOP_hill Class ----------------
class QUCOOP_hill:
    """
    BBOB class defines a framework for solving optimization problems using 
    QUBO and Hamiltonian representations with quantum and classical methods.

    Attributes:
        problem: The objective function to minimize.
        n: The dimensionality of the problem.
        n_trans: Transformed dimensionality of the problem (usually equals n).
        P_sol: The probability distribution matrix of the solution (initialized as an identity matrix).
    """
    def __init__(self, problem):
        self.problem = problem
        self.n = problem.dimension
        self.n_trans = self.n
        self.P_sol = np.eye(self.n)

    # ---------------- Numerical Gradient ----------------
    def numerical_grad(self, f, x, eps=1e-6):
        """
        Compute the numerical gradient of the objective function.

        Args:
            f: Objective function.
            x: Input point (numpy array).
            eps: Small step size for numerical differentiation.

        Returns:
            grad: Gradient vector (numpy array).
        """
        x = np.array(x, dtype=float)
        grad = np.zeros_like(x)
        for i in range(len(x)):
            dx = np.zeros_like(x)
            dx[i] = eps
            grad[i] = (f(x + dx) - f(x - dx)) / (2 * eps)
        return grad

    # ---------------- Build QUBO ----------------
    def taylor_expansion_second_order(self, problem, x0, epsilon=1e-5):
        """
        Computes the second-order Taylor expansion information for the function `problem` at the point `x0`, 
        including the function value, gradient, and Hessian matrix.

        Parameters:
        - problem: Callable objective function, which takes a vector `x` as input and returns the corresponding function value (fitness).
        - x0: ndarray, the target point, input vector (point).
        - epsilon: float, a small perturbation value used for numerical computation of gradients and Hessians (default is 1e-5).

        Returns:
        - f_x0: float, the value of the function at `x0`.
        - grad: ndarray, the gradient of the function at `x0` (vector).
        - hessian: ndarray, the Hessian matrix (second-order derivatives) of the function at `x0`.
        """
        
        # Compute the function value f(x0)
        f_x0 = problem(x0)
        
        # Initialize gradient and Hessian
        n = len(x0)  # Dimensionality of the variables
        grad = np.zeros(n)  # Gradient vector
        hessian = np.zeros((n, n))  # Hessian matrix
        
        # Compute the gradient (∇f)
        for i in range(n):
            # Create unit directional perturbations
            x_forward = np.array(x0, dtype=float)
            x_backward = np.array(x0, dtype=float)
            x_forward[i] += epsilon
            x_backward[i] -= epsilon
            
            # Use central difference to compute the gradient
            grad[i] = (problem(x_forward) - problem(x_backward)) / (2 * epsilon)
        
        # Compute the Hessian (H)
        for i in range(n):
            for j in range(n):
                # Create double directional perturbations
                x_ij_forward = np.array(x0, dtype=float)
                x_ij_backward = np.array(x0, dtype=float)
                x_i_forward_j_backward = np.array(x0, dtype=float)
                x_i_backward_j_forward = np.array(x0, dtype=float)
                
                if i == j:
                    # Diagonal elements (∂²f / ∂x_i²)
                    x_ij_forward[i] += epsilon
                    x_ij_backward[i] -= epsilon
                    hessian[i, j] = (
                        problem(x_ij_forward) - 2 * f_x0 + problem(x_ij_backward)
                    ) / (epsilon ** 2)
                else:
                    # Off-diagonal elements (∂²f / ∂x_i∂x_j)
                    x_ij_forward[i] += epsilon
                    x_ij_forward[j] += epsilon
                    
                    x_i_forward_j_backward[i] += epsilon
                    x_i_forward_j_backward[j] -= epsilon
                    
                    x_i_backward_j_forward[i] -= epsilon
                    x_i_backward_j_forward[j] += epsilon
                    
                    x_ij_backward[i] -= epsilon
                    x_ij_backward[j] -= epsilon
                    
                    hessian[i, j] = (
                        problem(x_ij_forward)
                        - problem(x_i_forward_j_backward)
                        - problem(x_i_backward_j_forward)
                        + problem(x_ij_backward)
                    ) / (4 * epsilon ** 2)
        
        return f_x0, grad, hessian

    # ---------------- Build Hamiltonian ----------------
    def build_ham(self, A):
        """
        Construct the Hamiltonian representation of the problem.

        Args:
            A: QUBO matrix (numpy array).

        Returns:
            ham: Hamiltonian object (mindquantum.core.operators.Hamiltonian).
            raw_ham: Raw Hamiltonian (mindquantum.core.operators.QubitOperator).
        """
        ham = QubitOperator()
        for i in range(self.n_trans):
            for j in range(self.n_trans):
                if i < j:
                    ham += A[i, j] / 2 * QubitOperator(f'Z{i} Z{j}')
        for i in range(self.n_trans):
            for j in range(self.n_trans):
                ham += A[i, j] / 2 * QubitOperator(f'Z{j}')
        for i in range(self.n_trans):
            for j in range(self.n_trans):
                if i < j:
                    ham += A[i, j] / 2
        return Hamiltonian(ham), ham

    # ---------------- Build QAOA Circuit ----------------
    def build_qaoa_circuit(self, ham, p):
        """
        Build a QAOA (Quantum Approximate Optimization Algorithm) circuit.

        Args:
            ham: Raw Hamiltonian (mindquantum.core.operators.QubitOperator).
            p: Number of QAOA layers.

        Returns:
            circ: QAOA circuit (mindquantum.core.circuit.Circuit).
        """
        circ = Circuit()
        circ += UN(H, list(range(self.n_trans)))
        for layer in range(p):
            for (term, coeff) in ham.terms.items():
                if len(term) == 2:  # Two-qubit terms
                    qubits = [q[0] for q in term]
                    circ += Rzz(2 * coeff * f'alpha{layer}').on(qubits)
                elif len(term) == 1:  # Single-qubit terms
                    qubit = term[0][0]
                    circ += RZ(2 * coeff * f'alpha{layer}').on(qubit)
            circ += UN(RX(f'beta{layer}'), list(range(self.n_trans)))
        return circ

    # ---------------- Single-Step QuCOOP ----------------
    def step_qucoop(self, x0, method="SA", layers=4, sampler=None):
        """
        Perform a single step of the QuCOOP optimization method.

        Args:
            x0: Initial solution.
            method: Optimization method ("SA" for Simulated Annealing, "QAOA" for Quantum Optimization).
            layers: Number of QAOA layers (used only if method="QAOA").
            sampler: Sampler object for simulated annealing (used only if method="SA").

        Returns:
            x_new: Updated solution (numpy array).
        """
        if method == "SA":
            _, grad, hessian = self.taylor_expansion_second_order(self.problem, x0)
            Q = hessian/2
            c = grad- np.dot(hessian, x0)
            W = {(i, j): Q[i, j] for i in range(self.n_trans) for j in range(self.n_trans)}
            c_dict = {i: c[i] for i in range(self.n_trans)}
            bqm = dimod.BinaryQuadraticModel(c_dict, W, vartype=dimod.BINARY)
            if sampler is None:
                sampler = neal.SimulatedAnnealingSampler()
            response = sampler.sample(bqm, num_reads=200)
            best = response.first.sample
            return np.array(list(best.values()))

        elif method == "QAOA":
            _, grad, hessian = self.taylor_expansion_second_order(self.problem, x0)
            Dij = hessian/2
            Dii = grad- np.dot(hessian, x0)
            W = Dij + np.diag(Dii)
            ham, h_raw = self.build_ham(W)
            circ = self.build_qaoa_circuit(h_raw, layers)
            sim = Simulator('mqvector', self.n_trans)
            grad_ops = sim.get_expectation_with_grad(ham, circ)

            def objective(params):
                energy, grad_val = grad_ops(params)
                return np.real(energy)[0, 0], np.real(grad_val)[0, 0]

            params_init = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
            result = minimize(objective, params_init, jac=True, method="BFGS",
                              options={"gtol": 1e-12, "maxiter": 200, "disp": False})
            pr = ParameterResolver(dict(zip(circ.params_name, result.x)))
            bound_circuit = circ.apply_value(pr)
            bound_circuit.measure_all()
            samples = sim.sampling(bound_circuit, shots=200)
            freqs = samples.data
            most_probable = max(freqs, key=freqs.get)
            x_new = np.array([1 - int(bit) for bit in most_probable[::-1]])
            return x_new

    # ---------------- Hybrid Optimization ----------------
    def solve(self, total_steps=50, q_ratio=0.5, hc_iter=3, method="SA", schedule="alternate"):
        """
        Solve the optimization problem using a hybrid approach of QuCOOP and Hill Climbing.

        Args:
            total_steps: Total number of iterations.
            q_ratio: Proportion of steps allocated to QuCOOP (0 to 1).
            hc_iter: Number of iterations for hill climbing per step.
            method: Optimization method for QuCOOP ("SA" or "QAOA").
            schedule: Schedule type ("alternate" for alternating methods, "phased" for sequential).

        Returns:
            history: Objective function values at each iteration.
            x0: Best solution found.
            problem(x0): Objective function value of the best solution.
        """
        x0 = np.zeros(self.n_trans, dtype=int)
        history = [self.problem(x0)]
        sampler = neal.SimulatedAnnealingSampler()

        q_steps = int(total_steps * q_ratio)
        hc_steps = total_steps - q_steps

        if schedule == "alternate":
            # Alternate between QuCOOP and hill climbing
            steps = min(q_steps, hc_steps)
            for _ in range(steps):
                x0 = self.step_qucoop(x0, method=method, sampler=sampler)
                history.append(self.problem(x0))
                hc_hist, x0, _ = hill_climbing(self.problem, x0, max_iter=hc_iter)
                history.extend(hc_hist)

        elif schedule == "phased":
            # Run QuCOOP first, then hill climbing
            for _ in range(q_steps):
                x0 = self.step_qucoop(x0, method=method, sampler=sampler)
                history.append(self.problem(x0))
            for _ in range(hc_steps):
                hc_hist, x0, _ = hill_climbing(self.problem, x0, max_iter=hc_iter)
                history.extend(hc_hist)

        self.P_sol = x0
        return history, x0, self.problem(x0)