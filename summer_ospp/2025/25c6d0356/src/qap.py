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

import utils


np.set_printoptions(linewidth=120)


class QAP:
    """
    Solver for the Quadratic Assignment Problem (QAP) using:
    - Quantum Approximate Optimization Algorithm (QAOA)
    - Simulated Annealing (SA)
    """

    def __init__(self, Q: np.ndarray, alpha: float, P_star: np.ndarray):
        self.Q = Q
        self.alpha = alpha
        self.P_star = P_star
        self.n = int(np.sqrt(Q.shape[0]))
        self.T = utils.t(self.n)
        self.n_trans = self.T.shape[1]
        self.P_sol = np.eye(self.n)

    # ----------------------------------------------------------------------
    # Utility functions
    # ----------------------------------------------------------------------
    def objective(self, P: np.ndarray) -> float:
        """Compute the QAP objective value."""
        P = P.reshape(-1)[:, None]
        return float((P.T @ self.Q @ P).squeeze())

    def build_qaoa_circuit(self, ham: QubitOperator, p: int) -> Circuit:
        """Build a QAOA circuit for a given Hamiltonian."""
        n_qubits = self.n_trans
        circ = Circuit()

        # Initial Hadamard state
        circ += UN(H, list(range(n_qubits)))

        # Alternating operators
        for layer in range(p):
            # Cost Hamiltonian evolution
            for (term, coeff) in ham.terms.items():
                if len(term) == 2:
                    qubits = [q[0] for q in term]
                    circ += Rzz(2 * coeff * ParameterResolver({f"alpha{layer}": 1})).on(qubits)
                elif len(term) == 1:
                    qubit = term[0][0]
                    circ += RZ(2 * coeff * ParameterResolver({f"alpha{layer}": 1})).on(qubit)

            # Mixer Hamiltonian
            circ += UN(RX(f"beta{layer}"), list(range(n_qubits)))

        return circ

    def build_hamiltonian(self, A: np.ndarray):
        """Convert matrix A into an Ising Hamiltonian."""
        ham = QubitOperator()
        n = self.n_trans

        # Quadratic terms
        for i in range(n):
            for j in range(i + 1, n):
                ham += A[i, j] / 2 * QubitOperator(f"Z{i} Z{j}")

        # Linear terms
        for i in range(n):
            for j in range(n):
                ham += A[i, j] / 2 * QubitOperator(f"Z{j}")

        # Constant offsets
        for i in range(n):
            for j in range(i + 1, n):
                ham += A[i, j] / 2

        return Hamiltonian(ham), ham

    # ----------------------------------------------------------------------
    # Solver methods
    # ----------------------------------------------------------------------
    def _solve_qaoa(self, Q_eff: np.ndarray, maxiter: int, layers: int,
                    x0: np.ndarray, history: list):
        """Internal loop for QAOA solver."""
        sim = Simulator("mqvector", self.n_trans)

        for j in range(maxiter):
            # Expand around current solution
            Px0, dPx0 = utils.eval_p(self.T, x0)
            Pc = Px0 - dPx0 @ x0
            Dij = dPx0.T @ Q_eff @ dPx0
            Dii = (2 * Pc.T @ Q_eff @ dPx0).flatten()
            W_matrix = Dij + np.diag(Dii)

            # Hamiltonian and circuit
            ham, h_prob = self.build_hamiltonian(W_matrix)
            circ = self.build_qaoa_circuit(h_prob, layers)
            grad_ops = sim.get_expectation_with_grad(ham, circ)

            def objective_qaoa(params):
                energy, grad = grad_ops(params)
                return np.real(energy)[0, 0], np.real(grad)[0, 0]

            # Optimize circuit parameters
            params_init = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
            result = minimize(
                objective_qaoa,
                params_init,
                jac=True,
                method="BFGS",
                options={"gtol": 1e-12, "maxiter": 20000, "disp": False},
            )

            best_params = result.x

            # Build optimized circuit
            sim.reset()
            pr = ParameterResolver(dict(zip(circ.params_name, best_params)))
            bound_circuit = circ.apply_value(pr)
            bound_circuit.measure_all()

            # Sample measurement outcomes
            samples = sim.sampling(bound_circuit, shots=1000).data
            probabilities = {bs: count / 1000 for bs, count in samples.items()}
            most_probable = max(probabilities, key=probabilities.get)

            # Update solution
            x0 = np.array([1 - int(b) for b in most_probable[::-1]]).reshape(-1, 1)
            P_sol = (Pc + dPx0 @ x0).reshape(self.n, self.n)
            sol = self.objective(P_sol)
            history.append(sol)

            logging.info(
                f"\n[QAOA Iter {j}] objective = {sol}\nPermutation matrix:\n{P_sol}\n"
            )

            if history[-1] == history[-2]:
                break

        self.P_sol = P_sol
        return history

    def _solve_sa(self, Q_eff: np.ndarray, maxiter: int, x0: np.ndarray, history: list):
        """Internal loop for Simulated Annealing solver."""
        sampler = neal.SimulatedAnnealingSampler()

        for j in range(maxiter):
            Px0, dPx0 = utils.eval_p(self.T, x0)
            Pc = Px0 - dPx0 @ x0
            Dij = dPx0.T @ Q_eff @ dPx0
            Dii = (2 * Pc.T @ Q_eff @ dPx0).flatten()

            W = {(i, j): Dij[i, j] for i in range(self.n_trans) for j in range(self.n_trans)}
            c = {i: Dii[i] for i in range(self.n_trans)}
            bqm = dimod.BinaryQuadraticModel(c, W, vartype=dimod.BINARY)

            response = sampler.sample(bqm, num_reads=500)
            best = response.first.sample
            q = np.array(list(best.values())).reshape(self.n_trans, 1)

            # Update solution
            x0 = q
            P_sol = (Pc + dPx0 @ x0).reshape(self.n, self.n)
            sol = self.objective(P_sol)
            history.append(sol)

            logging.info(
                f"\n[SA Iter {j}] objective = {sol}\nPermutation matrix:\n{P_sol}\n"
            )

            if len(history) > 1 and history[-1] == history[-2]:
                break

        self.P_sol = P_sol
        return history

    # ----------------------------------------------------------------------
    # Public interface
    # ----------------------------------------------------------------------
    def solve(self, maxiter: int = 15, layers: int = 10, method: str = "sa"):
        """
        Solve QAP with the chosen method.

        Parameters
        ----------
        maxiter : int
            Maximum iterations.
        layers : int
            QAOA layers (only used if method="qaoa").
        method : str
            "qaoa" for Quantum Approximate Optimization,
            "sa" for Simulated Annealing.

        Returns
        -------
        list[float]
            Objective value history.
        """
        x0 = np.zeros((self.n_trans, 1))

        # Effective Q matrix with penalty
        Q_eff = self.Q - self.alpha * np.eye(self.n * self.n)

        # Initial evaluation
        P_sol = utils.eval_p(self.T, x0)[0].reshape(self.n, self.n)
        sol = self.objective(P_sol)
        history = [sol]

        # Degenerate case
        if np.allclose(Q_eff, np.zeros_like(Q_eff)):
            return P_sol, history

        if method.lower() == "qaoa":
            logging.info("Using QAOA solver...")
            return self._solve_qaoa(Q_eff, maxiter, layers, x0, history)
        elif method.lower() == "sa":
            logging.info("Using Simulated Annealing solver...")
            return self._solve_sa(Q_eff, maxiter, x0, history)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose 'qaoa' or 'sa'.")

