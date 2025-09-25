import numpy as np
import matplotlib.pyplot as plt
import dimod
import neal
import utils
import logging
from scipy.optimize import minimize

from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite
from mindquantum.core.circuit import Circuit, UN
from mindquantum.core.gates import RX, RZ, Rzz, H
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator
from mindquantum.core import ParameterResolver

np.set_printoptions(linewidth=100)


class PointSetRegistration:
    """
    Point Set Registration class.
    Supports Simulated Annealing (SA) and QAOA algorithms
    for solving rotation and permutation matching problems.
    """

    def __init__(self, ref_points, tmp_points, alpha, beta, R_star, P_star):
        """
        Initialize the registration problem.

        Args:
            ref_points (ndarray): Reference point set (d x n).
            tmp_points (ndarray): Transformed point set (d x n).
            alpha (float): Regularization parameter for rotation.
            beta (float): Regularization parameter for permutation.
            R_star (ndarray): Ground truth rotation matrix.
            P_star (ndarray): Ground truth permutation matrix.
        """
        self.ref_points = ref_points
        self.tmp_points = tmp_points
        self.alpha = alpha
        self.beta = beta
        self.R_star = R_star
        self.P_star = P_star
        self.d, self.n = ref_points.shape
        self.p = int(self.d * (self.d - 1) / 2)  # number of params
        self.T = utils.t(self.n)
        self.n_trans = self.T.shape[1]
        # Current best solution
        self.P_sol = np.eye(self.n)
        self.R_sol = np.eye(self.d)

    def objective(self, R, P):
        """Compute Frobenius-norm objective."""
        return np.linalg.norm(R @ self.tmp_points - self.ref_points @ P, ord="fro") ** 2

    def build_qaoa_circuit(self, ham, p, K):
        """
        Build QAOA circuit based on Hamiltonian.

        Args:
            ham (QubitOperator): Problem Hamiltonian.
            p (int): Number of QAOA layers.
            K (int): Precision parameter.

        Returns:
            Circuit: QAOA circuit.
        """
        n_qubits = self.n_trans + self.p * K
        circ = Circuit()

        # Initial state with Hadamards
        circ += UN(H, list(range(n_qubits)))

        # QAOA layers
        for layer in range(p):
            for (term, coeff) in ham.terms.items():
                # ZZ interactions
                if len(term) == 2:
                    qubits = [q[0] for q in term]
                    circ += Rzz(2 * coeff * f"alpha{layer}").on(qubits)
                # Z fields
                elif len(term) == 1:
                    qubit = term[0][0]
                    circ += RZ(2 * coeff * f"alpha{layer}").on(qubit)

            # Mixer Hamiltonian (X rotations)
            circ += UN(RX(f"beta{layer}"), list(range(n_qubits)))

        return circ

    def build_ham(self, A, K):
        """
        Build Ising Hamiltonian from weight matrix A.

        Args:
            A (ndarray): Coefficient matrix.
            K (int): Precision.

        Returns:
            Hamiltonian, QubitOperator: Hamiltonian objects.
        """
        ham = QubitOperator()
        n_vars = self.n_trans + self.p * K

        # Quadratic terms
        for i in range(n_vars):
            for j in range(n_vars):
                if i < j:
                    ham += A[i, j] / 2 * QubitOperator(f"Z{i} Z{j}")

        # Linear terms
        for i in range(n_vars):
            for j in range(n_vars):
                ham += A[i, j] / 2 * QubitOperator(f"Z{j}")

        # Constant terms
        for i in range(n_vars):
            for j in range(n_vars):
                if i < j:
                    ham += A[i, j] / 2

        return Hamiltonian(ham), ham

    def solve(self, K, maxiter=15, method="sa", layers=20):
        """
        Main entry for solving the registration problem.

        Args:
            K (int): Precision parameter.
            maxiter (int): Max number of iterations.
            method (str): "sa" for simulated annealing, "qaoa" for QAOA.
            layers (int): Number of QAOA layers.

        Returns:
            list: Objective value history.
        """
        eval_r = utils.eval_r_2d if self.d == 2 else utils.eval_r_3d
        scale = 2**K - 1
        delta = np.pi * np.ones((self.p, 1))
        tau = 4 * delta / scale

        x0_rot = np.zeros((self.p, 1))
        x0_perm = np.zeros((self.n_trans, 1))

        # Initial objective
        P_sol = np.eye(self.n)
        R_sol = np.eye(self.d)
        sol = self.objective(R_sol, P_sol)
        history = [sol]

        X = np.kron(self.ref_points, P_sol)
        Y = np.kron(R_sol, self.tmp_points)

        if method == "sa":
            return self._solve_with_annealing(
                X, Y, x0_rot, x0_perm, scale, delta, tau,
                K, maxiter, history, eval_r
            )
        if method == "qaoa":
            return self._solve_with_qaoa(
                X, Y, x0_rot, x0_perm, scale, delta, tau,
                K, maxiter, history, eval_r, layers
            )

    def _solve_with_annealing(
        self, X, Y, x0_rot, x0_perm, scale, delta, tau,
        K, maxiter, history, eval_r
    ):
        """Simulated Annealing solver."""
        sampler = neal.SimulatedAnnealingSampler()
        alpha_rot = self.alpha * np.linalg.norm(self.tmp_points, ord="fro") ** 2
        alpha_perm = self.beta * np.linalg.norm(self.ref_points, ord="fro") ** 2

        for j in range(maxiter):
            # Rotation update
            Rx0, dRx0 = eval_r(x0_rot)
            Rc = Rx0 - dRx0 @ delta

            # Permutation update
            Px0, dPx0 = utils.eval_p(self.T, x0_perm)
            Pc = Px0 - dPx0 @ x0_perm

            # Encoding
            U = (2 / scale) * np.kron(
                np.diag(delta.flatten()), np.array([2**k for k in range(K)])
            )

            # Gradients
            x_dPx0 = np.einsum("di, ij -> dj", X, dPx0)
            y_dRx0 = np.einsum("di, dj -> ij", Y, dRx0)

            # Cost terms
            Dij_rot = U.T @ np.einsum("di, dj -> ij", dRx0, dRx0) @ U * alpha_rot
            Dii_rot = 2 * np.einsum("di, d -> i", dRx0, Rc.flatten()) @ U * alpha_rot
            Dij_perm = np.einsum("dj, dk -> jk", dPx0, dPx0) * alpha_perm
            Dii_perm = 2 * np.einsum("dk, d -> k", dPx0, Pc.flatten()) * alpha_perm
            Dij_mixed = -U.T @ np.einsum("di, dj -> ij", y_dRx0, x_dPx0) / 2
            Dii_mixed = -np.block([
                np.einsum("di, d -> i", y_dRx0, X @ Pc.flatten()) @ U,
                np.einsum("dk, d -> k", x_dPx0, Rc.flatten() @ Y),
            ])

            Dij = np.block([[Dij_rot, Dij_mixed], [Dij_mixed.T, Dij_perm]])
            Dii = np.block([Dii_rot, Dii_perm]) + Dii_mixed

            # Build BQM
            W = {(i, j): Dij[i, j] for i in range(self.p * K + self.n_trans)
                 for j in range(self.p * K + self.n_trans)}
            c = {i: Dii[i] for i in range(self.p * K + self.n_trans)}
            bqm = dimod.BinaryQuadraticModel(c, W, vartype=dimod.BINARY)
            response = sampler.sample(bqm, num_reads=100)

            # Best sample
            best = response.first.sample
            q = np.array(list(best.values()))
            q_rot = q[:K * self.p].reshape(K * self.p, 1)
            q_perm = q[K * self.p:].reshape(self.n_trans, 1)

            # Update
            x_new_tmp = U @ q_rot
            R_sol = (Rc + dRx0 @ x_new_tmp).reshape(self.d, self.d)
            P_sol = (Pc + dPx0 @ q_perm).reshape(self.n, self.n)
            sol = self.objective(R_sol, P_sol)
            history.append(sol)

            # Decrease search window
            x0_rot = x0_rot - delta + x_new_tmp
            x0_perm = q_perm
            if (j + 1) % 2 == 0:
                delta = delta / 2
                tau = tau / 2

            logging.info(
                f"Iter {j}, delta = {np.round(delta.flatten(), 8)}, "
                f"error_R = {round(np.linalg.norm(R_sol - self.R_star), 8) if self.R_star is not None else '-'}, "
                f"error_P = {round((np.linalg.norm(P_sol - self.P_star) ** 2) / 2, 8) if self.P_star is not None else '-'}, "
                f"objective = {round(sol, 8)}"
            )

            self.R_sol, self.P_sol = R_sol, P_sol

        return history

    def _solve_with_qaoa(
        self, X, Y, x0_rot, x0_perm, scale, delta, tau,
        K, maxiter, history, eval_r, layers
    ):
        """QAOA solver."""
        # print("Using QAOA for solving the problem...")
        sim = Simulator("mqvector", self.p * K + self.n_trans)
        alpha_rot = self.alpha * np.linalg.norm(self.tmp_points, ord="fro") ** 2
        alpha_perm = self.beta * np.linalg.norm(self.ref_points, ord="fro") ** 2

        for j in range(maxiter):
            # Rotation update
            Rx0, dRx0 = eval_r(x0_rot)
            Rc = Rx0 - dRx0 @ delta

            # Permutation update
            Px0, dPx0 = utils.eval_p(self.T, x0_perm)
            Pc = Px0 - dPx0 @ x0_perm

            # Encoding
            U = (2 / scale) * np.kron(
                np.diag(delta.flatten()), np.array([2**k for k in range(K)])
            )

            # Gradients
            x_dPx0 = np.einsum("di, ij -> dj", X, dPx0)
            y_dRx0 = np.einsum("di, dj -> ij", Y, dRx0)

            # Cost terms
            Dij_rot = U.T @ np.einsum("di, dj -> ij", dRx0, dRx0) @ U * alpha_rot
            Dii_rot = 2 * np.einsum("di, d -> i", dRx0, Rc.flatten()) @ U * alpha_rot
            Dij_perm = np.einsum("dj, dk -> jk", dPx0, dPx0) * alpha_perm
            Dii_perm = 2 * np.einsum("dk, d -> k", dPx0, Pc.flatten()) * alpha_perm
            Dij_mixed = -U.T @ np.einsum("di, dj -> ij", y_dRx0, x_dPx0) / 2
            Dii_mixed = -np.block([
                np.einsum("di, d -> i", y_dRx0, X @ Pc.flatten()) @ U,
                np.einsum("dk, d -> k", x_dPx0, Rc.flatten() @ Y),
            ])

            Dij = np.block([[Dij_rot, Dij_mixed], [Dij_mixed.T, Dij_perm]])
            Dii = np.block([Dii_rot, Dii_perm]) + Dii_mixed
            W = Dij + np.diag(Dii)

            # Build Hamiltonian
            ham, h_prob = self.build_ham(W, K)
            circ = self.build_qaoa_circuit(h_prob, layers, K)
            grad_ops = sim.get_expectation_with_grad(ham, circ)

            # Define objective for optimizer
            def objective_QAOA(params):
                energy, grad = grad_ops(params)
                return np.real(energy)[0, 0], np.real(grad)[0, 0]

            # Optimize
            params_init = np.random.uniform(-np.pi, np.pi, len(circ.params_name))
            result = minimize(objective_QAOA, params_init, jac=True,
                              method="BFGS",
                              options={"gtol": 1e-12, "maxiter": 20000, "disp": True})

            # Extract best parameters
            best_params = result.x
            sim.reset()
            pr = ParameterResolver(dict(zip(circ.params_name, best_params)))
            bound_circuit = circ.apply_value(pr)
            bound_circuit.measure_all()
            samples = sim.sampling(bound_circuit, shots=1000)

            # Get most probable bitstring
            probabilities = {b: c / 1000 for b, c in samples.data.items()}
            most_probable_bitstring = max(probabilities, key=probabilities.get)
            q = np.array([1 - int(bit) for bit in most_probable_bitstring[::-1]])

            # Update
            q_rot = q[:K * self.p].reshape(K * self.p, 1)
            q_perm = q[K * self.p:].reshape(self.n_trans, 1)
            x_new_tmp = U @ q_rot

            R_sol = (Rc + dRx0 @ x_new_tmp).reshape(self.d, self.d)
            P_sol = (Pc + dPx0 @ q_perm).reshape(self.n, self.n)
            sol = self.objective(R_sol, P_sol)
            history.append(sol)

            x0_rot = x0_rot - delta + x_new_tmp
            x0_perm = q_perm
            if (j + 1) % 2 == 0:
                delta = delta / 2
                tau = tau / 2

            logging.info(
                f"Iter {j}, delta = {np.round(delta.flatten(), 8)}, "
                f"error_R = {round(np.linalg.norm(R_sol - self.R_star), 8) if self.R_star is not None else '-'}, "
                f"error_P = {round((np.linalg.norm(P_sol - self.P_star) ** 2) / 2, 8) if self.P_star is not None else '-'}, "
                f"objective = {round(sol, 8)}"
            )

            self.R_sol, self.P_sol = R_sol, P_sol

        return history