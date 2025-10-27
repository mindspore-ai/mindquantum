from itertools import permutations
import numpy as np
import math


class LQA_HUBO_Solver():

    def __init__(self, problem_sparse, N, n_iter=1000, gamma=0.1, dt=1.0, x=None):
        self.N = N
        self.n_iter = n_iter
        self.gamma = gamma
        self.dt = dt
        self.problem_sparse = problem_sparse

        self.x = x if x is not None else 0.2 * (np.random.rand(self.N, 1) - 0.5)

    def _calculate_hubo_grad_sparse(self, z):
        z = z.flatten()
        grad_contribution = np.zeros(self.N)

        for order, terms_list in self.problem_sparse.items():
            for indices, coeff in terms_list:
                for idx_k in range(order):
                    k = indices[idx_k]
                    product_others = 1.0
                    for idx_m in range(order):
                        if idx_k != idx_m:
                            m = indices[idx_m]
                            product_others *= z[m]

                    grad_contribution[k] += order * coeff * product_others

        return grad_contribution.reshape(-1, 1)

    def initialize(self):
        """Initialize spin values."""
        if self.x is None:
            self.x = 0.2 * (np.random.rand(self.N, self.batch_size) - 0.5)

        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

    def update(self, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        LQA annealing process for Hubo problem.
        """
        # Initialize the momentum and velocity variables for the ADAM optimizer
        m_dx = 0
        v_dx = 0

        for i in range(1, self.n_iter + 1):
            # Calculate the current annealing progress t and intermediate variables z, y
            t = i / self.n_iter
            tmp = np.pi / 2 * np.tanh(self.x)
            z = np.sin(tmp)
            y = np.cos(tmp)

            # calculate grad
            grad_contribution = self._calculate_hubo_grad_sparse(z)

            # calculate full dx
            a_w = 1 - np.tanh(self.x) ** 2
            # dx = np.pi / 2 * (-t * self.gamma * grad_contribution * y + (1 - t) * z) * a_w
            dx = np.pi / 2 * (t * self.gamma * grad_contribution * y + (1 - t) * z) * a_w
            # adam
            m_dx = beta1 * m_dx + (1 - beta1) * dx
            v_dx = beta2 * v_dx + (1 - beta2) * dx ** 2

            # bias amend
            m_dx_corr = m_dx / (1 - beta1 ** i)
            v_dx_corr = v_dx / (1 - beta2 ** i)

            self.x = self.x - self.dt * m_dx_corr / (np.sqrt(v_dx_corr) + epsilon)


    def get_solution(self):
        return np.sign(self.x).flatten()

    def calc_energy_sparse(self, s=None):
        """calculate energy on sparse question"""
        if s is None:
            s = self.get_solution()
        s = s.flatten()

        total_energy = 0.0
        for order, terms_list in self.problem_sparse.items():
            for indices, coeff in terms_list:
                term_energy = coeff * np.prod([s[i] for i in indices])
                total_energy += term_energy
        return total_energy

def run_grid_hubo_problem(M, N, coefficient=-1.0):
    """
    Construct a HUBO problem based on an M x N grid.

    Args:
        M (int): Number of rows in the grid.
        N (int): Number of columns in the grid.
        coefficient (float): Coupling coefficient for each plaquette term.

    Returns:
        tuple: (
            N_total (int): Total number of variables,
            problem_sparse (dict): Sparse dictionary defining the HUBO problem
        )
    """
    # --- Step 1: Compute total number of edges and define indexing ---
    num_horizontal_edges = N * (M + 1)
    num_vertical_edges = M * (N + 1)
    N_total = num_horizontal_edges + num_vertical_edges

    # Indexing scheme:
    # - Horizontal edges: 0 to num_horizontal_edges - 1
    # - Vertical edges: num_horizontal_edges to N_total - 1

    def get_horizontal_edge_index(row, col):
        # row in [0, M], col in [0, N-1]
        return row * N + col

    def get_vertical_edge_index(row, col):
        # row in [0, M-1], col in [0, N]
        return num_horizontal_edges + col * M + row

    problem_sparse = {
        4: []
    }

    # Each plaquette is identified by the coordinates of its lower-left corner
    for row in range(M):
        for col in range(N):
            bottom_edge = get_horizontal_edge_index(row, col)
            right_edge = get_vertical_edge_index(row, col + 1)
            top_edge = get_horizontal_edge_index(row + 1, col)
            left_edge = get_vertical_edge_index(row, col)
            # Add the 4-body interaction term to the HUBO problem
            indices = (bottom_edge, right_edge, top_edge, left_edge)
            term = (indices, coefficient)
            problem_sparse[4].append(term)

    return N_total, problem_sparse

def create_grid_hubo_problem(M, N, coefficient=-1.0):
    """
    Construct a HUBO problem on an M x N grid.

    Args:
        M (int): Number of grid rows (height).
        N (int): Number of grid columns (width).
        coefficient (float): Coupling coefficient for each plaquette term.

    Returns:
        tuple: (
            N_total (int): Total number of variables,
            problem_sparse (dict): Sparse problem definition dictionary required by the solver
        )
    """
    num_horizontal_edges = N * (M + 1)
    num_vertical_edges = M * (N + 1)
    N_total = num_horizontal_edges + num_vertical_edges

    def get_horizontal_edge_index(row, col):
        return row * N + col

    def get_vertical_edge_index(row, col):
        return num_horizontal_edges + col * M + row

    problem_sparse = {
        4: []
    }

    for row in range(M):
        for col in range(N):
            bottom_edge = get_horizontal_edge_index(row, col)
            right_edge = get_vertical_edge_index(row, col + 1)
            top_edge = get_horizontal_edge_index(row + 1, col)
            left_edge = get_vertical_edge_index(row, col)

            indices = (bottom_edge, right_edge, top_edge, left_edge)
            term = (indices, coefficient)
            problem_sparse[4].append(term)

    return N_total, problem_sparse


def create_lgt_problem_wegner(M, N, coefficient=-1.0):
    """
    Args:
        M (int): Number of rows in the grid.
        N (int): Number of columns in the grid.
        coefficient (float): First-order energy coefficient associated with each variable.

    Returns:
        tuple: (
            N_total (int): Total number of variables,
            problem_sparse (dict): Sparse dictionary defining the HUBO problem
        )
    """

    N_total = M * N

    problem_sparse = {
        1: []
    }

    for row in range(M):
        for col in range(N):
            plaquette_index = row * N + col
            indices = (plaquette_index,)
            term = (indices, coefficient)
            problem_sparse[1].append(term)

    return N_total, problem_sparse

def run_lgt_problem_wegner(M_rows, N_cols):

    total_vars, sparse_problem = create_lgt_problem_wegner(M=M_rows, N=N_cols, coefficient=-1.0)

    solver = LQA_HUBO_Solver(problem_sparse=sparse_problem, N=total_vars, n_iter=3000)
    solver.update()
    solution = solver.get_solution()
    energy = solver.calc_energy_sparse(solution)
    solution = [int(s) for s in solution]

    return solution, energy

def run_lgt_problem_classical(M_rows, N_cols):
    total_vars, sparse_problem = create_grid_hubo_problem(M=M_rows, N=N_cols, coefficient=-1.0)
    solver = LQA_HUBO_Solver(problem_sparse=sparse_problem, N=total_vars, n_iter=3000)
    solver.update()
    solution = solver.get_solution()
    energy = solver.calc_energy_sparse(solution)
    solution = [int(s) for s in solution]
    return solution, energy
