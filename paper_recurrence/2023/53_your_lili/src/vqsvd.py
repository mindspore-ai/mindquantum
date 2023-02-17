"""
VQSVDTrainer is used to train/update the weight(ansatz parameters) with
'gradient-descent' or 'adam' method.

requirement:
mindspore==2.0.0a0
mindquantum==0.8.0
"""

import tqdm
import numpy as np
import scipy.sparse as sparse
import mindspore as ms
import mindspore.nn as nn
from mindquantum import Hamiltonian
from mindquantum import Simulator


class VQSVDTrainer:
    """Train/Update the weight (ansatz parameters) and reconstructed matrix
    with trained weight."""

    def __init__(self, n_qubit, mat_item, rank, ansatz_u, ansatz_v, q, basis,
                 lr=0.5, method='adam'):
        """
        Args:
            n_qubit: number of qubit.
            mat_item: the decomposed matrix.
            rank: rank T.
            ansatz_u: ansatz U(alpha).
            ansatz_v: ansatz V(beta).
            q: the weighted list value for different basis.
            basis: the chosen orthonormal basis.
            lr: learning rate of optimizer.
            method: gradient descent {'gd', 'adam'}
        """
        self.n_qubit = n_qubit
        self.mat_item = mat_item
        self.rank = rank
        self.ansatz_u = ansatz_u
        self.ansatz_v = ansatz_v
        self.q = q
        self.basis = basis
        self.n_param = len(ansatz_u.params_name) + len(ansatz_v.params_name)
        self.weight = np.random.uniform(0.0, 2 * np.pi, size=self.n_param)
        self.sim = Simulator('mqvector', self.n_qubit)
        self.lr = lr
        self.method = method
        self.circuits_left = [b + ansatz_u for b in basis]
        self.circuits_right = [b + ansatz_v for b in basis]
        # used when `method` is 'adam'
        self.ms_param = ms.Parameter(ms.Tensor(self.weight, dtype=ms.float64),
                                     name="weight")
        self.optimizer = nn.Adam((self.ms_param,), self.lr)

    def train_one_loop(self):
        """Iterating over the basis and update the weight."""
        expect = 0.0
        for i, (cir_left, cir_right) in enumerate(\
                zip(self.circuits_left, self.circuits_right)):
            grad = np.zeros_like(self.weight, dtype=np.complex128)
            fval = 0.0

            for (coef, mat) in self.mat_item:
                ham = Hamiltonian(sparse.csr_matrix(mat))
                ops = self.sim.get_expectation_with_grad(ham, cir_right, cir_left)
                f, g = ops(self.weight)
                fval += coef * f[0][0].conj()
                grad += coef * g[0][0].conj()
            # Update weight
            weighted_grad = self.q[i] * grad.real
            self.update_weight(weighted_grad)
            expect += self.q[i] * fval.real
        return expect

    def update_weight(self, grad):
        """Update weight."""
        if self.method == 'adam':
            grad = -grad
            grad = ms.Tensor(grad, dtype=ms.float64)
            self.optimizer((grad,))
            self.weight = self.optimizer.parameters[0].asnumpy()
        else:
            self.weight += self.lr * grad

    def train(self, epoch=50, ftol=1e-4):
        """Train the VQSVD weight with `epoch` times."""
        self.expect_record = []
        for i in tqdm.tqdm(range(epoch)):
            fval = self.train_one_loop()
            self.expect_record.append(fval)
            if i > 1:
                tol = np.abs(self.expect_record[-2] - fval)
                if tol < ftol:
                    print(f"Reach the ftol = {ftol} with epoch = {i}, "
                          f"stop in advance.")
                    break
                if i == epoch - 1:
                    print(f"Reach maximum epoch = {epoch}, "
                          f"current tolerance = {tol}.")
