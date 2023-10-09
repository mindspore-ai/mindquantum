# coding:utf-8
"""
Using QAOA to get better solution than Babai's algorithm in Schnorr's algorithm.
"""

import time
from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import mindspore.nn as nn
from mindquantum import H, RX, RZ, ZZ, UN
from mindquantum import Circuit, Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator
from mindquantum import ParameterResolver


def dot(v1, v2) -> float:
    """Calculate the inner product of two vector.
    """
    return np.ravel(v1) @ np.ravel(v2)


def norm2(v):
    """Calculate the square of norm for input vector.
    """
    return dot(v, v)


@dataclass
class QAOAConfig:
    """Configuration for `QAOASearch`"""
    n_layer: int = 4                # Layers of QAOA circuit.
    # Optimizer for QAOA, only support Adam now.
    optimizer: str = "adam"
    learning_rate: float = 0.002    # Learning rate of optimizer.
    max_iter: int = 1000            # Number of iterations in QAOA.
    n_sample_shot: int = 1000       # Number of samples.
    verbose: int = 0                # Output log if `verbose` greater than 0.


class QAOASearch:
    """QAOA Search in SQIF algorithm.
    """

    def __init__(self, config: QAOAConfig):
        """
        Args:
            config: The QAOA configuration.
            verbose: If it's greater than 0, then output some log. Otherwise no output.
        """
        self.config = config
        self.verbose = config.verbose

    def _get_hamil_coef(self, mat_d, diff, s):
        """Get coefficients for hamiltonian.
        Args:
            mat_d (np.ndarray): shape=(m,n). LLL-deduced basis.
            t (np.ndarray): shape=(m,1). Target vector, which is a column vector.
            s (np.ndarray): shape=(n). s[i] means symbol of ith variable. 1 means cj <= muj.
                -1 means cj > muj. See formula (S48) in paper.
        """
        n = mat_d.shape[1]
        mat_f = np.zeros((n, n))      # Coefficient matrix.
        g = np.zeros(n)
        l = 0.0

        mat_d = mat_d.copy()
        diff = diff.copy()

        mat_d *= s
        diff += mat_d.sum(axis=1, keepdims=True) / 2.0
        mat_d /= (-2.0)

        for i in range(n):
            for j in range(i, n):
                di = mat_d[:, i:i+1]
                dj = mat_d[:, j:j+1]
                if i == j:
                    mat_f[i, j] = dot(di, dj)
                else:
                    mat_f[i, j] = 2 * dot(di, dj)
        for i in range(n):
            di = mat_d[:, i:i+1]
            g[i] = 2 * dot(di, diff)
        for i in range(n):
            l += mat_f[i, i]
            mat_f[i, i] = 0
        l += norm2(diff)
        return mat_f, g, l

    def _build_hamil_circuit(self, param, mat_f, g) -> Circuit:
        """Build hamiltonian circuit.
        Args:
            mat_f (np.ndarray): shape=(n, n)
            g (np.ndarray): shape=(n)
            param (string): Parameter name.
        Return:
            circ (Circuit): Hamiltonian circuit.
        """
        n_qubit = self.n_qubit
        circ = Circuit()
        for i in range(n_qubit):
            for j in range(i+1, n_qubit):
                circ += ZZ(ParameterResolver(param) * mat_f[i, j]).on((i, j))
        for i in range(n_qubit):
            circ += RZ(ParameterResolver(param) * g[i]).on(i)
        circ.barrier()
        return circ

    def _build_base_circuit(self, param):
        """Build circuit for base state.
        """
        circ = Circuit()
        for i in range(self.n_qubit):
            circ += RX(param).on(i)
        circ.barrier()
        return circ

    def _build_qaoa_circuit(self, mat_f, g):
        """Build QAOA circuit.
        """
        circ = Circuit()
        circ += UN(H, self.n_qubit)
        for i in range(self.config.n_layer):
            circ += self._build_hamil_circuit(f'g{i}', mat_f, g)
            circ += self._build_base_circuit(f'b{i}')
        return circ

    def _build_qaoa_hamil(self, mat_f, g):
        """Build QAOA hamiltonian.
        """
        n = mat_f.shape[0]
        assert mat_f.shape[1] == n, "Matrix `mat_f` shape should be (n x n)"
        assert len(g) == n, "`g` length error."

        ham = QubitOperator()
        for i in range(n):
            ham += float(g[i]) * QubitOperator(f'Z{i}')
        for i in range(n):
            for j in range(i+1, n):
                ham += float(mat_f[i, j]) * QubitOperator(f'Z{i} Z{j}')
        if self.verbose:
            print("Built hamiltonian is:\n", ham)
        return Hamiltonian(ham)

    def _train(self):
        """Build QAOA and train.
        """
        t1 = time.time()
        circ = deepcopy(self.circ)
        sim = Simulator('mqvector', self.n_qubit)
        grad_ops = sim.get_expectation_with_grad(self.ham, circ)
        net = MQAnsatzOnlyLayer(grad_ops)
        opti = nn.Adam(net.trainable_params(),
                       learning_rate=self.config.learning_rate)
        train_net = nn.TrainOneStepCell(net, opti)

        loss_record = []
        for i in range(self.config.max_iter):
            loss = train_net()
            loss = float(loss)
            loss_record.append(loss)
            if self.verbose and i % (self.config.max_iter//10) == 0:
                print(f"Iter [{i}]: loss = {loss:.3f}")
        t2 = time.time()
        if self.verbose:
            print(f"It spends {t2 - t1:.2f} seconds to train.")
        self.loss_record = loss_record

        circ.measure_all()
        pr = dict(zip(circ.params_name, net.weight.asnumpy()))
        out = sim.sampling(circ, pr=pr, shots=self.config.n_sample_shot)
        if self.verbose:
            print(out)
        out_data = out.bit_string_data
        # Get the state that has the minimum energy.
        bit_string_max = max(out_data, key=out_data.get)
        bit_float_max = [float(c) for c in bit_string_max][::-1]
        disturbance = np.array(bit_float_max) * np.array(self.symbol)
        return disturbance

    def run(self, mat_d, diff, s, bop):
        """The outer interface that get the disturbance.
        Args:
            mat_d: The LLL-reduced matrix.
            diff: The difference between target vector and the closest vector.
            s: The symbol which element is in {-1, 1}.
            bop: The closest vector from Schnorr's algorithm.
        Return:
            vnew: The new closest vector optimized based on `bop`.
        """
        self.n_qubit = mat_d.shape[1]
        mat_f, g, _ = self._get_hamil_coef(mat_d, diff, s)
        self.ham = self._build_qaoa_hamil(mat_f, g)
        self.circ = self._build_qaoa_circuit(mat_f, g)
        self.symbol = s
        disturbance = self._train()
        vnew = (mat_d * disturbance).sum(axis=1).reshape((-1, 1)) + bop
        return vnew

    def __call__(self, mat_d, diff, s, bop):
        return self.run(mat_d, diff, s, bop)


if __name__ == "__main__":
    mat_d = np.array(
        [[1, -4, -3],
         [-2,  1,  2],
         [2,  2,  0],
         [3, -2,  4]], np.float64)

    config = QAOAConfig(n_layer=5, max_iter=8001, verbose=1)
    qaoa = QAOASearch(config)

    diff = np.array([0, 4, 4, 2], np.float64).reshape((-1, 1))
    s = np.array([-1, -1, -1], np.float64)
    t = np.array([0, 0, 0, 240], np.float64).reshape((-1, 1))
    bop = np.array([0, 4, 4, 242], np.float64).reshape((-1, 1))
    vnew = qaoa.run(mat_d, diff, s, bop)
    print(f"target vector `t`:\n{t.ravel().tolist()}")
    print(f"schnorr's algorithm closest vector `bop`:\n{bop.ravel().tolist()}")
    print(f"new vector `vnew`:\n{vnew.ravel().tolist()}")
    print(f"norm(bop - t) = {np.linalg.norm(bop - t):.2f}")
    print(f"norm(vnew - t) = {np.linalg.norm(vnew - t):.2f}")
