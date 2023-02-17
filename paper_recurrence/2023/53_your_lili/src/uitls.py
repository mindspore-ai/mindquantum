"""
The utils functions.

requirement:
mindspore==2.0.0a0
mindquantum==0.8.0
"""

import itertools as it
from typing import List
import numpy as np
from mindquantum.core import Circuit
from mindquantum.core.gates import RY, CNOT, X
from mindquantum.core.circuit import add_prefix

from src.vqsvd import VQSVDTrainer


def get_ansatz(n_qubit: int, depth: int, kind='a') -> Circuit:
    """Get ansatz circuit.

    Args:
        n_qubit: number of qubit used.
        depth: number of block repetation.
        kind: circuit type, optional value: {'a', 'b', 'c', 'd'}.

    Return:
        Two ansatz circuits with different parameter names.
    """
    ansatz = Circuit()

    if kind == 'a':
        for layer in range(depth+1):
            for i in range(n_qubit):
                ansatz += RY(f'{layer * n_qubit + i + 1}').on(i)
            for i in range(n_qubit - 1):
                ansatz += CNOT(i+1, i)
    elif kind == 'b':
        assert n_qubit == 3, "this circuit only support n_qubit = 3"
        for layer in range(depth+1):
            ansatz += Circuit([
                RY(f'{8 * layer + 0}').on(0),
                RY(f'{8 * layer + 1}').on(1),
                CNOT(1, 0),
                RY(f'{8 * layer + 2}').on(0),
                RY(f'{8 * layer + 3}').on(1),
                RY(f'{8 * layer + 4}').on(1),
                RY(f'{8 * layer + 5}').on(2),
                CNOT(2, 1),
                RY(f'{8 * layer + 6}').on(1),
                RY(f'{8 * layer + 7}').on(2)
            ])
    elif kind == 'c':
        assert n_qubit == 3, "this circuit only support n_qubit = 3"
        for layer in range(depth+1):
            ansatz += Circuit([
                RY(f'{3 * layer + 0}').on(0),
                RY(f'{3 * layer + 1}').on(1),
                RY(f'{3 * layer + 2}').on(2),
                CNOT(1, 0),
                CNOT(2, 1),
                CNOT(0, 2)
            ])
    elif kind == 'd':
        for layer in range(depth+1):
            ansatz += Circuit([
                RY(f'{6 * layer + 0}').on(0),
                RY(f'{6 * layer + 1}').on(1),
                RY(f'{6 * layer + 2}').on(2),
                CNOT(1, 0),
                CNOT(2, 1),
                RY(f'{6 * layer + 0}').on(0),
                RY(f'{6 * layer + 1}').on(1),
                RY(f'{6 * layer + 2}').on(2),
                CNOT(0, 2),
                CNOT(1, 2)
            ])
    else:
        assert kind in {'a', 'b', 'c', 'd'}, \
            "Parameter kind should be one of {'a', 'b', 'c', 'd'}"

    ansatz_u = add_prefix(ansatz, 'alpha')
    ansatz_v = add_prefix(ansatz, 'beta')
    return (ansatz_u, ansatz_v)


def get_basis(n_qubit: int, rank=5) -> List:
    """Get basis which are orthonormal each other and encoded with circuit.
    Here just return $|0...00\rangle$, $|0...01\rangle$ ... in turn until
    `rank` basis.

    Args:
        n_qubit: number of qubit used.
        rank: number of basis.

    Return:
        Basis encoded with circuit.
    """
    basis = []
    for k, ps in enumerate(it.product([0, 1], repeat=n_qubit)):
        if k >= rank:
            break
        cir = Circuit()
        for i, p in enumerate(ps):
            if p == 1:
                cir += X.on(i)
        basis.append(cir)
    return basis


def decompose_matrix(n_qubit: int, in_mat: np.ndarray) -> List[tuple]:
    """Decompose a arbitrary matrix with the linear combination of tensor
    product of pauli operators {X, Y, Z, I}.

    Args:
        n_qubit: number of qubit.
        in_mat: input matrix that will be decompose.

    Return:
        List((coef, mat)): `mat` is tensor product of pauli operator and `coef`
            is corresponding coefficient.
    """
    def hs_product(m1, m2):
        """Hilbert-Schmidt-Product of two matrices `m1`, `m2`"""
        return (np.dot(m1.conjugate().transpose(), m2)).trace()

    def krons(arrs):
        """Get kron product of matrix array."""
        res = arrs[0]
        for a in arrs[1:]:
            res = np.kron(res, a)
        return res

    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ey = np.array([[1, 0], [0, 1]], dtype=np.complex128)
    op_list = np.array([sx, sy, sz, ey])

    # All combination of pauli gates on circuit.
    items = list(it.product([0, 1, 2, 3], repeat=n_qubit))

    result = []
    for item in items:
        mat = krons(op_list[list(item)])
        coef = hs_product(mat, in_mat) / (2**n_qubit)
        result.append((coef, mat))
    return result


def matrix_distance(m1, m2):
    """The distance between `m1` and `m2`"""
    return np.linalg.norm(m1 - m2)


def get_svd_norm(in_mat, ranks):
    """Get distance between `in_mat` and classical SVD reconstruction matrix
    with different ranks.
    """
    u, sigma, v = np.linalg.svd(in_mat)
    ds = []
    for r in ranks:
        re_mat = u[:, :r].dot(np.diag(sigma[:r])).dot(v[:r, :])
        d = matrix_distance(in_mat, re_mat)
        ds.append(d)
    return ds


def reconstruct_by_svd(in_mat, rank):
    """Get the matrix constructed by classical SVD with specific rank."""
    u, sigma, v = np.linalg.svd(in_mat)
    re_mat = u[:, :rank].dot(np.diag(sigma[:rank])).dot(v[:rank, :])
    return re_mat


def reconstruct_by_vqsvd(n_qubit, in_mat, basis, ansatz_u, ansatz_v, weight):
    """Reconstructing the input matrix with quantum circuit."""
    re_mat = np.zeros((2**n_qubit, 2**n_qubit),
                      dtype=np.complex128)
    eigv = []
    n_param = len(weight)
    for base in basis:
        circ_left = base + ansatz_u
        circ_right = base + ansatz_v
        qs1 = circ_left.get_qs(pr=weight[:n_param//2])
        qs2 = circ_right.get_qs(pr=weight[n_param//2:])
        expect = qs1.dot(in_mat).dot(qs2)
        re_mat += expect * (qs1.reshape((-1, 1)) * qs2)
        eigv.append(expect)
    re_mat = re_mat.real
    return re_mat


def run(n_qubit, rank, ansatz_uv, in_mat, epoch=50, lr=1e-2, method='adam'):
    """Run QVSVD process with specific parameters.

    Args:
        n_qubit: number of qubit.
        rank: rank T.
        ansatz_uv: U and V ansatz circuits.
        in_mat: the input matrix.
        epoch: the epoch of training.
        lr: learning rate of optimizer.

    Return:
        re_mat: the reconstructed matrix.
    """
    mat_item = decompose_matrix(n_qubit, in_mat)
    ansatz_u, ansatz_v = ansatz_uv
    q = np.arange(rank, 0, -1)
    basis = get_basis(n_qubit, rank)

    vqsvd = VQSVDTrainer(n_qubit, mat_item, rank, ansatz_u, ansatz_v, q, basis,
                         lr=lr, method=method)
    vqsvd.train(epoch)
    re_mat = reconstruct_by_vqsvd(n_qubit, in_mat, basis, ansatz_u, ansatz_v,
                                  weight=vqsvd.weight)
    expect_record = vqsvd.expect_record
    return re_mat, expect_record


def run_light(n_qubit, rank, ansatz_uv, in_mat):
    """Run QVSVD process with specific parameters.

    Args:
        n_qubit: number of qubit.
        rank: rank T.
        ansatz_uv: U and V ansatz circuits.
        in_mat: the input matrix.
        epoch: the epoch of training.
        lr: learning rate of optimizer.

    Return:
        re_mat: the reconstructed matrix.
    """
    import scipy.optimize as optimize

    ansatz_u, ansatz_v = ansatz_uv
    q = np.arange(rank, 0, -1)
    basis = get_basis(n_qubit, rank)
    n_param = len(ansatz_u.params_name) + len(ansatz_v.params_name)
    weight = np.random.uniform(0.0, 2 * np.pi, size=n_param)
    ansatz_u, ansatz_v = ansatz_uv

    circuits_left = [b + ansatz_u for b in basis]
    circuits_right = [b + ansatz_v for b in basis]

    def loss_func(params):
        loss = 0.0
        for k, (circ_left, circ_right) in enumerate(\
                zip(circuits_left, circuits_right)):
            expect = circ_left.get_qs(pr=params[:n_param//2]).real \
                .dot(in_mat) \
                .dot(circ_right.get_qs(pr=params[n_param//2:]).real)
            loss -= q[k] * expect
        return loss

    print("Begin trainning ...")
    result = optimize.minimize(loss_func, x0=weight, method="L-BFGS-B",
                               options={"maxiter": 500,
                                        "ftol": 1e-5,
                                        "gtol": 1e-5})
    weight = result.x
    print("Finish training.")
    re_mat = reconstruct_by_vqsvd(n_qubit, in_mat, basis, ansatz_u, ansatz_v, weight)
    return re_mat
