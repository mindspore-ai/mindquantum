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
"""Utils functions."""

from functools import reduce
from math import atan2, sqrt
from typing import List

import numpy as np
from scipy import linalg

from mindquantum.core import gates
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import QuantumGate
from mindquantum.utils.f import is_power_of_two

M = np.array([[1, 0, 0, 1j], [0, 1j, 1, 0], [0, 1j, -1, 0], [1, 0, 0, -1j]]) / sqrt(2)

M_DAG = M.conj().T

A = np.array([[1, 1, -1, 1], [1, 1, 1, -1], [1, -1, -1, -1], [1, -1, 1, 1]])


def kron_factor_4x4_to_2x2s(mat: np.ndarray):
    """
    Split a 4x4 matrix U = kron(A, B) into A, B, and a global factor.

    Requires the matrix to be the kronecker product of two 2x2 unitaries.
    Requires the matrix to have a non-zero determinant.
    Giving an incorrect matrix will cause garbage output.

    Args:
        mat: The 4x4 unitary matrix to factor.

    Returns:
        A scalar factor and a pair of 2x2 unit-determinant matrices. The
        kronecker product of all three is equal to the given matrix.

    Raises:
        ValueError:
            The given matrix can't be tensor-factored into 2x2 pieces.
    """
    # Use the entry with the largest magnitude as a reference point.
    a, b = max(((i, j) for i in range(4) for j in range(4)), key=lambda t: abs(mat[t]))

    # Extract sub-factors touching the reference cell.
    f1 = np.zeros((2, 2), dtype=np.complex128)
    f2 = np.zeros((2, 2), dtype=np.complex128)
    for i in range(2):
        for j in range(2):
            f1[(a >> 1) ^ i, (b >> 1) ^ j] = mat[a ^ (i << 1), b ^ (j << 1)]
            f2[(a & 1) ^ i, (b & 1) ^ j] = mat[a ^ i, b ^ j]

    # Rescale factors to have unit determinants.
    f1 /= np.sqrt(np.linalg.det(f1)) or 1
    f2 /= np.sqrt(np.linalg.det(f2)) or 1

    # Determine global phase.
    denominator = f1[a >> 1, b >> 1] * f2[a & 1, b & 1]
    if denominator == 0:
        raise ZeroDivisionError("denominator cannot be zero.")
    g = mat[a, b] / denominator
    if np.real(g) < 0:
        f1 *= -1
        g = -g

    return g, f1, f2


def kron_decomp(mat: np.ndarray):
    """
    Kronecker product decomposition (KPD) algorithm for 4x4 4*4 matrix.

    Note:
        This function is not absolutely robust (without tolerance).

    References:
        'New Kronecker product decompositions and its applications.'
        https://www.researchinventy.com/papers/v1i11/F0111025030.pdf
    """
    m00, m01, m10, m11 = mat[:2, :2], mat[:2, 2:], mat[2:, :2], mat[2:, 2:]
    k = np.vstack([m00.ravel(), m01.ravel(), m10.ravel(), m11.ravel()])
    if np.linalg.matrix_rank(k) == 1:
        # it is in form of tensor product
        l_list = [not np.allclose(np.zeros(4), k[i]) for i in range(4)]
        idx = l_list.index(True)  # the first non-zero block
        b = k[idx]
        a = np.array([])
        for i in range(4):
            if l_list[i]:
                a_i = times_two_matrix(k[i], b)
            else:
                a_i = 0
            a = np.append(a, a_i)
        a = a.reshape(2, 2)
        b = b.reshape(2, 2)
        return a, b
    return None, None


def is_tensor_prod(mat: np.ndarray) -> bool:
    """Distinguish whether a 4x4 matrix is the tensor product of two 2x2 matrices."""
    _, _ = kron_decomp(mat)
    if _ is None:
        return False
    return True


def params_zyz(mat: np.ndarray):
    r"""
    ZYZ decomposition of a 2x2 unitary matrix.

    .. math::
        U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)

    Args:
        mat: 2x2 unitary matrix

    Returns:
        `\alpha`, `\theta`, `\phi`, `\lambda`, four phase angles.
    """
    mat = mat.astype(np.complex128)
    if mat.shape != (2, 2):
        raise ValueError('Input matrix should be a 2*2 matrix')
    coe = linalg.det(mat) ** (-0.5)
    alpha = -np.angle(coe)
    v = coe * mat
    v = v.round(10)
    theta = 2 * atan2(abs(v[1, 0]), abs(v[0, 0]))
    phi_lam_sum = 2 * np.angle(v[1, 1])
    phi_lam_diff = 2 * np.angle(v[1, 0])
    phi = (phi_lam_sum + phi_lam_diff) / 2
    lam = (phi_lam_sum - phi_lam_diff) / 2
    return alpha, (theta, phi, lam)


def params_u3(mat: np.ndarray, return_phase=False):
    r"""
    Obtain the U3 parameters of a 2x2 unitary matrix.

    .. math::
        U = exp(i p) U3(\theta, \phi, \lambda)

    Args:
        mat: 2x2 unitary matrix
        return_phase: whether return the global phase `p`.

    Returns:
        Global phase `p` and three parameters `\theta`, `\phi`, `\lambda` of a standard U3 gate.
    """
    alpha, (theta, phi, lam) = params_zyz(mat)
    phase = alpha - (phi + lam) / 2
    if return_phase:
        return phase, (theta, phi, lam)
    return theta, phi, lam


def params_abc(mat: np.ndarray):
    r"""
    ABC decomposition of 2*2 unitary operator.

    .. math::
        \begin{align}
            U &= e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)\\
              &= e^{i\alpha} [R_z(\phi)R_y(\frac{\theta}{2})] X
                [R_y(-\frac{\theta}{2})R_z(-\frac{\phi+\lambda}{2})] X
                [R_z(\frac{\lambda-\phi}{2})]\\
              &=e^{i\alpha} A X B X C
        \end{align}

    Args:
        mat: 2x2 unitary matrix

    Returns:
        alpha (float), a (2x2 unitary), b (2x2 unitary), c (2x2 unitary).

    """
    if mat.shape != (2, 2):
        raise ValueError('Input matrix should be a 2*2 matrix')
    alpha, (theta, phi, lam) = params_zyz(mat)
    a = gates.RZ(phi).matrix() @ gates.RY(theta / 2).matrix()
    b = gates.RY(-theta / 2).matrix() @ gates.RZ(-(phi + lam) / 2).matrix()
    c = gates.RZ((lam - phi) / 2).matrix()
    return alpha, (a, b, c)


def limit_angle(a: float) -> float:
    """Limit equivalent rotation angle into (-pi, pi]."""
    if a >= 0:
        r = a % (2 * np.pi)
        if r < 0 or r > np.pi:
            r -= 2 * np.pi
        return r
    r = (-a) % (2 * np.pi)
    if 0 <= r <= np.pi:
        return -r
    return 2 * np.pi - r


def glob_phase(mat: np.ndarray) -> float:
    r"""
    Extract the global phase `\alpha` from a d*d matrix.

    .. math::
        U = e^{i\alpha} S

    in which S is in SU(d).

    Args:
        mat: d*d unitary matrix

    Returns:
        Global phase rad, in range of (-pi, pi].
    """
    d = mat.shape[0]
    if d == 0:
        raise ZeroDivisionError("Dimension of mat can not be zero.")
    exp_alpha = linalg.det(mat) ** (1 / d)
    return np.angle(exp_alpha)


def remove_glob_phase(mat: np.ndarray) -> np.ndarray:
    r"""
    Remove the global phase of a 2x2 unitary matrix by means of ZYZ decomposition.

    That is, remove

    .. math::

        e^{i\alpha}

    from

    .. math::
        U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)

    and return

    .. math::
        R_z(\phi) R_y(\theta) R_z(\lambda)

    Args:
        mat: 2x2 unitary matrix

    Returns:
        2x2 matrix without global phase.
    """
    alpha = glob_phase(mat)
    return mat * np.exp(-1j * alpha)


def is_equiv_unitary(mat1: np.ndarray, mat2: np.ndarray) -> bool:
    """Distinguish whether two unitary operators are equivalent, regardless of the global phase."""
    if mat1.shape != mat2.shape:
        raise ValueError(f'Input matrices have different dimensions: {mat1.shape}, {mat2.shape}.')
    d = mat1.shape[0]
    if not np.allclose(mat1 @ mat1.conj().T, np.identity(d)):
        raise ValueError('mat1 is not unitary')
    if not np.allclose(mat2 @ mat2.conj().T, np.identity(d)):
        raise ValueError('mat2 is not unitary')
    mat1f = mat1.ravel()
    mat2f = mat2.ravel()
    idx_uf = np.flatnonzero(mat1f.round(4))  # cut to some precision
    idx_vf = np.flatnonzero(mat2f.round(4))
    try:
        if np.allclose(idx_uf, idx_vf, atol=1e-4):
            coe = mat1f[idx_uf] / mat2f[idx_vf]
            return np.allclose(coe / coe[0], np.ones(len(idx_uf)), atol=1e-4)
        return False
    except ValueError:
        return False


def tensor_1_slot(mat: np.ndarray, n: int, tq: int) -> np.ndarray:
    """
    Given a 2x2 matrix, compute the matrix expanded to the whole Hilbert space (totally n qubits).

    Args:
        mat: matrix with size [2,2].
        n: total number of qubit subspaces.
        tq: target qubit index.

    Returns:
        Matrix, expanded via tensor product, with size [2^n, 2^n].
    """
    if tq not in range(n):
        raise ValueError('qubit index is out of range')
    ops = [np.identity(2)] * n
    ops[tq] = mat
    return reduce(np.kron, ops)


def tensor_slots(mat: np.ndarray, n: int, indices: List[int]) -> np.ndarray:
    """
    Given a matrix, compute the matrix expanded to the whole Hilbert space (totally n qubits).

    Args:
        mat: matrix with size
        n: total number of qubit subspaces
        indices: target qubit indices

    Returns:
        Matrix, expanded via tensor product, with size [2^n, 2^n].
    """
    if not is_power_of_two(mat.shape[0]):
        raise ValueError(f"Dimension of input matrix need should be power of 2, but get {mat.shape[0]}")
    m = int(np.log2(mat.shape[0]))
    if len(indices) != m or max(indices) >= n:
        raise ValueError(f'input indices {indices} does not consist with dimension of input matrix')

    if m == 1:
        return tensor_1_slot(mat, n, indices[0])

    arr_list = [mat] + [np.identity(2)] * (n - m)
    res = reduce(np.kron, arr_list).reshape([2] * 2 * n)
    idx = np.repeat(-1, n)
    for i, k in enumerate(indices):
        idx[k] = i
    idx[idx < 0] = range(m, n)
    idx_latter = [i + n for i in idx]
    return res.transpose(idx.tolist() + idx_latter).reshape(2**n, 2**n)


def times_two_matrix(mat1: np.ndarray, mat2: np.ndarray):
    """Calculate the coefficient a, s.t., U = a V. If a does not exist, return None."""
    if mat1.shape != mat2.shape:
        raise ValueError(f'Input matrices have different dimensions: {mat1.shape}, {mat2.shape}.')
    idx1 = np.flatnonzero(mat1.round(6))  # cut to some precision
    idx2 = np.flatnonzero(mat2.round(6))
    if not np.allclose(idx1, idx2):
        raise ValueError('Input matrices is not equivalent with a scalar factor')
    return mat1.ravel()[idx1[0]] / mat2.ravel()[idx2[0]]


def simult_svd(mat1: np.ndarray, mat2: np.ndarray):
    r"""
    Simultaneous SVD of two matrices, based on Eckart-Young theorem.

    Given two real matrices A and B who satisfy the condition of simultaneous SVD, then

    .. math::
        A = U D_1 V^{\dagger}, B = U D_2 V^{\dagger}

    Args:
        mat1: real matrix
        mat2: real matrix

    Returns:
        Four real matrices: u, v, d1, d2. u an v are both in SO(2). d1 and d2 are diagonal.

    References:
        'An Introduction to Cartan's KAK Decomposition for QC Programmers'
        https://arxiv.org/abs/quant-ph/0507171
    """
    if mat1.shape != mat2.shape:
        raise ValueError(f'mat1 and mat2 have different dimensions: {mat1.shape}, {mat2.shape}.')
    d = mat1.shape[0]

    # real orthogonal matrices decomposition
    u_a, d_a, v_a_h = linalg.svd(mat1)
    u_a_h = u_a.conj().T
    v_a = v_a_h.conj().T

    if np.count_nonzero(d_a) != d:
        raise ValueError('Not implemented yet for the situation that mat1 is not full-rank')
    # g commutes with d
    g = u_a_h @ mat2 @ v_a
    # because g is hermitian, eigen-decomposition is its spectral decomposition
    _, p = linalg.eigh(g)  # p is unitary or orthogonal
    u = u_a @ p
    v = v_a @ p

    # ensure det(u_a) == det(v_a) == +1
    if linalg.det(u) < 0:
        u[:, 0] *= -1
    if linalg.det(v) < 0:
        v[:, 0] *= -1

    d1 = u.conj().T @ mat1 @ v
    d2 = u.conj().T @ mat2 @ v
    return (u, v), (d1, d2)


def is_so4(mat: np.ndarray) -> bool:
    """Distinguish if a matrix is in SO(4) (4-dimension Special Orthogonal group)."""
    if mat.shape != (4, 4):
        raise ValueError('Input matrix is not 4x4 matrix')
    return np.allclose(mat @ mat.conj().T, np.identity(4)) and np.allclose(linalg.det(mat), 1)


def so4_to_magic_su2s(mat: np.ndarray):
    """
    Decompose 1 SO(4) operator into 2 SU(2) operators with Magic matrix transformation: U = Mdag @ kron(A, B) @ M.

    Args:
        mat: an SO(4) matrix.

    Returns:
        two SU(2) matrices, or, raise error.
    """
    if not is_so4(mat):
        raise ValueError('Input matrix is not in SO(4)')
    # KPD is definitely feasible when the input matrix is in SO(4)
    return kron_decomp(M @ mat @ M_DAG)


def circuit_to_unitary(circuit: Circuit, msb: bool = False) -> np.ndarray:
    """
    Convert a quantum circuit to a unitary matrix.

    Args:
        circuit (Circuit): Quantum circuit instance
        msb (bool): if True, means the most significant bit (MSB) is on the left, i.e., little-endian representation

    Returns:
        Matrix, Equivalent unitary matrix representation.
    """
    ops = []
    n = circuit.n_qubits
    for g in circuit:
        if isinstance(g, (gates.Measure, gates.BarrierGate)):
            continue
        if not g.ctrl_qubits:
            mat = tensor_slots(g.matrix(), n, g.obj_qubits)
        else:
            mat = controlled_unitary_matrix(g.matrix(), len(g.ctrl_qubits))
            mat = tensor_slots(mat, n, g.ctrl_qubits + g.obj_qubits)
        ops.append(mat)

    unitary = reduce(np.dot, reversed(ops))
    if msb:
        return tensor_slots(unitary, n, list(range(n - 1, -1, -1)))
    return unitary


def controlled_unitary_matrix(mat: np.ndarray, num_ctrl: int = 1) -> np.ndarray:
    """Construct the controlled-unitary matrix based on input unitary matrix."""
    proj_0, proj_1 = np.diag([1, 0]), np.diag([0, 1])
    for _ in range(num_ctrl):
        ident = reduce(np.kron, [np.identity(2)] * int(np.log2(mat.shape[0])))
        mat = np.kron(proj_0, ident) + np.kron(proj_1, mat)
    return mat


def multiplexor_matrix(n: int, tq: int, *args) -> np.ndarray:
    """
    Construct a quantum multiplexor in form of matrix.

    Args:
        n: total qubit index range (0 ~ n-1)
        tq: target qubit index
        *args: matrix components of the multiplexor

    Returns:
        Matrix, in type of np.ndarray.
    """
    if not len(args) == 2 ** (n - 1):
        raise ValueError(f'Number of input matrix components is not equal to {n}')
    qubits = list(range(n - 1))
    qubits.insert(tq, n - 1)
    mat = linalg.block_diag(*args)
    mat = mat.reshape([2] * 2 * n)
    return mat.transpose(qubits + [q + n for q in qubits]).reshape(2**n, 2**n)


def optimize_circuit(circuit: Circuit) -> Circuit:
    """
    Optimize the quantum circuit, i.e., removing identity operators.

    Args:
        circuit (Circuit): original input circuit.

    Returns:
        Circuit, the optimized quantum circuit.
    """
    circuit_opt = Circuit()
    for g in circuit:
        if isinstance(g, QuantumGate) and is_equiv_unitary(g.matrix(), np.identity(2)):
            continue
        circuit_opt.append(g)
    return circuit_opt
