# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Qudit symmetric mapping module."""

from typing import List
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
from mindquantum.utils.f import is_power_of_two
from mindquantum.core import QubitOperator

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import X, RX, RY, RZ, U3, GlobalPhase, UnivMathGate

optional_basis = ["zyz", "u3"]


def _symmetric_state_index(dim: int, n_qudits: int) -> dict:
    """
    The index of the qudit state or matrix element corresponding to the qubit symmetric state or matrix during mapping.

    Args:
        dim (int): the dimension of qudit state or matrix.
        n_qudits (int): the number fo qudit state or matrix.

    Returns:
        dict, which keys are the index of the qudit state or matrix,
        values are the corresponding index of qubit symmetric state or matrix.

    Examples:
        >>> from mindquantum.algorithm.library.qudit_mapping import _symmetric_state_index
        >>> _symmetric_state_index(dim=3, n_qudits=1)
        {0: [0], 1: [1, 2], 2: [3]}
        >>> _symmetric_state_index(dim=4, n_qudits=1)
        {0: [0], 1: [1, 2, 4], 2: [3, 5, 6], 3: [7]}
        >>> _symmetric_state_index(dim=3, n_qudits=2)
        {0: [0], 1: [1, 2], 2: [3], 3: [4, 8], 4: [5, 6, 9, 10], 5: [7, 11], 6: [12], 7: [13, 14], 8: [15]}
    """
    if not isinstance(dim, int):
        raise ValueError(f"Wrong type of dimension {type(dim)} {dim}.")
    if not isinstance(n_qudits, int):
        raise ValueError(f"Wrong type of n_qudits {type(n_qudits)} {n_qudits}.")
    if n_qudits == 1:
        ind = {}
        for i in range(2**(dim - 1)):
            num = bin(i).count("1")
            if num in ind:
                ind[num].append(i)
            else:
                ind[num] = [i]
    else:
        ind, ind_ = {}, {}
        for i in range(2**(dim - 1)):
            num = bin(i).count("1")
            i_ = bin(i)[2::].zfill(dim - 1)
            if num in ind_:
                ind_[num].append(i_)
            else:
                ind_[num] = [i_]
        for i in range(dim**n_qudits):
            multi = [""]
            base = np.base_repr(i, dim).zfill(n_qudits)
            for j in range(n_qudits):
                multi = [x + y for x in multi for y in ind_[int(base[j])]]
            ind[i] = [int(x, 2) for x in multi]
    return ind


def _is_symmetric(qubit: np.ndarray, n_qubits: int = 1) -> bool:
    """
    Check whether the qubit state or matrix is symmetric.

    Args:
        qubit (np.ndarray): the qubit state or matrix that needs to be checked whether it is symmetric.
        n_qubits (int): the number of qubits in the qubit symmetric state or matrix. Default: ``1``.

    Returns:
        bool, whether the qubit state or matrix is symmetric.
    """
    if qubit.ndim == 2 and (qubit.shape[0] == 1 or qubit.shape[1] == 1):
        qubit = qubit.flatten()
    if qubit.ndim == 2 and qubit.shape[0] != qubit.shape[1]:
        raise ValueError(f"Wrong qubit matrix shape {qubit.shape}.")
    if qubit.ndim != 1 and qubit.ndim != 2:
        raise ValueError(f"Wrong qubit matrix shape {qubit.shape}.")
    is_sym = True
    n = qubit.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f"Wrong qubit matrix size {n} is not a power of 2.")
    nq = int(np.log2(n))
    dim = nq // n_qubits + 1
    if nq % n_qubits == 0 and nq != n_qubits:
        ind = _symmetric_state_index(dim, n_qubits)
    else:
        raise ValueError(f"Wrong qubit matrix shape {qubit.shape} or number of qubits {n_qubits}.")
    if qubit.ndim == 1:
        for i in range(dim**n_qubits):
            i_ = ind[i]
            if len(i_) != 1:
                a = qubit[i_]
                is_sym = is_sym & np.allclose(a, a[0])
    elif qubit.ndim == 2:
        for i in range(dim**n_qubits):
            i_ = ind[i]
            for j in range(dim**n_qubits):
                j_ = ind[j]
                if len(i_) != 1 or len(j_) != 1:
                    a = qubit[np.ix_(i_, j_)]
                    is_sym = is_sym & np.allclose(a, a[0][0])
    return is_sym


def qudit_symmetric_decoding(qubit: np.ndarray, n_qubits: int = 1) -> np.ndarray:
    r"""
    Qudit symmetric decoding, decodes a qubit symmetric state or matrix into a qudit state or matrix.

    .. math::
        \begin{align}
        \ket{00\cdots00}&\to\ket{0} \\[.5ex]
        \frac{\ket{0\cdots01}+\ket{0\cdots010}+\ket{10\cdots0}}{\sqrt{d-1}}&\to\ket{1} \\
        \frac{\ket{0\cdots011}+\ket{0\cdots0101}+\ket{110\cdots0}}{\sqrt{d-1}}&\to\ket{2} \\
        \vdots&\qquad\vdots \\[.5ex]
        \ket{11\cdots11}&\to\ket{d-1}
        \end{align}

    Args:
        qubit (np.ndarray): the qubit symmetric state or matrix that needs to be decoded,
        where the qubit state or matrix must preserve symmetry.
        n_qubits (int): the number of qubits in the qubit symmetric state or matrix. Default: ``1``.

    Returns:
        np.ndarray, the qudit state or matrix obtained after the qudit symmetric decoding.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.library.qudit_mapping import qudit_symmetric_decoding
        >>> qubit = np.array([1., 2., 2., 3.])
        >>> qubit /= np.linalg.norm(qubit)
        >>> print(qubit)
        [0.23570226 0.47140452 0.47140452 0.70710678]
        >>> print(qudit_symmetric_decoding(qubit))
        [0.23570226+0.j 0.66666667+0.j 0.70710678+0.j]
    """
    if qubit.ndim == 2 and (qubit.shape[0] == 1 or qubit.shape[1] == 1):
        qubit = qubit.flatten()
    if qubit.ndim == 2 and qubit.shape[0] != qubit.shape[1]:
        raise ValueError(f"Wrong qubit matrix shape {qubit.shape}.")
    if qubit.ndim != 1 and qubit.ndim != 2:
        raise ValueError(f"Wrong qubit matrix shape {qubit.shape}.")
    n = qubit.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f"Wrong qubit matrix size {n} is not a power of 2.")
    nq = int(np.log2(n))
    dim = nq // n_qubits + 1
    if nq % n_qubits == 0 and nq != n_qubits:
        ind = _symmetric_state_index(dim, n_qubits)
    else:
        raise ValueError(f"Wrong qubit matrix shape {qubit.shape} or number of qubits {n_qubits}.")
    if qubit.ndim == 1:
        qudit = np.zeros(dim**n_qubits, dtype=np.complex128)
        for i in range(dim**n_qubits):
            i_ = ind[i]
            qubit_i = qubit[i_]
            if np.allclose(qubit_i, qubit_i[0]):
                qudit[i] = qubit_i[0] * np.sqrt(len(i_))
            else:
                raise ValueError("Qubit matrix is not symmetric.")
    elif qubit.ndim == 2:
        qudit = np.zeros([dim**n_qubits, dim**n_qubits], dtype=np.complex128)
        for i in range(dim**n_qubits):
            i_ = ind[i]
            for j in range(dim**n_qubits):
                j_ = ind[j]
                qubit_ij = qubit[np.ix_(i_, j_)]
                if np.allclose(qubit_ij, qubit_ij[0][0]):
                    div = np.sqrt(len(i_)) * np.sqrt(len(j_))
                    qudit[i, j] = qubit_ij[0][0] * div
                else:
                    raise ValueError("Qubit matrix is not symmetric.")
    return qudit


def qudit_symmetric_encoding(qudit: np.ndarray, n_qudits: int = 1, is_csr: bool = False) -> np.ndarray:
    r"""
    Qudit symmetric encoding, encodes a qudit state or matrix into a qubit symmetric state or matrix.

    .. math::

        \begin{align}
        \ket{0}&\to\ket{00\cdots00} \\[.5ex]
        \ket{1}&\to\frac{\ket{0\cdots01}+\ket{0\cdots010}+\ket{10\cdots0}}{\sqrt{d-1}} \\
        \ket{2}&\to\frac{\ket{0\cdots011}+\ket{0\cdots0101}+\ket{110\cdots0}}{\sqrt{d-1}} \\
        \vdots&\qquad\vdots \\[.5ex]
        \ket{d-1}&\to\ket{11\cdots11}
        \end{align}

    Args:
        qudit (np.ndarray): the qudit state or matrix that needs to be encoded.
        n_qudits (int): the number of qudits in the qudit state or matrix. Default: ``1``.

    Returns:
        np.ndarray, the qubit symmetric state or matrix obtained after the qudit symmetric encoding.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.library.qudit_mapping import qudit_symmetric_encoding
        >>> qudit = np.array([1., 2., 3.])
        >>> qudit /= np.linalg.norm(qudit)
        >>> print(qudit)
        [0.26726124 0.53452248 0.80178373]
        >>> print(qudit_symmetric_encoding(qudit))
        [0.26726124+0.j 0.37796447+0.j 0.37796447+0.j 0.80178373+0.j]
    """
    if qudit.ndim == 2 and (qudit.shape[0] == 1 or qudit.shape[1] == 1):
        qudit = qudit.flatten()
    if qudit.ndim == 2 and qudit.shape[0] != qudit.shape[1]:
        raise ValueError(f"Wrong qudit matrix shape {qudit.shape}.")
    if qudit.ndim != 1 and qudit.ndim != 2:
        raise ValueError(f"Wrong qudit matrix shape {qudit.shape}.")
    dim = round(qudit.shape[0]**(1 / n_qudits), 12)
    if dim % 1 == 0:
        dim = int(dim)
        n = 2**((dim - 1) * n_qudits)
        ind = _symmetric_state_index(dim, n_qudits)
    else:
        raise ValueError(f"Wrong qudit matrix shape {qudit.shape} or number of qudits {n_qudits}.")
    if qudit.ndim == 1:
        qubit = csr_matrix((n, 1), dtype=np.complex128)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            data = np.ones(num_i) * qudit[i] / np.sqrt(num_i)
            i_ = (ind_i, np.zeros(num_i))
            qubit += csr_matrix((data, i_), shape=(n, 1))
        if not is_csr:
            qubit = qubit.toarray().flatten()
    elif qudit.ndim == 2:
        qubit = csr_matrix((n, n), dtype=np.complex128)
        for i in range(dim**n_qudits):
            ind_i = ind[i]
            num_i = len(ind_i)
            for j in range(dim**n_qudits):
                ind_j = ind[j]
                num_j = len(ind_j)
                i_ = np.repeat(ind_i, num_j)
                j_ = np.tile(ind_j, num_i)
                div = np.sqrt(num_i) * np.sqrt(num_j)
                data = np.ones(num_i * num_j) * qudit[i, j] / div
                qubit += csr_matrix((data, (i_, j_)), shape=(n, n))
        if not is_csr:
            qubit = qubit.toarray()
    return qubit


def _two_level_unitary_synthesis(basis: str, ind: List[int], pr_str: List[str], obj: List[int]) -> Circuit:
    """
    Synthesize a qutrit two-level unitary gate with qubit circuit.

    Args:
       basis (str): decomposition basis, can be one of ``"zyz"`` or ``"u3"``.
       ind (List[int]): the subspace index of the qutrit two-level unitary gate.
       pr_str (List[str]): the params name of the qutrit two-level unitary gate.
       obj (List[int]): object qubits.

    Returns:
        :class:`~.core.circuit.Circuit`, qubit circuit that can synthesize a qutrit two-level unitary gate.
    """
    if len(ind) != 2:
        raise ValueError(f"The qutrit unitary index length {len(ind)} should be 2.")
    if len(set(ind)) != len(ind):
        raise ValueError(f"The qutrit unitary index {ind} cannot be repeated")
    if min(ind) < 0 or max(ind) >= 3:
        raise ValueError(f"The qutrit unitary index {ind} should in 0 to 2.")
    if len(pr_str) != 3:
        raise ValueError(f"The qutrit unitary params length {len(pr_str)} should be 3.")
    circ = Circuit()
    if ind == [0, 1]:
        corr = Circuit() + X(obj[1], obj[0]) + RY(np.pi / 2).on(obj[0], obj[1]) + X(obj[1], obj[0]) + X(obj[1])
    elif ind == [0, 2]:
        corr = Circuit() + X(obj[0]) + X(obj[1], obj[0]) + X(obj[0])
    elif ind == [1, 2]:
        corr = Circuit() + X(obj[1], obj[0]) + RY(-np.pi / 2).on(obj[0], obj[1]) + X(obj[1], obj[0])
    circ += corr
    if basis == "zyz":
        circ += RZ(pr_str[0]).on(obj[0], obj[1])
        circ += RY(pr_str[1]).on(obj[0], obj[1])
        circ += RZ(pr_str[2]).on(obj[0], obj[1])
    elif basis == "u3":
        theta, phi, lam = pr_str
        circ += U3(theta, phi, lam).on(obj[0], obj[1])
    else:
        raise ValueError(f"{basis} is not a supported decomposition method of {optional_basis}.")
    circ += corr.hermitian()
    return circ


def _single_qutrit_unitary_synthesis(basis: str, name: str, obj: List[int]) -> Circuit:
    """
    Synthesize a single qutrit unitary gate with qubit circuit.

    Args:
        basis (str): decomposition basis, can be one of ``"zyz"`` or ``"u3"``.
        name (str): the name of the single qutrit unitary gate.
        obj (List[int]): object qubits.

    Returns:
        :class:`~.core.circuit.Circuit`, qubit circuit that can synthesize a single qutrit unitary gate.
    """
    circ = Circuit()
    index = [[0, 1], [0, 2], [1, 2]]
    if basis == "zyz":
        for i, ind in enumerate(index):
            pr_ind = f"{''.join(str(i) for i in ind)}_{i}"
            pr_str = [f"{name}RZ{pr_ind}", f"{name}RY{pr_ind}", f"{name}Rz{pr_ind}"]
            circ += _two_level_unitary_synthesis(basis, ind, pr_str, obj)
    elif basis == "u3":
        for i, ind in enumerate(index):
            pr_ind = f"{''.join(str(i) for i in ind)}_{i}"
            pr_str = [f"{name}𝜃{pr_ind}", f"{name}𝜑{pr_ind}", f"{name}𝜆{pr_ind}"]
            circ += _two_level_unitary_synthesis(basis, ind, pr_str, obj)
    else:
        raise ValueError(f"{basis} is not a supported decomposition method of {optional_basis}.")
    return circ


def _controlled_rotation_synthesis(ind: List[int], name: str, obj: int, ctrl: List[int], state: int) -> Circuit:
    """
    Synthesize a qutrit controlled rotation gate with qubit circuit.

    Args:
        ind (List[int]): the subspace index of the qutrit controlled rotation gate.
        name (str): the name of the qutrit controlled rotation gate.
        obj (int): object qubit.
        ctrl (List[int]): control qubits.
        state (int): the control state of the qutrit controlled rotation gate.

    Returns:
        :class:`~.core.circuit.Circuit`, qubit circuit that can synthesize a qutrit controlled rotation gate.
    """
    circ = Circuit()
    if state == 0:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(obj, ctrl[1:]) + X(ctrl[0], ctrl[1:] + [obj]) + X(
                obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[1]) + X(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(-np.pi / 2).on(
                obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj])
    elif state == 1:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(
                np.pi / 2).on(obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(obj, ctrl[1:]) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[1], ctrl[2]) + RY(np.pi / 2).on(ctrl[2]) + X(ctrl[0], ctrl[1:] + [obj]) + RY(
                -np.pi / 2).on(obj, ctrl) + X(ctrl[0], ctrl[1:] + [obj])
    elif state == 2:
        if ind == [0, 1]:
            corr = Circuit() + X(ctrl[0], ctrl[1:] + [obj]) + RY(np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj]) + X(ctrl[0], ctrl[1:])
        elif ind == [0, 2]:
            corr = Circuit() + X(obj, ctrl[1:]) + X(ctrl[0], ctrl[1:] + [obj]) + X(obj, ctrl[1:])
        elif ind == [1, 2]:
            corr = Circuit() + X(ctrl[0], ctrl[1:] + [obj]) + RY(-np.pi / 2).on(obj, ctrl) + X(
                ctrl[0], ctrl[1:] + [obj])
    circ += corr
    if "RX" in name:
        circ = circ + RX(name).on(obj, ctrl)
    elif "RY" in name:
        circ = circ + RY(name).on(obj, ctrl)
    elif "RZ" in name:
        circ = circ + RZ(name).on(obj, ctrl)
    elif "GP" in name:
        circ = circ + GlobalPhase(name).on(obj, ctrl)
    circ += corr.hermitian()
    return circ


def _controlled_diagonal_synthesis(name: str, obj: int, ctrl: List[int], state: int) -> Circuit:
    """
    Synthesize a qutrit controlled diagonal gate with qubit circuit.

    Args:
        name (str): the name of the qutrit controlled diagonal gate.
        obj (int): object qubit.
        ctrl (List[int]): control qubits.
        state (int): the control state of the qutrit controlled diagonal gate.

    Returns:
        :class:`~.core.circuit.Circuit`, qubit circuit that can synthesize a qutrit controlled diagonal gate.
    """
    circ = Circuit()
    circ += _controlled_rotation_synthesis([0, 1], f"{name}RZ01", obj, ctrl, state)
    circ += _controlled_rotation_synthesis([0, 2], f"{name}RZ02", obj, ctrl, state)
    circ += _controlled_rotation_synthesis([0, 1], f"{name}GP", obj, ctrl, state)
    circ += _controlled_rotation_synthesis([0, 2], f"{name}GP", obj, ctrl, state)
    circ += _controlled_rotation_synthesis([1, 2], f"{name}GP", obj, ctrl, state)
    return circ


def qutrit_symmetric_ansatz(gate: UnivMathGate, basis: str = "zyz", with_phase: bool = False) -> Circuit:
    r"""
    Construct a qubit ansatz that preserves the symmetry of encoding for arbitrary qutrit gate.

    Reference: `Synthesis of multivalued quantum logic circuits by elementary gates
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.87.012325>`_,
    `Optimal synthesis of multivalued quantum circuits
    <https://journals.aps.org/pra/abstract/10.1103/PhysRevA.92.062317>`_.

    Args:
        gate (:class:`~.core.gates.UnivMathGate`): symmetry-preserving qubit gate encoded by qutrit gate.
        basis (str): decomposition basis, can be one of ``"zyz"`` or ``"u3"``. Default: ``"zyz"``.
        with_phase (bool): whether return global phase in form of a :class:`~.core.gates.GlobalPhase` gate
        on the qubit circuit. Default: ``False``.

    Returns:
        :class:`~.core.circuit.Circuit`, qubit ansatz that preserves the symmetry of qutrit encoding.

    Examples:
        >>> from scipy.stats import unitary_group
        >>> from mindquantum.core.circuit import Circuit
        >>> from mindquantum.core.gates import UnivMathGate
        >>> from qudit_mapping import qutrit_symmetric_ansatz, qudit_symmetric_encoding
        >>> qutrit_unitary = unitary_group.rvs(3)
        >>> qutrit_projector = np.eye(4) - qudit_symmetric_encoding(np.eye(3))
        >>> qubit_unitary = qudit_symmetric_encoding(qutrit_unitary) + qutrit_projector
        >>> qubit_gate = UnivMathGate('U', qubit_unitary).on([0, 1])
        >>> print(Circuit() + qubit_gate)
        q0: ──U──
              │
        q1: ──U──
        >>> print(qutrit_symmetric_ansatz(qubit_gate))
        q0: ──●────RY(π/2)────●─────────RZ(U_RZ01_0)────RY(U_RY01_0)────RZ(U_Rz01_0)─────────●────RY(-π/2)────●────X──>>
              │       │       │              │               │               │               │       │        │       >>
        q1: ──X───────●───────X────X─────────●───────────────●───────────────●──────────X────X───────●────────X───────>>
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        q0: <<──●────X────RZ(U_RZ02_1)────RY(U_RY02_1)────RZ(U_Rz02_1)────X────●────X────●────RY(-π/2)────●──>>
            <<  │              │               │               │               │         │       │        │  >>
        q1: <<──X──────────────●───────────────●───────────────●───────────────X─────────X───────●────────X──>>
        ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        q0: <<──RZ(U_RZ12_2)────RY(U_RY12_2)────RZ(U_Rz12_2)────●────RY(π/2)────●──
            <<       │               │               │          │       │       │
        q1: <<───────●───────────────●───────────────●──────────X───────●───────X──
    """
    if gate.ctrl_qubits:
        raise ValueError(f"Currently not applicable for a controlled gate {gate}.")
    basis = basis.lower()
    if basis not in optional_basis:
        raise ValueError(f"{basis} is not a supported decomposition method of {optional_basis}.")
    circ = Circuit()
    obj = gate.obj_qubits
    name = f"{gate.name}_"
    if not _is_symmetric(gate.matrix(), int(len(obj) / 2)):
        raise ValueError(f"{gate} is not a symmetric gate.")
    if len(obj) == 2:
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}", obj)
    elif len(obj) == 4:
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U1_", obj[:2])
        circ += _controlled_diagonal_synthesis(f"{name}CD1_", obj[0], obj[1:], 1)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U2_", obj[:2])
        circ += _controlled_diagonal_synthesis(f"{name}CD2_", obj[0], obj[1:], 2)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U3_", obj[:2])
        circ += _controlled_rotation_synthesis([1, 2], f"{name}RY1_2", obj[-1], obj[::-1][1:], 2)
        circ += _controlled_rotation_synthesis([1, 2], f"{name}RY1_1", obj[-1], obj[::-1][1:], 1)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U4_", obj[:2])
        circ += _controlled_diagonal_synthesis(f"{name}CD3_", obj[0], obj[1:], 2)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U5_", obj[:2])
        circ += _controlled_rotation_synthesis([0, 1], f"{name}RY2_2", obj[-1], obj[::-1][1:], 2)
        circ += _controlled_rotation_synthesis([0, 1], f"{name}RY2_1", obj[-1], obj[::-1][1:], 1)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U6_", obj[:2])
        circ += _controlled_diagonal_synthesis(f"{name}CD4_", obj[0], obj[1:], 0)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U7_", obj[:2])
        circ += _controlled_rotation_synthesis([1, 2], f"{name}RY3_2", obj[-1], obj[::-1][1:], 2)
        circ += _controlled_rotation_synthesis([1, 2], f"{name}RY3_1", obj[-1], obj[::-1][1:], 1)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U8_", obj[:2])
        circ += _controlled_diagonal_synthesis(f"{name}CD5_", obj[0], obj[1:], 2)
        circ += _single_qutrit_unitary_synthesis(basis, f"{name}U9_", obj[:2])
    else:
        raise ValueError(
            "Currently only applicable when the n_qutrits is 1 or 2, which means the n_qubits must be 2 or 4.")
    if with_phase:
        for i in obj:
            circ += GlobalPhase(f"{name}phase").on(i)
    return circ


def mat_to_op(mat, little_endian: bool = True) -> QubitOperator:
    """
    Convert a matrix to a QubitOperator. Default output is in little endian.

    Args:
        mat: the qubit matrix that needs to be converted to a QubitOperator.
        little_endian (bool): whether the qubit order is little endian. This means
        the leftmost qubit is the qubit with the highest index. Default: ``True``.

    Returns:
        :class:`~.core.QubitOperator`, the QubitOperator obtained after the matrix conversion.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.algorithm.library.qudit_mapping import mat_to_op
        >>> mat = np.array([[1, 0, 0, 1], [0, 1, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1]])
        >>> print(mat_to_op(mat, 2))
        1 [] +
        1 [X0 X1]
    """
    def pairs_to_op(i, j):
        bin_i = bin(i)[2:].zfill(n_qubits)
        bin_j = bin(j)[2:].zfill(n_qubits)
        if little_endian:
            bin_i = bin_i[::-1]
            bin_j = bin_j[::-1]
        term = QubitOperator('')
        for ind, (b1, b2) in enumerate(zip(bin_i, bin_j)):
            if b1 + b2 == '00':
                term *= QubitOperator(f'I{ind}', 1/2) + QubitOperator(f'Z{ind}', 1/2)
            elif b1 + b2 == '11':
                term *= QubitOperator(f'I{ind}', 1/2) + QubitOperator(f'Z{ind}', -1/2)
            elif b1 + b2 == '01':
                term *= QubitOperator(f'X{ind}', 1/2) + QubitOperator(f'Y{ind}', 1/2 * 1j)
            elif b1 + b2 == '10':
                term *= QubitOperator(f'X{ind}', 1/2) + QubitOperator(f'Y{ind}', -1/2 * 1j)
        return term

    if np.shape(mat)[0] != np.shape(mat)[1]:
        raise ValueError(f"Not a legal qubit matrix {mat}.")
    if np.ceil(np.log2(np.shape(mat)[0])) != np.floor(np.log2(np.shape(mat)[0])):
        raise ValueError(f"Not a legal qubit matrix {mat}.")
    n_qubits = int(np.ceil(np.log2(np.shape(mat)[0])))
    res = QubitOperator()

    if sp.sparse.issparse(mat):
        coo_mat = sp.sparse.coo_matrix(mat)
        for (i, j, value) in zip(coo_mat.row, coo_mat.col, coo_mat.data):
            if np.abs(value) == 0:
                continue
            term = pairs_to_op(i, j)
            res += term * value
    else:
        for i in range(2 ** n_qubits):
            for j in range(2 ** n_qubits):
                if np.abs(mat[i][j]) == 0:
                    continue
                term = pairs_to_op(i, j)
                res += term * mat[i][j]
    return res.compress()
