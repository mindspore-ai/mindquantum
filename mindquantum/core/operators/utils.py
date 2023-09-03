#   Portions Copyright (c) 2020 Huawei Technologies Co.,ltd.
#   Portions Copyright 2017 The OpenFermion Developers.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   You may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""This module provide some useful function related to operators."""

from ...simulator.available_simulator import SUPPORTED_SIMULATOR
from ..operators.fermion_operator import FermionOperator
from ..operators.polynomial_tensor import PolynomialTensor
from ..operators.qubit_excitation_operator import QubitExcitationOperator
from ..operators.qubit_operator import QubitOperator


def count_qubits(operator):
    """
    Calculate the number of qubits on which operator acts before removing the unused qubit.

    Note:
        In some case, we need to remove the unused index.

    Args:
        operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]): FermionOperator or QubitOperator or
            QubitExcitationOperator.

    Returns:
        int, The minimum number of qubits on which operator acts.

    Raises:
       TypeError: Operator of invalid type.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator,FermionOperator, count_qubits
        >>> qubit_op = QubitOperator("X1 Y2")
        >>> count_qubits(qubit_op)
        3
        >>> fer_op = FermionOperator("1^")
        >>> count_qubits(fer_op)
        2
    """
    # Handle FermionOperator.
    valuable_type = (
        FermionOperator,
        QubitOperator,
        QubitExcitationOperator,
    )
    if hasattr(operator, 'count_qubits'):
        return operator.count_qubits()

    if isinstance(operator, valuable_type):
        num_qubits = 0
        for term in operator.terms:
            # a tuple compose of single (qubit_index,operator) subterms
            if term == ():
                qubit_index = (0,)
            else:
                qubit_index, _ = zip(*term)
            num_qubits = max(max(qubit_index) + 1, num_qubits)  # index start with 0
        return num_qubits

    if isinstance(operator, PolynomialTensor):
        return operator.n_qubits

    raise TypeError(f"Unsupported type of operator {operator}")


def commutator(left_operator, right_operator):
    """
    Compute the commutator of two operators.

    Args:
        left_operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            FermionOperator or QubitOperator.
        right_operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            FermionOperator or QubitOperator.

    Raises:
        TypeError: `left_operator` and `right_operator` are not of the same type.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator, FermionOperator, commutator
        >>> qub_op1 = QubitOperator("X1 Y2")
        >>> qub_op2 = QubitOperator("X1 Z2")
        >>> commutator(qub_op1, qub_op1)
        0
        >>> commutator(qub_op1, qub_op2)
        (2j) [X2]
    """
    if not isinstance(left_operator, type(right_operator)):
        raise TypeError('operator_a and operator_b are not of the same type.')
    valuable_type = (QubitOperator, FermionOperator, QubitExcitationOperator)
    if not isinstance(left_operator, valuable_type):
        raise TypeError("Operator should be QubitOperator, FermionOperator or QubitExcitationOperator.")

    result = left_operator * right_operator
    result -= right_operator * left_operator
    return result


def normal_ordered(fermion_operator):
    r"""
    Calculate and return the normal order of the FermionOperator.

    By convention, normal ordering implies terms are ordered from highest mode index (on left) to lowest (on right).
    Also, creation operators come first then follows the annihilation operator.
    e.g 3 4^ :math:`\rightarrow` - 4^ 3.

    Args:
        fermion_operator(FermionOperator): Only Fermion type Operator has such forms.

    Returns:
        FermionOperator, the FermionOperator with normal order.

    Examples:
        >>> from mindquantum.core.operators import FermionOperator, normal_ordered
        >>> op = FermionOperator("3 4^", 'a')
        >>> normal_ordered(op)
        -a [4^ 3]
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise ValueError("The operator should be FermionOperator!")
    return fermion_operator.normal_ordered()


def number_operator(n_modes=None, mode=None, coefficient=1.0):
    """
    Return a fermionic number operator for the reverse_jordan_wigner transform.

    Args:
        n_modes (int): The number of modes in the system. Default: ``None``.
        mode (int, optional): The mode on which to return the number
            operator. If ``None``, return total number operator on all sites. Default: ``None``.
        coefficient (float): The coefficient of the term. Default: ``1.0``.

    Returns:
        FermionOperator, a fermionic number operator for the reverse_jordan_wigner transform.

    Examples:
        >>> from mindquantum.core.operators import FermionOperator, number_operator
        >>> n_mode = 3
        >>> number_operator(n_mode)
        1 [0^ 0] +
        1 [1^ 1] +
        1 [2^ 2]
        >>> mode = 3
        >>> number_operator(None, mode)
        1 [3^ 3]
    """
    if (mode is None and n_modes is None) or (mode is not None and n_modes is not None):
        raise ValueError("Please provide only one parameter between n_modes and mode.")

    operator = FermionOperator()
    if mode is None:
        for mode_idx in range(n_modes):
            operator += FermionOperator(((mode_idx, 1), (mode_idx, 0)), coefficient)
        return operator
    return FermionOperator(((mode, 1), (mode, 0)), coefficient)


def get_fermion_operator(operator):
    """
    Convert the tensor (PolynomialTensor) to FermionOperator.

    Args:
        operator (PolynomialTensor): The `PolynomialTensor` you want to convert to `FermionOperator`.

    Returns:
        FermionOperator, An instance of the FermionOperator class.
    """
    if isinstance(operator, PolynomialTensor):
        return FermionOperator(operator)

    raise TypeError(f"Unsupported type of oeprator {operator}")


def hermitian_conjugated(operator):
    """
    Return Hermitian conjugate of FermionOperator or QubitOperator.

    Args:
        operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            The input operator.

    Returns:
        operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]),
        the hermitian form of the input operator.

    Examples:
        >>> from mindquantum.core.operators import QubitOperator, hermitian_conjugated
        >>> q = QubitOperator('X0', {'a' : 2j})
        >>> hermitian_conjugated(q)
        (-2j)*a [X0]
    """
    # Handle FermionOperator and QubitOperator
    if isinstance(operator, (FermionOperator, QubitOperator, QubitExcitationOperator)):
        return operator.hermitian()

    # Unsupported type
    raise TypeError(f'Taking the hermitian conjugate of a {type(operator).__name__} is not supported.')


def up_index(index):
    """
    Up index getter function.

    The index order, by default we set the spinless orbits as
    even-odd-even-odd (0,1,2,3,...). The spin_up orbitals (alpha orbitals) with index even.

    Args:
        index (int): spatial orbital index.

    Returns:
        int, the index of the associated spin-up orbital.

    Examples:
        >>> from mindquantum.core.operators import up_index
        >>> up_index(1)
        2
    """
    return 2 * index


def down_index(index):
    """
    Down index getter function.

    The index order, by default we set the spinless orbits as even-odd-even-odd (0,1,2,3,...). The spin_down orbitals
    (beta orbital) with index odd.

    Args:
        index (int): spatial orbital index.

    Returns:
        int, the index of the associated spin-down orbital.

    Examples:
        >>> from mindquantum.core.operators import down_index
        >>> down_index(1)
        3
    """
    return 2 * index + 1


def sz_operator(n_spatial_orbitals):
    """
    Return the sz operator.

    Note:
        The default index order spin_up (alpha) corresponds to even index,
        while the spin_down (beta) corresponds to odd.

    Args:
        n_spatial_orbitals (int): number of spatial orbitals (n_qubits // 2).

    Returns:
        FermionOperator, corresponding to the sz operator over
        `n_spatial_orbitals`.

    Examples:
        >>> from mindquantum.core.operators import sz_operator
        >>> sz_operator(3)
        1/2 [0^ 0] +
        -1/2 [1^ 1] +
        1/2 [2^ 2] +
        -1/2 [3^ 3] +
        1/2 [4^ 4] +
        -1/2 [5^ 5]
    """
    if not isinstance(n_spatial_orbitals, int) or n_spatial_orbitals < 0:
        raise TypeError("n_orbitals must be specified as an integer")

    sz_up = FermionOperator()
    sz_down = FermionOperator()
    for orbit in range(n_spatial_orbitals):
        sz_up += number_operator(None, up_index(orbit), 0.5)
        sz_down += number_operator(None, down_index(orbit), 0.5)

    return sz_up - sz_down


def ground_state_of_sum_zz(ops: QubitOperator, sim='mqvector') -> float:
    """
    Find the ground state energy of qubit operator that only has pauli :math:`Z` term.

    Args:
        ops (QubitOperator): qubit operator that only has pauli :math:`Z` term.
        sim (str): use which simulator to do calculation. Currently, we support
            ``'mqvector'`` and ``'mqvector_gpu'``. Default: ``'mqvector'``.

    Returns:
        float, the ground state energy of given qubit operator.

    Examples:
        >>> from mindquantum.core.operators import ground_state_of_sum_zz, QubitOperator
        >>> h = QubitOperator('Z0 Z1', 1.0) + QubitOperator('Z0 Z2', -1.5)
        >>> ground_state_of_sum_zz(h)
        -2.5
        >>> import numpy as np
        >>> np.min(np.diag(h.matrix().toarray()))
        (-2.5+0j)
    """
    # pylint: disable=import-outside-toplevel
    if sim.startswith('mqmatrix'):
        raise ValueError("mqmatrix simulator not support this method yet.")
    c_module = SUPPORTED_SIMULATOR.c_module(sim)
    ground_state_of_zs = getattr(c_module, "ground_state_of_zs")
    masks_value = {}
    terms = ops.terms
    for k, v in terms.items():
        mask = 0
        for idx, z in k:
            if str(z) != 'Z':
                raise ValueError("ops should contains only pauli z operator.")
            mask += 1 << idx
        masks_value[mask] = v.const.real
    return ground_state_of_zs(masks_value, ops.count_qubits())
