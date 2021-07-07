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
"""This module provide some useful function related to operators"""

import projectq.ops as pjops
import openfermion.ops as ofops
from mindquantum.ops.fermion_operator import FermionOperator
from mindquantum.ops.qubit_operator import QubitOperator
from mindquantum.ops.qubit_excitation_operator import QubitExcitationOperator
from mindquantum.ops.polynomial_tensor import PolynomialTensor


def count_qubits(operator):
    """
    Calculate the number of qubits on which operator acts before
    removing the unused qubit

    Note:
        In some case, we need to remove the unused index.

    Args:
        operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            FermionOperator or QubitOperator or QubitExcitationOperator.

    Returns:
        int, The minimum number of qubits on which operator acts.

    Raises:
       TypeError: Operator of invalid type.

    Examples:
        >>> from mindquantum.ops import QubitOperator,FermionOperator
        >>> from mindquantum.utils import count_qubits
        >>> qubit_op = QubitOperator("X1 Y2")
        >>> count_qubits(qubit_op)
        3
        >>> fer_op = FermionOperator("1^")
        >>> count_qubits(fer_op)
        2
    """
    # Handle FermionOperator.
    valueable_type = (FermionOperator, QubitOperator, QubitExcitationOperator,
                      ofops.FermionOperator, ofops.QubitOperator,
                      pjops.QubitOperator)
    if isinstance(operator, valueable_type):
        num_qubits = 0
        for term in operator.terms:
            # a tuple compose of single (qubit_index,operator) subterms
            if term == ():
                qubit_index = (0,)
            else:
                qubit_index, _ = zip(*term)
            num_qubits = max(max(qubit_index) + 1,
                             num_qubits)  # index start with 0
        return num_qubits

    if isinstance(operator, PolynomialTensor):
        return operator.n_qubits

    raise TypeError("Unsupported type of operator {}".format(operator))


def commutator(left_operator, right_operator):
    """
    Compute the commutator of two operators.

    Args:
        left_operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            FermionOperator or QubitOperator.
        right_operator (Union[FermionOperator, QubitOperator, QubitExcitationOperator]):
            FermionOperator or QubitOperator.

    Raises:
        TypeError: operator_a and operator_b are not of the same type.

    Examples:
        >>> from mindquantum.ops import QubitOperator,FermionOperator
        >>> from mindquantum.utils import commutator
        >>> qub_op1 = QubitOperator("X1 Y2")
        >>> qub_op2 = QubitOperator("X1 Z2")
        >>> commutator(qub_op1, qub_op1)
        0
        >>> commutator(qub_op1, qub_op2)
        2j [X2]
    """
    if not isinstance(left_operator, type(right_operator)):
        raise TypeError('operator_a and operator_b are not of the same type.')
    valueable_type = (QubitOperator, FermionOperator, QubitExcitationOperator)
    if not isinstance(left_operator, valueable_type):
        raise TypeError(
            "Operator should be QubitOperator, FermionOperator or QubitExcitationOperator."
        )

    result = left_operator * right_operator
    result -= right_operator * left_operator
    return result


def _normal_ordered_term(term, coefficient):
    r"""
    return the normal order order of a fermion operator with
    larger index and creation operator in front.
    eg. :math:`a_4\dagger a3_\dagger a_2 a_1`.
    """
    term = list(term)
    ordered_term = FermionOperator()
    for i in range(1, len(term)):
        for j in range(i, 0, -1):
            left_sub_term = term[j - 1]
            right_sub_term = term[j]
            # Swap operators if left operator is annihilation op and right operator is
            # a\dagger operator
            if not left_sub_term[1] and right_sub_term[1]:
                term[j], term[j - 1] = left_sub_term, right_sub_term
                coefficient = coefficient * -1
                # If indice are same, employ the anti-commutation relationship
                # And generate the new term
                if left_sub_term[0] == right_sub_term[0]:
                    new_term = term[:(j - 1)] + term[(j + 1):]
                    ordered_term += _normal_ordered_term(
                        new_term, -coefficient)
            # Deal with the case with same operator
            elif left_sub_term[1] == right_sub_term[1]:
                # If indice are same,evaluate it to zero.
                if left_sub_term[0] == right_sub_term[0]:
                    return ordered_term
                # Swap them if same operator but lower index on left
                if left_sub_term[0] < right_sub_term[0]:
                    term[j], term[j - 1] = left_sub_term, right_sub_term
                    coefficient = coefficient * -1

    # Add the term and return.
    ordered_term += FermionOperator(tuple(term), coefficient)
    return ordered_term


def normal_ordered(fermion_operator):
    r"""
    Calculate and return the normal order of the FermionOperator.
    By convention, normal ordering implies terms are ordered
    from highest mode index (on left) to lowest (on right).
    Also, creation operators come first then follows the annihilation operator.
    e.g 3 4^ :math:`\rightarrow` - 4^ 3.

    Args:
        fermion_operator(FermionOperator): Only Fermion type Operator has
            such forms.

    Returns:
        FermionOperator, the normal_ordered FermionOperator.

    Examples:
        >>> from mindquantum.ops import FermionOperator
        >>> from mindquantum.utils import normal_ordered
        >>> op = FermionOperator("3 4^", 'a')
        >>> normal_ordered(op)
        -a [4^ 3]
    """
    if not isinstance(fermion_operator, FermionOperator):
        raise ValueError("The operator should be FermionOperator!")
    ordered_op = FermionOperator()
    for term, coeff in fermion_operator.terms.items():
        ordered_op += _normal_ordered_term(term, coeff)
    return ordered_op


def get_fermion_operator(operator):
    """Convert the tensor (PolynomialTensor) to FermionOperator.

    Returns:
        fermion_operator, An instance of the FermionOperator class.
    """
    fermion_operator = FermionOperator()

    if isinstance(operator, PolynomialTensor):
        for term in operator:
            fermion_operator += FermionOperator(term, operator[term])
        return fermion_operator

    raise TypeError("Unsupported type of oeprator {}".format(operator))


def number_operator(n_modes=None, mode=None, coefficient=1.):
    """
    Return a fermionic number operator for the reverse_jordan_wigner transform.

    Args:
        n_modes (int): The number of modes in the system. Default: None.
        mode (int, optional): The mode on which to return the number
            operator. If None, return total number operator on all sites. Default: None.
        coefficient (float): The coefficient of the term. Default: 1.

    Returns:
        FermionOperator, a fermionic number operator for the reverse_jordan_wigner transform.

    Examples:
        >>> from mindquantum.ops import FermionOperator
        >>> from mindquantum.utils import number_operator
        >>> nmode = 3
        >>> number_operator(nmode)
        1.0 [0^ 0] +
        1.0 [1^ 1] +
        1.0 [2^ 2]
    """
    if (mode is None and n_modes is None) or (mode and n_modes):
        raise ValueError(
            "Please provide the correct value of n_modes and mode.")

    operator = FermionOperator()
    if mode is None:
        for m in range(n_modes):
            operator += FermionOperator(((m, 1), (m, 0)), coefficient)
    else:
        operator = FermionOperator(((mode, 1), (mode, 0)), coefficient)
    return operator


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
        >>> from mindquantum.ops import QubitOperator
        >>> from mindquantum.utils import hermitian_conjugated
        >>> q = QubitOperator('X0', {'a' : 2j})
        >>> hermitian_conjugated(q)
        -2.0*I*a [X0]
    """
    # Handle FermionOperator
    if isinstance(operator, FermionOperator):
        conjugate_operator = FermionOperator()
        for term, coefficient in operator.terms.items():
            # reverse the order and change the action from 0(1) to 1(0)
            conjugate_term = tuple([(index, 1 - op)
                                    for (index, op) in reversed(term)])
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()

    # Handle QubitOperator
    elif isinstance(operator, QubitOperator):
        conjugate_operator = QubitOperator()
        for term, coefficient in operator.terms.items():
            conjugate_operator.terms[term] = coefficient.conjugate()

    # Handle QubitExcitationOperator
    elif isinstance(operator, QubitExcitationOperator):
        conjugate_operator = QubitExcitationOperator()
        for term, coefficient in operator.terms.items():
            # reverse the order and change the action from 0(1) to 1(0)
            conjugate_term = tuple([(index, 1 - op)
                                    for (index, op) in reversed(term)])
            conjugate_operator.terms[conjugate_term] = coefficient.conjugate()

    # Unsupported type
    else:
        raise TypeError('Taking the hermitian conjugate of a {} is not '
                        'supported.'.format(type(operator).__name__))

    return conjugate_operator


def up_index(index):
    """
    The index order, by default we set the spinless orbits as
    even-odd-even-odd (0,1,2,3,...). The spin_up orbitals (alpha orbitals) with index even.

    Args:
        index (int): spatial orbital index.

    Returns:
        An integer that is the index of the associated spin-up orbital.

    Examples:
        >>> from mindquantum.utils import up_index
        >>> up_index(1)
        2
    """
    return 2 * index


def down_index(index):
    """
    The index order, by default we set the spinless orbits as
    even-odd-even-odd (0,1,2,3,...). The spin_down orbitals (beta orbital) with index odd.

    Args:
        index (int): spatial orbital index.

    Returns:
        An integer that is the index of the associated spin-down orbital.

    Examples:
        >>> from mindquantum.utils import down_index
        >>> down_index(1)
        3
    """
    return 2 * index + 1


def sz_operator(n_spatial_orbitals):
    """
    Return the sz operator.

    Args:
        n_spatial_orbitals (int): number of spatial orbitals (n_qubits // 2).

    Returns:
        FermionOperator, corresponding to the sz operator over
        n_spatial_orbitals.

    Note:
        The default index order spin_up(alpha) corresponds to even index,
        while the spin_down(beta) corresponds to odd index.rpartition()

    Examples:
        >>> from mindquantum.utils import sz_operator
        >>> sz_operator(3)
        0.5 [0^ 0] +
        -0.5 [1^ 1] +
        0.5 [2^ 2] +
        -0.5 [3^ 3] +
        0.5 [4^ 4] +
        -0.5 [5^ 5]
    """
    if not isinstance(n_spatial_orbitals, int) or n_spatial_orbitals < 0:
        raise TypeError("n_orbitals must be specified as an integer")

    sz_up = FermionOperator()
    sz_down = FermionOperator()
    for orbit in range(n_spatial_orbitals):
        sz_up += number_operator(None, up_index(orbit), 0.5)
        sz_down += number_operator(None, down_index(orbit), 0.5)

    return sz_up - sz_down
