#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""Compatibility module (temporary) for Python terms operators."""

from mindquantum.core.operators.utils import number_operator

from ....core.operators import FermionOperator, QubitOperator


def _transform_ladder_operator(ladder_operator, x1, y1, z1, x2, y2, z2):  # pylint: disable=too-many-arguments
    r"""
    Make transformation to qubits for :math:`a`, :math:`a^\dagger` operators.

    .. math::
        a = 1/2 X(x1) Y(y1) Z (z1)+ i/2 X(x2) Y(y2) Z(z2)
        a^\dagger = 1/2 X(x1) Y(y1) Z (z1)- i/2 X(x2) Y(y2) Z(z2)

    Args:
        ladder_operator (tuple[int, int]): the ladder operator
        n_qubits (int): the number of qubits
        x1, y1, z1, x2, y2, z2 (list[int] or set[int]):
        lists or sets of indices of qubits,
        to which corresponding Pauli gates are applied.
        These lists (sets) are defined for each transform.

    Returns:
        QubitOperator
    """
    coefficient_1 = 0.5
    coefficient_2 = -0.5j if ladder_operator[1] else 0.5j
    transf_op_1 = QubitOperator(
        tuple((index, 'X') for index in x1) + tuple((index, 'Y') for index in y1) + tuple((index, 'Z') for index in z1),
        coefficient_1,
    )
    transf_op_2 = QubitOperator(
        tuple((index, 'X') for index in x2) + tuple((index, 'Y') for index in y2) + tuple((index, 'Z') for index in z2),
        coefficient_2,
    )
    return transf_op_1 + transf_op_2


def jordan_wigner(operator):
    r"""
    Apply Jordan-Wigner transform.

    The Jordan-Wigner transform holds the initial occupation number locally, which change the formular of
    fermion operator into qubit operator following the equation.

    .. math::
        a^\dagger_{j}\rightarrow \sigma^{-}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i}
        a_{j}\rightarrow \sigma^{+}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i},
    where the :math:`\sigma_{+}= \sigma^{X} + i \sigma^{Y}` and :math:`\sigma_{-} = \sigma^{X} - i\sigma^{Y}` is the
    Pauli spin raising and lowring operator.

    Returns:
        QubitOperator, qubit operator after jordan_wigner transformation.
    """
    transf_op = QubitOperator()
    for term in operator.terms:
        # Initialize identity matrix.
        transformed_term = QubitOperator((), 1 * operator.terms[term])

        # Loop through operators, transform and multiply.
        for ladder_operator in term:
            # Define lists of qubits to apply Pauli gates
            index = ladder_operator[0]
            x1 = [index]
            y1 = []
            z1 = list(range(index))
            x2 = []
            y2 = [index]
            z2 = list(range(index))
            transformed_term *= _transform_ladder_operator(ladder_operator, x1, y1, z1, x2, y2, z2)
        transf_op += transformed_term

    return transf_op


def reversed_jordan_wigner(self):
    """
    Apply reversed Jordan-Wigner transform.

    Returns:
        FermionOperator, fermion operator after reversed_jordan_wigner transformation.
    """
    if not isinstance(self.operator, QubitOperator):
        raise TypeError('This method can be only applied for QubitOperator.')
    transf_op = FermionOperator()

    # Loop through terms.
    for term in self.operator.terms:
        transformed_term = FermionOperator(())
        if term:
            working_term = QubitOperator(term)
            pauli_operator = term[-1]
            while pauli_operator is not None:

                # Handle Pauli Z.
                if pauli_operator[1] == 'Z':
                    transformed_pauli = FermionOperator(()) + number_operator(None, pauli_operator[0], -2.0)

                # Handle Pauli X and Y.
                else:
                    raising_term = FermionOperator(((pauli_operator[0], 1),))
                    lowering_term = FermionOperator(((pauli_operator[0], 0),))
                    if pauli_operator[1] == 'Y':
                        raising_term *= 1.0j
                        lowering_term *= -1.0j

                    transformed_pauli = raising_term + lowering_term

                    # Account for the phase terms.
                    for j in reversed(range(pauli_operator[0])):
                        z_term = QubitOperator(((j, 'Z'),))
                        working_term = z_term * working_term
                    term_key = list(working_term.terms)[0]
                    transformed_pauli *= working_term.terms[term_key]
                    working_term.terms[list(working_term.terms)[0]] = 1.0

                # Get next non-identity operator acting below
                # 'working_qubit'.
                working_qubit = pauli_operator[0] - 1
                for working_operator in reversed(list(working_term.terms)[0]):
                    if working_operator[0] <= working_qubit:
                        pauli_operator = working_operator
                        break
                    pauli_operator = None

                # Multiply term by transformed operator.
                transformed_term *= transformed_pauli
        # Account for overall coefficient
        transformed_term *= self.operator.terms[term]
        transf_op += transformed_term
    return transf_op


def parity(operator, n_qubits):
    r"""
    Apply parity transform.

    The parity transform stores the initial occupation number nonlocally,
    with the formular:

    .. math::
        \left|f_{M-1}, f_{M-2},\cdots, f_0\right> \rightarrow \left|q_{M−1}, q_{M−2},\cdots, q_0\right>,
    where

    .. math::
        q_{m} = \left|\left(\sum_{i=0}^{m-1}f_{i}\right) mod\ 2 \right>
    Basically, this formular could be written as this,

    .. math::
        p_{i} = \sum{[\pi_{n}]_{i,j}} f_{j},
    where :math:`\pi_{n}` is the :math:`N\times N` square matrix,
    :math:`N` is the total qubit number. The operator changes follows the following equation as:

    .. math::
        a^\dagger_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
        \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}-i\sigma_j^Y\right) X \sigma^{Z}_{j-1}
        a_{j}\rightarrow\frac{1}{2}\left(\prod_{i=j+1}^N
        \left(\sigma_i^X X\right)\right)\left( \sigma^{X}_{j}+i\sigma_j^Y\right) X \sigma^{Z}_{j-1}

    Returns:
        QubitOperator, qubits operator after parity transformation.
    """
    if not isinstance(operator, FermionOperator):
        raise TypeError('This method can be only applied for FermionOperator.')
    transf_op = QubitOperator()
    for term in operator.terms:
        # Initialize identity matrix.
        transformed_term = QubitOperator((), 1 * operator.terms[term])

        # Loop through operators, transform and multiply.
        for ladder_operator in term:
            # Define lists of qubits to apply Pauli gates
            index = ladder_operator[0]
            x1 = list(range(index, n_qubits))
            y1 = []
            z1 = [index - 1] if index > 0 else []
            x2 = list(range(index + 1, n_qubits))
            y2 = [index]
            z2 = []
            transformed_term *= _transform_ladder_operator(ladder_operator, x1, y1, z1, x2, y2, z2)
        transf_op += transformed_term

    return transf_op
