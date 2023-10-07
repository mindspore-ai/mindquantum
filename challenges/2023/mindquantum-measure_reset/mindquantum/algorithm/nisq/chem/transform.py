#   Copyright (c) 2020 Huawei Technologies Co.,ltd.
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
"""
Module implementing a conversion from fermion type operators to qubit type operators.

Thus can be simulated in quantum computer.
"""

from mindquantum import _math

# pylint: disable=no-member,protected-access
from mindquantum.core.operators.utils import (
    FermionOperator,
    QubitOperator,
    count_qubits,
)


class Transform:
    r"""
    Class for transforms of fermionic and qubit operators.

    `jordan_wigner`, `parity`, `bravyi_kitaev`, `bravyi_kitaev_tree`,
    `bravyi_kitaev_superfast` will transform `FermionOperator` to
    `QubitOperator`. `reversed_jordan_wigner` will transform `QubitOperator`
    to `FermionOperator`.

    Args:
        operator (Union[FermionOperator, QubitOperator]): The input
            FermionOperator or QubitOperator that need to do transform.
        n_qubits (int): The total qubits of given operator. If ``None``, then we will count it automatically.
            Default: ``None``.

    Examples:
        >>> from mindquantum.core.operators import FermionOperator
        >>> op1 = FermionOperator('1^')
        >>> op1
        1.0 [1^]
        >>> op_transform = Transform(op1)
        >>> from mindquantum.algorithm.nisq import Transform
        >>> op_transform.jordan_wigner()
        0.5 [Z0 X1] +
        -0.5j [Z0 Y1]
        >>> op_transform.parity()
        0.5 [Z0 X1] +
        -0.5j [Y1]
        >>> op_transform.bravyi_kitaev()
        0.5 [Z0 X1] +
        -0.5j [Y1]
        >>> op_transform.ternary_tree()
        0.5 [X0 Z1] +
        -0.5j [Y0 X2]
        >>> op2 = FermionOperator('1^', 'a')
        >>> Transform(op2).jordan_wigner()
        0.5*a [Z0 X1] +
        -0.5*I*a [Z0 Y1]
    """

    def __init__(self, operator, n_qubits=None):
        """Initialize a Transform object."""
        if not isinstance(operator, (FermionOperator, QubitOperator)):
            raise TypeError("Operator must be a FermionOperator or QubitOperator")
        if n_qubits is None:
            n_qubits = count_qubits(operator)
        if n_qubits < count_qubits(operator):
            raise ValueError('Invalid number of qubits specified.')

        self.n_qubits = n_qubits
        self.operator = operator

    def jordan_wigner(self):
        r"""
        Apply Jordan-Wigner transform.

        The Jordan-Wigner transform holds the initial occupation number locally, which change the formular of
        fermion operator into qubit operator following the equation.

        .. math::

            a^\dagger_{j}\rightarrow \sigma^{-}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i}

            a_{j}\rightarrow \sigma^{+}_{j} X \prod_{i=0}^{j-1}\sigma^{Z}_{i},

        where the :math:`\sigma_{+} = \sigma^{X} + i \sigma^{Y}` and
        :math:`\sigma_{-} = \sigma^{X} - i\sigma^{Y}` is the
        Pauli spin raising and lowring operator.

        Returns:
            QubitOperator, qubit operator after jordan_wigner transformation.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError('This method can be only applied for FermionOperator.')
        return QubitOperator(_math.ops.jordan_wigner(self.operator), internal=True)

    def parity(self):
        r"""
        Apply parity transform.

        The parity transform stores the initial occupation number nonlocally,
        with the formular:

        .. math::

            \left|f_{M-1}, f_{M-2},\cdots, f_0\right> \rightarrow \left|q_{M-1}, q_{M-2},\cdots, q_0\right>,

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
        if not isinstance(self.operator, FermionOperator):
            raise TypeError('This method can be only applied for FermionOperator.')
        return QubitOperator(_math.ops.parity(self.operator, self.n_qubits))

    def bravyi_kitaev(self):  # pylint: disable=too-many-locals
        r"""
        Apply Bravyi-Kitaev transform.

        The Bravyi-Kitaev basis is a middle between Jordan-Wigner
        and parity transform. That is, it balances the locality of occupation and parity information
        for improved simulation efficiency. In this scheme, qubits store the parity
        of a set of :math:`2^x` orbitals, where :math:`x \ge 0`. A qubit of index :math:`j` always
        stores orbital :math:`j`.
        For even values of :math:`j`, this is the only orbital
        that it stores, but for odd values of :math:`j`, it also stores a certain
        set of adjacent orbitals with index less than :math:`j`.
        For the occupation transformation, we follow the
        formular:

        .. math::

            b_{i} = \sum{[\beta_{n}]_{i,j}} f_{j},

        where :math:`\beta_{n}` is the :math:`N\times N` square matrix,
        :math:`N` is the total qubit number.
        The qubits index are divide into three sets,
        the parity set, the update set and flip set.
        The parity of this set of qubits has
        the same parity as the set of orbitals with index less than :math:`j`,
        and so we will call this set of qubit indices the "parity set" of
        index :math:`j`, or :math:`P(j)`.

        The update set of index :math:`j`, or :math:`U(j)` contains the set of qubits (other than
        qubit :math:`j`) that must be updated when the occupation of orbital :math:`j`

        The flip set of index :math:`j`, or :math:`F(j)` contains the set of BravyiKitaev qubits determines
        whether qubit :math:`j` has the same parity or inverted parity with
        respect to orbital :math:`j`.

        Please see some detail explanation in the paper `The Bravyi-Kitaev transformation for
        quantum computation of electronic structure <https://doi.org/10.1063/1.4768229>`_.

        Implementation from `Fermionic quantum computation <https://arxiv.org/abs/quant-ph/0003137>`_ and
        `A New Data Structure for Cumulative Frequency Tables <https://doi.org/10.1002/spe.4380240306>`_
        by Peter M. Fenwick.

        Returns:
            QubitOperator, qubit operator after bravyi_kitaev transformation.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError('This method can be only applied for FermionOperator.')
        return QubitOperator(_math.ops.bravyi_kitaev(self.operator, self.n_qubits))

    def bravyi_kitaev_superfast(self):
        r"""
        Apply Bravyi-Kitaev Superfast transform.

        Implementation from `Bravyi-Kitaev Superfast simulation of fermions on a
        quantum computer <https://arxiv.org/abs/1712.00446>`_.

        Note that only hermitian operators of form will be transformed.

        .. math::

            C + \sum_{p, q} h_{p, q} a^\dagger_p a_q +
                \sum_{p, q, r, s} h_{p, q, r, s} a^\dagger_p a^\dagger_q a_r a_s

        where :math:`C` is a constant.

        Returns:
            QubitOperator, qubit operator after bravyi_kitaev_superfast.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError('This method can be only applied for FermionOperator.')
        return QubitOperator(_math.ops.bravyi_kitaev_superfast(self.operator))

    def ternary_tree(self):  # pylint: disable=too-many-locals
        """
        Apply Ternary tree transform.

        Implementation from `Optimal fermion-to-qubit mapping via ternary trees with
        applications to reduced quantum states learning <https://arxiv.org/abs/1910.10746>`_.

        Returns:
            QubitOperator, qubit operator after ternary_tree transformation.
        """
        if not isinstance(self.operator, FermionOperator):
            raise TypeError('This method can be only applied for FermionOperator.')
        return QubitOperator(_math.ops.ternary_tree(self.operator, self.n_qubits))

    def reversed_jordan_wigner(self):
        """
        Apply reversed Jordan-Wigner transform.

        Returns:
            FermionOperator, fermion operator after reversed_jordan_wigner transformation.
        """
        if not isinstance(self.operator, QubitOperator):
            raise TypeError('This method can be only applied for QubitOperator.')

        return FermionOperator(_math.ops.reverse_jordan_wigner(self.operator, self.n_qubits))
