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

#   This module we develop is default being licensed under Apache 2.0 license,
#   and also uses or refactor Fermilib and OpenFermion licensed under
#   Apache 2.0 license.
"""
This is the base class that to represent fermionic molecualr or Hamiltonian.
"""

import copy
import itertools
import numpy

EQ_TOLERANCE = 1e-8


class PolynomialTensorError(Exception):
    r"""The particular Exception for this particular class"""


class PolynomialTensor:
    r"""
    Class to store the coefficient of the fermionic ladder operators
    in a tensor form.
    For instance, in a molecular Hamiltonian (degree 4 polynomial) which
    conserves particle number, there are only three kinds of terms,
    namely constant term, single excitation :math:`a^\dagger_p a_q` and
    double excitation terms :math:`a^\dagger_p a^\dagger_q a_r a_s`,
    and their corresponding coefficients can be stored in an scalar,
    :math:`n_\text{qubits}\times n_\text{qubits}` matrix and
    :math:`n_\text{qubits}\times n_\text{qubits} n_\text{qubits}\times n_\text{qubits}` matrix.
    Note that each tensor must have an even number of dimensions due to
    the parity conservation.
    Much of the functionality of this class is similar to that of
    FermionOperator.

    Args:
        n_body_tensors(dict): A dictionary storing the tensors describing
            n-body interactions. The keys are tuples that indicate the
            type of tensor.
            For instance, n_body_tensors[()] would return a constant,
            while a n_body_tensors[(1, 0)] would be an
            :math:`n_\text{qubits}\times n_\text{qubits}` numpy
            array, and n_body_tensors[(1,1,0,0)]
            would return a
            :math:`n_\text{qubits}\times n_\text{qubits} n_\text{qubits}\times n_\text{qubits}`
            numpy array
            and those constant and array represent the coefficients of terms of
            the form identity, :math:`a^\dagger_p a_q`,
            :math:`a^\dagger_p a^\dagger_q a_r a_s`, respectively. Default: None.

    Note:
        Here '1' represents :math:`a^\dagger`, while '0' represent :math:`a`.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.ops import PolynomialTensor
        >>> constant = 1
        >>> one_body_term = np.array([[1,0],[0,1]])
        >>> two_body_term = two_body_term = np.array([[[[1,0],[0,1]],[[1,0],[0,1]]],[[[1,0],[0,1]],[[1,0],[0,1]]]])
        >>> n_body_tensors = {(): 1, (1,0): one_body_term,(1,1,0,0):two_body_term}
        >>> poly_op = PolynomialTensor(n_body_tensors)
        >>> poly_op
        () 1
        ((0, 1), (0, 0)) 1
        ((1, 1), (1, 0)) 1
        ((0, 1), (0, 1), (0, 0), (0, 0)) 1
        ((0, 1), (0, 1), (1, 0), (1, 0)) 1
        ((0, 1), (1, 1), (0, 0), (0, 0)) 1
        ((0, 1), (1, 1), (1, 0), (1, 0)) 1
        ((1, 1), (0, 1), (0, 0), (0, 0)) 1
        ((1, 1), (0, 1), (1, 0), (1, 0)) 1
        ((1, 1), (1, 1), (0, 0), (0, 0)) 1
        ((1, 1), (1, 1), (1, 0), (1, 0)) 1
        >>> # get the constant
        >>> poly_op.constant
        1
        >>> # set the constant
        >>> poly_op.constant = 2
        >>> poly_op.constant
        2
        >>> poly_op.n_qubits
        2
        >>> poly_op.one_body_tensor
        array([[1, 0],
               [0, 1]])
        >>> poly_op.two_body_tensor
        array([[[[1, 0],
                 [0, 1]],
                [[1, 0],
                 [0, 1]]],
               [[[1, 0],
                 [0, 1]],
                 [[1, 0],
                  [0, 1]]]])
    """
    __hash__ = None

    def __init__(self, n_body_tensors=None):
        self.n_body_tensors = n_body_tensors
        self.n_qubits = 0
        for key, _ in self.n_body_tensors.items():
            if key == ():
                pass
            elif len(key) == 2 or len(key) == 4:  # one body tensors
                self.n_qubits = self.n_body_tensors[key].shape[0]
            else:
                PolynomialTensorError("Unexpected type of n-body-tensors!")

    @property
    def constant(self):
        """get the value of the identity term"""
        return self.n_body_tensors.get(())

    @constant.setter
    def constant(self, value):
        """set the value of the identity term"""
        self.n_body_tensors[()] = value

    @property
    def one_body_tensor(self):
        """get the one-body term"""
        if (1, 0) in self.n_body_tensors:
            return self.n_body_tensors[(1, 0)]

        return 0

    @one_body_tensor.setter
    def one_body_tensor(self, value):
        """
        set the value of the one body term,
        the value should numpy array with size n_qubits x n_qubits
        """
        self.n_body_tensors[(1, 0)] = value

    @property
    def two_body_tensor(self):
        """get the two body term"""
        if (1, 1, 0, 0) in self.n_body_tensors:
            return self.n_body_tensors[(1, 1, 0, 0)]

        return 0

    @two_body_tensor.setter
    def two_body_tensor(self, value):
        """
        set the two body term, the value should be of numpy
        array with size n_qubits x n_qubits x n_qubits x n_qubits
        """
        self.n_body_tensors[(1, 1, 0, 0)] = value

    def __getitem__(self, args):
        r"""
        Look up the matrix table.

        Args:
            args(tuples): Tuples indicating which coefficient to get.
            For instance,
            `my_tensor[(3, 1), (4, 1), (2, 0)]` means look for the coefficient
            of fermionic ladder operator (a^\dagger_3 a^\dagger_4 a_2 )
                returns
                `my_tensor.n_body_tensors[1, 1, 0][3, 4, 2]`

        Note: this supports single element extraction
        """
        if args == ():
            return self.constant

        # change it into array
        index, key = tuple(zip(*args))[0], tuple(zip(*args))[1]
        return self.n_body_tensors[key][index]

    def __setitem__(self, args, value):
        """
        Set matrix element.

        Args:
            args(tuples): Tuples indicating which terms to set the
            corresponding coefficient.
        """
        if args == ():
            self.constant = value
        else:
            # handle with the case (1,0) or ((1,0),(2,1)) they both have the
            # length 2
            index, key = tuple(zip(*args))[0], tuple(zip(*args))[1]
            self.n_body_tensors[key][index] = value

    def __eq__(self, other):
        # first check qubits number
        if self.n_qubits != other.n_qubits:
            return False
        # then check the maximum difference whether within the EQ_TOLERANCE
        diff = 0.
        self_keys = set(self.n_body_tensors.keys())
        other_keys = set(other.n_body_tensors.keys())
        # check the intersection part
        for key in self_keys.intersection(other_keys):
            if key == () or key is not None:
                self_tensor = self.n_body_tensors[key]
                other_tensor = other.n_body_tensors[key]
                discrepancy = numpy.amax(
                    numpy.absolute(self_tensor - other_tensor))
                diff = max(diff, discrepancy)

        # check the difference part
        for key in self_keys.symmetric_difference(other_keys):
            if key == () or key is not None:
                tensor = self.n_body_tensors[key] if self.n_body_tensors[key]\
                    is not None \
                    else other.n_body_tensors[key]
                discrepancy = numpy.amax(numpy.abs(tensor))
                diff = max(diff, discrepancy)
        return diff < EQ_TOLERANCE

    def __ne__(self, other):
        return not self == other

    def __iadd__(self, addend):
        """
        In-place method for += addition of PolynomialTensor.

        Args:
            addend (PolynomialTensor): The addend.

        Returns:
            sum (PolynomialTensor), Mutated self.

        Raises:
            TypeError: Cannot add invalid addend type.
        """
        if not isinstance(addend, type(self)):
            raise PolynomialTensorError(
                "Cannot add invalid type! \n Expect {}".format(type(self)))
        # check dimension, self.n_qubits
        if self.n_qubits != addend.n_qubits:
            raise PolynomialTensorError(
                "Can not add invalid type, the shape does not match!")
        # add the common part
        self_keys = set(self.n_body_tensors.keys())
        addend_keys = set(addend.n_body_tensors.keys())
        for key in self_keys.intersection(addend_keys):
            self.n_body_tensors[key] = numpy.add(self.n_body_tensors[key],
                                                 addend.n_body_tensors[key])

        for key in addend_keys.difference(
                self_keys):  # the term in added but not in self
            if key:
                self.n_body_tensors[key] = addend.n_body_tensors[key]
        return self

    def __add__(self, addend):
        """
        Args:
            added(PolynomialTensor): The addend.

        Returns:
            sum (PolynomialTensor), un-mutated self, but has new instance

        Raises:
            TypeError: Cannot add invalid operator type.
        """
        sum_addend = copy.deepcopy(self)
        sum_addend += addend
        return sum_addend

    def __neg__(self):
        """Return negation of the PolynomialTensor,mutated itself"""
        for key in self.n_body_tensors:
            self.n_body_tensors[key] = numpy.negative(self.n_body_tensors[key])
        return self

    def __isub__(self, subtractend):
        """
        In-place method for -= subtraction of PolynomialTensor.

        Args:
            subtractend (PolynomialTensor): subtractend.

        Returns:
            subtract (PolynomialTensor), Mutated self.

        Raises:
            TypeError: Cannot sub invalid addend type.
        """
        if not isinstance(subtractend, type(self)):
            raise PolynomialTensorError(
                "Cannot sub invalid type! \n Expect {}".format(type(self)))
        # check dimension, self.n_qubits
        if self.n_qubits != subtractend.n_qubits:
            raise PolynomialTensorError(
                "Cannot sub invalid type, the shape does not match!")
        # sub the common part
        self_keys = set(self.n_body_tensors.keys())
        sub_keys = set(subtractend.n_body_tensors.keys())
        for key in self_keys.intersection(sub_keys):
            self.n_body_tensors[key] = numpy.subtract(
                self.n_body_tensors[key], subtractend.n_body_tensors[key])
        for key in sub_keys.difference(
                self_keys):  # the term in sub but not in self
            if key:
                self.n_body_tensors[key] = numpy.negative(
                    subtractend.n_body_tensors[key])
        return self

    def __sub__(self, subtractend):
        """
        Args:
            subtractend(PolynomialTensor): The subtractend.

        Returns:
            subtractend (PolynomialTensor), un-mutated self,
            but has new instance

        Raises:
            TypeError: Cannot sub invalid operator type.
        """
        subend = copy.deepcopy(self)
        subend -= subtractend
        return subend

    def __imul__(self, multiplier):
        """
        In-place multiply (*=) with scalar or operator of the same type.
        Default implementation is to multiply coefficients and concatenate
        terms (same operator).

        Args:
            multiplier(complex, float, or PolynomialTensor): multiplier

        Returns:
            products(PolynomialTensor)

        Raises:
            TypeError: cannot multiply invalid type of multiplier.
        """
        # hand with scalar
        if isinstance(multiplier, (int, float, complex)):
            for key in self.n_body_tensors:
                self.n_body_tensors[key] *= multiplier

        elif isinstance(multiplier, type(self)):
            if self.n_qubits != multiplier.n_qubits:
                raise PolynomialTensorError(
                    "Cannot multiply invalid type, the shape does not match!")
            # note we do not deal with the key multiplication,
            # unlike that in FermionOperator, which is possible
            self_keys = set(self.n_body_tensors.keys())
            multiply_keys = set(multiplier.n_body_tensors.keys())

            for key in self_keys.intersection(multiply_keys):
                self.n_body_tensors[key] = numpy.multiply(
                    self.n_body_tensors[key], multiplier.n_body_tensors[key])

            for key in self_keys.difference(
                    multiply_keys):  # the term in added but not in self
                if key == ():
                    self.constant = 0
                else:
                    self.n_body_tensors[key] = numpy.zeros(
                        self.n_body_tensors[key].shape)
        else:
            raise PolynomialTensorError("Cannot multiply invalid type!")

        return self

    def __mul__(self, multiplier):
        """
        Method for * addition of PolynomialTensor.

        Args:
            multiplier (PolynomialTensor): The multiplier to multiply.

        Returns:
            multiply (PolynomialTensor), un-Mutated self.

        Raises:
            TypeError: Cannot multiply invalid type.
        """
        if isinstance(multiplier, (int, float, complex, type(self))):
            # make use of the *= method
            product_results = copy.deepcopy(self)
            product_results *= multiplier
        else:
            raise PolynomialTensorError(
                'Cannot multiply invalid type to {}.'.format(type(self)))
        return product_results

    def __rmul__(self, multiplier):
        """
        Return multiplier * self

        Args:
            multiplier: The operator to multiply.

        Returns:
            a new instance of  PolynomialTensor

        Raises:
            TypeError: Cannot multiply invalid type.
        """
        if isinstance(multiplier,
                      (int, float,
                       complex)):  # make use of the * method, basically scalar
            return self * multiplier

        raise PolynomialTensorError(
            'Cannot multiply invalid operator type to {}.'.format(type(self)))

    def __itruediv__(self, divisor):
        """
        Returns self/divisor for the scalar

        Args:
            divisor(int, float, complex): scalar

        Returns:
            a new instance of PolynomialTensor

        Raises:
            TypeError: cannot divide non-numeric type.

        """
        if isinstance(divisor, (int, float, complex)) and divisor != 0:
            for key in self.n_body_tensors:
                self.n_body_tensors[key] /= divisor
        else:
            raise PolynomialTensorError(
                'Cannot divide the {} by non_numeric type or \
            the divisor is 0.'.format(type(self)))
        return self

    def __truediv__(self, divisor):
        if isinstance(divisor, (int, float, complex)) and divisor != 0:
            quotient = copy.deepcopy(self)
            quotient /= divisor
        else:
            raise PolynomialTensorError(
                'Cannot divide the {} by non_numeric type or \
            the divisor is 0.'.format(type(self)))
        return quotient

    # ba careful with this function
    def __div__(self, divisor):
        """ For compatibility with Python 2. """
        return self.__truediv__(divisor)

    def __iter__(self):
        """Iterate over non-zero elements in the PolynomialTensor."""
        def sort_key(key):
            """
            This determines how the keys to n_body_tensors
            should be sorted by mapping it to the corresponding integer.
            """
            if key == ():
                return 0

            key_int = int(''.join(map(str, key)))
            return key_int

        for key in sorted(self.n_body_tensors, key=sort_key):
            if key == ():
                yield ()
            else:
                n_body_tensors = self.n_body_tensors[key]  # get the matrix
                # look up the non-zero elements in the n_body_tensors
                for index in itertools.product(range(self.n_qubits),
                                               repeat=len(key)):
                    if n_body_tensors[index]:
                        yield tuple(zip(index, key))

    def __str__(self):
        """Print out the non-zero elements of PolynomialTensor."""
        strings = []
        for key in self:
            strings.append('{} {}\n'.format(key, self[key]))
        return ''.join(strings) if strings else '0'

    def __repr__(self):
        return str(self)
