# Portions Copyright 2021 Huawei Technologies Co., Ltd
# Portions Copyright 2017 The OpenFermion Developers.
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

# pylint: disable=duplicate-code
"""This module is generated the Fermion Operator."""


from openfermion import FermionOperator as OFFermionOperator

from mindquantum.core.parameterresolver import ParameterResolver
from mindquantum.mqbackend import complex_pr, real_pr

# NB: C++ actually supports FermionOperatorD and FermionOperatorCD that are purely numerical FermionOperator classes
from .._mindquantum_cxx.ops import (
    FermionOperatorBase,
    FermionOperatorPRCD,
    FermionOperatorPRD,
)
from ._terms_operators import TermsOperator

# ==============================================================================


class FermionOperator(TermsOperator):
    r"""
    Definition of a Fermion Operator.

    The Fermion Operator such as FermionOperator(' 4^ 3 9 3^ ') are used to represent :math:`a_4^\dagger a_3 a_9
    a_3^\dagger`.


    These are the Basic Operators to describe a fermionic system, such as a Molecular system.
    The FermionOperator are follows the anti-commutation relationship.

    Args:
        terms (str): The input term of fermion operator. Default: None.
        coefficient (Union[numbers.Number, str, ParameterResolver]): The coefficient for the corresponding single
            operators Default: 1.0.

    Examples:
        >>> from mindquantum.core.operators import FermionOperator
        >>> a_p_dagger = FermionOperator('1^')
        >>> a_p_dagger
        1.0 [1^]
        >>> a_q = FermionOperator('0')
        >>> a_q
        1.0 [0]
        >>> zero = FermionOperator()
        >>> zero
        0
        >>> identity= FermionOperator('')
        >>> identity
        1.0 []
        >>> para_op = FermionOperator('0 1^', 'x')
        >>> para_op
        x [0 1^]
        >>> para_dt = {'x':2}
        >>> op = para_op.subs(para_dt)
        >>> op
        2 [0 1^]
    """

    cxx_base_klass = FermionOperatorBase
    real_pr_klass = FermionOperatorPRD
    complex_pr_klass = FermionOperatorPRCD
    openfermion_klass = OFFermionOperator

    @classmethod
    def create_cpp_obj(cls, term, coeff):
        """
        Create a C++ FermionOperator object based on the coefficient type.

        Args:
            term (str): The input term of fermion operator. Default: None.
            coeff (Union[numbers.Number, str, ParameterResolver]): The coefficient for the corresponding single
                operators Default: 1.0.
        """
        klass = None

        if isinstance(coeff, float):
            if coeff is not None:
                coeff = complex_pr(coeff)
            klass = FermionOperatorPRD
        elif isinstance(coeff, complex):
            if coeff is not None:
                coeff = real_pr(coeff)
            klass = FermionOperatorPRCD
        elif isinstance(coeff, ParameterResolver):
            if isinstance(ParameterResolver._cpp_obj, real_pr):
                klass = FermionOperatorPRD
            else:
                klass = FermionOperatorPRCD
            coeff = coeff._cpp_obj
        elif isinstance(coeff, real_pr):
            klass = FermionOperatorPRD
        elif isinstance(coeff, complex_pr):
            klass = FermionOperatorPRCD
        elif coeff is not None:
            return TypeError(f'FermionOperator does not support {type(coeff)} as coefficient type.')

        if term is None:
            return klass()
        if coeff is None:
            return klass(term)
        return klass(term, coeff)

    def normal_ordered(self) -> "FermionOperator":
        """
        Return the normal ordered form of the Fermion Operator.

        Returns:
            FermionOperator, the normal ordered FermionOperator.

        Examples:
            >>> from mindquantum.core.operators import FermionOperator
            >>> origin = FermionOperator('0 1^')
            >>> origin
            1.0 [0 1^]
            >>> origin.normal_ordered()
            -1.0 [1^ 0]
        """
        return FermionOperator(self._cpp_obj.normal_ordered())


__all__ = ['FermionOperator']
