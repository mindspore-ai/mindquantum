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

from ...core.operators.polynomial_tensor import PolynomialTensor
from ...core.parameterresolver import ParameterResolver
from ...mqbackend import complex_pr, real_pr

# NB: C++ actually supports FermionOperatorD and FermionOperatorCD that are purely numerical FermionOperator classes
from .._mindquantum_cxx.ops import (
    FermionOperatorBase,
    FermionOperatorPRCD,
    FermionOperatorPRD,
)
from ..utils import TermValue
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

    _type_conversion_table = {
        complex_pr: complex_pr_klass,
        complex: complex_pr_klass,
        real_pr: real_pr_klass,
        float: real_pr_klass,
    }

    def __init__(self, *args):
        """
        Initialize a FermionOperator instance.

        Args:
            *args: Variable length argument list:
                - PolynomialTensor
                - Any type as specified by TermsOperator.__init__
        """
        if len(args) == 1 and isinstance(args[0], PolynomialTensor):
            terms = {}
            for term in args[0]:
                terms[tuple((i, TermValue[j]) for i, j in term)] = ParameterResolver(args[0][term])
            super().__init__(terms)
        else:
            super().__init__(*args)

    def __str__(self) -> str:
        """Return string expression of a TermsOperator."""
        out = []
        for terms, coeff in self.terms.items():
            terms_str = ' '.join([f"{idx}{'^' if TermValue[fermion] else ''}" for idx, fermion in terms])
            out.append(f"{coeff.expression()} [{terms_str}] ")
        if not out:
            return "0"
        return "+\n".join(out)

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
