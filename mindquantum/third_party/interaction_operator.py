#   Copyright 2017 The OpenFermion Developers.
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
This is the class that to store the non-zero coefficient of the molecular
Hamiltonian. It can be further used to construct the molecular Hamiltonian.
"""
# Note this module, we did not modify much of the OpenFermion file

import itertools
from mindquantum.ops.polynomial_tensor import PolynomialTensor


class InteractionOperator(PolynomialTensor):
    r"""
    Class to store 'interaction opeartors' which are used to configure
    a ferinonic molecular Hamiltonian.

    The Hamiltonian including one-body and two-body terms which conserve spin
    and parity. In this module, the stored coefficient could be represented the
    molecualr Hamiltonians througth the FermionOperator class.

    Note:
        The operators stored in this class has the form:

        .. math::

            C + \sum_{p, q} h_{[p, q]} a^\dagger_p a_q +
            \sum_{p, q, r, s} h_{[p, q, r, s]} a^\dagger_p a^\dagger_q a_r a_s.

        Where :math:`C` is a constant.

    Args:
        constant (numbers.Number): A constant term in the operator given as a
                float. For instance, the nuclear repulsion energy.
        one_body_tensor (numpy.ndarray): The coefficients of the one-body terms (h[p, q]).
            This is an :math:`n_\text{qubits}\times n_\text{qubits}` numpy array of floats.
            By default we store the numpy array with keys: :math:`a^\dagger_p a_q` (1,0).
        two_body_tensor (numpy.ndarray): The coefficients of the two-body terms
            (h[p, q, r, s]). This is an
            :math:`n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}\times n_\text{qubits}`
            numpy array of floats.By default we store the numpy array
            with keys: :math:`a^\dagger_p a^\dagger_q a_r a_s` (1, 1, 0, 0).
    """
    def __init__(self, constant, one_body_tensor, two_body_tensor):
        # make sure only non-zero tensor elements exist in the normal-ordered
        # form
        super().__init__({
            (): constant,
            (1, 0): one_body_tensor,
            (1, 1, 0, 0): two_body_tensor
        })

    def unique_iter(self, complex_valued=False):
        r"""
        Iterate all terms that are not in the same symmetry group.

        Four point symmetry:
            1. pq = qp.
            2. pqrs = srqp = qpsr = rspq.
        Eight point symmetry(when complex_valued is False):
            1. pq = qp.
            2. pqrs = rqps = psrq = srqp = qpsr = rspq = spqr = qrsp.

        Args:
            complex_valued (bool):
                Whether the operator has complex coefficients. Default: False.
        """
        # Constant.
        if self.constant:
            yield ()

        # One-body terms.
        for p in range(self.n_qubits):
            for q in range(p + 1):
                if self.one_body_tensor[p, q]:
                    yield (p, 1), (q, 0)

        # Two-body terms.
        two_body_index = set()
        for quad in itertools.product(range(self.n_qubits), repeat=4):
            if self.two_body_tensor[quad] and quad not in two_body_index:
                two_body_index |= set(
                    _symmetric_two_body_terms(quad, complex_valued))
                yield tuple(zip(quad, (1, 1, 0, 0)))


def _symmetric_two_body_terms(quad, complex_valued):
    """symmetric_two_body_terms."""
    p, q, r, s = quad
    # four point symmetry
    yield p, q, r, s
    yield q, p, s, r
    yield s, r, q, p
    yield r, s, p, q
    # complex_value false, then eight symmetry
    if not complex_valued:
        yield p, s, r, q
        yield q, r, s, p
        yield s, p, q, r
        yield r, q, p, s
