# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from mindquantum._math.ops import QubitOperator as QubitOperator_
from mindquantum.core.tensor import dtype as mqtype
from mindquantum.core.tensor.parameterresolver import ParameterResolver


class QubitOperator(QubitOperator_):
    def __init__(self, terms, coefficient=1.0, internal=False):
        if isinstance(terms, QubitOperator_):
            internal = True
        if internal:
            QubitOperator_.__init__(self, terms)
        else:
            QubitOperator_.__init__(self, terms, ParameterResolver(coefficient))

    @property
    def imag(self):
        """
        Convert the coefficient to its imag part.

        Returns:
            QubitOperator, the imag part of this qubit operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.imag.compress()
            2 [X0]
        """
        return QubitOperator(QubitOperator_.imag(self))

    @property
    def real(self):
        """
        Convert the coefficient to its real part.

        Returns:
            QubitOperator, the real part of this qubit operator.

        Examples:
            >>> from mindquantum.core.operators import QubitOperator
            >>> f = QubitOperator('X0', 1 + 2j) + QubitOperator('Y0', 'a')
            >>> f.real.compress()
            1 [X0] +
            a [Y0]
        """
        return QubitOperator(QubitOperator_.real(self))


if __name__ == "__main__":
    a = QubitOperator("X0", 1.3 + 34j)
    print(a)
