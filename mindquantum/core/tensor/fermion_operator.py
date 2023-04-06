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
import numbers

from mindquantum._math.ops import FermionOperator as FermionOperator_
from mindquantum._math.ops import f_term_value
from mindquantum.core.tensor import dtype as mqtype
from mindquantum.core.tensor.parameterresolver import ParameterResolver
from mindquantum.core.tensor.term_value import TermValue


class FermionOperator(FermionOperator_):
    def __init__(self, terms=None, coefficient=1.0, internal=False):
        if terms is None:
            FermionOperator_.__init__(self)
        if isinstance(terms, FermionOperator_):
            internal = True
        if internal:
            FermionOperator_.__init__(self, terms)
        else:
            FermionOperator_.__init__(self, terms, ParameterResolver(coefficient))

    @property
    def imag(self):
        return FermionOperator(FermionOperator_.imag(self))

    @property
    def real(self):
        return FermionOperator(FermionOperator_.real(self))

    def __len__(self):
        return FermionOperator_.size(self)

    def __copy__(self):
        return FermionOperator(FermionOperator_.__copy__(self))

    def __deepcopy__(self):
        return FermionOperator(FermionOperator_.__copy__(self))

    def __add__(self, other):
        if not isinstance(other, FermionOperator_):
            return FermionOperator(FermionOperator_.__add__(self, FermionOperator("", ParameterResolver(other))))
        return FermionOperator(FermionOperator_.__add__(self, other))

    def __iadd__(self, other):
        if not isinstance(other, FermionOperator_):
            FermionOperator_.__iadd__(self, FermionOperator("", ParameterResolver(other)))
            return self
        FermionOperator_.__iadd__(self, other)
        return self

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if not isinstance(other, FermionOperator_):
            return FermionOperator(FermionOperator_.__mul__(self, FermionOperator("", ParameterResolver(other))))
        return FermionOperator(FermionOperator_.__mul__(self, other))

    def __imul__(self, other):
        if not isinstance(other, FermionOperator_):
            FermionOperator_.__imul__(self, FermionOperator("", ParameterResolver(other)))
            return self
        FermionOperator_.__imul__(self, other)
        return self

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1.0 / other)

    def __itruediv__(self, other):
        self.__imul__(1.0 / other)
        return self

    def astype(self, dtype):
        return FermionOperator(FermionOperator_.astype(self, dtype))

    @property
    def terms(self):
        origin_dict = FermionOperator_.get_terms(self)
        out = {}
        for key, value in origin_dict:
            out_key = []
            for idx, t in key:
                out_key.append((idx, 0 if t == f_term_value.a else 1))
            out[tuple(out_key)] = ParameterResolver(value, internal=True)
        return out

    @property
    def constant(self):
        return ParameterResolver(FermionOperator_.get_coeff(self, []), internal=True)

    @constant.setter
    def constant(self, value):
        FermionOperator_.set_coeff(self, [], ParameterResolver(value))

    def count_qubits(self):
        return FermionOperator_.count_qubits(self)

    def get_coeff(self, term):
        return ParameterResolver(FermionOperator_.get_coeff(self, [(i, TermValue[j]) for i, j in term]), internal=True)


if __name__ == "__main__":
    a = FermionOperator("1", 1.3)
    b = FermionOperator("1 0", 1.3)
    c = a + b
    d = c + FermionOperator("0 1^", 3j) + 3
