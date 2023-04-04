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

from mindquantum._math import pr as ParameterResolver_
from mindquantum._math.tensor import Tensor as Tensor_
from mindquantum.core.tensor import dtype as mqtype


class ParameterResolver(ParameterResolver_):
    """
    ParameterResolver('a'[, dtype=mq.float32])
    ParameterResolver(1.0[, dtype=mq.float32])
    ParameterResolver({'a': 2}[, const=1.2, dtype=mq.float32])
    """

    def __init__(self, data=None, const=None, dtype=None, internal=False):
        if internal:
            if dtype != None:
                ParameterResolver_.__init__(data, dtype)
            else:
                ParameterResolver_.__init__(data)
        else:
            if isinstance(data, str):
                if dtype is None:
                    dtype = mqtype.float64
                if const is None:
                    const = 0.0
                ParameterResolver_.__init__(data, const, dtype)  # PR('a'[, 1.0, mq.float64])
            elif isinstance(data, dict):
                if dtype is None:
                    dtype = mqtype.float64
                    for v in data.values():
                        if isinstance(v, numbers.Number) and not isinstance(v, numbers.Real):
                            dtype = mqtype.complex128
                            break
                    if const is not None:
                        if isinstance(v, numbers.Number) and not isinstance(v, numbers.Real):
                            dtype = mqtype.complex128
                if const is None:
                    const = 0.0
                # PR({'a': 1.0}[, 2.0, mq.float64])
                ParameterResolver_.__init__({i: j for i, j in data.items()}, const, dtype)
            elif isinstance(data, numbers.Number):
                if dtype is None:
                    dtype = mqtype.float64
                    if isinstance(v, numbers.Number) and not isinstance(v, numbers.Real):
                        dtype = mqtype.complex128
                ParameterResolver_.__init__(data, dtype)  # PR(1.0[, mq.float64])

    def __str__(self) -> str:
        return ParameterResolver_.__str__()
