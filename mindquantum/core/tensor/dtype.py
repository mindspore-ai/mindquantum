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

from mindquantum._math import dtype

__dtype__ = [
    'float32',
    'float64',
    'complex64',
    'complex128',
]

float32 = dtype.float32
float64 = dtype.float64
complex64 = dtype.complex64
complex128 = dtype.complex128

str_dtype_map = {
    str(float32): float32,
    str(float64): float64,
    str(complex64): complex64,
    str(complex128): complex128,
}

__all__ = []
__all__.extend(__dtype__)
__all__.extend(['str_dtype_map'])
