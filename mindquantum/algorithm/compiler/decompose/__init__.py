# Copyright 2021 Huawei Technologies Co., Ltd
#
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

"""Decompose rule for gate."""

from . import fixed_decompose, universal_decompose, utils
from .fixed_decompose import (
    ccx_decompose,
    ch_decompose,
    crx_decompose,
    crxx_decompose,
    cry_decompose,
    cryy_decompose,
    crz_decompose,
    cs_decompose,
    cswap_decompose,
    ct_decompose,
    cy_decompose,
    cz_decompose,
    rxx_decompose,
    ryy_decompose,
    rzz_decompose,
    swap_decompose,
)
from .universal_decompose import (
    abc_decompose,
    cu_decompose,
    demultiplex_pair,
    demultiplex_pauli,
    euler_decompose,
    kak_decompose,
    qs_decompose,
    tensor_product_decompose,
)

__all__ = []
__all__.extend(fixed_decompose.__all__)
__all__.extend(universal_decompose.__all__)
__all__.sort()
