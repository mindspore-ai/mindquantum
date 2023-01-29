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

from . import fixed_decompose
from . import universal_decompose
from . import utils

from .fixed_decompose import (
    ch_decompose, crx_decompose, cry_decompose, crz_decompose,
    swap_decompose, cswap_decompose, cs_decompose, ct_decompose,
    ccx_decompose, cxx_decompose, xx_decompose, cyy_decompose, yy_decompose,
    cy_decompose, cz_decompose, zz_decompose
)
from .universal_decompose import euler_decompose
from .universal_decompose import tensor_product_decompose, abc_decompose, kak_decompose
from .universal_decompose import qs_decompose, cu_decompose, demultiplex_pair, demultiplex_pauli

__all__ = []
__all__.extend(fixed_decompose.__all__)
__all__.extend(universal_decompose.__all__)
__all__.sort()
