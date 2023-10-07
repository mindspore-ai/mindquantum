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

"""Decompose rule for fixed gates."""

from . import (
    h_related,
    rx_related,
    rxx_related,
    ry_related,
    ryy_related,
    rz_related,
    rzz_related,
    s_related,
    swap_related,
    t_related,
    x_related,
    y_related,
    z_related,
)
from .h_related import ch_decompose
from .rx_related import crx_decompose
from .rxx_related import crxx_decompose, rxx_decompose
from .ry_related import cry_decompose
from .ryy_related import cryy_decompose, ryy_decompose
from .rz_related import crz_decompose
from .rzz_related import rzz_decompose
from .s_related import cs_decompose
from .swap_related import cswap_decompose, swap_decompose
from .t_related import ct_decompose
from .x_related import ccx_decompose
from .y_related import cy_decompose
from .z_related import cz_decompose

__all__ = []
__all__.extend(h_related.__all__)
__all__.extend(rx_related.__all__)
__all__.extend(rxx_related.__all__)
__all__.extend(ry_related.__all__)
__all__.extend(ryy_related.__all__)
__all__.extend(rz_related.__all__)
__all__.extend(rzz_related.__all__)
__all__.extend(s_related.__all__)
__all__.extend(swap_related.__all__)
__all__.extend(t_related.__all__)
__all__.extend(x_related.__all__)
__all__.extend(y_related.__all__)
__all__.extend(z_related.__all__)
__all__.sort()
