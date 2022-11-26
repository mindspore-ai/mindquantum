# Copyright 2022 Huawei Technologies Co., Ltd
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
"""MindQuantum acculated CPU/GPU simulator."""
import warnings

import mindquantum as mq

from . import _mq_densitymatrix as mqmatrix
from . import simulator

__all__ = ['mqmatrix', 'simulator']
SUPPORTED_BACKEND = ['cpu']
try:
    from . import _mq_sim_gpu as mq_sim_gpu

    __all__.append('mq_sim_gpu')
    SUPPORTED_BACKEND.append('gpu')
except:
    warnings.warn("GPU simulator not built.", stacklevel=2)


__all__.sort()
__version__ = '0.1.0'
