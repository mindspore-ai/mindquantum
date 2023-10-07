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
"""Algorithm for IQP Encoding."""

from . import arxiv_1905_10876
from .arxiv_1905_10876 import (
    Ansatz1,
    Ansatz2,
    Ansatz3,
    Ansatz4,
    Ansatz5,
    Ansatz6,
    Ansatz7,
    Ansatz8,
    Ansatz9,
    Ansatz10,
    Ansatz11,
    Ansatz12,
    Ansatz13,
    Ansatz14,
    Ansatz15,
    Ansatz16,
    Ansatz17,
    Ansatz18,
    Ansatz19,
)
from .iqp_encoding import IQPEncoding
from .strongly_entangling import StronglyEntangling

__all__ = ['IQPEncoding', 'StronglyEntangling']
__all__.extend(arxiv_1905_10876.__all__)
__all__.sort()
