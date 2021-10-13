# -*- coding: utf-8 -*-
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
"""
Circuit.

Quantum circuit module.
"""

from .circuit import Circuit
from .module_circuit import UN, SwapParts, U3
from .utils import decompose_single_term_time_evolution
from .utils import pauli_word_to_circuits
from .utils import controlled
from .utils import dagger
from .utils import apply
from .utils import add_prefix
from .utils import change_param_name
from .utils import C
from .utils import A
from .utils import D
from .utils import AP
from .utils import CPN

__all__ = [
    'Circuit', 'U3', 'UN', 'SwapParts', 'C', 'A', 'D', 'AP', 'CPN',
    'decompose_single_term_time_evolution', 'pauli_word_to_circuits',
    'controlled', 'dagger', 'apply', 'add_prefix', 'change_param_name'
]
__all__.sort()
