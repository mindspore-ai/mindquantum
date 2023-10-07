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
Noisy Intermediate Scale Quantum (NISQ) algorithms.

In NISQ, the quantum qubits number and quantum circuit depth are very limited
and the quantum gate fidelity is also limited.
"""

from . import chem, qaoa, qnn
from ._ansatz import Ansatz
from .barren_plateau import ansatz_variance
from .chem import *
from .qaoa import *
from .qnn import *

__all__ = ['Ansatz', 'ansatz_variance']
__all__.extend(chem.__all__)
__all__.extend(qaoa.__all__)
__all__.extend(qnn.__all__)
__all__.sort()
