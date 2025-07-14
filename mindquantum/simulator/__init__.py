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
"""Quantum simulator that simulate evolution of quantum system."""

from .available_simulator import SUPPORTED_SIMULATOR
from .noise import NoiseBackend
from .simulator import Simulator, fidelity, get_supported_simulator, inner_product
from .stabilizer import decompose_stabilizer, get_stabilizer_string, get_tableau_string
from . import mqchem
from .utils import GradOpsWrapper

__all__ = [
    'Simulator',
    'GradOpsWrapper',
    'get_supported_simulator',
    'inner_product',
    'SUPPORTED_SIMULATOR',
    'NoiseBackend',
    'fidelity',
    'get_stabilizer_string',
    'get_tableau_string',
    'decompose_stabilizer',
    'mqchem',
]
__all__.sort()
