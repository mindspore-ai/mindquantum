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
"""NISQ algorithms"""

from ._ansatz import Ansatz
from . import chem
from . import qaoa
from .chem import *
from .qaoa import *

__all__ = ['Ansatz']
__all__.extend(chem.__all__)
__all__.extend(qaoa.__all__)
__all__.sort()
