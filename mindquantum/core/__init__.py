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

"""MindQuantum core features (eDSL)."""


# Provide alias for convenience
from . import circuit, gates, operators, parameterresolver, third_party
from .circuit import *  # noqa: F401,F403
from .gates import *  # noqa: F401,F403
from .operators import *  # noqa: F401,F403
from .parameterresolver import *  # noqa: F401,F403
from .third_party import *  # noqa: F401,F403

__all__ = []
__all__.extend(circuit.__all__)
__all__.extend(gates.__all__)
__all__.extend(operators.__all__)
__all__.extend(parameterresolver.__all__)
__all__.extend(third_party.__all__)
__all__.sort()
