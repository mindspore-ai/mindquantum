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
""".. MindQuantum package."""

import os
import warnings
import sys

from . import core
from .core import gates
from .core import operators
from . import engine
from . import framework
from . import utils
from . import algorithm
from . import simulator
from . import io
from .core import *
from .algorithm import *
from .utils import *
from .simulator import *
from .framework import *
from .io import *


if sys.version_info < (3, 8):  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError
else:  # pragma: no cover
    from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mindquantum")
    __version_info__ = tuple(__version__.split('.'))
    __all__ = ['__version__', '__version_info__']
except PackageNotFoundError:
    __all__ = []


__all__.extend(core.__all__)
__all__.extend(algorithm.__all__)
__all__.extend(utils.__all__)
__all__.extend(simulator.__all__)
__all__.extend(framework.__all__)
__all__.extend(io.__all__)
__all__.sort()
