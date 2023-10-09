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
import sys
import warnings

import numpy as np

from .mqbackend import logging

# isort: split


from . import (
    algorithm,
    config,
    core,
    device,
    dtype,
    engine,
    framework,
    io,
    simulator,
    utils,
)
from .algorithm import *
from .config import *
from .core import *
from .core import gates, operators
from .device import *
from .dtype import *
from .framework import *
from .io import *
from .simulator import *
from .utils import *

if sys.version_info < (3, 8):  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version
else:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mindquantum")
    __version_info__ = tuple(__version__.split('.'))
    __all__ = ['__version__', '__version_info__']
except (PackageNotFoundError, AttributeError):
    __all__ = []

__all__.extend(core.__all__)
__all__.extend(algorithm.__all__)
__all__.extend(utils.__all__)
__all__.extend(simulator.__all__)
__all__.extend(framework.__all__)
__all__.extend(io.__all__)
__all__.extend(config.__all__)
__all__.extend(device.__all__)
__all__.extend(dtype.__all__)
__all__.sort()
