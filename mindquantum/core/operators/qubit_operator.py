#   Copyright 2022 <Huawei Technologies Co., Ltd>
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""This is the module for the Qubit Operator."""

import os
import warnings

try:
    if int(os.environ.get('MQ_PY_TERMSOP', False)):
        warnings.warn("Using Python QubitOperator class")
        raise ImportError()

    from ...experimental.ops import TermValue
    from ...experimental.ops.qubit_operator import QubitOperator
except ImportError:
    TermValue = {k: k for k in ('X', 'Y', 'Z', 'I')}
    from ._qubit_operator import QubitOperator  # noqa: F401
