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
"""Quantum neural networks operators and cells."""
import warnings

__all__ = []
try:
    import mindspore

    from .layer import (
        MQAnsatzOnlyLayer,
        MQLayer,
        MQN2AnsatzOnlyLayer,
        MQN2Layer,
        QRamVecLayer,
    )
    from .operations import (
        MQAnsatzOnlyOps,
        MQEncoderOnlyOps,
        MQN2AnsatzOnlyOps,
        MQN2EncoderOnlyOps,
        MQN2Ops,
        MQOps,
        QRamVecOps,
    )

    __all__.extend(
        [
            "MQAnsatzOnlyLayer",
            "MQN2AnsatzOnlyLayer",
            "MQLayer",
            "MQN2Layer",
            "MQOps",
            "MQN2Ops",
            "MQAnsatzOnlyOps",
            "MQN2AnsatzOnlyOps",
            "MQEncoderOnlyOps",
            "MQN2EncoderOnlyOps",
            "QRamVecOps",
            "QRamVecLayer",
        ]
    )
    import packaging.version

    ms_version = mindspore.__version__
    if "rc" in ms_version:
        ms_version = ms_version[: ms_version.find('rc')]
    ms_requires = packaging.version.parse('1.4.0')
    if packaging.version.parse(ms_version) < ms_requires:
        warnings.warn(
            "Current version of MindSpore is not compatible with MindSpore Quantum. "
            "Some functions might not work or even raise error. Please install MindSpore "
            "version >= 1.4.0. For more details about dependency setting, please check "
            "the instructions at MindSpore official website https://www.mindspore.cn/install "
            "or check the README.md at https://gitee.com/mindspore/mindquantum",
            stacklevel=2,
        )

except ImportError:
    warnings.warn(
        "MindSpore not installed, you may not be able to use hybrid quantum classical neural network.",
        stacklevel=2,
    )

__all__.sort()
