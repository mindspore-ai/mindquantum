# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Public backend discovery and selection errors."""

import pytest

import mindquantum as mq
from mindquantum.simulator import Simulator, get_supported_simulator
from mindquantum.simulator.available_simulator import MQVECTOR_GPU_SUPPORTED
from mindquantum.utils.error import SimNotAvailableError

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


def test_supported_cpu_backends():
    """
    Feature: Backend discovery.
    Description: Query the public list of supported simulators.
    Expectation: Both precisions of the vector and matrix backends and stabilizer are advertised.
    """
    entries = get_supported_simulator()
    for backend in ('mqvector', 'mqmatrix'):
        for dtype in (mq.complex64, mq.complex128):
            assert [backend, dtype] in entries
    assert 'stabilizer' in entries


def test_unknown_backend():
    """
    Feature: Backend selection.
    Description: Construct a simulator with an unknown backend.
    Expectation: The public availability error is raised rather than a registry KeyError.
    """
    with pytest.raises(SimNotAvailableError):
        Simulator('not-a-backend', 2)


@pytest.mark.skipif(MQVECTOR_GPU_SUPPORTED, reason='GPU backend is available')
def test_missing_gpu_backend():
    """
    Feature: Optional GPU backend.
    Description: Request a GPU simulator from a build without a usable GPU backend.
    Expectation: Selection explains the GPU requirement.
    """
    with pytest.raises(ValueError, match='GPU backend is not available'):
        Simulator('mqvector_gpu', 2)
