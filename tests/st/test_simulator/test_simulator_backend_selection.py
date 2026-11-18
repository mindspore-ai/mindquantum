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
"""Backend discovery and selection of ``mindquantum.simulator``.

0.13 turns the optional backends (``mqmps``, ``mqvector_npu``,
``mqvector_gpu``, ``mqvector_cq``) into build-time decisions that are discovered
without importing the extension.  A missing backend must be reported at
selection time, not at import time.
"""

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.simulator import Simulator, get_supported_simulator
from mindquantum.simulator.available_simulator import MQVECTOR_GPU_SUPPORTED, SUPPORTED_SIMULATOR
from mindquantum.simulator.mqsim import MQSim
from mindquantum.simulator.stabilizer import Stabilizer
from mindquantum.utils.error import SimNotAvailableError

pytestmark = pytest.mark.level0


def test_supported_simulator_listing():
    """
    Feature: Listing of the compiled simulators.

    Description: Enumerate the simulators of this build.

    Expectation: The three CPU backends are listed for both precisions and the stabilizer backend is listed once.
    """
    entries = get_supported_simulator()
    pairs = {tuple(entry) for entry in entries if isinstance(entry, list)}
    assert ('mqvector', mq.complex64) in pairs
    assert ('mqvector', mq.complex128) in pairs
    assert ('mqmatrix', mq.complex64) in pairs
    assert ('mqmatrix', mq.complex128) in pairs
    assert 'stabilizer' in entries


def test_backend_lookup():
    """
    Feature: Python class and C++ module lookup of a backend.

    Description: Resolve the state vector, density matrix and stabilizer backends by name.

    Expectation: The returned classes match the implementation selected by the public simulator.
    """
    assert SUPPORTED_SIMULATOR.py_class('mqvector') is MQSim
    assert SUPPORTED_SIMULATOR.py_class('mqmatrix') is MQSim
    assert SUPPORTED_SIMULATOR.py_class('stabilizer') is Stabilizer
    assert Simulator('mqvector', 1).backend.name == 'mqvector'


def test_available_predicate():
    """
    Feature: Availability predicate of the backend registry.

    Description: Query the registry with a backend object and with unsupported names or precisions.

    Expectation: Only configured backends are reported as available.
    """
    backend = Simulator('mqvector', 1).backend
    assert SUPPORTED_SIMULATOR.is_available(backend, mq.complex128)
    assert SUPPORTED_SIMULATOR.is_available('stabilizer', mq.complex128)
    assert SUPPORTED_SIMULATOR.is_available('mqvector', mq.complex128)
    assert not SUPPORTED_SIMULATOR.is_available('mqvector', np.float64)
    assert not SUPPORTED_SIMULATOR.is_available('not-a-backend', mq.complex128)


def test_unknown_backend_is_rejected():
    """
    Feature: Selection of an unknown backend.

    Description: Build a simulator with an unknown name and query the registry directly.

    Expectation: ``SimNotAvailableError`` is raised instead of a bare ``KeyError``.
    """
    with pytest.raises(SimNotAvailableError):
        Simulator('not-a-backend', 2)
    with pytest.raises(SimNotAvailableError):
        SUPPORTED_SIMULATOR.c_module('not-a-backend')
    with pytest.raises(SimNotAvailableError):
        SUPPORTED_SIMULATOR.py_class('not-a-backend')
    with pytest.raises(SimNotAvailableError):
        SUPPORTED_SIMULATOR.c_module('mqvector', np.float64)


@pytest.mark.skipif(MQVECTOR_GPU_SUPPORTED, reason='A GPU backend is available in this build')
def test_gpu_backend_reports_the_build_requirement():
    """
    Feature: Diagnostic of an unavailable GPU backend.

    Description: Request the GPU backend on a build where it cannot be used, either because the
        extension is missing or because the CUDA driver is insufficient.

    Expectation: The error explains why the GPU backend is not available.
    """
    with pytest.raises(ValueError, match='GPU backend is not available'):
        Simulator('mqvector_gpu', 2)
    with pytest.raises(ValueError, match='GPU backend is not available'):
        SUPPORTED_SIMULATOR.py_class('mqvector_gpu')
    with pytest.warns(UserWarning, match='GPU backend is not available'):
        with pytest.raises(SimNotAvailableError):
            SUPPORTED_SIMULATOR.c_module('mqvector_gpu')


def test_stabilizer_backend_accepts_any_precision():
    """
    Feature: Precision handling of the stabilizer backend.

    Description: Build a stabilizer simulator with both complex precisions.

    Expectation: The backend is created and applies a Clifford circuit.
    """
    for dtype in (mq.complex64, mq.complex128):
        simulator = Simulator('stabilizer', 2, dtype=dtype)
        simulator.apply_circuit(mq.Circuit().h(0).x(1, 0))
        assert np.isfinite(simulator.get_qs()).all()
