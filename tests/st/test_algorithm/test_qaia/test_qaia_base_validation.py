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
"""QAIA constructor validation shared by the NumPy solvers."""

import importlib.util

import numpy as np
import pytest

from mindquantum.algorithm.qaia import BSB, QAIA
from mindquantum.algorithm.qaia.SB import GPU_AVAILABLE

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]
_COUPLING = np.array([[0.0, -1.0], [-1.0, 0.0]])


@pytest.mark.parametrize(
    'kwargs,error,match',
    [
        pytest.param({'backend': 'tpu-float32'}, ValueError, 'backend must be one of', id='backend-name'),
        pytest.param({'backend': 1}, TypeError, 'requires a string', id='backend-type'),
        pytest.param({'h': [0.5, -0.5]}, TypeError, 'h requires numpy.array', id='field-type'),
        pytest.param({'h': np.zeros(3)}, ValueError, 'h must have shape', id='field-length'),
        pytest.param({'h': np.zeros((2, 2))}, ValueError, 'h must have shape', id='field-columns'),
        pytest.param({'x': [[0.1], [0.2]]}, TypeError, 'x requires numpy.array', id='spins-type'),
        pytest.param({'x': np.zeros(2)}, ValueError, 'x must be a 2D array', id='spins-rank'),
        pytest.param({'x': np.zeros((3, 1))}, ValueError, 'x must have shape', id='spins-count'),
        pytest.param({'x': np.zeros((2, 2))}, ValueError, 'x must have shape', id='replica-count'),
        pytest.param({'n_iter': 0}, ValueError, 'not less than 1', id='iterations-zero'),
        pytest.param({'n_iter': 1.5}, TypeError, 'requires an int', id='iterations-float'),
        pytest.param({'batch_size': 0}, ValueError, 'not less than 1', id='batch-zero'),
        pytest.param({'batch_size': True}, TypeError, 'requires an int', id='batch-bool'),
        pytest.param({'J': np.zeros((2, 3))}, ValueError, 'square', id='matrix-shape'),
        pytest.param({'J': [[0, -1], [-1, 0]]}, TypeError, 'J requires', id='matrix-type'),
        pytest.param({'J': np.eye(2)}, ValueError, 'diagonal', id='matrix-diagonal'),
        pytest.param({'J': np.array([[0, -1], [1, 0]])}, ValueError, 'symmetric', id='matrix-symmetry'),
    ],
)
def test_qaia_constructor_validation(kwargs, error, match):
    """
    Feature: QAIA input validation.
    Description: Supply invalid problem dimensions, types, counters or backend selection.
    Expectation: Construction rejects each malformed problem explicitly.
    """
    with pytest.raises(error, match=match):
        QAIA(**({'J': _COUPLING} | kwargs))


@pytest.mark.skipif(importlib.util.find_spec('torch') is not None, reason='PyTorch is installed')
@pytest.mark.parametrize(
    'backend,match',
    [('gpu-float32', 'Please install pytorch'), ('npu-float32', 'Please install torch_npu')],
)
def test_missing_torch(backend, match):
    """
    Feature: Optional Torch dependency.
    Description: Select a Torch backend on an installation without PyTorch.
    Expectation: ImportError identifies the missing package.
    """
    with pytest.raises(ImportError, match=match):
        QAIA(_COUPLING, backend=backend)


@pytest.mark.parametrize(
    'kwargs,error,match',
    [
        pytest.param({'device_id': 1}, ValueError, 'only supported by Ascend', id='cpu-device-id'),
        pytest.param(
            {'backend': 'gpu-float16'},
            RuntimeError,
            'not available',
            marks=pytest.mark.skipif(GPU_AVAILABLE, reason='GPU SB extension is available'),
            id='missing-gpu-float16',
        ),
        pytest.param(
            {'backend': 'gpu-int8'},
            RuntimeError,
            'not available',
            marks=pytest.mark.skipif(GPU_AVAILABLE, reason='GPU SB extension is available'),
            id='missing-gpu-int8',
        ),
    ],
)
def test_bsb_backend_validation(kwargs, error, match):
    """
    Feature: Bounded SB backend validation.
    Description: Request a CPU device id or an unavailable quantized GPU backend.
    Expectation: Construction rejects the unsupported backend configuration explicitly.
    """
    with pytest.raises(error, match=match):
        BSB(_COUPLING, n_iter=2, **kwargs)


@pytest.mark.parametrize('method', ['calc_cut', 'calc_energy'])
def test_qaia_unknown_backend_dispatch(method):
    """
    Feature: QAIA observable backend dispatch.
    Description: Evaluate an observable after replacing the backend with an unknown name.
    Expectation: An invalid backend raises ValueError rather than silently returning None.
    """
    solver = QAIA(_COUPLING)
    solver.backend = 'cpu-unknown'
    with pytest.raises(ValueError, match='invalid backend'):
        getattr(solver, method)()
