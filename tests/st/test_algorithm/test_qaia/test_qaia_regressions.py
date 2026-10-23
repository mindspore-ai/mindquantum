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
"""Regression tests for QAIA input conversion and coupling normalization."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mindquantum.algorithm.qaia import ASB, BSB, CAC, CFC, DSB, GBSB, QAIA, SFC

pytestmark = pytest.mark.level0

# Matrix powers and element-wise powers have different sums for this graph.
_COUPLING = np.array([[0.0, 1.0, -2.0], [1.0, 0.0, 3.0], [-2.0, 3.0, 0.0]])


@pytest.fixture(
    name='backend',
    params=[
        pytest.param('cpu-float32', marks=pytest.mark.platform_x86_cpu),
        pytest.param('gpu-float32', marks=[pytest.mark.platform_x86_gpu_training, pytest.mark.env_onecard]),
        pytest.param(
            'npu-float32',
            marks=[
                pytest.mark.platform_arm_ascend_training,
                pytest.mark.platform_arm_ascend910b_training,
                pytest.mark.env_onecard,
            ],
        ),
    ],
)
def backend_fixture(request):
    """Select an available backend without requiring Torch for CPU tests."""
    if request.param != 'cpu-float32':
        torch = pytest.importorskip('torch')
        if request.param == 'gpu-float32':
            if not torch.cuda.is_available():
                pytest.skip('CUDA is not available.')
        else:
            torch_npu = pytest.importorskip('torch_npu')
            if not torch_npu.npu.is_available():
                pytest.skip('Ascend NPU is not available.')
    return request.param


@pytest.fixture(name='coupling', params=['dense', 'sparse'])
def coupling_fixture(request):
    """Exercise both supported coupling matrix representations."""
    if request.param == 'sparse':
        return csr_matrix(_COUPLING)
    return _COUPLING.copy()


@pytest.fixture(name='external_field', params=['none', 'vector', 'column'])
def external_field_fixture(request):
    """Exercise absent and both supported external-field shapes."""
    if request.param == 'none':
        return None
    field = np.array([0.5, -1.0, 1.5])
    if request.param == 'column':
        return field[:, np.newaxis]
    return field


def _as_numpy(value):
    """Copy device results to the host before comparing with NumPy."""
    if isinstance(value, np.ndarray):
        return value
    return value.cpu().numpy()


@pytest.mark.parametrize('solver_class', [CAC, CFC, SFC])
def test_cim_coupling_strength(solver_class, backend, coupling):
    """
    Feature: QAIA coupling-strength normalization.
    Description: Initialize CAC/CFC/SFC with dense and sparse matrices on each available backend.
    Expectation: xi equals the normalization computed from element-wise squares.
    """
    solver = solver_class(coupling, n_iter=2, backend=backend)
    expected = np.sqrt(2 * _COUPLING.shape[0] / np.square(_COUPLING).sum())
    assert float(solver.xi) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('solver_class', [ASB, BSB, GBSB, DSB])
def test_sb_coupling_strength(solver_class, backend, coupling, external_field):
    """
    Feature: SB external-field normalization.
    Description: Initialize SB variants with no field, a vector field, or a column field on each available backend.
    Expectation: Default xi matches the NumPy reference without implicit device-to-NumPy conversion.
    """
    solver = solver_class(coupling, h=external_field, n_iter=2, backend=backend)
    norm_squared = np.square(_COUPLING).sum()
    if external_field is not None:
        norm_squared += 2 * np.square(external_field / 2).sum()
    expected = 0.5 * np.sqrt(_COUPLING.shape[0] - 1) / np.sqrt(norm_squared)
    assert float(solver.xi) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize('input_kind', ['float_array', 'int_array', 'cpu_tensor', 'device_tensor'])
def test_qaia_observable_inputs(backend, coupling, external_field, input_kind):
    """
    Feature: QAIA cut and energy input conversion.
    Description: Evaluate stored spins and explicit NumPy/CPU/device Tensor inputs on each available backend.
    Expectation: Results match NumPy and tensors use the coupling matrix's device and dtype,
        without modifying the explicit input or stored state.
    """
    if backend == 'cpu-float32' and input_kind.endswith('tensor'):
        pytest.skip('Tensor inputs are only supported by Torch backends.')

    initial_x = np.array([[0.2, 0.4], [-0.3, 0.1], [0.6, -0.2]])
    solver = QAIA(coupling, h=external_field, x=initial_x.copy(), batch_size=2, backend=backend)
    solver.initialize()
    spins = np.array([[1, -1], [0, 1], [-1, -1]])
    if input_kind == 'float_array':
        explicit_x = spins.astype(np.float64)
    elif input_kind == 'int_array':
        explicit_x = spins.astype(np.int64)
    else:
        torch = pytest.importorskip('torch')
        if input_kind == 'cpu_tensor':
            explicit_x = torch.tensor(spins, dtype=torch.float64)
        else:
            explicit_x = torch.tensor(spins, dtype=solver.J.dtype, device=solver.J.device)

    for x, sign in [(None, np.sign(initial_x)), (explicit_x, spins)]:
        interaction = np.sum((_COUPLING @ sign) * sign, axis=0)
        expected_cut = 0.25 * interaction - 0.25 * _COUPLING.sum()
        expected_energy = -0.5 * interaction
        if external_field is not None:
            expected_energy = expected_energy[np.newaxis, :] - external_field.reshape(1, -1) @ sign

        cut = solver.calc_cut(x)
        energy = solver.calc_energy(x)
        np.testing.assert_allclose(_as_numpy(cut), expected_cut, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(_as_numpy(energy), expected_energy, rtol=1e-6, atol=1e-6)
        if backend != 'cpu-float32':
            assert cut.device == energy.device == solver.J.device
            assert cut.dtype == energy.dtype == solver.J.dtype

    np.testing.assert_allclose(_as_numpy(solver.x), initial_x, rtol=1e-6)
    np.testing.assert_array_equal(_as_numpy(explicit_x), spins)
