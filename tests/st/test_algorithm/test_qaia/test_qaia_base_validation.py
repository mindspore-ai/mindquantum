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
"""Validation contract of the QAIA base class.

Every solver of ``mindquantum.algorithm.qaia`` inherits the constructor of
``QAIA``, so the checks below guard the whole family: coupling matrix, external
field, initial spin state, backend name and iteration counters.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mindquantum.algorithm.qaia import BSB, CAC, QAIA

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]

_COUPLING = np.array([[0.0, -1.0], [-1.0, 0.0]])


@pytest.mark.parametrize('backend', ['gpu-float32', 'npu-float32'])
def test_unavailable_device_backend_reports_the_required_package(backend):
    """
    Feature: Backend availability check of ``QAIA``.

    Description: Ask for a Torch based backend on a NumPy only installation.

    Expectation: An ``ImportError`` explains which package is required.
    """
    pytest.importorskip('scipy')
    try:
        import torch  # noqa: F401  pylint: disable=unused-import
    except ImportError:
        pass
    else:
        pytest.skip('PyTorch is installed, the import guard is not exercised.')

    with pytest.raises(ImportError, match='Please install pytorch|Please install torch_npu'):
        QAIA(_COUPLING, backend=backend)


@pytest.mark.parametrize('value', ['tpu-float32', 'cpu', ''])
def test_invalid_backend_name_is_rejected(value):
    """
    Feature: Backend name validation of ``QAIA``.

    Description: Provide an unsupported backend name.

    Expectation: A ``ValueError`` lists the supported backends.
    """
    with pytest.raises(ValueError, match='backend must be one of'):
        QAIA(_COUPLING, backend=value)


def test_non_string_backend_is_rejected():
    """
    Feature: Backend type validation of ``QAIA``.

    Description: Provide a non string backend.

    Expectation: A ``TypeError`` reports the expected type.
    """
    with pytest.raises(TypeError, match='backend requires a string'):
        QAIA(_COUPLING, backend=1)


def test_external_field_type_and_shape_validation():
    """
    Feature: External field validation of ``QAIA``.

    Description: Provide a list, a wrong length vector and a 2D field with two columns.

    Expectation: Each malformed field raises the documented exception.
    """
    with pytest.raises(TypeError, match='h requires numpy.array'):
        QAIA(_COUPLING, h=[0.5, -0.5])
    with pytest.raises(ValueError, match='h must have shape'):
        QAIA(_COUPLING, h=np.zeros(3))
    with pytest.raises(ValueError, match='h must have shape'):
        QAIA(_COUPLING, h=np.zeros((2, 2)))


def test_initial_state_type_and_shape_validation():
    """
    Feature: Initial spin validation of ``QAIA``.

    Description: Provide a list, a 1D array and a 2D array with the wrong number of replicas.

    Expectation: Each malformed state raises the documented exception.
    """
    with pytest.raises(TypeError, match='x requires numpy.array'):
        QAIA(_COUPLING, x=[[0.1], [0.2]])
    with pytest.raises(ValueError, match='x must be a 2D array'):
        QAIA(_COUPLING, x=np.zeros(2))
    with pytest.raises(ValueError, match='x must have shape'):
        QAIA(_COUPLING, x=np.zeros((3, 1)))
    with pytest.raises(ValueError, match='x must have shape'):
        QAIA(_COUPLING, x=np.zeros((2, 2)))


@pytest.mark.parametrize(
    'kwargs,match',
    [
        pytest.param({'n_iter': 0}, 'should be not less than 1', id='n_iter-zero'),
        pytest.param({'n_iter': 1.5}, 'requires an int', id='n_iter-float'),
        pytest.param({'batch_size': 0}, 'should be not less than 1', id='batch-zero'),
        pytest.param({'batch_size': True}, 'requires an int', id='batch-bool'),
    ],
)
def test_iteration_counter_validation(kwargs, match):
    """
    Feature: Iteration counter validation of ``QAIA``.

    Description: Provide a non positive or non integer iteration counter.

    Expectation: The documented exception is raised.
    """
    with pytest.raises((TypeError, ValueError), match=match):
        QAIA(_COUPLING, **kwargs)


@pytest.mark.parametrize(
    'coupling,match,error',
    [
        pytest.param(np.zeros((2, 3)), 'must be a square matrix', ValueError, id='non-square'),
        pytest.param([[0.0, -1.0], [-1.0, 0.0]], 'requires numpy.array or scipy sparse matrix', TypeError, id='list'),
        pytest.param(np.diag([1.0, 1.0]), 'diagonal elements of J are not all 0', ValueError, id='diag'),
        pytest.param(np.array([[0.0, -1.0], [1.0, 0.0]]), 'must be a symmetric matrix', ValueError, id='asymmetric'),
    ],
)
def test_coupling_matrix_validation(coupling, match, error):
    """
    Feature: Coupling matrix validation of ``QAIA``.

    Description: Provide a non square, non array, asymmetric coupling matrix and a non zero diagonal.

    Expectation: The documented exception is raised.
    """
    with pytest.raises(error, match=match):
        QAIA(coupling)


def test_sparse_coupling_matrix_is_accepted():
    """
    Feature: Sparse coupling matrix support of ``QAIA``.

    Description: Build a solver from a CSR matrix.

    Expectation: The matrix and its shape are preserved.
    """
    solver = QAIA(csr_matrix(_COUPLING))
    assert solver.N == 2
    assert solver.J.shape == (2, 2)


def test_cut_and_energy_reject_an_unknown_backend():
    """
    Feature: Backend dispatch of ``QAIA.calc_cut`` and ``QAIA.calc_energy``.

    Description: Corrupt the backend attribute after construction.

    Expectation: A ``ValueError`` reports the invalid backend instead of returning a wrong value.
    """
    solver = QAIA(_COUPLING)
    solver.backend = 'cpu-unknown'
    with pytest.raises(ValueError, match='invalid backend'):
        solver.calc_cut()
    with pytest.raises(ValueError, match='invalid backend'):
        solver.calc_energy()


def test_inherited_validation_applies_to_subclasses():
    """
    Feature: Inheritance of the ``QAIA`` validation.

    Description: Build two subclasses with a non positive batch size and an asymmetric matrix.

    Expectation: The same exceptions are raised as for the base class.
    """
    with pytest.raises(ValueError, match='should be not less than 1'):
        BSB(_COUPLING, batch_size=0)
    with pytest.raises(ValueError, match='must be a symmetric matrix'):
        CAC(np.array([[0.0, -1.0], [1.0, 0.0]]))


def test_device_id_is_rejected_by_cpu_backends():
    """
    Feature: Device selection of the bounded SB backends.

    Description: Pass a non zero ``device_id`` to a CPU backend.

    Expectation: A ``ValueError`` explains that only the Ascend CANN backend accepts a device id.
    """
    with pytest.raises(ValueError, match='only supported by Ascend CANN backends'):
        BSB(_COUPLING, n_iter=2, device_id=1)


@pytest.mark.parametrize('backend', ['gpu-float16', 'gpu-int8'])
def test_quantized_gpu_backend_reports_missing_extension(backend):
    """
    Feature: Quantized GPU backends of ``BSB``.

    Description: Request a GPU quantized backend on a build without the ``_qaia_sb`` extension.

    Expectation: A ``RuntimeError`` explains that the backend is not available.
    """
    from mindquantum.algorithm.qaia.SB import GPU_AVAILABLE  # pylint: disable=import-outside-toplevel

    if GPU_AVAILABLE:
        pytest.skip('The GPU SB extension is available in this build.')
    with pytest.raises(RuntimeError, match='not available'):
        BSB(_COUPLING, n_iter=2, backend=backend)
