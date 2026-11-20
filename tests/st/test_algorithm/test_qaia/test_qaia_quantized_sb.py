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
"""One-step references for logarithmic and uniform SB quantization."""

import warnings

import numpy as np
import pytest

from mindquantum.algorithm.qaia import LSB, USB

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]

_COUPLING = np.array([[0.0, 0.5], [0.5, 0.0]])
_XI = 0.5


def _one_step(solver, x0):
    """Return the momentum and position after exactly one Euler step."""
    solver.x = x0.copy()
    solver.y = np.zeros_like(x0)
    solver.update()
    return solver.x, solver.y


def test_lsb_reference_step():
    """
    Feature: Logarithmic quantization of ``LSB``.

    Description: Replay one Euler step of ``LSB`` with a hand computed quantization of the spin state.

    Expectation: The momentum increment matches ``-x + xi * J @ v`` with ``v = sign(x) * 2 ** ceil(log2 |x|)``.
    """
    x0 = np.array([[0.5], [0.1]])
    solver = LSB(_COUPLING, n_iter=1, xi=_XI, x=x0.copy())
    x1, y1 = _one_step(solver, x0)

    expected_y = np.array(
        [
            [-0.46875],  # -1/2 + (1/2)(1/2)(1/8), with v[1] = 2**ceil(log2(0.1)) = 1/8.
            [0.025],  # -1/10 + (1/2)(1/2)(1/2), with v[0] = 2**ceil(log2(0.5)) = 1/2.
        ]
    )
    np.testing.assert_allclose(y1, expected_y, atol=1e-12)

    expected_x = x0 + expected_y
    np.testing.assert_allclose(x1, expected_x, atol=1e-12)


def test_lsb_drops_entries_below_the_threshold():
    """
    Feature: Logarithmic quantization cut-off of ``LSB``.

    Description: Use a spin whose logarithm is below ``-threshold`` so that its quantized value vanishes.

    Expectation: The corresponding coupling term is zero and the update only follows ``-x``.
    """
    threshold = 8
    x0 = np.array([[0.5], [1e-4]])
    solver = LSB(_COUPLING, n_iter=1, xi=_XI, threshold=threshold, x=x0.copy())
    x1, y1 = _one_step(solver, x0)

    v = np.array([[0.5], [0.0]])
    expected_y = -x0 + _XI * _COUPLING.dot(v)
    np.testing.assert_allclose(y1, expected_y, atol=1e-12)
    np.testing.assert_allclose(x1, x0 + expected_y, atol=1e-12)


def test_lsb_handles_zero_spins_without_division():
    """
    Feature: Zero spin handling of ``LSB``.

    Description: Start from an all-zero spin state, where ``log2(0)`` would be undefined.

    Expectation: No floating point warning is emitted and the resulting state stays finite.
    """
    solver = LSB(_COUPLING, n_iter=5, xi=_XI, batch_size=3, x=np.zeros((2, 3)))
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        solver.update()
    assert np.isfinite(solver.x).all()
    assert np.isfinite(solver.y).all()


def test_usb_quantization_reference_step():
    """
    Feature: Uniform quantization of ``USB``.

    Description: Replay one Euler step of ``USB`` with a hand computed uniform quantization.

    Expectation: The momentum increment matches the documented ``u_x`` rule.
    """
    threshold = 10
    x0 = np.array([[0.5], [0.1]])
    solver = USB(_COUPLING, n_iter=1, xi=_XI, threshold=threshold, x=x0.copy())
    x1, y1 = _one_step(solver, x0)

    expected_y = np.array(
        [
            [-0.5 + 0.5 / 21],  # -1/2 + (1/2)(1/2)(2/21), with round(21 * 0.1 / 2) = 1.
            [-0.1 + 2.5 / 21],  # -1/10 + (1/2)(1/2)(10/21), with round(21 * 0.5 / 2) = 5.
        ]
    )
    np.testing.assert_allclose(y1, expected_y, atol=1e-12)
    np.testing.assert_allclose(x1, x0 + expected_y, atol=1e-12)


def test_usb_default_state_has_nonzero_quantized_coupling(monkeypatch):
    """
    Feature: USB default initialization.
    Description: Initialize without explicit spins using a fixed non-central random sample and evolve one step.
    Expectation: Quantization preserves a nonzero coupling force instead of rounding all initial spins to zero.
    """
    monkeypatch.setattr(np.random, 'rand', lambda *shape: np.full(shape, 0.9))
    solver = USB(_COUPLING, n_iter=1, xi=_XI)
    initial = solver.x.copy()
    solver.y = np.zeros_like(initial)
    solver.update()
    assert np.any(np.abs(solver.y + initial) > 1e-12)


def test_usb_saturates_at_unit_spin():
    """
    Feature: Saturation of the uniform quantizer of ``USB``.

    Description: Use a spin above the largest quantization level.

    Expectation: The quantized value saturates at the sign of the spin.
    """
    threshold = 10
    x0 = np.array([[-0.99], [0.99]])
    solver = USB(_COUPLING, n_iter=1, xi=_XI, threshold=threshold, x=x0.copy())
    x1, y1 = _one_step(solver, x0)

    u_x = np.sign(x0)
    expected_y = -x0 + _XI * _COUPLING.dot(u_x)
    np.testing.assert_allclose(y1, expected_y, atol=1e-12)
    np.testing.assert_allclose(x1, x0 + expected_y, atol=1e-12)
