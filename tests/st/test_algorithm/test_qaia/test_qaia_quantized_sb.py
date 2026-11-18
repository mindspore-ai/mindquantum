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
"""Functional tests for the quantized simulated bifurcation solvers.

MindQuantum 0.13 fixed two defects of the quantized SB family:

* ``LSB`` reversed the sign of the logarithmic scaling factor and divided by
  ``x = 0``;
* ``USB`` initialized its spin state in ``[-0.01, 0.01]``, a range that the
  uniform quantizer maps to zero, so the dynamics never started.

The tests below pin down the quantization rules themselves by replaying one
Euler step of the documented equations, so a regression in the scaling factor
or in the initialization range is caught even when the final cut value would
still look reasonable.
"""

import warnings

import numpy as np
import pytest

from mindquantum.algorithm.qaia import LSB, TSB, USB

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

    l_x = np.ceil(np.log2(np.abs(x0)))
    scale = np.ldexp(np.ones_like(x0), l_x.astype(int))
    v = np.sign(x0) * scale
    expected_y = (-x0 + _XI * _COUPLING.dot(v)) * 1.0
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
        warnings.simplefilter('error')
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

    p_x = np.round((2 * threshold + 1) * np.abs(x0) / 2)
    u_x = np.where(p_x <= threshold - 1, np.sign(x0) * 2 * p_x / (2 * threshold + 1), np.sign(x0))
    expected_y = -x0 + _XI * _COUPLING.dot(u_x)
    np.testing.assert_allclose(y1, expected_y, atol=1e-12)
    np.testing.assert_allclose(x1, x0 + expected_y, atol=1e-12)


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


def test_usb_initial_state_range_is_wide_enough():
    """
    Feature: Initial state range of ``USB``.

    Description: Draw the default initial state of a 64 spin problem and inspect its range.

    Expectation: Values cover ``[-0.1, 0.1]`` so that the uniform quantizer is not identically zero.
    """
    np.random.seed(2026)
    solver = USB(np.zeros((64, 64)), n_iter=2, xi=1.0)
    assert solver.x.shape == (64, 1)
    assert solver.x.min() >= -0.1
    assert solver.x.max() <= 0.1
    assert np.abs(solver.x).max() > 0.01


@pytest.mark.parametrize('strategy', ['linear', 'exponential', 'logarithmic'])
def test_tsb_time_strategy_is_monotonic(strategy):
    """
    Feature: Time schedule of the ternary solver ``TSB``.

    Description: Evaluate the selected schedule over the annealing window.

    Expectation: Values start at zero for the linear and logarithmic schedules and never decrease.
    """
    solver = TSB(_COUPLING, n_iter=10, xi=_XI, strategy=strategy)
    values = [solver.time_strategy[strategy](i, 10) for i in range(10)]
    assert all(np.diff(values) >= 0)
    if strategy in ('linear', 'logarithmic'):
        assert values[0] == pytest.approx(0.0)
    assert values[-1] < 0.7


def test_tsb_schedule_formulas():
    """
    Feature: Time schedule formulas of ``TSB``.

    Description: Compare each schedule with its documented closed form at several steps.

    Expectation: The formulas match exactly.
    """
    solver = TSB(_COUPLING, n_iter=100, xi=_XI)
    n_iter = 100
    for step in (0, 1, 25, 50, 99):
        ratio = step / n_iter
        assert solver.linear_time(step, n_iter) == pytest.approx(0.7 * ratio)
        assert solver.exponential_time(step, n_iter) == pytest.approx(0.7 * 2 ** (ratio - 1))
        assert solver.logarithmic_time(step, n_iter) == pytest.approx(0.7 * np.log2(ratio + 1))
