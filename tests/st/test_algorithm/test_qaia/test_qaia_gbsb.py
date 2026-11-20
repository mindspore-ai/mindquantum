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
"""Bounded GBSB dynamics and independent batch trajectories."""

import numpy as np
import pytest

from mindquantum.algorithm.qaia import GBSB

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]
_COUPLING = np.array([[0.0, -1.0, 0.5], [-1.0, 0.0, 0.5], [0.5, 0.5, 0.0]])


@pytest.mark.parametrize('bifurcation', [0.0, 0.2, 2.0], ids=['zero', 'default', 'large'])
def test_gbsb_box_bounds(bifurcation):
    """
    Feature: GBSB bounded dynamics.
    Description: Evolve explicit spins with small and large bifurcation strengths.
    Expectation: Spins remain in the unit box and momentum and pressure stay finite.
    """
    solver = GBSB(_COUPLING, n_iter=30, A=bifurcation, x=np.array([[0.1], [-0.2], [0.3]]))
    solver.y = np.zeros_like(solver.x)
    solver.update()
    assert np.all(np.abs(solver.x) <= 1)
    assert np.isfinite(solver.y).all()
    assert np.isfinite(solver.p_i).all()
    assert np.all(solver.p_i >= 0)
    assert (solver.p_i <= 1).all()


def test_gbsb_batch_independence():
    """
    Feature: Independent GBSB replicas.
    Description: Compare a batch of distinct spin columns with separate single-column evolutions.
    Expectation: Every column's position, momentum and pressure match its isolated trajectory.
    """
    initial = np.array([[0.1, -0.4], [-0.2, 0.3], [0.3, 0.2]])
    batched = GBSB(_COUPLING, n_iter=20, batch_size=2, x=initial.copy())
    batched.y = np.zeros_like(initial)
    batched.update()
    for index in range(initial.shape[1]):
        single = GBSB(_COUPLING, n_iter=20, x=initial[:, index : index + 1].copy())
        single.y = np.zeros_like(single.x)
        single.update()
        for name in ('x', 'y', 'p_i'):
            np.testing.assert_allclose(getattr(batched, name)[:, index : index + 1], getattr(single, name), atol=1e-12)


@pytest.mark.parametrize(
    'value,error,match',
    [
        pytest.param(-0.1, ValueError, 'not less than 0', id='negative'),
        pytest.param('0.2', TypeError, 'requires a number', id='string'),
    ],
)
def test_gbsb_bifurcation_validation(value, error, match):
    """
    Feature: GBSB bifurcation validation.
    Description: Pass a negative or non-numeric bifurcation strength.
    Expectation: Invalid values are rejected before evolution.
    """
    with pytest.raises(error, match=match):
        GBSB(_COUPLING, A=value)
