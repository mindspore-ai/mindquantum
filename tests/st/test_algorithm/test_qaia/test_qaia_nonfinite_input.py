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
"""Regression tests for non-finite QAIA coefficients.

Reported by `mindquantum issue #466
<https://gitcode.com/mindspore/mindquantum/issues/466>`_: the CPU path shared by
``BSB``/``DSB``/``GBSB`` accepts a coupling matrix, an external field, an
initial spin state or an integration parameter containing ``NaN``/``Inf``.
The constructor succeeds, the first ``update()`` fills ``x``/``y`` with ``NaN``
and only NumPy ``RuntimeWarning``s are emitted.  The Ascend backend validates
every input through ``mindquantum.algorithm.qaia._ascend_sb`` and raises a
``ValueError`` saying that the values must be finite, so the CPU path is
expected to do the same.

Note:
    ``J`` with a ``NaN`` entry is currently rejected by the symmetry check with
    the misleading message "J must be a symmetric matrix".  These tests require
    a finiteness error instead, so that the real cause of the rejection is
    reported to the user.

    The checks that need the missing CPU validation are marked ``xfail``
    (``strict=False``, see ``_ISSUE_466``) so that the level-0 CI stays green
    while issue #466 is open.  Remove the marker once the validation lands; the
    tests then enforce the new behaviour instead of reporting an expected
    failure.
"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mindquantum.algorithm.qaia import ASB, BSB, DSB, GBSB

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]

_SOLVER_CLASSES = [ASB, BSB, DSB, GBSB]
_NONFINITE = [np.nan, np.inf, -np.inf]
_COUPLING = np.array([[0.0, -1.0], [-1.0, 0.0]])
_N_ITER = 4

_ISSUE_466 = pytest.mark.xfail(
    reason='Open issue #466: the CPU QAIA path does not validate non-finite J/h/x/dt/xi/A yet.',
    strict=False,
)


def _solver_kwargs(solver_class):
    """Extra keyword arguments needed by the individual SB variants."""
    if solver_class is GBSB:
        return {'A': 0.2}
    return {}


@pytest.mark.parametrize('solver_class', _SOLVER_CLASSES)
@pytest.mark.parametrize('representation', ['dense', 'sparse'])
@_ISSUE_466
def test_nonfinite_coupling_matrix_is_rejected(solver_class, representation):
    """
    Feature: QAIA coupling matrix validation.

    Description: Build an SB variant where one off-diagonal coupling entry is ``NaN`` or ``Inf``.

    Expectation: A ``ValueError`` about non-finite values is raised before the solver is initialized.
    """
    for value in _NONFINITE:
        coupling = np.array([[0.0, value], [value, 0.0]])
        if representation == 'sparse':
            coupling = csr_matrix(coupling)
        with pytest.raises(ValueError, match='finite'):
            solver_class(coupling, n_iter=_N_ITER, **_solver_kwargs(solver_class))


@pytest.mark.parametrize('solver_class', _SOLVER_CLASSES)
@_ISSUE_466
def test_nonfinite_external_field_is_rejected(solver_class):
    """
    Feature: QAIA external field validation.

    Description: Provide an external field ``h`` whose second entry is ``NaN`` or ``Inf``.

    Expectation: A ``ValueError`` about non-finite values is raised, instead of a silent ``NaN`` run.
    """
    for value in _NONFINITE:
        with pytest.raises(ValueError, match='finite'):
            solver_class(
                _COUPLING,
                h=np.array([0.5, value]),
                n_iter=_N_ITER,
                **_solver_kwargs(solver_class),
            )


@pytest.mark.parametrize('solver_class', _SOLVER_CLASSES)
@_ISSUE_466
def test_nonfinite_initial_spins_are_rejected(solver_class):
    """
    Feature: QAIA initial spin validation.

    Description: Provide an initial spin state ``x`` containing ``NaN`` or ``Inf``.

    Expectation: A ``ValueError`` about non-finite values is raised before the state is used.
    """
    for value in _NONFINITE:
        with pytest.raises(ValueError, match='finite'):
            solver_class(
                _COUPLING,
                x=np.full((2, 1), value),
                n_iter=_N_ITER,
                **_solver_kwargs(solver_class),
            )


@pytest.mark.parametrize('solver_class', _SOLVER_CLASSES)
@pytest.mark.parametrize('name', ['dt', 'xi'])
@_ISSUE_466
def test_nonfinite_integration_parameters_are_rejected(solver_class, name):
    """
    Feature: QAIA integration parameter validation.

    Description: Provide a non-finite step size ``dt`` or coupling strength ``xi``.

    Expectation: A ``ValueError`` about non-finite values is raised.
    """
    for value in _NONFINITE:
        kwargs = {'n_iter': _N_ITER}
        kwargs[name] = value
        kwargs.update(_solver_kwargs(solver_class))
        with pytest.raises(ValueError, match='finite'):
            solver_class(_COUPLING, **kwargs)


@_ISSUE_466
def test_nonfinite_gbsb_bifurcation_parameter_is_rejected():
    """
    Feature: GBSB individual bifurcation parameter validation.

    Description: Provide a non-finite bifurcation parameter ``A``.

    Expectation: A ``ValueError`` about non-finite values is raised.
    """
    for value in _NONFINITE:
        with pytest.raises(ValueError, match='finite'):
            GBSB(_COUPLING, n_iter=_N_ITER, A=value)


@pytest.mark.parametrize('solver_class', _SOLVER_CLASSES)
def test_finite_solver_never_produces_nonfinite_state(solver_class):
    """
    Feature: QAIA numerical safety with finite inputs.

    Description: Run a small finite problem and inspect the state, the cut and the energy.

    Expectation: Every produced value is finite, so the finiteness validation does not reject valid input.
    """
    solver = solver_class(_COUPLING, n_iter=_N_ITER, batch_size=2, **_solver_kwargs(solver_class))
    solver.update()
    assert np.isfinite(solver.x).all()
    assert np.isfinite(solver.y).all()
    assert np.isfinite(solver.calc_cut()).all()
    assert np.isfinite(solver.calc_energy()).all()


@pytest.mark.parametrize('solver_class', _SOLVER_CLASSES)
def test_large_finite_coefficients_are_accepted(solver_class):
    """
    Feature: QAIA numerical range.

    Description: Use large but finite couplings and external fields.

    Expectation: The solver is constructed, the coupling strength stays finite and the run is stable.
    """
    coupling = _COUPLING * 1e3
    field = np.array([1e5, -1e5])
    solver = solver_class(
        coupling,
        h=field,
        n_iter=_N_ITER,
        **_solver_kwargs(solver_class),
    )
    assert np.isfinite(float(solver.xi))
    solver.update()
    assert np.isfinite(solver.x).all()
