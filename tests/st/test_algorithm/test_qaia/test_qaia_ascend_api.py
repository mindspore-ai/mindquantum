# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""CPU-only contracts for optional Ascend QAIA and backend selection."""
# J follows the public QAIA API.
# pylint: disable=invalid-name
# ruff: noqa: N806

import builtins
import importlib
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mindquantum.algorithm.qaia import (
    ASB,
    BSB,
    CAC,
    CFC,
    DSB,
    GBSB,
    LQA,
    LSB,
    NMFA,
    QAIA,
    SFC,
    TSB,
    USB,
    SimCIM,
)

pytestmark = [pytest.mark.level0, pytest.mark.platform_x86_cpu]


@pytest.mark.parametrize('solver_class', [BSB, DSB])
@pytest.mark.parametrize(
    'kwargs,match',
    [
        ({'J': np.ones((2, 3))}, 'square'),
        ({'J': np.empty((0, 0))}, 'square'),
        ({'J': np.array([[np.nan]])}, 'finite'),
        ({'J': np.array([[1e100]])}, 'finite'),
        ({'J': csr_matrix(([3e38, 3e38], [0, 0], [0, 2]), shape=(1, 1))}, 'finite'),
        ({'J': np.ones((3, 3), complex)}, 'real'),
        ({'J': np.eye(2)}, 'diagonal'),
        ({'J': np.array([[0.0, 1.0], [2.0, 0.0]])}, 'symmetric'),
        ({'J': csr_matrix([[0.0, np.inf], [np.inf, 0.0]])}, 'finite'),
        ({'J': np.array([[0.0, 1e100], [1e100, 0.0]])}, 'finite'),
        ({'batch_size': 0}, 'batch_size'),
        ({'batch_size': 1.5}, 'batch_size'),
        ({'batch_size': True}, 'batch_size'),
        ({'batch_size': 2**30}, 'indexing'),
        ({'batch_size': np.int32(2**30)}, 'indexing'),
        ({'n_iter': 0}, 'n_iter'),
        ({'n_iter': 2**31}, 'n_iter'),
        ({'h': np.ones((2, 2))}, 'h must'),
        ({'h': np.array([0.0, np.nan])}, 'finite'),
        ({'x': np.array([[np.inf], [0.0]])}, 'finite'),
        ({'dt': 0}, 'positive'),
        ({'dt': 1e100}, 'finite'),
        ({'xi': np.inf}, 'finite'),
        ({'J': np.zeros((3, 3))}, 'specify xi'),
        ({'device_id': -1}, 'device_id'),
        ({'device_id': 2**31}, 'device_id'),
        ({'device_id': True}, 'device_id'),
    ],
)
def test_invalid_problem_precedes_backend_availability(solver_class, kwargs, match, monkeypatch):
    """
    Feature: Host-side validation before checking optional Ascend backend availability.
    Description: Construct BSB/DSB with invalid inputs while blocking native import and reporting no extension.
    Expectation: Raise ValueError for the input, not an availability error, without importing CANN.
    """
    ascend = importlib.import_module('mindquantum.algorithm.qaia._ascend_sb')
    monkeypatch.setattr(ascend, 'ASCEND_AVAILABLE', False)
    original = builtins.__import__

    def reject_native(name, *args, **kw):
        if name == 'mindquantum._qaia_sb_ascend':
            raise AssertionError('Invalid input must fail before importing CANN')
        return original(name, *args, **kw)

    monkeypatch.setattr(builtins, '__import__', reject_native)
    with pytest.raises(ValueError, match=match):
        solver_class(**({'J': np.array([[0.0, 1.0], [1.0, 0.0]]), 'backend': 'ascend-float32'} | kwargs))


@pytest.mark.parametrize('solver_class', [BSB, DSB])
@pytest.mark.parametrize('matrix', [object(), [[0.0, 1.0], [1.0, 0.0]]])
def test_qaia_matrix_type_precedes_conversion_and_field_validation(solver_class, matrix):
    """
    Feature: Type validation of coupling matrix J in BSB and DSB on Ascend backends.
    Description: Pass non-array/non-sparse objects as J with backend='ascend-float32'.
    Expectation: Raise TypeError stating that J requires a NumPy array or SciPy sparse matrix.
    """
    with pytest.raises(TypeError, match='J requires'):
        solver_class(matrix, h=np.zeros(2), n_iter=1, backend='ascend-float32')


@pytest.mark.parametrize('solver_class', [BSB, DSB])
@pytest.mark.parametrize(
    'name,value,error,match',
    [
        ('x', np.zeros(2), ValueError, 'x must have shape'),
        ('y', np.zeros(2), ValueError, 'y must have shape'),
        ('x', np.full((2, 1), np.nan), ValueError, 'finite'),
        ('y', np.zeros((2, 3)), ValueError, 'y must have shape'),
        ('J', object(), TypeError, 'J requires'),
        ('J', np.zeros((2, 2, 2)), ValueError, 'square'),
        ('J', np.zeros((3, 3)), ValueError, 'number of spins'),
        ('J', np.eye(2), ValueError, 'diagonal'),
        ('J', np.array([[0.0, 1.0], [2.0, 0.0]]), ValueError, 'symmetric'),
        ('h', np.array([0.0, np.inf]), ValueError, 'finite'),
        ('p', np.array([np.inf]), ValueError, 'finite'),
        ('p', np.array([0.0, 1.0]), ValueError, 'p must'),
        ('delta', -1, ValueError, 'positive'),
        ('dt', 0, ValueError, 'positive'),
        ('xi', np.inf, ValueError, 'finite'),
    ],
)
def test_invalid_qaia_edits_precede_native_import(solver_class, name, value, error, match, monkeypatch):
    """
    Feature: Pre-launch validation of mutated QAIA attributes before importing the native Ascend module.
    Description: Mutate x, y, J, h, or p to invalid values on an initialized BSB/DSB instance and call update().
    Expectation: Raise the expected TypeError or ValueError before importing _qaia_sb_ascend.
    """
    original = builtins.__import__

    def reject_native(module, *args, **kwargs):
        if module == 'mindquantum._qaia_sb_ascend':
            raise AssertionError('Invalid input must fail before importing CANN')
        return original(module, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', reject_native)
    # Use CPU construction so this host-validation test needs no native layout
    # implementation. Explicit CANN construction loads the extension itself.
    solver = solver_class(np.array([[0.0, 1.0], [1.0, 0.0]]), n_iter=1)
    solver.backend = 'ascend-float32'
    setattr(solver, name, value)
    with pytest.raises(error, match=match):
        solver.update()


@pytest.mark.parametrize('algorithm', [QAIA, ASB, GBSB, CAC, CFC, SFC, LQA, NMFA, SimCIM, TSB, USB, LSB])
@pytest.mark.parametrize('backend', ['ascend-float32', 'gpu-float16', 'gpu-int8'])
def test_specialized_backends_are_not_accepted_by_other_algorithms(algorithm, backend):
    """
    Feature: Backend allowlist enforcement across QAIA algorithms.
    Description: Pass BSB/DSB-specific GPU and Ascend backends to other QAIA algorithm classes.
    Expectation: Raise ValueError indicating the backend is not supported by that algorithm.
    """
    with pytest.raises(ValueError, match='backend must be one of'):
        algorithm(np.array([[0.0, -1.0], [-1.0, 0.0]]), backend=backend)


@pytest.mark.parametrize('algorithm', [TSB, USB, LSB])
def test_quantized_backends_reject_torch_npu(algorithm):
    """
    Feature: Backend validation for quantized SB algorithms (TSB, USB, LSB).
    Description: Instantiate TSB, USB, and LSB with backend='npu-float32'.
    Expectation: Raise ValueError rejecting unsupported backends.
    """
    with pytest.raises(ValueError, match='backend must be one of'):
        algorithm(np.array([[0.0, -1.0], [-1.0, 0.0]]), backend='npu-float32')


def test_import_and_validated_missing_backend_in_fresh_process():
    """
    Feature: Lazy native extension loading and clear diagnostic when Ascend support is not built.
    Description: Import qaia and construct BSB and DSB in a subprocess without the native module.
    Expectation: Package import does not load the native extension, and solver construction raises RuntimeError.
    """
    # Model a CPU installation even on builders with the extension present.
    # An import finder rejects execution, rather than only checking sys.modules
    # after a potentially side-effecting import has already happened.
    source = '''
import builtins
import importlib.util
import sys

find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *a, **k: (
    None if name == 'mindquantum._qaia_sb_ascend' else find_spec(name, *a, **k)
)
original = builtins.__import__
attempts = []
def no_native(name, *a, **k):
    if name == 'mindquantum._qaia_sb_ascend':
        attempts.append(name)
        raise ImportError('CPU installation')
    return original(name, *a, **k)
builtins.__import__ = no_native
import mindquantum
from mindquantum.algorithm.qaia import BSB, DSB
import numpy as np
assert not attempts
for algorithm in (BSB, DSB):
    try:
        algorithm(np.array([[0., 1.], [1., 0.]]), backend='ascend-float32')
    except RuntimeError as exc:
        assert 'ENABLE_ASCEND' in str(exc)
    else:
        raise AssertionError('Missing backend accepted')
assert not attempts
'''
    subprocess.run([sys.executable, '-c', source], check=True, env=os.environ.copy(), timeout=60)
