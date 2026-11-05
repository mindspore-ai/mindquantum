# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Host input validation and optional native-plan discovery for BSB/DSB."""
# J follows the public QAIA API and the Ising equations throughout this module.
# pylint: disable=invalid-name
# ruff: noqa: N803, N806

import importlib.util
from numbers import Integral

import numpy as np
from scipy.sparse import issparse

# Discovery must not load CANN libraries or initialize a device.
ASCEND_AVAILABLE = importlib.util.find_spec('mindquantum._qaia_sb_ascend') is not None
ASCEND_UNAVAILABLE_REASON = (
    "Ascend QAIA backend is not available. Build MindQuantum with "
    "-DENABLE_ASCEND=ON and -DMQ_ASCEND_SOC=Ascend910B[1-4], and source CANN set_env.sh."
)


def _array(value, name, shapes=None):
    """Validate and convert an array-like input to finite float32."""
    value = _matrix(value, name).toarray() if issparse(value) else value
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    try:
        with np.errstate(over="raise", invalid="raise"):
            array = np.asarray(array, dtype=np.float32)
    except FloatingPointError as exc:
        raise ValueError(f"{name} must be finite float32 values") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float32 values")
    if shapes is not None and array.shape not in shapes:
        raise ValueError(f"{name} must have shape {' or '.join(map(str, shapes))}")
    return array


def _matrix(value, name="J"):
    if not issparse(value):
        return _array(value, name)
    # Canonicalize in the input dtype before rounding: duplicate entries may
    # cancel. CSR also gives LIL/DOK/BSR inputs one flat numeric representation.
    matrix = value.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.data = _array(matrix.data, name)
    return matrix


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 < value <= 2**31 - 1:
        raise ValueError(f"{name} must be a positive int32")
    return int(value)


def _check_device_id(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= 2**31 - 1:
        raise ValueError("device_id must be a nonnegative int32")


def _check_dimensions(J, batch_size, n_iter):
    """Check logical dimensions without loading the optional native extension."""
    batch_size = _positive_int(batch_size, "batch_size")
    _positive_int(n_iter, "n_iter")
    shape = np.shape(J)
    if len(shape) != 2 or shape[0] != shape[1] or shape[0] == 0:
        raise ValueError("J must be a nonempty square matrix")
    if shape[0] * batch_size > 2**31 - 1:
        raise ValueError("state exceeds int32 indexing")


def _parameters(delta, xi, dt):
    parameters = _array([delta, xi, dt], "delta/xi/dt")
    if parameters[0] <= 0 or parameters[2] <= 0:
        raise ValueError("delta and dt must be positive")
    return tuple(map(float, parameters))


def _get_plan_type():
    """Load the native boundary on explicit CANN use, without initializing ACL."""
    if not ASCEND_AVAILABLE:
        raise RuntimeError(ASCEND_UNAVAILABLE_REASON)
    try:
        from mindquantum._qaia_sb_ascend import Plan
    except ImportError as exc:
        raise RuntimeError(ASCEND_UNAVAILABLE_REASON) from exc
    return Plan
