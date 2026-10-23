"""Batched DSB/BSB on Ascend 910B, using CANN 9.1 Cube GEMM and graph replay.

Solver snapshots its coefficients on construction. run(x, y) never mutates its
inputs. Reuse the Solver for repeated starts; use a context manager to release
device resources promptly. FP32 is the default; HF32 is an explicit accuracy /
throughput tradeoff, not a fallback. No CPU execution is hidden in this API.
"""

from numbers import Integral

import numpy as np
from scipy.sparse import issparse


def _array(value, name):
    value = value.toarray() if issparse(value) else value
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    with np.errstate(over="raise", invalid="raise"):
        array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite float32 values")
    return array


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 < value <= 2**31 - 1:
        raise ValueError(f"{name} must be a positive int32")
    return int(value)


class Solver:
    """An immutable problem and schedule, with reusable, privately owned state.

    J: square dense or SciPy sparse matrix; h: (N,) or (N,1).
    x/y for run: (N,batch_size), or (N,) when batch_size=1; any input layout.
    Returns two independent Fortran-contiguous FP32 arrays (N,batch_size).
    Calls on one Solver are serialized; distinct Solvers own separate contexts.
    device_id is the ACL logical device ID (not npu-smi's physical card ID).
    """

    def __init__(self, J, h=None, *, algorithm="dsb", batch_size=1, n_iter=1000,
                 p=None, delta=1.0, xi=None, dt=1.0, device_id=0, precision="fp32"):
        if algorithm not in ("dsb", "bsb"):
            raise ValueError("algorithm must be 'dsb' or 'bsb'")
        if precision not in ("fp32", "hf32"):
            raise ValueError("precision must be 'fp32' or 'hf32'")
        self.batch_size = _positive_int(batch_size, "batch_size")
        self.n_iter = _positive_int(n_iter, "n_iter")
        J = _array(J, "J")
        if J.ndim != 2 or J.shape[0] != J.shape[1] or J.shape[0] == 0:
            raise ValueError("J must be a nonempty square matrix")
        self.n = J.shape[0]
        # CompareScalar on 910B consumes full 256-byte repeats (64 FP32s).
        # This alignment also satisfies DMA and Cube, without special tail math.
        self._padded_n = (self.n + 63) // 64 * 64
        if self._padded_n * self.batch_size > 2**31 - 1:
            raise ValueError("padded state exceeds int32 indexing")
        if h is None:
            h = np.zeros(self.n, dtype=np.float32)
        h = _array(h, "h")
        if h.shape not in ((self.n,), (self.n, 1)):
            raise ValueError("h must have shape (N,) or (N,1)")
        if p is None:
            p = np.linspace(0, 1, self.n_iter, dtype=np.float32)
        p = _array(p, "p")
        if p.shape != (self.n_iter,):
            raise ValueError("p must have shape (n_iter,)")
        if xi is None:
            norm = np.sqrt(np.einsum("ij,ij->", J, J, dtype=np.float64))
            if norm == 0:
                raise ValueError("specify xi explicitly for a zero coupling matrix")
            xi = 0.5 * np.sqrt(self.n - 1) / norm
        parameters = _array([delta, xi, dt], "delta/xi/dt")
        if parameters[0] <= 0 or parameters[2] <= 0:
            raise ValueError("delta and dt must be positive")
        if isinstance(device_id, bool) or not isinstance(device_id, Integral) or device_id < 0:
            raise ValueError("device_id must be a nonnegative integer")
        self.algorithm, self.precision = algorithm, precision
        self.delta, self.xi, self.dt = map(float, parameters)

        # One-time owned snapshots. Padding is internal, never exposed to users.
        matrix = np.zeros((self._padded_n, self._padded_n), dtype=np.float32)
        matrix[:self.n, :self.n] = J
        field = np.zeros(self._padded_n, dtype=np.float32)
        field[:self.n] = h.reshape(-1)
        pressure = np.array(p, dtype=np.float32, order="C", copy=True)
        from _kernel_sb import Plan

        self._plan = Plan(matrix, field, pressure, self.batch_size, algorithm == "dsb",
                          self.delta, self.xi, self.dt, int(device_id), 0 if precision == "fp32" else 3)

    def _state(self, value, name):
        value = _array(value, name)
        if value.ndim == 1 and self.batch_size == 1:
            value = value.reshape(-1, 1)
        if value.shape != (self.n, self.batch_size):
            raise ValueError(f"{name} must have shape ({self.n}, {self.batch_size})")
        if self._padded_n == self.n:
            return np.asfortranarray(value)
        padded = np.zeros((self._padded_n, self.batch_size), dtype=np.float32, order="F")
        padded[:self.n] = value
        return padded

    def run(self, x, y):
        """Run the complete schedule from explicit initial states, synchronously."""
        x, y = self._plan.run(self._state(x, "x"), self._state(y, "y"))
        return np.asfortranarray(x[:self.n]), np.asfortranarray(y[:self.n])

    @property
    def last_device_ms(self):
        """ACL event time of the last graph replay, excluding state transfers."""
        return self._plan.last_device_ms

    def close(self):
        self._plan.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def _solve(J, h=None, x=None, y=None, *, algorithm, batch_size=1, seed=None, **kwargs):
    # Explicit states determine batch size, as in the original convenience API.
    initial = x if x is not None else y
    if initial is not None:
        shape = np.shape(initial)
        if len(shape) not in (1, 2):
            raise ValueError("x/y must be one- or two-dimensional")
        batch_size = shape[1] if len(shape) == 2 else 1
    with Solver(J, h, algorithm=algorithm, batch_size=batch_size, **kwargs) as solver:
        rng = np.random.default_rng(seed)
        shape = (solver.n, solver.batch_size)
        if x is None:
            x = rng.uniform(-0.01, 0.01, shape).astype(np.float32)
        if y is None:
            y = rng.uniform(-0.01, 0.01, shape).astype(np.float32)
        return solver.run(x, y)


def kernel_dsb(J, h=None, x=None, y=None, **kwargs):
    """One-shot DSB; prefer Solver for repeated starts. Inputs are not mutated."""
    return _solve(J, h, x, y, algorithm="dsb", **kwargs)


def kernel_bsb(J, h=None, x=None, y=None, **kwargs):
    """One-shot BSB; prefer Solver for repeated starts. Inputs are not mutated."""
    return _solve(J, h, x, y, algorithm="bsb", **kwargs)
