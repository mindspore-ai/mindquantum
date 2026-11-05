# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Hardware tests. Build first; failures are not replaced with CPU fallbacks."""

# J follows the QAIA API and the independent reference equations.
# pylint: disable=invalid-name
# ruff: noqa: N803, N806
import ctypes
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import coo_matrix, csr_matrix

from mindquantum.algorithm.qaia import BSB, DSB

DEVICE = int(os.environ.get("SB_TEST_DEVICE", "0"))
pytestmark = [
    pytest.mark.level0,
    pytest.mark.platform_arm_ascend910b_training,
    pytest.mark.env_onecard,
    pytest.mark.skipif(find_spec('mindquantum._qaia_sb_ascend') is None, reason='Ascend QAIA extension not built'),
]


def reference(J, h, x, y, p, algorithm, delta, xi, dt):
    """Compute the FP64 reference trajectory for DSB and BSB."""
    # FP64 oracle, with an independently expressed wall and update.
    x, y = x.astype(np.float64), y.astype(np.float64)
    for pressure in p:
        force = J.astype(np.float64) @ (np.sign(x) if algorithm == "dsb" else x)
        velocity = y + dt * ((pressure - delta) * x + xi * (force + h))
        position = x + dt * delta * velocity
        outside = np.abs(position) > 1
        x = np.where(outside, np.sign(position), position)
        y = np.where(outside, 0, velocity)
    return x, y


@pytest.mark.parametrize('solver_class,algorithm', [(BSB, 'bsb'), (DSB, 'dsb')])
@pytest.mark.parametrize(
    "n,batch", [(1, 1), (17, 3), (64, 65), (1023, 3), (1024, 1), (1025, 193), (2049, 2), (5001, 1)]
)
def test_trajectory_and_tail(solver_class, algorithm, n, batch):
    """
    Feature: Ascend SB trajectory accuracy and padded tail handling.
    Description: Run DSB and BSB across unaligned spin counts and batch sizes against an FP64 reference.
    Expectation: Final positions and momenta match the FP64 reference within FP32 tolerance.
    """
    rng = np.random.default_rng(n)
    upper = np.triu(rng.uniform(-0.07, 0.09, (n, n)).astype(np.float32), 1)
    J = upper + upper.T
    h = rng.uniform(-0.3, 0.5, (n, 1)).astype(np.float32)
    x = rng.uniform(-0.8, 0.7, (n, batch)).astype(np.float32)
    x[::7] = 0
    y = rng.uniform(-0.4, 0.6, (n, batch)).astype(np.float32)
    p = np.array([0.13, 0.27, -0.08, 1.11], np.float32)
    delta, xi, dt = map(np.float32, (0.8, 0.2, 0.13))
    original = x.copy(), y.copy()
    expected = reference(J, h, x, y, p, algorithm, delta, xi, dt)
    solver = solver_class(
        J, h, x=x, batch_size=batch, n_iter=len(p), xi=xi, dt=dt, backend='ascend-float32', device_id=DEVICE
    )
    solver.y, solver.p, solver.delta = y, p, delta
    solver.update()
    actual = solver.x, solver.y
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=3e-6)
        assert a.dtype == np.float32 and a.flags.f_contiguous
    for got, want in zip((x, y), original):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('solver_class', [BSB, DSB])
def test_padded_layout_rejected_before_state_allocation(solver_class, monkeypatch):
    """
    Feature: Native layout limits checked before allocating initial states.
    Description: Use a logical state that fits int32 but exceeds its limit after native padding.
    Expectation: Constructor and initialize reject the layout without allocating random states.
    """
    J = np.zeros((65, 65), np.float32)
    solver = solver_class(J, xi=0.5, n_iter=1, backend='ascend-float32', device_id=DEVICE)

    def no_allocation(*_):
        pytest.fail('Invalid padded layout must fail before random state allocation')

    monkeypatch.setattr(np.random, 'rand', no_allocation)
    with pytest.raises(ValueError, match='padded state exceeds int32 indexing'):
        solver_class(J, xi=0.5, n_iter=1, batch_size=2**24, backend='ascend-float32', device_id=DEVICE)
    solver.x = None
    solver.batch_size = 2**24
    with pytest.raises(ValueError, match='padded state exceeds int32 indexing'):
        solver.initialize()


@pytest.mark.parametrize('solver_class', [BSB, DSB])
def test_strict_wall_and_zero_sign(solver_class):
    """
    Feature: Strict bifurcating wall boundary condition and zero-sign convention.
    Description: Run one step on states crossing, touching, or staying inside the wall across 64-lane repeats.
    Expectation: Positions outside [-1, 1] are clamped with zeroed momentum while boundary states keep momentum.
    """
    # Six rows deliberately do not divide the 64-lane mask repeat. Exercise
    # different predicates across repeats, replica boundaries and padded tails.
    x = np.zeros((1032, 6), np.float32)
    y = np.tile(np.array([[2, -2], [1, -1], [0.875, -0.875], [0.25, -0.75], [0, 0], [-1.5, 1.5]], np.float32), (172, 3))
    solver = solver_class(
        np.zeros((1032, 1032), np.float32),
        x=x,
        batch_size=6,
        n_iter=1,
        xi=1,
        backend='ascend-float32',
        device_id=DEVICE,
    )
    solver.y = y
    solver.p[:] = 1
    solver.update()
    actual_x, actual_y = solver.x, solver.y
    np.testing.assert_array_equal(
        actual_x, np.tile([[1, -1], [1, -1], [0.875, -0.875], [0.25, -0.75], [0, 0], [-1, 1]], (172, 3))
    )
    np.testing.assert_array_equal(
        actual_y, np.tile([[0, 0], [1, -1], [0.875, -0.875], [0.25, -0.75], [0, 0], [0, 0]], (172, 3))
    )


@pytest.mark.parametrize('algorithm', ['dsb', 'bsb'])
@pytest.mark.parametrize('n,layout', [(64, 'F'), (65, 'F'), (65, 'C'), (65, 'strided'), (65, 'broadcast')])
def test_native_snapshot_replay_isolation_and_concurrent_calls(algorithm, n, layout):
    """
    Feature: Native layout, coefficient snapshot isolation and thread-safe graph replay.
    Description: Check nonsymmetric GEMM with strided states, mutate coefficients, and replay native plans concurrently.
    Expectation: Unpadded FP32 results match FP64, plans preserve inputs and snapshots, and RAII isolates resources.
    """
    from mindquantum._qaia_sb_ascend import Plan

    rng = np.random.default_rng(72)
    # Only the native boundary accepts nonsymmetric J. Symmetric algorithm
    # inputs cannot distinguish J @ X from an incorrectly transposed GEMM.
    J = rng.normal(0, 0.1, (n, n)).astype(np.float32)
    h = rng.normal(0, 0.2, n).astype(np.float32)
    p = np.linspace(0, 1, 11, dtype=np.float32)
    states = []
    for _ in range(2):
        values = rng.normal(0, 0.1, (n, 5)).astype(np.float32)
        if layout == 'strided':
            # Both negative strides and an unaligned address must be supported.
            storage = np.ndarray((2 * n, 10), dtype=np.float32, buffer=bytearray(2 * n * 10 * 4 + 1), offset=1)
            state = storage[::-2, ::-2]
            state[:] = values
        elif layout == 'broadcast':
            state = np.broadcast_to(values[:, :1], (n, 5))
        else:
            state = np.array(values, order=layout)
        states.append(state)
    x, y = states
    original = x.copy(), y.copy()
    oracle = reference(J, h[:, None], x, y, p, algorithm, 1.0, 0.5, 0.1)
    a = Plan(J, h, p, 5, algorithm == 'dsb', 1.0, 0.5, 0.1, DEVICE)
    expected = a.run(x, y)
    for got, want in zip(expected, oracle):
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=3e-6)
        assert got.shape == (n, 5) and got.dtype == np.float32 and got.flags.f_contiguous
    J[:] = -3
    h[:] = 12
    p[:] = 0.81
    b = Plan(J, h, p, 5, algorithm == 'dsb', 1.0, 0.5, 0.1, DEVICE)
    other = b.run(x, y)
    assert not np.allclose(other[0], expected[0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = list(pool.map(lambda _: a.run(x, y), range(4)))
    for result in runs:
        for got, want in zip(result, expected):
            np.testing.assert_array_equal(got, want)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(s.run, x, y) for s in (a, b)]
        for future, want in zip(futures, (expected, other)):
            for got, target in zip(future.result(), want):
                np.testing.assert_array_equal(got, target)
    del b
    for got, want in zip(a.run(x, y), expected):
        np.testing.assert_array_equal(got, want)
    del a
    for got, want in zip((x, y), original):
        np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize('solver_class', [BSB, DSB])
def test_long_schedule_and_restart(solver_class):
    """
    Feature: Reusable BSB/DSB execution plans for long schedules.
    Description: Run 1000 iterations twice from the same explicit initial states.
    Expectation: Both runs return identical, finite, bounded state trajectories.
    """
    J = np.array([[0, 0.25, -0.5], [0.25, 0, 0.125], [-0.5, 0.125, 0]], np.float32)
    rng = np.random.default_rng(37)
    x, y = (rng.uniform(-0.01, 0.01, (3, 3)).astype(np.float32) for _ in range(2))
    solver = solver_class(J, x=x, n_iter=1000, batch_size=3, backend='ascend-float32', device_id=DEVICE)
    solver.y = y
    solver.update()
    first = solver.x, solver.y
    solver.x, solver.y = x, y
    solver.update()
    second = solver.x, solver.y
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
    assert np.max(np.abs(first[0])) <= 1


@pytest.mark.parametrize('solver_class', [BSB, DSB])
def test_mismatched_and_nonfinite_states(solver_class):
    """
    Feature: Runtime state shape and finiteness validation on cached BSB/DSB updates.
    Description: Capture a plan, then set mismatched state shapes or NaN state entries.
    Expectation: Raise ValueError before launching the device graph.
    """
    solver = solver_class(
        np.ones((17, 17)) - np.eye(17), batch_size=3, n_iter=1, backend='ascend-float32', device_id=DEVICE
    )
    solver.update()
    solver.y = np.zeros((17, 2))
    with pytest.raises(ValueError, match="y must have shape"):
        solver.update()
    solver.y = np.zeros((17, 3))
    solver.x[:] = np.nan
    with pytest.raises(ValueError, match="finite"):
        solver.update()


@pytest.mark.parametrize('sparse_format', ['csr', 'csc', 'coo', 'bsr', 'dia', 'lil', 'dok'])
def test_sparse_coefficients_preserve_values_and_inputs(sparse_format):
    """
    Feature: SciPy sparse coupling matrix canonicalization and input preservation.
    Description: Construct DSB from each SciPy sparse format with duplicate entries that cancel in FP64.
    Expectation: Duplicate entries are summed before FP32 rounding and the caller's sparse matrix is unchanged.
    """
    # The duplicate at (0, 1) sums to 1 in FP64, but to 0 if cast before summing.
    sparse = coo_matrix(
        ([16777217.0, -16777216.0, -2.0, 0.5] * 2, ([0, 0, 1, 2, 1, 1, 2, 0], [1, 1, 2, 0, 0, 0, 1, 2])), shape=(3, 3)
    ).asformat(sparse_format)
    J = np.array([[0.0, 1.0, 0.5], [1.0, 0.0, -2.0], [0.5, -2.0, 0.0]])
    original_nnz = sparse.nnz
    x, y = np.array([[0.3], [-0.2], [0.5]]), np.array([[0.7], [0.1], [-0.3]])
    xi = 0.5 * np.sqrt(2) / np.sqrt(10.5)
    expected = reference(J, np.zeros((3, 1)), x, y, [0.0, 1.0], 'dsb', 1.0, xi, 0.1)
    solver = DSB(sparse, x=x, n_iter=2, dt=0.1, backend='ascend-float32', device_id=DEVICE)
    solver.y = y
    solver.update()
    for got, want in zip((solver.x, solver.y), expected):
        np.testing.assert_allclose(got, want, atol=1e-7, rtol=1e-6)
    assert sparse.shape == (3, 3) and sparse.dtype == np.float64 and sparse.nnz == original_nnz
    np.testing.assert_array_equal(sparse.toarray(), J)


@pytest.mark.parametrize('solver_class,algorithm', [(BSB, 'bsb'), (DSB, 'dsb')])
def test_qaia_restarts_and_coefficient_edits(solver_class, algorithm):
    """
    Feature: QAIA BSB/DSB Ascend backend graph reuse and cache invalidation on coefficient edits.
    Description: Repeatedly run update() after restarting or mutating J, h, p, dt, delta, xi, n_iter, and batch_size.
    Expectation: Each update matches the FP64 reference and invalid edits raise ValueError.
    """
    J = np.array([[0.0, 2.0, -0.5], [2.0, 0.0, -0.75], [-0.5, -0.75, 0.0]])
    h = np.array([[0.7], [-0.3], [0.2]])
    x = np.array([[0.0, 0.1], [-0.25, 0.8], [0.6, -0.2]])
    y = np.array([[0.5, 0.2], [-0.7, 0.3], [0.1, -0.4]])
    solver = solver_class(J, h, x=x.copy(), n_iter=3, batch_size=2, dt=0.25, backend='ascend-float32', device_id=DEVICE)
    expected_xi = 0.5 * np.sqrt(2) / np.sqrt((J**2).sum() + 0.5 * (h**2).sum())
    assert solver.xi == pytest.approx(expected_xi)
    for edit in ('none', 'restart', 'J', 'h', 'p', 'dt', 'delta', 'xi', 'n_iter', 'batch_size'):
        if edit == 'J':
            solver.J.data *= -1.7
        elif edit == 'h':
            solver.h[:] += 1.3
        elif edit == 'p':
            solver.p[:] = [0.75, -0.25, 1.1]
        elif edit == 'n_iter':
            solver.n_iter = 2
            solver.p = np.array([0.3, -0.2])
        elif edit == 'batch_size':
            solver.batch_size = 1
            x, y = x[:, :1], y[:, :1]
        elif edit in ('dt', 'delta', 'xi'):
            setattr(solver, edit, getattr(solver, edit) * 0.6)
        solver.x = x.copy()
        solver.initialize()
        solver.y = y.copy()
        expected = reference(
            solver.J.toarray(), solver.h, x, y, solver.p, algorithm, solver.delta, solver.xi, solver.dt
        )
        solver.update()
        for got, want in zip((solver.x, solver.y), expected):
            np.testing.assert_allclose(got, want, atol=3e-6, rtol=2e-5)
        spin = np.sign(solver.x)
        np.testing.assert_allclose(
            solver.calc_cut(), 0.25 * np.sum(spin * (solver.J @ spin), axis=0) - 0.25 * solver.J.sum()
        )
        np.testing.assert_allclose(
            solver.calc_energy(), -0.5 * np.sum(spin * (solver.J @ spin), axis=0, keepdims=True) - solver.h.T @ spin
        )
    # False compares equal to device 0 in a cache key, but is not a device ID.
    solver.device_id = False
    with pytest.raises(ValueError, match='device_id'):
        solver.update()
    solver.device_id = DEVICE
    solver.J[0, 1] += 1
    with pytest.raises(ValueError, match='symmetric'):
        solver.update()


@pytest.mark.parametrize('solver_class,algorithm', [(BSB, 'bsb'), (DSB, 'dsb')])
def test_qaia_matrix_representation_preserves_validation(solver_class, algorithm):
    """
    Feature: Dense vs. sparse symmetry validation across QAIA Ascend graph cache keys.
    Description: Alternate between a near-symmetric dense matrix and its CSR conversion on the same solver instance.
    Expectation: Dense matrix passes tolerant symmetry validation while CSR matrix raises ValueError.
    """
    # These values pass dense allclose, but fail exact sparse symmetry. Their
    # normalized CSR bytes are identical, so content alone cannot cache validity.
    near_symmetric = np.array([[0.0, 1.0], [1.000002, 0.0]])
    x, y = np.array([[0.2], [-0.3]]), np.array([[0.4], [0.1]])
    solver = solver_class(
        np.array([[0.0, 1.0], [1.0, 0.0]]), n_iter=2, xi=0.3, dt=0.1, backend='ascend-float32', device_id=DEVICE
    )
    expected = reference(near_symmetric, np.zeros((2, 1)), x, y, [0.0, 1.0], algorithm, 1.0, 0.3, 0.1)
    for _ in range(2):
        solver.J = near_symmetric.copy()
        solver.x, solver.y = x.copy(), y.copy()
        solver.update()
        for got, want in zip((solver.x, solver.y), expected):
            np.testing.assert_allclose(got, want, atol=1e-7, rtol=1e-6)
        solver.J = csr_matrix(near_symmetric)
        with pytest.raises(ValueError, match='symmetric'):
            solver.update()


@pytest.mark.parametrize('solver_class', [BSB, DSB])
def test_qaia_g43_quality(solver_class):
    """
    Feature: End-to-end solution quality of BSB and DSB on the G43 benchmark graph.
    Description: Run BSB and DSB on G43 with and without an external field using the Ascend FP32 backend.
    Expectation: Mean cut value and mean Ising energy match the reference benchmark within 5% relative tolerance.
    """
    # Read the fixture, not another test module with import-time I/O and RNG
    # changes. Keep the established G43 benchmark's exact initial conditions.
    path = Path(__file__).resolve().parents[2] / 'G43.txt'
    n, edges = np.loadtxt(path, max_rows=1, dtype=int)
    row, col, weight = np.loadtxt(path, skiprows=1, dtype=int).T
    assert len(weight) == edges
    graph = coo_matrix((-weight, (row - 1, col - 1)), shape=(n, n)).tocsr()
    graph = graph + graph.T
    rng = np.random.RandomState(666)
    h = rng.rand(n, 1)
    x = 0.01 * (rng.rand(n, 5) - 0.5)
    # SB.initialize generates y from the global RNG; set it explicitly instead.
    y = 0.02 * (np.random.RandomState(666).rand(n, 5) - 0.5)

    for field in (None, h):
        solver = solver_class(
            graph, h=field, x=x.copy(), n_iter=300, batch_size=5, backend='ascend-float32', device_id=DEVICE
        )
        solver.y = y.copy()
        solver.update()
        if field is None:
            assert np.mean(solver.calc_cut()) == pytest.approx(6660.0, rel=0.05)
        else:
            assert np.mean(solver.calc_energy()) == pytest.approx(-3347.0, rel=0.05)
        del solver


def test_context_isolation_with_simulator_and_failed_construction():
    """
    Feature: ACL context isolation between BSB, mqvector_npu Simulator, and caller contexts.
    Description: Interleave BSB updates, Simulator circuit evolution, and failed native plan construction.
    Expectation: The caller's active ACL context is preserved across all operations and error paths.
    """
    from mindquantum.core.circuit import Circuit
    from mindquantum.simulator import Simulator

    # Bootstrap ACL, then install a caller-owned context as torch_npu would.
    bootstrap = BSB(np.zeros((1, 1)), xi=1, n_iter=1, backend='ascend-float32', device_id=DEVICE)
    bootstrap.update()
    del bootstrap
    acl = ctypes.CDLL('libascendcl.so')
    pointer = ctypes.c_void_p
    acl.aclrtCreateContext.argtypes = [ctypes.POINTER(pointer), ctypes.c_int]
    acl.aclrtGetCurrentContext.argtypes = [ctypes.POINTER(pointer)]
    acl.aclrtDestroyContext.argtypes = [pointer]
    foreign = pointer()
    assert acl.aclrtCreateContext(ctypes.byref(foreign), DEVICE) == 0

    def check_owner():
        current = pointer()
        assert acl.aclrtGetCurrentContext(ctypes.byref(current)) == 0
        assert current.value == foreign.value

    solver = simulator = None
    try:
        simulator = Simulator('mqvector_npu', 2, device_id=DEVICE)
        check_owner()
        solver = BSB(np.zeros((3, 3)), xi=1, n_iter=2, backend='ascend-float32', device_id=DEVICE)
        check_owner()
        solver.x, solver.y = np.zeros((3, 1)), np.full((3, 1), 0.2)
        solver.update()
        check_owner()
        simulator.apply_circuit(Circuit().h(0).x(1, 0))
        np.testing.assert_allclose(simulator.get_qs(), [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-6)
        check_owner()
        solver = None
        gc.collect()
        check_owner()
        with pytest.raises(RuntimeError):
            invalid = BSB(np.zeros((3, 3)), xi=1, n_iter=1, backend='ascend-float32', device_id=2**30)
            invalid.update()
        check_owner()
        # A reported QAIA error must not poison the caller's next native launch.
        simulator.reset()
        np.testing.assert_array_equal(simulator.get_qs(), [1, 0, 0, 0])
        check_owner()
    finally:
        solver = simulator = None
        gc.collect()
        try:
            check_owner()
        finally:
            assert acl.aclrtDestroyContext(foreign) == 0


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
@pytest.mark.parametrize('parameter', ['delta', 'xi', 'dt'])
def test_native_release_build_rejects_nonfinite_parameters(value, parameter):
    """
    Feature: Native extension boundary validation of nonfinite integration parameters.
    Description: Construct _qaia_sb_ascend.Plan directly with NaN or infinite delta, xi, or dt.
    Expectation: Raise ValueError even in release builds where assertions are disabled.
    """
    from mindquantum._qaia_sb_ascend import Plan

    parameters = {"delta": 1.0, "xi": 0.5, "dt": 0.1}
    parameters[parameter] = value
    with pytest.raises(ValueError, match='integration parameters'):
        Plan(
            np.eye(64, dtype=np.float32),
            np.zeros(64, np.float32),
            np.zeros(1, np.float32),
            1,
            True,
            **parameters,
            device_id=DEVICE,
        )
