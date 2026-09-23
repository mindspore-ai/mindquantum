"""Hardware tests. Build first; failures are not replaced with CPU fallbacks."""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pytest

from kernel_sb import Solver, kernel_bsb, kernel_dsb

DEVICE = int(os.environ.get("SB_TEST_DEVICE", "0"))


def reference(J, h, x, y, p, algorithm, delta, xi, dt):
    # FP64 oracle, with an independently expressed wall and update. Deliberately
    # use nonsymmetric J: a reversed GEMM transpose must not pass these tests.
    x, y = x.astype(np.float64), y.astype(np.float64)
    for pressure in p:
        force = J.astype(np.float64) @ (np.sign(x) if algorithm == "dsb" else x)
        velocity = y + dt * ((pressure - delta) * x + xi * (force + h))
        position = x + dt * delta * velocity
        outside = np.abs(position) > 1
        x = np.where(outside, np.sign(position), position)
        y = np.where(outside, 0, velocity)
    return x, y


@pytest.mark.parametrize("algorithm", ["dsb", "bsb"])
@pytest.mark.parametrize("n,batch", [(1, 1), (17, 3), (64, 65), (1023, 3), (1024, 1),
                                       (1025, 193), (2049, 2), (5001, 1)])
def test_trajectory_and_tail(algorithm, n, batch):
    rng = np.random.default_rng(n)
    J = rng.uniform(-0.07, 0.09, (n, n)).astype(np.float32)
    h = rng.uniform(-0.3, 0.5, (n, 1)).astype(np.float32)
    x = rng.uniform(-0.8, 0.7, (n, batch)).astype(np.float32)
    x[::7] = 0
    y = rng.uniform(-0.4, 0.6, (n, batch)).astype(np.float32)
    p = np.array([0.13, 0.27, -0.08, 1.11], np.float32)
    delta, xi, dt = map(np.float32, (0.8, 0.2, 0.13))
    expected = reference(J, h, x, y, p, algorithm, delta, xi, dt)
    with Solver(J, h, algorithm=algorithm, batch_size=batch, n_iter=len(p), p=p,
                delta=delta, xi=xi, dt=dt, device_id=DEVICE) as solver:
        actual = solver.run(x, y)
        assert solver.last_device_ms > 0
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=2e-5, atol=3e-6)
        assert a.dtype == np.float32 and a.flags.f_contiguous


@pytest.mark.parametrize("algorithm", ["dsb", "bsb"])
def test_strict_wall_and_zero_sign(algorithm):
    # Six rows deliberately do not divide the 64-lane mask repeat. Exercise
    # different predicates across repeats, replica boundaries and padded tails.
    x = np.zeros((1032, 6), np.float32)
    y = np.tile(np.array([[2, -2], [1, -1], [.875, -.875], [.25, -.75], [0, 0], [-1.5, 1.5]],
                        np.float32), (172, 3))
    with Solver(np.zeros((1032, 1032), np.float32), algorithm=algorithm, batch_size=6,
                p=[1.0], n_iter=1, xi=1, device_id=DEVICE) as solver:
        actual_x, actual_y = solver.run(x, y)
    np.testing.assert_array_equal(actual_x, np.tile(
        [[1, -1], [1, -1], [.875, -.875], [.25, -.75], [0, 0], [-1, 1]], (172, 3)))
    np.testing.assert_array_equal(actual_y, np.tile(
        [[0, 0], [1, -1], [.875, -.875], [.25, -.75], [0, 0], [0, 0]], (172, 3)))


def test_snapshot_replay_isolation_and_concurrent_calls():
    rng = np.random.default_rng(72)
    J = rng.normal(0, .1, (33, 33)).astype(np.float32)
    h = rng.normal(0, .2, 33).astype(np.float32)
    p = np.linspace(0, 1, 11, dtype=np.float32)
    x = np.asfortranarray(rng.normal(0, .1, (33, 5)).astype(np.float32))
    y = x * -.7
    original = x.copy(), y.copy()
    with Solver(J, h, n_iter=11, p=p, batch_size=5, dt=.1, device_id=DEVICE) as a:
        expected = a.run(x, y)
        J[:] = -3
        h[:] = 12
        p[:] = .81
        with Solver(J, h, n_iter=11, p=p, batch_size=5, dt=.1, device_id=DEVICE) as b:
            other = b.run(x, y)
            assert not np.allclose(other[0], expected[0])
            with ThreadPoolExecutor(max_workers=2) as pool:
                runs = list(pool.map(lambda _: a.run(x, y), range(4)))
            for result in runs:
                for got, want in zip(result, expected):
                    np.testing.assert_array_equal(got, want)
        np.testing.assert_array_equal(a.run(x, y)[0], expected[0])
    for got, want in zip((x, y), original):
        np.testing.assert_array_equal(got, want)
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        a.run(x, y)


@pytest.mark.parametrize("algorithm,one_shot", [("dsb", kernel_dsb), ("bsb", kernel_bsb)])
def test_long_schedule_and_seed(algorithm, one_shot):
    J = np.array([[0, .25, -.5], [.25, 0, .125], [-.5, .125, 0]], np.float32)
    first = one_shot(J, n_iter=1000, batch_size=3, seed=37, device_id=DEVICE)
    second = one_shot(J, n_iter=1000, batch_size=3, seed=37, device_id=DEVICE)
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(a).all()
    assert np.max(np.abs(first[0])) <= 1


@pytest.mark.parametrize("algorithm", ["dsb", "bsb"])
def test_explicit_hf32_mode(algorithm):
    rng = np.random.default_rng(89)
    J = rng.normal(0, .2, (257, 257)).astype(np.float32)
    x = rng.uniform(-.5, .5, (257, 19)).astype(np.float32)
    y = x * .3
    h = np.zeros((257, 1), np.float32)
    expected = reference(J, h, x, y, [.3], algorithm, 1., .2, .1)
    with Solver(J, algorithm=algorithm, precision="hf32", n_iter=1, p=[.3],
                batch_size=19, xi=.2, dt=.1, device_id=DEVICE) as solver:
        actual = solver.run(x, y)
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=2e-3, atol=1e-4)


@pytest.mark.parametrize("kwargs,match", [
    ({"J": np.ones((2, 3))}, "square"),
    ({"J": np.empty((0, 0))}, "square"),
    ({"J": np.array([[np.nan]])}, "finite"),
    ({"J": np.ones((3, 3), complex)}, "real"),
    ({"batch_size": 0}, "batch_size"),
    ({"batch_size": 1.5}, "batch_size"),
    ({"n_iter": 0}, "n_iter"),
    ({"n_iter": 2, "p": [0.]}, "p must"),
    ({"h": np.ones((3, 2))}, "h must"),
    ({"dt": 0}, "positive"),
    ({"xi": np.inf}, "finite"),
    ({"algorithm": "other"}, "algorithm"),
    ({"precision": "fp16"}, "precision"),
    ({"J": np.zeros((3, 3))}, "specify xi"),
])
def test_invalid_problem_fails_before_device_allocation(kwargs, match):
    args = {"J": np.eye(3)} | kwargs
    with pytest.raises(ValueError, match=match):
        Solver(**args)


def test_mismatched_and_nonfinite_states():
    with Solver(np.eye(17), batch_size=3, n_iter=1, device_id=DEVICE) as solver:
        with pytest.raises(ValueError, match="y must have shape"):
            solver.run(np.zeros((17, 3)), np.zeros((17, 2)))
        with pytest.raises(ValueError, match="finite"):
            solver.run(np.full((17, 3), np.nan), np.zeros((17, 3)))


@pytest.mark.parametrize("algorithm", ["dsb", "bsb"])
def test_cpu_benchmark_equations(algorithm):
    from benchmarking import cpu_solve
    from scipy.sparse import csr_matrix

    J = np.array([[0, 2, -.5], [-1, .25, .75], [1, -2, .5]], np.float32)
    h = np.array([[.7], [-.3], [.2]], np.float32)
    x = np.array([[0, .1], [-.25, .8], [.6, -.2]], np.float32)
    y = np.array([[.5, .2], [-.7, .3], [.1, -.4]], np.float32)
    p = np.array([.125, .7, 1.], np.float32)
    expected = reference(J, h, x, y, p, algorithm, 1, .5, .5)
    for matrix in (J, csr_matrix(J)):
        for got, want in zip(cpu_solve(matrix, h, x, y, p, algorithm, np.float32(.5), np.float32(.5)), expected):
            np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-7)
