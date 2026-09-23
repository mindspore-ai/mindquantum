"""Reproducible solve-level benchmarks; JSON lines go to stdout.

Setup (validation, upload, operator warmup, graph capture), synchronized run
wall time (including state transfers), and device event time are separate.
CPU uses NumPy/SciPy FP32 with exactly the same equations, inputs and schedule.
Final heuristic quality is measured in the original objective, not by assuming
that chaotic long trajectories must match bit-for-bit across GEMM reductions.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import scipy
from scipy.sparse import csr_matrix

from kernel_sb import Solver


def arguments(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--algorithm", nargs="+", choices=["dsb", "bsb"], default=["dsb", "bsb"])
    parser.add_argument("--precision", nargs="+", choices=["fp32", "hf32"], default=["fp32"])
    parser.add_argument("--batch", type=int, default=100)
    parser.add_argument("--niter", type=int, default=1000)
    parser.add_argument("--repeat", type=int, default=3, help="independent, matched seeds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--device-id", type=int, default=0, help="ACL logical device ID")
    parser.add_argument("--cpu", choices=["sparse", "dense", "none"], default="sparse")
    return parser


def cpu_solve(J, h, x, y, p, algorithm, xi, dt):
    x, y = x.copy(), y.copy()
    for pressure in p:
        coupling = J @ (np.sign(x) if algorithm == "dsb" else x)
        y += dt * ((pressure - np.float32(1)) * x + xi * (coupling + h))
        x += dt * y
        outside = np.abs(x) > 1
        y[outside] = 0
        np.clip(x, -1, 1, out=x)
    return x, y


def benchmark(name, J, h, objective, sense, args):
    if min(args.batch, args.niter, args.repeat) <= 0:
        raise ValueError("batch, niter and repeat must be positive")
    J = np.asarray(J, dtype=np.float32, order="C")
    h = np.asarray(h, dtype=np.float32).reshape(-1, 1)
    n = J.shape[0]
    norm = np.sqrt(np.einsum("ij,ij->", J, J, dtype=np.float64))
    if norm == 0:
        raise ValueError("benchmark requires a nonzero coupling matrix")
    xi = np.float32(0.5 * np.sqrt(n - 1) / norm)
    p = np.linspace(0, 1, args.niter, dtype=np.float32)
    dt = np.float32(args.dt)
    starts = []
    for repeat in range(args.repeat):
        rng = np.random.default_rng(args.seed + repeat)
        starts.append(tuple(np.asfortranarray(rng.uniform(-.01, .01, (n, args.batch)).astype(np.float32))
                            for _ in range(2)))
    cpu_J = csr_matrix(J) if args.cpu == "sparse" else J

    def scores(x, y):
        if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any(np.abs(x) > 1):
            raise AssertionError("solver returned nonfinite or out-of-bounds state")
        # Tie at zero maps to binary 1 (spin +1), never an invalid third state.
        values = np.asarray(objective((x >= 0).astype(np.float64)))
        return float(values.max() if sense == "maximize" else values.min())

    rows = []
    for algorithm in args.algorithm:
        cpu_times, cpu_scores = [], []
        if args.cpu != "none":
            cpu_solve(cpu_J, h, *starts[0], p[:1], algorithm, xi, dt)
            for x, y in starts:
                begin = time.perf_counter()
                result = cpu_solve(cpu_J, h, x, y, p, algorithm, xi, dt)
                cpu_times.append((time.perf_counter() - begin) * 1000)
                cpu_scores.append(scores(*result))
        for precision in args.precision:
            begin = time.perf_counter()
            with Solver(J, h, algorithm=algorithm, precision=precision, batch_size=args.batch,
                        n_iter=args.niter, p=p, xi=xi, dt=dt, device_id=args.device_id) as solver:
                setup_ms = (time.perf_counter() - begin) * 1000
                begin = time.perf_counter()
                solver.run(*starts[0])
                first_run_ms = (time.perf_counter() - begin) * 1000
                wall, device, quality = [], [], []
                for x, y in starts:
                    begin = time.perf_counter()
                    result = solver.run(x, y)
                    wall.append((time.perf_counter() - begin) * 1000)
                    device.append(solver.last_device_ms)
                    quality.append(scores(*result))
            row = dict(instance=name, n=n, batch=args.batch, niter=args.niter, seed=args.seed,
                       algorithm=algorithm, precision=precision, device_id=args.device_id,
                       delta=1.0, xi=float(xi), dt=float(dt),
                       environment=dict(python=platform.python_version(), numpy=np.__version__,
                                        scipy=scipy.__version__, cann_home=os.getenv("ASCEND_HOME_PATH"),
                                        openblas_threads=os.getenv("OPENBLAS_NUM_THREADS")),
                       setup_ms=setup_ms, first_run_ms=first_run_ms, wall_ms=wall, device_ms=device,
                       wall_median_ms=float(np.median(wall)), device_median_ms=float(np.median(device)),
                       objective_sense=sense, best_objectives=quality,
                       cpu_backend=args.cpu, cpu_ms=cpu_times, cpu_best_objectives=cpu_scores)
            if cpu_times:
                row["cpu_speedup"] = float(np.median(cpu_times) / np.median(wall))
            print(json.dumps(row, allow_nan=False), flush=True)
            rows.append(row)
    return rows
