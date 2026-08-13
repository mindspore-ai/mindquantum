import argparse
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix

from mindquantum.algorithm.qaia import DSB

def test():
    try:
        import kernel_sb
    except ImportError:
        print(
            "Module 'kernel_sb' not found. Ensure the .so file is in the python path."
        )
        sys.exit(1)

    N = 16
    BATCH_SIZE = 2

    J = np.random.uniform(-0.1, 0.1, (N, N)).astype(np.float32)

    x = 0.02 * (np.random.rand(N, BATCH_SIZE) - 0.5).astype(np.float32)
    x = np.asfortranarray(x)  # Ensure column-major for kernel (N contiguous)

    y = np.random.uniform(-0.5, 0.5, (N, BATCH_SIZE)).astype(np.float32)
    y = np.asfortranarray(y)

    delta = 1.0
    xi = 1.0
    dt = 0.1
    niter = 100

    p = np.linspace(0, 1, niter).astype(np.float32)

    print("Running simulation (test)...")
    kernel_sb.solve(J, x, y, delta, xi, dt, p, niter)

    print("Success (test)!")


def _import_kernel_sb():
    try:
        import kernel_sb  # type: ignore
    except ImportError:
        print("Module 'kernel_sb' not found. Ensure the .so file is in the python path.")
        sys.exit(1)
    return kernel_sb


def read_gset(filename: str, *, negate: bool = True, dtype=np.float32):
    arr = np.fromfile(filename, sep=" ", dtype=np.float32)
    if arr.size < 2:
        raise ValueError(f"Invalid GSET file: {filename}")

    n_v = int(arr[0])
    n_e = int(arr[1])
    triples = arr[2:]
    if triples.size % 3 != 0:
        raise ValueError(f"Invalid GSET file (edge triples not multiple of 3): {filename}")

    edges = triples.reshape(-1, 3)
    if edges.shape[0] != n_e:
        raise ValueError(f"Edge count mismatch: header={n_e}, parsed={edges.shape[0]}")

    src = edges[:, 0].astype(np.int64) - 1
    dst = edges[:, 1].astype(np.int64) - 1
    w = edges[:, 2].astype(dtype)
    if negate:
        w = -w

    data = np.concatenate([w, w])
    row = np.concatenate([src, dst])
    col = np.concatenate([dst, src])
    return coo_matrix((data, (row, col)), shape=(n_v, n_v), dtype=dtype)


def _run_cpu(*, J, x0, y0, p, xi: float, dt: float, niter: int):
    solver = DSB(J, xi=xi, batch_size=x0.shape[1], n_iter=niter, dt=dt)
    solver.x = x0.copy(order="F")
    solver.y = y0.copy(order="F")
    solver.p = p.copy()
    t1 = time.perf_counter()
    solver.update()
    t2 = time.perf_counter()
    cut = solver.calc_cut()
    energy = solver.calc_energy()
    return {"time_s": t2 - t1, "cut_max": float(np.max(cut)), "energy_min": float(np.min(energy)), "x": solver.x}


def _run_npu(*, J_dense, x0, y0, p, xi: float, dt: float, niter: int,  warmup: int):
    delta = 1.0
    kernel_sb = _import_kernel_sb()
    if warmup > 0:
        x_w = x0.copy(order="F")
        y_w = y0.copy(order="F")
        p_w = p[:warmup].copy()
        if hasattr(kernel_sb, "solve_cached"):
            kernel_sb.solve_cached(J_dense, x_w, y_w, delta, xi, dt, p_w, warmup, True)
        else:
            kernel_sb.solve(J_dense, x_w, y_w, delta, xi, dt, p_w, warmup)

    x = x0.copy(order="F")
    y = y0.copy(order="F")
    t1 = time.perf_counter()
    if hasattr(kernel_sb, "solve_cached"):
        kernel_sb.solve_cached(J_dense, x, y, delta, xi, dt, p, niter, False)
    else:
        kernel_sb.solve(J_dense, x, y, delta, xi, dt, p, niter)
    t2 = time.perf_counter()
    return {"time_s": t2 - t1, "x": x}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="./data/gset/G22", help="Path to GSET graph file (default: G22)")
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--niter", type=int, default=1000)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--dtype", choices=["fp32", "fp16"], default="fp32")
    ap.add_argument("--npu-id", type=int, default=None, help="NPU card id for power monitoring (default: auto-detect)")
    ap.add_argument("--no-power", action="store_true", help="Disable NPU power monitoring")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup iterations for NPU (default: 5)")
    args = ap.parse_args()

    dtype = np.float32 if args.dtype == "fp32" else np.float16
    G = read_gset(args.graph, dtype=np.float32)
    n_v = G.shape[0]
    print(f"Loaded {args.graph}: N={n_v}, nnz={G.nnz}")

    solver_init = DSB(G, batch_size=args.batch, n_iter=args.niter, dt=args.dt)
    xi = float(solver_init.xi)
    x0 = np.asfortranarray(solver_init.x.astype(dtype, copy=False))
    y0 = np.asfortranarray(solver_init.y.astype(dtype, copy=False))
    p = np.asarray(solver_init.p, dtype=dtype)

    J_dense = np.asarray(G.toarray(), dtype=dtype, order="C")

    cpu_res = _run_cpu(
        J=G,
        x0=x0.astype(np.float32),
        y0=y0.astype(np.float32),
        p=p.astype(np.float32),
        xi=xi,
        dt=args.dt,
        niter=args.niter,
    )
    npu_res = _run_npu(
        J_dense=J_dense,
        x0=x0,
        y0=y0,
        p=p,
        xi=xi,
        dt=args.dt,
        niter=args.niter,
        warmup=max(0, min(args.warmup, args.niter)),
    )

    solver_eval = DSB(G, xi=xi, batch_size=args.batch, n_iter=args.niter, dt=args.dt)
    cut_npu = solver_eval.calc_cut(npu_res["x"])
    energy_npu = solver_eval.calc_energy(npu_res["x"])

    print("\n=== Results ===")
    print(
        f"CPU(fp32):  time={cpu_res['time_s']:.4f}s  maxCut={cpu_res['cut_max']:.6f}  minEnergy={cpu_res['energy_min']:.6f}"
    )
    print(
        f"NPU({args.dtype}):  time={npu_res['time_s']:.4f}s  maxCut={float(np.max(cut_npu)):.6f}  minEnergy={float(np.min(energy_npu)):.6f}"
    )


if __name__ == "__main__":
    test()
    main()
