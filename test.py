import numpy as np
import sys
import os
import time

from scipy.sparse import coo_matrix
from mindquantum.algorithm.qaia import DSB


try:
    import kernel_sb
except ImportError:
    print("Module 'kernel_sb' not found. Ensure the .so file is in the python path.")
    sys.exit(1)


def test():
    N = 64
    BATCH_SIZE = 2
    print(f"\n--- Running test (N={N}, Batch={BATCH_SIZE}) ---")

    J = np.random.uniform(-0.1, 0.1, (N, N)).astype(np.float32)

    x = 0.02 * (np.random.rand(N, BATCH_SIZE) - 0.5).astype(np.float32)
    x = np.asfortranarray(x)  # Ensure column-major for kernel (N contiguous)

    y = np.random.uniform(-0.5, 0.5, (N, BATCH_SIZE)).astype(np.float32)
    y = np.asfortranarray(y)

    print("Initial x[:10, 0]:", x[:10, 0])
    print(f"x\n{x}\ny\n{y}")

    delta = 1.0
    xi = 1.0
    dt = 0.1
    niter = 100

    p = np.linspace(0, 1, niter).astype(np.float32)

    print("Running simulation (test)...")
    t1 = time.time()
    x, y = kernel_sb.kernel_dsb(
        J,
        x=x,
        y=y,
        delta=delta,
        xi=xi,
        dt=dt,
        p=p,
        n_iter=niter,
        batch_size=BATCH_SIZE,
    )
    t2 = time.time()
    print(f"DSB Done in {t2-t1:.4f}s")

    print("Final x[:10, 0]:", x[:10, 0])
    print("Final y[:10, 0]:", y[:10, 0])

    t1 = time.time()
    x, y = kernel_sb.kernel_bsb(
        J,
        x=x,
        y=y,
        delta=delta,
        xi=xi,
        dt=dt,
        p=p,
        n_iter=niter,
        batch_size=BATCH_SIZE,
    )
    t2 = time.time()
    print(f"BSB Done in {t2-t1:.4f}s")

    print("Final x[:10, 0]:", x[:10, 0])
    print("Final y[:10, 0]:", y[:10, 0])

    print("Success (test)!")


def generate_random_matrix(size=1024):
    if coo_matrix is None:
        raise RuntimeError("scipy is required for generate_random_matrix (missing scipy.sparse)")
    np.random.seed(1024)
    matrix = np.triu(np.random.choice([-1, 1], size=(size, size)), 1)
    symmetric_matrix = matrix + matrix.T - np.diag(np.diag(matrix))
    sym_matrix = coo_matrix(symmetric_matrix)

    return sym_matrix


def compare_dsb():
    N = 16
    print("\n--- Running compare_dsb ---")
    J = generate_random_matrix(N)

    # Prepare DSB solver (CPU/MindSpore)
    solver = DSB(J, xi=1, batch_size=1, n_iter=1000)
    x_init = solver.x.copy()
    y_init = solver.y.copy()
    p = solver.p.copy()
    print(f"solver.x shape: {solver.x.shape}\n{x_init}")
    print(f"y init:\n{y_init}")

    target_N = N

    J_padded = np.zeros((target_N, target_N), dtype=np.float32)

    J_dense = J.toarray().astype(np.float32)
    rows, cols = J_dense.shape
    J_padded[:rows, :cols] = J_dense

    print(f"J:{J}")
    print(f"J_padded:\n{J_padded}")

    # NPU Params
    delta = 1.0
    xi = 1.0
    dt = 1.0
    # p = 0.1
    niter = 1000

    print("Launching NPU solve...")
    t3 = time.time()
    kernel_sb.solve(J_padded, x_init, y_init, delta, xi, dt, p, niter)
    t4 = time.time()
    print(f"NPU solve done in {t4-t3:.4f}s")

    # Run CPU solver for comparison
    print("Running CPU solver...")
    t1 = time.time()
    solver.update()
    t2 = time.time()

    cut_cpu = solver.calc_cut()
    energy_cpu = solver.calc_energy()

    # NPU results evaluation
    x_npu_reshaped = np.sign(x_init)

    cut_npu = solver.calc_cut(x_npu_reshaped)
    energy_npu = solver.calc_energy(x_npu_reshaped)

    print(f"CPU:\nCut: {cut_cpu}\nEnergy: {energy_cpu}")
    print(f"NPU:\nCut: {cut_npu}\nEnergy: {energy_npu}")
    print(f"CPU use time:{t2-t1}\nNPU use time:{t4-t3}")


if __name__ == "__main__":
    test()
    # compare_dsb()
