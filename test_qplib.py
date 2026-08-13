import numpy as np
import time
import sys
from scipy.sparse import csr_matrix

from mindquantum.algorithm.qaia import DSB

try:
    import kernel_sb
    print(kernel_sb.__file__)
except ImportError:
    print(
            "Module 'kernel_sb' not found. Ensure the .so file is in the python path."
        )
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

    delta = 1.0
    xi = 1.0
    dt = 0.1
    niter = 100

    p = np.linspace(0, 1, niter).astype(np.float32)

    print("Running simulation (test)...")
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

    print("Success (test)!")


def qplib_translate(data_name: None):

    data = None
    with open(data_name, "r") as file:
        data = file.read()

    data = data.split("\n")
    Node = int(data[3].split()[0])
    Edge = int(data[4].split()[0])
    Jdata = [
        [int(i.split()[0]), int(i.split()[1]), float(i.split()[2])]
        for i in data[5 : 5 + Edge]
    ]
    hvalue = float(data[5 + Edge].split()[0])
    hnum = int(data[5 + Edge + 1].split()[0])
    hdata = [
        [int(i.split()[0]), float(i.split()[1])]
        for i in data[7 + Edge : 7 + Edge + hnum]
    ]

    J_info = {}
    h_info = {}

    for j in Jdata:
        J_info[f"({j[0]},{j[1]})"] = j[2] / 2
    for hi in hdata:
        h_info[f"{hi[0]-1}"] = hi[1]

    edge_list = []
    edge_weight = []

    for key, value in J_info.items():
        edge_list.append(list(eval(key)))
        edge_weight.append(value)
    edge_list = np.array(edge_list)
    edge_weight = np.array(edge_weight)
    n_vertex = np.max(edge_list)

    J = csr_matrix(
        (
            np.concatenate([edge_weight, edge_weight]),
            (
                np.concatenate([edge_list[:, 0] - 1, edge_list[:, 1] - 1]),
                np.concatenate([edge_list[:, 1] - 1, edge_list[:, 0] - 1]),
            ),
        ),
        shape=(n_vertex, n_vertex),
    )

    if h_info != {}:
        h = np.zeros((n_vertex, 1))
        h[:] = hvalue
        h[np.array(list(h_info.keys())).astype(np.int64), 0] = np.array(
            list(h_info.values())
        )

    J = J / 8
    offset = -J.sum() - h.sum() / 2
    h = h / 2 + 2 * J.sum(axis=1)
    J = -2 * J
    h = -np.asarray(h)
    scale_factor = np.max([np.abs(J).max(), np.abs(h).max()])
    input_J = -J / scale_factor
    input_h = -h / scale_factor

    return input_J, input_h, J, h, offset


if __name__ == "__main__":
    test()
    for qpname in [3506,3565,3642,3650,3693,3705,3706,3738,3745,3822,3832,3838,3850,3852,3877,5721,5725,5755,5875,5881,5882,5909,5922]:
        # test()
        data_name = f"./data/QPLIB_{qpname}.qplib"
        j_value, h_value, J, h, offset = qplib_translate(data_name)

        batch_size = 100
        n_iter = 1000

        j_value_f32 = j_value.astype(np.float32)
        h_value_f32 = np.asarray(h_value, dtype=np.float32)
        solver = DSB(J=j_value_f32, h=h_value_f32, batch_size=batch_size, n_iter=n_iter)

        x0 = solver.x.astype(np.float32, copy=True)
        y0 = solver.y.astype(np.float32, copy=True)
        p = solver.p.astype(np.float32, copy=True)

        J_dense = np.asarray(j_value_f32.toarray(), dtype=np.float32, order="C")
        h_dense = np.asarray(h_value_f32, dtype=np.float32).reshape((-1, 1))

        x_npu = np.asfortranarray(x0.copy())
        y_npu = np.asfortranarray(y0.copy())
        x_cpu = np.asfortranarray(x0.copy())
        y_cpu = np.asfortranarray(y0.copy())

        t_npu0 = time.time()
        x_npu, y_npu = kernel_sb.kernel_dsb(
            J_dense,
            h=h_dense,
            n_iter=n_iter,
            batch_size=batch_size,
        )
        t_npu1 = time.time()

        t_cpu0 = time.time()
        solver = DSB(J=j_value_f32, h=h_value_f32, batch_size=batch_size, n_iter=n_iter)
        solver.update()
        x_cpu=solver.x
        t_cpu1 = time.time()

        solution_cpu = np.sign(x_cpu)
        solution_npu = np.sign(x_npu)

        energy_cpu = (
            -(
                (J.dot(solution_cpu) * solution_cpu).sum(axis=0, keepdims=True) / 2
                + h.T.dot(solution_cpu)
            )
        ) - offset
        energy_npu = (
            -(
                (J.dot(solution_npu) * solution_npu).sum(axis=0, keepdims=True) / 2
                + h.T.dot(solution_npu)
            )
        ) - offset

        print(f"{qpname}.qplib")
        print(f"CPU best energy: {float(np.max(energy_cpu))}")
        print(f"NPU best energy: {float(np.max(energy_npu))}")
        print(f"|CPU-NPU| best energy: {float(np.abs(np.max(energy_cpu) - np.max(energy_npu)))}")
        print(f"CPU time: {t_cpu1 - t_cpu0:.4f}s, NPU time: {t_npu1 - t_npu0:.4f}s")

