# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Test QAIA algorithm NPU backend."""
# pylint: disable=invalid-name
from pathlib import Path
import subprocess

import numpy as np
from scipy.sparse import coo_matrix

from mindquantum.algorithm.qaia import ASB, BSB, CAC, CFC, DSB, LQA, SFC, NMFA, SimCIM
from mindquantum.utils.fdopen import fdopen

import pytest

OPTIMAL_CUT_G43 = 6660.0
TOLERANCE = 0.05  # 5% relative tolerance for cut value
BENCHMARK_ENERGY_H = -3347.0  # Benchmark energy for scaled h
ENERGY_TOLERANCE = 0.05  # Uniform 5% relative tolerance for energy with h
N_ITER = 300  # Global default iteration count
BATCH_SIZE = 5  # Global default batch size
SEED = 666


try:
    subprocess.check_output('npu-smi info', shell=True)
    _HAS_NPU = True
except (FileNotFoundError, subprocess.CalledProcessError):
    _HAS_NPU = False

torch = pytest.importorskip("torch", reason="This test case require torch")


def read_gset(filename, negate=True):
    """
    Reading Gset and transform it into sparse matrix

    Args:
        filename (str): The path and file name of the data.
        negate (bool): whether negate the weight of matrix or not.  Default: ``True``.

    Returns:
        coo_matrix, matrix representation of graph.
    """
    with fdopen(filename, "r") as f:
        data = f.readlines()

    n_v, n_e = (int(i) for i in data[0].strip().split(" "))
    graph = np.array([[int(i) for i in j.strip().split(" ")] for j in data[1:]])
    if n_e != graph.shape[0]:
        raise ValueError(f"The number of edges is not matched, {n_e} != {graph.shape[0]}")
    # Keep COO format for PyTorch sparse tensor conversion
    out = coo_matrix(
        (
            np.concatenate([graph[:, -1], graph[:, -1]]),
            (
                np.concatenate([graph[:, 0] - 1, graph[:, 1] - 1]),
                np.concatenate([graph[:, 1] - 1, graph[:, 0] - 1]),
            ),
        ),
        shape=(n_v, n_v),
    )

    if negate:
        return -out

    return out


G = read_gset(str(Path(__file__).parent.parent.parent / 'G43.txt'))
N = G.shape[0]
np.random.seed(SEED)
h_test = np.random.rand(N, 1)  # Scaled h
x_init_test = 0.01 * (np.random.rand(N, BATCH_SIZE) - 0.5)


# Helper function to run and check NPU algorithm
def run_and_check_npu(
    solver_class,
    G_mat,
    h_vec,
    x_init,
    n_iter,
    batch_size,
    optimal_cut,
    cut_tolerance,
    benchmark_energy,
    energy_tolerance,
    seed,
    backend,
    **kwargs,
):
    """Runs QAIA NPU solver and checks cut value and energy with h."""
    # Test without h (check cut value)
    np.random.seed(seed)
    solver = solver_class(G_mat, x=x_init.copy(), n_iter=n_iter, batch_size=batch_size, backend=backend, **kwargs)
    solver.update()
    cuts = solver.calc_cut()
    if hasattr(cuts, 'tolist'):
        cuts_np = np.array(cuts.tolist())
    elif hasattr(cuts, 'cpu'):
        cuts_np = cuts.cpu().numpy()
    else:
        cuts_np = np.array(cuts)

    avg_cut = np.mean(cuts_np)
    print(f"{solver_class.__name__} ({backend}) Avg Cut: {avg_cut:.2f}, Optimal Cut: {optimal_cut}")
    assert np.abs(avg_cut - optimal_cut) / np.abs(optimal_cut) < cut_tolerance

    # Test with external field h (check energy value)
    if h_vec is not None:
        np.random.seed(seed)
        solver_h = solver_class(
            G_mat, h=h_vec, x=x_init.copy(), n_iter=n_iter, batch_size=batch_size, backend=backend, **kwargs
        )
        solver_h.update()
        energy_h = solver_h.calc_energy()
        assert energy_h is not None

        if hasattr(energy_h, 'tolist'):
            energy_h_np = np.array(energy_h.tolist())
        elif hasattr(energy_h, 'cpu'):
            energy_h_np = energy_h.cpu().numpy()
        else:
            energy_h_np = np.array(energy_h)

        if energy_h_np.ndim > 1:
            energy_h_np = energy_h_np.flatten()

        avg_energy_h = np.mean(energy_h_np)
        print(
            f"{solver_class.__name__} ({backend}) with h Avg Energy: {avg_energy_h:.2f}, "
            f"Benchmark Energy: {benchmark_energy}"
        )
        assert np.abs(avg_energy_h - benchmark_energy) / np.abs(benchmark_energy) < energy_tolerance


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_aSB_npu():
    """
    Description: Test ASB NPU
    Expectation: success
    """
    run_and_check_npu(
        ASB,
        G,
        h_test,
        x_init_test,
        N_ITER,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_bSB_npu():
    """
    Description: Test BSB NPU
    Expectation: success
    """
    run_and_check_npu(
        BSB,
        G,
        h_test,
        x_init_test,
        N_ITER,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_dSB_npu():
    """
    Description: Test DSB NPU
    Expectation: success
    """
    run_and_check_npu(
        DSB,
        G,
        h_test,
        x_init_test,
        N_ITER,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_LQA_npu():
    """
    Description: Test LQA NPU
    Expectation: success
    """
    run_and_check_npu(
        LQA,
        G,
        h_test,
        x_init_test,
        N_ITER,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_CAC_npu():
    """
    Description: Test CAC NPU
    Expectation: success
    """
    run_and_check_npu(
        CAC,
        G,
        h_test,
        x_init_test,
        N_ITER,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_CFC_npu():
    """
    Description: Test CFC NPU
    Expectation: success
    """
    run_and_check_npu(
        CFC,
        G,
        h_test,
        x_init_test,
        N_ITER,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_SFC_npu():
    """
    Description: Test SFC NPU (specific N_ITER)
    Expectation: success
    """
    sfc_n_iter = 1000
    run_and_check_npu(
        SFC,
        G,
        h_test,
        x_init_test,
        sfc_n_iter,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_NMFA_npu():
    """
    Description: Test NMFA NPU (specific N_ITER)
    Expectation: success
    """
    nmfa_n_iter = 1000
    run_and_check_npu(
        NMFA,
        G,
        h_test,
        x_init_test,
        nmfa_n_iter,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        ENERGY_TOLERANCE,
        SEED,
        'npu-float32',
    )


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_NPU, reason='Machine does not has Ascend NPU.')
def test_SimCIM_npu():
    """
    Description: Test SimCIM NPU (specific N_ITER)
    Expectation: success
    """
    simcim_n_iter = 5000
    simcim_energy_tolerance = 0.1
    run_and_check_npu(
        SimCIM,
        G,
        h_test,
        x_init_test,
        simcim_n_iter,
        BATCH_SIZE,
        OPTIMAL_CUT_G43,
        TOLERANCE,
        BENCHMARK_ENERGY_H,
        simcim_energy_tolerance,
        SEED,
        'npu-float32',
    )
