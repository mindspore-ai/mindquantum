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
"""Test QAIA algorithm."""
# pylint: disable=invalid-name
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix

from mindquantum.algorithm.qaia import ASB, BSB, CAC, CFC, DSB, LQA, SFC, NMFA, SimCIM
from mindquantum.utils.fdopen import fdopen

import pytest

OPTIMAL_CUT_G43 = 6660.0
TOLERANCE = 0.05  # 5% relative tolerance for cut value
BENCHMARK_ENERGY_H = -3347.0  # Benchmark energy for scaled h, based on DSB
ENERGY_TOLERANCE = 0.05  # Uniform 5% relative tolerance for energy with h
N_ITER = 300  # Global default iteration count
BATCH_SIZE = 5  # Global default batch size
SEED = 666


def read_gset(filename, negate=True):
    """
    Reading Gset and transform it into sparse matrix

    Args:
        filename (str): The path and file name of the data.
        negate (bool): whether negate the weight of matrix or not.  Default: ``True``.

    Returns:
        coo_matrix, matrix representation of graph.

    Examples:
        >>> from qaia.utils import read
        >>> G = read.read_gset('data/Gset/G1.txt')
    """
    with fdopen(filename, "r") as f:
        data = f.readlines()

    n_v, n_e = (int(i) for i in data[0].strip().split(" "))
    graph = np.array([[int(i) for i in j.strip().split(" ")] for j in data[1:]])
    if n_e != graph.shape[0]:
        raise ValueError(f"The number of edges is not matched, {n_e} != {graph.shape[0]}")
    # Convert to CSR format which is preferred by QAIA CPU backend
    out = coo_matrix(
        (
            np.concatenate([graph[:, -1], graph[:, -1]]),
            (
                np.concatenate([graph[:, 0] - 1, graph[:, 1] - 1]),
                np.concatenate([graph[:, 1] - 1, graph[:, 0] - 1]),
            ),
        ),
        shape=(n_v, n_v),
    ).tocsr()

    if negate:
        return -out

    return out


G = read_gset(str(Path(__file__).parent.parent.parent / 'G43.txt'))
N = G.shape[0]
np.random.seed(SEED)  # Ensure h_test is reproducible
h_test = np.random.rand(N, 1)  # Scaled h
x_init_test = 0.01 * (np.random.rand(N, BATCH_SIZE) - 0.5)


# Helper function to run and check algorithm
def run_and_check(
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
    **kwargs,
):
    """Runs QAIA solver and checks cut value and energy with h."""
    # Test without h (check cut value)
    np.random.seed(seed)
    solver = solver_class(G_mat, x=x_init.copy(), n_iter=n_iter, batch_size=batch_size, **kwargs)
    solver.update()
    avg_cut = np.mean(solver.calc_cut())
    print(f"{solver_class.__name__} Avg Cut: {avg_cut:.2f}, Optimal Cut: {optimal_cut}")
    assert np.abs(avg_cut - optimal_cut) / np.abs(optimal_cut) < cut_tolerance

    # Test with external field h (check energy value)
    if h_vec is not None:
        np.random.seed(seed)
        solver_h = solver_class(G_mat, h=h_vec, x=x_init.copy(), n_iter=n_iter, batch_size=batch_size, **kwargs)
        solver_h.update()
        energy_h = solver_h.calc_energy()
        assert energy_h is not None
        if energy_h.ndim > 1:
            energy_h = energy_h.flatten()
        avg_energy_h = np.mean(energy_h)
        print(f"{solver_class.__name__} with h Avg Energy: {avg_energy_h:.2f}, Benchmark Energy: {benchmark_energy}")
        assert np.abs(avg_energy_h - benchmark_energy) / np.abs(benchmark_energy) < energy_tolerance


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_aSB():
    """
    Description: Test ASB CPU
    Expectation: success
    """
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_bSB():
    """
    Description: Test BSB CPU
    Expectation: success
    """
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dSB():
    """
    Description: Test DSB CPU
    Expectation: success
    """
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_LQA():
    """
    Description: Test LQA CPU
    Expectation: success
    """
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_CAC():
    """
    Description: Test CAC CPU
    Expectation: success
    """
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_CFC():
    """
    Description: Test CFC CPU
    Expectation: success
    """
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_SFC():
    """
    Description: Test SFC CPU (specific N_ITER)
    Expectation: success
    """
    sfc_n_iter = 1000
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_NMFA():
    """
    Description: Test NMFA CPU (specific N_ITER)
    Expectation: success
    """
    nmfa_n_iter = 1000
    run_and_check(
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
    )


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_SimCIM():
    """
    Description: Test SimCIM CPU (specific N_ITER)
    Expectation: success
    """
    simcim_n_iter = 5000
    simcim_energy_tolerance = 0.1
    run_and_check(
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
    )
