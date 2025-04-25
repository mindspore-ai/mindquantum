# Copyright 2023 Huawei Technologies Co., Ltd
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
import subprocess

import numpy as np
from scipy.sparse import coo_matrix

from mindquantum.algorithm.qaia import ASB, BSB, CAC, CFC, DSB, LQA, SFC, NMFA, SimCIM
from mindquantum.utils.fdopen import fdopen

import pytest

try:
    subprocess.check_output('nvidia-smi')
    _HAS_GPU = True
except FileNotFoundError:
    _HAS_GPU = False

torch = pytest.importorskip("torch", reason="This test case require torch")

# Temporarily disable QAIA GPU module in version 0.10
# _HAS_GPU = False


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
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


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_aSB_gpu():
    """
    Description: Test ASB GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100
    dt = 0.9
    xi = 0.1

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = ASB(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float32 test
    solver_gpu_fp32 = ASB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = ASB(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_gpu_h = ASB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float32')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy().tolist())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_bSB_gpu():
    """
    Description: Test BSB GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100
    dt = 0.9
    xi = 0.1

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = BSB(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float16 test
    solver_gpu_fp16 = BSB(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')
    solver_gpu_fp16.update()
    energy_gpu_fp16 = np.mean(solver_gpu_fp16.calc_energy())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp16 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = BSB(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_gpu_h = BSB(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')
    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1

    # GPU int8 test
    solver_int8 = BSB(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8.update()
    energy_int8 = np.mean(solver_int8.calc_energy())
    assert np.abs(energy_int8 - energy_cpu) / np.abs(energy_cpu) < 0.1

    # GPU int8 + external field test
    solver_int8_h = BSB(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8_h.update()
    energy_int8_h = np.mean(solver_int8_h.calc_energy())
    assert np.abs(energy_int8_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1

    # GPU float32 test
    solver_gpu_fp32 = BSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1

    # GPU float32 + external field test
    solver_fp32_h = BSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float32')
    solver_fp32_h.update()
    energy_fp32_h = np.mean(solver_fp32_h.calc_energy().tolist())
    assert np.abs(energy_fp32_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason="Machine does not has GPU.")
def test_dSB_gpu():
    """
    Description: Test DSB GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100
    dt = 0.9
    xi = 0.1

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = DSB(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float16 test
    solver_gpu_fp16 = DSB(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')
    solver_gpu_fp16.update()
    energy_gpu_fp16 = np.mean(solver_gpu_fp16.calc_energy())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp16 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = DSB(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_gpu_h = DSB(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1

    # GPU int8 test
    solver_int8 = DSB(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8.update()
    energy_int8 = np.mean(solver_int8.calc_energy())
    assert np.abs(energy_int8 - energy_cpu) / np.abs(energy_cpu) < 0.1

    # GPU int8 + external field test
    solver_int8_h = DSB(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8_h.update()
    energy_int8_h = np.mean(solver_int8_h.calc_energy())
    assert np.abs(energy_int8_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1

    # GPU float32 test
    solver_gpu_fp32 = DSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1

    # GPU float32 + external field test
    solver_fp32_h = DSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float32')
    solver_fp32_h.update()
    energy_fp32_h = np.mean(solver_fp32_h.calc_energy().tolist())
    assert np.abs(energy_fp32_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_LQA_gpu():
    """
    Description: Test LQA GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = LQA(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float32 test
    solver_gpu_fp32 = LQA(G, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = LQA(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_gpu_h = LQA(G, h=h, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy().tolist())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_CAC_gpu():
    """
    Description: Test CAC GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = CAC(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float32 test
    solver_gpu_fp32 = CAC(G, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = CAC(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_gpu_h = CAC(G, h=h, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy().tolist())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_CFC_gpu():
    """
    Description: Test CFC GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = CFC(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float32 test
    solver_gpu_fp32 = CFC(G, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = CFC(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_gpu_h = CFC(G, h=h, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy().tolist())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_SFC_gpu():
    """
    Description: Test SFC GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = SFC(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float32 test
    solver_gpu_fp32 = SFC(G, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = SFC(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_gpu_h = SFC(G, h=h, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy().tolist())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_NMFA_gpu():
    """
    Description: Test NMFA GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 1000

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = NMFA(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float32 test
    solver_gpu_fp32 = NMFA(G, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = NMFA(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_gpu_h = NMFA(G, h=h, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy().tolist())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.skipif(not _HAS_GPU, reason='Machine does not has GPU.')
def test_SimCIM_gpu():
    """
    Description: Test SimCIM GPU implementation end-to-end performance
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    n_iter = 1000  # Increase the number of iterations for end-to-end testing
    batch_size = 100

    # Initialize the same random state
    x = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = SimCIM(G, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_cpu.update()
    energy_cpu = np.mean(solver_cpu.calc_energy())

    # GPU float32 test
    solver_gpu_fp32 = SimCIM(G, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')
    solver_gpu_fp32.update()
    energy_gpu_fp32 = np.mean(solver_gpu_fp32.calc_energy().tolist())

    # Compare energy values, allowing some error
    assert np.abs(energy_gpu_fp32 - energy_cpu) / np.abs(energy_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = SimCIM(G, h=h, x=x.copy(), n_iter=n_iter, batch_size=batch_size, backend='cpu-float32')
    solver_gpu_h = SimCIM(G, h=h, n_iter=n_iter, batch_size=batch_size, backend='gpu-float32')

    solver_cpu_h.update()
    solver_gpu_h.update()
    energy_cpu_h = np.mean(solver_cpu_h.calc_energy())
    energy_gpu_h = np.mean(solver_gpu_h.calc_energy().tolist())
    assert np.abs(energy_gpu_h - energy_cpu_h) / np.abs(energy_cpu_h) < 0.1
