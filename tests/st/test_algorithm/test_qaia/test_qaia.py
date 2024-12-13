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

from mindquantum.algorithm.qaia import ASB, BSB, CAC, CFC, DSB, LQA, SFC
from mindquantum.utils.fdopen import fdopen

import pytest

try:
    subprocess.check_output('nvidia-smi')
    _HAS_GPU = True
except FileNotFoundError:
    _HAS_GPU = False


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
@pytest.mark.platform_x86_cpu
def test_aSB():
    """
    Description: Test ASB
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    y = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = ASB(G, n_iter=1)
    solver.x = x.copy()
    solver.y = y.copy()
    solver.update()
    for _ in range(2):
        x += solver.dm * y
        y -= (x**3 + (1 - solver.p[0]) * x) * solver.dm
    np.allclose(x, solver.x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_bSB():
    """
    Description: Test BSB
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    y = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = BSB(G, n_iter=1)
    solver.x = x.copy()
    solver.y = y.copy()
    solver.update()
    y += (-(1 - solver.p[0]) * x + solver.xi * G @ x) * solver.dt
    x += y * solver.dt
    x = np.where(np.abs(x) > 1, np.sign(x), x)
    np.allclose(x, solver.x)


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
    y = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = BSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_cpu.x = x.copy()
    solver_cpu.y = y.copy()
    solver_cpu.update()
    cut_cpu = np.mean(solver_cpu.calc_cut())

    # GPU float16 test
    solver_gpu_fp16 = BSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')
    solver_gpu_fp16.x = x.copy()
    solver_gpu_fp16.y = y.copy()
    solver_gpu_fp16.update()
    cut_gpu_fp16 = np.mean(solver_gpu_fp16.calc_cut())

    # Compare cut values, allowing some error
    assert np.abs(cut_gpu_fp16 - cut_cpu) / np.abs(cut_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = BSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_gpu_h = BSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')

    solver_cpu_h.x = x.copy()
    solver_cpu_h.y = y.copy()
    solver_gpu_h.x = x.copy()
    solver_gpu_h.y = y.copy()

    solver_cpu_h.update()
    solver_gpu_h.update()
    cut_cpu_h = np.mean(solver_cpu_h.calc_cut())
    cut_gpu_h = np.mean(solver_gpu_h.calc_cut())
    assert np.abs(cut_gpu_h - cut_cpu_h) / np.abs(cut_cpu_h) < 0.1

    # GPU int8 test
    solver_int8 = BSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8.x = x.copy()
    solver_int8.y = y.copy()
    solver_int8.update()
    cut_int8 = np.mean(solver_int8.calc_cut())
    assert np.abs(cut_int8 - cut_cpu) / np.abs(cut_cpu) < 0.1

    # GPU int8 + external field test
    solver_int8_h = BSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8_h.x = x.copy()
    solver_int8_h.y = y.copy()
    solver_int8_h.update()
    cut_int8_h = np.mean(solver_int8_h.calc_cut())
    assert np.abs(cut_int8_h - cut_cpu_h) / np.abs(cut_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_dSB():
    """
    Description: Test DSB
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    y = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = DSB(G, n_iter=1)
    solver.x = x.copy()
    solver.y = y.copy()
    solver.update()
    y += (-(1 - solver.p[0]) * x + solver.xi * G @ np.sign(x)) * solver.dt
    x += y * solver.dt
    x = np.where(np.abs(x) > 1, np.sign(x), x)
    np.allclose(x, solver.x)


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
    y = 0.01 * (np.random.rand(N, batch_size) - 0.5)

    # CPU float32 baseline test
    solver_cpu = DSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_cpu.x = x.copy()
    solver_cpu.y = y.copy()
    solver_cpu.update()
    cut_cpu = np.mean(solver_cpu.calc_cut())

    # GPU float16 test
    solver_gpu_fp16 = DSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')
    solver_gpu_fp16.x = x.copy()
    solver_gpu_fp16.y = y.copy()
    solver_gpu_fp16.update()
    cut_gpu_fp16 = np.mean(solver_gpu_fp16.calc_cut())

    # Compare cut values, allowing some error
    assert np.abs(cut_gpu_fp16 - cut_cpu) / np.abs(cut_cpu) < 0.1  # Allow 10% relative error

    # Use external field for testing
    h = np.random.rand(N, 1)
    solver_cpu_h = DSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='cpu-float32')
    solver_gpu_h = DSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-float16')

    solver_cpu_h.x = x.copy()
    solver_cpu_h.y = y.copy()
    solver_gpu_h.x = x.copy()
    solver_gpu_h.y = y.copy()

    solver_cpu_h.update()
    solver_gpu_h.update()
    cut_cpu_h = np.mean(solver_cpu_h.calc_cut())
    cut_gpu_h = np.mean(solver_gpu_h.calc_cut())
    assert np.abs(cut_gpu_h - cut_cpu_h) / np.abs(cut_cpu_h) < 0.1

    # GPU int8 test
    solver_int8 = DSB(G, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8.x = x.copy()
    solver_int8.y = y.copy()
    solver_int8.update()
    cut_int8 = np.mean(solver_int8.calc_cut())
    assert np.abs(cut_int8 - cut_cpu) / np.abs(cut_cpu) < 0.1

    # GPU int8 + external field test
    solver_int8_h = DSB(G, h=h, n_iter=n_iter, batch_size=batch_size, dt=dt, xi=xi, backend='gpu-int8')
    solver_int8_h.x = x.copy()
    solver_int8_h.y = y.copy()
    solver_int8_h.update()
    cut_int8_h = np.mean(solver_int8_h.calc_cut())
    assert np.abs(cut_int8_h - cut_cpu_h) / np.abs(cut_cpu_h) < 0.1


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_LQA():
    """
    Description: Test LQA
    Expectation: success
    """
    N = G.shape[0]
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = LQA(G, n_iter=1)
    solver.x = x.copy()
    solver.update()
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 10e-8
    m_dx = 0
    v_dx = 0
    t = 1
    tmp = np.pi / 2 * np.tanh(x)
    z = np.sin(tmp)
    y = np.cos(tmp)
    dx = np.pi / 2 * (-t * solver.gamma * G.dot(z) * y + (1 - t) * z) * (1 - np.tanh(x) ** 2)
    # momentum beta1
    m_dx = beta1 * m_dx + (1 - beta1) * dx
    # rms beta2
    v_dx = beta2 * v_dx + (1 - beta2) * dx**2
    # bias correction
    m_dx_corr = m_dx / (1 - beta1)
    v_dx_corr = v_dx / (1 - beta2)

    x = x - solver.dt * m_dx_corr / (np.sqrt(v_dx_corr) + epsilon)
    np.allclose(x, solver.x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_CAC():
    """
    Description: Test CAC
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = CAC(G, n_iter=1)
    solver.x = x.copy()
    solver.update()
    y = np.ones_like(x)
    x = x + (-(x**3) + (solver.p[0] - 1) * x + solver.xi * y * (G @ x)) * solver.dt
    cond = np.abs(x) > (1.5 * np.sqrt(solver.alpha[0]))
    x = np.where(cond, 1.5 * np.sign(x) * np.sqrt(solver.alpha[0]), solver.x)
    np.allclose(x, solver.x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_CFC():
    """
    Description: Test CFC
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = CFC(G, n_iter=1)
    solver.x = x.copy()
    solver.update()
    y = np.ones_like(x)

    x = x + (-(x**3) + (solver.p[0] - 1) * x + solver.xi * y * (G @ x)) * solver.dt
    cond = np.abs(x) > 1.5
    x = np.where(cond, 1.5 * np.sign(x), solver.x)
    np.allclose(x, solver.x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_SFC():
    """
    Description: Test SFC
    Expectation: success
    """
    N = G.shape[0]
    np.random.seed(666)
    x = 0.01 * (np.random.rand(N, 1) - 0.5)
    solver = SFC(G, n_iter=1)
    solver.x = x.copy()
    solver.update()
    y = np.zeros_like(x)
    z = -solver.xi * (G @ x)
    x = x + (-(x**3) + (solver.p[0] - 1) * x - np.tanh(solver.c[0] * z) - solver.k * (z - y)) * solver.dt
    np.allclose(x, solver.x)
