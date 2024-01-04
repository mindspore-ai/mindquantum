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

import numpy as np
from scipy.sparse import coo_matrix

from mindquantum.algorithm.qaia import ASB, BSB, CAC, CFC, DSB, LQA, SFC
from mindquantum.utils.fdopen import fdopen


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


G = read_gset(str(Path(__file__).parent.parent.parent / 'G1.txt'))


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
