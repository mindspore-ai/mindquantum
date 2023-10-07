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
"""Test mindquantum provide arxiv ansatz."""

from mindquantum.algorithm import nisq
import mindquantum.core.gates as G
from mindquantum.core.circuit import Circuit

import numpy as np
import pytest

qubits = 4


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz1():
    """
    Description: test arxiv 1905 ansatz1.
    Expectation: success
    """
    ansatz1_cir = nisq.Ansatz1(qubits, 1).circuit
    pr = dict(zip(ansatz1_cir.params_name, np.random.rand(qubits * 2)))
    ansatz1_matrix = ansatz1_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz1_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz2():
    """
    Description: test arxiv 1905 ansatz2.
    Expectation: success
    """
    ansatz2_cir = nisq.Ansatz2(qubits, 1).circuit
    pr = dict(zip(ansatz2_cir.params_name, np.random.rand(qubits * 2)))
    ansatz2_matrix = ansatz2_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    for j in range(qubits - 1):
        test_cir += G.CNOT(qubits - 2 - j, qubits - 1 - j)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz2_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz3():
    """
    Description: test arxiv 1905 ansatz3.
    Expectation: success
    """
    ansatz3_cir = nisq.Ansatz3(qubits, 1).circuit
    pr = dict(zip(ansatz3_cir.params_name, np.random.rand(qubits * 3 - 1)))
    ansatz3_matrix = ansatz3_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    for j in range(qubits - 1):
        test_cir += G.RZ(f'p{qubits*2+j}').on(qubits - 2 - j, qubits - 1 - j)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz3_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz4():
    """
    Description: test arxiv 1905 ansatz4.
    Expectation: success
    """
    ansatz4_cir = nisq.Ansatz4(qubits, 1).circuit
    pr = dict(zip(ansatz4_cir.params_name, np.random.rand(qubits * 3 - 1)))
    ansatz4_matrix = ansatz4_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    for j in range(qubits - 1):
        test_cir += G.RX(f'p{qubits*2+j}').on(qubits - 2 - j, qubits - 1 - j)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz4_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz5():
    """
    Description: test arxiv 1905 ansatz5.
    Expectation: success
    """
    ansatz5_cir = nisq.Ansatz5(qubits, 1).circuit
    pr = dict(zip(ansatz5_cir.params_name, np.random.rand(qubits * (qubits + 3))))
    ansatz5_matrix = ansatz5_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    pr_count = qubits * 2
    for i in range(qubits)[::-1]:
        for j in range(qubits)[::-1]:
            if i == j:
                continue
            test_cir += G.RZ(f'p{pr_count}').on(j, i)
            pr_count += 1
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RX(f'p{i+pr_count}').on(i)
        test_cir += G.RZ(f'p{i+qubits+pr_count}').on(i)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz5_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz6():
    """
    Description: test arxiv 1905 ansatz6.
    Expectation: success
    """
    ansatz6_cir = nisq.Ansatz6(qubits, 1).circuit
    pr = dict(zip(ansatz6_cir.params_name, np.random.rand(qubits * (qubits + 3))))
    ansatz6_matrix = ansatz6_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    pr_count = qubits * 2
    for i in range(qubits)[::-1]:
        for j in range(qubits)[::-1]:
            if i == j:
                continue
            test_cir += G.RX(f'p{pr_count}').on(j, i)
            pr_count += 1
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RX(f'p{i+pr_count}').on(i)
        test_cir += G.RZ(f'p{i+qubits+pr_count}').on(i)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz6_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz7():
    """
    Description: test arxiv 1905 ansatz7.
    Expectation: success
    """
    ansatz7_cir = nisq.Ansatz7(qubits, 1).circuit
    pr = dict(zip(ansatz7_cir.params_name, np.random.rand(qubits * 5 - 1)))
    ansatz7_matrix = ansatz7_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    pr_count = qubits * 2
    for i in range(qubits)[:qubits:2]:
        test_cir += G.RZ(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RX(f'p{i+pr_count}').on(i)
        test_cir += G.RZ(f'p{i+qubits+pr_count}').on(i)
    pr_count += qubits * 2
    test_cir += G.BARRIER
    for i in range(qubits)[1:qubits - 1:2]:
        test_cir += G.RZ(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz7_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz8():
    """
    Description: test arxiv 1905 ansatz8.
    Expectation: success
    """
    ansatz8_cir = nisq.Ansatz8(qubits, 1).circuit
    pr = dict(zip(ansatz8_cir.params_name, np.random.rand(qubits * 5 - 1)))
    ansatz8_matrix = ansatz8_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    pr_count = qubits * 2
    for i in range(qubits)[:qubits:2]:
        test_cir += G.RX(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RX(f'p{i+pr_count}').on(i)
        test_cir += G.RZ(f'p{i+qubits+pr_count}').on(i)
    pr_count += qubits * 2
    test_cir += G.BARRIER
    for i in range(qubits)[1:qubits - 1:2]:
        test_cir += G.RX(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz8_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz9():
    """
    Description: test arxiv 1905 ansatz9.
    Expectation: success
    """
    ansatz9_cir = nisq.Ansatz9(qubits, 1).circuit
    pr = dict(zip(ansatz9_cir.params_name, np.random.rand(qubits)))
    ansatz9_matrix = ansatz9_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.H.on(i)
    test_cir += G.BARRIER
    for i in range(qubits)[qubits - 2::-1]:
        test_cir += G.Z.on(i, i + 1)
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz9_matrix, test_matrix, 1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz10():
    """
    Description: test arxiv 1905 ansatz10.
    Expectation: success
    """
    ansatz10_cir = nisq.Ansatz10(qubits, 1).circuit
    pr = dict(zip(ansatz10_cir.params_name, np.random.rand(qubits * 2)))
    ansatz10_matrix = ansatz10_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RY(f'p{i}').on(i)
    test_cir += G.BARRIER
    for i in range(qubits)[qubits - 1::-1]:
        if i == 0:
            test_cir += G.Z.on(i, qubits - 1)
        else:
            test_cir += G.Z.on(i, i - 1)
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RY(f'p{i+qubits}').on(i)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz10_matrix, test_matrix, 1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz11():
    """
    Description: test arxiv 1905 ansatz11.
    Expectation: success
    """
    ansatz11_cir = nisq.Ansatz11(qubits, 1).circuit
    pr = dict(zip(ansatz11_cir.params_name, np.random.rand(4 * (qubits - 1))))
    ansatz11_matrix = ansatz11_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RY(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    for i in range(qubits)[:qubits - (qubits % 2):2]:
        test_cir += G.X.on(i, i + 1)
    test_cir += G.BARRIER
    for i in range(qubits)[1:-1]:
        test_cir += G.RY(f'p{i+2*qubits-1}').on(i)
        test_cir += G.RZ(f'p{i+(3*qubits-2)-1}').on(i)
    test_cir += G.BARRIER
    for i in range(qubits)[1:qubits - (qubits % 2) - 1:2]:
        test_cir += G.X.on(i, i + 1)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz11_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz12():
    """
    Description: test arxiv 1905 ansatz12.
    Expectation: success
    """
    ansatz12_cir = nisq.Ansatz12(qubits, 1).circuit
    pr = dict(zip(ansatz12_cir.params_name, np.random.rand(4 * (qubits - 1))))
    ansatz12_matrix = ansatz12_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RY(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    for i in range(qubits)[:qubits - (qubits % 2):2]:
        test_cir += G.Z.on(i, i + 1)
    test_cir += G.BARRIER
    for i in range(qubits)[1:-1]:
        test_cir += G.RY(f'p{i+2*qubits-1}').on(i)
        test_cir += G.RZ(f'p{i+(3*qubits-2)-1}').on(i)
    test_cir += G.BARRIER
    for i in range(qubits)[1:qubits - (qubits % 2) - 1:2]:
        test_cir += G.Z.on(i, i + 1)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz12_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz13():
    """
    Description: test arxiv 1905 ansatz13.
    Expectation: success
    """
    ansatz13_cir = nisq.Ansatz13(qubits, 1).circuit
    pr = dict(zip(ansatz13_cir.params_name, np.random.rand(4 * qubits)))
    ansatz13_matrix = ansatz13_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RY(f'p{i}').on(i)
    test_cir += G.BARRIER
    test_cir += G.RZ(f'p{qubits}').on(0, qubits - 1)
    for i in range(qubits)[:0:-1]:
        test_cir += G.RZ(f'p{2*qubits-i}').on(i, i - 1)
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RY(f'p{i+2*qubits}').on(i)
    test_cir += G.RZ(f'p{3*qubits}').on(qubits - 2, qubits - 1)
    test_cir += G.RZ(f'p{3*qubits+1}').on(qubits - 1, 0)
    for i in range(qubits)[:-2]:
        test_cir += G.RZ(f'p{i+3*qubits+2}').on(i, i + 1)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz13_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz14():
    """
    Description: test arxiv 1905 ansatz14.
    Expectation: success
    """
    ansatz14_cir = nisq.Ansatz14(qubits, 1).circuit
    pr = dict(zip(ansatz14_cir.params_name, np.random.rand(4 * qubits)))
    ansatz14_matrix = ansatz14_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RY(f'p{i}').on(i)
    test_cir += G.BARRIER
    test_cir += G.RX(f'p{qubits}').on(0, qubits - 1)
    for i in range(qubits)[:0:-1]:
        test_cir += G.RX(f'p{2*qubits-i}').on(i, i - 1)
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RY(f'p{i+2*qubits}').on(i)
    test_cir += G.RX(f'p{3*qubits}').on(qubits - 2, qubits - 1)
    test_cir += G.RX(f'p{3*qubits+1}').on(qubits - 1, 0)
    for i in range(qubits)[:-2]:
        test_cir += G.RX(f'p{i+3*qubits+2}').on(i, i + 1)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz14_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz15():
    """
    Description: test arxiv 1905 ansatz15.
    Expectation: success
    """
    ansatz15_cir = nisq.Ansatz15(qubits, 1).circuit
    pr = dict(zip(ansatz15_cir.params_name, np.random.rand(2 * qubits)))
    ansatz15_matrix = ansatz15_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RY(f'p{i}').on(i)
    test_cir += G.BARRIER
    test_cir += G.X.on(0, qubits - 1)
    for i in range(qubits)[:0:-1]:
        test_cir += G.X.on(i, i - 1)
    test_cir += G.BARRIER
    for i in range(qubits):
        test_cir += G.RY(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    test_cir += G.X.on(qubits - 2, qubits - 1)
    test_cir += G.X.on(qubits - 1, 0)
    for i in range(qubits)[:-2]:
        test_cir += G.X.on(i, i + 1)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz15_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz16():
    """
    Description: test arxiv 1905 ansatz16.
    Expectation: success
    """
    ansatz16_cir = nisq.Ansatz16(qubits, 1).circuit
    pr = dict(zip(ansatz16_cir.params_name, np.random.rand(3 * qubits - 1)))
    ansatz16_matrix = ansatz16_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    pr_count = 2 * qubits
    for i in range(qubits)[:qubits - (qubits % 2):2]:
        test_cir += G.RZ(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    for i in range(qubits)[1:qubits - (qubits % 2) - 1:2]:
        test_cir += G.RZ(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz16_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz17():
    """
    Description: test arxiv 1905 ansatz17.
    Expectation: success
    """
    ansatz17_cir = nisq.Ansatz17(qubits, 1).circuit
    pr = dict(zip(ansatz17_cir.params_name, np.random.rand(3 * qubits - 1)))
    ansatz17_matrix = ansatz17_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    pr_count = 2 * qubits
    for i in range(qubits)[:qubits - (qubits % 2):2]:
        test_cir += G.RX(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    for i in range(qubits)[1:qubits - (qubits % 2) - 1:2]:
        test_cir += G.RX(f'p{pr_count}').on(i, i + 1)
        pr_count += 1
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz17_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz18():
    """
    Description: test arxiv 1905 ansatz18.
    Expectation: success
    """
    ansatz18_cir = nisq.Ansatz18(qubits, 1).circuit
    pr = dict(zip(ansatz18_cir.params_name, np.random.rand(3 * qubits)))
    ansatz18_matrix = ansatz18_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    test_cir += G.RZ(f'p{2*qubits}').on(0, qubits - 1)
    for i in range(qubits)[:0:-1]:
        test_cir += G.RZ(f'p{3*qubits-i}').on(i, i - 1)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz18_matrix, test_matrix, atol=1e-6)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_ansatz19():
    """
    Description: test arxiv 1905 ansatz19.
    Expectation: success
    """
    ansatz19_cir = nisq.Ansatz19(qubits, 1).circuit
    pr = dict(zip(ansatz19_cir.params_name, np.random.rand(3 * qubits)))
    ansatz19_matrix = ansatz19_cir.matrix(pr)

    test_cir = Circuit()
    for i in range(qubits):
        test_cir += G.RX(f'p{i}').on(i)
        test_cir += G.RZ(f'p{i+qubits}').on(i)
    test_cir += G.BARRIER
    test_cir += G.RX(f'p{2*qubits}').on(0, qubits - 1)
    for i in range(qubits)[:0:-1]:
        test_cir += G.RX(f'p{3*qubits-i}').on(i, i - 1)
    test_matrix = test_cir.matrix(pr)

    assert np.allclose(ansatz19_matrix, test_matrix, atol=1e-6)
