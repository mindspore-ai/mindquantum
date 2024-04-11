# Copyright 2024 Huawei Technologies Co., Ltd
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
"""Test qudit mapping functions."""

from functools import reduce
import pytest
import numpy as np
from numpy.linalg import norm
from mindquantum.algorithm.library import qudit_mapping

qudit_original = {3: np.array([0.26570339 + 0.42469573j, 0.67444848 + 0.11068155j, 0.51928555 + 0.11066444j]), 4: np.array([0.03199561 + 0.01133906j, 0.47713693 + 0.5342791j, 0.33112684 + 0.45855468j, 0.39004489 + 0.11696793j]), 5: np.array([0.15829351 + 0.2535388j, 0.15966864 + 0.53266802j, 0.26486777 + 0.12144082j, 0.45684342 + 0.25433583j, 0.37604349 + 0.31894797j])}  # pylint: disable=line-too-long
qubit_original = {3: np.array([0.26570339 + 0.42469573j, 0.47690709 + 0.07826368j, 0.47690709 + 0.07826368j, 0.51928555 + 0.11066444j]), 4: np.array([0.03199561 + 0.01133906j, 0.27547514 + 0.30846618j, 0.27547514 + 0.30846618j, 0.19117617 + 0.26474667j, 0.27547514 + 0.30846618j, 0.19117617 + 0.26474667j, 0.19117617 + 0.26474667j, 0.39004489 + 0.11696793j]), 5: np.array([0.15829351 + 0.2535388j, 0.07983432 + 0.26633401j, 0.07983432 + 0.26633401j, 0.10813182 + 0.04957801j, 0.07983432 + 0.26633401j, 0.10813182 + 0.04957801j, 0.10813182 + 0.04957801j, 0.22842171 + 0.12716791j, 0.07983432 + 0.26633401j, 0.10813182 + 0.04957801j, 0.10813182 + 0.04957801j, 0.22842171 + 0.12716791j, 0.10813182 + 0.04957801j, 0.22842171 + 0.12716791j, 0.22842171 + 0.12716791j, 0.37604349 + 0.31894797j])}  # pylint: disable=line-too-long
index_original = {'d3n1': {0: [0], 1: [1, 2], 2: [3]}, 'd4n1': {0: [0], 1: [1, 2, 4], 2: [3, 5, 6], 3: [7]}, 'd5n1': {0: [0], 1: [1, 2, 4, 8], 2: [3, 5, 6, 9, 10, 12], 3: [7, 11, 13, 14], 4: [15]}, 'd3n2': {0: [0], 1: [1, 2], 2: [3], 3: [4, 8], 4: [5, 6, 9, 10], 5: [7, 11], 6: [12], 7: [13, 14], 8: [15]}, 'd4n2': {0: [0], 1: [1, 2, 4], 2: [3, 5, 6], 3: [7], 4: [8, 16, 32], 5: [9, 10, 12, 17, 18, 20, 33, 34, 36], 6: [11, 13, 14, 19, 21, 22, 35, 37, 38], 7: [15, 23, 39], 8: [24, 40, 48], 9: [25, 26, 28, 41, 42, 44, 49, 50, 52], 10: [27, 29, 30, 43, 45, 46, 51, 53, 54], 11: [31, 47, 55], 12: [56], 13: [57, 58, 60], 14: [59, 61, 62], 15: [63]}, 'd5n2': {0: [0], 1: [1, 2, 4, 8], 2: [3, 5, 6, 9, 10, 12], 3: [7, 11, 13, 14], 4: [15], 5: [16, 32, 64, 128], 6: [17, 18, 20, 24, 33, 34, 36, 40, 65, 66, 68, 72, 129, 130, 132, 136], 7: [19, 21, 22, 25, 26, 28, 35, 37, 38, 41, 42, 44, 67, 69, 70, 73, 74, 76, 131, 133, 134, 137, 138, 140], 8: [23, 27, 29, 30, 39, 43, 45, 46, 71, 75, 77, 78, 135, 139, 141, 142], 9: [31, 47, 79, 143], 10: [48, 80, 96, 144, 160, 192], 11: [49, 50, 52, 56, 81, 82, 84, 88, 97, 98, 100, 104, 145, 146, 148, 152, 161, 162, 164, 168, 193, 194, 196, 200], 12: [51, 53, 54, 57, 58, 60, 83, 85, 86, 89, 90, 92, 99, 101, 102, 105, 106, 108, 147, 149, 150, 153, 154, 156, 163, 165, 166, 169, 170, 172, 195, 197, 198, 201, 202, 204], 13: [55, 59, 61, 62, 87, 91, 93, 94, 103, 107, 109, 110, 151, 155, 157, 158, 167, 171, 173, 174, 199, 203, 205, 206], 14: [63, 95, 111, 159, 175, 207], 15: [112, 176, 208, 224], 16: [113, 114, 116, 120, 177, 178, 180, 184, 209, 210, 212, 216, 225, 226, 228, 232], 17: [115, 117, 118, 121, 122, 124, 179, 181, 182, 185, 186, 188, 211, 213, 214, 217, 218, 220, 227, 229, 230, 233, 234, 236], 18: [119, 123, 125, 126, 183, 187, 189, 190, 215, 219, 221, 222, 231, 235, 237, 238], 19: [127, 191, 223, 239], 20: [240], 21: [241, 242, 244, 248], 22: [243, 245, 246, 249, 250, 252], 23: [247, 251, 253, 254], 24: [255]}}  # pylint: disable=line-too-long


def random_qudits(dim: int, n_qudits: int) -> np.ndarray:
    """Generate random n-qudit states."""
    qudit_list = [np.random.rand(dim) + 1j * np.random.rand(dim) for _ in range(n_qudits)]
    qudits = np.reshape(reduce(np.kron, qudit_list), (-1, 1))
    qudits /= norm(qudits)
    return qudits


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_symmetric_state_index():
    """
    Feature: symmetric state index.
    Description: test arbitrary symmetric state index.
    Expectation: success.
    """
    index_test = {f'd{dim}n{n_qudits}': qudit_mapping._symmetric_state_index(dim, n_qudits) for n_qudits in [1, 2] for dim in [3, 4, 5]}  # pylint: disable=line-too-long,protected-access
    assert index_test == index_original


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_qudit_symmetric_decoding():
    """
    Feature: qudit symmetric decoding.
    Description: test arbitrary qudit symmetric decoding.
    Expectation: success.
    """
    np.random.seed(42)
    for dim in [3, 4, 5]:
        qudit_test = qudit_mapping.qudit_symmetric_decoding(qubit_original[dim])
        assert np.allclose(qudit_test, qudit_original[dim])


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_qudit_symmetric_encoding():
    """
    Feature: qudit symmetric encoding.
    Description: test arbitrary qudit symmetric encoding.
    Expectation: success.
    """
    np.random.seed(42)
    for dim in [3, 4, 5]:
        qubit_test = qudit_mapping.qudit_symmetric_encoding(random_qudits(dim, 1))
        assert np.allclose(qubit_test, qubit_original[dim])

@pytest.mark.level0
@pytest.mark.platform_x86_cpu
def test_mat_to_op():
    """
    Feature: Matrix to QubitOperator transformation.
    Description: test arbitrary Qubit matrix transform to QubitOperators.
    Expectation: success.
    """
    np.random.seed(42)
    state = random_qudits(2, 2)
    mat = state @ np.transpose(np.conj(state))
    qubit_test = qudit_mapping.mat_to_op(mat, little_endian=True)
    assert np.allclose(qubit_test.matrix().A, mat)
