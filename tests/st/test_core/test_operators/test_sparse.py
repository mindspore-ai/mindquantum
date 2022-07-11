# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test operator sparsing."""

from pathlib import Path

import numpy as np
from openfermion import get_sparse_operator
from openfermion.chem import MolecularData

from mindquantum.algorithm.nisq import Transform
from mindquantum.core.operators import get_fermion_operator
from mindquantum.third_party.interaction_operator import InteractionOperator


def test_sparsing_operator():
    """
    Description: Test sparsing operator
    Expectation: success
    """

    molecular = Path(__file__).parent.parent.parent / 'H4.hdf5'

    mol = MolecularData(filename=str(molecular))
    mol.load()

    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = get_fermion_operator(inter_ops)

    ham = Transform(ham_hiq).jordan_wigner()

    hamiltonian = ham.to_openfermion()
    matrix1 = get_sparse_operator(hamiltonian).toarray()
    matrix2 = ham.matrix().toarray()
    matrix3 = ham_hiq.matrix().toarray()
    eigen_v1 = np.real(np.linalg.eigvals(matrix1))
    eigen_v2 = np.real(np.linalg.eigvals(matrix2))
    eigen_v3 = np.real(np.linalg.eigvals(matrix3))
    eigen_v1.sort()
    eigen_v2.sort()
    eigen_v3.sort()
    assert np.allclose(eigen_v1, eigen_v2)
    assert np.allclose(eigen_v1, eigen_v3)
