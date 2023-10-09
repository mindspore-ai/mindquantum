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

import os
from pathlib import Path

import numpy as np
import pytest

import mindquantum as mq
from mindquantum.algorithm.nisq import Transform
from mindquantum.core.operators import FermionOperator
from mindquantum.third_party.interaction_operator import InteractionOperator

_HAS_OPENFERMION = True
try:
    from openfermion import get_sparse_operator
    from openfermion.chem import MolecularData
except (ImportError, AttributeError):
    _HAS_OPENFERMION = False
_FORCE_TEST = bool(os.environ.get("FORCE_TEST", False))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.parametrize('dtype', [mq.complex128, mq.complex64])
@pytest.mark.skipif(not _HAS_OPENFERMION, reason='OpenFermion is not installed')
@pytest.mark.skipif(not _FORCE_TEST, reason='set not force test')
def test_sparsing_operator(dtype):
    """
    Description: Test sparsing operator
    Expectation: success
    """
    molecular = Path(__file__).parent.parent.parent / 'H4.hdf5'

    mol = MolecularData(filename=str(molecular))
    mol.load()

    ham_of = mol.get_molecular_hamiltonian()
    inter_ops = InteractionOperator(*ham_of.n_body_tensors.values())
    ham_hiq = FermionOperator(inter_ops)

    ham = Transform(ham_hiq).jordan_wigner()

    hamiltonian = ham.to_openfermion()
    matrix1 = get_sparse_operator(hamiltonian).toarray().astype(mq.to_np_type(dtype))
    matrix2 = ham.astype(dtype).matrix().toarray()
    matrix3 = ham_hiq.astype(dtype).matrix().toarray()
    eigen_v1 = np.real(np.linalg.eigvals(matrix1))
    eigen_v2 = np.real(np.linalg.eigvals(matrix2))
    eigen_v3 = np.real(np.linalg.eigvals(matrix3))
    eigen_v1.sort()
    eigen_v2.sort()
    eigen_v3.sort()
    assert np.allclose(eigen_v1, eigen_v2, atol=1e-6)
    assert np.allclose(eigen_v1, eigen_v3, atol=1e-6)
