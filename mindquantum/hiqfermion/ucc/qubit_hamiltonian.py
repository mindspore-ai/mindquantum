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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Get qubit hamiltonian"""

from mindquantum.ops import InteractionOperator
from mindquantum.utils import get_fermion_operator
from mindquantum.hiqfermion.transforms import Transform


def get_qubit_hamiltonian(mol):
    r"""
    Get the qubit hamiltonian of a molecular data.

    Args:
        mol (MolecularData): molecular data.

    Returns:
        QubitOperator, qubit operator of this molecular.
    """
    m_ham = mol.get_molecular_hamiltonian()
    int_ham = InteractionOperator(*(m_ham.n_body_tensors.values()))
    f_ham = get_fermion_operator(int_ham)
    q_ham = Transform(f_ham).jordan_wigner()
    return q_ham
