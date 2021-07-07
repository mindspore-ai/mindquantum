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
"""Hamiltonian module."""

from projectq.ops import QubitOperator as pq_operator
from openfermion.ops import QubitOperator as of_operator
from mindquantum.ops import QubitOperator as hiq_operator


class Hamiltonian:
    """
    A QubitOperator hamiltonian wrapper.

    Args:
        hamiltonian (QubitOperator): The pauli word qubit operator.

    Examples:
        >>> from mindquantum.ops import QubitOperator
        >>> from mindquantum import Hamiltonian
        >>> ham = Hamiltonian(QubitOperator('Z0 Y1', 0.3))
        >>> ham.mindspore_data()
        {'hams_pauli_coeff': [0.3],
         'hams_pauli_word': [['Z', 'Y']],
         'hams_pauli_qubit': [[0, 1]]}
    """
    def __init__(self, hamiltonian):
        if not isinstance(hamiltonian,
                          (pq_operator, of_operator, hiq_operator)):
            raise TypeError("Require a QubitOperator, but get {}!".format(
                type(hamiltonian)))
        self.hamiltonian = hamiltonian
        self.ham_termlist = [(i, j) for i, j in hamiltonian.terms.items()]

    def __str__(self):
        return self.hamiltonian.__str__()

    def __repr__(self):
        return self.hamiltonian.__repr__()

    def mindspore_data(self):
        """
        Generate hamiltonian information for PQC operator.
        """
        m_data = {
            "hams_pauli_coeff": [],
            "hams_pauli_word": [],
            "hams_pauli_qubit": []
        }
        for term, coeff in self.ham_termlist:
            m_data["hams_pauli_coeff"].append(float(coeff))
            m_data["hams_pauli_word"].append([])
            m_data["hams_pauli_qubit"].append([])
            for qubit, word in term:
                m_data["hams_pauli_qubit"][-1].append(qubit)
                m_data["hams_pauli_word"][-1].append(word)
        return m_data
