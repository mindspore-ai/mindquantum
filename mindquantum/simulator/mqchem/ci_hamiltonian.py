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
# pylint: disable=c-extension-no-member,too-few-public-methods
"""CI Hamiltonian for CI-basis expectation and application in MQ Chemistry simulator."""
from typing import List, Tuple

from ...utils.type_value_check import _check_input_type
from ...core.operators import FermionOperator


class CIHamiltonian:
    r"""
    A wrapper for a fermionic Hamiltonian to be used with the
    :class:`~.simulator.mqchem.MQChemSimulator`.

    This class stores a fermionic Hamiltonian for efficient expectation
    value calculations within a specific CI space.

    Note:
        This Hamiltonian object is specifically designed for the `MQChemSimulator`
        and is not compatible with the standard state-vector `Simulator`.

    Args:
        fermion_hamiltonian (FermionOperator): A normal-ordered fermionic Hamiltonian.

    Examples:
        >>> from mindquantum.core.operators import FermionOperator
        >>> from mindquantum.simulator import mqchem
        >>> ham_op = FermionOperator('0^ 0', 1.0) + FermionOperator('1^ 1', 0.5)
        >>> ci_ham = mqchem.CIHamiltonian(ham_op)
        >>> ci_ham
          1 [0^ 0] +
        1/2 [1^ 1]
    """

    def __init__(self, fermion_hamiltonian: FermionOperator):
        """Initialize a CIHamiltonian."""
        _check_input_type("fermion_hamiltonian", FermionOperator, fermion_hamiltonian)
        self.fermion_hamiltonian = fermion_hamiltonian
        self.ham_data: List[Tuple[List[Tuple[int, bool]], float]] = []
        for term, coef in fermion_hamiltonian.terms.items():
            ops = list(term)
            val = coef.const
            self.ham_data.append((ops, val))
        self._cpp_obj_cache = {}

    def __str__(self):
        return str(self.fermion_hamiltonian)

    def __repr__(self):
        return self.__str__()

    def get_cpp_obj(self, backend, n_qubits, n_electrons):
        """
        Return the underlying C++ CppCIHamiltonian object for a given backend.

        Note:
            This method is for internal use by the :class:`~.simulator.mqchem.MQChemSimulator`.

        Args:
            backend: The C++ backend module (_mq_chem.float or _mq_chem.double).
            n_qubits (int): The total number of qubits (spin-orbitals) in the system.
            n_electrons (int): The total number of electrons in the system.

        Returns:
            The C++ object used for simulation.
        """
        cache_key = (backend, n_qubits, n_electrons)
        if cache_key not in self._cpp_obj_cache:
            self._cpp_obj_cache[cache_key] = backend.CppCIHamiltonian(self.ham_data, n_qubits, n_electrons)
        return self._cpp_obj_cache[cache_key]
