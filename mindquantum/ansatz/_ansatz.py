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
"""Basic class of ansatz."""

from abc import abstractmethod
from mindquantum.circuit import Circuit


class Ansatz:
    """
    Basic class for Ansatz.

    Args:
        name (str): The name of this ansatz.
        n_qubits (int): How many qubits this ansatz act on.
    """
    def __init__(self, name, n_qubits, *args, **kwargs):
        self.name = name
        self.n_qubits = n_qubits
        self._circuit = Circuit()
        self._implement(*args, **kwargs)

    @abstractmethod
    def _implement(self, *args, **kwargs):
        """Implement of ansatz."""

    @property
    def circuit(self):
        """
        Get the quantum circuit of this ansatz.

        Returns:
            Circuit, the quantum circuit of this ansatz.
        """
        return self._circuit
