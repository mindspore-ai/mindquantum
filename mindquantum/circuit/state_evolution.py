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
"""Evaluate a quantum circuit."""

from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

from mindspore import Tensor
from mindquantum.parameterresolver import ParameterResolver as PR
from mindquantum.nn import generate_evolution_operator
from mindquantum.utils import normalize
from mindquantum.utils import ket_string
from mindquantum.circuit import Circuit


def _generate_n_qubits_index(n_qubits):
    out = []
    for i in range(1 << n_qubits):
        out.append(bin(i)[2:].zfill(n_qubits))
    return out


class StateEvolution:
    """
    Calculate the final state of a parameterized or non parameterized quantum circuit.

    Args:
        circuit (Circuit): The circuit that you want to do evolution.

    Examples:
        >>> from mindquantum.circuit import StateEvolution
        >>> from mindquantum.circuit import qft
        >>> print(StateEvolution(qft([0, 1])).final_state(ket=True))
        0.5¦00⟩
        0.5¦01⟩
        0.5¦10⟩
        0.5¦11⟩
    """
    def __init__(self, circuit):
        if not isinstance(circuit, Circuit):
            raise TypeError(
                f'Input circuit should be a quantum circuit, but get {type(circuit)}'
            )
        self.circuit = circuit
        self.evol = generate_evolution_operator(self.circuit)
        self.index = _generate_n_qubits_index(self.circuit.n_qubits)

    def final_state(self, param=None, ket=False):
        """
        Get the final state of the input quantum circuit.

        Args:
            param (Union[Tensor, numpy.ndarray, ParameterResolver, dict]): The
                parameter for the parameterized quantum circuit. If None, the
                quantum circuit should be a non parameterized quantum circuit.
                Default: None.
            ket (bool): Whether to print the final state in ket format. Default: False.

        Returns:
            numpy.ndarray, the final state in numpy array format.
        """
        if param is None:
            if self.circuit.para_name:
                raise ValueError(
                    "Require a non parameterized quantum circuit, since not parameters specified."
                )
            return self.evol() if not ket else '\n'.join(
                ket_string(self.evol()))
        if isinstance(param, np.ndarray):
            return self.evol(Tensor(param)) if not ket else '\n'.join(
                ket_string(self.evol(Tensor(param))))
        if isinstance(param, Tensor):
            return self.evol(param) if not ket else '\n'.join(
                ket_string(self.evol(param)))
        if isinstance(param, (PR, dict)):
            data = [param[i] for i in self.circuit.para_name]
            data = Tensor(np.array(data).astype(np.float32))
            return self.evol(data) if not ket else '\n'.join(
                ket_string(self.evol(data)))
        raise TypeError(
            f"parameter requires a numpy array or a ParameterResolver or a dict, ut get {type(param)}"
        )

    def sampling(self, shots=1, param=None, show=False):
        """
        Sampling the bit string based on the final state.

        Args:
            shots (int): How many samples you want to get. Default: 1.
            param (Union[Tensor, numpy.ndarray, ParameterResolver, dict]): The
                parameter for the parameterized quantum circuit. If None, the
                quantum circuit should be a non parameterized quantum circuit.
                Default: None.
            show (bool): Whether to show the sampling result in bar plot. Default: False.

        Returns:
            dict, a dict with key as bit string and value as number of samples.

        Examples:
            >>> from mindquantum.circuit import StateEvolution
            >>> from mindquantum.circuit import qft
            >>> import numpy as np
            >>> np.random.seed(42)
            >>> StateEvolution(qft([0, 1])).sampling(100)
            {'00': 29, '01': 24, '10': 23, '11': 24}
        """
        final_state = self.final_state(param)
        amps = normalize(np.abs(final_state)**2)**2
        sampling = Counter(np.random.choice(self.index, p=amps, size=shots))
        result = dict(zip(self.index, [0] * len(self.index)))
        result.update(sampling)
        if show:
            plt.bar(result.keys(), result.values())
            if self.circuit.n_qubits > 2:
                plt.xticks(rotation=45)
            plt.show()
        return result
