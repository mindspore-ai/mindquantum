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
"""Quantum Neuron."""

import numpy as np
from mindquantum.core.gates import RY, Y, RZ, Measure
from mindquantum.core.circuit import dagger, Circuit


class QuantumNeuron:
    """
    A quantum neuron implementation based on RUS (Repeat-Until-Success) strategy,
    simulating classical neuron behavior and activation functions through quantum circuits.

    For more details, see `Quantum neuron: an elementary building block for machine
    learning on quantum computers <https://arxiv.org/abs/1711.11240>`_.

    Working principle:
    - Uses RUS circuit, repeatedly executing quantum circuits until target measurement is obtained
    - Measurement result '0' indicates successful application of non-linear function rotation
    - Measurement result '1' triggers recovery operation and repeats until success

    Note:
        - For input qubits in superposition states, the output state has a functional
          relationship with the number of failures (measurement '1') before the final
          success (measurement '0') in the RUS circuit. See Appendix in arXiv:1711.11240
          for detailed mathematical derivation.
        - The recovery rotation angle in the original paper (arXiv:1711.11240) is RY(-π/2), but our
          implementation uses RY(π/2) based on experimental validation. Users are advised to carefully
          verify the behavior in their specific use cases.

    Args:
        weight (Union[List[float], np.ndarray]): List of weights or numpy array. Length must equal
            number of input qubits, each weight corresponds to an input qubit.
        gamma (Union[int, float]): Scaling factor to adjust the weight impact. Default: 1
        bias (Union[int, float]): Bias term. Default: 0
        input_qubits (Optional[List[int]]): List of qubit indices used as inputs. If None, will use
            [0, 1, ..., len(weight)-1]. Default: None
        output_qubit (Optional[int]): Index of the qubit which is the output of the neuron. If None,
            will be set to ancilla_qubit + 1. Default: None
        ancilla_qubit (Optional[int]): Index of the auxiliary qubit used for computation. If None,
            will be set to len(input_qubits) + 1. Default: None

    Examples:
        >>> from mindquantum.simulator import Simulator
        >>> from mindquantum import Circuit, H,X
        >>> from mindquantum.algorithm import QuantumNeuron
        >>> # Create 2-qubit input state
        >>> circ = Circuit()
        >>> circ += H.on(0)
        >>> circ += H.on(1)
        >>> sim = Simulator('mqvector', 4)
        >>> sim.apply_circuit(circ)
        >>> qn = QuantumNeuron(weight=[1, 1], input_qubits=[0, 1], output_qubit=3, ancilla_qubit=2)
        >>> # Repeat quantum neuron circuit until success
        >>> while True:
        >>>     result = sim.apply_circuit(qn.circuit)
        >>>     if next(iter(result.data))[0] == '1':
        >>>         sim.apply_circuit(qn.recovery_circuit)
        >>>     else:
        >>>         print("Non-linear function rotation successfully applied to output qubit")
        >>>         break
    """

    def __init__(self, weight, gamma=1, bias=0, input_qubits=None, output_qubit=None, ancilla_qubit=None):
        """Initialize a quantum neuron."""
        if not isinstance(weight, (list, np.ndarray)):
            raise TypeError("weight must be a list or numpy array")
        if not all(isinstance(w, (int, float)) for w in weight):
            raise TypeError("all weights must be numeric")
        if not isinstance(gamma, (int, float)):
            raise TypeError("gamma must be numeric")
        if not isinstance(bias, (int, float)):
            raise TypeError("bias must be numeric")
        self.weight = weight
        self.gamma = gamma
        self.bias = bias
        if input_qubits is None:
            input_qubits = [i for i in range(len(self.weight))]
        if ancilla_qubit is None:
            ancilla_qubit = len(input_qubits) + 1
        if output_qubit is None:
            output_qubit = ancilla_qubit + 1
        self.input_qubits = input_qubits
        self.output_qubit = output_qubit
        self.ancilla_qubit = ancilla_qubit
        if len(self.input_qubits) != len(self.weight):
            raise ValueError("The length of weight must be equal to the number of input qubits")

        self._encoding_circ = None
        self._circuit = None
        self._recovery_circ = None

    @property
    def _weight_encoding_circuit(self) -> Circuit:
        """
        Generate the quantum circuit for encoding weights and bias.

        This circuit applies RY rotations controlled by input qubits to encode weights,
        followed by a bias rotation on the ancilla qubit.

        Returns:
            Circuit, A quantum circuit that encodes the weights and bias
        """
        if self._encoding_circ is None:
            self._encoding_circ = Circuit()
            for i, w in enumerate(self.weight):
                self._encoding_circ += RY(2 * self.gamma * w).on(self.ancilla_qubit, self.input_qubits[i])
            self._encoding_circ += RY(2 * self.bias).on(self.ancilla_qubit)
        return self._encoding_circ

    @property
    def circuit(self) -> Circuit:
        """
        The quantum circuit of the quantum neuron.

        Returns:
            Circuit: The quantum circuit of the quantum neuron
        """
        if self._circuit is None:
            self._circuit = Circuit()
            self._circuit += self._weight_encoding_circuit
            self._circuit += Y.on(self.output_qubit, self.ancilla_qubit)
            self._circuit += RZ(-np.pi / 2).on(self.ancilla_qubit)
            self._circuit += dagger(self._weight_encoding_circuit)
            self._circuit += Measure(reset_to=0).on(self.ancilla_qubit)
        return self._circuit

    @property
    def recovery_circuit(self) -> Circuit:
        """
        The recovery circuit when measurement result is '1'.

        This circuit applies a π/2 rotation around Y-axis on the output qubit
        to recover from an unsuccessful measurement result.

        Note:
            While the original paper (arXiv:1711.11240) suggests using a RY(-π/2) rotation,
            our implementation uses RY(π/2) based on experimental validation. Users should
            verify this behavior in their specific applications.

        Returns:
            Circuit, A quantum circuit for recovery operation
        """
        if self._recovery_circ is None:
            self._recovery_circ = Circuit().ry(np.pi / 2, self.output_qubit)
        return self._recovery_circ
