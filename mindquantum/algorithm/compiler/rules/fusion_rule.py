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
"""Single qubit gate to U3 fusion and conversion rule."""

import numpy as np
from mindquantum.core.gates import U3, GlobalPhase
from mindquantum.core.gates.basic import FunctionalGate
from mindquantum.utils.type_value_check import _check_input_type
from .basic_rule import BasicCompilerRule
from ..dag import DAGCircuit, GateNode
from .compiler_logger import CompileLog as CLog
from .compiler_logger import LogIndentation


def _matrix_to_u3_params(unitary):
    """
    Convert a 2x2 unitary matrix to U3 gate parameters and global phase.

    Args:
        unitary (np.ndarray): 2x2 unitary matrix

    Returns:
        tuple: (theta, phi, lambda, global_phase) where:
            - theta: rotation angle in [0, 2π]
            - phi: first phase angle in [0, 2π]
            - lambda: second phase angle in [0, 2π]
            - global_phase: global phase angle in [0, 2π]

    Raises:
        ValueError: If matrix is not 2x2 or not a valid U(2) matrix
    """

    def _validate_shape(unitary_matrix):
        if unitary_matrix.shape != (2, 2):
            raise ValueError("Matrix must be 2x2")

    def _extract_elements(unitary_matrix):
        return (
            complex(unitary_matrix[0, 0]),
            complex(unitary_matrix[0, 1]),
            complex(unitary_matrix[1, 0]),
            complex(unitary_matrix[1, 1]),
        )

    def _decompose_diagonal(u00, u11):
        # Handle diagonal matrix case (theta = 0)
        return 0.0, 0.0, np.angle(u11 / u00), u00

    def _decompose_antidiagonal(u01, u10):
        # Handle anti-diagonal matrix case (theta = pi)
        if np.isclose(abs(u10), 0.0, atol=1e-8):
            raise ValueError("Invalid U(2) matrix for theta=pi decomposition.")
        return np.pi, 0.0, np.angle(-u01 / u10), u10

    def _decompose_general(u00, u01, u10):
        rotation_angle = 2 * np.arctan2(abs(u10), abs(u00))
        rotation_angle = rotation_angle if rotation_angle >= 0 else rotation_angle + 2 * np.pi

        phase_phi = np.angle(u10 / u00)
        phase_lambda = np.angle(-u01 / u00)

        cos_half_theta = np.cos(rotation_angle / 2)
        if np.isclose(cos_half_theta, 0.0, atol=1e-8):
            phase_factor = u10 / (np.exp(1j * phase_phi) * np.sin(rotation_angle / 2))
        else:
            phase_factor = u00 / cos_half_theta

        return rotation_angle, phase_phi, phase_lambda, phase_factor

    def _validate_phase_factor(u00, u10, theta, phi, phase_factor):
        sin_half_theta = np.sin(theta / 2)
        cos_half_theta = np.cos(theta / 2)

        reconstructed_u00 = phase_factor * cos_half_theta
        reconstructed_u10 = phase_factor * np.exp(1j * phi) * sin_half_theta

        if not (np.isclose(reconstructed_u00, u00, atol=1e-5) and np.isclose(reconstructed_u10, u10, atol=1e-5)):
            alt_phase_factor = u10 / (np.exp(1j * phi) * sin_half_theta)
            if np.isclose(abs(alt_phase_factor), 1.0, atol=1e-5):
                return alt_phase_factor
            raise ValueError("Decomposition failed. Inconsistent global phase calculation.")
        return phase_factor

    _validate_shape(unitary)
    u00, u01, u10, u11 = _extract_elements(unitary)

    # Choose decomposition method based on matrix type
    if np.allclose([abs(u00), abs(u10)], [1.0, 0.0], atol=1e-8):
        theta, phi, lambda_, phase_factor = _decompose_diagonal(u00, u11)
    elif np.isclose(abs(u00), 0.0, atol=1e-8):
        theta, phi, lambda_, phase_factor = _decompose_antidiagonal(u01, u10)
    else:
        theta, phi, lambda_, phase_factor = _decompose_general(u00, u01, u10)
        phase_factor = _validate_phase_factor(u00, u10, theta, phi, phase_factor)

    return theta, phi, lambda_, np.angle(phase_factor)


class U3Fusion(BasicCompilerRule):
    """
    Fuse consecutive single qubit gates into one U3 gate.

    This rule scans through the circuit and combines consecutive single qubit gates
    acting on the same qubit into a single U3 gate. For standalone single qubit gates,
    they will also be converted to U3 form.

    Optionally, it can also track and include the global phase.

    Args:
        rule_name (str): Name of this compiler rule. Default: "U3Fusion"
        log_level (int): Display log level. Default: 0
        with_global_phase (bool): Whether to include global phase gate. Default: False

    Examples:
        >>> from mindquantum.algorithm.compiler import U3Fusion, DAGCircuit
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().rx(1.0, 0).ry(0.5, 0).rz(0.7, 0)
        >>> circ
              ┏━━━━━━━┓ ┏━━━━━━━━━┓ ┏━━━━━━━━━┓
        q0: ──┨ RX(1) ┠─┨ RY(1/2) ┠─┨ RZ(0.7) ┠───
              ┗━━━━━━━┛ ┗━━━━━━━━━┛ ┗━━━━━━━━━┛
        >>> dag_circ = DAGCircuit(circ)
        >>> compiler = U3Fusion()
        >>> compiler.do(dag_circ)
        >>> dag_circ.to_circuit()
              ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        q0: ──┨ U3(θ=1.0768, φ=-0.5722, λ=0.995) ┠───
              ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
    """

    def __init__(self, rule_name="U3Fusion", log_level=0, with_global_phase=False):
        """Initialize a U3Fusion compiler rule."""
        super().__init__(rule_name, log_level)
        self.with_global_phase = with_global_phase

    def _fuse_gates(self, gates_to_fuse, qubit):
        """
        Fuse and convert single qubit gates to U3.

        Returns:
            bool: True if fusion was performed, False otherwise
        """
        if not gates_to_fuse:
            return False

        for gate in gates_to_fuse:
            if gate.gate.parameterized:
                raise ValueError(
                    "U3Fusion cannot handle parameterized gates. Please assign values to all parameters before fusion."
                )

        matrix = gates_to_fuse[0].gate.matrix()
        for gate in gates_to_fuse[1:]:
            matrix = gate.gate.matrix() @ matrix

        theta, phi, lambda_, global_phase = _matrix_to_u3_params(matrix)

        fused_gate = U3(theta, phi, lambda_).on(qubit)
        fused_node = GateNode(fused_gate)

        # Update node connections
        first_father = gates_to_fuse[0].father[qubit]
        last_child = gates_to_fuse[-1].child[qubit]

        if self.with_global_phase and not np.isclose(global_phase, 0.0, atol=1e-8):
            phase_gate = GlobalPhase(global_phase).on(qubit)
            phase_node = GateNode(phase_gate)

            first_father.child[qubit] = phase_node
            phase_node.father[qubit] = first_father

            phase_node.child[qubit] = fused_node
            fused_node.father[qubit] = phase_node
        else:
            first_father.child[qubit] = fused_node
            fused_node.father[qubit] = first_father

        fused_node.child[qubit] = last_child
        last_child.father[qubit] = fused_node

        return True

    def do(self, dag_circuit: DAGCircuit) -> bool:
        """
        Apply single qubit gate fusion to the circuit.

        Args:
            dag_circuit (DAGCircuit): Input circuit in DAG form

        Returns:
            bool: True if any fusion was performed, False otherwise
        """
        _check_input_type('dag_circuit', DAGCircuit, dag_circuit)

        compiled = False
        fusion_count = 0
        total_gates_before = 0
        total_gates_after = 0

        CLog.log(f"Running {CLog.R1(self.rule_name)}", 1, self.log_level)

        for qubit, head in dag_circuit.head_node.items():
            current = head
            gates_to_fuse = []

            while current.child.get(qubit) is not None:
                next_node = current.child[qubit]

                if (
                    isinstance(next_node, GateNode)
                    and len(next_node.gate.obj_qubits) == 1
                    and not next_node.gate.ctrl_qubits
                    and not isinstance(next_node.gate, FunctionalGate)
                ):
                    gates_to_fuse.append(next_node)
                    total_gates_before += 1
                else:
                    if gates_to_fuse:
                        msg = (
                            f"{CLog.R1(self.rule_name)}: Found "
                            f"{CLog.B(len(gates_to_fuse))} consecutive single qubit gates "
                            f"on qubit {CLog.B(qubit)}"
                        )
                        CLog.log(msg, 2, self.log_level)
                        with LogIndentation() as _:
                            CLog.log(
                                f"Gates to fuse: {CLog.B([node.gate for node in gates_to_fuse])}", 2, self.log_level
                            )

                        if self._fuse_gates(gates_to_fuse, qubit):
                            compiled = True
                            fusion_count += 1
                            total_gates_after += 1
                    gates_to_fuse = []

                current = next_node

            if gates_to_fuse:
                msg = (
                    f"{CLog.R1(self.rule_name)}: Found "
                    f"{CLog.B(len(gates_to_fuse))} consecutive single qubit gates "
                    f"on qubit {CLog.B(qubit)}"
                )
                CLog.log(msg, 2, self.log_level)
                with LogIndentation() as _:
                    CLog.log(f"Gates to fuse: {CLog.B([node.gate for node in gates_to_fuse])}", 2, self.log_level)

                if self._fuse_gates(gates_to_fuse, qubit):
                    compiled = True
                    fusion_count += 1
                    total_gates_after += 1

        if compiled:
            reduction = total_gates_before - total_gates_after
            CLog.log(
                f"{CLog.R1(self.rule_name)}: {CLog.P('successfully fused')} {CLog.B(fusion_count)} groups of gates, "
                f"reduced gate count from {CLog.B(total_gates_before)} to {CLog.B(total_gates_after)} "
                f"({CLog.B(reduction)} gates reduction)",
                1,
                self.log_level,
            )
        else:
            CLog.log(f"{CLog.R1(self.rule_name)}: nothing to change", 1, self.log_level)

        return compiled
