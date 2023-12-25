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

"""HiQASM support module."""

import numpy as np

from mindquantum.utils import fdopen
from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_seed,
    _check_value_should_not_less,
)

HIQASM_GATE_SET = {
    '0.1': {
        'np': ['X', 'Y', 'Z', 'S', 'T', 'H', 'CNOT', 'CZ', 'ISWAP', 'CCNOT'],
        'p': ['RX', 'RY', 'RZ', 'U', 'CRX', 'CRY', 'CRZ', 'CCRX', 'CCRY', 'CCRZ'],
    }
}


def _find_qubit_id(cmd):
    """Find qubit id in openqasm cmd."""
    left = []
    right = []
    for i, j in enumerate(cmd):
        if j == '[':
            left.append(i)
        elif j == ']':
            right.append(i)
    if len(left) != len(right):
        raise ValueError(f"Parsing failed for cmd {cmd}")
    idx = []
    for name_left, name_right in zip(left, right):
        idx.append(int(cmd[name_left + 1 : name_right]))  # noqa: E203
    return idx


def u3(theta, psi, lambd, qubit):
    """Decompose u3 gate."""
    from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
        Circuit,
    )

    circ = Circuit().rz(psi + 3 * np.pi, qubit)
    circ.rx(np.pi / 2, qubit).rz(theta + np.pi, qubit)
    circ.rx(np.pi / 2, qubit).rz(lambd, qubit)
    return circ


def random_hiqasm(n_qubits, gate_num, version='0.1', seed=42):  # pylint: disable=too-many-branches,too-many-statements
    """
    Generate random HiQASM supported circuit.

    Args:
        n_qubits (int): Total number of qubit in this quantum circuit.
        gate_num (int): Total number of gate in this quantum circuit.
        version (str): version of HIQASM. Default: ``'0.1'``.
        seed (int): The random seed to generate this random quantum circuit. Default: ``42``.

    Returns:
        str, quantum circuit in HIQASM format.

    Examples:
        >>> from mindquantum.io import random_hiqasm
        >>> from mindquantum.io import HiQASM
        >>> HiQASM_str = random_hiqasm(2, 5)
        >>> HiQASM = HiQASM()
        >>> circuit = HiQASM.from_string(HiQASM_str)
        >>> circuit
              ┏━━━━━━━━━━━━━┓ ┏━━━━━━━━━━━━━┓ ┏━━━━━━━━━━━━┓ ┍━━━━━━┑
        q0: ──┨ RZ(-2.5134) ┠─┨ RZ(-3.0123) ┠─┨ RX(0.7382) ┠─┤ M k0 ├─────────
              ┗━━━━━━━━━━━━━┛ ┗━━━━━━━━━━━━━┛ ┗━━━━━━┳━━━━━┛ ┕━━━━━━┙
              ┏━━━┓                                  ┃       ┏━━━┓ ┍━━━━━━┑
        q1: ──┨ S ┠──────────────────────────────────■───────┨ Z ┠─┤ M k1 ├───
              ┗━━━┛                                          ┗━━━┛ ┕━━━━━━┙
    """
    _check_int_type("n_qubits", n_qubits)
    _check_value_should_not_less("n_qubits", 1, n_qubits)
    _check_int_type("gate_num", gate_num)
    _check_value_should_not_less("gate_num", 1, gate_num)
    _check_input_type("version", str, version)
    _check_seed(seed)
    np.random.seed(seed)
    if version not in HIQASM_GATE_SET:
        raise NotImplementedError(f"version of {version} not implement yet!")
    gate_set = HIQASM_GATE_SET[version]
    np_set = gate_set['np']
    p_set = gate_set['p']
    if version == '0.1':
        qasm = ['# HIQASM 0.1', '# Instruction stdins', '', f'ALLOCATE q {n_qubits}', 'RESET q']
        if n_qubits == 1:
            np_set = np_set[:6]
            p_set = p_set[:4]
        elif n_qubits == 2:
            np_set = np_set[:9]
            p_set = p_set[:10]
        while len(qasm) - 5 < gate_num:
            g_set = [np_set, p_set][int(np.random.choice([0, 1]))]
            gate = np.random.choice(g_set)
            pval = np.random.uniform(-np.pi, np.pi, 3)

            qubit_list = np.arange(n_qubits)
            np.random.shuffle(qubit_list)
            qubit_strings = [f'q[{idx}]' for idx in qubit_list]
            param_string = ''

            if gate in ['X', 'Y', 'Z', 'S', 'T', 'H']:
                qubit_strings = qubit_strings[:1]
            elif gate in ['CNOT', 'CZ', 'ISWAP']:
                qubit_strings = qubit_strings[:2]
            elif gate == 'CCNOT':
                qubit_strings = qubit_strings[:3]
            elif gate in ['RX', 'RY', 'RZ']:
                qubit_strings = qubit_strings[:1]
                param_string = f' {pval[0]}'
            elif gate == 'U':
                qubit_strings = qubit_strings[:1]
                param_string = f' {",".join(map(str, pval))}'
            elif gate in ['CRX', 'CRY', 'CRZ']:
                qubit_strings = qubit_strings[:2]
                param_string = f' {pval[0]}'
            elif gate in ['CCRX', 'CCRY', 'CCRZ']:
                qubit_strings = qubit_strings[:3]
                param_string = f' {pval[0]}'
            else:
                raise NotImplementedError(f"gate {gate} not implement in HIQASM {version}")
            qasm.append(f'{gate} {",".join(qubit_strings)}{param_string}')
        qasm.append('MEASURE q')
        qasm.append('DEALLOCATE q')
        qasm.append('')
        return '\n'.join(qasm)
    raise NotImplementedError(f'version {version} not implemented')


class HiQASM:
    """
    Convert a circuit to HiQASM format.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.io.qasm import HiQASM
        >>> from mindquantum.core import Circuit
        >>> circuit = Circuit().rx(0.3, 0).z(0, 1).rzz(np.pi, [0, 1])
        >>> HiQASM = HiQASM()
        >>> circuit_str = HiQASM.to_string(circuit)
        >>> print(circuit_str[68: 80])
        CZ q[1],q[0]
        >>> circuit_2 = HiQASM.from_string(circuit_str)
        >>> circuit_2
        q0: ──RX(3/10)────Z────ZZ(π)──
                          │      │
        q1: ──────────────●────ZZ(π)──
    """

    def __init__(self):
        """Initialize a HiQASM object."""
        from mindquantum import Circuit  # pylint: disable=import-outside-toplevel

        self.circuit = Circuit()
        self.cmds = []

    def _filter(self, cmds):
        """Filter empty cmds and head."""
        out = []
        version = None
        n_qubits = None
        for cmd in cmds:
            cmd = cmd.strip()
            if not cmd:
                continue
            if _startswithany(cmd, '#', 'ALLOCATE', 'RESET', 'DEALLOCATE'):
                if cmd.startswith('# HIQASM'):
                    version = cmd.split(' ')[-1]
                if cmd.startswith('ALLOCATE q '):
                    n_qubits = int(cmd.split(' ')[-1])
                continue
            out.append(cmd)
        if n_qubits is None:
            raise ValueError('Can not find qubit number in qasm')
        if version is None:
            raise ValueError('Can not find version in qasm')
        return out, version, n_qubits

    def to_string(self, circuit, version='0.1'):  # pylint: disable=too-many-branches
        """
        Convert circuit to HiQASM.

        Args:
            circuit (Circuit): The quantum circuit you want to translated to HiQASM.
            version (str): The HiQASM version you want to use. Default: ``'0.1'``.

        Returns:
            str, The HiQASM format of input circuit.

        Raises:
            TypeError: if `circuit` is not a Circuit.
            TypeError: if `version` is not a str.
            NotImplementedError: if HiQASM version not implement.
            ValueError: if gate not implement in this version.
        """
        from mindquantum import (  # pylint: disable=import-outside-toplevel
            Circuit,
            gates,
        )

        _check_input_type("circuit", Circuit, circuit)
        _check_input_type("version", str, version)
        if version == '0.1':
            if circuit.parameterized:
                raise ValueError("Cannot convert parameterized circuit to HIQASM")
            self.circuit = circuit
            self.cmds = [f"# HIQASM {version}", "# Instruction stdins", "", f'ALLOCATE q {circuit.n_qubits}', 'RESET q']
            for gate in circuit:
                ctrl_qubits = gate.ctrl_qubits
                n_ctrl_qubits = len(ctrl_qubits)
                obj_qubits = gate.obj_qubits
                n_obj_qubits = len(obj_qubits)
                if n_ctrl_qubits > 2:
                    raise ValueError(f"HIQASM do not support more than two control qubits gate: {gate}")
                if n_obj_qubits > 2:
                    raise ValueError(f"HIQASM do not support more than two object qubit gate: {gate}")
                if self._to_string_non_parametric(gate, ctrl_qubits, obj_qubits, version):
                    pass
                elif self._to_string_parametric(gate, ctrl_qubits, obj_qubits, version):
                    pass
                elif isinstance(gate, gates.ISWAPGate):
                    if n_ctrl_qubits == 0:
                        self.cmds.append(f'ISWAP q[{obj_qubits[0]}],q[{obj_qubits[1]}]')
                    else:
                        _not_implement(version, gate)
                elif isinstance(gate, gates.Measure):
                    if n_ctrl_qubits == 0:
                        self.cmds.append(f'MEASURE q[{obj_qubits[0]}]')
                    else:
                        _not_implement(version, gate)
                else:
                    _not_implement(version, gate)
            self.cmds.append('DEALLOCATE q')
            self.cmds.append('')
        else:
            raise NotImplementedError(f"version of {version} for HiQASM not implement yet.")
        return '\n'.join(self.cmds)

    def _to_string_non_parametric(self, gate, ctrl_qubits, obj_qubits, version):  # pylint: disable=too-many-branches
        """Conversion of simple gates to string."""
        from mindquantum.core import (  # pylint: disable=import-outside-toplevel,cyclic-import
            gates,
        )

        n_ctrl_qubits = len(ctrl_qubits)

        if isinstance(gate, gates.XGate):
            if n_ctrl_qubits == 0:
                self.cmds.append(f'X q[{obj_qubits[0]}]')
            elif n_ctrl_qubits == 1:
                self.cmds.append(f'CNOT q[{ctrl_qubits[0]}],q[{obj_qubits[0]}]')
            elif n_ctrl_qubits == 2:
                self.cmds.append(f'CCNOT q[{ctrl_qubits[0]}],q[{ctrl_qubits[1]}],q[{obj_qubits[0]}]')
            else:
                _not_implement(version, gate)
        elif isinstance(gate, gates.YGate):
            if n_ctrl_qubits == 0:
                self.cmds.append(f'Y q[{obj_qubits[0]}]')
            else:
                _not_implement(version, gate)
        elif isinstance(gate, gates.ZGate):
            if n_ctrl_qubits == 0:
                self.cmds.append(f'Z q[{obj_qubits[0]}]')
            elif n_ctrl_qubits == 1:
                self.cmds.append(f'CZ q[{ctrl_qubits[0]}],q[{obj_qubits[0]}]')
            else:
                _not_implement(version, gate)
        elif isinstance(gate, gates.SGate):
            if n_ctrl_qubits == 0:
                if gate.hermitianed:
                    _not_implement(version, gate)
                self.cmds.append(f'S q[{obj_qubits[0]}]')
            else:
                _not_implement(version, gate)
        elif isinstance(gate, gates.TGate):
            if n_ctrl_qubits == 0:
                if gate.hermitianed:
                    _not_implement(version, gate)
                self.cmds.append(f'T q[{obj_qubits[0]}]')
            else:
                _not_implement(version, gate)
        elif isinstance(gate, gates.HGate):
            if n_ctrl_qubits == 0:
                self.cmds.append(f'H q[{obj_qubits[0]}]')
            else:
                _not_implement(version, gate)
        else:
            return False
        return True

    def _to_string_parametric(self, gate, ctrl_qubits, obj_qubits, version):
        """Conversion of parametric gates to string."""
        from mindquantum.core import (  # pylint: disable=import-outside-toplevel,cyclic-import
            gates,
        )

        n_ctrl_qubits = len(ctrl_qubits)
        if gate.parameterized:
            raise ValueError(f"Cannot convert parameterized gate {gate} to hiqasm format.")

        if isinstance(gate, (gates.RX, gates.RY, gates.RZ)):
            if n_ctrl_qubits == 0:
                self.cmds.append(f'{gate.name} q[{obj_qubits[0]}] {gate.coeff.const}')
            elif n_ctrl_qubits == 1:
                self.cmds.append(f'C{gate.name} q[{ctrl_qubits[0]}],q[{obj_qubits[0]}] {gate.coeff.const}')
            elif n_ctrl_qubits == 2:
                self.cmds.append(
                    f'CC{gate.name} q[{ctrl_qubits[0]}],q[{ctrl_qubits[1]}],q[{obj_qubits[0]}] {gate.coeff.const}'
                )
            else:
                _not_implement(version, gate)
        elif isinstance(gate, (gates.Rxx, gates.Ryy, gates.Rzz)):
            gate_name = gate.name[1:].upper()
            if n_ctrl_qubits == 0:
                self.cmds.append(f'{gate_name} q[{obj_qubits[0]}],q[{obj_qubits[1]}] {gate.coeff.const}')
            else:
                _not_implement(version, gate)
        else:
            return False
        return True

    def from_string(self, string):
        """
        Read a HiQASM string.

        Args:
            string (str): The HiQASM string of a Circuit.

        Returns:
            Circuit, The quantum circuit translated from HiQASM string.
        """
        _check_input_type('string', str, string)
        cmds = string.split('\n')
        self.cmds, version, n_qubits = self._filter(cmds)
        if version == '0.1':
            self._trans_v01(self.cmds, n_qubits)
        else:
            raise ValueError(f'HIQASM {version} not implement yet')
        return self.circuit

    def from_file(self, file_name):
        """
        Read a HiQASM file.

        Args:
            file_name (str): The path of file that stored quantum circuit in HiQASM format.

        Returns:
            Circuit, the quantum circuit translated from HiQASM file.
        """
        _check_input_type('file_name', str, file_name)
        with fdopen(file_name, 'r') as fd:
            cmds = fd.readlines()
        self.from_string('\n'.join(cmds))
        return self.circuit

    def to_file(self, file_name, circuit, version='0.1'):
        """
        Convert a quantum circuit to HiQASM format and save in file.

        Args:
            file_name (str): The file name you want to save the HiQASM file.
            circuit (Circuit): The Circuit you want to convert.
            version (str): The version of HiQASM. Default: ``'0.1'``.

        Raises:
            TypeError: if `file_name` is not a str.
            TypeError: if `circuit` is not a Circuit.
            TypeError: if `version` is not a str.
        """
        from mindquantum.core import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Circuit,
        )

        _check_input_type('file_name', str, file_name)
        _check_input_type('circuit', Circuit, circuit)
        _check_input_type('version', str, version)
        circuit_string = self.to_string(circuit, version)
        with fdopen(file_name, 'w') as fd:
            fd.writelines(circuit_string)
        print(f"write circuit to {file_name} finished!")

    def _trans_v01(self, cmds, n_qubits):  # pylint: disable=too-many-branches
        """Trans method for HiQASM version 0.1."""
        from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Circuit,
            gates,
        )

        self.circuit = Circuit()
        for cmd in cmds:
            qubit = _find_qubit_id(cmd)
            if cmd.startswith('CNOT '):
                self.circuit.x(qubit[1], qubit[0])
            elif cmd.startswith('CZ '):
                self.circuit.z(qubit[1], qubit[0])
            elif cmd.startswith('ISWAP '):
                self.circuit += gates.ISWAP.on(qubit[:2])
            elif cmd.startswith('CCNOT '):
                self.circuit.x(qubit[-1], qubit[:2])
            elif cmd.startswith('CRX '):
                self.circuit.rx(*_extr_parameter(cmd), qubit[1], qubit[0])
            elif cmd.startswith('CRY '):
                self.circuit.ry(*_extr_parameter(cmd), qubit[1], qubit[0])
            elif cmd.startswith('CRZ '):
                self.circuit.rz(*_extr_parameter(cmd), qubit[1], qubit[0])
            elif cmd.startswith('CCRX '):
                self.circuit.rx(*_extr_parameter(cmd), qubit[-1], qubit[:2])
            elif cmd.startswith('CCRY '):
                self.circuit.ry(*_extr_parameter(cmd), qubit[-1], qubit[:2])
            elif cmd.startswith('CCRZ '):
                self.circuit.rz(*_extr_parameter(cmd), qubit[-1], qubit[:2])
            elif cmd.startswith('MEASURE '):
                qubit = _find_qubit_id(cmd)
                if qubit:
                    self.circuit.measure(f'k{self.circuit.all_measures.size}', qubit[0])
                else:
                    for midx in range(n_qubits):
                        self.circuit.measure(f'k{self.circuit.all_measures.size}', midx)
            elif self._trans_v01_single_qubit(cmd, qubit[0]):
                pass
            else:
                raise ValueError(f"transfer cmd {cmd} not implement yet!")

    def _trans_v01_single_qubit(self, cmd, qubit):
        """Trans method for HiQASM version 0.1 (single-qubit gates)."""
        from mindquantum.core import (  # pylint: disable=import-outside-toplevel,cyclic-import
            gates,
        )

        if cmd.startswith('H '):
            self.circuit.h(qubit)
        elif cmd.startswith('X '):
            self.circuit.x(qubit)
        elif cmd.startswith('Y '):
            self.circuit.y(qubit)
        elif cmd.startswith('Z '):
            self.circuit.z(qubit)
        elif cmd.startswith('S '):
            self.circuit += gates.S.on(qubit)
        elif cmd.startswith('T '):
            self.circuit += gates.T.on(qubit)
        elif cmd.startswith('U '):
            self.circuit += u3(*_extr_parameter(cmd), qubit)
        elif cmd.startswith('RX '):
            self.circuit.rx(*_extr_parameter(cmd), qubit)
        elif cmd.startswith('RY '):
            self.circuit.ry(*_extr_parameter(cmd), qubit)
        elif cmd.startswith('RZ '):
            self.circuit.rz(*_extr_parameter(cmd), qubit)
        else:
            return False
        return True


def _extr_parameter(cmd):
    """Extra parameter for parameterized gate in HiQASM cmd."""
    return [float(i) for i in cmd.split(' ')[-1].split(',')]


def _startswithany(cmd, *s):
    """Checkout whether cmd starts with any string in s."""
    for i in s:
        if cmd.startswith(i):
            return True
    return False


def _not_implement(version, gate):
    """Not implemented error."""
    raise ValueError(f'{gate} not implement in HiQASM {version}')
