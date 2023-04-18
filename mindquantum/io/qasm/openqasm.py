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

"""OpenQASM support module."""

import numpy as np

from mindquantum.utils import fdopen


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


def _extr_parameter(cmd):
    """Extra parameter for parameterized gate in openqasm cmd."""
    param_start = cmd.find('(')
    idx = cmd.find(')')
    if param_start == -1 or idx == -1:
        raise ValueError(f"no parameter found in cmd {cmd}")
    all_expre = cmd[param_start + 1 : idx]  # noqa: E203
    all_expre = all_expre.split(',')
    out = []
    for expre in all_expre:
        if 'pi' in expre:
            expre = expre.replace('pi', str(np.pi))
        if '*' in expre:
            tmp = expre.split('*')
            if len(tmp) != 2:
                raise ValueError(f"cannot parse cmd {cmd}")
            expre = str(float(tmp[0]) * float(tmp[1]))
        if '/' in expre:
            tmp = expre.split('/')
            if len(tmp) != 2:
                raise ValueError(f"cannot parse cmd {cmd}")
            expre = str(float(tmp[0]) / float(tmp[1]))
        out.append(float(expre))
    return out[0] if len(all_expre) == 1 else out


def u3(theta, psi, lambd, qubit):
    """Decompose u3 gate."""
    from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
        Circuit,
    )

    circ = Circuit().rz(psi + 3 * np.pi, qubit)
    circ.rx(np.pi / 2, qubit).rz(theta + np.pi, qubit)
    circ.rx(np.pi / 2, qubit).rz(lambd, qubit)
    return circ


def u1(lambd, qubit):
    """Openqasm u1 gate."""
    from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
        Circuit,
    )

    return Circuit().rz(lambd, qubit)


def isgateinstance(gate, gates):
    """Check whether gate is any instance of supported gate type."""
    if isinstance(gates, list):
        gates = (gates,)
    for gate_list in gates:
        for gate_set in gate_list:
            if isinstance(gate, gate_set):
                return True
    return False


def _gate_not_implement_to_openqasm(gate):
    """Raise an error that cannot convert gate to openqasm."""
    raise ValueError(f"cannot convert {gate} to openqasm.")


class OpenQASM:
    """
    Convert a circuit to openqasm format.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core import Circuit
        >>> from mindquantum.io import OpenQASM
        >>> circuit = Circuit().rx(0.3, 0).z(0, 1).zz(np.pi, [0, 1])
        >>> openqasm = OpenQASM()
        >>> circuit_str = openqasm.to_string(circuit)
        >>> circuit_str[47:60]
        'rx(0.3) q[0];'
    """

    def __init__(self):
        """Initialize an OpenQASM object."""
        from mindquantum import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Circuit,
        )

        self.circuit = Circuit()
        self.cmds = []

    def to_string(self, circuit, version="2.0"):  # pylint: disable=R0912,R0914,R0915
        """
        Convert circuit to openqasm.

        Args:
            circuit (Circuit): The quantum circuit you want to translated to openqasm.
            version (str): The openqasm version you want to use. Default: ``'2.0'``.

        Returns:
            str, The openqasm format of input circuit.

        Raises:
            TypeError: if `circuit` is not a Circuit.
            TypeError: if `version` is not a str.
            NotImplementedError: if openqasm version not implement.
            ValueError: if gate not implement in this version.
        """
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum import Circuit, gates

        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit requires Circuit, but get {type(circuit)}.")
        if not isinstance(version, str):
            raise TypeError(f"version requires a str, but get {type(version)}")
        single_np = [gates.XGate, gates.YGate, gates.ZGate, gates.HGate, gates.SGate, gates.TGate]
        single_p = [gates.RX, gates.RY, gates.RZ, gates.PhaseShift]
        double_np = [gates.SWAPGate, gates.CNOTGate]
        double_p = [gates.Rxx, gates.Ryy, gates.Rzz]
        if version == "2.0":
            self.circuit = circuit
            self.cmds = [f"OPENQASM {version};", "include \"qelib1.inc\";"]
            self.cmds.append(f"qreg q[{circuit.n_qubits}];")
            for gate in self.circuit:
                if isgateinstance(gate, (single_np, single_p)):
                    if isinstance(gate, gates.XGate):
                        if not gate.ctrl_qubits:
                            self.cmds.append(f"x q[{gate.obj_qubits[0]}];")
                            continue
                        if len(gate.ctrl_qubits) == 1:
                            self.cmds.append(f"cx q[{gate.ctrl_qubits[0]}],q[{gate.obj_qubits[0]}];")
                            continue
                        if len(gate.ctrl_qubits) == 2:
                            ctrl0, ctrl1 = gate.ctrl_qubits
                            obj0 = gate.obj_qubits[0]
                            self.cmds.append(f"ccx q[{ctrl0}],q[{ctrl1}],q[{obj0}];")
                            continue
                        _gate_not_implement_to_openqasm(gate)
                    if isinstance(gate, gates.TGate):
                        t_name = 'tdg' if gate.hermitianed else 't'
                        if not gate.ctrl_qubits:
                            self.cmds.append(f"{t_name} q[{gate.obj_qubits[0]}];")
                            continue
                        _gate_not_implement_to_openqasm(gate)
                    if isinstance(gate, gates.SGate):
                        s_name = 'sdg' if gate.hermitianed else 's'
                        if not gate.ctrl_qubits:
                            self.cmds.append(f"{s_name} q[{gate.obj_qubits[0]}];")
                            continue
                        _gate_not_implement_to_openqasm(gate)
                    if isinstance(gate, gates.HGate):
                        if not gate.ctrl_qubits:
                            self.cmds.append(f"h q[{gate.obj_qubits[0]}];")
                            continue
                        if len(gate.ctrl_qubits) == 1:
                            self.cmds.append(f"ch q[{gate.ctrl_qubits[0]}],q[{gate.obj_qubits[0]}];")
                            continue
                        _gate_not_implement_to_openqasm(gate)
                    if isinstance(gate, gates.PhaseShift):
                        param = gate.coeff
                        if not param.is_const():
                            raise ValueError(f"Cannot convert parameterized gate {gate} to OpenQASM.")
                        if not gate.ctrl_qubits:
                            self.cmds.append(f"p({param.const}) q[{gate.obj_qubits[0]}];")
                            continue
                        if len(gate.ctrl_qubits) == 1:
                            self.cmds.append(f"cp({param.const}) q[{gate.ctrl_qubits[0]}],q[{gate.obj_qubits[0]}];")
                            continue
                        _gate_not_implement_to_openqasm(gate)
                    if len(gate.ctrl_qubits) > 1:
                        raise ValueError(f"Multiple control for gate {gate} not implement")
                    if isgateinstance(gate, single_np):
                        obj = gate.obj_qubits[0]
                        if gate.ctrl_qubits:
                            ctrl = gate.ctrl_qubits[0]
                            self.cmds.append(f"c{gate.name.lower()} q[{ctrl}],q[{obj}];")
                        else:
                            self.cmds.append(f"{gate.name.lower()} q[{obj}];")
                    else:
                        obj = gate.obj_qubits[0]
                        param = gate.coeff
                        if not param.is_const():
                            raise ValueError(f"Cannot convert parameterized gate {gate} to OpenQASM.")
                        param = param.const
                        if gate.ctrl_qubits:
                            ctrl = gate.ctrl_qubits[0]
                            self.cmds.append(f"c{gate.name.lower()}({param}) q[{ctrl}],q[{obj}];")
                        else:
                            self.cmds.append(f"{gate.name.lower()}({param}) q[{obj}];")
                if isgateinstance(gate, (double_np, double_p)):
                    if gate.ctrl_qubits:
                        raise ValueError(f"control two qubits gate {gate} not implement")
                    if isgateinstance(gate, double_np):
                        obj = gate.obj_qubits
                        if isinstance(gate, gates.SWAPGate):
                            self.cmds.append(f"swap q[{obj[0]}],q[{obj[1]}];")
                        if isinstance(gate, gates.CNOTGate):
                            self.cmds.append(f"cx q[{obj[1]}],q[{obj[0]}];")
                    else:
                        obj = gate.obj_qubits
                        param = ",".join([str(i.const) for i in gate.get_parameters()])
                        self.cmds.append(f"r{gate.name[1:].lower()}({param}) q[{obj[0]}],q[{obj[1]}];")
        else:
            raise NotImplementedError(f"openqasm version {version} not implement")
        return '\n'.join(self.cmds)

    def to_file(self, file_name, circuit, version="2.0"):
        """
        Convert a quantum circuit to openqasm format and save in file.

        Args:
            file_name (str): The file name you want to save the openqasm file.
            circuit (Circuit): The Circuit you want to convert.
            version (str): The version of openqasm. Default: ``'2.0'``.

        Raises:
            TypeError: if `file_name` is not a str.
            TypeError: if `circuit` is not a Circuit.
            TypeError: if `version` is not a str.
        """
        from mindquantum.core import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Circuit,
        )

        if not isinstance(file_name, str):
            raise TypeError(f'file_name requires a str, but get {type(file_name)}')
        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit requires a Circuit, but get {type(circuit)}")
        if not isinstance(version, str):
            raise TypeError(f'version requires a str, but get {type(version)}')
        with fdopen(file_name, 'w') as fd:
            fd.writelines(self.to_string(circuit, version))
        print(f"write circuit to {file_name} finished!")

    def from_file(self, file_name):
        """
        Read a openqasm file.

        Args:
            file_name (str): The path of file that stored quantum circuit in openqasm format.

        Returns:
            Circuit, the quantum circuit translated from openqasm file.
        """
        with fdopen(file_name, 'r') as fd:
            cmds = fd.readlines()
        self.cmds, version = self._filter(cmds)
        if version == '2.0':
            self._trans_v2(self.cmds)
        else:
            raise ValueError(f"OPENQASM {version} not implement yet")
        return self.circuit

    def _filter(self, cmds):
        """Filter empty cmds and head."""
        out = []
        version = None
        for cmd in cmds:
            cmd = cmd.strip()
            if not cmd or cmd.startswith('//') or cmd.startswith('include') or cmd.startswith("qreg"):
                pass
            elif cmd.startswith('OPENQASM'):
                version = cmd.split(' ')[-1][:-1]
            else:
                out.append(cmd[:-1])
        return out, version

    def _trans_v2(self, cmds):  # pylint: disable=too-many-branches
        """Trans method for openqasm version 2."""
        # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum import Circuit
        from mindquantum.core.circuit import controlled

        self.circuit = Circuit()
        for cmd in cmds:
            qubit = _find_qubit_id(cmd)
            if cmd.startswith("h "):
                self.circuit.h(qubit[0])
            elif cmd.startswith("x "):
                self.circuit.x(qubit[0])
            elif cmd.startswith("y "):
                self.circuit.y(qubit[0])
            elif cmd.startswith("z "):
                self.circuit.z(qubit[0])
            elif cmd.startswith("cx "):
                self.circuit.x(qubit[1], qubit[0])
            elif cmd.startswith("cz "):
                self.circuit.z(*qubit[::-1])
            elif cmd.startswith("rz("):
                self.circuit.rz(_extr_parameter(cmd), qubit[0])
            elif cmd.startswith("ry("):
                self.circuit.ry(_extr_parameter(cmd), qubit[0])
            elif cmd.startswith("rx("):
                self.circuit.rx(_extr_parameter(cmd), qubit[0])
            elif cmd.startswith("u3("):
                self.circuit += u3(*_extr_parameter(cmd), qubit[0])
            elif cmd.startswith("cu1("):
                self.circuit += controlled(u1(_extr_parameter(cmd), qubit[1]))(qubit[0])
            else:
                raise ValueError(f"transfer cmd {cmd} not implement yet!")
