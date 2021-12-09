# -*- coding: utf-8 -*-
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
"""openqasm"""
import numpy as np


def _find_qubit_id(cmd):
    """Find qubit id in openqasm cmd"""
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
    for l, r in zip(left, right):
        idx.append(int(cmd[l + 1:r]))
    return idx


def _extr_parameter(cmd):
    """extra parameter for parameterized gate in openqasm cmd"""
    l = cmd.find('(')
    r = cmd.find(')')
    if l == -1 or r == -1:
        raise ValueError(f"no parameter found in cmd {cmd}")
    all_expre = cmd[l + 1:r]
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


def u3(theta, psi, lambd, q):
    """decomp u3 gate"""
    from mindquantum import Circuit
    circ = Circuit().rz(psi + 3 * np.pi, q)
    circ.rx(np.pi / 2, q).rz(theta + np.pi, q)
    circ.rx(np.pi / 2, q).rz(lambd, q)
    return circ


def u1(lambd, q):
    """openqasm u1 gate"""
    from mindquantum import Circuit
    return Circuit().rz(lambd, q)


def isgateinstance(gate, gates):
    """Check whether gate is any instance of supported gate type"""
    if isinstance(gates, list):
        gates = (gates, )
    for gate_test in gates:
        for g in gate_test:
            if isinstance(gate, g):
                return True
    return False


class OpenQASM:
    """
    Convert a circuit to openqasm format

    Examples:
        >>> import numpy as np
        >>> from mindquantum.io.qasm import OpenQASM
        >>> from mindquantum.core import Circuit
        >>> circuit = Circuit().rx(0.3, 0).z(0, 1).zz(np.pi, [0, 1])
        >>> openqasm = OpenQASM()
        >>> circuit_str = openqasm.to_string(circuit)
        >>> circuit_str[47:60]
        'rx(0.3) q[0];'
    """
    def __init__(self):
        from mindquantum import Circuit
        self.circuit = Circuit()
        self.cmds = []

    def to_string(self, circuit, version="2.0"):
        """
        Convert circuit to hiqasm.

        Args:
            circuit (Circuit): The quantum circuit you want to translated to openqasm.
            version (str): The HiQASM version you want to use. Default: '2.0'.

        Returns:
            str, The HiQASM format of input circuit.

        Raises:
            TypeError: if circuit is not a Circuit.
            TypeError: if version is not a str.
            NotImplementedError: if openqasm version not implement.
            ValueError: if gate not implement in this version.
        """
        from mindquantum import gates as G
        from mindquantum.core import Circuit
        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit requires Circuit, but get {type(circuit)}.")
        if not isinstance(version, str):
            raise TypeError(f"version requires a str, but get {type(version)}")
        single_np = [G.XGate, G.YGate, G.ZGate]
        single_p = [G.RX, G.RY, G.RZ, G.PhaseShift]
        double_np = [G.SWAPGate, G.CNOTGate]
        double_p = [G.XX, G.YY, G.ZZ]
        if version == "2.0":
            self.circuit = circuit
            self.cmds = [f"OPENQASM {version};", "include \"qelib1.inc\";"]
            self.cmds.append(f"qreg q[{circuit.n_qubits}];")
            for gate in self.circuit:
                if isgateinstance(gate, (single_np, single_p)):
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
                        p = gate.coeff
                        if gate.ctrl_qubits:
                            ctrl = gate.ctrl_qubits[0]
                            self.cmds.append(f"c{gate.name.lower()}({p}) q[{ctrl}],q[{obj}];")
                        else:
                            self.cmds.append(f"{gate.name.lower()}({p}) q[{obj}];")
                if isgateinstance(gate, (double_np, double_p)):
                    if gate.ctrl_qubits:
                        raise ValueError(f"control two qubits gate {gate} not implement")
                    if isgateinstance(gate, double_np):
                        obj = gate.obj_qubits
                        if isinstance(gate, G.SWAPGate):
                            self.cmds.append(f"cx q[{obj[1]}],q[{obj[0]}];")
                            self.cmds.append(f"cx q[{obj[0]}],q[{obj[1]}];")
                            self.cmds.append(f"cx q[{obj[1]}],q[{obj[0]}];")
                        if isinstance(gate, G.CNOTGate):
                            self.cmds.append(f"cx q[{obj[1]}],q[{obj[0]}];")
                    else:
                        obj = gate.obj_qubits
                        p = gate.coeff
                        self.cmds.append(f"{gate.name.lower()}({p}) q[{obj[0]}],q[{obj[1]}];")
        else:
            raise NotImplementedError(f"openqasm version {version} not implement")
        return '\n'.join(self.cmds)

    def to_file(self, file_name, circuit, version="2.0"):
        """
        Convert a quantum circuit to openqasm format and save in file.

        Args:
            file_name (str): The file name you want to save the openqasm file.
            circuit (Circuit): The Circuit you want to convert.
            version (str): The version of openqasm. Default: '2.0'.

        Raises:
            TypeError: if `file_name` is not a str.
            TypeError: if `circuit` is not a Circuit.
            TypeError: if `version` is not a str.
        """
        from mindquantum.core import Circuit
        if not isinstance(file_name, str):
            raise TypeError(f'file_name requires a str, but get {type(file_name)}')
        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit requires a Circuit, but get {type(circuit)}")
        if not isinstance(version, str):
            raise TypeError(f'version requires a str, but get {type(version)}')
        cs = self.to_string(circuit, version)
        with open(file_name, 'w') as f:
            f.writelines(cs)
        print(f"write circuit to {file_name} finished!")

    def from_file(self, file_name):
        """
        Read a openqasm file.

        Args:
            file_name (str): The path of file that stored quantum circuit in openqasm format.

        Returns:
            Circuit, the quantum circuit translated from openqasm file.
        """
        with open(file_name, 'r') as f:
            cmds = f.readlines()
        self.cmds, version = self._filter(cmds)
        if version == '2.0':
            self._trans_v2(self.cmds)
        else:
            raise ValueError(f"OPENQASM {version} not implement yet")

    def _filter(self, cmds):
        """
        filter empty cmds and head.
        """
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

    def _trans_v2(self, cmds):
        """
        trans method for openqasm version 2
        """
        from mindquantum import Circuit
        from mindquantum.core.circuit import controlled
        self.circuit = Circuit()
        for cmd in cmds:
            q = _find_qubit_id(cmd)
            if cmd.startswith("h "):
                self.circuit.h(q[0])
            elif cmd.startswith("x "):
                self.circuit.x(q[0])
            elif cmd.startswith("y "):
                self.circuit.y(q[0])
            elif cmd.startswith("cx "):
                self.circuit.x(q[1], q[0])
            elif cmd.startswith("cz "):
                self.circuit.z(*q[::-1])
            elif cmd.startswith("rz("):
                self.circuit.rz(_extr_parameter(cmd), q[0])
            elif cmd.startswith("ry("):
                self.circuit.ry(_extr_parameter(cmd), q[0])
            elif cmd.startswith("rx("):
                self.circuit.rx(_extr_parameter(cmd), q[0])
            elif cmd.startswith("u3("):
                self.circuit += u3(*_extr_parameter(cmd), q[0])
            elif cmd.startswith("cu1("):
                self.circuit += controlled(u1(_extr_parameter(cmd), q[1]))(q[0])
            else:
                raise ValueError(f"transfer cmd {cmd} not implement yet!")
