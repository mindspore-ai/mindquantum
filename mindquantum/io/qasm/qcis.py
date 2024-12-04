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
"""QCIS support module."""
# pylint: disable=import-outside-toplevel
import re
from typing import List, Tuple

import numpy as np

from mindquantum.utils import fdopen
from mindquantum.utils.type_value_check import _check_input_type
from mindquantum.core.parameterresolver import ParameterResolver

def _floatable(string) -> bool:
    """Whether the input string can be converted to a float number.
    This function will return `Ture` if the input string is composed only of digits, point, star and slash.

    Examples:
        >>> _floatable(2*a)
        False
        >>> _floatable("2/13")
        True
    """
    num_chars = re.findall(r"([\d\.\+\-\*\/])", string)
    return len(num_chars) == len(string)

def _to_float(string) -> float:
    """convert the input floatable string to a float number.

    Examples:
        >>> _to_float("-12/3.0")
        -4.0
        >>> _to_float("12")
        12.0
    """
    tmp = re.findall(r"([^\/]+)", string)
    if "/" in string:
        res = float(tmp[0]) / float(tmp[1])
    else:
        res = float(tmp[0])
    return res

def _string_to_param(string) -> ParameterResolver:
    """Convert the input string to a ParameterResolver object.

    Examples:
        >>> string0 = "2.0*pi-1.5*π"
        >>> _string_to_param(string0)
        π/2
        >>> string1 = a = "-2√2*a+3.4*b+2.2*3+0.5"
        >>> _string_to_param(string1)
        -2√2*a + 17/5*b + 7.1
    """
    if "√" in string:
        tmp = re.findall(r"(\d+\√\d)", string)
        if tmp: # there are digits before √.
            both = tmp[0].split("√")
            left = _to_float(both[0])
            right = _to_float(both[1])
            string = string.replace(tmp[0], str(left*np.sqrt(right)))
        else: # # there are not digits before √.
            tmp = re.findall(r"√(\d)", string)
            num = _to_float(tmp[0])
            string = string.replace("√" + tmp[0], str(np.sqrt(num)))
    # Add a + before and a space after to the string
    # to make it easier to split it with regular expression.
    if not string.startswith("-"):
        string = "+" + string
    string += " "
    # covert the pi or π to a float number
    string = string.replace("pi", str(np.pi)).replace("π", str(np.pi))
    terms = re.findall(r"(.+?)(?=[\+\-\s])", string)
    data = {}
    const = 0.
    for term in terms:
        tmp = re.findall(r"([^\*]+)", term)
        if len(tmp) == 1:
            tmp = tmp[0]
            if _floatable(tmp):
                const += _to_float(tmp)
            else:
                if tmp[0] == "-":
                    data[tmp[1:]] = -1
                else:
                    data[tmp[1:]] = 1
        else:
            if sum(_floatable(i) for i in tmp) == 2:
                const += _to_float(tmp[0]) * _to_float(tmp[1])
            else:
                if _floatable(tmp[0]):
                    data[tmp[1]] = _to_float(tmp[0])
                else:
                    data[tmp[0]] = _to_float(tmp[1])
    return ParameterResolver(data=data, const=const)

def _failed_from_qcis(gate_name):
    raise NotImplementedError(f"Cannot convert the {gate_name} from QCIS to gate.")

def _failed_to_qcis(gate):
    raise NotImplementedError(f"Cannot convert the {gate} from gate to QCIS.")

def _valid_angle(angle) -> float:
    """Restrict the angle to the interval [-pi, pi]."""
    return np.mod(angle + np.pi, np.pi * 2) - np.pi

class QCIS:
    """Convert a circuit to qcis format.

    Examples:
        >>> circ = Circuit()
        >>> circ.x(0).z(1,0).rx({"a":-2*np.sqrt(2)}, 0).sx(0).barrier()
        >>> circ.ry(ParameterResolver(data={'theta':-np.pi}, const=np.pi), 1)
        >>> string= QCIS().to_string(circ)
        >>> print(string)
        X Q0
        CZ Q0 Q1
        RX Q0 -2√2*a
        X2P Q0
        B Q0
        B Q1
        RY Q1 -π*theta + π
        >>> circ1 = QCIS().from_string(string)
        >>> print(circ1)
              ┏━━━┓       ┏━━━━━━━━━━━━┓ ┏━━━━┓
        q0: ──┨╺╋╸┠───■───┨ RX(-2√2*a) ┠─┨ SX ┠─▓────────────────────────
              ┗━━━┛   ┃   ┗━━━━━━━━━━━━┛ ┗━━━━┛ ▓
                    ┏━┻━┓                       ▓ ┏━━━━━━━━━━━━━━━━━━┓
        q1: ────────┨ Z ┠───────────────────────▓─┨ RY(-π*theta + π) ┠───
                    ┗━━━┛                         ┗━━━━━━━━━━━━━━━━━━┛
    """
    def __init__(self):
        from mindquantum.core import gates as G # pylint: disable=import-outside-toplevel,cyclic-import
        from mindquantum.core.circuit import Circuit # pylint: disable=import-outside-toplevel,cyclic-import
        self.gate_class = G
        self.circuit_class = Circuit

    def _get_gate_terms(self, string) -> List[str]:
        """Extract lines containing gate."""
        terms = string.split("\n")
        terms = [_.strip() for _ in terms if _.strip()]
        return terms

    def _get_gate_name(self, term) -> str:
        """Get the gate name from the term."""
        name = re.findall(r"^([\w]+)(?=\s)", term)[0]
        return name

    def _get_qubits(self, term) -> List[int]:
        """Get the qubits acted upon by the gate from the term."""
        qubits = re.findall(r"(?<=Q)(\d+)", term)
        return list(reversed([int(i) for i in qubits])) # [obj_qubit, ctrl_qubit]

    def _get_gate_info(self, term) -> Tuple[str, List[int], List[ParameterResolver]]:
        """Get the gate information (name, params and qubits) according to the term."""
        term = term.strip()
        gate_name = self._get_gate_name(term)
        qubits = self._get_qubits(term)
        params = []
        if gate_name.startswith("R"):
            param = "".join(term.split(" ")[2:])
            params = [_string_to_param(param)]
        return gate_name, qubits, params

    def _to_gate(self, gate_name: str, qubits: List[int], params: List[ParameterResolver]):
        """Create a gate according to the given information."""
        if gate_name in ["Z", "CZ"]:
            return self.gate_class.Z.on(*qubits)
        if gate_name == "I":
            return self.gate_class.I.on(*qubits)
        if gate_name == "X":
            return self.gate_class.X.on(*qubits)
        if gate_name == "Y":
            return self.gate_class.Y.on(*qubits)
        if gate_name == "H":
            return self.gate_class.H.on(*qubits)
        if gate_name == "S":
            return self.gate_class.S.on(*qubits)
        if gate_name == "SD":
            return self.gate_class.S.hermitian().on(*qubits)
        if gate_name == "T":
            return self.gate_class.T.on(*qubits)
        if gate_name == "TD":
            return self.gate_class.T.hermitian().on(*qubits)
        if gate_name == "B":
            return self.gate_class.BarrierGate().on(*qubits)
        if gate_name == "X2P":
            return self.gate_class.SX.on(*qubits)
        if gate_name == "X2M":
            return self.gate_class.SX.hermitian().on(*qubits)
        if gate_name == "Y2P":
            return self.gate_class.RY(np.pi/2).on(*qubits)
        if gate_name == "Y2M":
            return self.gate_class.RY(-np.pi/2).on(*qubits)
        if gate_name == "RX":
            return self.gate_class.RX(*params).on(*qubits)
        if gate_name == "RY":
            return self.gate_class.RY(*params).on(*qubits)
        if gate_name == "RZ":
            return self.gate_class.RZ(*params).on(*qubits)
        _failed_from_qcis(gate_name)
        return None

    def from_file(self, file_name: str):
        """Read a qcis file.

        Args:
            file_name (str): The path of file that stored quantum circuit in qcis format.

        Returns:
            Circuit, the quantum circuit translated from qcis file.
        """
        with fdopen(file_name, 'r') as fd:
            terms = fd.readlines()
        return self.from_string('\n'.join(terms))

    def from_string(self, string: str):
        r"""Read a QCIS string.

        Args:
            string (str): The QCIS string of Circuit.

        Returns:
            :class:`~.core.circuit.Circuit`, the quantum circuit translated from QCIS string.

        Examples:
            >>> string = "X Q0 \nCZ Q0 Q1 \nRX Q0 -2√2*a\nX2P Q0 \nB Q0\nB Q1\nRY Q1 -π*theta + π"
            >>> circ = QCIS().from_string(string)
            >>> print(circ)
                  ┏━━━┓       ┏━━━━━━━━━━━━┓ ┏━━━━┓
            q0: ──┨╺╋╸┠───■───┨ RX(-2√2*a) ┠─┨ SX ┠─▓────────────────────────
                  ┗━━━┛   ┃   ┗━━━━━━━━━━━━┛ ┗━━━━┛ ▓
                        ┏━┻━┓                       ▓ ┏━━━━━━━━━━━━━━━━━━┓
            q1: ────────┨ Z ┠───────────────────────▓─┨ RY(-π*theta + π) ┠───
                        ┗━━━┛                         ┗━━━━━━━━━━━━━━━━━━┛
        """
        from mindquantum.core.gates import BarrierGate
        _check_input_type('string', str, string)
        terms = self._get_gate_terms(string)
        circ = self.circuit_class()
        barrier_qubits = []
        for term in terms:
            gate_name, qubits, params = self._get_gate_info(term)
            gate = self._to_gate(gate_name, qubits, params)
            if isinstance(gate, BarrierGate):
                barrier_qubits.extend(gate.obj_qubits)
            else:
                if barrier_qubits:
                    circ += BarrierGate().on(barrier_qubits)
                barrier_qubits = []
                circ += gate
        if barrier_qubits:
            circ += BarrierGate().on(barrier_qubits)
        return circ

    def _qubits_str(self, gate) -> str:
        """Convert the gate's qubits to a string"""
        return "".join(f"Q{qubit} " for qubit in gate.ctrl_qubits + gate.obj_qubits)

    def to_string(self, circuit, parametric: bool = True) -> str:
        """Convert the input circuit to qcis.

        Args:
            circuit (Circuit): The quantum circuit you want to translated to qcis.
            parametric (bool): Whether to keep the parameters in gates. If it is ``False``
                , we will discard all parameters and rotation gates with zero angles. The
                remaining angles will be restricted to the interval [-pi, pi].
                Default: ``True``.

        Returns:
            str, The qcis format of the input circuit.

        Examples:
            >>> circ = Circuit()
            >>> circ.x(0).z(1,0).rx({"a":-2*np.sqrt(2)}, 0).sx(0).barrier()
            >>> circ.ry(ParameterResolver(data={'theta':-np.pi}, const=np.pi), 1)
            >>> string= QCIS().to_string(circ)
            >>> print(string)
            X Q0
            CZ Q0 Q1
            RX Q0 -2√2*a
            X2P Q0
            B Q0
            B Q1
            RY Q1 -π*theta + π

        Raises:
            TypeError: if `circuit` is not a Circuit.
            NotImplementedError: if the input circuit containing gates which is not supported by qcis.
        """
        _check_input_type('circuit', self.circuit_class, circuit)
        if circuit.is_noise_circuit:
            raise ValueError("Cannot convert noise circuit to QCIS.")
        string = []
        for gate in circuit: # type: ignore
            if isinstance(gate, self.gate_class.BarrierGate):
                if not gate.obj_qubits:
                    for i in range(circuit.n_qubits):
                        string.append(f"B Q{i}")
                else:
                    string.append("B " + self._qubits_str(gate))
                continue
            if isinstance(gate, self.gate_class.Measure):
                string.append("M " + self._qubits_str(gate))
                continue
            if isinstance(gate, (self.gate_class.YGate, self.gate_class.HGate)):
                if gate.ctrl_qubits:
                    _failed_to_qcis(gate)
                string.append(gate.name + " " + self._qubits_str(gate))
                continue
            if isinstance(gate, self.gate_class.XGate):
                if not gate.ctrl_qubits:
                    string.append("X " +self. _qubits_str(gate))
                    continue
                elif len(gate.ctrl_qubits) == 1:
                    string.append("H " + f"Q{gate.obj_qubits[0]}")
                    string.append("CZ " + self._qubits_str(gate))
                    string.append("H " + f"Q{gate.obj_qubits[0]}")
                    continue
                else:
                    _failed_to_qcis(gate)
            if isinstance(gate, self.gate_class.ZGate):
                if not gate.ctrl_qubits:
                    string.append("Z " + self._qubits_str(gate))
                else:
                    if len(gate.ctrl_qubits) > 1:
                        _failed_to_qcis(gate)
                    string.append("CZ " + self._qubits_str(gate))
                continue
            if isinstance(gate, (self.gate_class.TGate, self.gate_class.SGate)):
                if gate.ctrl_qubits:
                    _failed_to_qcis(gate)
                if gate.hermitianed:
                    string.append(f"{gate.name}D " + self._qubits_str(gate))
                else:
                    string.append(f"{gate.name} " + self._qubits_str(gate))
                continue
            if isinstance(gate, self.gate_class.SXGate):
                if gate.ctrl_qubits:
                    _failed_to_qcis(gate)
                if gate.hermitianed:
                    string.append(f"{gate.name[1]}2M " + self._qubits_str(gate))
                else:
                    string.append(f"{gate.name[1]}2P " + self._qubits_str(gate))
                continue
            if isinstance(gate, (self.gate_class.RX, self.gate_class.RY, self.gate_class.RZ)):
                if gate.ctrl_qubits:
                    _failed_to_qcis(gate)
                if parametric:
                    param = str(gate.coeff)
                else:
                    if not gate.coeff.const:
                        continue
                    param = str(_valid_angle(gate.coeff.const))
                string.append(f"{gate.name} " + self._qubits_str(gate) + param)
                continue
            if isinstance(gate, self.gate_class.IGate):
                if gate.ctrl_qubits:
                    _failed_to_qcis(gate)
                string.append("I " + self._qubits_str(gate))
                continue
            _failed_to_qcis(gate)
        return "\n".join(string)

    def to_file(self, file_name: str, circuit, parametric: bool = True) -> None:
        """
        Convert a quantum circuit to qcis format and save in file.

        Args:
            file_name (str): The file name you want to save the qcis file.
            circuit (Circuit): The Circuit you want to convert.
            parametric (bool): Whether to keep the parameters in gates. If it is ``False``
                , we will discard all parameters and rotation gates with zero angles. The
                remaining angles will be restricted to the interval [-pi, pi]. Default: ``True``.

        Raises:
            TypeError: if `file_name` is not a str.
            TypeError: if `circuit` is not a Circuit.
            NotImplementedError: if the input circuit containing gates which is not supported by qcis.
        """
        if not isinstance(file_name, str):
            raise TypeError(f'file_name requires a str, but get {type(file_name)}')
        if not isinstance(circuit, self.circuit_class):
            raise TypeError(f"circuit requires a Circuit, but get {type(circuit)}")
        with fdopen(file_name, 'w') as fd:
            fd.writelines(self.to_string(circuit, parametric))
        print(f"write circuit to {file_name} finished!")
