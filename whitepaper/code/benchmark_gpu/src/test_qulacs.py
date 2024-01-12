# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Benchmark Qulacs."""

import os
import re
import pytest
import json
import numpy as np
from scipy.optimize import minimize

from qulacs import (
    Observable,
    ParametricQuantumCircuit,
    QuantumCircuit,
    gate,
)
from qulacs import QuantumStateGpu as QuantumState
current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, os.path.abspath("../data"))

# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, qulacs


def raise_error(name, obj, ctrl):
    raise ValueError(f"Can not convert {name}({obj}, {ctrl})")


def convert_back_to_qulacs_circ(str_circ, n_qubits, circ):
    """Convert str gate back to Qulacs circuit."""
    for str_g in str_circ:
        name = str_g["name"]
        obj = str_g["obj"]
        ctrl = str_g["ctrl"]
        val = str_g.get("val", 0)
        if name in ["x", "z"]:
            if ctrl:
                try:
                    if name == "x":
                        getattr(circ, f"add_CNOT_gate")(*ctrl, *obj)
                    else:
                        getattr(circ, f"add_CZ_gate")(*ctrl, *obj)
                except:
                    raise_error(name, obj, ctrl)
            else:
                try:
                    circ.add_gate(getattr(gate, name.upper())(*obj))
                except:
                    raise_error(name, obj, ctrl)
        elif name in ["h", "y"]:
            if len(ctrl) > 2:
                raise_error(name, obj, ctrl)
            elif ctrl:
                g = gate.to_matrix_gate(getattr(gate, name.upper())(obj[0]))
                g.add_control_qubit(ctrl[0], 1)
                circ.add_gate(g)
            else:
                circ.add_gate(getattr(gate, name.upper())(obj[0]))
        elif name in ["rzz", "ryy", "rxx"]:
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                circ.add_multi_Pauli_rotation_gate(
                    obj, [ord(i) - 119 for i in name[1:]], val
                )
        elif name == "rx":
            if len(ctrl) > 1:
                raise_error(name, obj, ctrl)
            elif len(ctrl):
                circ.add_gate(gate.S(obj[0]))
                circ.add_gate(gate.CNOT(ctrl[0], obj[0]))
                circ.add_gate(gate.ParametricRY(obj[0], -val / 2))
                circ.add_gate(gate.CNOT(ctrl[0], obj[0]))
                circ.add_gate(gate.ParametricRY(obj[0], val / 2))
                circ.add_gate(gate.Sdag(obj[0]))
            else:
                circ.add_gate(gate.ParametricRX(obj[0], val))
        elif name in ["ps", "rz", "ry"]:
            if name == "ry":
                prg = gate.ParametricRY
            else:
                prg = gate.ParametricRZ
            if len(ctrl) > 1:
                raise_error(name, obj, ctrl)
            elif len(ctrl):
                circ.add_gate(prg(obj[0], val / 2))
                circ.add_gate(gate.CNOT(ctrl[0], obj[0]))
                circ.add_gate(prg(obj[0], -val / 2))
                circ.add_gate(gate.CNOT(ctrl[0], obj[0]))
            else:
                circ.add_gate(prg(*(obj + ctrl), val))
        elif name == "swap":
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                circ.add_gate(gate.SWAP(obj[0], obj[1]))
        elif name == "s":
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                circ.add_gate(gate.S(obj[0]))
        elif name == "sdag":
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                circ.add_gate(gate.Sdag(obj[0]))
        elif name == "t":
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                circ.add_gate(gate.T(obj[0]))
        elif name == "tdag":
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                circ.add_gate(gate.Tdag(obj[0]))
        else:
            raise ValueError(f"Can not convert {name}({obj}, {ctrl}) to qulacs")
    return circ


def get_task_file(task: str):
    """Get all data file name."""
    all_path = []
    for file_name in os.listdir(data_path):
        if file_name.startswith(task):
            full_path = os.path.join(data_path, file_name)
            all_path.append(full_path)
    return all_path


random_circuit_data_path = get_task_file("random_circuit")
random_circuit_data_path.sort()
random_circuit_data_path = random_circuit_data_path[:24]


def benchmark_random_circuit(state, circ):
    circ.update_quantum_state(state)


@pytest.mark.random_circuit
@pytest.mark.qulacs
@pytest.mark.parametrize("file_name", random_circuit_data_path)
def test_qulacs_random_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    state = QuantumState(n_qubits)
    circ = QuantumCircuit(n_qubits)
    convert_back_to_qulacs_circ(str_circ, n_qubits, circ)
    benchmark(benchmark_random_circuit, state, circ)


# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, qulacs

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:24]


@pytest.mark.simple_circuit
@pytest.mark.qulacs
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
def test_qulacs_simple_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    state = QuantumState(n_qubits)
    circ = QuantumCircuit(n_qubits)
    convert_back_to_qulacs_circ(str_circ, n_qubits, circ)
    benchmark(benchmark_random_circuit, state, circ)
