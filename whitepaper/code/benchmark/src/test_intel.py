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
"""Benchmark Intel SDK."""
from functools import partial
import os
import re
import pytest
import json
import numpy as np
import intelqs_py as iqs

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, os.path.abspath("../data"))


def get_task_file(task: str):
    """Get all data file name."""
    all_path = []
    for file_name in os.listdir(data_path):
        if file_name.startswith(task):
            full_path = os.path.join(data_path, file_name)
            all_path.append(full_path)
    return all_path


class Simulator:
    def __init__(self, n_qubits: int, str_circ: str):
        self.qs = iqs.QubitRegister(n_qubits, "base", 0, 0)
        self.circ = []

        for str_g in str_circ:
            name = str_g["name"]
            obj = str_g["obj"]
            ctrl = str_g["ctrl"]
            val = str_g.get("val", 0)
            if name == "h":
                if len(ctrl) == 1:
                    self.circ.append(partial(self.qs.ApplyCHadamard, ctrl[0], obj[0]))
                    continue
                self.circ.append(partial(self.qs.ApplyHadamard, obj[0]))
                continue
            if name == "rzz":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyCPauliX, obj[1], obj[0]))
                    self.circ.append(partial(self.qs.ApplyRotationZ, obj[0], val))
                    self.circ.append(partial(self.qs.ApplyCPauliX, obj[1], obj[0]))
                    continue

            if name == "rxx":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyHadamard, obj[0]))
                    self.circ.append(partial(self.qs.ApplyHadamard, obj[1]))
                    self.circ.append(partial(self.qs.ApplyCPauliX, obj[1], obj[0]))
                    self.circ.append(partial(self.qs.ApplyRotationZ, obj[0], val))
                    self.circ.append(partial(self.qs.ApplyCPauliX, obj[1], obj[0]))
                    self.circ.append(partial(self.qs.ApplyHadamard, obj[0]))
                    self.circ.append(partial(self.qs.ApplyHadamard, obj[1]))
                    continue

            if name == "ryy":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyRotationX, obj[0], np.pi / 2))
                    self.circ.append(partial(self.qs.ApplyRotationX, obj[1], np.pi / 2))
                    self.circ.append(partial(self.qs.ApplyCPauliX, obj[1], obj[0]))
                    self.circ.append(partial(self.qs.ApplyRotationZ, obj[0], val))
                    self.circ.append(partial(self.qs.ApplyCPauliX, obj[1], obj[0]))
                    self.circ.append(
                        partial(self.qs.ApplyRotationX, obj[0], -np.pi / 2)
                    )
                    self.circ.append(
                        partial(self.qs.ApplyRotationX, obj[1], -np.pi / 2)
                    )
                    continue
            if name == "x":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyPauliX, obj[0]))
                    continue
                if len(ctrl) == 1:
                    self.circ.append(partial(self.qs.ApplyCPauliX, ctrl[0], obj[0]))
                    continue
            if name == "y":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyPauliY, obj[0]))
                    continue
                if len(ctrl) == 1:
                    self.circ.append(partial(self.qs.ApplyCPauliY, ctrl[0], obj[0]))
                    continue
            if name == "z":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyPauliZ, obj[0]))
                    continue
                if len(ctrl) == 1:
                    self.circ.append(partial(self.qs.ApplyCPauliZ, ctrl[0], obj[0]))
                    continue
            if name == "ps":
                if not ctrl:
                    m = np.array([[1, 0], [0, np.exp(1j * val)]], dtype=np.complex128)
                    self.circ.append(partial(self.qs.Apply1QubitGate, obj[0], m))
                    continue
                if len(ctrl) == 1:
                    m = np.array([[1, 0], [0, np.exp(1j * val)]], dtype=np.complex128)
                    self.circ.append(
                        partial(self.qs.ApplyControlled1QubitGate, ctrl[0], obj[0], m)
                    )
                    continue
            if name == "rx":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyRotationX, obj[0], val))
                    continue
                if len(ctrl) == 1:
                    a, b = np.cos(val / 2), -1j * np.sin(val / 2)
                    m = np.array([[a, b], [b, a]], dtype=np.complex128)
                    self.circ.append(
                        partial(self.qs.ApplyControlled1QubitGate, ctrl[0], obj[0], m)
                    )
                    continue
            if name == "rz":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyRotationZ, obj[0], val))
                    continue
                if len(ctrl) == 1:
                    a = np.exp(-1j * val / 2)
                    m = np.array([[a, 0], [0, np.conj(a)]], dtype=np.complex128)
                    self.circ.append(
                        partial(self.qs.ApplyControlled1QubitGate, ctrl[0], obj[0], m)
                    )
                    continue
            if name == "ry":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplyRotationY, obj[0], val))
                    continue
                if len(ctrl) == 1:
                    a, b = np.cos(val / 2), -np.sin(val / 2)
                    m = np.array([[a, b], [-b, a]], dtype=np.complex128)
                    self.circ.append(
                        partial(self.qs.ApplyControlled1QubitGate, ctrl[0], obj[0], m)
                    )
                    continue
            if name == "swap":
                if not ctrl:
                    self.circ.append(partial(self.qs.ApplySwap, obj[0], obj[1]))
                    continue
            if name in ["sdag", "s"]:
                if not ctrl:
                    m = np.array([[1, 0], [0, 1j if name == "s" else -1j]])
                    self.circ.append(partial(self.qs.Apply1QubitGate, obj[0], m))
                    continue
            if name in ["tdag", "t"]:
                if not ctrl:
                    m = np.array(
                        [[1, 0], [0, (1 + (1j if name == "t" else -1j)) / np.sqrt(2)]]
                    )
                    self.circ.append(partial(self.qs.Apply1QubitGate, obj[0], m))
                    continue
            self.gate_not_implement(name, obj, ctrl)

    def run(self):
        for i in self.circ:
            i()

    def gate_not_implement(self, gate, obj, ctrl):
        raise ValueError(f"gate not implement: {gate}({obj}, {ctrl})")


# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, intel

random_circuit_data_path = get_task_file("random_circuit")
random_circuit_data_path.sort()
random_circuit_data_path = random_circuit_data_path[:24]


@pytest.mark.random_circuit
@pytest.mark.intel
@pytest.mark.parametrize("file_name", random_circuit_data_path)
def test_intel_random_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    sim = Simulator(n_qubits, str_circ)
    benchmark(sim.run)


# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, intel

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:24]


@pytest.mark.simple_circuit
@pytest.mark.intel
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
def test_intel_simple_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    sim = Simulator(n_qubits, str_circ)
    benchmark(sim.run)
