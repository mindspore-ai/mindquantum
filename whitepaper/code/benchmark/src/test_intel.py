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


# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, intel


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
                    self.circ.append(partial(self.apply_ch, obj[0], ctrl[0]))
                    continue
                self.circ.append(partial(self.apply_h, obj[0]))
                continue
            if name == "rzz":
                if not ctrl:
                    self.circ.append(partial(self.apply_cnot, obj[0], obj[1]))
                    self.circ.append(partial(self.apply_rz, obj[0], val))
                    self.circ.append(partial(self.apply_cnot, obj[0], obj[1]))
                    continue
            if name == 'y':
                if not ctrl:
                    self.circ.append(partial(self.apply_y, obj[0]))
                    continue
            self.gate_not_implement(name, obj, ctrl)

    def run(self):
        for i in self.circ:
            i()

    def gate_not_implement(self, gate, obj, ctrl):
        raise ValueError(f"gate not implement: {gate}({obj}, {ctrl})")

    def apply_rz(self, q, v):
        self.qs.ApplyPauliZ(q, v)

    def apply_y(self, q):
        self.qs.ApplyPauliY(q)

    def apply_x(self, q):
        self.qs.ApplyPauliX(q)

    def apply_cnot(self, q, c):
        self.qs.ApplyCPauliX(c, q)

    def apply_h(self, q):
        self.qs.ApplyHadamard(q)

    def apply_ch(self, q, c):
        self.qs.ApplyCHadamard(c, q)


random_circuit_data_path = get_task_file("random_circuit")
random_circuit_data_path.sort()
random_circuit_data_path = random_circuit_data_path[:5]


# @pytest.mark.random_circuit
# @pytest.mark.intel
# @pytest.mark.parametrize("file_name", random_circuit_data_path)
# def test_qulacs_random_circuit(benchmark, file_name):
file_name = random_circuit_data_path[0]
n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
with open(file_name, "r", encoding="utf-8") as f:
    str_circ = json.load(f)
sim = Simulator(n_qubits, str_circ)
# benchmark(sim.run)
