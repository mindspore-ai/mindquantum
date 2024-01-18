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
"""Benchmark projectq."""
from functools import partial
import os
import re
import pytest
import json
import numpy as np

from projectq import MainEngine
from projectq import ops

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


def get_qreg(obj, qregs):
    if isinstance(obj, int):
        return qregs[obj]
    return tuple([qregs[i] for i in obj])


def convert_back_to_projectq_circ(str_circ, qregs):
    """Convert str gate back to projectq circuit."""
    out = []
    for str_g in str_circ:
        name = str_g["name"]
        obj = str_g["obj"]
        ctrl = str_g["ctrl"]
        val = str_g.get("val", 0)
        if name == "rzz":
            if not ctrl:
                out.append([ops.Rzz(val), get_qreg(obj, qregs)])
                continue
        if name == "y":
            if not ctrl:
                out.append([ops.Y, get_qreg(obj, qregs)])
                continue
        if name == "ps":
            if not ctrl:
                out.append([ops.Ph(val), get_qreg(obj, qregs)])
                continue
            if len(ctrl) == 1:
                out.append([ops.C(ops.Ph(val)), get_qreg(ctrl + obj, qregs)])
                continue
        if name == "ryy":
            if not ctrl:
                out.append([ops.Ryy(val), get_qreg(obj, qregs)])
                continue
        if name == "rxx":
            if not ctrl:
                out.append([ops.Rxx(val), get_qreg(obj, qregs)])
                continue
        if name == "rx":
            if not ctrl:
                out.append([ops.Rx(val), get_qreg(obj, qregs)])
                continue
            if len(ctrl) == 1:
                out.append([ops.C(ops.Rx(val)), get_qreg(ctrl + obj, qregs)])
                continue
        if name == "rz":
            if not ctrl:
                out.append([ops.Rz(val), get_qreg(obj, qregs)])
                continue
            if len(ctrl) == 1:
                out.append([ops.C(ops.Rz(val)), get_qreg(ctrl + obj, qregs)])
                continue
        if name == "ry":
            if not ctrl:
                out.append([ops.Ry(val), get_qreg(obj, qregs)])
                continue
            if len(ctrl) == 1:
                out.append([ops.C(ops.Ry(val)), get_qreg(ctrl + obj, qregs)])
                continue
        if name == "z":
            if not ctrl:
                out.append([ops.Z, get_qreg(obj, qregs)])
                continue
            if len(ctrl) == 1:
                out.append([ops.C(ops.Z), get_qreg(ctrl + obj, qregs)])
                continue
        if name == "h":
            if not ctrl:
                out.append([ops.H, get_qreg(obj, qregs)])
                continue
            if len(ctrl) == 1:
                out.append([ops.C(ops.H), get_qreg(ctrl + obj, qregs)])
                continue
        if name == "x":
            if not ctrl:
                out.append([ops.X, get_qreg(obj, qregs)])
                continue
            if len(ctrl) == 1:
                out.append([ops.CNOT, get_qreg(ctrl + obj, qregs)])
                continue
        if name == "y":
            if len(ctrl) == 1:
                out.append([ops.C(ops.Y), get_qreg(ctrl + obj, qregs)])
                continue
        if name == "swap":
            if not ctrl:
                out.append([ops.Swap, get_qreg(obj, qregs)])
                continue
        if name in ["s", "sdag"]:
            if not ctrl:
                out.append([ops.S if name == "s" else ops.Sdag, get_qreg(obj, qregs)])
                continue
        if name in ["t", "tdag"]:
            if not ctrl:
                out.append([ops.T if name == "t" else ops.Tdag, get_qreg(obj, qregs)])
                continue
        raise ValueError(f"gate not implement: {name}({obj}, {ctrl})")
    return out


def run_circ(circs, eng, qregs):
    for i, j in circs:
        i | j
    ops.All(ops.Measure) | qregs
    eng.flush()


# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, projectq

random_circuit_data_path = get_task_file("random_circuit")
random_circuit_data_path.sort()
random_circuit_data_path = random_circuit_data_path[:24]


@pytest.mark.random_circuit
@pytest.mark.projectq
@pytest.mark.parametrize("file_name", random_circuit_data_path)
def test_projectq_random_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    eng = MainEngine()
    qregs = eng.allocate_qureg(n_qubits)
    circs = convert_back_to_projectq_circ(str_circ, qregs)
    benchmark(run_circ, circs, eng, qregs)


# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, projectq

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:24]


@pytest.mark.simple_circuit
@pytest.mark.projectq
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
def test_projectq_simple_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    eng = MainEngine()
    qregs = eng.allocate_qureg(n_qubits)
    circs = convert_back_to_projectq_circ(str_circ, qregs)
    benchmark(run_circ, circs, eng, qregs)
