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
"""Benchmark pyqpanda."""
from functools import partial
import os
import re
import pytest
import json
import numpy as np
import pyqpanda as pq


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

def get_qubits(qubits, idxs):
    if isinstance(idxs, int):
        return qubits[idxs]
    return [qubits[i] for i in idxs]

def convert_back_to_pyqpanda_circ(str_circ, qubits):
    """Convert str gate back to pyqpanda circuit."""
    prog = pq.QProg()
    for str_g in str_circ:
        name = str_g["name"]
        obj = str_g["obj"]
        ctrl = str_g["ctrl"]
        val = str_g.get("val", 0)
        if name in ['rxx', 'ryy', 'rzz']:
            if not ctrl:
                prog.insert(getattr(pq, name.upper())(*get_qubits(qubits, obj), val))
                continue
        if name in ['x', 'y', 'z', 'h']:
            if not ctrl:
                prog.insert(getattr(pq, name.upper())(get_qubits(qubits, obj[0])))
                continue
            if len(ctrl) == 1:
                prog.insert(getattr(pq, name.upper())(get_qubits(qubits, obj[0])).control(qubits[ctrl[0]]))
                continue
        if name == 'ps':
            if not ctrl:
                prog.insert(pq.P(get_qubits(qubits, obj[0]), val))
                continue
            if len(ctrl) == 1:
                prog.insert(pq.CP(get_qubits(qubits, ctrl[0]), get_qubits(qubits, obj[0]), val))
                continue
        if name in ['rx', 'ry', 'rz']:
            if not ctrl:
                prog.insert(getattr(pq, name.upper())(qubits[obj[0]], val))
                continue
            if len(ctrl) == 1:
                prog.insert(getattr(pq, name.upper())(qubits[obj[0]], val).control(qubits[ctrl[0]]))
                continue
        if name == 'swap':
            if not ctrl:
                prog.insert(pq.SWAP(*get_qubits(qubits, obj)))
                continue
        if name in ['s', 't']:
            if not ctrl:
                prog.insert(getattr(pq, name.upper())(qubits[obj[0]]))
                continue
        if name in ['sdag', 'tdag']:
            if not ctrl:
                prog.insert(getattr(pq, name.upper()[0])(qubits[obj[0]]).dagger())
                continue
        raise ValueError(f"gate not implement: {name}({obj}, {ctrl})")
    return prog

# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, pyqpanda

random_circuit_data_path = get_task_file("random_circuit")
random_circuit_data_path.sort()
random_circuit_data_path = random_circuit_data_path[:24]


@pytest.mark.random_circuit
@pytest.mark.pyqpanda
@pytest.mark.parametrize("file_name", random_circuit_data_path)
def test_pyqpanda_random_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    qvm = pq.GPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(n_qubits)
    prog = convert_back_to_pyqpanda_circ(str_circ, qubits)
    benchmark(qvm.directly_run, prog)

# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, pyqpanda

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:24]


@pytest.mark.simple_circuit
@pytest.mark.pyqpanda
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
def test_pyqpanda_simple_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    qvm = pq.GPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(n_qubits)
    prog = convert_back_to_pyqpanda_circ(str_circ, qubits)
    benchmark(qvm.directly_run, prog)


# Section Three
# Benchmark four regular qaoa
# Available pytest mark: regular_4, pyqpanda

regular_4_data_path = get_task_file("regular_4")
regular_4_data_path.sort()
regular_4_data_path = regular_4_data_path[:9]


def generate_qaoa_ansatz(edges, qubits):
    p0 = pq.var(
        np.random.uniform(-1, 1, len(edges) + len(qubits)).astype(np.float64)[:, None], True
    )
    circ = pq.VariationalQuantumCircuit()
    for i in qubits:
        circ.insert(pq.VariationalQuantumGate_H(i))
    for idx, (i, j) in enumerate(edges):
        circ.insert(pq.VariationalQuantumGate_CNOT(qubits[i], qubits[j]))
        circ.insert(pq.VariationalQuantumGate_RZ(qubits[j], p0[idx]))
        circ.insert(pq.VariationalQuantumGate_CNOT(qubits[i], qubits[j]))
    for i in range(len(qubits)):
        circ.insert(pq.VariationalQuantumGate_RX(qubits[i], p0[idx + 1 + i]))
    op = {}
    for i, j in edges:
        op[f"Z{i} Z{j}"] = 1
    hp = pq.PauliOperator(op)
    return p0, circ, hp

@pytest.mark.regular_4
@pytest.mark.pyqpanda
@pytest.mark.parametrize("file_name", regular_4_data_path)
def test_pyqpanda_regular_4(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        edges = [tuple(i) for i in json.load(f)]
    qvm = pq.GPUQVM()
    qvm.init_qvm()
    qubits = qvm.qAlloc_many(n_qubits)
    p0, circ, hp = generate_qaoa_ansatz(edges, qubits)
    loss = pq.qop(circ, hp, qvm, qubits)
    optimizer = pq.MomentumOptimizer.minimize(loss, 0.01, 0.8)
    leaves = optimizer.get_variables()
    N_ITER = 10
    def run():
        for i in range(N_ITER):
            optimizer.run(leaves, 0)
            loss_value = optimizer.get_loss()
        return loss_value
    benchmark(run)
