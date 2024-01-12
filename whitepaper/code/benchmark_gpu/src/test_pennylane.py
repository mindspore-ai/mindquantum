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
"""Benchmark pennylane."""
from functools import partial
import os
import re
import pytest
import json
import pennylane as qml
from pennylane import numpy as np
import numpy
import networkx as nx
from scipy.optimize import minimize

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

def convert_back_to_qiskit_circ(str_circ, n_qubits):
    """Convert str gate back to Qiskit circuit."""
    from qiskit import QuantumCircuit
    import qiskit.circuit.library as G
    circ = QuantumCircuit(n_qubits)
    for str_g in str_circ:
        name = str_g["name"]
        obj = str_g["obj"]
        ctrl = str_g["ctrl"]
        val = str_g.get("val", 0)
        if name in ["y", "x", "z", "h"]:
            if not ctrl:
                getattr(circ, name)(obj[0])
                continue
            if len(ctrl) == 1:
                getattr(circ, f"c{name}")(ctrl[0], obj[0])
                continue
        if name == "ps":
            if not ctrl:
                circ.p(val, obj[0])
                continue
            if len(ctrl) == 1:
                circ.append(G.CPhaseGate(val), ctrl + obj)
                continue
        if name in ["rx", "ry", "rz"]:
            if not ctrl:
                getattr(circ, name)(val, obj[0])
                continue
            if len(ctrl) == 1:
                circ.append(getattr(G, f"C{name.upper()}Gate")(val), ctrl + obj)
                continue
        if name == "swap":
            if not ctrl:
                circ.swap(obj[0], obj[1])
                continue
        if name in ["rxx", "ryy", "rzz"]:
            if not ctrl:
                getattr(circ, name)(val, obj[0], obj[1])
                continue
        if name in ['s', 't']:
            if not ctrl:
                circ.append(getattr(G, f"{name.upper()}Gate")(), obj)
                continue
        if name in ['sdag', 'tdag']:
            if not ctrl:
                circ.append(getattr(G, f"{name[0].upper()}dgGate")(), obj)
                continue
        raise ValueError(f"gate not implement: {name}({obj}, {ctrl})")
    return circ


# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, pennylane

random_circuit_data_path = get_task_file("random_circuit")
random_circuit_data_path.sort()
random_circuit_data_path = random_circuit_data_path[:24]


@pytest.mark.random_circuit
@pytest.mark.pennylane
@pytest.mark.parametrize("file_name", random_circuit_data_path)
def test_pennylane_random_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)

    qiskit_qc = convert_back_to_qiskit_circ(str_circ, n_qubits)

    dev = qml.device('lightning.gpu', wires=n_qubits)
    @qml.qnode(dev)
    def quantum_circuit_with_loaded_subcircuit():
        qml.from_qiskit(qiskit_qc)()
        return qml.state()
    benchmark(quantum_circuit_with_loaded_subcircuit)

# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, pennylane

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:24]


@pytest.mark.simple_circuit
@pytest.mark.pennylane
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
def test_pennylane_simple_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    qiskit_qc = convert_back_to_qiskit_circ(str_circ, n_qubits)

    dev = qml.device('lightning.gpu', wires=n_qubits)
    @qml.qnode(dev)
    def quantum_circuit_with_loaded_subcircuit():
        qml.from_qiskit(qiskit_qc)()
        return qml.state()
    benchmark(quantum_circuit_with_loaded_subcircuit)

# Section Three
# Benchmark four regular qaoa
# Available pytest mark: regular_4, pennylane

regular_4_data_path = get_task_file("regular_4")
regular_4_data_path.sort()
regular_4_data_path = regular_4_data_path[:10]

def f_and_g(weights, circuit, vag):
    f, g = circuit(weights), vag(weights)
    return f, numpy.array(g)

def benchmark_regular_4(weights, circuit, vag):
    res = minimize(f_and_g, weights,args=(circuit, vag), method='bfgs',jac=True)
    return res

@pytest.mark.regular_4
@pytest.mark.pennylane
@pytest.mark.parametrize("file_name", regular_4_data_path)
def test_pennylane_regular_4(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        edges = [tuple(i) for i in json.load(f)]
    graph = nx.from_edgelist(edges)
    cost_h, mixer_h = qml.qaoa.maxcut(graph)
    dev = qml.device('lightning.gpu', wires=n_qubits)
    weights = numpy.random.uniform(size=len(edges)+n_qubits)

    @qml.qnode(dev, diff_method='adjoint')
    def circuit(params):
        for i in range(n_qubits):
            qml.Hadamard(wires=i)
        for idx, (i, j) in enumerate(edges):
            qml.IsingZZ(params[idx], wires=[i, j])
        for i in range(n_qubits):
            qml.RX(params[len(edges) + i], wires=i)
        return qml.expval(cost_h)
    vag = qml.grad(circuit, argnum=0)
    benchmark(benchmark_regular_4, weights, circuit, vag)
