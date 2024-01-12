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
"""Benchmark MindQuantum."""

import os
import re
import pytest
import json
import numpy as np
from scipy.optimize import minimize

from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.simulator import Simulator
import mindquantum as mq
from mindquantum.algorithm.nisq.qaoa import MaxCutAnsatz

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, os.path.abspath("../data"))

# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, mindquantum

STR_GATE_MAP = {
    "x": G.XGate,
    "y": G.YGate,
    "z": G.ZGate,
    "h": G.HGate,
    "swap": G.SWAPGate,
    "rx": G.RX,
    "ry": G.RY,
    "rz": G.RZ,
    "rxx": G.Rxx,
    "ryy": G.Ryy,
    "rzz": G.Rzz,
    "ps": G.PhaseShift,
    "s": G.SGate,
    "sdag": G.SGate,
    "t": G.TGate,
    "tdag": G.TGate,
}


def convert_back_to_mq_circ(str_circ):
    """Convert str gate back to MindQuantum circuit."""
    circ = Circuit()
    for str_g in str_circ:
        name = str_g["name"]
        obj = str_g["obj"]
        ctrl = str_g["ctrl"]
        if "val" in str_g:
            gate = STR_GATE_MAP[name](str_g["val"]).on(obj, ctrl)
        else:
            gate = STR_GATE_MAP[name]().on(obj, ctrl)
        circ += gate.hermitian() if name.endswith("dag") else gate
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


def benchmark_random_circuit(sim, circ):
    sim.apply_circuit(circ)


@pytest.mark.random_circuit
@pytest.mark.mindquantum
@pytest.mark.parametrize("file_name", random_circuit_data_path)
@pytest.mark.parametrize("dtype", [mq.complex128, mq.complex64])
def test_mindquantum_random_circuit(benchmark, file_name, dtype):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    sim = Simulator("mqvector", n_qubits, dtype=dtype)
    circ = convert_back_to_mq_circ(str_circ)
    benchmark(benchmark_random_circuit, sim, circ)


# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, mindquantum

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:24]


@pytest.mark.simple_circuit
@pytest.mark.mindquantum
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
@pytest.mark.parametrize("dtype", [mq.complex128, mq.complex64])
def test_mindquantum_simple_circuit(benchmark, file_name, dtype):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    sim = Simulator("mqvector", n_qubits, dtype=dtype)
    circ = convert_back_to_mq_circ(str_circ)
    benchmark(benchmark_random_circuit, sim, circ)


# Section Three
# Benchmark four regular qaoa
# Available pytest mark: regular_4, mindquantum

regular_4_data_path = get_task_file("regular_4")
regular_4_data_path.sort()
regular_4_data_path = regular_4_data_path[:19]


def regular_4_fun(p, grad_ops):
    f, g = grad_ops(p)
    f = np.real(f)[0][0]
    g = np.real(g)[0][0]
    return f, g


def benchmark_regular_4(grad_ops, p0):
    res = minimize(regular_4_fun, p0, args=(grad_ops,), method="bfgs", jac=True)


@pytest.mark.regular_4
@pytest.mark.mindquantum
@pytest.mark.parametrize("file_name", regular_4_data_path)
@pytest.mark.parametrize("dtype", [mq.complex128, mq.complex64])
def test_mindquantum_regular_4(benchmark, file_name, dtype):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        edges = [tuple(i) for i in json.load(f)]
    ansatz = MaxCutAnsatz(edges, 1)
    circ = Circuit()
    for i in range(n_qubits):
        circ.h(i)
    for idx, (i, j) in enumerate(edges):
        circ.rzz(f"p_{idx}", [i, j])
    for i in range(n_qubits):
        circ.rx(f"p_{len(edges)+i}", i)
    ham = mq.Hamiltonian(-ansatz.hamiltonian).astype(dtype)
    sim = Simulator("mqvector", n_qubits, dtype=dtype)
    grad_ops = sim.get_expectation_with_grad(ham, circ)
    rng = np.random.default_rng(42)
    p0 = rng.uniform(-3, 3, len(circ.params_name))
    benchmark(benchmark_regular_4, grad_ops, p0)
