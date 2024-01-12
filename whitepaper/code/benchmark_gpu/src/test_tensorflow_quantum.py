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
"""Benchmark tensorflow quantum."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import re
import pytest
import json
import numpy as np
import sympy
from scipy.optimize import minimize

import tensorflow as tf
import tensorflow_quantum as tfq
import cirq

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, os.path.abspath("../data"))

# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, tensorflow_quantum

STR_GATE_MAP = {
    "x": cirq.X,
    "y": cirq.Y,
    "z": cirq.Z,
    "h": cirq.H,
    "swap": cirq.SWAP,
    "rx": cirq.rx,
    "ry": cirq.ry,
    "rz": cirq.rz,
    "rxx": cirq.XXPowGate,
    "ryy": cirq.YYPowGate,
    "rzz": cirq.ZZPowGate,
    "ps": cirq.rz,
    "s": cirq.S,
    "sdag": cirq.ZPowGate(exponent=-0.5),
    "t": cirq.T,
    "tdag": cirq.ZPowGate(exponent=-0.25),
}


def convert_back_to_tfq_circ(str_circ, n_qubits):
    """Convert str gate back to tensorflow quantum circuit."""
    circ = cirq.Circuit()
    qubits = cirq.LineQubit.range(n_qubits)
    for str_g in str_circ:
        name = str_g["name"]
        obj = str_g["obj"]
        ctrl = str_g["ctrl"]
        if "val" in str_g:
            if name in ["rxx", "ryy", "rzz"]:
                circ.append(
                    [
                        STR_GATE_MAP[name](exponent=str_g["val"])
                        .on(*[qubits[i] for i in obj])
                        .controlled_by(*[qubits[i] for i in ctrl])
                    ]
                )
            else:
                circ.append(
                    [
                        STR_GATE_MAP[name](str_g["val"])
                        .on(*[qubits[i] for i in obj])
                        .controlled_by(*[qubits[i] for i in ctrl])
                    ]
                )
        else:
            circ.append(
                [
                    STR_GATE_MAP[name](*[qubits[i] for i in obj]).controlled_by(
                        *[qubits[i] for i in ctrl]
                    )
                ]
            )
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


def benchmark_random_circuit(circ, op):
    state = op(circ, [], [[]])


@pytest.mark.random_circuit
@pytest.mark.tensorflow_quantum
@pytest.mark.parametrize("file_name", random_circuit_data_path)
def test_tensorflow_quantum_random_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    circ = convert_back_to_tfq_circ(str_circ, n_qubits)
    my_circuit_tensor = tfq.convert_to_tensor([circ])
    op = tfq.get_state_op()
    benchmark(benchmark_random_circuit, my_circuit_tensor, op)


# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, tensorflow_quantum

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:24]


@pytest.mark.simple_circuit
@pytest.mark.tensorflow_quantum
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
def test_tensorflow_quantum_simple_circuit(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    circ = convert_back_to_tfq_circ(str_circ, n_qubits)
    my_circuit_tensor = tfq.convert_to_tensor([circ])
    op = tfq.get_state_op()
    benchmark(benchmark_random_circuit, my_circuit_tensor, op)


# Section Three
# Benchmark four regular qaoa
# Available pytest mark: regular_4, tensorflow_quantum

regular_4_data_path = get_task_file("regular_4")
regular_4_data_path.sort()
regular_4_data_path = regular_4_data_path[:15]


def benchmark_regular_4(energy_function, p0):
    res = minimize(energy_function, p0, method="bfgs", jac=True)


@pytest.mark.regular_4
@pytest.mark.tensorflow_quantum
@pytest.mark.parametrize("file_name", regular_4_data_path)
def test_tensorflow_quantum_regular_4(benchmark, file_name):
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        edges = [tuple(i) for i in json.load(f)]
    n_p = len(edges) + n_qubits
    symbols = []
    for i in range(n_p):
        symbols.append(sympy.Symbol(f"p_{i}"))
    rng = np.random.default_rng(42)
    p0 = rng.uniform(-3, 3, n_p)

    qubits = cirq.LineQubit.range(n_qubits)
    circ = cirq.Circuit()
    for i in range(n_qubits):
        circ.append(cirq.H(qubits[i]))
    for idx, (i, j) in enumerate(edges):
        circ.append(cirq.ZZPowGate(exponent=symbols[idx]).on(qubits[i], qubits[j]))
    for i in range(n_qubits):
        circ.append(cirq.rx(symbols[len(edges) + i]).on(qubits[i]))
    cost_ham = 0.0
    for i, j in edges:
        cost_ham += cirq.PauliString(cirq.Z(qubits[i]) * cirq.Z(qubits[j]))
    my_op = tfq.get_expectation_op()
    adjoint_differentiator = tfq.differentiators.Adjoint()
    op = adjoint_differentiator.generate_differentiable_op(analytic_op=my_op)
    circ = tfq.convert_to_tensor([circ])
    cost_ham = tfq.convert_to_tensor([[cost_ham]])
    symbol_names = tf.convert_to_tensor([str(i) for i in symbols])

    def energy_function(x):
        values_tensor = tf.convert_to_tensor([x])
        with tf.GradientTape() as g:
            g.watch(values_tensor)
            expectations = op(circ, symbol_names, values_tensor, cost_ham)
            grads = g.gradient(expectations, values_tensor)
        return expectations[0, 0], grads[0]

    benchmark(benchmark_regular_4, energy_function, p0)
