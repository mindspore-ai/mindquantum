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
"""Benchmark Tensorcircuit."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import re
import pytest
import json
import numpy as np
from scipy.optimize import minimize

import tensorcircuit as tc

current_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_path, os.path.abspath("../data"))

# Section One
# Benchmark random circuit
# Available pytest mark: random_circuit, tensorcircuit


def raise_error(name, obj, ctrl):
    raise ValueError(f"Can not convert {name}({obj}, {ctrl})")


def convert_back_to_tc_circ(str_circ, circ):
    """Convert str gate back to tensorcircuit circuit."""
    for str_g in str_circ:
        name = str_g["name"]
        obj = str_g["obj"]
        ctrl = str_g["ctrl"]
        val = str_g.get("val", 0)
        if name in ["x", "y", "z", "swap"]:
            if len(ctrl) > 1:
                raise_error(name, obj, ctrl)
            elif ctrl:
                getattr(circ, "c" + name)(*ctrl, *obj)
            else:
                getattr(circ, name)(*obj)
        elif name in ["h"]:
            if len(ctrl) > 1:
                raise_error(name, obj, ctrl)
            elif ctrl:
                circ.multicontrol(ctrl[0], obj[0], unitary=tc.gates._h_matrix)
            else:
                getattr(circ, name)(*obj)
        elif name in ["rzz", "ryy", "rxx"]:
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                getattr(circ, name)(*obj, theta=val)
        elif name in ["rx", "ry", "rz"]:
            if len(ctrl) > 1:
                raise_error(name, obj, ctrl)
            elif ctrl:
                getattr(circ, "c" + name)(ctrl[0], obj[0], theta=val)
            else:
                getattr(circ, name)(*obj, theta=val)
        elif name == "ps":
            if len(ctrl) > 1:
                raise_error(name, obj, ctrl)
            elif len(ctrl):
                getattr(circ, "cphase")(ctrl[0], obj[0], theta=val)
            else:
                getattr(circ, "phase")(*obj, theta=val)
        elif name in ["s", "t"]:
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                getattr(circ, name)(*obj)
        elif name in ["sdag", "tdag"]:
            if ctrl:
                raise_error(name, obj, ctrl)
            else:
                getattr(circ, name[:2])(*obj)
        else:
            raise ValueError(f"Can not convert {name}({obj}, {ctrl}) to tensorcircuit")
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
random_circuit_data_path = random_circuit_data_path[:16]


def benchmark_random_circuit(qir):
    circ = tc.Circuit.from_qir(qir)
    circ.state()


@pytest.mark.random_circuit
@pytest.mark.tensorcircuit
@pytest.mark.parametrize("file_name", random_circuit_data_path)
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_tensorcircuit_random_circuit(benchmark, file_name, dtype):
    tc.set_dtype(dtype)
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    circ = tc.Circuit(n_qubits)
    circ = convert_back_to_tc_circ(str_circ, circ)
    qir = circ.to_qir()
    benchmark(benchmark_random_circuit, qir)


# Section Two
# Benchmark simple gate set circuit
# Available pytest mark: simple_circuit, tensorcircuit

simple_circuit_data_path = get_task_file("simple_circuit")
simple_circuit_data_path.sort()
simple_circuit_data_path = simple_circuit_data_path[:16]


@pytest.mark.simple_circuit
@pytest.mark.tensorcircuit
@pytest.mark.parametrize("file_name", simple_circuit_data_path)
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_tensorcircuit_simple_circuit(benchmark, file_name, dtype):
    tc.set_dtype(dtype)
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        str_circ = json.load(f)
    circ = tc.Circuit(n_qubits)
    circ = convert_back_to_tc_circ(str_circ, circ)
    qir = circ.to_qir()
    benchmark(benchmark_random_circuit, qir)


# Section Three
# Benchmark four regular qaoa
# Available pytest mark: regular_4, tensorcircuit

regular_4_data_path = get_task_file("regular_4")
regular_4_data_path.sort()
regular_4_data_path = regular_4_data_path[:15]

K = tc.set_backend("tensorflow")


def benchmark_regular_4(f_scipy, p0):
    res = minimize(f_scipy, p0, method="bfgs", jac=True)


@pytest.mark.regular_4
@pytest.mark.tensorcircuit
@pytest.mark.parametrize("file_name", regular_4_data_path)
@pytest.mark.parametrize("dtype", ["complex128", "complex64"])
def test_tensorcircuit_regular_4(benchmark, file_name, dtype):
    tc.set_dtype(dtype)
    n_qubits = int(re.search(r"qubit_\d+", file_name).group().split("_")[-1])
    with open(file_name, "r", encoding="utf-8") as f:
        edges = [tuple(i) for i in json.load(f)]
    n_p = len(edges) + n_qubits
    rng = np.random.default_rng(42)
    p0 = rng.uniform(-3, 3, n_p)

    def energy(params):
        c = tc.Circuit(n_qubits)
        for i in range(n_qubits):
            c.h(i)
        for idx, (i, j) in enumerate(edges):
            c.rzz(i, j, theta=params[idx])
        for i in range(n_qubits):
            c.rx(i, theta=params[len(edges) + i])
        e = 0.0
        for i, j in edges:
            e += c.expectation_ps(z=[i, j])
        return K.real(e)

    f_scipy = tc.interfaces.scipy_interface(energy, shape=[n_p], jit=True)
    benchmark(benchmark_regular_4, f_scipy, p0)
