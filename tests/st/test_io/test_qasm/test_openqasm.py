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
# wITHOUT wARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Test openqasm."""

import numpy as np

from mindquantum.core import U3, UN, Circuit, H, X
from mindquantum.io import OpenQASM
from mindquantum.simulator import Simulator
from mindquantum.utils import random_circuit


def test_openqasm():
    """
    test openqasm
    Description: test openqasm api
    Expectation: success
    """
    cir = Circuit().h(0).x(1).rz(0.1, 0, 1)
    test_openqasms = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\nh q[0];\nx q[1];\ncrz(0.1) q[1],q[0];'
    openqasm = OpenQASM().to_string(cir)
    assert len(openqasm) == 82
    assert openqasm[63:] == 'crz(0.1) q[1],q[0];'
    assert openqasm == test_openqasms
    cir = Circuit().rz(0.1, 0, 1)
    test_openqasms = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncrz(0.1) q[1],q[0];'
    test_cir = OpenQASM().from_string(test_openqasms)
    assert np.allclose(test_cir.matrix(), cir.matrix())


def test_openqasm_custom_gate1():
    """
    test custom gate in openqasm
    Description: test openqasm custom gate
    Expectation: success
    """
    qasm = """
// quantum ripple-carry adder from Cuccaro et al, quant-ph/0410184
OPENQASM 2.0;
include "qelib1.inc";
gate majority a,b,c
{
  cx c,b;
  cx c,a;
  ccx a,b,c;
}
gate unmaj a,b,c
{
  ccx a,b,c;
  cx c,a;
  cx a,b;
}
qreg cin[1];
qreg a[4];
qreg b[4];
qreg cout[1];
creg ans[5];
// set input states
x a[0]; // a = 0001
x b;    // b = 1111
// add a to b, storing result in b
majority cin[0],b[0],a[0];
majority a[0],b[1],a[1];
majority a[1],b[2],a[2];
majority a[2],b[3],a[3];
cx a[3],cout[0];
unmaj a[2],b[3],a[3];
unmaj a[1],b[2],a[2];
unmaj a[0],b[1],a[1];
unmaj cin[0],b[0],a[0];
measure b[0] -> ans[0];
measure b[1] -> ans[1];
measure b[2] -> ans[2];
measure b[3] -> ans[3];
measure cout[0] -> ans[4];
    """
    circ = OpenQASM().from_string(qasm)

    def majority(a, b, c):
        return Circuit().x(b, c).x(a, c).x(c, [a, b])

    def unmaj(a, b, c):
        return Circuit().x(c, [a, b]).x(a, c).x(b, a)

    cin = [0]
    a = [1, 2, 3, 4]
    b = [5, 6, 7, 8]
    cout = [9]
    exp_circ = sum(
        [
            Circuit().x(a[0]),
            UN(X, b),
            majority(cin[0], b[0], a[0]),
            majority(a[0], b[1], a[1]),
            majority(a[1], b[2], a[2]),
            majority(a[2], b[3], a[3]),
            Circuit().x(
                cout[0],
                a[3],
            ),
            unmaj(a[2], b[3], a[3]),
            unmaj(a[1], b[2], a[2]),
            unmaj(a[0], b[1], a[1]),
            unmaj(cin[0], b[0], a[0]),
        ]
    )
    exp_circ.measure(b[0])
    exp_circ.measure(b[1])
    exp_circ.measure(b[2])
    exp_circ.measure(b[3])
    exp_circ.measure(cout[0])
    init = random_circuit(circ.n_qubits, 30, seed=42)
    sim = Simulator('mqvector', circ.n_qubits)
    s_1 = sim.sampling(init + circ, shots=50, seed=42)
    s_2 = sim.sampling(init + exp_circ, shots=50, seed=42)
    assert np.all(s_1.samples == s_2.samples)


# pylint: disable=invalid-name
def test_openqasm_custom_gate2():
    """
    test custom gate in openqasm
    Description: test openqasm custom gate
    Expectation: success
    """
    qasm = """
// quantum Fourier transform
OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];
creg c[4];
x q[0];
x q[2];
barrier q;
h q[0];
cu1(pi/2) q[1],q[0];
h q[1];
cu1(pi/4) q[2],q[0];
cu1(pi/2) q[2],q[1];
h q[2];
cu1(pi/8) q[3],q[0];
cu1(pi/4) q[3],q[1];
cu1(pi/2) q[3],q[2];
h q[3];
measure q -> c;
    """
    circ = OpenQASM().from_string(qasm)

    q = [0, 1, 2, 3]
    exp_circ = sum(
        [
            Circuit().x(q[0]).x(q[2]).barrier().h(q[0]),
            U3(0, 0, np.pi / 2).on(q[0], q[1]),
            H.on(q[1]),
            U3(0, 0, np.pi / 4).on(q[0], q[2]),
            U3(0, 0, np.pi / 2).on(q[1], q[2]),
            H.on(q[2]),
            U3(0, 0, np.pi / 8).on(q[0], q[3]),
            U3(0, 0, np.pi / 4).on(q[1], q[3]),
            U3(0, 0, np.pi / 2).on(q[2], q[3]),
            H.on(q[3]),
        ]
    )
    exp_circ.measure_all()
    init = random_circuit(circ.n_qubits, 30, seed=42)
    sim = Simulator('mqvector', circ.n_qubits)
    s_1 = sim.sampling(init + circ, shots=50, seed=42)
    s_2 = sim.sampling(init + exp_circ, shots=50, seed=42)
    assert np.all(s_1.samples == s_2.samples)
