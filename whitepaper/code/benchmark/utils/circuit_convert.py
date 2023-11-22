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
"""Convert MindQuantum circuit to string format."""

from mindquantum.core.circuit import Circuit
from mindquantum.core import gates as G
from mindquantum.core.parameterresolver import ParameterResolver

FIX_GATE_MAP = {
    G.XGate: "x",
    G.YGate: "y",
    G.ZGate: "z",
    G.HGate: "h",
    G.SWAPGate: "swap",
    G.SGate: "s",
    G.TGate: "t",
}

PARA_GATE_MAP = {
    G.RX: "rx",
    G.RY: "ry",
    G.RZ: "rz",
    G.Rxx: "rxx",
    G.Ryy: "ryy",
    G.Rzz: "rzz",
    G.PhaseShift: "ps",
}

GATE_MAP = {}
GATE_MAP.update(FIX_GATE_MAP)
GATE_MAP.update(PARA_GATE_MAP)


def convert_circ(circ: Circuit):
    """Convert a quantum circuit to string format."""
    out = []
    for g in circ:
        obj = g.obj_qubits
        ctrl = g.ctrl_qubits
        g_type = g.__class__
        if isinstance(g, (G.SGate, G.TGate)):
            name = g.name.lower() + ("dag" if g.hermitianed else "")
        else:
            name = GATE_MAP[g_type]
        out.append({"name": name, "obj": obj, "ctrl": ctrl})
        if g_type in PARA_GATE_MAP:
            coeff: ParameterResolver = g.coeff
            if coeff.is_const():
                val = coeff.const
            else:
                val = coeff.params_name[0]
            out[-1]["val"] = val
    return out


if __name__ == "__main__":
    from mindquantum.utils import random_circuit

    circuit = random_circuit(3, 10)
    circuit.rx("a", 0)
    print(convert_circ(circuit))
