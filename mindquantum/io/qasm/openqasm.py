# Copyright 2021 Huawei Technologies Co., Ltd
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
"""OpenQASM support module."""
# pylint: disable=import-outside-toplevel
import ast
import operator as op
import re

import numpy as np

from mindquantum.utils import fdopen
from mindquantum.utils.type_value_check import _check_input_type


def join_qubit(gate):
    """Convert qubit list to openqasm style."""
    return ','.join(f'q[{i}]' for i in gate.ctrl_qubits + gate.obj_qubits)


def x_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if len(ctrl) == 1:
        return f"cx {jointed};"
    if not ctrl:
        return f"x {jointed};"
    if len(ctrl) == 2:
        return f"ccx {join_qubit};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def y_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if len(ctrl) == 1:
        return f"cy {jointed};"
    if not ctrl:
        return f"y {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def z_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if len(ctrl) == 1:
        return f"cz {jointed};"
    if not ctrl:
        return f"z {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def cnot_related(gate):
    """Convert mindquantum gate to qasm."""
    from mindquantum.core import gates as G  # noqa: N812

    return x_related(G.X.on(gate.obj_qubits[0], gate.obj_qubits[1:] + gate.ctrl_qubits))


def global_phase_related(gate):
    """Convert mindquantum gate to qasm."""
    if len(gate.ctrl_qubits) == 1:
        return f"p({-gate.coeff.const}) {gate.ctrl_qubits[0]};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def h_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if len(ctrl) == 1:
        return f"ch {jointed};"
    if not ctrl:
        return f"h {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def i_related(gate):
    """Convert mindquantum gate to qasm."""
    return f"id q[{gate.obj_qubits[0]}];"


def phase_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    coeff = gate.coeff.const
    if len(ctrl) == 1:
        return f"cp({coeff}) {jointed};"
    if not ctrl:
        return f"p({coeff}) {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def rx_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    coeff = gate.coeff.const
    if len(ctrl) == 1:
        return f"crx({coeff}) {jointed};"
    if not ctrl:
        return f"rx({coeff}) {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def ry_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    coeff = gate.coeff.const
    if len(ctrl) == 1:
        return f"cry({coeff}) {jointed};"
    if not ctrl:
        return f"ry({coeff}) {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def rz_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    coeff = gate.coeff.const
    if len(ctrl) == 1:
        return f"crz({coeff}) {jointed};"
    if not ctrl:
        return f"rz({coeff}) {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def rxx_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    coeff = gate.coeff.const
    if not ctrl:
        return f"rxx({coeff}) {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def rzz_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    coeff = gate.coeff.const
    if not ctrl:
        return f"rzz({coeff}) {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def s_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if not ctrl:
        if not gate.hermitianed:
            return f"s {jointed};"
        return f"sdg {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def t_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if not ctrl:
        if not gate.hermitianed:
            return f"t {jointed};"
        return f"tdg {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def sx_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if not ctrl:
        if not gate.hermitianed:
            return f"sx {jointed};"
        return f"sxdg {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def swap_related(gate):
    """Convert mindquantum gate to qasm."""
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if len(ctrl) == 1:
        return f"cswap {jointed};"
    if not ctrl:
        return f"swap {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def u3_related(gate):
    """Convert mindquantum gate to qasm."""
    p_0, p_1, p_2 = gate.prs
    p_0 = p_0.const
    p_1 = p_1.const
    p_2 = p_2.const
    ctrl = gate.ctrl_qubits
    jointed = join_qubit(gate)
    if len(ctrl) == 1:
        return f"cu({p_0},{p_1},{p_2}) {jointed};"
    if not ctrl:
        return f"u({p_0},{p_1},{p_2}) {jointed};"
    raise ValueError(f"Cannot convert {gate} to qasm.")


def extra_qid(cmd, qregs, cregs):
    """Get qubit id."""
    regs = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\[\d+\]', cmd)
    if regs:
        out = []
        for i in regs:
            reg_name = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', i)[0]
            reg_id = int(re.findall(r'\d+', i)[0])
            if reg_name in qregs:
                out.append(qregs[reg_name][reg_id])
            elif reg_name in cregs:
                out.append(cregs[reg_name][reg_id])
            else:
                raise ValueError(f"Registor {reg_name} not allocated.")
        return (out,)
    possible_name = [i for i in re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', cmd) if i in qregs or i in cregs]
    out = np.array([qregs[i] if i in qregs else cregs[i] for i in possible_name]).T
    return tuple([int(j) for j in i] for i in out)


def extra_params(cmd, dtype=float):
    """Get gate parameters."""
    matches = re.findall(r'\((.*)\)', cmd)
    out = []
    for i in matches:
        for j in i.split(','):
            pr = j.strip()
            if dtype == str:
                out.append(pr)
            else:
                if '*' in pr:
                    pr = pr.replace('pi', str(np.pi)).replace('π', str(np.pi))
                    pr = [float(i.strip()) for i in pr.split('*')]
                    out.append(pr[0] * pr[1])
                elif '/' in pr:
                    pr = pr.replace('pi', str(np.pi)).replace('π', str(np.pi))
                    pr = [float(i.strip()) for i in pr.split('/')]
                    out.append(pr[0] / pr[1])
                else:
                    out.append(float(pr))
    return out


def extra_gate_name(cmd: str):
    """Get gate name."""
    cmd = cmd.strip()
    space_idx = cmd.find(' ')
    if space_idx == -1:
        raise ValueError(f"Wrong command: {cmd}")
    sub_cmd = cmd[:space_idx]
    if '(' in sub_cmd:
        return sub_cmd[: sub_cmd.find('(')].strip()
    return sub_cmd.strip()


def parse_gate_cmd(cmd, qregs, cregs):
    """Parse a gate command."""
    try:
        cmd = cmd.strip()
        gate_name = extra_gate_name(cmd)
        qid = extra_qid(cmd, qregs, cregs)
        if gate_name == 'pauli':
            params = extra_params(cmd, dtype=str)
        else:
            params = extra_params(cmd)
        return gate_name, params, qid
    except:  # pylint: disable=raise-missing-from # noqa: E722
        raise ValueError(f"Cannot parse command: {cmd}")


def gene_reg(cmds):
    """Parse quantum and classical registor."""
    qregs = {}
    cregs = {}
    total_q, total_c = 0, 0
    for i in cmds:
        if i.startswith('creg'):
            creg_name = re.findall(r'\s(.*)?\[', i)[0]
            n_c_this = int(re.findall(r'\[(.*)?\]', i)[0])
            cregs[creg_name] = list(range(total_c, total_c + n_c_this))
            total_c += n_c_this
            continue
        if i.startswith('qreg'):
            qreg_name = re.findall(r'\s(.*)?\[', i)[0]
            n_q_this = int(re.findall(r'\[(.*)?\]', i)[0])
            qregs[qreg_name] = list(range(total_q, total_q + n_q_this))
            total_q += n_q_this
            continue
    return qregs, total_q, cregs, total_c


def extra_func_body(cmd):
    """Get definition of custom gate."""
    matches = re.findall(r'\{(.*)\}', cmd)
    return [i.strip() for i in matches[0].strip().split(';')]


def extra_func_head(cmd):
    """Get gate name and parameters for custom gate."""
    cmd = cmd.replace('gate ', '')
    cmd = cmd[: cmd.find('{')].strip()
    if '(' in cmd:
        name = cmd[: cmd.find('(')].strip()
        params = extra_params(cmd, dtype=str)
        qid = re.findall(r'\)(.*)', cmd)[0].strip().split(',')
        qid = [i.strip() for i in qid]
        return name, params, qid
    name = cmd[: cmd.find(' ')].strip()
    qid = cmd[cmd.find(' ') :].strip().split(',')
    qid = [i.strip() for i in qid]
    return name, [], qid


def eval_expr(expr):
    """Safe eval of string expression."""
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    """Safe eval of string expression."""
    operators = {
        ast.Add: op.add,
        ast.Sub: op.sub,
        ast.Mult: op.mul,
        ast.Div: op.truediv,
        ast.Pow: op.pow,
        ast.USub: op.neg,
    }

    if isinstance(node, ast.Num):
        return node.n
    if isinstance(node, ast.BinOp):
        return operators[type(node.op)](eval_(node.left), eval_(node.right))
    if isinstance(node, ast.UnaryOp):
        return operators[type(node.op)](eval_(node.operand))
    raise TypeError(node)


def eval_pr(pr, prs, gate_params):
    """Calculate the parameters."""
    pr = pr.replace('pi', str(np.pi)).replace('π', str(np.pi))
    for idx, p in enumerate(gate_params):
        pr = pr.replace(p, str(prs[idx]))
    return eval_expr(pr)


def generate_custom_gate(gate_def, custom_gate, gate_map_openqasm_mq):
    """Generate a custom gate defined in openqasm."""
    from mindquantum.core.circuit import Circuit

    gate_name, gate_params, gate_qids = extra_func_head(gate_def)
    body = extra_func_body(gate_def)

    def gate(prs, qids):
        circ = Circuit()
        for cmd in body:
            if not cmd:
                continue
            sub_gate_name = extra_gate_name(cmd)
            sub_gate_params = extra_params(cmd, str)
            if sub_gate_params:
                sub_gate_qid = cmd[cmd.find(')') + 1 :]
            else:
                sub_gate_qid = cmd[cmd.find(' ') :]
            sub_gate_qid = [i.strip() for i in sub_gate_qid.split(',')]
            real_qid = [qids[gate_qids.index(i)] for i in sub_gate_qid]
            real_prs = [eval_pr(i, prs, gate_params) for i in sub_gate_params]
            if sub_gate_name not in gate_map_openqasm_mq:
                if sub_gate_name not in custom_gate:
                    raise ValueError(f"{sub_gate_name} not defined.")
                circ += custom_gate[sub_gate_name](real_prs, real_qid)
            else:
                circ += gate_map_openqasm_mq[sub_gate_name](real_prs, real_qid)
        return circ

    custom_gate[gate_name] = gate


def mq_to_qasm_v2(circ, gate_map_mq_openqasm, version: str = '2.0'):
    """Convert mindquantum circuit to openqasm."""
    from mindquantum.algorithm.compiler import BasicDecompose, compile_circuit
    from mindquantum.core import gates as G  # noqa: N812
    from mindquantum.core.circuit import Circuit

    if version != '2.0':
        raise ValueError("Only support qasm version 2.0")
    if circ.is_noise_circuit:
        raise ValueError("Cannot convert noise circuit to qasm.")
    cmds = ['OPENQASM 2.0;', 'include "qelib1.inc";', f'qreg q[{circ.n_qubits}];']
    mea_qubits = sorted({m.obj_qubits[0] for m in circ.all_measures.keys()})
    reg_map = {j: i for i, j in enumerate(mea_qubits)}
    if mea_qubits:
        cmds.append(f"creg c[{len(mea_qubits)}];")
    for gate in circ:
        gate: G.BasicGate
        if isinstance(gate, G.BarrierGate):
            obj = gate.obj_qubits
            if not gate.obj_qubits:
                obj = list(range(circ.n_qubits))
            cmds.append(f"barrier {join_qubit(G.BARRIER.on(obj))};")
            continue
        if isinstance(gate, G.Measure):
            cmds.append(f"measure q[{gate.obj_qubits[0]}] -> c[{reg_map[gate.obj_qubits[0]]}];")
            continue
        if gate.__class__ in gate_map_mq_openqasm:
            cmds.append(gate_map_mq_openqasm[gate.__class__](gate))
            continue
        compiled_circ = compile_circuit(BasicDecompose(), Circuit([gate]))
        for sub_gate in compiled_circ:
            if sub_gate.__class__ in gate_map_mq_openqasm:
                cmds.append(gate_map_mq_openqasm[sub_gate.__class__](sub_gate))
            else:
                raise ValueError(f"Cannot convert {sub_gate} to openqasm after decompose gate {gate}")
    return '\n'.join(cmds)


def prepare_qasm(qasm: str):
    """Remove new line between bracket."""
    qasm = re.sub(r'//.*?\n', '\n', qasm.lower())
    new_qasm = ''
    start_body = False
    for i in qasm:
        if i == '{':
            start_body = True
            new_qasm = new_qasm.strip()
        if i == '\n' and start_body:
            continue
        if i == '}':
            start_body = False
        new_qasm += i
    return [i.strip() for i in new_qasm.strip().split('\n')]


# pylint: disable=too-many-locals,too-many-branches
def qasm_to_mq_v2(qasm: str, gate_map_openqasm_mq):
    """Convert openqasm to mindquantum circuit."""
    from mindquantum.core.circuit import Circuit

    cmds = prepare_qasm(qasm)
    circ = Circuit()
    custom_gate = {}
    measures = {}
    qregs, _, cregs, _ = gene_reg(cmds)
    for i in cmds:
        if i.startswith('openqasm'):
            if not i.endswith('2.0;'):
                raise ValueError(f"OpenQASM version not supported: {i}")
            continue
        if i.startswith("include"):
            if not i.endswith('"qelib1.inc";'):
                raise ValueError(f"Include other head file not supported: {i}")
            continue
        if i.startswith('qreg') or i.startswith('creg') or not i:
            continue
        if i.startswith('gate '):
            generate_custom_gate(i, custom_gate, gate_map_openqasm_mq)
        else:
            cmd = i.strip()
            gate_name, params, qids = parse_gate_cmd(cmd, qregs, cregs)
            if gate_name == 'barrier':
                circ += gate_map_openqasm_mq['barrier'](params, [q[0] for q in qids])
                continue
            for qid in qids:
                if gate_name in custom_gate:
                    circ += custom_gate[gate_name](params, qid)
                elif gate_name in gate_map_openqasm_mq:
                    mq_gate = gate_map_openqasm_mq[gate_name](params, qid)
                    if gate_name == 'measure':
                        mq_gate.key = f"q{qid[0]}_{measures.get(qid[0], 0)}"
                        measures[qid[0]] = measures.get(qid[0], 0) + 1
                    circ += mq_gate
                else:
                    raise ValueError(f"{cmd} not implement.")
    return circ


class OpenQASM:
    """
    Convert a circuit to openqasm format.

    Examples:
        >>> import numpy as np
        >>> from mindquantum.core import Circuit
        >>> from mindquantum.io import OpenQASM
        >>> circuit = Circuit().rx(0.3, 0).z(0, 1).zz(np.pi, [0, 1])
        >>> openqasm = OpenQASM()
        >>> circuit_str = openqasm.to_string(circuit)
        >>> print(circuit_str)
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        rx(0.3) q[0];
        cz q[1],q[0];
        rzz(6.283185307179586) q[0],q[1];
    """

    def __init__(self):
        """Construct a openqasm parse."""
        from mindquantum.core import gates as G  # noqa: N812
        from mindquantum.core.circuit import Circuit

        self.gate_map_openqasm_mq = {
            'barrier': lambda prs, qids: Circuit([G.BarrierGate().on(qids)]),
            'ccx': lambda prs, qids: G.X.on(qids[-1], qids[:-1]),
            'ch': lambda prs, qids: G.H.on(qids[-1], qids[:-1]),
            'cp': lambda prs, qids: G.PhaseShift(prs[0]).on(qids[-1], qids[:-1]),
            'crx': lambda prs, qids: G.RX(prs[0]).on(qids[-1], qids[:-1]),
            'cry': lambda prs, qids: G.RY(prs[0]).on(qids[-1], qids[:-1]),
            'crz': lambda prs, qids: G.RZ(prs[0]).on(qids[-1], qids[:-1]),
            'cswap': lambda prs, qids: G.SWAP.on(qids[1:], qids[0]),
            'cx': lambda prs, qids: G.X.on(qids[1], qids[0]),
            'csx': lambda prs, qids: Circuit(
                [
                    G.RX(np.pi / 2).on(qids[1], qids[0]),
                    G.PhaseShift(np.pi / 2).on(qids[0]),
                ]
            ),
            'cu': lambda prs, qids: Circuit(
                [
                    G.U3(prs[0], prs[1], prs[2]).on(qids[1], qids[0]),
                    G.PhaseShift(prs[3]).on(qids[0]),
                ]
            ),
            'cu1': lambda prs, qids: G.U3(0, 0, prs[0]).on(qids[1], qids[0]),
            'cu3': lambda prs, qids: G.U3(prs[0], prs[1], prs[2]).on(qids[1], qids[0]),
            'cy': lambda prs, qids: G.Y.on(qids[1], qids[0]),
            'cz': lambda prs, qids: G.Z.on(qids[1], qids[0]),
            'h': lambda prs, qids: G.H.on(qids[0]),
            'id': lambda prs, qids: G.I.on(qids[0]),
            'measure': lambda prs, qids: G.Measure().on(qids[0]),
            'p': lambda prs, qids: G.PhaseShift(prs[0]).on(qids[0]),
            'rccx': lambda prs, qids: Circuit(
                [
                    G.H.on(qids[2]),
                    G.T.on(qids[2]),
                    G.X.on(qids[2], qids[1]),
                    G.T.on(qids[2]).hermitian(),
                    G.X.on(qids[2], qids[0]),
                    G.T.on(qids[2]),
                    G.X.on(qids[2], qids[1]),
                    G.T.on(qids[2]).hermitian(),
                    G.H.on(qids[2]),
                ]
            ),
            'rx': lambda prs, qids: G.RX(prs[0]).on(qids),
            'ry': lambda prs, qids: G.RY(prs[0]).on(qids),
            'rz': lambda prs, qids: G.RZ(prs[0]).on(qids),
            'rxx': lambda prs, qids: G.Rxx(prs[0]).on(qids),
            'ryy': lambda prs, qids: G.Ryy(prs[0]).on(qids),
            'rzz': lambda prs, qids: G.Rzz(prs[0]).on(qids),
            's': lambda prs, qids: G.S.on(qids),
            'sdg': lambda prs, qids: G.S.on(qids).hermitian(),
            'swap': lambda prs, qids: G.SWAP.on(qids),
            'sx': lambda prs, qids: G.SX.on(qids),
            'sxdg': lambda prs, qids: G.SX.on(qids).hermitian(),
            't': lambda prs, qids: G.T.on(qids),
            'tdg': lambda prs, qids: G.T.on(qids).hermitian(),
            'u': lambda prs, qids: G.U3(*prs).on(qids),
            'u1': lambda prs, qids: G.U3(0, 0, prs[0]).on(qids),
            'u2': lambda prs, qids: G.U3(np.pi / 2, prs[0], prs[1]).on(qids),
            'u3': lambda prs, qids: G.U3(*prs).on(qids),
            'x': lambda prs, qids: G.X.on(qids),
            'y': lambda prs, qids: G.Y.on(qids),
            'z': lambda prs, qids: G.Z.on(qids),
        }

        self.gate_map_mq_openqasm = {
            G.XGate: x_related,
            G.YGate: y_related,
            G.ZGate: z_related,
            G.CNOTGate: cnot_related,
            G.GlobalPhase: global_phase_related,
            G.HGate: h_related,
            G.IGate: i_related,
            G.PhaseShift: phase_related,
            G.RX: rx_related,
            G.RY: ry_related,
            G.RZ: rz_related,
            G.Rxx: rxx_related,
            G.Rzz: rzz_related,
            G.SGate: s_related,
            G.TGate: t_related,
            G.SXGate: sx_related,
            G.SWAPGate: swap_related,
            G.U3: u3_related,
        }

    def to_string(self, circuit, version="2.0"):  # pylint: disable=R0912,R0914,R0915
        """
        Convert circuit to openqasm.

        Args:
            circuit (Circuit): The quantum circuit you want to translated to openqasm.
            version (str): The openqasm version you want to use. Default: ``'2.0'``.

        Returns:
            str, The openqasm format of input circuit.

        Raises:
            TypeError: if `circuit` is not a Circuit.
            TypeError: if `version` is not a str.
            NotImplementedError: if openqasm version not implement.
            ValueError: if gate not implement in this version.
        """
        return mq_to_qasm_v2(circuit, self.gate_map_mq_openqasm, version)

    def to_file(self, file_name, circuit, version="2.0"):
        """
        Convert a quantum circuit to openqasm format and save in file.

        Args:
            file_name (str): The file name you want to save the openqasm file.
            circuit (Circuit): The Circuit you want to convert.
            version (str): The version of openqasm. Default: ``'2.0'``.

        Raises:
            TypeError: if `file_name` is not a str.
            TypeError: if `circuit` is not a Circuit.
            TypeError: if `version` is not a str.
        """
        from mindquantum.core import (  # pylint: disable=import-outside-toplevel,cyclic-import
            Circuit,
        )

        if not isinstance(file_name, str):
            raise TypeError(f'file_name requires a str, but get {type(file_name)}')
        if not isinstance(circuit, Circuit):
            raise TypeError(f"circuit requires a Circuit, but get {type(circuit)}")
        if not isinstance(version, str):
            raise TypeError(f'version requires a str, but get {type(version)}')
        with fdopen(file_name, 'w') as fd:
            fd.writelines(self.to_string(circuit, version))
        print(f"write circuit to {file_name} finished!")

    def from_file(self, file_name):
        """
        Read a openqasm file.

        Args:
            file_name (str): The path of file that stored quantum circuit in openqasm format.

        Returns:
            Circuit, the quantum circuit translated from openqasm file.
        """
        with fdopen(file_name, 'r') as fd:
            cmds = fd.readlines()
        return qasm_to_mq_v2('\n'.join(cmds), self.gate_map_openqasm_mq)

    def from_string(self, string):
        """
        Read a OpenQASM string.

        Args:
            string (str): The OpenQASM string of Circuit.

        Returns:
            :class:`~.core.circuit.Circuit`, the quantum circuit translated from OpenQASM string.

        Examples:
            >>> from mindquantum.io import OpenQASM
            >>> from mindquantum.core.circuit import Circuit
            >>> circ = Circuit().x(0, 1).h(1)
            >>> string = OpenQASM().to_string(circ)
            >>> OpenQASM().from_string(string)
                  ┏━━━┓
            q0: ──┨╺╋╸┠─────────
                  ┗━┳━┛
                    ┃   ┏━━━┓
            q1: ────■───┨ H ┠───
                        ┗━━━┛
        """
        _check_input_type('string', str, string)
        return qasm_to_mq_v2(string, self.gate_map_openqasm_mq)
