import numpy as np
from typing import Any

from quri_parts.circuit import CONST, ParametricQuantumGate
from quri_parts.circuit.gate import QuantumGate

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, PhaseShift, U3, H, S, T, X, Y, Z, CNOT, SWAP, UnivMathGate, IGate
from mindquantum.core.parameterresolver import ParameterResolver as MQPR
from mindquantum.core.circuit import decompose_single_term_time_evolution
from mindquantum.core.gates import GroupedPauli

def quri_to_mindquantum(quri_circuit: Any) -> Circuit:
    """
    Convert a QURI Parts LinearMappedUnboundParametricQuantumCircuit to a MindQuantum Circuit.

    Args:
        quri_circuit (LinearMappedUnboundParametricQuantumCircuit): The QURI Parts circuit to convert.

    Returns:
        Circuit: The equivalent MindQuantum circuit.
    """
    # Collect unique input parameters and assign names 'p0', 'p1', ...
    quri_params = quri_circuit.param_mapping.in_params  # Tuple of Parameter objects
    param_names = [f'p{i}' for i in range(len(quri_params))]
    param_to_name = {p: param_names[i] for i, p in enumerate(quri_params)}

    # Initialize MindQuantum circuit
    mq_circ = Circuit()

    # Get parameter mappings as list of mappingproxy dicts (one per parametric gate)
    param_mappings = list(quri_circuit.param_mapping.mapping.values())

    # Counter for parametric positions
    param_pos = 0

    # Pauli ID mapping
    pauli_id_to_str = {1: 'X', 2: 'Y', 3: 'Z'}

    for gate in quri_circuit.gates:
        targets = list(gate.target_indices)
        controls = list(gate.control_indices)
        pauli_ids = list(gate.pauli_ids) if gate.pauli_ids else []

        if isinstance(gate, QuantumGate):  # Non-parametric gate
            params = gate.params
            if gate.name == 'Identity':
                g = IGate().on(targets[0], controls)
            elif gate.name == 'X':
                g = X.on(targets[0], controls)
            elif gate.name == 'Y':
                g = Y.on(targets[0], controls)
            elif gate.name == 'Z':
                g = Z.on(targets[0], controls)
            elif gate.name == 'H':
                g = H.on(targets[0], controls)
            elif gate.name == 'S':
                g = S.on(targets[0], controls)
            elif gate.name == 'Sdag':
                g = S.hermitian().on(targets[0], controls)
            elif gate.name == 'T':
                g = T.on(targets[0], controls)
            elif gate.name == 'Tdag':
                g = T.hermitian().on(targets[0], controls)
            elif gate.name == 'U1':
                g = PhaseShift(params[0]).on(targets[0], controls)
            elif gate.name == 'U2':
                g = U3(np.pi / 2, params[0], params[1]).on(targets[0], controls)
            elif gate.name == 'U3':
                g = U3(params[0], params[1], params[2]).on(targets[0], controls)
            elif gate.name == 'RX':
                g = RX(params[0]).on(targets[0], controls)
            elif gate.name == 'RY':
                g = RY(params[0]).on(targets[0], controls)
            elif gate.name == 'RZ':
                g = RZ(params[0]).on(targets[0], controls)
            elif gate.name == 'CNOT':
                g = CNOT.on(targets[0], controls)
            elif gate.name == 'CZ':
                g = Z.on(targets[0], controls)
            elif gate.name == 'SWAP':
                g = SWAP.on(targets, controls)
            elif gate.name == 'TOFFOLI':
                g = X.on(targets[0], controls)
            elif gate.name == 'Pauli':
                pauli_s = ''.join(pauli_id_to_str.get(p, 'I') for p in pauli_ids)
                g = GroupedPauli(pauli_s).on(targets)
            elif gate.name == 'PauliRotation':
                term = tuple((targets[i], pauli_id_to_str[p]) for i, p in enumerate(pauli_ids))
                angle = params[0]
                sub_circ = decompose_single_term_time_evolution(term, angle / 2)
                mq_circ += sub_circ
                continue
            elif gate.name == 'UnitaryMatrix':
                n_qubits = len(targets)
                mat = np.array(gate.unitary_matrix).reshape(2**n_qubits, 2**n_qubits)
                g = UnivMathGate(gate.name, mat).on(targets, controls)
            else:
                raise NotImplementedError(f"Conversion for fixed gate '{gate.name}' not implemented.")
            mq_circ += g

        elif isinstance(gate, ParametricQuantumGate):  # Parametric gate
            # Get the mapping for this parametric gate
            param_fn = param_mappings[param_pos]
            # Build pr_dict excluding CONST
            pr_dict = {param_to_name[k]: v for k, v in param_fn.items() if k is not CONST}
            pr = MQPR(pr_dict)
            # Add constant if present
            if CONST in param_fn:
                pr.const = param_fn[CONST]

            if gate.name == 'RX':
                g = RX(pr).on(targets[0], controls)
            elif gate.name == 'RY':
                g = RY(pr).on(targets[0], controls)
            elif gate.name == 'RZ':
                g = RZ(pr).on(targets[0], controls)
            elif gate.name == 'U1':
                g = PhaseShift(pr).on(targets[0], controls)
            elif gate.name == 'PauliRotation' or gate.name == 'ParametricPauliRotation':
                term = tuple((targets[i], pauli_id_to_str[p]) for i, p in enumerate(pauli_ids))
                pr = pr / 2  # PauliRotation angle is halved for decomposition
                sub_circ = decompose_single_term_time_evolution(term, pr)
                mq_circ += sub_circ
            else:
                raise NotImplementedError(f"Conversion for parametric gate '{gate.name}' not implemented.")
            if gate.name not in ['PauliRotation', 'ParametricPauliRotation']:
                mq_circ += g
            param_pos += 1

        else:
            raise TypeError(f"Unknown gate type: {type(gate)}")

    return mq_circ