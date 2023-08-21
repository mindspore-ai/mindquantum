"""Some circuits used in NISQ device."""

from .gates import X, RY, BaseGate
from .circuit import Circuit


def hardware_efficent_ansatz(
        n_qubit,
        single_rot_gate_seq=[RY],
        entangle_gate=X,
        depth=3):
    """Get a hardware efficient ansatz.
    
    Args:
        n_qubit (int): Number of qubit.
        single_rot_gate_seq (list): The rotate gate.
        entangle_gate (BaseGate): The entangle gate.
        depth (int): Number of repetitions of gates.
    """
    gates = []
    for i in range(n_qubit):
        for rgate in single_rot_gate_seq:
            gates.append(rgate().on(i))
    for _ in range(depth):
        for i in range(n_qubit - 1):
            gates.append(entangle_gate(i+1, i))
        for i in range(n_qubit):
            for rgate in single_rot_gate_seq:
                gates.append(rgate().on(i))
    return Circuit(gates)
