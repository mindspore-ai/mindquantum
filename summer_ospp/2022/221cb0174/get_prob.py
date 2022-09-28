# This code is part of Mindquantum.
# This code add function that measure the probabiliy of circuit without do samplings
# ============================================================================

from mindquantum.core.circuit import Circuit
import numpy as np


def get_prob(circ: Circuit, qubit):
    r"""
    Args:
      circuit (Circuit): The circuit that you want to evolution and do sampling.
      qubit (number): the index of qubit to be measured.

    Returns:
      Exact probability without making samplings.

    Examples:
      >>> circ = Circuit().ry(1.1, 0).ry(2.2, 1)
      >>> res = get_prob(circ)
      >>> print("exact probability is:", res)
      >>> circ += Measure('q0_0').on(0)
      >>> sim = Simulator('projectq', circ.n_qubits)
      >>> res = sim.sampling(circ, shots=1000, seed=42)
      >>> print(res)
      "exact probability is:" {'0': 0.7267980607127886, '1': 0.27320193928721137}
      shots: 1000
      Keys: q0_0│0.00   0.177       0.353        0.53       0.707       0.884
      ──────────┼───────────┴───────────┴───────────┴───────────┴───────────┴
               0│▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
                │
               1│▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒
                │
      {'0': 707, '1': 293}
    """
    state_vector = circ.get_qs()
    if qubit == 0:
        vec = state_vector.reshape(-1, 2)
        res = np.einsum("ab, ab -> b", np.conj(vec), vec).real
        return {"0": res[0], "1": res[1]}
    else:
        tensor_shape = [2**(circ.n_qubits-qubit-1), 2, 2**qubit]
        vec = state_vector.reshape(tensor_shape)
        res = np.einsum("abc, abc->b", np.conj(vec), vec).real
        return {"0": res[0], "1": res[1]}

