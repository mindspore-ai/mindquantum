import numpy as np
from mindquantum.core import gates as G
from mindquantum.core.parameterresolver import ParameterResolver

# Non parameterized gate
h = G.H.on(0, 1)  # A hadamard gate act on qubit 0 and controlled by qubit 1
# H(0 <-: 1)

# Controlled by any qubits
multi_control_h = G.H.on(
    0, [1, 2, 3])  # Controlled by qubit 1, qubit 2 and qubit 3.
# H(0 <-: 1 2 3)

# Parameterized gate
var_a = ParameterResolver('a')
rx = G.RX(var_a).on(0)
# RX(a|0)

# Or simple way
rx = G.RX('a').on(0)
# RX(a|0)

# Custom a matrix gate
h_mat = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
custom_h = G.UnivMathGate('H', h_mat).on(0, [1, 2, 3])

# H(0 <-: 1 2 3)


# Custom a parameterized gate, please make sure numba is installed.
def rx_mat(ang):
    return np.array([[np.cos(ang / 2), -np.sin(ang / 2) * 1j],
                     [-np.sin(ang / 2) * 1j,
                      np.cos(ang / 2)]])


def rx_diff_mat(ang):
    return np.array([[-np.sin(ang / 2), -np.cos(ang / 2) * 1j],
                     [-np.cos(ang / 2) * 1j,
                      -np.sin(ang / 2)]]) / 2
custom_rx = G.gene_univ_parameterized_gate('RX', rx_mat, rx_diff_mat)
rx = custom_rx('a').on(0)
# RX(a | 0)
