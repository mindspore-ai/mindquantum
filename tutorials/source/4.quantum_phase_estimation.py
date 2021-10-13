from mindquantum import Circuit
from mindquantum import Simulator
from mindquantum import UN, PhaseShift, qft, H, X, BARRIER
import numpy as np

n = 3
c = Circuit()
c += UN(H, n)
c += X.on(n)
print(c)

for i in range(n):
    c += PhaseShift({'phi': 2**i}).on(n, n - i - 1)
print(c)

c += BARRIER
c += qft(range(n)).hermitian()
print(c)

from mindquantum import Measure
sim = Simulator('projectq', c.n_qubits)
phi = 0.125
sim.apply_circuit(c, {'phi': 2 * np.pi * phi})
qs = sim.get_qs()
print(sim.get_qs(ket=True))
res = sim.sampling(UN(Measure(), c.n_qubits), shots=100)
print(res)

index = np.argmax(np.abs(qs))
print(index)

bit_string = bin(index)[2:].zfill(c.n_qubits)[1:]
print(bit_string)

theta_exp = int(bit_string[::-1], 2) / 2**n
print(theta_exp)
