import numpy as np
import math
import mindspore as ms
from mindspore.common.parameter import Parameter
import mindspore.context as context

from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from mindquantum.core.gates import X
from mindquantum.core.circuit import Circuit, pauli_word_to_circuits, controlled, dagger
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.algorithm.nisq import generate_uccsd

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")

dist = 2.0
geometry = [
    ["H", [0.0, 0.0, 1.0 * dist]],
    ["H", [0.0, 0.0, 2.0 * dist]],
    ["H", [0.0, 0.0, 3.0 * dist]],
    ["H", [0.0, 0.0, 4.0 * dist]],
]
basis = "sto3g"
spin = 0

molecule_of = MolecularData(geometry, basis, multiplicity=2 * spin + 1)
molecule_of = run_pyscf(molecule_of, run_scf=1, run_ccsd=1, run_fci=1)
print("Hartree-Fock energy: %20.16f Ha" % (molecule_of.hf_energy))
print("CCSD energy: %20.16f Ha" % (molecule_of.ccsd_energy))
print("FCI energy: %20.16f Ha" % (molecule_of.fci_energy))

molecule_of.save()
molecule_file = molecule_of.filename

hartreefock_wfn_circuit = Circuit(
    [X.on(i) for i in range(molecule_of.n_electrons)])
print(hartreefock_wfn_circuit)

ansatz_circuit, \
init_amplitudes, \
ansatz_parameter_names, \
hamiltonian_QubitOp, \
n_qubits, n_electrons = generate_uccsd(molecule_file)

total_circuit = hartreefock_wfn_circuit + ansatz_circuit
total_circuit.summary()

sim = Simulator('mqvector', total_circuit.n_qubits)
molecule_pqc = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_QubitOp),
                                             total_circuit)
molecule_pqcnet = MQAnsatzOnlyLayer(molecule_pqc, 'Zeros')
initial_energy = molecule_pqcnet()

optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(),
                          learning_rate=4e-2)
train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)

eps = 1.e-8
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while abs(energy_diff) > eps:
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 5 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", molecule_pqcnet.weight.asnumpy())

U = ansatz_circuit.apply_value(
    dict(zip(ansatz_parameter_names, list(molecule_pqcnet.weight.asnumpy()))))
Udag = dagger(U)

p = list(hamiltonian_QubitOp.split())
emptycirc = Circuit()
for i in range(n_qubits):
    emptycirc += X.on(i)
    emptycirc += X.on(i)
a = pauli_word_to_circuits(p[0][1])
circ = emptycirc + a
hamatrix = p[0][0].const * circ.matrix()
n_pau = len(p)
for i in range(1, n_pau):
    a = pauli_word_to_circuits(p[i][1])
    circ = emptycirc + a
    hamatrix += p[i][0].const * circ.matrix()
hamatrix = np.dot(np.dot(Udag.matrix(), hamatrix), U.matrix())
hamatrix = hamatrix.real

# with open('H.npy', 'wb') as f:
# np.save(f, hamatrix)
# with open('h4.npy', 'rb') as f:
# hamatrix = np.load(f)

hf = 2**molecule_of.n_electrons - 1

T = 700
dt = 0.01
zeta = 1.0
A = 10
maxpop = 10000

D = set([hf])
Nmax = 2**n_qubits
pospop = [0 for i in range(Nmax)]
pospop[hf] = 1
negpop = [0 for i in range(Nmax)]
Slast = 0
S = 0
Nlast = 1
N = 1
for n in range(T):
    toadd = []
    for i in D:
        if max(pospop[i], negpop[i]) == 0:
            continue
        isign = 1
        if negpop[i] > 0:
            isign = -1
        for w in range(Nmax):
            if w == i:
                hii = hamatrix[i, i]
                p_i = dt * (hii - S)
                ran = np.random.uniform(0, 1, max(pospop[i], negpop[i]))
                n_new = np.sum(ran < np.abs(p_i))
                if n_new > 0:
                    if p_i < 0:
                        if isign > 0:
                            pospop[i] += n_new
                        else:
                            negpop[i] += n_new
                        N += n_new
                    else:
                        if isign > 0:
                            pospop[i] -= n_new
                        else:
                            negpop[i] -= n_new
                        N -= n_new
                continue
            Hjisign = 1
            Hji = hamatrix[w, i]
            if Hji < 0:
                Hjisign = -1
            if np.abs(Hji) > 1e-8:
                ran = np.random.uniform(0, 1, max(pospop[i], negpop[i]))
                prob = dt * np.abs(Hji)
                n_new = np.sum(ran < prob)
                if n_new > 0:
                    toadd.append(w)
                    if isign * Hjisign > 0:
                        negpop[w] += n_new
                    else:
                        pospop[w] += n_new
                    N += n_new
    for i in D:
        if pospop[i] > negpop[i]:
            pospop[i] -= negpop[i]
            N -= 2 * negpop[i]
            negpop[i] = 0
        else:
            negpop[i] -= pospop[i]
            N -= 2 * pospop[i]
            pospop[i] = 0

    if n % A == A - 1:
        if N > maxpop:
            S = Slast - zeta / A / dt * np.log(N / Nlast)
            Slast = S
        Nlast = N

    for i in toadd:
        D.add(i)

    if n % 50 == 49:
        temp = sum(pospop) + sum(negpop)
        mixed_energy = energy_i[0]
        for i in range(Nmax):
            if i != 15:
                if pospop[i] > 0:
                    mixed_energy += hamatrix[i, hf] * pospop[i] / pospop[hf]
                else:
                    mixed_energy -= hamatrix[i, hf] * negpop[i] / pospop[hf]
        print('step %d: mixed energy = %.10f, total pupulation = %d' %
              (n + 1, mixed_energy, temp))

mixed_energy = energy_i[0]

for i in range(Nmax):
    if i != 15:
        if pospop[i] > 0:
            mixed_energy += hamatrix[i, hf] * pospop[i] / pospop[hf]
        else:
            mixed_energy -= hamatrix[i, hf] * negpop[i] / pospop[hf]

print('VQE: %.10f' % energy_i[0])
print('QQMC: %.10f' % mixed_energy)
print('FCI: %.10f' % molecule_of.fci_energy)