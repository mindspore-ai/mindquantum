import os

import pytest

from mindquantum import Circuit, Hamiltonian, Simulator
from mindquantum.algorithm.nisq.chem import generate_uccsd
from mindquantum.core import gates as G

os.environ.setdefault('OMP_NUM_THREADS', '8')

_has_mindspore = True
try:
    import mindspore as ms

    from mindquantum.framework import MQAnsatzOnlyLayer

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
except ImportError:
    _has_mindspore = False


@pytest.mark.skipif(not _has_mindspore, reason='MindSpore is not installed')
def test_vqe_net():
    """
    Description: Test vqe
    Expectation:
    """
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
    (
        ansatz_circuit,
        init_amplitudes,
        ansatz_parameter_names,
        hamiltonian_qubitop,
        n_qubits,
        n_electrons,
    ) = generate_uccsd('./tests/st/H4.hdf5', th=-1)
    hf_circuit = Circuit([G.X.on(i) for i in range(n_electrons)])
    vqe_circuit = hf_circuit + ansatz_circuit
    sim = Simulator('projectq', vqe_circuit.n_qubits)
    f_g_ops = sim.get_expectation_with_grad(Hamiltonian(hamiltonian_qubitop.real), vqe_circuit)
    molecule_pqcnet = MQAnsatzOnlyLayer(f_g_ops)
    optimizer = ms.nn.Adagrad(molecule_pqcnet.trainable_params(), learning_rate=4e-2)
    train_pqcnet = ms.nn.TrainOneStepCell(molecule_pqcnet, optimizer)
    eps = 1e-8
    energy_diff = 1.0
    energy_last = 1.0
    iter_idx = 0
    iter_max = 100
    while (abs(energy_diff) > eps) and (iter_idx < iter_max):
        energy_i = train_pqcnet().asnumpy()
        energy_diff = energy_last - energy_i
        energy_last = energy_i
        iter_idx += 1

    assert round(energy_i.item(), 3) == -2.166
