from mindquantum.algorithm import amplitude_encoder
from mindquantum.simulator import Simulator

def test_amplitude_encoder():
    sim = Simulator('projectq', 3)
    circuit, params = amplitude_encoder([0.5, 0.5, 0.5, 0.5])
    sim.apply_circuit(circuit, params)
    st = sim.get_qs(False)
    for i in range(8):
        assert abs(st[i].real - (0.5 if i < 4 else 0)) < 1e-10
    circuit, params = amplitude_encoder([0, 0, 0.5, 0.5, 0.5, 0.5])
    sim.reset()
    sim.apply_circuit(circuit, params)
    st = sim.get_qs(False)
    assert abs(st[1].real - 0.5) < 1e-10
    assert abs(st[2].real - 0.5) < 1e-10
    assert abs(st[5].real - 0.5) < 1e-10
    assert abs(st[6].real - 0.5) < 1e-10
