import os
import sys

sys.path.append(os.path.abspath(__file__))
from typing import Union
from mindquantum import Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.core import gates as G
from mindquantum.core import Circuit
import inspect
from noise_model import generate_noise_model


def _gate_restriction(g: G.BasicGate):
    if isinstance(g, (G.Measure, G.BarrierGate, G.NoiseGate)):
        return True
    if isinstance(g, G.XGate):
        if len(g.ctrl_qubits) <= 1:
            return True
        return False
    if isinstance(g, G.CNOTGate):
        if len(g.ctrl_qubits) == 0:
            return True
        return False
    if isinstance(g, G.YGate):
        if len(g.ctrl_qubits) == 0:
            return True
        return False
    if isinstance(g, G.ZGate):
        if len(g.ctrl_qubits) <= 1:
            return True
        return False
    if isinstance(g, G.HGate):
        if len(g.ctrl_qubits) == 0:
            return True
        return False
    if isinstance(g, (G.RX, G.RY, G.RZ)):
        if len(g.ctrl_qubits) == 0:
            return True
        return False
    return False


def gate_restriction(obj: Union[G.BasicGate, Circuit]):
    if isinstance(obj, G.BasicGate):
        return _gate_restriction(obj)
    if isinstance(obj, Circuit):
        for g in obj:
            if not _gate_restriction(g):
                return False
        return True
    return False


class ShotsCounter:

    def __init__(self):
        self.count = 0
        self.called = False

    def increase(self, shots: int):
        assert shots >= 0, 'shots should not be less than 0'
        self.count += shots
        self.called = True

    def __setattr__(self, __name, __value) -> None:
        if __name in ['__init__', 'increase']:
            raise RuntimeError("DO NOT be evil.")
        if __name != 'count':
            self.__dict__[__name] = __value
            return
        try:
            stack = inspect.stack()
            assert stack[1].function in ['__init__', 'increase']
            self.__dict__[__name] = __value
        except:
            raise RuntimeError("DO NOT try to modify count by your self!!!")


def init_shots_counter():
    global shots_counter
    shots_counter = ShotsCounter()


noise_model = generate_noise_model()


class HKSSimulator(Simulator):
    """
    黑客松比赛专用量子模拟器，使用方法跟现有模拟器类似，但是会自动限制能使用的量子门种类。
    模拟器会限制接口，只能使用量子计算机上支持的操作，且会自动给量子门添加噪声，也会自动统计采样次数。

    Examples:
        >>> sim = HKSSimulator('mqvector', 3)
        >>> circ = qft(range(3)).measure_all()
        >>> sim.sampling(circ, shots=100)
    """

    def sampling(self, circuit, pr=None, shots=1, seed=None):
        # if not gate_restriction(circuit):
        #     raise RuntimeError("Gate not allowed.")
        shots_counter.increase(shots)
        return super().sampling(noise_model(circuit), pr, shots, seed)

    def apply_gate(self, gate, pr=None, diff=False):
        if not gate_restriction(gate):
            raise RuntimeError("Gate not allowed.")
        return super().apply_circuit(noise_model(Circuit([gate]), pr))

    def apply_circuit(self, circuit, pr=None):
        if not gate_restriction(circuit):
            raise RuntimeError("Gate not allowed.")
        return super().apply_circuit(noise_model(circuit), pr)

    def get_expectation(self,
                        hamiltonian,
                        circ_right=None,
                        circ_left=None,
                        simulator_left=None,
                        pr=None):
        raise RuntimeError("Calling get_expectation is not allowed.")

    def get_expectation_with_grad(self,
                                  hams,
                                  circ_right,
                                  circ_left=None,
                                  simulator_left=None,
                                  parallel_worker=None,
                                  pr_shift=False):
        raise RuntimeError("Call get_expectation_with_grad is not allowed.")

    def apply_hamiltonian(self, hamiltonian: Hamiltonian):
        raise RuntimeError("Call apply_hamiltonian is not allowed.")

    def get_qs(self, ket=False):
        raise RuntimeError("Call get_qs is not allowed.")

    def copy(self):
        raise RuntimeError("Call copy is not allowed.")

    def set_qs(self, quantum_state):
        raise RuntimeError("Call set_qs is not allowed.")
