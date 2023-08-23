"""Quantum circuit."""

from copy import deepcopy
import collections
from typing import List, Tuple

import numpy as np
from mindspore import Tensor
import mindspore.nn as nn

from .gates import BaseGate, WithParamGate
from .gates import DEFAULT_PARAM_NAME
from .define import DTYPE
from .utils import bprint


class Circuit(nn.Cell):
    """Quantum Circuit.
    """
    def __init__(self, gates=None):
        """
        Args:
            gates (BaseGate | List[BaseGate]): The quantum gates.
        """
        super().__init__()
        self.n_qubit = 0
        self.gates = nn.SequentialCell()
        self.params_name = []
        self._append(gates)
        self.qs0 = self._init_qs0()

    def construct(self):
        n_qubit, qs = self.gates((self.n_qubit, self.qs0))
        return n_qubit, qs

    def _append(self, arg):
        """Append a gate, a list of gates, or a circuit to original circuit.
        """
        if isinstance(arg, BaseGate):
            gate = arg
            self.gates.append(deepcopy(gate))
            self.n_qubit = max(self.n_qubit, gate.max_qubit_index + 1)
            if isinstance(gate, WithParamGate):
                self.params_name.append(gate.param_name)

        if isinstance(arg, List):
            for gate in arg:
                self.gates.append(deepcopy(gate))
                self.n_qubit = max(self.n_qubit, gate.max_qubit_index + 1)
                if isinstance(gate, WithParamGate):
                    self.params_name.append(gate.param_name)

        if isinstance(arg, Circuit):
            for gate in arg.gates:
                self.gates.append(deepcopy(gate))
                self.n_qubit = max(self.n_qubit, gate.max_qubit_index + 1)
                if isinstance(gate, WithParamGate):
                    self.params_name.append(gate.param_name)

        # update init_qs0 after knowing `n_qubit`
        self.qs0 = self._init_qs0()
        return self

    def _init_qs0(self):
        """Initial state of circuit.
        """
        a = [0.0] * 2**self.n_qubit
        a[0] = 1.0
        re = Tensor(a, DTYPE).reshape([2] * self.n_qubit)
        im = Tensor([0] * 2**self.n_qubit, DTYPE).reshape([2] * self.n_qubit)
        return re, im

    def get_qs(self, pr=None):
        """Get quantum state of the circuit.

        Args:
            pr (numpy.ndarray | Tensor | list | dict): Assign parameter(s) value of the circuit.
        """
        self.set_ansatz_value(pr)
        self.set_encoder_value(pr)
        _, qs = self.gates((self.n_qubit, self.qs0))
        qs2 = np.ravel(qs[0].numpy() + 1j * qs[1].numpy())
        return qs2

    def as_encoder(self):
        """Set the circuit as encoder, which doesn't calculate the gradient of parameters.
        """
        self.no_grad()
        return self

    def no_grad(self):
        """No gradient for the circuit parameters.
        """
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                gate.no_grad()
        return self

    def as_ansatz(self):
        """Set the circuit as ansatz, which calculates the gradient of parameters.
        """
        self.with_grad()
        return self

    def with_grad(self):
        """With gradient for the circuit parameters.
        """
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                gate.with_grad()
        return self

    def set_encoder_value(self, pr):
        """Set the encoder value.

        Args:
            pr (numpy.ndarray | Tensor | list | dict): Assign parameter(s) value of the circuit.
        """
        self.assign_params_value(pr, trainable=False)

    def set_ansatz_value(self, pr):
        """Set the encoder value.

        Args:
            pr (numpy.ndarray | Tensor | list | dict): Assign parameter(s) value of the circuit.
        """
        self.assign_params_value(pr, trainable=True)

    def assign_params_value(self, pr, trainable=False):
        """Set the parameters value.

        Args:
            pr (numpy.ndarray | Tensor | list | dict): Assign parameter(s) value of the circuit.
            trainable (bool): If true, set the paramters only for gate parameters that `requires_grad=True`
                , vice versa.
        """
        if pr is None:
            return
        if isinstance(pr, (np.ndarray, List, Tensor)):
            params = []
            for gate in self.gates:
                if isinstance(gate, WithParamGate) and (trainable == gate.trainable):
                    params.append(gate.param)
            assert len(pr) == len(params), f"Circuit have {len(params)} parameters, "\
                                f"but giving {len(pr)} parameters."
            for value, param in zip(pr, params):
                value = Tensor(value, DTYPE)
                param.set_data(value)

        elif isinstance(pr, dict):
            assert set(pr.keys()).issubset(set(self.params_name)),\
                "the `pr` names should be the subset of ansatz parameters' name."
            for gate in self.gates:
                if isinstance(gate, WithParamGate) and (trainable == gate.trainable):
                    name = gate.param_name
                    if name != DEFAULT_PARAM_NAME and name in pr.keys():
                        value = Tensor(pr[name], DTYPE)
                        gate.param.set_data(value)
        else:
            raise TypeError("`pr` 's type should be list, numpy.ndarray or dict.")

    def summary(self):
        """Print the basic information of circuit.
        """
        num_non_para_gate = 0
        num_para_gate = 0
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                num_para_gate += 1
            else:
                num_non_para_gate += 1
        info = bprint(
            [
                f'Total number of gates: {num_para_gate + num_non_para_gate}.',
                f'Parameter gates: {num_para_gate}.',
                f"with {len(self.params_name)} parameters are: ",
                f"{', '.join(self.params_name[:10])}{'.' if len(self.params_name) <= 10 else '...'}",
                f'Number qubit of circuit: {self.n_qubit}',
            ],
            title='Circuit Summary',
        )
        for i in info:
            print(i)

    def sampling(self, pr=None, shots=1000, display=False):
        """Sampling according the circuit state.

        Args:
            pr (None | numpy.ndarray | Tensor | list | dict): Assign parameter(s) value of the circuit.
                If None, just use the current state.
            shots (int): Number of sampling points.
        """
        qs = self.get_qs(pr)
        p = (qs.conj() * qs).real
        points = np.random.choice(2**self.n_qubit, shots, p=p)
        counter = collections.Counter(points)
        count = [counter[i] for i in range(2**self.n_qubit)]
        formatter = "{:0" + str(self.n_qubit) + "b}"
        xlabel = [formatter.format(i) for i in range(2**self.n_qubit)]
        if display:
            import matplotlib.pyplot as plt
            plt.bar(xlabel, count)
            plt.xticks(rotation=90)
            plt.show()
        return xlabel, count

    @property
    def parameters(self):
        param = {}
        for gate in self.gates:
            if isinstance(gate, WithParamGate) and\
                    gate.param_name != DEFAULT_PARAM_NAME:
                param[gate.param_name] = gate.param.reshape((1)).numpy()[0]
        return param

    def __add__(self, arg):
        return self._append(arg)


def UN(gate, obj_qubits: Tuple[int, List]):
    """Uniting several gates.
    
    Args:
        gate (BaseGate): Gate that will be set on each qubit.
        obj_qubits (int, list): The qubits that gates on.
    """
    if isinstance(obj_qubits, int):
        obj_qubits = list(range(obj_qubits))
    return Circuit([gate(obj) for obj in obj_qubits])
