"""Quantum circuit for qudit."""

from typing import List, Tuple, Union, Iterable, Optional, Dict
import collections

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from quditop.global_var import DTYPE
from quditop.utils import bprint, str_ket
from quditop.gates import GateBase, WithParamGate
from quditop.common import get_complex_tuple


class Circuit(nn.Cell):
    """A list like container that contains qudit gates."""

    def __init__(self,
                 dim: int,
                 n_qudits: int,
                 gates: Optional[Iterable[GateBase]] = None) -> None:
        """Initialize.

        Args:
            dim: The dimension of qudits.
            n_qudits: The number of qudits this circuit contains.
            gates: The initialized quantum gates that on circuit.
        """
        super().__init__()
        self.dim = dim
        self.n_qudits = n_qudits
        self.param_name = []
        self.gates = nn.CellList()  # Quantum gates
        self.qs = None  # Quantum state
        self._set_initial_state()

        if gates is not None:
            for gate in gates:
                if gate.dim != self.dim:
                    raise ValueError(
                        f"The input gate.dim({gate.dim}) doesn't match the circuit dim({self.dim})."
                    )
                if max(gate.obj_qudits) >= self.n_qudits:
                    raise ValueError(
                        f"Gate {gate.name}.obj_qudits = {gate.obj_qudits} should less than {self.n_qudits}."
                    )
                if gate.ctrl_qudits and max(gate.ctrl_qudits) >= self.n_qudits:
                    raise ValueError(
                        f"Gate {gate.name}.obj_qudits = {gate.ctrl_qudits} should less than {self.n_qudits}."
                    )
                self.gates.append(gate)
                if isinstance(gate, WithParamGate):
                    self.param_name.append(gate.param_name)

    def append(
            self, arg: Union[GateBase, Iterable[GateBase],
                             "Circuit"]) -> "Circuit":
        """Add a gate, a series of gate or another circuit to this circuit inplace."""
        if isinstance(arg, GateBase):
            self.gates.append(arg)
        elif isinstance(arg, Iterable):
            self.gates.extend(arg)
        elif isinstance(arg, Circuit):
            self.gates.extend([g for g in arg.gates])
        else:
            raise TypeError(
                f"The input type should be GateBase or Iterable, but got {type(self.gates)}."
            )
        self.param_name.clear()
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                self.param_name.append(gate.param_name)

    def __iadd__(
            self, arg: Union[GateBase, Iterable[GateBase],
                             "Circuit"]) -> "Circuit":
        self.append(arg)
        return self

    def __add__(
            self, arg: Union[GateBase, Iterable[GateBase],
                             "Circuit"]) -> "Circuit":
        new_gates = []
        for gate in self.gates:
            new_gates.append(gate)
        if isinstance(arg, GateBase):
            new_gates.append(arg)
        elif isinstance(arg, Iterable):
            new_gates.extend(arg)
        elif isinstance(arg, Circuit):
            new_gates.extend([g for g in arg.gates])
        else:
            raise TypeError(
                f"The input type should be GateBase or Iterable, but got {type(self.gates)}."
            )
        self.gates = nn.CellList(new_gates)
        self.param_name.clear()
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                self.param_name.append(gate.param_name)
        return self

    def construct(self, qs=None):
        """This function will get the last state of circuit after gates apply on initial state."""
        if qs is None:
            qs = self.qs
        for gate in self.gates:
            qs = gate(qs)
        return qs

    def _set_initial_state(self):
        """Set the default initial state of the circuit."""
        re = ops.zeros(self.dim**self.n_qudits, dtype=DTYPE)
        re[0] = 1
        im = ops.zeros(self.dim**self.n_qudits, dtype=DTYPE)
        self.set_init_qs((re, im))

    def reset(self):
        """Reset the initial state as the default one."""
        self.qs = self._get_initial_state()

    def _detach_flatten_merge_qs(self,
                                 qs: Tuple[Tensor],
                                 endian_reverse=False):
        """A tool function that convert the representation format of quantum state."""
        re, im = qs[0], qs[1]

        if endian_reverse:
            indices = list(range(self.n_qudits))[::-1]
            re = re.permute(*indices)
            im = im.permute(*indices)
        return re.numpy().flatten() + 1j * im.numpy().flatten()

    def _assign_parameters(self, pr, trainable=False):
        """Assign parameter to circuit.

        Args:
            pr: The input parameters.
            trainable: If true, only assign to ansatz, otherwise only to encoder.
        """
        if pr is None:
            return
        param_gates = []
        if isinstance(pr, (np.ndarray, List, Tensor)):
            for gate in self.gates:
                if isinstance(gate, WithParamGate) and (trainable
                                                        == gate.trainable):
                    param_gates.append(gate)
            assert len(pr) == len(param_gates), (
                f"Circuit have {len(param_gates)} parameters, "
                f"but giving {len(pr)} parameters.")
            for value, gate in zip(pr, param_gates):
                gate.assign_param(value)

        elif isinstance(pr, dict):
            assert set(pr.keys()).issubset(
                set(self.param_name)
            ), f"The circuit parameters are {self.param_name}, while got {pr.keys()}."
            for gate in self.gates:
                if isinstance(gate, WithParamGate) and (trainable
                                                        == gate.trainable):
                    name = gate.param_name
                    if name in pr.keys():
                        gate.assign_param(pr[name])
        else:
            raise TypeError(
                "`pr` 's type should be list, numpy.ndarray or dict.")

    def assign_encoder_parameters(self, pr):
        """Assign parameters to encoder gates, which are non-trainable parameterized gates."""
        self._assign_parameters(pr, trainable=False)

    def assign_ansatz_parameters(self, pr):
        """Assign parameters to ansatz gates, which are trainable parameterized gates."""
        self._assign_parameters(pr, trainable=True)

    def get_qs_tuple(self):
        """Get the Tuple format of quantum state.
        Note: This function don't use `detach()`, so the gradient can be traced.
        """
        return self.construct(self.qs)

    def get_qs(self, pr=None, ket: bool = False, endian_reverse=False):
        """Get quantum state.

        Args:
            pr: The given parameters. If None, use the current value of parameters. Otherwise assign
                the value in `pr` to the parameters in circuit.
            ket: The output format is ket string (ket=True) or numpy.ndarray (ket=False).
        """
        self.assign_encoder_parameters(pr)
        self.assign_ansatz_parameters(pr)
        qs = self._detach_flatten_merge_qs(self.construct(self.qs),
                                           endian_reverse)
        if ket:
            return str_ket(self.dim, qs)
        return qs

    def set_init_qs(self, qs):
        """Set the initial state of circuit, initial state means the state that on the most left state of circuit.
        Note: This function will not check if the state is a quantum state (such as if norm is 1).
        """
        shape = (self.dim, ) * self.n_qudits
        self.qs = get_complex_tuple(qs, shape)

    def no_grad_(self):
        """Stop calculating gradient for all the parameterized gates, it's usually used as encoder.
        This operation is inplace.
        """
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                gate.no_grad_()

    def with_grad_(self):
        """Calculating gradient for all the parameterized gates, it's usually used as ansatz.
        This operation is inplace."""
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                gate.with_grad_()

    def as_encoder(self):
        """Set the circuit as encoder, which means stopping calculating gradient for parameters. This operation is
        inplace."""
        self.no_grad_()
        return self

    def as_ansatz(self):
        """Set the circuit as ansatz, which means it will calculate gradient for parameters. This operation is inplace.
        """
        self.with_grad_()
        return self

    def qs_probability_distribution(self, endian_reverse=True) -> Dict:
        """Get the probability of each quantum state."""
        qs = self.get_qs(endian_reverse)
        p = (qs.conj() * qs).real
        p /= p.sum()
        state_str = [
            np.base_repr(ind, self.dim).zfill(self.n_qudits)
            for ind in range(self.dim**self.n_qudits)
        ]
        return dict(zip(state_str, p))

    def sampling(self, shots: int = 1000, endian_reverse=True) -> None:
        """Measure the circuit `shots` times and calculate the result.

        Args:
            shots: The number of sampling.
            endian_reverse: If show the result in reversed endian. Since the endian is opposite to some other quantum
                frameworks(such as MindQuantum), set `endian_reverse=True` will return consistent result.
        """
        p = self.qs_probability_distribution(endian_reverse)
        points = np.random.choice(self.dim**self.n_qudits, shots, p=p)
        counter = collections.Counter(points)
        count = [counter[i] for i in range(self.dim**self.n_qudits)]
        state_str = [
            np.base_repr(ind, self.dim).zfill(self.n_qudits)
            for ind in range(self.dim**self.n_qudits)
        ]
        return dict(zip(state_str, count))

    def summary(self):
        """Print the basic information of circuit."""
        num_non_para_gate = 0
        num_para_gate = 0
        for gate in self.gates:
            if isinstance(gate, WithParamGate):
                num_para_gate += 1
            else:
                num_non_para_gate += 1
        info = bprint(
            [
                f"Total number of gates: {num_para_gate + num_non_para_gate}.",
                f"Parameter gates: {num_para_gate}.",
                f"with {len(self.param_name)} parameters are: ",
                f"{', '.join(self.param_name[:10])}{'.' if len(self.param_name) <= 10 else '...'}",
                f"Number qudits of circuit: {self.n_qudits}",
            ],
            title="Circuit Summary",
        )
        for i in info:
            print(i)


__all__ = ["Circuit"]
