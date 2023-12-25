# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Helper method to add noise channel."""
import typing
import warnings
from types import FunctionType, MethodType

from mindquantum.utils.type_value_check import (
    _check_input_type,
    _check_int_type,
    _check_value_should_not_less,
)

from .. import gates
from ..gates import BarrierGate, BasicGate, NoiseGate
from .circuit import Circuit


# pylint: disable=unused-argument,protected-access,too-few-public-methods
class ChannelAdderBase:
    """
    Add noise channel after or before quantum gate.

    This class is the base class for channel adder. In the derived class, you need to define the `_accepter`,
    `_excluder` and `_handler` method. The `_accepter` method is a set of accept rules that every gate
    you want to add channel should be accepted by those rules. The `_excluder` method is a set of deny
    rules that every gate you want to add channel should not be accepted by those rules. The `_handel`
    is the action you want to do when `_accepter` is accepted and `_excluder` is denied.

    Args:
        add_after (bool): Whether add channel after gate or before gate. Default: ``True``.
    """

    def __init__(self, add_after=True):
        """Initialize a ChannelAdderBase."""
        _check_input_type("add_after", bool, add_after)
        self.add_after = add_after
        self.accepter = []  # a list of function, which act as rules to accept considering gate to add noise channel.
        self.excluder = []  # a list of function, which act as rules to deny considering gate to add noise channel.
        self.accepter.extend(self._accepter())
        self.excluder.extend(self._excluder())

    def __call__(self, circ: Circuit) -> Circuit:
        """Add noise channel after acceptable quantum gate."""
        out = Circuit()
        for g in circ:
            if self.add_after:
                out += g
            if not isinstance(g, BarrierGate) and all(rule(g) for rule in self.accepter):
                if not any(rule(g) for rule in self.excluder):
                    out += self._handler(g)
            if not self.add_after:
                out += g
        return out

    def __repr__(self) -> str:
        """Return string expression of adder."""
        return f"{self.__class__.__name__}<>"

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rules."""
        return []

    def _excluder(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct excluder rules."""
        return []

    def _handler(self, g: BasicGate, *args, **kwargs) -> Circuit:
        """Create action you will do if a gate is acceptable."""
        return Circuit()


class ReverseAdder(ChannelAdderBase):
    """
    Reverse the accepter and excluder rules.

    Args:
        adder (:class:`~.core.circuit.ChannelAdderBase`): A channel adder.

    Examples:
        >>> from mindquantum.core.circuit import ReverseAdder, MeasureAccepter, BitFlipAdder, MixerAdder
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().rx('a', 0).measure_all()
        >>> only_measure = MixerAdder([BitFlipAdder(0.1), MeasureAccepter()])
        >>> only_measure(circ)
              ┏━━━━━━━┓ ┍━━━━━━┑ ╔═════════════╗
        q0: ──┨ RX(a) ┠─┤ M q0 ├─╢ BFC(p=1/10) ╟───
              ┗━━━━━━━┛ ┕━━━━━━┙ ╚═════════════╝
        >>> no_measure = ReverseAdder(only_measure)
        >>> no_measure(circ)
              ┏━━━━━━━┓ ╔═════════════╗ ┍━━━━━━┑
        q0: ──┨ RX(a) ┠─╢ BFC(p=1/10) ╟─┤ M q0 ├───
              ┗━━━━━━━┛ ╚═════════════╝ ┕━━━━━━┙
    """

    def __init__(self, adder: ChannelAdderBase):
        """Initialize a channel adder."""
        _check_input_type("adder", ChannelAdderBase, adder)
        self.adder = adder
        super().__init__(adder.add_after)

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rules."""
        return self.adder._excluder()

    def _excluder(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct excluder rules."""
        return self.adder._accepter()

    def __repr__(self) -> str:
        """Return string expression of adder."""
        strs = ["ReverseAdder<"]
        for i in self.adder.__repr__().split('\n'):
            strs.append("  " + i)
        strs.append(">")
        return '\n'.join(strs)

    def _handler(self, g: BasicGate, *args, **kwargs) -> Circuit:
        """Create action you will do if a gate is acceptable."""
        out = Circuit()
        out += self.adder._handler(g)
        return out


class MeasureAccepter(ChannelAdderBase):
    """
    Select measurement gate.

    Examples:
        >>> from mindquantum.core.circuit import MeasureAccepter, BitFlipAdder, MixerAdder
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().rx('a', 0).h(0).measure_all()
        >>> only_measure = MixerAdder([BitFlipAdder(0.1), MeasureAccepter()], add_after=False)
        >>> only_measure(circ)
              ┏━━━━━━━┓ ┏━━━┓ ╔═════════════╗ ┍━━━━━━┑
        q0: ──┨ RX(a) ┠─┨ H ┠─╢ BFC(p=1/10) ╟─┤ M q0 ├───
              ┗━━━━━━━┛ ┗━━━┛ ╚═════════════╝ ┕━━━━━━┙
    """

    def __init__(self):
        """Initialize a MeasureAccepter."""
        super().__init__()

    def _accepter(self, *args, **kwargs):
        """Construct accepter rules."""
        return [lambda x: isinstance(x, gates.Measure)]


class NoiseExcluder(ChannelAdderBase):
    """
    Exclude a noise channel.

    Args:
        add_after (bool): Whether add channel after gate or before gate. Default: ``True``.

    Examples:
        >>> from mindquantum.core.circuit import Circuit, NoiseExcluder, BitFlipAdder, MixerAdder
        >>> from mindquantum.core.gates import DepolarizingChannel
        >>> circ = Circuit().x(0)
        >>> circ += DepolarizingChannel(0.1).on(0)
        >>> circ
              ┏━━━┓ ╔════════════╗
        q0: ──┨╺╋╸┠─╢ DC(p=1/10) ╟───
              ┗━━━┛ ╚════════════╝
        >>> BitFlipAdder(0.1)(circ)
              ┏━━━┓ ╔═════════════╗ ╔════════════╗ ╔═════════════╗
        q0: ──┨╺╋╸┠─╢ BFC(p=1/10) ╟─╢ DC(p=1/10) ╟─╢ BFC(p=1/10) ╟───
              ┗━━━┛ ╚═════════════╝ ╚════════════╝ ╚═════════════╝
        >>> adder = MixerAdder([NoiseExcluder(), BitFlipAdder(0.1)])
        >>> adder(circ)
              ┏━━━┓ ╔═════════════╗ ╔════════════╗
        q0: ──┨╺╋╸┠─╢ BFC(p=1/10) ╟─╢ DC(p=1/10) ╟───
              ┗━━━┛ ╚═════════════╝ ╚════════════╝
    """

    def _excluder(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct excluder rules."""
        return [lambda x: isinstance(x, gates.NoiseGate)]


class BitFlipAdder(ChannelAdderBase):
    """
    Add BitFlip channel after or before quantum gate.

    Args:
        flip_rate (float): The flip rate for bit flip channel. For more detail please refers to
            :class:`~.core.gates.BitFlipChannel`.
        with_ctrl (bool): Whether add bit flip channel for control qubits. Default: ``True``.
        focus_on (int): Only add this noise channel on ``focus_on`` qubit. If ``None``, add to
            all qubits of selected quantum gate. Default: ``None``.
        add_after (bool): Whether add this channel after quantum gate or not. If ``False``, the
            channel will add before quantum gate. Default: ``True``.

    Examples:
        >>> from mindquantum.core.circuit import BitFlipAdder
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().h(0).x(1, 0)
        >>> adder1 = BitFlipAdder(0.1, with_ctrl=False)
        >>> adder1(circ)
              ┏━━━┓ ╔═════════════╗
        q0: ──┨ H ┠─╢ BFC(p=1/10) ╟───■─────────────────────
              ┗━━━┛ ╚═════════════╝   ┃
                                    ┏━┻━┓ ╔═════════════╗
        q1: ────────────────────────┨╺╋╸┠─╢ BFC(p=1/10) ╟───
                                    ┗━━━┛ ╚═════════════╝
        >>> adder2 = BitFlipAdder(0.1, with_ctrl=False, focus_on=1)
        >>> adder2(circ)
              ┏━━━┓
        q0: ──┨ H ┠───■─────────────────────
              ┗━━━┛   ┃
                    ┏━┻━┓ ╔═════════════╗
        q1: ────────┨╺╋╸┠─╢ BFC(p=1/10) ╟───
                    ┗━━━┛ ╚═════════════╝
    """

    def __init__(self, flip_rate: float, with_ctrl=True, focus_on: int = None, add_after: bool = True):
        """Initialize a BitFlipAdder."""
        _check_input_type("with_ctrl", bool, with_ctrl)
        super().__init__(add_after=add_after)
        self.flip_rate = flip_rate
        self.with_ctrl = with_ctrl
        if focus_on is not None:
            _check_int_type("focus_on", focus_on)
        self.focus_on = focus_on

    def __repr__(self) -> str:
        """Return string expression of adder."""
        return f"BitFlipAdder<flip_rate={self.flip_rate}, with_ctrl={self.with_ctrl}>"

    def _handler(self, g: BasicGate, *args, **kwargs) -> Circuit:
        """Create action you will do if a gate is acceptable."""
        circ = Circuit()
        if self.focus_on is None:
            for qubit in g.obj_qubits + (g.ctrl_qubits if self.with_ctrl else []):
                circ += gates.BitFlipChannel(self.flip_rate).on(qubit)
        else:
            if self.focus_on in g.obj_qubits or self.focus_on in g.ctrl_qubits:
                circ += gates.BitFlipChannel(self.flip_rate).on(self.focus_on)
        return circ


class NoiseChannelAdder(ChannelAdderBase):
    """
    Add single qubit quantum channel.

    Args:
        channel (:class:`~.core.gates.NoiseGate`): A single qubit quantum channel.
        with_ctrl (bool): Whether add quantum channel for control qubits. Default: ``True``.
        focus_on (int): Only add this noise channel on ``focus_on`` qubit. If ``None``, add to
            all qubits of selected quantum gate. Default: ``None``.
        add_after (bool): Whether add this channel after quantum gate or not. If ``False``, the
            channel will add before quantum gate. Default: ``True``.

    Examples:
        >>> from mindquantum.core.circuit import NoiseChannelAdder, Circuit
        >>> from mindquantum.core.gates import AmplitudeDampingChannel
        >>> circ = Circuit().h(0).x(1, 0)
        >>> channel = AmplitudeDampingChannel(0.3)
        >>> adder1 = NoiseChannelAdder(channel, with_ctrl=True, add_after=True)
        >>> adder1(circ)
              ┏━━━┓ ╔═════════════╗       ╔═════════════╗
        q0: ──┨ H ┠─╢ ADC(γ=3/10) ╟───■───╢ ADC(γ=3/10) ╟───
              ┗━━━┛ ╚═════════════╝   ┃   ╚═════════════╝
                                    ┏━┻━┓ ╔═════════════╗
        q1: ────────────────────────┨╺╋╸┠─╢ ADC(γ=3/10) ╟───
                                    ┗━━━┛ ╚═════════════╝
        >>> adder2 = NoiseChannelAdder(channel, with_ctrl=True, focus_on=1, add_after=True)
        >>> adder2(circ)
              ┏━━━┓
        q0: ──┨ H ┠───■─────────────────────
              ┗━━━┛   ┃
                    ┏━┻━┓ ╔═════════════╗
        q1: ────────┨╺╋╸┠─╢ ADC(γ=3/10) ╟───
                    ┗━━━┛ ╚═════════════╝
    """

    def __init__(self, channel: NoiseGate, with_ctrl=True, focus_on: int = None, add_after: bool = True):
        """Initialize a BitFlipAdder."""
        _check_input_type("channel", NoiseGate, channel)
        _check_input_type("with_ctrl", bool, with_ctrl)
        if channel.n_qubits != 1:
            raise ValueError(f"Requires a single qubit channel, but get {channel.n_qubits}, please customize a adder.")
        super().__init__(add_after=add_after)
        self.with_ctrl = with_ctrl
        self.channel = channel
        if focus_on is not None:
            _check_int_type("focus_on", focus_on)
        self.focus_on = focus_on

    def __repr__(self) -> str:
        """Return string expression of adder."""
        return f"NoiseChannelAdder<channel={self.channel}, with_ctrl={self.with_ctrl}>"

    def _handler(self, g: BasicGate, *args, **kwargs) -> Circuit:
        """Create action you will do if a gate is acceptable."""
        circ = Circuit()
        if self.focus_on is None:
            will_act = g.obj_qubits + (g.ctrl_qubits if self.with_ctrl else [])
            if isinstance(self.channel, gates.DepolarizingChannel) and len(will_act) > 1:
                warnings.warn(
                    "DepolarizingChannel will act on each qubit, but if you want a multi qubit "
                    "DepolarizingChannel, please use DepolarizingChannelAdder",
                    stacklevel=2,
                )
            for qubit in will_act:
                circ += self.channel.on(qubit)
        else:
            if self.focus_on in g.obj_qubits or self.focus_on in g.ctrl_qubits:
                circ += self.channel.on(self.focus_on)
        return circ


class DepolarizingChannelAdder(ChannelAdderBase):
    """
    Add DepolarizingChannel.

    Args:
        p (float): probability of occurred depolarizing error.
        n_qubits (int): the qubit number of depolarizing channel.
        add_after (bool): Whether add this channel after quantum gate or not. If ``False``, the
            channel will add before quantum gate. Default: ``True``.

    Examples:
        >>> from mindquantum.core.circuit import MixerAdder, DepolarizingChannelAdder, GateSelector
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().h(0).x(1, 0)
        >>> adder = MixerAdder([GateSelector('cx'), DepolarizingChannelAdder(0.1, 2)])
        >>> adder(circ)
              ┏━━━┓       ╔════════════╗
        q0: ──┨ H ┠───■───╢            ╟───
              ┗━━━┛   ┃   ║            ║
                    ┏━┻━┓ ║ DC(p=1/10) ║
        q1: ────────┨╺╋╸┠─╢            ╟───
                    ┗━━━┛ ╚════════════╝
    """

    def __init__(self, p: float, n_qubits: int, add_after: bool = True):
        """Initialize a DepolarizingChannelAdder."""
        _check_int_type("n_qubits", n_qubits)
        _check_value_should_not_less("n_qubits", 1, n_qubits)
        super().__init__(add_after=add_after)
        self.channel = gates.DepolarizingChannel(p, n_qubits)
        self.n_qubits = n_qubits

    def __repr__(self) -> str:
        """Return string expression of adder."""
        return f"DepolarizationChannelAdder<p={self.channel.p}, n_qubits={self.n_qubits}>"

    def _handler(self, g: BasicGate, *args, **kwargs) -> Circuit:
        """Create action you will do if a gate is acceptable."""
        circ = Circuit()
        if len(g.obj_qubits + g.ctrl_qubits) != self.n_qubits:
            raise ValueError(f"This DepolarizingChannel only work for {self.n_qubits} qubit gate, but" f" get {g}.")
        circ += self.channel.on(g.obj_qubits + g.ctrl_qubits)
        return circ


class QubitNumberConstrain(ChannelAdderBase):
    """
    Only add noise channel for ``n_qubits`` quantum gate.

    Args:
        n_qubits (int): The number qubit of quantum gate.
        with_ctrl (bool): Whether control qubits also contribute to `n_qubits` or not. Default: ``True``.
        add_after (bool): Whether add channel after gate or before gate. Default: ``True``.

    Examples:
        >>> from mindquantum.core.circuit import QubitNumberConstrain, Circuit, BitFlipAdder, MixerAdder
        >>> circ = Circuit().h(0).x(1, 0)
        >>> circ
              ┏━━━┓
        q0: ──┨ H ┠───■─────
              ┗━━━┛   ┃
                    ┏━┻━┓
        q1: ────────┨╺╋╸┠───
                    ┗━━━┛
        >>> adder = MixerAdder([
        ...     QubitNumberConstrain(2),
        ...     BitFlipAdder(0.1)
        ... ])
        >>> adder(circ)
              ┏━━━┓       ╔═════════════╗
        q0: ──┨ H ┠───■───╢ BFC(p=1/10) ╟───
              ┗━━━┛   ┃   ╚═════════════╝
                    ┏━┻━┓ ╔═════════════╗
        q1: ────────┨╺╋╸┠─╢ BFC(p=1/10) ╟───
                    ┗━━━┛ ╚═════════════╝
    """

    def __init__(self, n_qubits: int, with_ctrl: bool = True, add_after: bool = True):
        """Initialize a QubitNumberConstrain."""
        _check_int_type("n_qubits", n_qubits)
        _check_input_type("with_ctrl", bool, with_ctrl)
        self.n_qubits = n_qubits
        self.with_ctrl = with_ctrl
        super().__init__(add_after)

    def __repr__(self) -> str:
        """Return string expression of adder."""
        return f"QubitNumberConstrain<n_qubits={self.n_qubits}, with_ctrl={self.with_ctrl}>"

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rules."""
        return [lambda x: self.n_qubits == (len(x.obj_qubits) + len(x.ctrl_qubits) * self.with_ctrl)]


class QubitIDConstrain(ChannelAdderBase):
    """
    Select gate with qubit id in given list.

    Args:
        qubit_ids (Union[int, List[int]]): The qubit id list you want to select.
        add_after (bool): Whether add channel after gate or before gate. Default: ``True``.

    Examples:
        >>> from mindquantum.core.circuit import MixerAdder, BitFlipAdder, QubitIDConstrain, Circuit
        >>> circ = Circuit().h(0).h(1).h(2).x(1, 0).x(2, 1)
        >>> circ
              ┏━━━┓
        q0: ──┨ H ┠───■───────────
              ┗━━━┛   ┃
              ┏━━━┓ ┏━┻━┓
        q1: ──┨ H ┠─┨╺╋╸┠───■─────
              ┗━━━┛ ┗━━━┛   ┃
              ┏━━━┓       ┏━┻━┓
        q2: ──┨ H ┠───────┨╺╋╸┠───
              ┗━━━┛       ┗━━━┛
        >>> adder = MixerAdder([
        ...     QubitIDConstrain([0, 1]),
        ...     BitFlipAdder(0.1),
        ... ])
        >>> adder(circ)
              ┏━━━┓ ╔═════════════╗       ╔═════════════╗
        q0: ──┨ H ┠─╢ BFC(p=1/10) ╟───■───╢ BFC(p=1/10) ╟─────────
              ┗━━━┛ ╚═════════════╝   ┃   ╚═════════════╝
              ┏━━━┓ ╔═════════════╗ ┏━┻━┓ ╔═════════════╗
        q1: ──┨ H ┠─╢ BFC(p=1/10) ╟─┨╺╋╸┠─╢ BFC(p=1/10) ╟───■─────
              ┗━━━┛ ╚═════════════╝ ┗━━━┛ ╚═════════════╝   ┃
              ┏━━━┓                                       ┏━┻━┓
        q2: ──┨ H ┠───────────────────────────────────────┨╺╋╸┠───
              ┗━━━┛                                       ┗━━━┛
    """

    def __init__(self, qubit_ids: typing.Union[int, typing.List[int]], add_after: bool = True):
        """Initialize a QubitIDConstrain."""
        self.qubit_ids = []
        if isinstance(qubit_ids, int):
            self.qubit_ids.append(qubit_ids)
        elif isinstance(qubit_ids, list):
            for qubit_id in qubit_ids:
                _check_int_type("Element of qubit_ids", qubit_id)
            self.qubit_ids.extend(qubit_ids)
        else:
            raise TypeError(f"qubit_ids requires a int or a list, but get {type(qubit_ids)}.")
        super().__init__(add_after)

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rules."""
        return [lambda x: all(i in self.qubit_ids for i in x.obj_qubits + x.ctrl_qubits)]


class MixerAdder(ChannelAdderBase):
    """
    Execute each adder if all accepter and excluder are met.

    Args:
        adders (List[:class:`~.core.gates.BitFlipChannel`]): The adders you want to mix.
        add_after (bool): Whether add channel after quantum gate or not. If ``False``, the
            channel will add before quantum gate. This `add_after` will override all`add_after`
            of sub adder. Default: ``True``.

    Examples:
        >>> from mindquantum.core.circuit import MeasureAccepter, BitFlipAdder, MixerAdder
        >>> from mindquantum.core.circuit import Circuit
        >>> circ = Circuit().rx('a', 0).h(0).measure_all()
        >>> only_measure = MixerAdder([BitFlipAdder(0.1), MeasureAccepter()], add_after=False)
        >>> only_measure(circ)
              ┏━━━━━━━┓ ┏━━━┓ ╔═════════════╗ ┍━━━━━━┑
        q0: ──┨ RX(a) ┠─┨ H ┠─╢ BFC(p=1/10) ╟─┤ M q0 ├───
              ┗━━━━━━━┛ ┗━━━┛ ╚═════════════╝ ┕━━━━━━┙
    """

    def __init__(self, adders: typing.List[ChannelAdderBase], add_after=True):
        """Initialize a MixerAdder."""
        _check_input_type("adders", list, adders)
        for adder in adders:
            _check_input_type("Element of adders", ChannelAdderBase, adder)
        self.adders = adders
        super().__init__(add_after=add_after)

    def __repr__(self) -> str:
        """Return string expression of adder."""
        strs = ["MixerAdder<"]
        for adder in self.adders:
            for i in adder.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append(">")
        return '\n'.join(strs)

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rules."""
        return [item for adder in self.adders for item in adder._accepter()]

    def _excluder(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct excluder rules."""
        return [item for adder in self.adders for item in adder._excluder()]

    def _handler(self, g: BasicGate, *args, **kwargs) -> Circuit:
        """Create action you will do if a gate is acceptable."""
        out = Circuit()
        for adder in self.adders:
            out += adder._handler(g)
        return out


class SequentialAdder(ChannelAdderBase):
    """
    Execute each adder in this sequential.

    Args:
        adders (List[:class:`~.core.circuit.ChannelAdderBase`]): The adder you want to apply.

    Examples:
        >>> from mindquantum.core.circuit import SequentialAdder, MixerAdder, BitFlipAdder, NoiseChannelAdder
        >>> from mindquantum.core.circuit import MeasureAccepter, ReverseAdder, NoiseChannelAdder, Circuit
        >>> from mindquantum.core.circuit import NoiseExcluder
        >>> from mindquantum.core.gates import DepolarizingChannel
        >>> circ = Circuit().h(0).x(1, 0).measure_all()
        >>> circ
              ┏━━━┓       ┍━━━━━━┑
        q0: ──┨ H ┠───■───┤ M q0 ├───
              ┗━━━┛   ┃   ┕━━━━━━┙
                    ┏━┻━┓ ┍━━━━━━┑
        q1: ────────┨╺╋╸┠─┤ M q1 ├───
                    ┗━━━┛ ┕━━━━━━┙
        >>> bitflip_error_for_measure = MixerAdder([
        ...     BitFlipAdder(0.1),
        ...     MeasureAccepter(),
        ...     NoiseExcluder()
        ... ], add_after=False)
        >>> depolarizing_for_gate = MixerAdder([
        ...     NoiseChannelAdder(DepolarizingChannel(0.1)),
        ...     ReverseAdder(MeasureAccepter()),
        ...     NoiseExcluder()
        ... ])
        >>> adder = SequentialAdder([
        ...     bitflip_error_for_measure,
        ...     depolarizing_for_gate,
        ... ])
        >>> adder(circ)
              ┏━━━┓ ╔════════════╗       ╔════════════╗ ╔═════════════╗ ┍━━━━━━┑
        q0: ──┨ H ┠─╢ DC(p=1/10) ╟───■───╢ DC(p=1/10) ╟─╢ BFC(p=1/10) ╟─┤ M q0 ├───
              ┗━━━┛ ╚════════════╝   ┃   ╚════════════╝ ╚═════════════╝ ┕━━━━━━┙
                                   ┏━┻━┓ ╔════════════╗ ╔═════════════╗ ┍━━━━━━┑
        q1: ───────────────────────┨╺╋╸┠─╢ DC(p=1/10) ╟─╢ BFC(p=1/10) ╟─┤ M q1 ├───
                                   ┗━━━┛ ╚════════════╝ ╚═════════════╝ ┕━━━━━━┙
    """

    def __init__(self, adders: typing.List[ChannelAdderBase]):
        """Initialize a SequentialAdder."""
        _check_input_type("adders", list, adders)
        for adder in adders:
            _check_input_type("Element of adders", ChannelAdderBase, adder)
        super().__init__()
        self.adders = adders

    def __call__(self, circ: Circuit) -> Circuit:
        """Add noise channel after acceptable quantum gate."""
        for adder in self.adders:
            circ = adder(circ)
        return circ

    def __repr__(self) -> str:
        """Return string expression of adder."""
        strs = ["SequentialAdder<"]
        for adder in self.adders:
            for i in adder.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append(">")
        return '\n'.join(strs)


class GateSelector(ChannelAdderBase):
    """
    Select gate to add noise channel.

    Args:
        gate (str): Gate you want to add channel. Could be one of 'H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ',
            'CX', 'CZ', 'SWAP'.

    Examples:
        >>> from mindquantum.core.circuit import BitFlipAdder, GateSelector, Circuit, MixerAdder
        >>> circ = Circuit().h(0).x(1, 0)
        >>> circ
              ┏━━━┓
        q0: ──┨ H ┠───■─────
              ┗━━━┛   ┃
                    ┏━┻━┓
        q1: ────────┨╺╋╸┠───
                    ┗━━━┛
        >>> adder = MixerAdder([BitFlipAdder(0.1), GateSelector('cx')])
        >>> adder(circ)
              ┏━━━┓       ╔═════════════╗
        q0: ──┨ H ┠───■───╢ BFC(p=1/10) ╟───
              ┗━━━┛   ┃   ╚═════════════╝
                    ┏━┻━┓ ╔═════════════╗
        q1: ────────┨╺╋╸┠─╢ BFC(p=1/10) ╟───
                    ┗━━━┛ ╚═════════════╝
    """

    _single_qubit_gate = ['H', 'X', 'Y', 'Z', 'RX', 'RY', 'RZ']
    _double_qubits_gate = ['CX', 'CZ', 'SWAP']

    def __init__(self, gate: str):
        """Initialize a gate selector."""
        if gate.upper() not in self.supported_gate:
            raise ValueError(f"Now we support {self.supported_gate}, but get {gate}")
        self.gate = gate.upper()
        super().__init__()

    @property
    def supported_gate(self):
        """Get supported gate for gate selector."""
        return GateSelector._single_qubit_gate + GateSelector._double_qubits_gate

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rule."""
        special = {
            'CX': lambda x: isinstance(x, (gates.CNOTGate, gates.XGate)) and len(x.obj_qubits + x.ctrl_qubits) == 2,
            'CZ': lambda x: isinstance(x, gates.ZGate) and len(x.obj_qubits + x.ctrl_qubits) == 2,
        }
        if self.gate in special:
            return [special[self.gate]]
        suffix = ''
        if self.gate not in ['RX', 'RY', 'RZ']:
            suffix = 'Gate'
        return [lambda x: isinstance(x, getattr(gates, self.gate + suffix)) and not x.ctrl_qubits]

    def __repr__(self) -> str:
        """Return string expression of adder."""
        return f"GateSelector<gate={self.gate}>"


__all__ = [
    "ChannelAdderBase",
    "NoiseChannelAdder",
    "MeasureAccepter",
    "ReverseAdder",
    "NoiseExcluder",
    "BitFlipAdder",
    "MixerAdder",
    "SequentialAdder",
    "QubitNumberConstrain",
    "QubitIDConstrain",
    "GateSelector",
    "DepolarizingChannelAdder",
]
__all__.sort()
