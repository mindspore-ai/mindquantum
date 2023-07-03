# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http: //www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Helper method to add noise channel."""
import typing
from types import FunctionType, MethodType

from .. import gates
from ..gates import BasicGate
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
            if all(rule(g) for rule in self.accepter):
                if not any(rule(g) for rule in self.excluder):
                    out += self._handler(g)
            if not self.add_after:
                out += g
        return out

    def __repr__(self):
        """Return string expression of adder."""
        return f"{self.__class__.__name__}<>"

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rules."""
        return []

    def _excluder(self, *args, **kwargs):
        """Construct excluder rules."""
        return []

    def _handler(self, g: BasicGate, *args, **kwargs):
        """Create action you will do if a gate is acceptable."""
        return Circuit()


class ReverseAdder(ChannelAdderBase):
    """
    Reverse the accepter and excluder rules.

    Args:
        adder (:class:`~.core.circuit.ChannelAdderBase`): A channel adder.
    """

    def __init__(self, adder: ChannelAdderBase):
        """Initialize a channel adder."""
        super().__init__(adder.add_after)
        self.adder = adder

    def _accepter(self, *args, **kwargs) -> typing.List[typing.Union[FunctionType, MethodType]]:
        """Construct accepter rules."""
        return self.adder._excluder()

    def _excluder(self, *args, **kwargs):
        """Construct excluder rules."""
        return self.adder._accepter()


class MeasureAccepter(ChannelAdderBase):
    """Select measurement gate."""

    def __init__(self):
        """Initialize a MeasureAccepter."""
        super().__init__()

    def _accepter(self, *args, **kwargs):
        """Construct accepter rules."""
        return [lambda x: isinstance(x, gates.Measure)]


class NoiseExcluder(ChannelAdderBase):
    """Exclude a noise channel."""

    def _excluder(self, *args, **kwargs):
        """Construct excluder rules."""
        return [lambda x: isinstance(x, gates.NoiseGate)]


class BitFlipAdder(ChannelAdderBase):
    """
    Add BitFlip channel after or before quantum gate.

    Args:
        flip_rate (float): The flip rate for bit flip channel. For more detail please refers to
            :class:`~.core.circuit.BitFlipChannel`.
        with_ctrl (bool): Whether add bit flip channel for control qubits. Default: ``True``.
        add_after (bool): Whether add this channel after quantum gate or not. If ``False``, the
            channel will add before quantum gate. Default: ``True``.
    """

    def __init__(self, flip_rate: float, with_ctrl=True, add_after: bool = True):
        """Initialize a BitFlipAdder."""
        super().__init__(add_after=add_after)
        self.with_ctrl = True
        self.flip_rate = flip_rate
        self.with_ctrl = with_ctrl

    def __repr__(self):
        """Return string expression of adder."""
        return f"BitFlipAdder<flip_rate={self.flip_rate}, with_ctrl={self.with_ctrl}>"

    def _handler(self, g: BasicGate, *args, **kwargs):
        """Create action you will do if a gate is acceptable."""
        circ = Circuit()
        for qubit in g.obj_qubits + (g.ctrl_qubits if self.with_ctrl else []):
            circ += gates.BitFlipChannel(self.flip_rate).on(qubit)
        return circ


class MixerAdder(ChannelAdderBase):
    """
    Execute each adder if all accepter and excluder are met.

    Args:
        adders (List[:class:`~.core.circuit.BitFlipChannel`]): The adders you want to mix.
        add_after (bool): Whether add channel after quantum gate or not. If ``False``, the
            channel will add before quantum gate. Default: ``True``.
    """

    def __init__(self, adders: typing.List[ChannelAdderBase], add_after=True):
        """Initialize a MixerAdder."""
        self.adders = adders
        super().__init__(add_after=add_after)

    def __repr__(self):
        """Return string expression of adder."""
        strs = ["MixerAdder<"]
        for adder in self.adders:
            for i in adder.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append(">")
        return '\n'.join(strs)

    def _accepter(self, *args, **kwargs):
        """Construct accepter rules."""
        return [item for adder in self.adders for item in adder._accepter()]

    def _excluder(self, *args, **kwargs):
        """Construct excluder rules."""
        return [item for adder in self.adders for item in adder._excluder()]

    def _handler(self, g: BasicGate, *args, **kwargs):
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
    """

    def __init__(self, adders: typing.List[ChannelAdderBase]):
        """Initialize a SequentialAdder."""
        super().__init__()
        self.adders = adders

    def __call__(self, circ: Circuit):
        """Add noise channel after acceptable quantum gate."""
        for adder in self.adders:
            circ = adder(circ)
        return circ

    def __repr__(self):
        """Return string expression of adder."""
        strs = ["SequentialAdder<"]
        for adder in self.adders:
            for i in adder.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append(">")
        return '\n'.join(strs)


__all__ = [
    "ChannelAdderBase",
    "MeasureAccepter",
    "ReverseAdder",
    "NoiseExcluder",
    "BitFlipAdder",
    "MixerAdder",
    "SequentialAdder",
]
