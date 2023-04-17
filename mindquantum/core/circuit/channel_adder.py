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

from mindquantum.core import gates as G
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import BasicGate
from mindquantum.device.chip import NaiveChip


class ChannelAdderBase:
    def __init__(self, *args, **kwargs):
        self.accepter = []  # a list of function, which act as rules to accept considering gate to add noise channel.
        self.excluder = []  # a list of function, which act as rules to deny considering gate to add noise channel.
        self._accepter(self.accepter)
        self._excluder(self.excluder)

    def _accepter(self, accepter_list, *args, **kwargs):
        """Construct accepter rules."""

    def _excluder(self, excluder_list, *args, **kwargs):
        """Construct excluder rules."""

    def _handler(self, g: BasicGate, *args, **kwargs):
        """What you will do if a gate is acceptable."""
        return Circuit()

    def __call__(self, circ: Circuit) -> Circuit:
        out = Circuit()
        for g in circ:
            out += g
            if all(rule(g) for rule in self.accepter):
                if not any(rule(g) for rule in self.excluder):
                    out += self._handler(g)
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}<>"


class MeasureChecker(ChannelAdderBase):
    def _accepter(self, accepter_list):
        accepter_list.append(lambda x: isinstance(x, G.Measure))


class NoiseExcluder(ChannelAdderBase):
    def _excluder(self, excluder_list):
        excluder_list.append(lambda x: isinstance(x, G.NoiseGate))


class BitFlipAdder(ChannelAdderBase):
    def __init__(self, flip_rate: float = None, with_ctrl=True, device: NaiveChip = None):
        super().__init__()
        self.with_ctrl = True
        self.flip_rate = flip_rate
        self.device = device
        self.with_ctrl = with_ctrl

    def __repr__(self):
        if self.device is None:
            return f"BitFlipAdder<flip_rate={self.flip_rate}, with_ctrl={self.with_ctrl}>"
        return f"BitFlipAdder<device={self.device}, with_ctrl={self.with_ctrl}>"

    def _handler(self, g: BasicGate):
        circ = Circuit()
        for q in g.obj_qubits + (g.ctrl_qubits if self.with_ctrl else []):
            if self.device is not None:
                circ += self.device.gene_channel(
                    self.device, g, G.BitFlipChannel, self.with_ctrl, G.BitFlipChannel(self.flip_rate)
                ).on(q)
            else:
                circ += G.BitFlipChannel(self.flip_rate).on(q)
        return circ


class MixerAdder(ChannelAdderBase):
    def __init__(self, adders: typing.List[ChannelAdderBase]):
        self.adders = adders
        super().__init__()

    def _accepter(self, accepter_list, *args, **kwargs):
        for adder in self.adders:
            adder._accepter(accepter_list)

    def _excluder(self, excluder_list, *args, **kwargs):
        for adder in self.adders:
            adder._excluder(excluder_list)

    def _handler(self, g: BasicGate, *args, **kwargs):
        out = Circuit()
        for adder in self.adders:
            out += adder._handler(g)
        return out

    def __repr__(self):
        strs = ["MixerAdder<"]
        for adder in self.adders:
            for i in adder.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append(">")
        return '\n'.join(strs)


class SequentialAdder(ChannelAdderBase):
    def __init__(self, adders: typing.List[ChannelAdderBase]):
        super().__init__()
        self.adders = adders

    def __call__(self, circ: Circuit):
        for adder in self.adders:
            circ = adder(circ)
        return circ

    def __repr__(self):
        strs = ["SequentialAdder<"]
        for adder in self.adders:
            for i in adder.__repr__().split('\n'):
                strs.append("  " + i)
        strs.append(">")
        return '\n'.join(strs)


if __name__ == '__main__':
    from mindquantum import H, Measure

    bit_flip_adder = BitFlipAdder(0.1)
    mea_checker = MeasureChecker()
    mix_adder = MixerAdder(
        [
            bit_flip_adder,
            mea_checker,
        ]
    )

    seq_adder = SequentialAdder(
        [
            mix_adder,
            MixerAdder(
                [
                    BitFlipAdder(0.3),
                    NoiseExcluder(),
                ]
            ),
        ]
    )

    circ = Circuit() + H.on(0) + Measure().on(0)  # q0: ──H────M(q0)──
    circ1 = bit_flip_adder(circ)  # q0: ──H────BF(0.1)────M(q0)────BF(0.1)──
    circ2 = mea_checker(circ)  # q0: ──H────M(q0)──
    circ3 = mix_adder(circ)  # q0: ──H────M(q0)────BF(0.1)──
    circ4 = seq_adder(circ)  # q0: ──H────BF(0.3)────M(q0)────BF(0.3)────BF(0.1)──
    print(circ)
    print(circ4)
