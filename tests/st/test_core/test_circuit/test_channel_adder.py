# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test channel adder."""

from mindquantum.core.circuit import (
    BitFlipAdder,
    Circuit,
    MeasureAccepter,
    MixerAdder,
    NoiseChannelAdder,
    NoiseExcluder,
    QubitIDConstrain,
    QubitNumberConstrain,
    ReverseAdder,
    SequentialAdder,
)
from mindquantum.core.gates import (
    AmplitudeDampingChannel,
    BitFlipChannel,
    DepolarizingChannel,
    H,
    Measure,
    X,
)


def test_reverse_adder():
    """
    Description: test reverse adder.
    Expectation: success.
    """
    circ = Circuit().rx('a', 0).measure_all()
    only_measure = MixerAdder([BitFlipAdder(0.1), MeasureAccepter()])
    no_measure = ReverseAdder(only_measure)
    new_circ = no_measure(circ)
    exp_circ = (Circuit().rx('a', 0) + BitFlipChannel(0.1).on(0)).measure_all()
    assert new_circ == exp_circ


def test_measure_accepter():
    """
    Description: test measure accepter.
    Expectation: success.
    """
    circ = Circuit().rx('a', 0).h(0).measure_all()
    only_measure = MixerAdder([BitFlipAdder(0.1), MeasureAccepter()], add_after=False)
    new_circ = only_measure(circ)
    exp_circ = (Circuit().rx('a', 0).h(0) + BitFlipChannel(0.1).on(0)).measure_all()
    assert new_circ == exp_circ


def test_noise_excluder():
    """
    Description: test noise excluder.
    Expectation: success.
    """
    circ = Circuit().x(0)
    circ += DepolarizingChannel(0.1).on(0)
    adder = MixerAdder([NoiseExcluder(), BitFlipAdder(0.1)])
    new_circ = adder(circ)
    exp_circ = Circuit().x(0) + BitFlipChannel(0.1).on(0) + DepolarizingChannel(0.1).on(0)
    assert new_circ == exp_circ


def test_bit_flip_adder():
    """
    Description: test bit flip adder.
    Expectation: success.
    """
    circ = Circuit().h(0).x(1, 0)
    adder1 = BitFlipAdder(0.1, with_ctrl=False)
    channel = BitFlipChannel(0.1)
    new_circ = adder1(circ)
    exp_circ = Circuit().h(0) + channel.on(0) + X.on(1, 0) + channel.on(1)
    assert new_circ == exp_circ
    adder2 = BitFlipAdder(0.1, with_ctrl=False, focus_on=1)
    new_circ2 = adder2(circ)
    exp_circ2 = circ + channel.on(1)
    assert new_circ2 == exp_circ2


def test_noise_channel_adder():
    """
    Description: test noise channel adder.
    Expectation: success.
    """
    circ = Circuit().h(0).x(1, 0)
    channel = AmplitudeDampingChannel(0.3)
    adder1 = NoiseChannelAdder(channel, with_ctrl=True, add_after=True)
    new_circ = adder1(circ)
    exp_circ = Circuit().h(0) + channel.on(0) + X.on(1, 0) + channel.on(1) + channel.on(0)
    assert new_circ == exp_circ
    adder2 = NoiseChannelAdder(channel, with_ctrl=True, focus_on=1, add_after=True)
    new_circ2 = adder2(circ)
    exp_circ2 = circ + channel.on(1)
    assert new_circ2 == exp_circ2


def test_mixer_adder():
    """
    Description: test mixer adder.
    Expectation: success.
    """
    circ = Circuit().rx('a', 0).h(0).measure_all()
    only_measure = MixerAdder([BitFlipAdder(0.1), MeasureAccepter()], add_after=False)
    new_circ = only_measure(circ)
    exp_circ = Circuit().rx('a', 0).h(0) + BitFlipChannel(0.1).on(0)
    exp_circ.measure_all()
    assert new_circ == exp_circ


def test_sequential_adder():
    """
    Description: test sequential adder.
    Expectation: success.
    """
    circ = Circuit().h(0).x(1, 0).measure_all()
    bitflip_error_for_measure = MixerAdder([BitFlipAdder(0.1), MeasureAccepter(), NoiseExcluder()], add_after=False)
    depolarizing_for_gate = MixerAdder(
        [NoiseChannelAdder(BitFlipChannel(0.2)), ReverseAdder(MeasureAccepter()), NoiseExcluder()]
    )
    adder = SequentialAdder(
        [
            bitflip_error_for_measure,
            depolarizing_for_gate,
        ]
    )
    new_circ = adder(circ)
    noise_1 = BitFlipChannel(0.2)
    bfc = BitFlipChannel(0.1)
    exp_circ = Circuit().h(0) + noise_1.on(0) + X.on(1, 0) + noise_1.on(1)
    exp_circ += Circuit([noise_1.on(0), bfc.on(0), Measure().on(0), bfc.on(1)])
    exp_circ += Measure().on(1)
    assert exp_circ == new_circ


def test_qubit_number_constrain():
    """
    Description: test qubit number constrain.
    Expectation: success.
    """
    circ = Circuit().h(0).x(1, 0)
    adder = MixerAdder([QubitNumberConstrain(2), BitFlipAdder(0.1)])
    new_circ = adder(circ)
    bit_flip = BitFlipChannel(0.1)
    exp_circ = circ + bit_flip.on(1) + bit_flip.on(0)
    assert new_circ == exp_circ


def test_qubit_id_constrain():
    """
    Description: test qubit id constrain.
    Expectation: success.
    """
    circ = Circuit().h(0).h(1).h(2).x(1, 0).x(2, 1)
    adder = MixerAdder(
        [
            QubitIDConstrain([0, 1]),
            BitFlipAdder(0.1),
        ]
    )
    new_circ = adder(circ)
    bit_flip = BitFlipChannel(0.1)
    exp_circ = Circuit(
        [
            H.on(0),
            bit_flip.on(0),
            H.on(1),
            bit_flip.on(1),
            H.on(2),
            X.on(1, 0),
            bit_flip.on(1),
            bit_flip.on(0),
            X.on(2, 1),
        ]
    )
    assert new_circ == exp_circ
