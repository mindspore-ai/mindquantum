"""A noise model."""
from mindquantum.core.circuit import SequentialAdder, MixerAdder, MeasureAccepter, BitFlipAdder, QubitNumberConstrain, NoiseExcluder, DepolarizingChannelAdder, ReverseAdder, NoiseChannelAdder
from mindquantum.core.gates import ThermalRelaxationChannel

def generate_noise_model():
    """
    生成一个噪声模型。
    """
    noise = SequentialAdder([
        MixerAdder(
            [  # 测量具有比特翻转误差
                MeasureAccepter(),
                BitFlipAdder(0.05),
            ],
            add_after=False),
        MixerAdder([ # 单比特门有去极化噪声
            QubitNumberConstrain(1),
            NoiseExcluder(),
            DepolarizingChannelAdder(0.001, 1),
            ReverseAdder(MeasureAccepter()),
        ]),
        MixerAdder([ # 单比特门有热弛豫噪声
            QubitNumberConstrain(1),
            NoiseExcluder(),
            NoiseChannelAdder(ThermalRelaxationChannel(100000, 50000, 30)),
            ReverseAdder(MeasureAccepter()),
        ]),
        MixerAdder([ # 双比特门有去极化噪声
            QubitNumberConstrain(2),
            NoiseExcluder(),
            DepolarizingChannelAdder(0.004, 2),
            ReverseAdder(MeasureAccepter()),
        ]),
        MixerAdder([ # 双比特门有热弛豫噪声
            QubitNumberConstrain(2),
            NoiseExcluder(),
            NoiseChannelAdder(ThermalRelaxationChannel(100000, 50000, 80)),
            ReverseAdder(MeasureAccepter()),
        ])
    ])
    return noise


if __name__ == '__main__':
    from mindquantum import *
    circ = qft(range(3)).measure_all()
    noise_model = generate_noise_model()
    noise_circ = noise_model(circ)
    print(circ)
    print(noise_circ)
