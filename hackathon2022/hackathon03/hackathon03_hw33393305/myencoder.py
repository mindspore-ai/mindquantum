from mindquantum.algorithm.library import amplitude_encoder
import numpy as np
from mindquantum import Circuit, RY, X, BarrierGate, PhaseShift

def generate_3d_complex_encoder(data):
    pr_real = []
    pr_image = []
    encoder_real, real = amplitude_encoder(np.abs(data), 3)
    pr_real.append(real.para_value)
    for i in range(8):
        pr_image.append(np.angle(data[i]))

    encoder_image = Circuit()
    for i in range(8):
        x = bin(i)[2:].zfill(3)
        encoder_image.barrier()
        for j in reversed(range(3)):
            if x[j] == '0':
                encoder_image += X.on(2 - j)
        encoder_image += PhaseShift(f'beta{i}').on(2, [0, 1])
        for j in reversed(range(3)):
            if x[j] == '0':
                encoder_image += X.on(2 - j)
    return encoder_real.apply_value(real) + encoder_image.apply_value(dict(zip(encoder_image.params_name, pr_image)))

def generate_circuit():
    encoder_real = Circuit()
    encoder_real += RY('enc1_0').on(2)
    encoder_real += X.on(2)
    encoder_real += RY('enc1_1').on(1, 2)
    encoder_real += X.on(2)
    encoder_real += RY('enc1_2').on(1, 2)
    encoder_real += X.on(1)
    encoder_real += X.on(2)
    encoder_real += RY('enc1_3').on(0, [1, 2])
    encoder_real += X.on(1)
    encoder_real += RY('enc1_4').on(0, [1, 2])
    encoder_real += X.on(1)
    encoder_real += X.on(2)
    encoder_real += RY('enc1_5').on(0, [1, 2])
    encoder_real += X.on(1)
    encoder_real += RY('enc1_6').on(0, [1, 2])

    encoder_image = Circuit()
    for i in range(8):
        x = bin(i)[2:].zfill(3)
        encoder_image.barrier()
        for j in reversed(range(3)):
            if x[j] == '0':
                encoder_image += X.on(2 - j)
        encoder_image += PhaseShift(f'enc2_{i}').on(2, [0, 1])
        for j in reversed(range(3)):
            if x[j] == '0':
                encoder_image += X.on(2 - j)

    return encoder_real + encoder_image

def generate_value(data):
    pr_real = []
    pr_image = []
    encoder_real, real = amplitude_encoder(np.abs(data), 3)
    pr_real = real.para_value
    for i in range(8):
        pr_image.append(np.angle(data[i]))

    return pr_real + pr_image