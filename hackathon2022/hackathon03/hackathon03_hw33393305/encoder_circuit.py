from mindquantum import *


def generate_encoder():
    n_qubits = 3

    enc_layer = Circuit()
    for i in range(n_qubits):
        enc_layer += U3(f'a{i}', f'b{i}', f'c{i}', i)

    coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])

    encoder = Circuit()
    for i in range(2):
        encoder += add_prefix(enc_layer, f'l{i}')
        encoder += coupling_layer

    return encoder, encoder.params_name


encoder, paras = generate_encoder()
encoder