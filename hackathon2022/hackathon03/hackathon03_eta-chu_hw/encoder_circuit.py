from mindquantum import *


def generate_encoder():
    n_qubits = 3
    enc_layer = sum(
        [U3(f'a{i}', f'b{i}', f'c{i}', i) for i in range(n_qubits)])
    coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])
    encoder = sum(
        [add_prefix(enc_layer, f'l{i}') + coupling_layer for i in range(2)])
    return encoder, encoder.params_name
