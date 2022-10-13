# -*- coding: utf-8 -*-
"""
[1] https://tensorflow.google.cn/quantum/api_docs/python/tfq/datasets/tfi_chain
[2] https://doi.org/10.48550/arXiv.2111.05076

@NE
"""
import os
import numpy as np
from mindquantum import Circuit, UN, H, ZZ, RX
def build_dataset(nqubits, data_dir, n=None):
    """
    Build dataset `tfi_chain`.
    1D Transverse field Ising-model quantum data set.

    Args:
        nqubits (int): Number of qubits.
        data_dir (str): Dataset path.
                        See `tfi_chain`.
        n (int): Number of values are inserted between two points.
    """
    encoder, encoder_params_name, x, y = tfi_chain(nqubits, data_dir)
    if n is None:
        return encoder, encoder_params_name, x, y
    x = interpolate_linear(x, n)
    n = x.shape[0] // 2
    y = np.array([-1] * n + [0] + [1] * n)
    return encoder, encoder_params_name, x, y
def interpolate_linear(x, n):
    """
    Linear Interpolation.

    Args:
        x (np.ndarray): `x` to be interpolated.
        n (int): Number of values are inserted between two points.
    """
    n += 1
    xi = np.linspace(x[0], x[1], num=n, endpoint=False)
    for i in range(1, x.shape[0]-1):
        xi_ = np.linspace(x[i], x[i+1], num=n, endpoint=False)
        xi = np.concatenate((xi, xi_))
    xi = np.concatenate((xi, [x[-1]]))
    return xi
def unique_name():
    """Generator to generate an infinite number of unique names.
    Yields:
        Python `str` of the form "theta_<integer>".
    """
    num = 0
    while True:
        yield "theta_" + str(num)
        num += 1
def tfi_chain(nqubits, data_dir): # tfi_chain
    """
    Build dataset `tfi_chain`.
    1D Transverse field Ising-model quantum data set.

    Datasets `TFI_chain.zip` downloaded from:
    https://storage.googleapis.com/download.tensorflow.org/data/quantum/spin_systems/TFI_chain.zip
    """
    supported_n = [4, 8, 12]
    if nqubits not in supported_n:
        raise ValueError("Supported number of qubits are {}, received {}".format(
            supported_n, nqubits))
    depth = nqubits // 2
    data_path = data_dir + str(nqubits) + '/'

    name_generator = unique_name()
    # 2 * N/2 parameters.
    params_name = [next(name_generator) for _ in range(nqubits)]

    # Define the circuit.
    circuit = Circuit()
    circuit += UN(H, list(range(nqubits)))
    for d in range(depth):
        for q1, q2 in zip(range(nqubits), range(1, nqubits)):
            circuit += ZZ(params_name[d]).on([q1, q2])
        circuit += ZZ(params_name[d]).on([nqubits - 1, 0])
        for q in range(nqubits):
            circuit += RX(params_name[d + depth]).on(q)

    # Initiate lists.
    order_parameters = []
    labels = []
    params = []
    # Load the data and append to the lists.
    for i, dirn in enumerate(x for x in os.listdir(data_path)):
        g = float(dirn)
        # Set labels for the different phases.
        if g < 1.0:
            labels.append(-1)
        elif g > 1.0:
            labels.append(1)
        else:
            labels.append(0)
        order_parameters.append(g)
        pm = np.load(os.path.join(data_path, dirn, "params.npy")) / np.pi
        params.append(pm.flatten())

    # Make sure that the data is ordered from g=0.2 to g=1.8.
    _, params, labels = zip(*sorted(zip(order_parameters, params, labels)))
    return circuit, params_name, np.array(params), np.array(labels)

def _main():
    #path = './TFI_chain/closed/'
    path = './dataset/closed/'
    n = 4
    encoder, encoder_params_name, x, y = tfi_chain(n, path)
    print(encoder)
    print(encoder_params_name)
    print(x)
    print(y)
    print(y.shape)
if __name__ == '__main__':
    _main()
