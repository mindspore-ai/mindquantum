import numpy as np
from mindquantum import RX, RY, RZ, Z, Circuit, UN


def RP(rotation_angle):
    """
    :param rotation_angle: parameter of the gate;
    :return: randomly return one of RX, RY, and RZ with equal probability.
    """
    a = np.random.randint(0, 3)
    if a == 0:
        return RX(rotation_angle)
    elif a == 1:
        return RY(rotation_angle)
    elif a == 2:
        return RZ(rotation_angle)
    else:
        print("error in random Pauli gates")


def bpansatz(n, p):
    """
    :param n: number of qubits;
    :param p: number of layers;
    :return: an 1 dimensional random circuit with n qubits and p layers.
    """
    qc = Circuit()
    for j in range(p):
        for i in range(n):
            qc += RP(f'c({j},{i})').on(i)
        for ii in range(n - 1):
            qc += Z(ii, ii + 1)
        qc += Z(0, n-1)
    return qc


def init_state(n):
    """
    :param n: number of qubits;
    :return: a circuit for state preparation.
    """
    qc = Circuit()
    qc += UN(RY(np.pi / 4), range(n))
    return qc


if __name__ == '__main__':
    circ = init_state(5) + bpansatz(5, 1)
    print(circ)
    print('test over')
