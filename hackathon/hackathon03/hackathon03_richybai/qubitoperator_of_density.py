import itertools
import math
import numpy as np
from mindquantum import *


def get_coefficient(matrix):
    m, n = matrix.shape
    vector = []
    for i in range(m):
        for j in range(i, n):
            if i == j:
                vector.append(matrix[i, j].real)
            else:
                vector.append(matrix[i, j].real)
                vector.append(matrix[i, j].imag)
    vector = np.array(vector)
    return vector


def get_qubitoperator(vector):
    """
    vector 输入的是行向量, 直接取 train_y[n:n+1, :]
    """
    x = vector.transpose()
    n_qubits = int(math.log(len(x), 2))
    # print("num qubit is: ", n_qubits)
    density_matrix = x.dot(x.transpose().conjugate())
    # print(density_matrix)

    # 右端向量, 与 输入向量有关
    b = np.zeros((2**(n_qubits*2), 1))
    b[:, 0] = get_coefficient(density_matrix)

    # 系数矩阵, 与输入向量无关, 只与bit数有关
    A = np.zeros((2**(n_qubits*2), 2**(n_qubits*2)))
    
    literal_list = []
    for op in itertools.product(["I", "X", "Y", "Z"], repeat=n_qubits):
        literal = ""
        for n, pauli in enumerate(op):
            if pauli == "I":
                continue
            else:
                literal += (pauli + str(n) + " ")
        literal_list.append(literal)

    for i, literal in enumerate(literal_list):
        qubitoperator = QubitOperator(literal)
        A[:, i] = get_coefficient(qubitoperator.matrix(n_qubits=n_qubits).todense())
    x = np.linalg.inv(A).dot(b)
    x = list(x.flatten())

    final_qubitoperator = sum([coeff*QubitOperator(literal) for (coeff, literal) in zip(x, literal_list)])

    hamiltonian = final_qubitoperator.matrix(n_qubits=n_qubits).todense()
    
    if not np.isclose(hamiltonian, density_matrix).all():
        print("wrong!")

    return final_qubitoperator

if __name__ == "__main__":

    train_y = np.load("train_y.npy", allow_pickle=True)
    qubitoperator_list = []
    for i in range(len(train_y)):
        qubitoperator_list.append(get_qubitoperator(train_y[i: i+1, :]))
        if (i+1) % 10 == 0:
            print(i+1)
    
    data = np.array(qubitoperator_list)
    np.save("qubitoperator.npy", data)
    print("finished")