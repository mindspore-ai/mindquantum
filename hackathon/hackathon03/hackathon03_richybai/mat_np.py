from encoder_circuit import generate_encoder
from mindquantum import *
import numpy as np
import scipy

from mindquantum.core.gates import UnivMathGate

# np.set_printoptions(precision=3)

def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict


def get_vector(simulator, circuit, params):
        param_dict = param2dict(circuit.params_name, params)
        simulator.reset()
        simulator.apply_circuit(circuit, param_dict)
        return simulator.get_qs()


def is_unitary(matrix): 

    unitary = True
    n = matrix.shape[0]
    error = np.linalg.norm(np.eye(n) - matrix.dot( matrix.transpose().conjugate()))

    if not(error < np.finfo(matrix.dtype).eps * 10.0 *n):
        unitary = False

    return unitary


if __name__ == "__main__":

    # simulator = Simulator('projectq', 3)
    
    # encoder, _ = generate_encoder()
    # encoder.summary()
    # print(encoder)    
    # train_x = np.load("train_x.npy", allow_pickle=True)

    # train_y = np.load("train_y.npy", allow_pickle=True)

    # test_x = np.load("test_x.npy", allow_pickle=True)

    

    # train_x_mat = []
    # for i in range(len(train_x)):
    #     train_x_mat.append(get_vector(simulator, encoder, train_x[i]))
    # train_x_mat = np.array(train_x_mat)
    # print("train quantum state after encoder:", train_x_mat.shape)

    # test_x_mat = []
    # for i in range(len(test_x)):
    #     test_x_mat.append(get_vector(simulator, encoder, test_x[i]))
    # test_x_mat = np.array(test_x_mat)
    # print("test quantum state after encoder:", test_x_mat.shape)

    # data = {}
    # data["train_x"] = train_x_mat
    # data["train_y"] = train_y
    # data["test_x"] = test_x_mat
    # scipy.io.savemat("data.mat", data)

    # x = scipy.io.loadmat("data.mat")
    # print(x.keys())
    # print(x["train_x"].shape)
    # print(x["test_x"].shape)
    # np.save("train_x.npy", x["train_x"], allow_pickle=True)
    # np.save("test_x.npy", x["test_x"], allow_pickle=True)



    # x = scipy.io.loadmat("res.mat")
    # print(x.keys())
    # print(x["B_iter"].shape)
    # np.save("test_y.npy", x["B_iter"], allow_pickle=True)


    # test_y = np.load("test_y.npy", allow_pickle=True)

    # print(np.vdot(test_y[0], test_y[0]))

    unit = scipy.io.loadmat("Q.mat")
    print(unit.keys())
    unit = unit["Q"]
    print(unit.shape)    
    print(is_unitary(unit))

    # u = train_x[0: 8]
    # print(is_unitary(u))
    # gate = UnivMathGate('U',unit)
    # print(gate.matrix())