from encoder_circuit import generate_encoder
from ansatz import *
from qubitoperator_of_density import *
import mindspore as ms
from mindquantum import *
import numpy as np

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def param2dict(keys, values):
    param_dict = {}
    for (key, value) in zip(keys, values):
        param_dict[key] = value
    return param_dict


def get_vector(simulator, circuit, params, qs):
    param_dict = param2dict(circuit.params_name, params)
    simulator.reset()
    simulator.set_qs(qs)
    simulator.apply_circuit(circuit, param_dict)
    return simulator.get_qs()


if __name__ == "__main__":

    ansatz = ansatz()
    print("summary of ansatz:")
    ansatz.summary()

    # 已经把x转化成8维向量了
    train_x = np.load("train_x.npy", allow_pickle=True)

    eval_x = np.load("test_x.npy", allow_pickle=True)

    simulator = Simulator('mqvector', 3)

    qubitoperator_list = np.load("qubitoperator.npy", allow_pickle=True)
    qubitoperator_list = qubitoperator_list.tolist()

    params = np.load("params.npy", allow_pickle=True)

    # for epoch in range(20):
    #     print("epoch ", epoch)
    #     batch = 0
    #     vi = 0
    #     g = np.zeros(len(ansatz.params_name))
    #     for i in range(len(train_x)):

    #         simulator.reset()
    #         simulator.set_qs(train_x[i])

    #         ham = Hamiltonian(qubitoperator_list[i])

    #         grad_ops = simulator.get_expectation_with_grad(
    #             ham,
    #             ansatz,
    #             parallel_worker=5)

    #         f, gi = grad_ops(params)

    #         g += gi.real[0, 0]

    #         vi = 0.9*vi + 0.1*g
    #         if (i+1) % 20 == 0:
    #             batch += 1
    #             params = params + 0.001 * vi
    #             print(f"batch: {batch}, inner product: {f[0][0]}")
    #             g = np.zeros(len(ansatz.params_name))
    # if epoch == 10:
    #     break

    np.save("params.npy", params, allow_pickle=True)

    eval_y = []
    for i in range(len(eval_x)):
        eval_y.append(get_vector(simulator, ansatz, params, eval_x[i]))
    eval_y = np.array(eval_y)
    np.save("test_y.npy", eval_y)
