from mindquantum import *
import numpy as np
from scipy.optimize import minimize
import time

train_x = np.load('train_x.npy', allow_pickle=True)
train_y = np.load('train_y.npy', allow_pickle=True)
test_x = np.load('test_x.npy', allow_pickle=True)
test_y = np.load('test_y.npy', allow_pickle=True)
real_test_y = np.load('real_test_y.npy', allow_pickle=True)

n_qubits = 3


def generate_encoder():

    enc_layer = Circuit()
    for i in range(n_qubits):
        enc_layer += U3(f'a{i}', f'b{i}', f'c{i}', i)

    coupling_layer = UN(X, [1, 2, 0], [0, 1, 2])

    encoder = Circuit()
    for i in range(2):
        encoder += add_prefix(enc_layer, f'l{i}')
        encoder += coupling_layer

    return encoder, encoder.params_name


def train_circuit():

    layer = 3
    n_qubits = 4
    ansatz = Circuit()

    for k in range(layer):
        for i in range(n_qubits):
            ansatz += RX(f'beta{i + 20*k}').on(i)
            ansatz += RX(f'beta{i + 4 + 20*k}').on(i)

        ansatz += BarrierGate()

        ansatz += RX(f'beta{8  + 20*k}').on(2, 3)
        ansatz += RX(f'beta{9  + 20*k}').on(1, 3)
        ansatz += RX(f'beta{10 + 20*k}').on(0, 3)
        ansatz += RX(f'beta{11 + 20*k}').on(3, 2)
        ansatz += RX(f'beta{12 + 20*k}').on(1, 2)
        ansatz += RX(f'beta{13 + 20*k}').on(0, 2)
        ansatz += RX(f'beta{14 + 20*k}').on(3, 1)
        ansatz += RX(f'beta{15 + 20*k}').on(2, 1)
        ansatz += RX(f'beta{16 + 20*k}').on(0, 1)
        ansatz += RX(f'beta{17 + 20*k}').on(3, 0)
        ansatz += RX(f'beta{18 + 20*k}').on(2, 0)
        ansatz += RX(f'beta{19 + 20*k}').on(1, 0)

        ansatz += BarrierGate()

    for i in range(n_qubits):
        ansatz += RX(f'beta{60 + i}').on(i)
        ansatz += RZ(f'beta{64 + i}').on(i)

    return ansatz, ansatz.params_name


def _param(param_name, param_value):

    para_dic = {}
    for name, value in zip(param_name, param_value):
        para_dic[name] = value

    return para_dic


## 我们通个这个函数得到|\psi\rangle的量子态, 3个量子比特，800个训练数据， 得到(800,8)的数据
def generate_initial_quantum_state(train_data):

    initial_quantum_data = np.zeros((len(train_data), 8), dtype=complex)
    encoder, paras_name = generate_encoder(
    )  # generate encoder circuit and circuit's parameter's name
    for i, data in enumerate(train_data):
        sim = Simulator('mqvector', n_qubits)
        pr = _param(paras_name, data)
        sim.apply_circuit(encoder, pr)
        result = sim.get_qs()
        for j, element in enumerate(result):
            initial_quantum_data[i, j] = element
        sim.reset()

    return initial_quantum_data


## 用 ansatz 的线路作用到初始态上得到目标态:
def predict_quantum_state(psi, parameters):

    sim = Simulator('mqvector', n_qubits)

    sim.set_qs(psi)

    pr = _param(ansatz.params_name, parameters)
    sim.apply_circuit(ansatz, pr)
    result = sim.get_qs()

    return result


def _cost_function(parameters):

    phi = [
        predict_quantum_state(psi, parameters) for psi in initial_quantum_data
    ]
    cost = [(1 - abs(np.vdot(train_y[i], phi[i]))**2)**2
            for i in range(len(phi))]
    cost = 1 / len(phi) * sum(cost)

    phi_test = [
        predict_quantum_state(psi, parameters)
        for psi in train_test_quantum_data
    ]
    acc = np.real(
        np.mean([
            np.abs(np.vdot(normal(bra), ket))
            for bra, ket in zip(phi_test, train_y[750:])
        ]))
    print("The acc is: ", acc)

    return cost


def normal(state):
    return state / np.sqrt(np.abs(np.vdot(state, state)))


def pred_accuracy(parameters, quantum_state,
                  real_quantum_state):  # this quantum state is 18

    phi = [predict_quantum_state(psi, parameters) for psi in quantum_state]
    acc = np.real(
        np.mean([
            np.abs(np.vdot(normal(bra), ket))
            for bra, ket in zip(phi, real_quantum_state)
        ]))

    return acc


#### THE ENCODER AND THE ANSTAT
encoder, paras = generate_encoder()
ansatz, ansatz_param = train_circuit()

###  THE TRAING AND TESTING DATA
initial_quantum_data = generate_initial_quantum_state(train_x[:750])
train_test_quantum_data = generate_initial_quantum_state(train_x[750:])

np.random.seed(2)

#### THE COST BEFOR TRAINING
ansatz_parameters = np.random.rand(len(ansatz_param))
train_before = _cost_function(ansatz_parameters)
print("随机参数是: ", ansatz_parameters)
print("训练前的损失含数的值为: ", train_before)

### TRAIN
start = time.time()
result = minimize(_cost_function, ansatz_parameters, method='SLSQP')
end = time.time()
print('Running time: %s Seconds' % (end - start))

### GET THE OPTIMAL PARAMETER
result.fun
optimal_theta = result.x

train_after = _cost_function(optimal_theta)
print("优化参数是: ", optimal_theta)
print("训练后的损失含数的值为: ", train_after)

### GET THE PREDICTED TEST QUANTUM STATE
initial_quantum_data_test = generate_initial_quantum_state(test_x)
pred_quantum_state_test = np.zeros((500, 8), dtype=complex)
for i in range(500):
    pred = predict_quantum_state(initial_quantum_data_test[i], optimal_theta)
    for j, state in enumerate(pred):
        pred_quantum_state_test[i, j] = state

np.save('test_y', pred_quantum_state_test, allow_pickle=True, fix_imports=True)
