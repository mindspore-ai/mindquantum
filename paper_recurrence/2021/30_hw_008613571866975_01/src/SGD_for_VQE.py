#%%
from mindquantum import *
import math
import numpy as np
import collections
import matplotlib.pyplot as plt
import tqdm
#%%
N = 8
d = 10  #50
alpha = 0.005
# Beta_1 = 0.9
# Beta_2 = 0.999
t_max = 400  #1000
n = 9
#%%
entangle_layer = UN(X, [int(i) for i in np.arange(0, (N - 1) // 2) * 2 + 1],
                    [int(i) for i in np.arange(0, (N - 1) // 2) * 2 + 2])
entangle_layer += UN(X, [int(i) for i in np.arange(0, N // 2) * 2],
                     [int(i) for i in np.arange(0, N // 2) * 2 + 1])
prepare_layer = UN(RY(np.pi / 4), N) + entangle_layer
rx_layer = Circuit([RX(f'x{i}').on(i) for i in range(N)]) + entangle_layer
ry_layer = Circuit([RY(f'y{i}').on(i) for i in range(N)]) + entangle_layer
rz_layer = Circuit([RZ(f'z{i}').on(i) for i in range(N)]) + entangle_layer
layers = [rx_layer, ry_layer, rz_layer]
VariationalCircuit = sum([add_prefix(layers[i % 3], f'l{i//3}') for i in range(d)],
                         start=prepare_layer)


#%%
def expectation_ZZ(theta, j):
    sim = Simulator('mqvector', N)
    sim.reset()
    parameters = dict(zip(VariationalCircuit.params_name, theta))
    VariationalCircuit_Measure = VariationalCircuit + Measure().on(
        j) + Measure().on(j + 1)
    res = sim.sampling(VariationalCircuit_Measure,
                       parameters,
                       shots=n,
                       seed=41)
    temp = 0
    for key in res.data:
        if (collections.Counter(key)['1'] % 2) == 0:
            temp += res.data[key]
        if (collections.Counter(key)['1'] % 2) == 1:
            temp -= res.data[key]
    expectation = temp / n
    # expectation = (res.data['00'] + res.data['11'] - res.data['01'] - res.data['10'])/n
    return expectation


def expectation_X(theta, j):
    sim = Simulator('mqvector', N)
    sim.reset()
    parameters = dict(zip(VariationalCircuit.params_name, theta))
    VariationalCircuit_Measure = VariationalCircuit + H.on(j) + Measure().on(j)
    res = sim.sampling(VariationalCircuit_Measure,
                       parameters,
                       shots=n,
                       seed=41)
    temp = 0
    for key in res.data:
        if (collections.Counter(key)['1'] % 2) == 0:
            temp += res.data[key]
        if (collections.Counter(key)['1'] % 2) == 1:
            temp -= res.data[key]
    expectation = temp / n
    # expectation = (res.data['0'] - res.data['1'])/n
    return expectation


def PartialDerivativeEstimator_1_infi(i, theta):
    sim = Simulator('mqvector', N)
    e_i = np.zeros(d * N)
    e_i[i] += 1
    sim.reset()
    sim.apply_circuit(VariationalCircuit, theta + np.pi / 2 * e_i)
    a = sim.get_expectation(ham)
    #print(a)
    sim.reset()
    sim.apply_circuit(VariationalCircuit, theta - np.pi / 2 * e_i)
    b = sim.get_expectation(ham)
    PartialDerivative = 1 / 2 * (a - b)
    #print(b)
    return PartialDerivative


def PartialDerivativeEstimator_1(i, theta):
    e_i = np.zeros(d * N)
    e_i[i] += 1
    temp = 0
    for j in range(0, N - 1):
        temp += 1 / 2 * (expectation_ZZ(theta + np.pi / 2 * e_i, j) -
                         expectation_ZZ(theta - np.pi / 2 * e_i, j))
    for j in range(0, N):
        temp += 1 / 2 * (expectation_X(theta + np.pi / 2 * e_i, j) -
                         expectation_X(theta - np.pi / 2 * e_i, j))
    PartialDerivative = temp
    return PartialDerivative


def loss_estimation(theta):
    temp = 0
    for j in range(0, N - 1):
        temp += expectation_ZZ(theta, j)
    for j in range(0, N):
        temp += expectation_X(theta, j)
    expectation = temp
    return expectation


# N = 8 ham

ham = Hamiltonian(
    sum([QubitOperator(f'Z{i} Z{i+1}')
         for i in range(N - 1)] + [QubitOperator(f'X{i}') for i in range(N)]))


#%%
def main():
    theta = np.ones(d * N)
    t = 0
    Loss = np.zeros(t_max)
    while t < t_max:
        for i in tqdm.tqdm(range(d * N)):
            g_i = PartialDerivativeEstimator_1(i, theta)
            # print(g_i)
            # g_i_infi = PartialDerivativeEstimator_1_infi(i,theta)
            # print(g_i_infi)
            theta[i] -= alpha * g_i
            # print('t=', t, 'i=', i)
        sim = Simulator('mqvector', N)
        sim.reset()
        print(theta)
        sim.apply_circuit(VariationalCircuit, theta)
        print(sim)
        loss = sim.get_expectation(ham)
        print('get_expectation_loss=', loss)
        Loss[t] = loss.real
        print(Loss)
        # my_loss = loss_estimation(theta)
        # print('my_loss=',my_loss)
        t += 1

    plt.xlabel("optimization steps")
    plt.ylabel("energy")
    plt.plot(Loss, c="c", label="9 shot")
    plt.legend()
    plt.show()
    plt.savefig('result.png')


main()