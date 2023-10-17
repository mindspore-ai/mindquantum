from mindquantum import *
import math
import numpy as np
import collections
import matplotlib.pyplot as plt

N = 8
d = 10  #50
alpha = 0.005
# Beta_1 = 0.9
# Beta_2 = 0.999
t_max = 400  #1000
n = 9

VariationalCircuit = Circuit()
#first layer
for i in range(0, N):
    VariationalCircuit += RY(math.pi / 4).on(i)
for i in range(0, N):
    if (i % 2) == 1 and i < N - 1:
        VariationalCircuit += X.on(i + 1, i)
for i in range(0, N):
    if (i % 2) == 0 and i < N - 1:
        VariationalCircuit += X.on(i + 1, i)
#Blocks
for j in range(0, d // 3):
    for i in range(0, N):
        VariableName = 'theta_' + str(j * 3 * N + i + 1)
        VariationalCircuit += RX(VariableName).on(i)
    for i in range(0, N):
        if (i % 2) == 1 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        if (i % 2) == 0 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        VariableName = 'theta_' + str(j * 3 * N + N + i + 1)
        VariationalCircuit += RY(VariableName).on(i)
    for i in range(0, N):
        if (i % 2) == 1 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        if (i % 2) == 0 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        VariableName = 'theta_' + str(j * 3 * N + N + N + i + 1)
        VariationalCircuit += RZ(VariableName).on(i)
    for i in range(0, N):
        if (i % 2) == 1 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        if (i % 2) == 0 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
#the remaining layers
if (d % 3) == 1:
    for i in range(0, N):
        VariableName = 'theta_' + str((d // 3) * 3 * N + i + 1)
        VariationalCircuit += RX(VariableName).on(i)
    for i in range(0, N):
        if (i % 2) == 1 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        if (i % 2) == 0 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
if (d % 3) == 2:
    for i in range(0, N):
        VariableName = 'theta_' + str((d // 3) * 3 * N + i + 1)
        VariationalCircuit += RX(VariableName).on(i)
    for i in range(0, N):
        if (i % 2) == 1 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        if (i % 2) == 0 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        VariableName = 'theta_' + str((d // 3) * 3 * N + N + i + 1)
        VariationalCircuit += RY(VariableName).on(i)
    for i in range(0, N):
        if (i % 2) == 1 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
    for i in range(0, N):
        if (i % 2) == 0 and i < N - 1:
            VariationalCircuit += X.on(i + 1, i)
# VariationalCircuit


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
    QubitOperator('Z0 Z1') + QubitOperator('Z1 Z2') + QubitOperator('Z2 Z3') +
    QubitOperator('Z3 Z4') + QubitOperator('Z4 Z5') + QubitOperator('Z5 Z6') +
    QubitOperator('Z6 Z7') + QubitOperator('X0') + QubitOperator('X1') +
    QubitOperator('X2') + QubitOperator('X3') + QubitOperator('X4') +
    QubitOperator('X5') + QubitOperator('X6') + QubitOperator('X7'))


def main():
    theta = np.ones(d * N)
    t = 0
    Loss = np.zeros(t_max)
    while t < t_max:
        for i in range(d * N):
            g_i = PartialDerivativeEstimator_1(i, theta)
            # print(g_i)
            # g_i_infi = PartialDerivativeEstimator_1_infi(i,theta)
            # print(g_i_infi)
            theta[i] -= alpha * g_i
            print('t=', t, 'i=', i)
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