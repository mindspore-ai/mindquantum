import numpy as np
from mindquantum import Circuit, X, H, RZ, RY, PhaseShift, Hamiltonian, Simulator, QubitOperator, Hamiltonian, MQAnsatzOnlyLayer
import mindspore as ms
from mindspore.common.parameter import Parameter
import mindspore.context as context

context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")


def N(count, i):
    temp = Circuit()
    temp += RZ(np.pi / 2).on(i + 1)
    temp += X.on(i, i + 1)
    temp += RZ({f'p{count}' : 2}).on(i)
    temp += RZ(-np.pi / 2).on(i)
    temp += RY(np.pi / 2).on(i + 1)
    temp += RY({f'p{count}' : -2}).on(i + 1)
    temp += X.on(i + 1, i)
    temp += RY({f'p{count}' : 2}).on(i + 1)
    temp += RY(-np.pi / 2).on(i + 1)
    temp += X.on(i, i + 1)
    temp += RZ(-np.pi / 2).on(i)
    return temp


NMtable = {4:2, 8:3, 10:3, 16:5, 20:6}
num = 8


temp = num
while num % 2 == 0 and num > 4:
    num = num // 2
m = NMtable[num]
assert num % 2 == 0

print(f'Current N: {num}')


encoder = Circuit()
for i in range(0, num, 2):
    encoder += X.on(i)
    encoder += X.on(i + 1)
    encoder += H.on(i)
    encoder += X.on(i + 1, i)
    
# print(encoder)
# encoder.summary()


ansatz = Circuit()
count = 1

for j in range(m):
    for i in range(0, num - 1, 2):
        ansatz += N(count, i)
        count += 1
    for i in range(1, num - 1, 2):
        ansatz += N(count, i)
        count += 1
    for i in range(0, num // 2):
        ansatz += PhaseShift('p%d' % count).on(i)
        ansatz += PhaseShift({'p%d' % count : -1}).on(num - i - 1)
        count += 1

ansatz = encoder + ansatz

# print(ansatz)
# ansatz.summary()


J = 1
ham = QubitOperator('')
for i in range(num - 1):
    ham += QubitOperator('X%d X%d' % (i, i + 1), J)
    ham += QubitOperator('Y%d Y%d' % (i, i + 1), J)
    ham += QubitOperator('Z%d Z%d' % (i, i + 1), J)
ham -= QubitOperator('')

# print(ham)


val = np.random.randn(count - 1)

sim = Simulator('projectq', num)
pqc = sim.get_expectation_with_grad(Hamiltonian(ham), ansatz)
pqcnet = MQAnsatzOnlyLayer(pqc)
pqcnet.weight = Parameter(ms.Tensor(val, pqcnet.weight.dtype))

initial_energy = pqcnet()
print("Initial energy: %20.16f" % (initial_energy.asnumpy()))


optimizer = ms.nn.Adam(pqcnet.trainable_params(), learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
train_pqcnet = ms.nn.TrainOneStepCell(pqcnet, optimizer)

eps = 1.e-8
print("eps: ", eps)
energy_diff = eps * 1000
energy_last = initial_energy.asnumpy() + energy_diff
iter_idx = 0
while (abs(energy_diff) > eps):
    energy_i = train_pqcnet().asnumpy()
    if iter_idx % 100 == 0:
        print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
    energy_diff = energy_last - energy_i
    energy_last = energy_i
    iter_idx += 1

print("Optimization completed at step %3d" % (iter_idx - 1))
print("Optimized energy: %20.16f" % (energy_i))
print("Optimized amplitudes: \n", pqcnet.weight.asnumpy())


while num < temp:
    num2 = 2 * num
    m2 = NMtable[num2]
    # print(num2, m2)

    print(f'Current N: {num2}')


    encoder = Circuit()
    for i in range(0, num2, 2):
        encoder += X.on(i)
        encoder += X.on(i + 1)
        encoder += H.on(i)
        encoder += X.on(i + 1, i)

    ansatz = Circuit()
    count = 1

    for j in range(m2):
        for i in range(0, num2 - 1, 2):
            ansatz += N(count, i)
            count += 1
        for i in range(1, num2 - 1, 2):
            ansatz += N(count, i)
            count += 1
        for i in range(0, num2 // 2):
            ansatz += PhaseShift('p%d' % count).on(i)
            ansatz += PhaseShift({'p%d' % count : -1}).on(num2 - i - 1)
            count += 1

    ansatz = encoder + ansatz
    # print(ansatz)
    # ansatz.summary()

    J = 1
    ham = QubitOperator('')
    for i in range(num2 - 1):
        ham += QubitOperator('X%d X%d' % (i, i + 1), J)
        ham += QubitOperator('Y%d Y%d' % (i, i + 1), J)
        ham += QubitOperator('Z%d Z%d' % (i, i + 1), J)
    ham = Hamiltonian(ham - QubitOperator(''))

    # print(ham)


    pre = pqcnet.weight.asnumpy()
    half = num // 2
    L1 = half * 3 - 1
    L2 = num * 3 - 1
    # print(L1, L2)
    val = np.random.randn(L2 * m2)
    # for j in range(m):
        # val[j * L2 + num + half - 1] = 0    # To utilize Attempt 1
    # val = np.zeros(L2 * m2)    # To utilize Attempt 2

    for j in range(m):
        for i in range(half):
            val[i + j * L2] = pre[i + j * L1]
            val[i + j * L2 + half] = pre[i + j * L1]
    for j in range(m):
        for i in range(half - 1):
            val[i + j * L2 + num] = pre[i + j * L1 + half]
            val[i + j * L2 + num + half] = pre[i + j * L1 + half]
    for j in range(m):
        for i in range(half):
            val[i + j * L2 + num2 - 1] = pre[i + j * L1 + num - 1]
            val[num - i - 2 + j * L2 + num2] = -pre[i + j * L1 + num - 1]

    '''print('old parameters:')
    for i in range(L1 * m):
        print(f'p{i + 1}', pre[i])
    print('new parameters:')
    for i in range(L2 * m2):
        print(f'p{i + 1}', val[i])'''


    sim = Simulator('projectq', num2)
    pqc = sim.get_expectation_with_grad(ham, ansatz)
    pqcnet = MQAnsatzOnlyLayer(pqc)
    pqcnet.weight = Parameter(ms.Tensor(val, pqcnet.weight.dtype))

    initial_energy = pqcnet()
    print("Initial energy: %20.16f" % (initial_energy.asnumpy()))

    optimizer = ms.nn.Adam(pqcnet.trainable_params(), learning_rate = 0.01, beta1 = 0.9, beta2 = 0.999, eps = 1e-8)
    train_pqcnet = ms.nn.TrainOneStepCell(pqcnet, optimizer)

    eps = 1.e-8
    print("eps: ", eps)
    energy_diff = eps * 1000
    energy_last = initial_energy.asnumpy() + energy_diff
    iter_idx = 0
    while (abs(energy_diff) > eps):
        energy_i = train_pqcnet().asnumpy()
        if iter_idx % 100 == 0:
            print("Step %3d energy %20.16f" % (iter_idx, float(energy_i)))
        energy_diff = energy_last - energy_i
        energy_last = energy_i
        iter_idx += 1

    print("Optimization completed at step %3d" % (iter_idx - 1))
    print("Optimized energy: %20.16f" % (energy_i))
    print("Optimized amplitudes: \n", pqcnet.weight.asnumpy())

    num = num2
    m = m2