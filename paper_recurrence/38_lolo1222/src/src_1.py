'''
This file is to present VQSVD using in random matrix
'''
import os

import mindspore as ms
from mindquantum import Simulator, MQAnsatzOnlyLayer, add_prefix
from mindquantum import Hamiltonian, Circuit, RY, RZ, X
import numpy as np

from scipy.sparse import csr_matrix
from scipy.linalg import norm
from matplotlib import pyplot
import tqdm

os.environ['OMP_NUM_THREADS'] = '1'

n_qubits = 3  # qbits number
cir_depth = 20  # circuit depth
N = 2**n_qubits
rank = 8  # learning rank
step = 3
ITR = 200  # iterations
LR = 0.02  # learning rate

# Set equal learning weights
if step == 0:
    weight = ms.Tensor(np.ones(rank))
else:
    weight = ms.Tensor(np.arange(rank * step, 0, -step))

# Define random seed
np.random.seed(42)


def mat_generator():
    '''
    Generate a random complex matrix
    '''
    matrix = np.random.randint(
        10, size=(N, N)) + 1j * np.random.randint(10, size=(N, N))
    return matrix


# Generate matrix M which will be decomposed
M = mat_generator()

# m_copy is generated to error analysis
m_copy = np.copy(M)

# Get SVD results
U, D, v_dagger = np.linalg.svd(M, full_matrices=True)


class Ansatz:
    '''
    Define ansatz
    '''

    def __init__(self, n, depth):
        self.circ = Circuit()
        num = 0
        for _ in range(depth):

            for i in range(n):
                self.circ += RY('theta' + str(num)).on(i)
                num += 1

            for i in range(n):
                self.circ += RZ('theta' + str(num)).on(i)
                num += 1

            for i in range(n - 1):
                self.circ += X.on(i + 1, i)

            self.circ += X.on(0, n - 1)


def loss_plot(loss):
    '''
    Plot loss over iteration
    '''
    pyplot.plot(list(range(1, len(loss) + 1)), loss)
    pyplot.xlabel('iteration')
    pyplot.ylabel('loss')
    pyplot.title('Loss Over Iteration')
    pyplot.suptitle('step = ' + str(step))
    pyplot.show()


def quantnet(qubits_num, hams, circ_right, circ_left=None, base=None):
    '''
    Generate quantum net using hams, circ_right and circ_left under given base
    '''
    sim = Simulator('projectq', qubits_num)

    if base is None:
        pass
    else:
        sim.set_qs(base)
    grad_ops = sim.get_expectation_with_grad(hams, circ_right, circ_left)

    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

    quantumnet = MQAnsatzOnlyLayer(grad_ops, 'ones')

    return quantumnet


# Define ansatz
u_ansatz = add_prefix(Ansatz(n_qubits, cir_depth).circ, 'u')
v_ansatz = add_prefix(Ansatz(n_qubits, cir_depth).circ, 'v')

# Embed M matrix into Hamiltonian ham
ham = Hamiltonian(csr_matrix(M))

i_matrix = np.identity(N)
quantum_models = dict()
quantum_models['net_0'] = quantnet(n_qubits, ham, v_ansatz, u_ansatz,
                                   i_matrix[0])
for s in range(1, rank):
    quantum_models["net_" + str(s)] = quantnet(n_qubits, ham, v_ansatz,
                                               u_ansatz, i_matrix[s])
    quantum_models["net_" + str(s)].weight = quantum_models['net_0'].weight


class MyNet(ms.nn.Cell):
    '''
    define quantum-classic net
    '''

    def __init__(self):
        super(MyNet, self).__init__()

        self.build_block = ms.nn.CellList()
        for j in range(rank):
            self.build_block.append(quantum_models["net_" + str(j)])

    def construct(self):
        x = self.build_block[0]() * weight[0]
        k = 1

        for layer in self.build_block[1:]:
            x += layer() * weight[k]
            k += 1

        return -x


# Define network
net = MyNet()

# Define optimizer
opt = ms.nn.Adam(net.trainable_params(), learning_rate=LR)

# Simple gradient descent
train_net = ms.nn.TrainOneStepCell(net, opt)

# Start to train net
loss_list = list()
for itr in tqdm.tqdm(range(ITR)):
    res = train_net()
    loss_list.append(res.asnumpy().tolist())

# Get singular value results
singular_value = list()

for _, qnet in quantum_models.items():
    singular_value.append(qnet().asnumpy()[0])

# Plot loss over iteration
loss_plot(loss_list)

print('Predicted singular values from large to small:', singular_value)
print("True singular values from large to small:", D)

# Get parameters value
value = quantum_models['net_0'].weight.asnumpy()
v_value = value[:120]
u_value = value[120:]

# Calculate U and V
u_learned = u_ansatz.matrix(u_value)
v_learned = v_ansatz.matrix(v_value)

v_dagger_learned = np.conj(v_learned.T)
d_learned = np.array(singular_value)

err_subfull, err_local, err_svd = [], [], []
U, D, v_dagger = np.linalg.svd(M, full_matrices=True)

# Calculate Frobenius-norm error
for t in range(rank):
    lowrank_mat = np.matrix(U[:, :t]) * np.diag(D[:t]) * np.matrix(
        v_dagger[:t, :])
    recons_mat = np.matrix(u_learned[:, :t]) * np.diag(
        d_learned[:t]) * np.matrix(v_dagger_learned[:t, :])
    err_local.append(norm(lowrank_mat - recons_mat))
    err_subfull.append(norm(m_copy - recons_mat))
    err_svd.append(norm(m_copy - lowrank_mat))

# Plot SVD error and VQSVD error
fig, ax = pyplot.subplots()
ax.plot(list(range(1, rank + 1)),
        err_subfull,
        "o-.",
        label='Reconstruction via VQSVD')
ax.plot(list(range(1, rank + 1)),
        err_svd,
        "^--",
        label='Reconstruction via SVD')
# ax.plot(list(range(1, rank + 1)), err_local, "*--", label='SVD V/S QSVD')
pyplot.xlabel('Singular Value Used (Rank)', fontsize=14)
pyplot.ylabel('Norm Distance', fontsize=14)
leg = pyplot.legend(frameon=True)
leg.get_frame().set_edgecolor('k')
