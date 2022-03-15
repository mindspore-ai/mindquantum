'''
This file is to present VQSVD using in compressing picture
'''
import os

import mindspore as ms
from mindquantum import Simulator, MQAnsatzOnlyLayer, add_prefix
from mindquantum import Hamiltonian, Circuit, RY, X

import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot
import tqdm
from PIL import Image

os.environ['OMP_NUM_THREADS'] = '1'

# Open figure MNIST_32.jpg and get matrix form
img = Image.open(r'.\\figure\\MNIST_32.png')
imgmat = np.array(list(img.getdata(band=0)), float)
imgmat.shape = (img.size[1], img.size[0])
imgmat = np.matrix(imgmat) / 255

# Get SVD results and show it
U, sigma, V = np.linalg.svd(imgmat)

for t in range(5, 16, 5):
    reconstimg = np.matrix(U[:, :t]) * np.diag(sigma[:t]) * np.matrix(V[:t, :])
    pyplot.imshow(reconstimg, cmap='gray')
    title = "n = %s" % t
    pyplot.title(title)
    pyplot.show()

# VQSVD results
# Set super parameters
n_qubits = 5  # qbits number
cir_depth = 40  # circuit depth
N = 2**n_qubits
rank = 8  # learning rank
step = 2
ITR = 200  # iterations
LR = 0.02  # learning rate
SEED = 14  # random seed

# Set equal learning weights
if step == 0:
    weight = ms.Tensor(np.ones(rank))
else:
    weight = ms.Tensor(np.arange(rank * step, 0, -step))


def mat_generator(image):
    '''
    Generate matrix by input image
    '''
    img_matrix = np.array(list(image.getdata(band=0)), float)
    img_matrix.shape = (image.size[1], image.size[0])
    img_np = np.matrix(img_matrix)
    return img_np


# Generate matrix M which will be decomposed
M = mat_generator(img)

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

            for i in range(n - 1):
                self.circ += X.on(i + 1, i)


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

# Get parameters value
value = quantum_models['net_0'].weight.asnumpy()
v_value = value[:200]
u_value = value[200:]

# Calculate U and V
u_learned = u_ansatz.matrix(u_value)
v_learned = v_ansatz.matrix(v_value)

v_dagger_learned = np.conj(v_learned.T)
d_learned = np.array(singular_value)

# Calculate recombined matrix mat
mat = np.matrix(u_learned[:, :rank]) * np.diag(d_learned[:rank]) * np.matrix(
    v_dagger_learned[:rank, :])

# Show recombination result
reconstimg = np.abs(mat)
pyplot.imshow(reconstimg, cmap='gray')
