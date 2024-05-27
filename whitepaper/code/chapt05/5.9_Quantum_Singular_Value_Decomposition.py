import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import norm
from matplotlib import pyplot as plt
import tqdm
import mindspore as ms
from mindquantum import Simulator, MQAnsatzOnlyLayer, add_prefix
from mindquantum import Hamiltonian, Circuit, RY, RZ, X

# Setting the required constants and weights
n_qubits = 3  # qubits number
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


# Matrix generator function
def mat_generator():
    """
    Generate a random complex matrix
    """
    matrix = np.random.randint(10, size=(N, N)) + 1j * np.random.randint(
        10, size=(N, N)
    )
    return matrix


# Generate matrix M which will be decomposed
M = mat_generator()
# m_copy is generated for error analysis
m_copy = np.copy(M)
print("Random matrix M is: ")
print(M)
# Get SVD results
U, D, v_dagger = np.linalg.svd(M, full_matrices=True)


# Ansatz class
class Ansatz:
    def __init__(self, n, depth):
        self.circ = Circuit()
        num = 0
        for _ in range(depth):
            for i in range(n):
                self.circ += RY("theta" + str(num)).on(i)
                num += 1
            for i in range(n):
                self.circ += RZ("theta" + str(num)).on(i)
                num += 1
            for i in range(n - 1):
                self.circ += X.on(i + 1, i)
            self.circ += X.on(0, n - 1)


# Quantum Network definition
def quantnet(qubits_num, hams, circ_right, circ_left=None, base=None):
    sim = Simulator("mqvector", qubits_num)
    if base is None:
        pass
    else:
        sim.set_qs(base)
    grad_ops = sim.get_expectation_with_grad(hams, circ_right, circ_left)

    quantumnet = MQAnsatzOnlyLayer(grad_ops, "ones")
    return quantumnet


# Sparring the decomposed 8 x 8 matrix M and generating the corresponding Hamiltonian H
u_ansatz = add_prefix(Ansatz(n_qubits, cir_depth).circ, "u")
v_ansatz = add_prefix(Ansatz(n_qubits, cir_depth).circ, "v")
ham = Hamiltonian(csr_matrix(M))
i_matrix = np.identity(N)
quantum_models = dict()
quantum_models["net_0"] = quantnet(n_qubits, ham, v_ansatz, u_ansatz, i_matrix[0])
for s in range(1, rank):
    quantum_models["net_" + str(s)] = quantnet(
        n_qubits, ham, v_ansatz, u_ansatz, i_matrix[s]
    )
    quantum_models["net_" + str(s)].weight = quantum_models["net_0"].weight


# Hybrid Quantum-Classical Network definition
class MyNet(ms.nn.Cell):
    """
    define quantum-classic net
    """

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


# Instantiate the hybrid quantum-classical network and start training using MindSpore
net = MyNet()
# Define optimizer
opt = ms.nn.Adam(net.trainable_params(), learning_rate=LR)
# Simple gradient descent
train_net = ms.nn.TrainOneStepCell(net, opt)
# Start training
loss_list = list()
for itr in tqdm.tqdm(range(ITR)):
    res = train_net()
    loss_list.append(res.asnumpy().tolist())

import matplotlib.pyplot as plt

# Finish training and read the training results
singular_value = list()
for _, qnet in quantum_models.items():
    singular_value.append(qnet().asnumpy()[0])

# Displaying results
print("Predicted singular values from large to small:", singular_value)
print("True singular values from large to small:", D)

# Plotting loss curve
plt.plot(loss_list)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()
