import numpy as np
import math
import mindspore as ms
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from mindquantum.core.gates import RX, RY, RZ
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.simulator import Simulator
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.transforms import Compose
from mindspore import nn, ops

seed = 99
np.random.seed(seed)
ds.config.set_seed(seed)
qubits = 5
depth = 4
gate = 3
img_size = 16
batch_size = 16
epoch = 1
lr_g = 0.7
lr_d = 0.01
ite = 400
DATA_DIR_MNIST = "./mnist/"
fixed_noise = ms.Tensor(np.random.uniform(0, 1, (1, qubits)) * math.pi / 2, dtype=ms.float32)

def show_img(img_list):
    for i, im in enumerate(img_list):
        ax = plt.subplot(1, len(img_list), i+1)
        plt.axis('off')
        ax.imshow(im.squeeze(0), cmap = 'gray')
        plt.show()

def generage_base_i_hamiltonian(i, n_qubit):
    """
    construct |0><i|
    """
    row = np.array([0])
    col = np.array([i])
    data = np.array([1.0])
    coo = coo_matrix((data, (row, col)), shape=(2**n_qubit, 2**n_qubit))
    return Hamiltonian(coo.tocsr())

def load_data(img_size, dir, batch_size):
    dataset_mnist = ds.MnistDataset(dir, usage="train", shuffle=True)
    dataset_mnist = dataset_mnist.filter(predicate=lambda label: label == 0, input_columns=["label"])
    transforms_list = [vision.Resize(size=img_size), vision.CenterCrop(img_size), vision.ToTensor()]
    dataset_pre = dataset_mnist.map(operations=Compose(transforms_list))
    data_set = dataset_pre.batch(batch_size, drop_remainder=True)
    return data_set

def quantum_circuit():
    encoder = Circuit()
    for i in range(qubits):
        encoder += RY(f'noise{i}').on(i)
    encoder = encoder.no_grad()
    ansatz = Circuit()
    for i in range(depth):
        for k in range(gate):
            for j in range(qubits):
                if k == 0:
                    ansatz += RX(f'weights{i}{j}{k}').on(j)
                elif k == 1:
                    ansatz += RZ(f'weights{i}{j}{k}').on(j)
                elif k == 2:
                    if j < 4:
                        ansatz += RX(f'weights{i}{j}{k}').on(j + 1, j)
                    else:
                        ansatz += RX(f'weights{i}{j}{k}').on(0, j)
    circuit = encoder.as_encoder() + ansatz.as_ansatz()
    circuit.summary()
    return circuit

def quantum_measure(qubit):
    hams = [generage_base_i_hamiltonian(i, qubit) for i in range(2 ** qubit)]
    print(hams)
    return hams

def quantum_layer(qubits):
    circuit = quantum_circuit()
    circ_l = Circuit()
    hams = quantum_measure(qubits)
    sim = Simulator('mqvector', circuit.n_qubits)
    sim_l = Simulator('mqvector', qubits)
    grad_ops = sim.get_expectation_with_grad(hams, circuit, circ_l, sim_l, parallel_worker=4)

    QuantumNet = MQLayer(grad_ops,
                         weight=ms.Tensor(np.random.uniform(-np.pi, np.pi, len(circuit.ansatz_params_name)),
                         dtype=ms.dtype.float32))
    return QuantumNet

class QuantumGenerator(nn.Cell):
    def __init__(self):
        super(QuantumGenerator, self).__init__()
        self.quantumlayer = quantum_layer(qubits)
        self.linear = nn.Dense(2**qubits, img_size * img_size, weight_init='uniform')
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def construct(self, x):
        x = self.quantumlayer(x)
        x = x/(abs(x).max())
        x = self.linear(x)
        out = self.sigmoid(x)
        return out

class Discriminator(nn.Cell):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.linear1 = nn.Dense(img_size * img_size, 64, weight_init='uniform')
        self.linear2 = nn.Dense(64, 32, weight_init='uniform')
        self.linear3 = nn.Dense(32, 1, weight_init='uniform')
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.sigmoid(x)
        return x

class LossG(nn.Cell):
    def __init__(self, netD, netG, loss_fn):
        super(LossG, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.loss_fn = loss_fn
    def construct(self, latent_code):
        fake_data = self.netG(latent_code)
        out = self.netD(fake_data)
        label_real = ops.OnesLike()(out)
        loss = self.loss_fn(out, label_real)
        return loss

class LossD(nn.Cell):
    def __init__(self, netD, netG, loss_fn):
        super(LossD, self).__init__(auto_prefix=True)
        self.netD = netD
        self.netG = netG
        self.loss_fn = loss_fn
    def construct(self, real_data, latent_code):
        out_real = self.netD(real_data)
        label_real = ops.OnesLike()(out_real)
        loss_real = self.loss_fn(out_real, label_real)
        fake_data = self.netG(latent_code)
        fake_data = ops.stop_gradient(fake_data)
        out_fake = self.netD(fake_data)
        label_fake = ops.ZerosLike()(out_fake)
        loss_fake = self.loss_fn(out_fake, label_fake)
        return loss_real + loss_fake

class QGAN(nn.Cell):
    def __init__(self, myTrainOneStepCellForD, myTrainOneStepCellForG):
        super(QGAN, self).__init__(auto_prefix=True)
        self.myTrainOneStepCellForD = myTrainOneStepCellForD
        self.myTrainOneStepCellForG = myTrainOneStepCellForG
    def construct(self, real_data, latent_code):
        output_D = self.myTrainOneStepCellForD(real_data, latent_code).view(-1)
        netD_loss = output_D.mean()
        output_G = self.myTrainOneStepCellForG(latent_code).view(-1)
        netG_loss = output_G.mean()
        return netD_loss, netG_loss

dataset = load_data(img_size, DATA_DIR_MNIST, batch_size)
size = dataset.get_dataset_size()
data_loader = dataset.create_dict_iterator(output_numpy=True, num_epochs=epoch)
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
netD = Discriminator()
netG = QuantumGenerator()
loss = nn.BCELoss()
loss = nn.BCELoss()
optimizerD = nn.SGD(netD.trainable_params(), learning_rate=lr_d)
optimizerG = nn.SGD(netG.trainable_params(), learning_rate=lr_g)

netD_with_criterion, netG_with_criterion = LossD(netD, netG, loss), LossG(netD, netG, loss)
myTrainOneStepCellForD = nn.TrainOneStepCell(netD_with_criterion, optimizerD)
myTrainOneStepCellForG = nn.TrainOneStepCell(netG_with_criterion, optimizerG)
qcgan = QGAN(myTrainOneStepCellForD, myTrainOneStepCellForG)
qcgan.set_train()
G_losses, D_losses, loss_real, loss_fake, image_list= [], [], [], [], []

print("Starting Training Loop...")
for e in range(epoch):
    for i, d in enumerate(data_loader):
        noise = ms.Tensor(np.random.uniform(-1, 1, (batch_size, qubits)) * math.pi / 2, dtype=ms.float32)
        real_data = ms.Tensor(d['image']).reshape(-1, img_size * img_size)
        netD_loss, netG_loss = qcgan(real_data, noise)
        if i % 1 == 0 or i == size - 1:
            print('[%2d/%d][%3d/%d]   Loss_D:%7.4f  Loss_G:%7.4f' % (
                e + 1, epoch, i + 1, size, netD_loss.asnumpy(), netG_loss.asnumpy()))
            D_losses.append(netD_loss.asnumpy())
            G_losses.append(netG_loss.asnumpy())
            fake = netG(fixed_noise)
            p_real = netD(real_data).mean().squeeze().asnumpy()
            p_fake = netD(fake).mean().squeeze().asnumpy()
            loss_real.append(p_real)
            loss_fake.append(p_fake)
        if i % 50 == 0:
            img = netG(fixed_noise)
            for j in range(img.shape[1]):
                if img[0][j] <0.08:
                    img[0][j] = 0.0
            img = netG(fixed_noise).reshape(1, 1, img_size, img_size)
            img = img.reshape(1, 1, img_size, img_size)
            image_list.append(img.transpose(0, 2, 3, 1).asnumpy())
    # ms.save_checkpoint(netG, "Generator.ckpt")
    # ms.save_checkpoint(netD, "Discriminator.ckpt")
show_img(image_list)