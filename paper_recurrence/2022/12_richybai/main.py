import mindspore
from src.QGAN import generator_photonic_circuit, discriminator_circuit, gene_amplitude_encoder, vector_to_angle
from src.QGAN import Generator, Discriminator, GenewithDiscrim
from src.utils import ForwardWithLoss, TrainOneStep
import mindspore as ms
from mindspore import nn
import mindspore.numpy as np

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
# ms.set_seed(100)


if __name__ == "__main__":
    n_qubits = 2
    lr = 1e-2
    epoch = 10
    # 生成相应的线路
    g = generator_photonic_circuit(num_qubits=n_qubits)
    d = discriminator_circuit(num_qubits=n_qubits, layers=2)
    encoder = gene_amplitude_encoder(num_qubits=n_qubits)
    # 把线路输入，生成相应的QMLayer
    G = Generator(g)
    D = Discriminator(encoder, d)

    loss = nn.MSELoss(reduction='mean')
    optimizerG = nn.Adam(G.trainable_params(), learning_rate=lr)
    optimizerD = nn.Adam(D.trainable_params(), learning_rate=lr)
    G_with_D = GenewithDiscrim(G, vector_to_angle, D)

    D_with_loss = ForwardWithLoss(D, loss)
    G_D_with_loss = ForwardWithLoss(G_with_D, loss)

    train_one_step_D_with_true_data = TrainOneStep(D_with_loss, optimizerD)
    train_one_step_D_with_fake_data = TrainOneStep(G_D_with_loss, optimizerD)
    train_one_step_G = TrainOneStep(G_D_with_loss, optimizerG)

    train_true = np.ones([1, 3]).astype("float32")
    train_true_label = np.ones([1, 1])
    train_fake_label = np.ones([1, 1]) * -1
    # 总的训练epoch
    for i in range(epoch):
        print(f"epoch{i+1}")
        # 每个epoch里面D训练的轮数, 增加D的准确率
        for _ in range(5):        
            train_one_step_D_with_true_data(train_true, train_true_label)
            train_one_step_D_with_fake_data(train_fake_label)
        # 每个epoch里面G训练的轮数, 增加G生成数据的能力
        for _ in range(100):
            train_one_step_G(train_true_label)
        
        print("true data logits:", D(train_true), "fake data logits:", G_with_D())
