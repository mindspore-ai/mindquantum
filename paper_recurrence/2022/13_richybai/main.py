from src.config import n_train, P, T, epochs, lstm_layers
from src.metaqaoa import gene_random_instance, gene_qaoa_layers, MetaQAOA, MetaQAOALoss, fig7_instance1, fig7_instance2
from src.utils import ForwardWithLoss, TrainOneStep
import numpy as np
from mindspore import Tensor, ops, nn, Parameter
import mindspore as ms
import os
import matplotlib.pyplot as plt 
os.environ["OMP_NUM_THREADS"] = "1"
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(2022)

if __name__ == "__main__":

    # g = gene_random_instance(n_train)
    g = fig7_instance1()
    # g = fig7_instance2()
    print("number of nodes:", len(g.nodes))
    print("number of deges:", len(g.edges))

    # 相当于是输入模型的数据
    qnn = gene_qaoa_layers(g, P)

    metaqaoa = MetaQAOA(T=T, QNN=qnn, lstm_layers=lstm_layers)
    loss_fn = MetaQAOALoss()
    optimizer = nn.Adam(metaqaoa.trainable_params(), learning_rate=0.001)


    forward_with_loss = ForwardWithLoss(metaqaoa, loss_fn)
    train_one_step = TrainOneStep(forward_with_loss, optimizer)

    theta = Tensor(np.ones([1, P, 2]).astype(np.float32))
    h = Tensor(np.zeros([lstm_layers, 1, 2]).astype(np.float32))
    c = Tensor(np.zeros([lstm_layers, 1, 2]).astype(np.float32))
    meta_loss = []
    qnn_expectation = []
    for epoch in range(epochs):
        # batch_size, seq_len, input_size
        # num_directions * num_layers, batch_size, hidden_size

        loss = train_one_step(theta, h, c)

        loss = forward_with_loss(theta, h, c)
        theta, h, c, _ = metaqaoa(theta, h, c)
        expectation = qnn(theta.reshape(1, -1))
        if (epoch+1) % 100 == 0:
            print("epoch: ", epoch+1,"metaqaoa loss: ", loss, "qnn expectation: ", expectation.squeeze())
        meta_loss.append(loss.asnumpy().squeeze())
        qnn_expectation.append(expectation.asnumpy().squeeze())

    plt.plot(range(1, epochs+1), meta_loss)
    plt.xlabel('epoch')
    plt.ylabel('meta loss')
    plt.title('meta loss')
    plt.savefig('meta loss.png')

    plt.clf()
    plt.plot(range(1, epochs+1), qnn_expectation)
    plt.xlabel('epoch')
    plt.ylabel('qnn expectation')
    plt.title('qnn expectation')
    plt.savefig('expectation.png')

    np.save("loss.npy", [meta_loss, qnn_expectation])