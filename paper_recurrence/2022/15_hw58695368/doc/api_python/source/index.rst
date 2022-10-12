.. api_python documentation master file, created by
   sphinx-quickstart on Thu Oct 13 04:32:17 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

赛题十五：利用MindQuantum实现量子卷积神经网络求解量子多体问题
================================================================

`昇腾AI创新大赛2022-昇思赛道 <https://www.hiascend.com/zh/developer/contests/details/48c53c2c697c482ba464111aaabb47ce>`_

赛题十五：利用MindQuantum实现量子卷积神经网络求解量子多体问题

论文：

`An Application of Quantum Machine Learning on Quantum Correlated Systems: Quantum Convolutional Neural Network as a Classifier for Many-Body Wavefunctions from the Quantum Variational Eigensolver <https://arxiv.org/abs/2111.05076>`_

复现要求：

基于MindQuantum实现图4中的量子卷积神经网络，并在N=4、8、12的情况下实现对顺磁性和铁磁性的分类，精度要求达到90%以上

@NPark-NoEvaa

API参考
----------------------------------------------------------------

.. toctree::
   :maxdepth: 1

   src.dataset
   src.ansatz_qcnn
   src.loss
   src.qcnn

样例：
----------------------------------------------------------------

>>> import numpy as np
    import mindspore as ms
    from mindspore.dataset import NumpySlicesDataset
    from mindspore.train.callback import LossMonitor, Callback
    import matplotlib.pyplot as plt
    from src.dataset import build_dataset
    from src.qcnn import QCNNet
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
>>> class StepAcc(Callback):
        def __init__(self, model, test_x, test_y):
            self.model = model
            self.test_x = test_x
            self.test_y = test_y.flatten()
            self.acc = []
        def step_end(self, run_context):
            pred_y = self.model.predict(self.test_x)
            self.acc.append((self.test_y == pred_y).mean())
>>> N = 8
    path = './TFI_chain/closed/'
    encoder, encoder_params_name, x, y = build_dataset(N, path)
    x = x[y!=0]
    y = y[y!=0]
    x_train = np.concatenate((x[:15], x[65:]))
    y_train = np.array([-1]*15+[1]*15)
>>> batch = 30
    epoch = 50
    y_train = y_train.reshape((y_train.shape[0], -1))
    train_loader = NumpySlicesDataset({'features': x_train, 'labels': y_train}, shuffle=False).batch(batch)
>>> ms.set_seed(1202)
    model = QCNNet(N, encoder)
    monitor = LossMonitor(10)
    acc = StepAcc(model, x, y)
    callbacks=[monitor, acc]
>>> model.train(epoch, train_loader, callbacks)
>>> _, _, x, y = build_dataset(N, path, 2)
    pred_y = model.predict(x)
>>> plt.plot(np.linspace(0.2,1.8,pred_y.shape[0]), pred_y)
    plt.title('N=8(batch=30, epoch=50)')
    plt.ylabel('predicted labels')
    plt.xlabel('g')
    plt.show()
