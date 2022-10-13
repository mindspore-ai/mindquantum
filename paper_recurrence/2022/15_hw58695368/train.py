# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
import mindspore as ms
from mindspore.dataset import NumpySlicesDataset
from mindspore.train.callback import LossMonitor, Callback
from sklearn.model_selection import train_test_split
from src.dataset import build_dataset
from src.qcnn import QCNNet
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
def _train(data_dir_path, N, epoch, batch, seed):
    """Train."""
    encoder, encoder_params_name, x, y = build_dataset(N, data_dir_path, 5)
    print(f'N = {N}')
    print(f'epoch = {epoch}')
    print(f'batch = {batch}')
    print(f'seed = {seed}')
    ms.set_seed(seed)
    model = QCNNet(N, encoder)
    y = y.reshape((y.shape[0], -1))
    X_train, X_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        shuffle=True)
    train_loader = NumpySlicesDataset({'features': X_train, 'labels': y_train}, shuffle=False).batch(batch)
    monitor = LossMonitor(5)
    callbacks=[monitor]
    model.train(epoch, train_loader, callbacks)
    pred_y = model.predict(X_test)
    print('test acc:', (y_test.flatten() == pred_y.flatten()).mean())
    model.export_trained_parameters(f'model_N{N}.ckpt')
def train():
    """Train."""
    nlist = [(4, 5, 60),
             (8, 4, 30),
             (12, 1, 30)]
    path = './TFI_chain/closed/'
    for n in nlist:
        _train(path, *n, 1202)
    print('Finished')

if __name__ == '__main__':
    train()
