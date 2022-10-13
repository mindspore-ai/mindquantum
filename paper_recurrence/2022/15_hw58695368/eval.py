# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
import numpy as np
import mindspore as ms
from mindspore.dataset import NumpySlicesDataset
from src.dataset import build_dataset
from src.qcnn import QCNNet
ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1202)
def _test(N, data_dir_path):
    """Test."""
    encoder, encoder_params_name, x, y = build_dataset(N, data_dir_path)
    print(f'N = {N}')
    model = QCNNet(N, encoder)
    model.load_trained_parameters(f'model_N{N}.ckpt')
    y = y.reshape((y.shape[0], -1))
    predict = model.predict(x)
    acc = np.mean(y.flatten() == predict)
    print(f"Acc: {acc}")
def test():
    """Test."""
    nlist = [4, 8, 12]
    path = './TFI_chain/closed/'
    for n in nlist:
        _test(n, path)

if __name__ == '__main__':
    test()
