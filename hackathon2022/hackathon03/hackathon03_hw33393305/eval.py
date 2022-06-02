import numpy as np


def normal(state):
    return state / np.sqrt(np.abs(np.vdot(state, state)))


# test_y = np.load('test_y.npy', allow_pickle=True)
test_y = np.load('predict_y.npy', allow_pickle=True)
real_test_y = np.load('train_y.npy', allow_pickle=True)
acc = np.real(
    np.mean([
        np.abs(np.vdot(normal(bra), ket))
        for bra, ket in zip(test_y, real_test_y)
    ]))
print(f"Acc: {acc}")
