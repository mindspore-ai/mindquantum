#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/05/03

# 理解数据是如何造出来的: y = H @ x + n
# - 只要噪声 n 能被完全估计出来，那么 ZF 方法就是完全可行的 (但这不可能)
# - 只要噪声 n 的估计是错误的，ZF 会导致很大的误差积累

import pickle as pkl
from pathlib import Path

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt

from run_baseline import modulate_and_transmit, bits_to_number

BASE_PATH = Path(__file__).parent
IMG_PATH = BASE_PATH / 'img' ; IMG_PATH.mkdir(exist_ok=True)


def test_data_gen(idx:int):
    fp = f'MLD_data/{idx}.pickle'
    with open(fp, 'rb') as fh:
        data = pkl.load(fh)
        H: ndarray = data['H']
        y: ndarray = data['y']
        bits: ndarray = data['bits'].astype(np.int32)
        nbps: int = data['num_bits_per_symbol']
        SNR: int = data['SNR']
        ZF_ber: float = data['ZF_ber']

    print('H.shape:', H.shape)
    print('y.shape:', y.shape)
    print('bits.shape:', bits.shape)
    print('nbps:', nbps)
    print('SNR:', SNR)
    print('ZF_ber:', ZF_ber)

    color = bits_to_number(bits)
    x, y_hat = modulate_and_transmit(bits, H, nbps, SNR)

    # 0. perfect recon, when noise is known (impossible)
    #x_hat = np.linalg.inv(H) @ (y_hat - noise)
    # 1. ZF-method (seemingly ok in cases)
    x_hat = np.linalg.inv(H) @ y_hat
    # 2. ZF-method with resampled noise (even also seemingly ok in cases)
    #noise2 = np.random.normal(scale=sigma**0.5, size=x.shape)
    #x_hat = np.linalg.inv(H) @ (y_hat - noise2)
    # 3. ZF-method with GT data (not ok in almost all cases, because y=y_hat+noise3, the noise will be amplified by H')
    #noise2 = np.random.normal(scale=sigma**0.5, size=x.shape)
    #x_hat = np.linalg.inv(H) @ (y - noise2)

    plt.subplot(221) ; plt.scatter(x.real,     x.imag,     c=color, cmap='Spectral') ; plt.title('x = QAM(bits)')
    plt.subplot(222) ; plt.scatter(y_hat.real, y_hat.imag, c=color, cmap='Spectral') ; plt.title('y_hat = H @ x + noise')
    plt.subplot(223) ; plt.scatter(x_hat.real, x_hat.imag, c=color, cmap='Spectral') ; plt.title('x_hat = inv(H) @ y_hat')
    plt.subplot(224) ; plt.scatter(y.real,     y.imag,     c=color, cmap='Spectral') ; plt.title('y (GT)')
    plt.suptitle(f'id={idx} Ht={H.shape[1]} SNR={SNR} nbps={nbps}')
    save_dp = IMG_PATH / 'H_channels' ; save_dp.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_dp / f'{idx}.png', dpi=400)
    plt.close()

    return np.std(x_hat - x), ZF_ber


if __name__ == '__main__':
    var_x, ZF_ber = [], []
    for i in range(150):
        v, ref = test_data_gen(i)
        var_x.append(v)
        ZF_ber.append(ref)

    plt.subplot(211) ; plt.plot(var_x)  ; plt.title('var(x_hat - x)')
    plt.subplot(212) ; plt.plot(ZF_ber) ; plt.title('ZF_ber')
    plt.suptitle('var_ref')
    plt.tight_layout()
    plt.savefig(IMG_PATH / 'var_ref.png', dpi=400)
    plt.close()
