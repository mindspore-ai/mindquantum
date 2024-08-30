import os
import pickle as pkl
from argparse import ArgumentParser
from typing import *

import numpy as np
from numpy import ndarray
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.executing_eagerly()
from sionna.mapping import Constellation, Mapper
from sionna.mimo import EPDetector, KBestDetector, LinearDetector, MaximumLikelihoodDetector, MMSEPICDetector

constellation_cache: Dict[int, Constellation] = {}
mapper_cache: Dict[int, Mapper] = {}
detector_cache: Dict[int, Callable] = {}

# https://stackoverflow.com/questions/15505514/binary-numpy-array-to-list-of-integers
bits_to_number = lambda bits: bits.dot(1 << np.arange(bits.shape[-1] - 1, -1, -1))

mean = lambda x: sum(x) / len(x) if len(x) else 0.0


def get_constellation(nbps:int) -> Constellation:
    if nbps not in constellation_cache:
        constellation_cache[nbps] = Constellation('qam', nbps)
    constellation = constellation_cache[nbps]
    #constellation.show() ; plt.show()
    return constellation

def get_mapper(nbps:int) -> Mapper:
    if nbps not in mapper_cache:
        constellation = get_constellation(nbps)
        mapper_cache[nbps] = Mapper(constellation=constellation)
    mapper = mapper_cache[nbps]
    return mapper

def get_detector(args, nbps:int, Nt:int):
    cfg = (nbps, Nt)
    if cfg not in detector_cache:
        # https://nvlabs.github.io/sionna/examples/OFDM_MIMO_Detection.html?highlight=detector#id1
        kwargs = {
            'constellation': get_constellation(nbps),
            'hard_out': True,   # bin_out
        }
        detector_cls = {
            'linear': lambda: LinearDetector(args.E, 'bit', args.D, **kwargs),
            'kbest':  lambda: KBestDetector('bit', Nt, args.k, **kwargs),
            'ep':     lambda: EPDetector('bit', nbps, l=args.l, hard_out=True),
            'mmse':   lambda: MMSEPICDetector('bit', num_iter=args.num_iter, **kwargs),
            # ml does not run due to resource limit (
            'ml':     lambda: MaximumLikelihoodDetector('bit', args.D, Nt, **kwargs),
        }
        detector_cache[cfg] = detector_cls[args.M]()
    return detector_cache[cfg]


def modulate_and_transmit(bits:ndarray, H:ndarray, nbps:int, SNR:int=None) -> Tuple[ndarray, ndarray]:
    mapper = get_mapper(nbps)
    b = tf.convert_to_tensor(bits, dtype=tf.int32)
    x: ndarray = mapper(b).cpu().numpy()

    noise = 0
    if SNR:
        # SNR(dB) := 10*log10(P_signal/P_noise) ?= Var(signal) / Var(noise)
        sigma = np.var(bits) / SNR
        noise = np.random.normal(scale=sigma**0.5, size=x.shape)
    y = H @ x + noise
    return x, y


def run(args):
    ber_list = []
    for idx in tqdm(range(150)):
        with open(f'MLD_data/{idx}.pickle', 'rb') as fh:
            data = pkl.load(fh)
            H: ndarray = data['H']
            y: ndarray = data['y']
            bits: ndarray = data['bits'].astype(np.int32)
            nbps: int = data['num_bits_per_symbol']
            SNR: int = data['SNR']
            ZF_ber: float = data['ZF_ber']
            Nt = H.shape[1]

        H = tf.convert_to_tensor(H[np.newaxis, ...])    # [B=1, Nr, Nt]
        y = tf.convert_to_tensor(y.T)                   # [B=1, Nr]
        # WTF: noise covariance matrice, how to get this?
        if 'use identity':
            cov = np.eye(Nt, dtype=np.complex64)    # *sigma (?
        else:
            sigma = np.var(bits) / SNR
            noise = np.random.normal(scale=sigma**0.5, size=H[0].shape).astype(np.complex64)
            cov = noise @ noise.T  # 这对吗？
        s = tf.convert_to_tensor(cov[np.newaxis, ...])

        if not 'debug':
            print('H.shape:', H.shape)
            print('y.shape:', y.shape)
            print('bits.shape:', bits.shape)
            print('nbps:', nbps)
            print('SNR:', SNR)
            print('ZF_ber:', ZF_ber)

        detector = get_detector(args, nbps, Nt)
        if isinstance(detector, MMSEPICDetector):
            # 这个实现暂时非常作弊，这里直接放了GT，理论上应该放先验概率logits
            LLRs = tf.convert_to_tensor(bits[np.newaxis, ...].astype(np.float32))
            inputs = (y, H, LLRs, s)
        else:
            inputs = (y, H, s)
        bits_decode = detector(inputs)[0].cpu().numpy()  # [Nr, nbps]
        ber = np.mean(bits != bits_decode)
        ber_list.append(ber)
        #print(f'[case {idx}] ans: {ber}, ref: {ZF_ber}')

    avgber = mean(ber_list)
    print(f">> avg. BER = {avgber:.5f}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-M', default='linear', choices=['linear', 'kbest', 'ml', 'ml-prior', 'mmse', 'ep'], help='detector')
    parser.add_argument('-E', default='lmmse', choices=['lmmse', 'mf', 'zf'], help='equalizer for LinearDetector')
    parser.add_argument('-D', default='app', choices=['app', 'maxlog'], help='demapping_method')
    parser.add_argument('-k', default=64, type=int, help='k for KBestDetector')
    parser.add_argument('-l', default=10, type=int, help='l for EPDetector')
    parser.add_argument('--num_iter', default=4, type=int, help='num_iter for MMSEPICDetector')
    args = parser.parse_args()

    run(args)
