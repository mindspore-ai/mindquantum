import pickle
from time import time
from glob import glob

import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_ber(solution:ndarray, bits:ndarray) -> float:
    '''
    Compute BER for the solution from QAIAs.

    Firstly, both the solution from QAIAs and generated bits should be transformed into gray-coded, 
    and then compute the ber.

    Reference
    ---------
    [1] Kim M, Venturelli D, Jamieson K. Leveraging quantum annealing for large MIMO processing in centralized radio access networks. 
        Proceedings of the ACM special interest group on data communication. 2019: 241-255.\
    
    Input
    -----
    solution: [rb*2*Nt, ], np.int
        The binary array filled with ones and minus ones.

    bits: [Nt, nbps], np.int
        The binary array filled with ones and zeros.
    Ouput
    -----
    ber: np.float
        A scalar, the BER.
    '''
    solution = solution.astype(np.int32)
    bits = bits.astype(np.int32)

    # convert the bits from sionna style to constellation style
    # Sionna QAM16 map: https://nvlabs.github.io/sionna/examples/Hello_World.html
    '''
    [sionna-style]
        1011 1001 0001 0011
        1010 1000 0000 0010
        1110 1100 0100 0110
        1111 1101 0101 0111
    [constellation-style] i.e. the "gray code" in QuAMax paper
        0010 0110 1110 1010
        0011 0111 1111 1011
        0001 0101 1101 1001
        0000 0100 1100 1000
    '''
    bits_constellation = 1 - np.concatenate([bits[..., 0::2], bits[..., 1::2]], axis=-1)

    # Fig. 2 from arXiv:2001.04014, the QuAMax paper converting QuAMax to gray coded
    num_bits_per_symbol = bits_constellation.shape[1]
    rb = num_bits_per_symbol // 2
    bits_hat = solution.reshape(rb, 2, -1)  # [rb, c=2, Nt]
    bits_hat = np.concatenate([bits_hat[:, 0], bits_hat[:, 1]], axis=0) # [2*rb, Nt]
    bits_hat = bits_hat.T.copy()            # [Nt, 2*rb]
    bits_hat[bits_hat == -1] = 0            # convert Ising {-1, 1} to QUBO {0, 1}
    # QuAMax => intermediate code
    '''
    [QuAMax-style]
        0011 0111 1011 1111
        0010 0110 1010 1110
        0001 0101 1001 1101
        0000 0100 1000 1100
    [intermediate-style]
        0011 0100 1011 1100
        0010 0101 1010 1101
        0001 0110 1001 1110
        0000 0111 1000 1111
    '''
    output_bit = bits_hat.copy()                        # copy b[0]
    index = np.nonzero(bits_hat[:, rb-1] == 1)[0]       # select even columns
    bits_hat[index, rb:] = 1 - bits_hat[index, rb:]     # invert bits of high part (flip upside-down)
    # Differential bit encoding, intermediate code => gray code (constellation-style)
    for i in range(1, num_bits_per_symbol):             # b[i] = b[i] ^ b[i-1]
        output_bit[:, i] = np.logical_xor(bits_hat[:, i], bits_hat[:, i - 1])
    # calc BER
    ber = np.mean(bits_constellation != output_bit)
    return ber


class Judger:

    def __init__(self, test_cases):
        self.test_cases = test_cases

    @staticmethod
    def infer(ising_generator, qaia_mld_solver, H, y, num_bits_per_symbol, snr):
        J, h = ising_generator(H, y, num_bits_per_symbol, snr)
        bits = qaia_mld_solver(J, h)
        return bits

    def benchmark(self, ising_gen, qaia_mld_solver):
        from collections import defaultdict
        ber_list = []
        ZF_ber_list = []
        avgber_per_Nt = defaultdict(list)
        avgber_per_snr = defaultdict(list)
        avgber_per_nbps = defaultdict(list)

        avgber = 0
        for i, case in enumerate(tqdm(self.test_cases)):
            H, y, bits_truth, num_bits_per_symbol, snr, ZF_ber = case
            bits_decode = self.infer(ising_gen, qaia_mld_solver, H, y, num_bits_per_symbol, snr)
            ber = compute_ber(bits_decode, bits_truth)
            avgber += ber
            print(f'[case {i}] ber: {ber}, ref_ber: {ZF_ber}')

            ber_list.append(ber)
            ZF_ber_list.append(ZF_ber)
            avgber_per_Nt[H.shape[1]].append(ber)
            avgber_per_snr[snr].append(ber)
            avgber_per_nbps[num_bits_per_symbol].append(ber)

        print('>> avgber_per_Nt:')
        for Nt in sorted(avgber_per_Nt):
            print(f'  {Nt}: {np.asarray(avgber_per_Nt[Nt]).mean()}')
        print('>> avgber_per_snr:')
        for snr in sorted(avgber_per_snr):
            print(f'  {snr}: {np.asarray(avgber_per_snr[snr]).mean()}')
        print('>> avgber_per_nbps:')
        for nbps in sorted(avgber_per_nbps):
            print(f'  {nbps}: {np.asarray(avgber_per_nbps[nbps]).mean()}')

        if 'plot':
            from pathlib import Path
            BASE_PATH = Path(__file__).parent
            LOG_PATH = BASE_PATH / 'log' ; LOG_PATH.mkdir(exist_ok=True)
            pairs = list(zip(ZF_ber_list, ber_list))
            pairs.sort(reverse=True)        # decrease order by ZF_ber
            ber_list = [ber for ZF_ber, ber in pairs]
            ZF_ber_list = [ZF_ber for ZF_ber, ber in pairs]
            plt.plot(ber_list,    label='ours')
            plt.plot(ZF_ber_list, label='ZF')
            plt.ylim(0, 0.55)
            plt.legend()
            plt.suptitle('BER')
            plt.tight_layout()
            plt.savefig(LOG_PATH / 'solut.png', dpi=400)
            plt.close()

        avgber /= len(self.test_cases)
        return avgber


if __name__ == "__main__":
    from main import ising_generator, qaia_mld_solver

    dataset = []
    filelist = glob(f'MLD_data/*.pickle')
    # filelist = ['MLD_data/16x16_snr10.pickle']
    for filename in filelist:
        with open(filename, 'rb') as fh:
            data = pickle.load(fh)
        dataset.append([data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR'], data['ZF_ber']])

    # 测试选手的平均ber，越低越好
    ref_ts = 234.09     # BSB baseline (B=100, n_iter=100)
    judger = Judger(dataset)
    t = time()
    avgber = judger.benchmark(ising_generator, qaia_mld_solver)
    ts = time() - t
    print(f'>> time cost: {ts:.2f}')
    print(f">> avg. BER = {avgber:.5f}")
    print(f'>> score:', (1 - avgber) * (ref_ts / ts))
