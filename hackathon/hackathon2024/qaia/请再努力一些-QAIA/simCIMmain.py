import pickle
import numpy as np
from qaia import SimCIM
from judger import Judger
from glob import glob

'''
本赛题旨在引领选手探索新的量子启发式算法, 用于求解现代无线通信系统中的MIMO (Multiple Input Multiple Output) 检测问题.
'''


def to_ising(H, y, num_bits_per_symbol):
    '''
    Reduce MIMO detection problem into Ising problem.

    Reference
    ---------
    [1] Singh A K, Jamieson K, McMahon P L, et al. Ising machines’ dynamics and regularization for near-optimal mimo detection.
        IEEE Transactions on Wireless Communications, 2022, 21(12): 11080-11094.

    Input
    -----
    H: [Nr, Nt], np.complex
        Channel matrix

    y: [Nr, 1], np.complex
        Received signal

    num_bits_per_symbol: int
        The number of bits per constellation symbol, e.g., 4 for QAM16.


    Output
    ------
    J: [2*Nt, 2*Nt], np.float
        The coupling matrix of Ising problem

    h: [2*Nt, 1], np.float
        The external field

    '''
    # the size of constellation
    M = 2 ** num_bits_per_symbol
    Nr, Nt = H.shape
    N = 2 * Nt
    rb = int(num_bits_per_symbol / 2)
    qam_var = 1 / (2 ** (rb - 2)) * np.sum(np.linspace(1, 2 ** rb - 1, 2 ** (rb - 1)) ** 2)
    I = np.eye(N)
    T = (2 ** (rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    T = T.reshape(-1, N).T
    Nr, Nt = H.shape
    H_real = H.real
    H_imag = H.imag
    H_tilde = np.vstack([np.hstack([H_real, -H_imag]), np.hstack([H_imag, H_real])])
    y_tilde = np.concatenate([y.real, y.imag])
    # This is different from the original paper because we use normalized transmitted symbol
    z = y_tilde / np.sqrt(qam_var) - H_tilde @ T @ np.ones((N * rb, 1)) / qam_var + (
                np.sqrt(M) - 1) * H_tilde @ np.ones((N, 1)) / qam_var
    J = -2 * T.T @ H_tilde.T @ H_tilde @ T / qam_var
    diag_index = np.diag_indices_from(J)
    J[diag_index] = 0
    h = 2 * z.T @ H_tilde @ T
    return J, h.T


# 选手提供的Ising模型生成函数，可以用我们提供的to_tsing
def ising_generator(H, y, num_bits_per_symbol, snr):
    return to_ising(H, y, num_bits_per_symbol)


# 使用SimCIM算法：
def qaia_mld_solver(J, h):
    solver = SimCIM(J, h, n_iter=1000, x=None,
        batch_size=1,
        dt=0.0201,
        momentum=0.9,
        sigma=0.0508,
        pt=6.5,)
    solver.update()
    sample = np.sign(solver.x)
    energy = solver.calc_energy()
    opt_index = np.argmin(energy)
    solution = sample[:, opt_index]
    return solution



if __name__ == "__main__":
    dataset = []
    filelist = glob('MLD_data/*.pickle')
    # filelist = ['MLD_data/16x16_snr10.pickle', 'MLD_data/16x16_snr10.pickle']

    for filename in filelist:
        # 读取数据
        data = pickle.load(open(filename, 'rb'))
        dataset.append([data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR']])

    judger = Judger(dataset)
    avgber = judger.benchmark(ising_generator, qaia_mld_solver)
    # 测试选手的平均ber，越低越好
    print(f"baseline avg. BER = {avgber}")