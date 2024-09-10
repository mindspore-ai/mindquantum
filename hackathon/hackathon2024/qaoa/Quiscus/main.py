from typing import Dict, Tuple

import numpy as np
from numpy import ndarray
from scipy.interpolate import interp1d

from utils.path import LOG_PATH
from utils.lookup_table import load_lookup_table, load_lookup_table_ex
from utils.lookup_table import SIM_EQ, NON_EQ

THRESHOLD = 1.0

is_ex = True
#lookup_table = load_lookup_table(LOG_PATH / 'ft-ada-decay' / 'lookup_table-iter=9400.json')
#lookup_table = load_lookup_table(LOG_PATH / 'ft-ada-decay-moment-fast' / 'lookup_table-iter=3800.json')
#lookup_table = load_lookup_table(LOG_PATH / 'ft-ada-decay-moment-fast_ft' / 'lookup_table-iter=9800.json')
lookup_table_ex = load_lookup_table_ex(LOG_PATH / 'ft-ada-moment-fast_ft-ex' / 'lookup_table-iter=12300.json')


def ave_D(Jc, nq):      # average degree
    return 2 * len(Jc) / nq

def order(Jc):          # graph order
    return max([len(key) for key in Jc.keys()])

def trans_gamma(gammas, D):
    # Eq. 10 from arXiv:2201.11785, without 1/|w|
    return gammas * np.arctan(1 / np.sqrt(D - 1)) 

def rescale_factor_arXiv_2201_11785(Jc):
    # Eq. 10 from arXiv:2201.11785, i.e. 1/Σ|w|
    keys_len = {}
    for key in Jc.keys():
        if len(key) in keys_len:
            keys_len[len(key)] += 1
        else:
            keys_len[len(key)] = 1
    norm = 0
    for key, val in Jc.items():
        norm += abs(val/keys_len[len(key)])
    return 1 / norm

def rescale_factor(Jc):
    # Eq. 87 from arXiv:2305.15201, i.e. 1/sqrt(Σw**2)
    keys_len = {}
    for key in Jc.keys():
        if len(key) in keys_len:
            keys_len[len(key)] += 1
        else:
            keys_len[len(key)] = 1
    norm = 0
    for key, val in Jc.items():
        norm += val**2/keys_len[len(key)]
    norm = np.sqrt(norm)
    return 1 / norm


def main_baseline(Jc_dict:Dict[Tuple[int], float], p:int, Nq:int=12) -> Tuple[ndarray, ndarray]:
    '''
        The main function you need to change!!!
    Args:
        Jc_dict (dict): the ising model
        p (int): the depth of qaoa circuit
    Returns:
        gammas (Union[numpy.ndarray, List[float]]): the gamma parameters, the length should be equal to depth p.
        betas (Union[numpy.ndarray, List[float]]): the beta parameters, the length should be equal to depth p.
    '''
    D = ave_D(Jc_dict, Nq)
    k = order(Jc_dict)
    import csv
    # Read the parameters of infinite size limit, and the maximum order is 6 surported here.
    k = min(k, 6)
    # const data sheet from arXiv:2110.14206 Tbl. 4 & 5
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row[0] == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3 + 2 * p:
                    gammas = np.array([float(new_row[i]) for i in range(3, 3 + p)] )
                    betas = np.array([float(new_row[i]) for i in range(3 + p, 3 + 2 * p)])
    # rescale the parameters for specific case
    gammas = trans_gamma(gammas, D)
    factor = rescale_factor(Jc_dict)
    return gammas * factor, betas

def main(Jc_dict:Dict[Tuple[int], float], p:int, Nq:int=12) -> Tuple[ndarray, ndarray]:
    global lookup_table

    D = ave_D(Jc_dict, Nq)
    k = order(Jc_dict)
    k = min(k, 6)
    if is_ex:
        vals_std = np.asarray(list(Jc_dict.values())).std()
        w = SIM_EQ if vals_std < THRESHOLD else NON_EQ
        lookup_table = lookup_table_ex[w]       # get sub table

    if p in [4, 8] and k in [2, 3, 4, 5]:       # these are directly trained
        params = lookup_table[p][k]
    else:                                       # interp for other case
        p4 = lookup_table[4][k]
        p8 = lookup_table[8][k]
        params = interp_expand(p, p4, p8)
    gammas, betas = np.split(params, 2)
    gammas = trans_gamma(gammas, D)
    factor = rescale_factor(Jc_dict) * 1.165    # 1.275, 1.165
    return gammas * factor, betas


def interp_expand(p:int, p4:ndarray, p8:ndarray) -> ndarray:
    # build model
    p4_g, p4_b = np.split(p4, 2)
    p8_g, p8_b = np.split(p8, 2)
    p4_g_interp = np.asarray([p8_g[0]] + [(x+y)/2 for x, y in zip(p4_g, p4_g[1:])])
    p4_b_interp = np.asarray([(x+y)/2 for x, y in zip(p4_b, p4_b[1:])] + [p8_b[-1]])
    p4_g_expand = np.asarray([[x, y] for x, y in zip(p4_g_interp, p4_g)]).flatten()
    p4_b_expand = np.asarray([[x, y] for x, y in zip(p4_b, p4_b_interp)]).flatten()
    p4_expand = np.concatenate([p4_g_expand, p4_b_expand], axis=0)
    fused = (p8 + p4_expand) / 2
    p_g, p_b = np.split(fused, 2)
    x_virtual = np.linspace(0.0, 1.0, len(p_g))
    func_g = interp1d(x_virtual, p_g)
    func_b = interp1d(x_virtual, p_b)

    # lerp: N points => p points
    x_sample = np.linspace(0.0, 1.0, p)
    p_s_g = func_g(x_sample)
    p_s_b = func_b(x_sample)
    p_s = np.concatenate([p_s_g, p_s_b], axis=0)

    if not 'debug':
        import matplotlib.pyplot as plt
        plt.subplot(211)
        plt.plot(p4_expand, label='p4 (ex.)')
        plt.plot(p8,        label='p8')
        plt.plot(fused,     label='fused')
        plt.legend()
        plt.subplot(212)
        plt.plot(p_s, label='p_s')
        plt.legend()
        plt.suptitle(f'interp demo: 4/8 => {p}')
        plt.show()

    return p_s


if __name__ == '__main__':
    k = 4
    p4 = lookup_table_ex[NON_EQ][4][k]
    p8 = lookup_table_ex[NON_EQ][8][k]
    params = interp_expand(6, p4, p8)
    print('params.shape:', params.shape)
    print('params:', params)
