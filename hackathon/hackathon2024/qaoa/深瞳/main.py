import numpy as np


def main(Jc_dict, p, Nq=14):
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
    with open('utils/transfer_data.csv', 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if (row[0]) == str(k):
                new_row = [item for item in row if item != '']
                length = len(new_row)
                if length == 3 + 2 * p:
                    gammas = np.array([float(new_row[i]) for i in range(3, 3 + p)])
                    betas = np.array([float(new_row[i]) for i in range(3 + p, 3 + 2 * p)])
    # rescale the parameters for specific case
    factor, s = rescale_factor(Jc_dict)
    gammas = trans_gamma(p, gammas, D, s, k)
    return gammas * factor, betas


def trans_gamma(p, gammas, D, s, k):
    d = D - 1
    # print(f'2阶项数量{s}')
    triangles = np.floor(s / 3 - 12)
    factor0 = []
    if triangles > 0:
        if k == 2:
            for e in range(p):
                factor0.append(1 / (np.sqrt(d) - (e + 7) * 0.1))
            # print('a')
        if k == 3:
            for e in range(p):
                if p > 4:
                    factor0.append(1 / (np.sqrt(d) - (e + 16) * 0.1))
                else:
                    factor0.append(1 / (np.sqrt(d) - (e + 13) * 0.1))
                # factor0.append(1 / (np.sqrt(d) - (e + 16) * 0.1))
            # print('c')
        if k == 4:
            for e in range(p):
                if p > 4:
                    factor0.append(1 / (np.sqrt(d) - (e + 18) * 0.1))
                else:
                    factor0.append(1 / (np.sqrt(d + 6)))
                # factor0.append(1 / (np.sqrt(d) - (e + 1) * 0.1))    跳变由k >= p导致？
            # print('e')
        if k >= 5:
            factor0 = 1 / (np.sqrt(d) - 1)
    else:
        if k == 2:
            for e in range(p):
                if p > 4:
                    factor0.append(1 / (np.sqrt(d) - (e - 1) * 0.1))
                else:
                    factor0.append(1 / (np.sqrt(d) - (e + 1) * 0.1))
                # factor0.append(1 / (np.sqrt(d) - e * 0.1))
            # print('b')
        if k == 3:
            for e in range(p):
                if p > 4:
                    factor0.append(1 / (np.sqrt(d) - (e + 7) * 0.1))
                else:
                    factor0.append(1 / (np.sqrt(d) - (e + 10) * 0.1))
                # factor0.append(1 / (np.sqrt(d) - (e + 9) * 0.1))
            # print('d')
        if k == 4:
            for e in range(p):
                if p > 4:
                    factor0.append(1 / (np.sqrt(d) - (e + 21) * 0.1))
                else:
                    factor0.append(1 / (np.sqrt(d) - (e + 22) * 0.1))
                # factor0.append(1 / (np.sqrt(d) - (e + 21) * 0.1))
            # print('f')
        if k >= 5:
            factor0 = 1 / (np.sqrt(d) - 1)
    return gammas * factor0


def rescale_factor(Jc):
    '''
    Get the rescale factor, a technique from arXiv:2305.15201v1
    '''
    import copy
    Jc_dict = copy.deepcopy(Jc)
    keys_len = {}
    s = 0
    for key in Jc_dict.keys():
        if len(key) == 2:
            s += 1
        if len(key) in keys_len:
            keys_len[len(key)] += 1
        else:
            keys_len[len(key)] = 1
    norm = 0
    for key, val in Jc_dict.items():
        norm += val ** 2 / keys_len[len(key)]
    norm = np.sqrt(norm)
    return 1 / norm, s


def ave_D(Jc, nq):
    return 2 * len(Jc) / nq


def order(Jc):
    return max([len(key) for key in Jc.keys()])
