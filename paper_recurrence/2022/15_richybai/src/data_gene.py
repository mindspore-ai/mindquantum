import numpy as np


def linear_interpolation(sample0, sample1, n_term):
    """
    输入两个sample，格式为sample = [gamma[float], param[dict]]。
    n是插值点的个数，包含原来的两个点。
    输出两个列表，gammas，params，里面包含了n个样本相应的数据
    """
    gammas = []
    params = []
    lambdas = np.linspace(0, 1, n_term)
    for l in lambdas[1: -1]:
        gamma = sample0[0] + l * (sample1[0] - sample0[0])
        gammas.append(gamma)
        param = {}
        for theta in sample0[1]:
            param[theta] = sample0[1][theta] + l * (sample1[1][theta] - sample0[1][theta])
        params.append(param)
    return gammas, params


if __name__ == "__main__":
    for n_qubits in [4, 8, 12]:
        n_term = 13 # 插值项数，包括了起点和终点
        gammas, params = np.load(f"../data/{n_qubits}qbsdata.npy", allow_pickle=True)
        gammas = gammas.tolist()
        params = params.tolist()

        print(f"total data number: {len(params)}")
        print(f"params in one sample: {len(params[0])}")
        additional_gammas = []
        additional_params = []
        additional_gammas += gammas
        additional_params += params
        for i in range(len(gammas)-1):
            gamma, param = linear_interpolation([gammas[i], params[i]], [gammas[i+1], params[i+1]], n_term)
            additional_gammas += gamma
            additional_params += param

        data = list(zip(additional_gammas, additional_params))
        print("after augmentation sample number: ", len(data))
        data = sorted(data, key=lambda x: x[0])
        data = list(zip(*data))
        np.save(f"../data/{n_qubits}additional_data.npy", data)
        print(f"saved into ../data/{n_qubits}additional_data.npy")