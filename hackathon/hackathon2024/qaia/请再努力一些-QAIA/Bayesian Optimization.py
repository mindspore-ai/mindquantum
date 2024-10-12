import pickle
import numpy as np
import optuna
import time
from qaia import SimCIM
from judger import Judger
from glob import glob
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_parallel_coordinate

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

def ising_generator(H, y, num_bits_per_symbol, snr):
    return to_ising(H, y, num_bits_per_symbol)

def qaia_mld_solver(J, h, n_iter, batch_size, dt, momentum, sigma, pt, st_v, fi_v, x_sat):
    solver = SimCIM(J, h, n_iter=n_iter, batch_size=batch_size, dt=dt, momentum=momentum, sigma=sigma, pt=pt, st_v=st_v, fi_v=fi_v, x_sat=x_sat)
    solver.update()
    sample = np.sign(solver.x)
    energy = solver.calc_energy()
    opt_index = np.argmin(energy)
    solution = sample[:, opt_index]
    return solution

def objective(trial):
    # 定义搜索空间
    n_iter = trial.suggest_categorical('n_iter', [100, 200, 500, 1000])
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 5, 10])
    sigma = trial.suggest_float('sigma', 0.01, 0.1, log=True)
    dt = trial.suggest_float('dt', 0.001, 0.05, log=True)
    momentum = trial.suggest_float('momentum', 0.9, 1)
    pt = trial.suggest_float('pt', 1, 10)
    st_v = trial.suggest_float('st_v', -10, -1)
    fi_v = -st_v
    x_sat = trial.suggest_float('x_sat', 0, 10)
 # 评估参数
    def qaia_mld_solver_with_params(J, h):
        return qaia_mld_solver(J, h, n_iter, batch_size, dt, momentum, sigma, pt, st_v, fi_v, x_sat)

    avgber = judger.benchmark(ising_generator, qaia_mld_solver_with_params)
    return avgber

if __name__ == "__main__":
    start_time = time.time()  # 开始计时
    dataset = []
    filelist = glob('MLD_data/*.pickle')

    for filename in filelist:
        data = pickle.load(open(filename, 'rb'))
        dataset.append([data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR']])

    judger = Judger(dataset)

    # 使用Optuna进行参数优化
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=10)

    # 计算实验结果的统计数据
    trials_results = [trial.value for trial in study.trials]
    avg_result = np.mean(trials_results)
    min_result = np.min(trials_results)
    max_result = np.max(trials_results)

    # 输出统计数据
    print(f"Average BER over 50 trials: {avg_result}")
    print(f"Minimum BER over 50 trials: {min_result}")
    print(f"Maximum BER over 50 trials: {max_result}")

    # 输出最优参数
    print(f"Best parameters: {study.best_params}")
    print(f"Best avg. BER: {study.best_value}")

    # 结束计时并输出时间
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

    # 可视化优化历史
    fig_optimization_history = plot_optimization_history(study)
    fig_optimization_history.show()

    # 可视化参数重要性
    fig_param_importances = plot_param_importances(study)
    fig_param_importances.show()

    # 可视化并行坐标图
    fig_parallel_coordinate = plot_parallel_coordinate(study)
    fig_parallel_coordinate.show()

    # 保存图像
    fig_optimization_history.write_image("optimization_history.png")
    fig_param_importances.write_image("param_importances.png")
    fig_parallel_coordinate.write_image("parallel_coordinate.png")
