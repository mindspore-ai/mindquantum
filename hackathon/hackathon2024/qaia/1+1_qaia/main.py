import pickle
import numpy as np
from qaia import YJC001
from judger import Judger
from glob import glob

def to_ising(H, y, num_bits_per_symbol):

    # the size of constellation
    M = 2**num_bits_per_symbol
    Nr, Nt = H.shape
    N = 2 * Nt
    rb = int(num_bits_per_symbol/2)
    qam_var = 1/(2**(rb-2))*np.sum(np.linspace(1,2**rb-1, 2**(rb-1))**2)
    I = np.eye(N)
    T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    T = T.reshape(-1, N).T
    Nr, Nt = H.shape
    H_real = H.real
    H_imag = H.imag
    H_tilde = np.vstack([np.hstack([H_real, -H_imag]), np.hstack([H_imag, H_real])])
    y_tilde = np.concatenate([y.real, y.imag])
    # This is different from the original paper because we use normalized transmitted symbol
    z = y_tilde/np.sqrt(qam_var) - H_tilde @ T @ np.ones((N * rb, 1))/qam_var + (np.sqrt(M) - 1) * H_tilde @ np.ones((N, 1))/qam_var
    J = -2*T.T@H_tilde.T @ H_tilde @ T/qam_var
    diag_index = np.diag_indices_from(J)
    J[diag_index] = 0
    h = 2 * z.T @ H_tilde @ T
    return J, h.T

# 选手提供的Ising模型生成函数，可以用我们提供的to_tsing
def ising_generator(H, y, num_bits_per_symbol, snr):
    return to_ising(H, y, num_bits_per_symbol)

# 选手提供的qaia MLD求解器，用mindquantum.algorithms.qaia
def qaia_mld_solver(J, h):

    param= {  
            "dt": 0.4,
            "gamma": 0.05,
            "beta1": 0.6,
            "n_iter": 20,
    }

    if (J.shape[0]<=300):    
        param['dt']=0.4979   
        param['gamma']=0.03461
        param['beta1']=0.6445
        param['n_iter']=20

    if (J.shape[0]<=400)&(J.shape[0]>300):
        param['dt']=0.4319
        param['gamma']=0.09259
        param['beta1']=0.6479
        param['n_iter']=20

    if (J.shape[0]<=600)&(J.shape[0]>400):
        param['dt']=0.2262
        param['gamma']=0.03922
        param['beta1']=0.6819
        param['n_iter']=20

    if (J.shape[0]<=800)&(J.shape[0]>600):
        param['dt']=0.2703
        param['gamma']=0.05115
        param['beta1']=0.4481
        param['n_iter']=10

    if (J.shape[0]>800):
        param['dt']=0.3930
        param['gamma']=0.02999
        param['beta1']=0.4595
        param['n_iter']=10

    solver = YJC001(J, h, batch_size=1, n_iter=param['n_iter'],dt=param['dt'],gamma=param['gamma'])   
    solver.update(beta1=param['beta1'])
    sample = np.sign(solver.x)
    energy = solver.calc_energy()
    opt_index = np.argmin(energy)
    solution = sample[:, opt_index]
    return solution


if __name__ == "__main__":
    dataset = []
    filelist = glob('MLD_data/*.pickle')

    for filename in filelist:
        # 读取数据
        data = pickle.load(open(filename, 'rb'))
        dataset.append([data['H'], data['y'], data['bits'], data['num_bits_per_symbol'], data['SNR']])

    import time
    start_time = time.time()  # 记录开始时间

    judger = Judger(dataset)
    avgber = judger.benchmark(ising_generator, qaia_mld_solver)
    # 测试选手的平均ber，越低越好
    print(f"baseline avg. BER = {avgber}")

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过时间
    print("Elapsed time: {:.2f} seconds".format(elapsed_time))