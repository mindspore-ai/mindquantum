import pickle
import numpy as np
from qaia import QAIA
from judger import Judger
from glob import glob

def to_ising(H, y, num_bits_per_symbol,SNR):
    y = y.reshape(-1)
    Nr, Nt = H.shape
    N = 2 * Nt
    rb = num_bits_per_symbol//2
    M = 2**num_bits_per_symbol
    qam_var = 1/(2**(rb-2))*np.sum(np.linspace(1,2**rb-1, 2**(rb-1))**2)
    I = np.eye(N)
    T = (2**(rb - 1 - np.arange(rb)))[:, np.newaxis, np.newaxis] * I[np.newaxis, ...]
    T = T.reshape(-1, N).T/np.sqrt(qam_var)
    
    y_tilde = np.concatenate([y.real, y.imag])
    H_real = H.real
    H_imag = H.imag
    H_tilde = np.vstack([np.hstack([H_real, -H_imag]), np.hstack([H_imag, H_real])])
    
    H_problom = (H_tilde@T).T
    y_problom = y_tilde - np.ones((N * rb))@H_problom + ((np.sqrt(M)-1)/np.sqrt(qam_var))*H_tilde@np.ones((N))
    
    if SNR == 10:
        lam = 30
    elif SNR == 15:
        lam = 10
    elif SNR == 20:
        lam = 5
    else:
        lam = 200//SNR
        
    H_inv = np.linalg.inv(H_problom.T@H_problom+lam*np.eye(H_problom.shape[1]))
    J = H_problom@H_inv@H_problom.T
    diag_index = np.diag_indices_from(J)
    J[diag_index] = 0
    h = -2*y_problom@H_inv@H_problom.T
    return J, h

class BSB(QAIA):
    def __init__(self,J,h=None,x=None,n_iter=100,batch_size=1):
        super().__init__(J, h, x, n_iter, batch_size)
        self.delta = 0.91
        self.dt = 3.
        self.p = np.linspace(1/n_iter, 1, n_iter)
        xi = self.delta/(2*self.N**0.5*(((J*J).sum()/(self.N*(self.N-1)))**0.5))
        self.xi = xi
        self.x = x
        self.initialize()

    def initialize(self):
        if self.x is None:
            self.x = np.zeros((self.N, self.batch_size))
        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")
        self.y = np.random.randn(self.N, self.batch_size)
    
    def update(self):
        for i in range(self.n_iter):
            self.x += self.dt*self.y*self.delta
            self.x = np.clip(self.x,-1.,1.)
            self.y += (-(self.delta-self.p[i])*self.x-self.xi*(self.J.dot(self.x)+0.5*self.h))*self.dt
            cond = np.abs(self.x) > 1
            self.y = np.where(cond, np.zeros_like(self.x), self.y)

def ising_generator(H, y, num_bits_per_symbol, snr):
    return to_ising(H, y, num_bits_per_symbol, snr)
def qaia_mld_solver(J, h):
    solver = BSB(J, h, batch_size=1, n_iter=8)
    solver.update()
    solution = np.sign(np.average(solver.x,1))
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