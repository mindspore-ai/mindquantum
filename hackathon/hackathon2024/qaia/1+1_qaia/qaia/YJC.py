import numpy as np
from .QAIA import QAIA
from scipy.stats import qmc
from scipy.sparse import csr_matrix

class YJC001(QAIA):

    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        gamma=0.1,
        dt=1.0,
    ):
        """Construct YJC algorithm."""
        super().__init__(J, h, x, n_iter, batch_size)

        self.J = J
        self.gamma = gamma
        self.dt = dt
        self.initialize()

    def initialize(self):
        """Initialize spin values."""

        # if self.x is None:
        #     self.x = 0.2 * (np.random.rand(self.N, self.batch_size))- 0.1

        # # 使用Sobol序列生成器
        sampler = qmc.Sobol(d=self.N, scramble=False)
        # 生成batch_size个样本，并缩放到[-0.1, 0.1]
        self.x = (0.2 * sampler.random_base2(m=int(np.ceil(np.log2(self.batch_size))))-0.1).T

    def update(self, beta1=0.9, beta2=0.999, epsilon=10e-8):
        m_dx = np.zeros_like(self.x)
        v_dx = np.zeros_like(self.x)
        # 将参数传递给 JIT 优化的函数
        self.x = run_update(self.x,self.J,self.h, self.dt,m_dx,v_dx, self.gamma,self.n_iter,beta1, beta2=0.999, epsilon=10e-8)
   
def run_update(x, J, h, dt, m_dx, v_dx, gamma, n_iter, beta1, beta2=0.999, epsilon=1e-8):

    pi_half = np.pi / 2
    beta1_inv = 1 - beta1
    beta2_inv = 1 - beta2
    
    for i in range(1, n_iter):  
        t = i / n_iter
        tanh_x = np.tanh(x)
        cos_tmp = np.cos(pi_half * tanh_x)
        sin_tmp = np.sin(pi_half * tanh_x)

        dx = pi_half * (-t * gamma * (J.dot(sin_tmp) + h) * cos_tmp + (1 - t) * sin_tmp) * (1 - tanh_x ** 2)

        m_dx = beta1 * m_dx + beta1_inv * dx
        v_dx = beta2 * v_dx + beta2_inv * dx ** 2

        m_dx_corr = m_dx / (1 - beta1 ** i)
        v_dx_corr = v_dx / (1 - beta2 ** i)

        x -= dt * m_dx_corr / (np.sqrt(v_dx_corr) + epsilon)

    return x