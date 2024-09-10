import numpy as np
from scipy.sparse import csr_matrix

from .QAIA import QAIA
from multiprocessing import Pool

class LQA(QAIA):

    def __init__(
        self,
        J,
        h=None,
        x=None,
        n_iter=1000,
        batch_size=1,
        gamma=0.1,
        dt=1.0,
        momentum=0.99,
    ):
        """Construct LQA algorithm."""
        super().__init__(J, h, x, n_iter, batch_size)
        self.J = csr_matrix(self.J)
        self.gamma = gamma
        self.dt = dt
        self.momentum = momentum

        self.initialize()

    def initialize(self):
        """Initialize spin values."""
        # if self.x is None:
        #     self.x = 0.2 * (np.random.rand(self.N, self.batch_size) - 0.5)

        from scipy.stats import qmc
        # 使用Sobol序列生成器
        sampler = qmc.Sobol(d=self.N, scramble=False)
        # 生成batch_size个样本，并缩放到[-0.1, 0.1]
        points = sampler.random_base2(m=int(np.ceil(np.log2(self.batch_size))))
        points = 0.2 * points - 0.1
        self.x = points.T


        if self.x.shape[0] != self.N:
            raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

    def update(self, beta1=0.9, beta2=0.999, epsilon=10e-8):
        """
        Dynamical evolution with Adam.

        Args:
            beta1 (float): Beta1 parameter. Default: ``0.9``.
            beta2 (float): Beta2 parameter. Default: ``0.999``.
            epsilon (float): Epsilon parameter. Default: ``10e-8``.
        """
        m_dx = 0
        v_dx = 0

        for i in range(1, self.n_iter):
            t = i / self.n_iter
            tmp = np.pi / 2 * np.tanh(self.x)
            z = np.sin(tmp)
            y = np.cos(tmp)
            if self.h is None:
                dx = np.pi / 2 * (-t * self.gamma * self.J.dot(z) * y + (1 - t) * z) * (1 - np.tanh(self.x) ** 2)
            else:
                dx = (
                    np.pi
                    / 2
                    * (-t * self.gamma * (self.J.dot(z) + self.h) * y + (1 - t) * z)
                    * (1 - np.tanh(self.x) ** 2)
                )

            # momentum beta1
            m_dx = beta1 * m_dx + (1 - beta1) * dx
            # rms beta2
            v_dx = beta2 * v_dx + (1 - beta2) * dx**2
            # bias correction
            m_dx_corr = m_dx / (1 - beta1**i)
            v_dx_corr = v_dx / (1 - beta2**i)

            self.x = self.x - self.dt * m_dx_corr / (np.sqrt(v_dx_corr) + epsilon)



# class LQA(QAIA):
#     def __init__(
#         self,
#         J,
#         h=None,
#         x=None,
#         n_iter=1000,
#         batch_size=1,
#         gamma=0.1,
#         dt=1.0,
#         momentum=0.99,
#     ):
#         """Construct LQA algorithm."""
#         super().__init__(J, h, x, n_iter, batch_size)
#         self.J = csr_matrix(self.J)
#         self.gamma = gamma
#         self.dt = dt
#         self.momentum = momentum
#         self.batch_size=batch_size
#         self.initialize()

#     def initialize(self):
#         """Initialize spin values."""
#         if self.x is None:
#             self.x = 0.2 * (np.random.rand(self.N, self.batch_size) - 0.5)

#         if self.x.shape[0] != self.N:
#             raise ValueError(f"The size of x {self.x.shape[0]} is not equal to the number of spins {self.N}")

#     def compute_on_batch(self,data):

#         m_dx = 0
#         v_dx = 0

#         for i in range(1, self.n_iter):
#             t = i / self.n_iter
#             tmp = np.pi / 2 * np.tanh(data)
#             z = np.sin(tmp)
#             y = np.cos(tmp)

#             dx = (np.pi/ 2 * (-t * self.gamma * (self.J.dot(z) + self.h) * y + (1 - t) * z) * (1 - np.tanh(data) ** 2))

#             # momentum beta1
#             m_dx = self.beta1 * m_dx + (1 - self.beta1) * dx
#             # rms beta2
#             v_dx = self.beta2 * v_dx + (1 - self.beta2) * dx**2
#             # bias correction
#             m_dx_corr = m_dx / (1 - self.beta1**i)
#             v_dx_corr = v_dx / (1 - self.beta2**i)

#             data = data - self.dt * m_dx_corr / (np.sqrt(v_dx_corr) + self.epsilon)

#         return data

#     def update(self, beta1=0.9, beta2=0.999, epsilon=10e-8):

#         self.beta1=beta1
#         self.beta2=beta2
#         self.epsilon=epsilon
#         # 使用 multiprocessing 处理每个batch
#         # print(self.x.shape)

#         # results =self.compute_on_batch(self.x[:, 0].reshape(-1, 1))
#         batch=1
#         with Pool() as pool:
#             # results = pool.map(self.compute_on_batch, (self.x[:, i].reshape(-1, 1) for i in range(self.batch_size)))
#             results = pool.map(self.compute_on_batch, (self.x[:, batch*i:batch*i+batch].reshape(-1, batch) for i in range(int(self.batch_size/batch))))

#         results = np.concatenate(results, axis=1)

#         # print(results.shape)
#         self.x = results
