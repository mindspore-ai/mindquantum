import os
import numpy as np
import tensorflow as tf
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

# 设置 NumPy 打印选项
np.set_printoptions(threshold=np.inf)  # threshold=np.inf 表示不省略任何内容


class QAIA:
    r"""
    The base class of QAIA.

    This class contains the basic and common functions of all the algorithms.

    Note:
        The solver is implemented using TensorFlow and can be run on either CPU or GPU.
        The input data `J` and `coeff` define the structure and weights of the energy function.

    Args:
        J (two-dimensional ragged tensor): The coupling relationship include the external field.
        coeff (numpy.array): The coefficients corresponding to each term in `J`.
        x (numpy.array): The initialized spin value with shape (batch_size x N).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``100``.
        batch_size (int): The number of sampling. Default: ``1``.
        backend (str): Computation backend to use: 'CPU','GPU'. Default: ``'CPU'``.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, J, coeff, x=None, n_iter=100, batch_size=1, backend='CPU'):
        """Construct a QAIA algorithm."""
        valid_backends = {'CPU', 'GPU'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        if x is not None:
            if not isinstance(x, np.ndarray):
                raise TypeError(f"x requires numpy.array, but get {type(x)}")
            if len(x.shape) != 2:
                raise ValueError(f"x must be a 2D array, but got shape {x.shape}")
            if x.shape[0] != J.shape[0] or x.shape[1] != batch_size:
                raise ValueError(f"x must have shape ({J.shape[0]}, {batch_size}), but got {x.shape}")

        J_high_order = [row for row in J if len(row) != 1]
        self.N = sum(1 for row in J if len(row) == 1)
        self.coeff_high_order = tf.ragged.constant(coeff[:sum(1 for row in J if len(row) != 1)], dtype=np.float32)
        self.coeff_one_order = tf.ragged.constant(coeff[-self.N:], dtype=np.float32)
        self.x = x
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.backend = backend

        high_order_term = []
        for index, sublist in enumerate(J_high_order):
            for num in sublist:
                high_order_term.append([index, num])

        high_order_term = np.array(high_order_term)

        rows, cols = high_order_term[:, 0], high_order_term[:, 1]
        rows = rows.astype(np.int64)
        cols = cols.astype(np.int64)
        self.J_ragged = tf.RaggedTensor.from_value_rowids(cols, rows)

        sortids = tf.argsort(cols)
        cols = tf.gather(cols, sortids)
        rows = tf.gather(rows, sortids)
        self.J_col_ragged = tf.RaggedTensor.from_value_rowids(rows, cols)

    def calc_energy(self, x=None):
        r"""
        Calculate energy.

        Args:
            x (numpy.array): The spin value with shape (batch_size x N).
                If ``None``, the initial spin will be used. Default: ``None``.
        """

        if self.backend in ["CPU", 'GPU']:
            if x is None:
                sign = tf.sign(self.x)
            else:
                sign = tf.sign(x)

            # 计算高阶项的多比特乘法：σi×σj×σk×σl
            poly_term = tf.reduce_prod(tf.gather(sign, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
            # 计算总能量
            energy = tf.reduce_sum(poly_term * self.coeff_high_order[None, :], axis=-1) + tf.reduce_sum(
                sign * self.coeff_one_order[None, :], axis=-1)
            return -energy

# class ASB(QAIA):
#     r"""
#         Adiabatic SB algorithm.
#
#         Reference: `Combinatorial optimization by simulating adiabatic bifurcations in nonlinear
#         Hamiltonian systems <https://www.science.org/doi/10.1126/sciadv.aav2372>`_.
#     Note:
#
#
#     Args:
#         J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
#         h (numpy.array): The external field with shape (N, ).
#         x (numpy.array): The initialized spin value with shape (N x batch_size).
#             Will be modified during optimization. If not provided (``None``), will be initialized as
#             random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
#         n_iter (int): The number of iterations. Default: ``1000``.
#         batch_size (int): The number of sampling. Default: ``1``.
#         dt (float): The step size. Default: ``1``.
#         xi (float): positive constant with the dimension of frequency. Default: ``None``.
#         backend (str): Computation backend and precision to use: 'CPU','GPU'. Default: ``'CPU'``.
#
#     Examples:
#          import numpy as np
#
#          J = tf.ragged.constant([[0, 1, 2, 3], [4, 5], [0], [1], [2], [3], [4], [5]])
#          coeff = np.array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
#          solver = DSB(J, coeff, batch_size=5, backend='CPU')
#          slover.initialize()
#          print(solver.update())
#     """
#
#     def __init__(
#         self,
#         J,
#         coeff,
#         x=None,
#         n_iter=10,
#         batch_size=1,
#         dt=1,
#         xi=0.5,
#         delta=1,
#         M=2,
#         backend='CPU',
#     ):
#         """Construct ASB algorithm."""
#         valid_backends = {'CPU', 'GPU'}
#         if not isinstance(backend, str):
#             raise TypeError(f"backend requires a string, but get {type(backend)}")
#         if backend not in valid_backends:
#             raise ValueError(f"backend must be one of {valid_backends}")
#
#         if backend in ['CPU','GPU']:
#             QAIA.__init__(self,J, coeff, x, n_iter, batch_size, backend)
#             self.xi = xi
#             self.dt = dt
#             self.K = 1
#             self.M = M
#             self.delta = delta
#             self.dm = self.dt / self.M
#             self.p = np.linspace(0, 1, self.n_iter)
#
#     def initialize(self):
#         """Randomly initialize spin values."""
#         if self.x is None:
#             if self.backend == "CPU":
#                 self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
#                 self.y = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
#             elif self.backend == "GPU":
#                 self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
#                 self.y = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
#
#     def update(self):
#         """Dynamical evolution based on Modified explicit symplectic Euler method."""
#         if self.backend == 'CPU':
#             for i in range(self.n_iter):
#                 for _ in range(self.M):
#                     self.x += self.dm * self.y * self.delta
#                     self.y -= (self.K * self.x ** 3 + (self.delta - self.p[i]) * self.x) * self.dm
#
#                     poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
#                     poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
#                     grad_val = poly_sum_bit / (self.x + 1e-9) + self.coeff_one_order[None, :]
#                     self.y += self.xi * grad_val * self.dt
#
#             return ((self.x - 1) / -2)
#         elif self.backend == 'GPU':
#             for i in range(self.n_iter):
#                 for _ in range(self.M):
#                     self.x += self.dm * self.y * self.delta
#                     self.y -= (self.K * self.x ** 3 + (self.delta - self.p[i]) * self.x) * self.dm
#
#                     poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
#                     poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
#                     grad_val = poly_sum_bit / (self.x + 1e-9) + self.coeff_one_order[None, :]
#                     self.y += self.xi * grad_val * self.dt
#
#             return ((self.x - 1) / -2)



class BSB(QAIA):
    r"""
        Ballistic SB algorithm.

        Reference: `High-performance combinatorial optimization based on classical
        mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_.

    Note:


    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'CPU','GPU'. Default: ``'CPU'``.

    Examples:
         import numpy as np

         J = tf.ragged.constant([[0, 1, 2, 3], [4, 5], [0], [1], [2], [3], [4], [5]])
         coeff = np.array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
         solver = DSB(J, coeff, batch_size=5, backend='CPU')
         slover.initialize()
         print(solver.update())
    """

    def __init__(
        self,
        J,
        coeff,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=1,
        xi=0.5,
        delta=1,
        backend='CPU',
    ):
        """Construct BSB algorithm."""
        valid_backends = {'CPU', 'GPU'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        if backend in ['CPU','GPU']:
            QAIA.__init__(self,J, coeff, x, n_iter, batch_size, backend)
            self.xi = xi
            self.dt = dt
            self.delta = delta
            self.p = np.linspace(0, 1, self.n_iter)

    def initialize(self):
        """Randomly initialize spin values."""
        if self.x is None:
            if self.backend == "CPU":
                with tf.device('/CPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.y = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
            elif self.backend == "GPU":
                with tf.device('/CPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.y = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))

    def update(self):
        if self.backend == 'CPU':
            with tf.device('/CPU:0'):
                for i in range(self.n_iter):

                    poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
                    grad_val = poly_sum_bit / self.x + self.coeff_one_order[None, :]

                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * grad_val) * self.dt
                    self.x += self.y * self.dt * self.delta

                    cond = np.abs(self.x) > 1
                    self.x = tf.where(cond, tf.sign(self.x), self.x)
                    self.y = tf.where(cond, tf.zeros_like(self.x), self.y)
                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return ((self.opt_x - 1) / -2)
        elif self.backend == 'GPU':
            with tf.device('/GPU:0'):
                for i in range(self.n_iter):
                    poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
                    grad_val = poly_sum_bit / self.x + self.coeff_one_order[None, :]

                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * grad_val) * self.dt
                    self.x += self.y * self.dt * self.delta

                    cond = np.abs(self.x) > 1
                    self.x = tf.where(cond, tf.sign(self.x), self.x)
                    self.y = tf.where(cond, tf.zeros_like(self.x), self.y)
                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return ((self.opt_x - 1) / -2)



class DSB(QAIA):
    r"""
    Discrete SB algorithm.

    Reference: `High-performance combinatorial optimization based on classical
    mechanics <https://www.science.org/doi/10.1126/sciadv.abe7953>`_.

    Note:


    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'CPU','GPU'. Default: ``'CPU'``.

    Examples:
         import numpy as np

         J = tf.ragged.constant([[0, 1, 2, 3], [4, 5], [0], [1], [2], [3], [4], [5]])
         coeff = np.array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
         solver = DSB(J, coeff, batch_size=5, backend='CPU')
         slover.initialize()
         print(solver.update())
    """

    def __init__(
        self,
        J,
        coeff,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=1,
        xi=0.5,
        delta=1,
        backend='CPU',
    ):
        """Construct DSB algorithm."""
        valid_backends = {'CPU', 'GPU'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        if backend in ['CPU','GPU']:
            QAIA.__init__(self,J, coeff, x, n_iter, batch_size, backend)
            self.xi = xi
            self.dt = dt
            self.delta = delta
            self.p = np.linspace(0, 1, self.n_iter)

    def initialize(self):
        """Randomly initialize spin values."""
        if self.x is None:
            if self.backend == "CPU":
                with tf.device('/CPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.y = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
            elif self.backend == "GPU":
                with tf.device('/GPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.y = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))

    def update(self):
        """Dynamical evolution based on Modified explicit symplectic Euler method."""
        if self.backend == 'CPU':
            with tf.device('/CPU:0'):
                for i in range(self.n_iter):

                    x_sign = tf.sign(self.x)

                    poly_term = tf.reduce_prod(tf.gather(x_sign, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
                    grad_val = poly_sum_bit / x_sign + self.coeff_one_order[None, :]

                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * grad_val) * self.dt
                    self.x += self.y * self.dt * self.delta

                    cond = np.abs(self.x) > 1
                    self.x = tf.where(cond, tf.sign(self.x), self.x)
                    self.y = tf.where(cond, tf.zeros_like(self.x), self.y)
                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return ((self.opt_x - 1) / -2)
        elif self.backend == 'GPU':
            with tf.device('/GPU:0'):
                for i in range(self.n_iter):

                    x_sign = tf.sign(self.x)

                    poly_term = tf.reduce_prod(tf.gather(x_sign, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
                    grad_val = poly_sum_bit / x_sign + self.coeff_one_order[None, :]

                    self.y += (-(self.delta - self.p[i]) * self.x + self.xi * grad_val) * self.dt
                    self.x += self.y * self.dt * self.delta

                    cond = np.abs(self.x) > 1
                    self.x = tf.where(cond, tf.sign(self.x), self.x)
                    self.y = tf.where(cond, tf.zeros_like(self.x), self.y)
                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return ((self.opt_x - 1) / -2)


class CAC(QAIA):
    r"""
        Coherent Ising Machine with chaotic amplitude control algorithm.

        Reference: `Coherent Ising machines with optical error correction
        circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_.
    Note:


    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'CPU','GPU'. Default: ``'CPU'``.

    Examples:
         import numpy as np

         J = tf.ragged.constant([[0, 1, 2, 3], [4, 5], [0], [1], [2], [3], [4], [5]])
         coeff = np.array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
         solver = DSB(J, coeff, batch_size=5, backend='CPU')
         slover.initialize()
         print(solver.update())
    """

    def __init__(
        self,
        J,
        coeff,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=0.075,
        xi=0.5,
        backend='CPU',
    ):
        """Construct CAC algorithm."""
        valid_backends = {'CPU', 'GPU'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        if backend in ['CPU','GPU']:
            QAIA.__init__(self,J, coeff, x, n_iter, batch_size, backend)
            self.xi = xi
            self.dt = dt
            self.Tr = int(0.9 * self.n_iter)
            self.Tp = self.n_iter - self.Tr
            self.p = np.hstack([np.linspace(-0.5, 1, self.Tr), np.ones(self.Tp)])
            self.alpha = np.hstack([np.linspace(1, 3, self.Tr), 3.0 * np.ones(self.Tp)])
            self.beta = 0.3

    def initialize(self):
        """Randomly initialize spin values."""
        if self.x is None:
            if self.backend == "CPU":
                with tf.device('/CPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.e = tf.ones((self.batch_size, self.N))
            elif self.backend == "GPU":
                with tf.device('/CPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.e = tf.ones((self.batch_size, self.N))

    def update(self):
        if self.backend == 'CPU':
            with tf.device('/CPU:0'):
                for i in range(self.n_iter):

                    poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
                    grad_val = poly_sum_bit / self.x + self.coeff_one_order[None, :]

                    self.x = self.x + (-self.x ** 3 + (self.p[i] - 1) * self.x + self.xi * self.e * grad_val) * self.dt
                    self.e = self.e + (-self.beta * self.e * (self.x ** 2 - self.alpha[i])) * self.dt
                    cond = np.abs(self.x) > 1.5 * np.sqrt(self.alpha[i])
                    self.x = tf.where(cond, 1.5 * tf.sign(self.x) * np.sqrt(self.alpha[i]), self.x)

                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return tf.sign((self.opt_x - 1) / -2)
        elif self.backend == 'GPU':
            with tf.device('/GPU:0'):
                for i in range(self.n_iter):

                    poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0),axis=-1)
                    grad_val = poly_sum_bit / self.x + self.coeff_one_order[None, :]

                    self.x = self.x + (-self.x ** 3 + (self.p[i] - 1) * self.x + self.xi * self.e * grad_val) * self.dt
                    self.e = self.e + (-self.beta * self.e * (self.x ** 2 - self.alpha[i])) * self.dt
                    cond = np.abs(self.x) > 1.5 * np.sqrt(self.alpha[i])
                    self.x = tf.where(cond, 1.5 * tf.sign(self.x) * np.sqrt(self.alpha[i]), self.x)
                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return tf.sign((self.opt_x - 1) / -2)

class CFC(QAIA):
    r"""
        Coherent Ising Machine with chaotic feedback control algorithm.

        Reference: `Coherent Ising machines with optical error correction
        circuits <https://onlinelibrary.wiley.com/doi/full/10.1002/qute.202100077>`_.
    Note:


    Args:
        J (Union[numpy.array, scipy.sparse.spmatrix]): The coupling matrix with shape (N x N).
        h (numpy.array): The external field with shape (N, ).
        x (numpy.array): The initialized spin value with shape (N x batch_size).
            Will be modified during optimization. If not provided (``None``), will be initialized as
            random values uniformly distributed in [-0.01, 0.01]. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
        backend (str): Computation backend and precision to use: 'CPU','GPU'. Default: ``'CPU'``.

    Examples:
         import numpy as np

         J = tf.ragged.constant([[0, 1, 2, 3], [4, 5], [0], [1], [2], [3], [4], [5]])
         coeff = np.array(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)
         solver = DSB(J, coeff, batch_size=5, backend='CPU')
         slover.initialize()
         print(solver.update())
    """

    def __init__(
        self,
        J,
        coeff,
        x=None,
        n_iter=1000,
        batch_size=1,
        dt=0.075,
        xi=0.5,
        backend='CPU',
    ):
        """Construct CFC algorithm."""
        valid_backends = {'CPU', 'GPU'}
        if not isinstance(backend, str):
            raise TypeError(f"backend requires a string, but get {type(backend)}")
        if backend not in valid_backends:
            raise ValueError(f"backend must be one of {valid_backends}")

        if backend in ['CPU','GPU']:
            QAIA.__init__(self,J, coeff, x, n_iter, batch_size, backend)
            self.xi = xi
            self.dt = dt
            self.Tr = int(0.9 * self.n_iter)
            self.Tp = self.n_iter - self.Tr
            self.p = np.hstack([np.linspace(-0.5, 1, self.Tr), np.ones(self.Tp)])
            self.alpha = 1.0
            self.beta = 0.15

    def initialize(self):
        """Randomly initialize spin values."""
        if self.x is None:
            if self.backend == "CPU":
                with tf.device('/CPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.e = tf.ones((self.batch_size, self.N))
            elif self.backend == "GPU":
                with tf.device('/GPU:0'):
                    self.x = 0.01 * tf.random.normal(shape=(self.batch_size, self.N))
                    self.e = tf.ones((self.batch_size, self.N))

    def update(self):
        if self.backend == 'CPU':
            with tf.device('/CPU:0'):
                for i in range(self.n_iter):

                    poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0), axis=-1)
                    grad_val = poly_sum_bit / self.x + self.coeff_one_order[None, :]

                    self.x = self.x + (-self.x ** 3 + (self.p[i] - 1) * self.x + self.xi * self.e * grad_val) * self.dt
                    self.e = self.e + (-self.beta * self.e * ((self.xi * self.e * grad_val) ** 2 - self.alpha)) * self.dt

                    cond = tf.abs(self.x) > 1.5
                    self.x = tf.where(cond, 1.5 * tf.sign(self.x), self.x)
                    self.e = tf.where(self.e < 0.01, 0.01, self.e)

                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return tf.sign((self.opt_x - 1) / -2)
        elif self.backend == 'GPU':
            with tf.device('/GPU:0'):
                for i in range(self.n_iter):

                    poly_term = tf.reduce_prod(tf.gather(self.x, self.J_ragged, axis=-1, batch_dims=0), axis=-1)
                    poly_sum_bit = tf.reduce_sum(
                        tf.gather(poly_term * self.coeff_high_order[None, :], self.J_col_ragged, axis=-1, batch_dims=0),
                        axis=-1)
                    grad_val = poly_sum_bit / self.x + self.coeff_one_order[None, :]

                    self.x = self.x + (-self.x ** 3 + (self.p[i] - 1) * self.x + self.xi * self.e * grad_val) * self.dt
                    self.e = self.e + (-self.beta * self.e * ((self.xi * self.e * grad_val) ** 2 - self.alpha)) * self.dt

                    cond = tf.abs(self.x) > 1.5
                    self.x = tf.where(cond, 1.5 * tf.sign(self.x), self.x)
                    self.e = tf.where(self.e < 0.01, 0.01, self.e)
                energy = self.calc_energy(self.x)
                opt_ind = tf.argmin(energy)
                # 能量最低的解opt_x，大小为batch_size（副本数）*64
                self.opt_x = self.x[opt_ind]
                return tf.sign((self.opt_x - 1) / -2)

