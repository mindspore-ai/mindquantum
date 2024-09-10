import numpy as np

from qaia.SB import SB
class DUSB(SB):
    r"""
    Deep Unfolded Simulated Bifurcation

    Reference: `Deep Unfolded Simulated Bifurcation for Massive
    MIMO Signal Detection <https://arxiv.org/pdf/2306.16264>`_.

    Args:
        J (Union[numpy.array, csr_matrix]): The coupling matrix with shape :math:`(N x N)`.
        h (numpy.array): The external field with shape :math:`(N, )`.
        x (numpy.array): The initialized spin value with shape :math:`(N x batch_size)`. Default: ``None``.
        n_iter (int): The number of iterations. Default: ``1000``.
        batch_size (int): The number of sampling. Default: ``1``.
        dt (float): The step size. Default: ``1``.
        xi (float): positive constant with the dimension of frequency. Default: ``None``.
    """

    # pylint: disable=too-many-arguments
    def __init__(
            self,
            J,
            h=None,
            x=None,
            n_iter=1000,
            batch_size=1,
            dt=1,
            eta=1,
            xi=None,
    ):
        """Construct DUSB algorithm."""
        super().__init__(J, h, x, n_iter, batch_size, dt, xi)
        # 这里我们多使用了一个参数$eta$做控制
        self.eta = eta
        self.initialize()

    # pylint: disable=attribute-defined-outside-init
    def update(self):
        """Differentiable update rule using back-propagation"""
        for i in range(self.n_iter):
            if self.h is None:
                self.y += (-(self.delta - self.p[i]) * self.x + self.eta * self.xi * self.J.dot(self.x)) * self.dt
            else:
                self.y += (-(self.delta - self.p[i]) * self.x + self.eta * self.xi * (self.J.dot(self.x) + self.h)) * self.dt
            self.x += self.dt * self.y * self.delta

            # 我们将边界条件改成可微形式
            a = self.x.copy()
            self.x = (a + 1) / (1 + np.exp(-10 * (a + 1))) - (a - 1) / (1 + np.exp(-10 * (a - 1))) - 1
            self.y = self.y * (1 - 1 / (1 + np.exp(-100 * (np.abs(a) - 1))))

