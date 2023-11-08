import mindspore as ms
import numpy as np
from mindspore import nn
from typing import List
from .loss import data_loss


class DenseODENet(nn.Cell):
    def __init__(self, depth: int, max_dt: float, h_dim: int, init_range: List[float],  dtype = ms.float32):
        super().__init__()
        self.depth = depth
        self.max_dt = max_dt
        self.h_dim = h_dim
        if init_range[1] <= init_range[0]:
            raise ValueError('latter item of range should be larger than the former one.')
        self.init_range = init_range
        self.dtype = dtype
        self._init_w()

    def _init_w(self):
        mask = ms.numpy.zeros((self.depth + 1, self.depth + 1), dtype=self.dtype)
        frozen_w = ms.numpy.zeros((self.depth + 1, self.depth + 1), dtype=self.dtype)
        # identity mapping
        frozen_w[0, self.depth] = 1
        for des in range(1, self.depth + 1):
            mask[des, des] = 1
            # all train:
            if des < self.depth:
                mask[0, des] = 1

        for src in range(1, self.depth):
            for des in range(src + 1, self.depth + 1):
                mask[src, des] = 1
        self.frozen_w = frozen_w
        self.mask = mask
        trainable_w = ms.numpy.rand((self.depth + 1, self.depth + 1), dtype=self.dtype)
        trainable_w = trainable_w * (self.init_range[1] - self.init_range[0]) + self.init_range[0]
        self.trainable_w = ms.Parameter(trainable_w)
        return

    def dense_w(self):
        return self.trainable_w * self.mask + self.frozen_w

    def _extend_h(self, H):
        r"""
        H: (2, h_dim, h_dim)
        return: extended -iH: (2 * h_dim, 2 * h_dim)
        """
        assert H.shape == (2, self.h_dim, self.h_dim)
        expand_h = ms.numpy.zeros((2 * self.h_dim, 2 * self.h_dim))
        h_real, h_img = H[0], H[1]
        ih_real = h_img
        ih_img = - h_real
        expand_h[: self.h_dim, : self.h_dim] = ih_real
        expand_h[: self.h_dim, self.h_dim: 2 * self.h_dim] = - ih_img
        expand_h[self.h_dim: 2 * self.h_dim, : self.h_dim] = ih_img
        expand_h[self.h_dim: 2 * self.h_dim, self.h_dim: 2 * self.h_dim] = ih_real
        return expand_h

    def step(self, extend_H, S, dt):
        assert dt <= self.max_dt
        S = ms.numpy.swapaxes(S, 0, 1)
        hidden_states = [S, ]
        output = S
        dense_weight = self.dense_w()
        for depth_idx in range(1, self.depth + 1):
            hidden = ms.ops.matmul(extend_H, output)
            hidden_states.append(hidden)
            output = dense_weight[0, depth_idx] * hidden_states[0]
            for src in range(1, depth_idx + 1):
                output = output + dense_weight[src, depth_idx] * hidden_states[src] * dt
        output = ms.numpy.swapaxes(output, 0, 1)
        return output

    def construct(self, H, S0, T):
        r"""
        Args:
            H: (2, h_dim, h_dim); H_real and H_img
            S0: (batch_size, h_dim + h_dim); S_real and S_img
            T: evolution time
        """
        if isinstance(T, ms.Tensor):
            T = T.asnumpy()
        steps = int(np.ceil(T / self.max_dt))
        dt = T / steps
        H = self._extend_h(H)
        S = S0
        for idx in range(steps):
            S = self.step(H, S, dt)
        return S


class DenseODENetWithLoss:
    def __init__(self,
                 ode_net: DenseODENet):
        self.ode_net = ode_net

    def get_loss(self, H, batch_trajectories, t_points):
        r"""
        (dim, dim); (batch_size, t_len + 1, dim + dim); (t_len + 1, )
        """
        current_data_loss = 0
        batch_former = batch_trajectories[:, 0]
        for t_idx in range(len(t_points) - 1):
            T = t_points[t_idx + 1] - t_points[t_idx]
            batch_latter = self.ode_net.construct(H=H, S0=batch_former, T=T)
            current_data_loss += data_loss(predict=batch_latter, label=batch_trajectories[:, t_idx + 1])
            batch_former = batch_latter
        return current_data_loss






# aa = DenseODENet(depth=4, max_dt=0.1, h_dim=5, init_range=[0.01, 0.02])
# print(aa.dense_weight.value())
#
# bb = ms.numpy.ones((3, 3))
# cc = ms.numpy.ones(3, )
# dd = ms.ops.matmul(bb, cc)
# print(dd)