# -*- coding: utf-8 -*-
"""
@NoEvaa
"""
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from src.mbe_loss import MBELoss
class MBEOps(nn.Cell):
    """
    MBE operator.
    """
    def __init__(self, loss):
        super(MBEOps, self).__init__()
        self.loss = loss
    def extend_repr(self):
        return ''
    def construct(self, x):
        x = x.asnumpy()
        f, g = self.loss.get_loss(x, True)
        f = ms.Tensor(np.real(f[0]), dtype=ms.float32)
        self.g = np.real(g[0])
        return f
    def bprop(self, x, out, dout):
        dout = dout.asnumpy()
        grad = dout @ self.g
        return ms.Tensor(grad, dtype=ms.float32)
class MBELayer(nn.Cell):
    """
    MBE trainable layer.
    The parameters of ansatz circuit are trainable parameters.
    """
    def __init__(self, loss, weight='normal'):
        super(MBELayer, self).__init__()
        if not isinstance(loss, MBELoss):
            raise ValueError('Improper operation!')
        self.evolution = MBEOps(loss)
        weight_size = len(loss.circ.params_name)
        if isinstance(weight, ms.Tensor):
            if weight.ndim != 1 or weight.shape[0] != weight_size:
                raise ValueError(f"Weight init shape error, required ({weight_size}, ), but get {weight.shape}.")
        self.weight = Parameter(initializer(weight, weight_size, dtype=ms.float32), name='ansatz_weight')
    def construct(self):
        return self.evolution(self.weight)
