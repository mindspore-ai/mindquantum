# -*- coding: utf-8 -*-
"""
[1] https://www.mindspore.cn/docs/zh-CN/r1.7/api_python/nn/mindspore.nn.SoftMarginLoss.html#mindspore.nn.SoftMarginLoss

@NoEvaa
"""
from mindspore.nn import LossBase
from mindspore.common.tensor import Tensor
from mindspore.nn.loss.loss import LossBase, _check_is_tensor
import mindspore.ops as P

class MySoftMarginLoss(LossBase):
    """
    """
    def __init__(self, reduction='mean'):
        super(MySoftMarginLoss, self).__init__()
        self.exp = P.Exp()
        self.log = P.Log()

    def construct(self, logits, labels):
        _check_is_tensor('logits', logits, self.cls_name)
        _check_is_tensor('labels', labels, self.cls_name)
        x = logits.flatten()
        y = labels.flatten()
        return self.log(1 + self.exp(- y * x)).mean()
