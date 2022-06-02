from mindspore.nn.loss.loss import LossBase
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.ops as ops
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel

class MyLoss(LossBase):

    def __init__(self, reduction='mean'):
        super(MyLoss, self).__init__(reduction)
        self.abs = ops.Abs()

    def construct(self, logits, labels):
        x = self.abs(logits - labels)
        output = self.get_loss(x)
        return output