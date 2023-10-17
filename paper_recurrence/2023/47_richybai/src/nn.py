import numpy as np
from mindspore import dataset as ds
from mindspore import nn, ops
from mindspore.nn.loss.loss import LossBase


class HingeLoss(LossBase):
    def __init__(self, reduction='mean'):
        super(HingeLoss, self).__init__(reduction)

    def construct(self, logits, labels):
        x = 1 - logits * labels
        x[x < 0] = 0
        return self.get_loss(x)


class Network(nn.Cell):
    def __init__(self, n_inputs):
        super().__init__()
        self.dense_relu_sequential = nn.SequentialCell(
            # nn.Dense(n_inputs, 1, has_bias=False),
            nn.Dense(n_inputs, 1),
            nn.ReLU(),
            # nn.Dense(1, 1, has_bias=False)
            nn.Dense(1, 1)
        )

    def construct(self, x):
        logits = self.dense_relu_sequential(x)
        return logits


def count_params(net):
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params


class ForwardWithLoss(nn.Cell):
    """定义损失网络"""

    def __init__(self, backbone, loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(ForwardWithLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, data, label):
        """连接前向网络和损失函数"""
        out = self.backbone(data)
        return self.loss_fn(out, label)

    def backbone_network(self):
        """要封装的骨干网络"""
        return self.backbone


class TrainOneStep(nn.TrainOneStepCell):
    """定义训练流程"""

    def __init__(self, network, optimizer):
        """参数初始化"""
        super(TrainOneStep, self).__init__(network, optimizer)
        self.grad = ops.GradOperation(get_by_list=True)

    def construct(self, data, label):
        """构建训练过程"""
        weights = self.weights
        loss = self.network(data, label)
        grads = self.grad(self.network, weights)(data, label)
        return loss, self.optimizer(grads)


class LossAgent(nn.Metric):
    """定义Loss"""

    def __init__(self):
        super(LossAgent, self).__init__()
        self.clear()

    def clear(self):
        self.loss = 0
        self.epoch_num = 0

    def update(self, *loss):
        loss = loss[0].asnumpy()
        self.loss += loss
        self.epoch_num += 1

    def eval(self):
        return self.loss / self.epoch_num


class BinaryAccAgent(nn.Metric):
    """
    定义Binary 分类acc统计的类
    updata 方法输入的是[y_pred, y], y_pred 是 logits, 网络直接的输出，预期是-1 和 1 分别代表两类
    """

    def __init__(self):
        super(BinaryAccAgent, self).__init__()
        self.clear()

    def clear(self):
        self.correct_num = 0
        self.samples_num = 0
        self.confusion_matrix = np.zeros([2, 2])

    def update(self, *inputs):
        y_pred = inputs[0]
        y = inputs[1]
        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.asnumpy()
        if not isinstance(y, np.ndarray):
            y = y.asnumpy()

        y_predout = np.zeros_like(y_pred, dtype=np.int32)
        y_predout[y_pred < 0] = -1
        y_predout[y_pred > 0] =  1
        # 统计acc
        self.correct_num += np.sum(y == y_predout)
        self.samples_num += y.shape[0]
        # 统计confusion matrix
        self.confusion_matrix[0, 0] += np.sum((y_predout == y)[y == 1])
        self.confusion_matrix[1, 1] += np.sum((y_predout == y)[y == -1])
        self.confusion_matrix[0, 1] += np.sum((y_predout != y)[y == 1])
        self.confusion_matrix[1, 0] += np.sum((y_predout != y)[y == -1])



    def eval(self):
        return self.correct_num / self.samples_num

    def get_confusion_matrix(self):
        return self.confusion_matrix

