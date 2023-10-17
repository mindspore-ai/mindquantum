from mindspore import nn, ops


class ForwardWithLoss(nn.Cell):
    """定义损失网络"""

    def __init__(self, backbone, loss_fn):
        """实例化时传入前向网络和损失函数作为参数"""
        super(ForwardWithLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = loss_fn

    def construct(self, *inputs):
        """连接前向网络和损失函数"""
        if len(inputs) == 2:
            data, label = inputs
            out = self.backbone(data)
            return self.loss_fn(out, label)
        if len(inputs) == 1:
            label = inputs[0]
            out = self.backbone()
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

    def construct(self, *inputs):
        """构建训练过程"""
        weights = self.weights
        if len(inputs) == 2:
            data, label = inputs
            loss = self.network(data, label)
            grads = self.grad(self.network, weights)(data, label)
            return loss, self.optimizer(grads)
        if len(inputs) == 1:
            label = inputs[0]
            loss = self.network(label)
            grads = self.grad(self.network, weights)(label)
            return loss, self.optimizer(grads)