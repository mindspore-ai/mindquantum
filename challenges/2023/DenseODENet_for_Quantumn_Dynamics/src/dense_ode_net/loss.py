import mindspore as ms


def data_loss(predict, label):
    r"""
        MSE loss of (predict, label) at a certain step.
        :param predict: (batch_size, dim + dim)
        :param label: (batch_size, dim + dim)
        :return:
        """
    loss = ms.ops.mean(ms.ops.square(predict - label))
    return loss
