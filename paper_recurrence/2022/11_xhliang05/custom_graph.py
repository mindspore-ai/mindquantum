import numpy as np
import networkx as nx
import mindspore as ms
from mindquantum import *
from mindspore import nn, ops, Tensor

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")
ms.set_seed(1)
np.random.seed(1)


# 量子网络的基本层结构
def QLayer(qubit_num=4, prefix='0'):
    circ_ = Circuit()
    for qubit in range(qubit_num):
        circ_ += RY(f'0_{qubit}').on(qubit)
    for qubit in range(0, qubit_num - 1, 2):
        circ_ += Z.on(qubit + 1, qubit)
    for qubit in range(qubit_num):
        circ_ += RY(f'1_{qubit}').on(qubit)
    for qubit in range(1, qubit_num - 1, 2):
        circ_ += Z.on(qubit + 1, qubit)
    circ_ += Z.on(0, qubit_num - 1)
    circ_ = add_prefix(circ_, prefix)
    return circ_


class MyLoss(nn.LossBase):  # 构建损失函数

    def __init__(self, edges, reduction='mean'):
        super(MyLoss, self).__init__(reduction)
        self.tanh = ops.Tanh()
        self.edges = edges

    def construct(self, logits):
        x = self.tanh(logits)
        out = 0
        for edge in self.edges:
            out += x[edge[0]] * x[edge[1]]
        return self.get_loss(out)


class MyWithLossCell(nn.Cell):  # 量子网络与损失函数结合

    def __init__(self, backbone, loss_fn):
        super(MyWithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn

    def construct(self):
        out = self._backbone()
        return self._loss_fn(out)


# 构造图结构
class Custom_graph(object):

    def __init__(self, edges=[[0, 1], [1, 2], [2, 3], [1, 3]]):
        for edge in edges:
            if len(edge) != 2:
                raise ValueError(f'一条边只能有两个顶点，您输入的边 {edge} 顶点数为：{len(edge)}。')
            if edge[0] == edge[1]:
                raise ValueError(f'一条边的两个顶点不能相同，您输入的边 {edge} 的两个顶点相同。')
        edges_set = set([f'{edge}' for edge in edges])
        if len(edges_set) != len(edges):
            raise ValueError('您输入的边中有重复项。')
        if min(map(min, edges)) != 0:
            raise ValueError(
                f'图顶点应该从 0 开始标记，但您输入的图从 {min(map(min, edges))} 开始标记。')
        if max(map(max, edges)) + 1 < 3:
            raise ValueError(
                f'抱歉，目前只支持 3 个顶点以上的图的最大割求解，您输入的图顶点数为 {max(map(max, edges))+1}。'
            )
        self.edges = edges
        self.node_num = max(map(max, self.edges)) + 1

    def exhaustion(self):  ## 经典穷举法
        if self.node_num > 10:
            return f'本程序经典穷举法支持的图顶点数范围为 [3, 10]， 您输入图的顶点数为 {self.node_num}，故不能计算。'
        g = nx.Graph()
        for i in range(len(self.edges)):
            nx.add_path(g, self.edges[i])

        max_cut = 0  # 通过比较，得到最大割数
        for i in g.nodes:
            max_cut = max(max_cut,
                          nx.cut_size(g, [i]))  # 一组 1 个顶点、另一组 node_num - 1 个顶点
            for j in range(i):
                max_cut = max(max_cut, nx.cut_size(
                    g, [i, j]))  # 一组 2 个顶点、另一组 node_num - 2 个顶点
                if self.node_num > 5:
                    for k in range(j):
                        max_cut = max(max_cut, nx.cut_size(
                            g, [i, j, k]))  # 一组 3 个顶点、另一组 node_num - 3 个顶点
                        if self.node_num > 7:
                            for m in range(k):
                                max_cut = max(
                                    max_cut, nx.cut_size(
                                        g,
                                        [i, j, k, m
                                         ]))  # 一组 4 个顶点、另一组 node_num - 4 个顶点
                                if self.node_num > 9:
                                    for n in range(m):
                                        max_cut = max(
                                            max_cut,
                                            nx.cut_size(g, [i, j, k, m, n])
                                        )  # 一组 5 个顶点、另一组 node_num - 5 个顶点

        return f'经典穷举法的最大割数为：{max_cut}'

    def MBE(self):  ## 量子 MBE 法
        qubit_num = int((self.node_num + 1) / 2)  # 比特数为顶点数的一半
        layer_num = 2 * qubit_num  # 层数为比特数的两倍

        ansatz = Circuit()
        for layer in range(layer_num):  # 拟设结构
            ansatz += QLayer(qubit_num=qubit_num, prefix=f'{layer}')

        sim = Simulator('mqvector', ansatz.n_qubits)
        if self.node_num % 2 == 0:
            ham =  [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(qubit_num)] + \
                        [Hamiltonian(QubitOperator(f'X{i}')) for i in range(qubit_num)]
        else:
            ham =  [Hamiltonian(QubitOperator(f'Z{i}')) for i in range(qubit_num-1)] + \
                        [Hamiltonian(QubitOperator(f'X{i}')) for i in range(qubit_num-1)] + \
                        [Hamiltonian(QubitOperator(f'Z{qubit_num-1}'))]
        grad_ops = sim.get_expectation_with_grad(
            ham, ansatz)

        QuantumNet = MQAnsatzOnlyLayer(grad_ops)
        loss = MyLoss(self.edges)
        net_with_criterion = MyWithLossCell(QuantumNet, loss)
        opti = nn.Adam(QuantumNet.trainable_params(), learning_rate=0.05)
        net = nn.TrainOneStepCell(net_with_criterion, opti)

        # 训练 200 次
        for i in range(200):
            res = net()

        round = ops.Round()
        out = QuantumNet()
        result = 0
        for edge in self.edges:
            result += (1 - round(out[edge[0]]) * round(out[edge[1]])) / 2

        return f'量子 MBE 法得到的最大割数为：{int(result.asnumpy() + 0.5)}'