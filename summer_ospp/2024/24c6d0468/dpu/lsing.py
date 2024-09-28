import numpy as np
import random
import mindspore
from mindspore import nn, ops, Tensor, CSRTensor, COOTensor

import math
import networkx as nx
import dwave_networkx as dnx
from collections import defaultdict


class lsing_model(nn.Cell):
    def __init__(self, input_node_num=28*28, label_class_num=10, label_node_num=50, all_node_num=4264):
        super().__init__()

        self.input_node_num = input_node_num
        self.label_class_num = label_class_num
        self.label_node_num = label_node_num
        self.all_node_num = all_node_num

        self.create_pegasus_pegasus()
        self.group()
        self.create_J_H()

    def create_pegasus_pegasus(self):
        ori_G = dnx.pegasus_graph(15, node_list=np.arange(self.all_node_num))

        # 随机打乱节点顺序
        nodes = list(ori_G.nodes)
        random.shuffle(nodes)

        # 映射节点为从0开始的序列，并修改相应的边
        nodes_map = {node : idx for idx, node in enumerate(nodes)}
        nodes = nodes_map.keys()
        edges = [(nodes_map[x], nodes_map[y]) for x, y in ori_G.edges]

        # 构建graph
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)

    def group(self):
        color_map = nx.greedy_color(self.graph, strategy='DSATUR')
        self.group_all = defaultdict(list)
        self.group_hidden = defaultdict(list)
        self.group_test = defaultdict(list)
        for node, color in color_map.items():
            if True:
                self.group_all[color].append(node)
            if node >= self.input_node_num + self.label_node_num:
                self.group_hidden[color].append(node)
            if node < self.input_node_num or node >= self.input_node_num + self.label_node_num:
                self.group_test[color].append(node)
        for key, value in self.group_all.items():
            value.sort()
            self.group_all[key] = Tensor(value)
        for key, value in self.group_hidden.items():
            value.sort()
            self.group_hidden[key] = Tensor(value)
        for key, value in self.group_test.items():
            value.sort()
            self.group_test[key] = Tensor(value)

    def create_J_H(self):
        self.J = ops.zeros((self.all_node_num, self.all_node_num))
        for x, y in self.graph.edges:
            x = int(x)
            y = int(y)
            self.J[x, y] = self.J[y, x] = Tensor(np.random.randn(1)) * 0.01
        self.J_mask = mindspore.numpy.where(self.J!=0, 1., 0.)

        self.H = mindspore.numpy.zeros(self.all_node_num)
        visible_num = self.input_node_num + self.label_node_num
        self.H[:visible_num] = math.log( (visible_num/self.all_node_num) / (1 - visible_num/self.all_node_num))

        self.deta_J_all = 0
        self.deta_H_all = 0

    def create_m(self, images_batch=None, labels_batch=None):
        m = np.random.randint(0, 2, (images_batch.shape[0], self.all_node_num))
        if images_batch != None:
            m[:, :self.input_node_num] = images_batch

        if labels_batch != None:
            labels = np.zeros((labels_batch.shape[0], self.label_node_num))
            for i, label in enumerate(labels_batch):
                labels[i][int(label)::self.label_class_num] = 1
            m[:, self.input_node_num : self.input_node_num + self.label_node_num] = labels

        m = Tensor(m)
        m = mindspore.numpy.where(m==0, -1., 1.)

        return m
    
    from line_profiler import profile
    @profile
    def construct(self, m, group, sample_num=1e+3):

        def csr(x):
            indptr = [0]
            indices = []
            values = []
            count = 0
            for h, line in enumerate(x):
                for w, data in enumerate(line):
                    if data != 0:
                        values.append(data)
                        indices.append(w)
                indptr.append(count)
            return mindspore.CSRTensor(indptr, indices, values, shape=x.shape)


        J_group = [self.J[idxs] for idxs in group.values()]
        # J_group = [csr(self.J[idxs]) for idxs in group.values()]
        H_group = [self.H[idxs] for idxs in group.values()]

        for _ in range(int(sample_num)):
            for idxs, J, H in zip(group.values(), J_group, H_group):
                # I = J.mv(m.T).T + H
                I = mindspore.ops.mm(J, m.T).T + H
                a = mindspore.ops.tanh(I)
                b = (np.random.rand(I.shape[0], I.shape[1])*2-1)
                b = Tensor(b)
                # b = mindspore.ops.rand_like(I)*2-1
                c = a - b
                m[:, idxs] = mindspore.ops.sign(c)

        return m

    def updateParams(self, m_data, m_model, batch_size, lr=3e-3, momentum=0.6):
        deta_J_new = (mindspore.ops.mm(m_data.T, m_data) - mindspore.ops.mm(m_model.T, m_model)) / batch_size
        deta_H_new = (m_data - m_model).mean(axis=0)

        self.deta_J_all = momentum * self.deta_J_all + deta_J_new * lr
        self.deta_H_all = momentum * self.deta_H_all + deta_H_new * lr

        self.J = self.J + self.deta_J_all * self.J_mask
        self.H = self.H + self.deta_H_all
