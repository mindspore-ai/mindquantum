import mindspore
import numpy as np
from mindspore import nn, ops, Tensor, CSRTensor, COOTensor
import networkx as nx
from collections import defaultdict

device="cuda"
class CNN_lsing(nn.Cell):
    def __init__(self, inputsize=28, kenrnal_size=5, stride=3, kenrnal_num=32, label_class_num=10, label_node_num=50) -> None:
        super().__init__()
        self.inputsize = inputsize

        self.kenrnal_size = kenrnal_size
        self.stride = stride
        self.kenrnal_num = kenrnal_num

        self.label_class_num = label_class_num
        self.label_node_num = label_node_num
        self.output_node_num = self.label_node_num
        


        idxs = Tensor(range(self.inputsize))
        idxs = idxs.unfold(dimension=0, size=self.kenrnal_size, step=self.stride)
        image_idxs = []
        for H in idxs:
            for W in idxs:
                patch = []
                for h in H:
                    for w in W:
                        patch.append(h*self.inputsize + w)
                image_idxs.append(patch)
        self.image_idxs = Tensor(image_idxs)
        self.img_node_num = self.inputsize * self.inputsize
        self.input_node_num = self.img_node_num
        self.image_idxs_inv = [[]]*(self.inputsize*self.inputsize)
        for i, idx in enumerate(self.image_idxs.flatten()):
            self.image_idxs_inv[idx].append(i)


        self.cnn_idxs = self.image_idxs.repeat(self.kenrnal_num, 1)
        self.cnn_flags = Tensor([[i]*self.image_idxs.shape[0] for i in range(self.kenrnal_num)]).flatten()
        self.cnn_node_num = len(self.cnn_flags)

        self.all_node_num = self.img_node_num + self.cnn_node_num + self.output_node_num

        self.J_CNN = Tensor(np.random.randn(self.kenrnal_num, self.kenrnal_size*self.kenrnal_size)) * 0.01
        self.J_MLP = Tensor(np.random.randn(self.cnn_node_num, self.output_node_num)) * 0.01
        self.H_IMG = Tensor(np.random.randn(self.img_node_num)) * 0.01
        # self.H_CNN = Tensor(np.random.randn(self.cnn_node_num)) * 0.01
        self.H_CNN = ops.zeros(self.cnn_node_num)
        self.H_MLP = Tensor(np.random.randn(self.output_node_num)) * 0.01

        self.deta_J_CNN_all = 0
        self.deta_J_MLP_all = 0
        self.deta_H_IMG_all = 0
        self.deta_H_CNN_all = 0
        self.deta_H_MLP_all = 0

        self.group()
        self.create_J_H()

    def group(self):
        nodes = range(self.all_node_num)
        edges = []
        for i, image_idxs in enumerate(self.cnn_idxs):
            i += self.img_node_num
            for image_idx in image_idxs:
                edges.append((i, int(image_idx)))
                edges.append((int(image_idx), i))
        for i in range(self.img_node_num, self.img_node_num+self.cnn_node_num):
            for j in range(self.img_node_num+self.cnn_node_num, self.all_node_num):
                edges.append((i, j))
                edges.append((j, i))
        self.graph = nx.Graph()
        self.graph.add_nodes_from(nodes)
        self.graph.add_edges_from(edges)
        color_map = nx.greedy_color(self.graph, strategy='DSATUR')
        self.group_all = defaultdict(list)
        self.group_hidden = defaultdict(list)
        self.group_clssify = defaultdict(list)
        self.group_gen = defaultdict(list)
        for node, color in color_map.items():
            if True:
                self.group_all[color].append(node)
            if node >= self.img_node_num and node < (self.img_node_num + self.cnn_node_num):
                self.group_hidden[color].append(node)
            if node >= self.img_node_num:
                self.group_clssify[color].append(node)
            if node < self.img_node_num + self.cnn_node_num:
                self.group_gen[color].append(node)
        for key, value in self.group_all.items():
            value.sort()
            self.group_all[key] = Tensor(value)
        for key, value in self.group_hidden.items():
            value.sort()
            self.group_hidden[key] = Tensor(value)
        for key, value in self.group_clssify.items():
            value.sort()
            self.group_clssify[key] = Tensor(value)
        for key, value in self.group_gen.items():
            value.sort()
            self.group_gen[key] = Tensor(value)

    def create_J_H(self):
        J = ops.zeros((self.all_node_num, self.all_node_num))
        for i, (image_idxs, cnn_flag) in enumerate(zip(self.cnn_idxs, self.cnn_flags)):
            i += self.img_node_num
            for j, image_idx in enumerate(image_idxs):
                J[i, image_idx] = J[image_idx, i] = self.J_CNN[cnn_flag, j]

        J[self.img_node_num:self.img_node_num+self.cnn_node_num, self.img_node_num+self.cnn_node_num:] = self.J_MLP
        J[self.img_node_num+self.cnn_node_num:, self.img_node_num:self.img_node_num+self.cnn_node_num] = self.J_MLP.transpose(1, 0)

        # H = ops.zeros(self.all_node_num)
        # for i, idx in enumerate(self.image_idxs.flatten()):
        #     H[i] = self.H_IMG[idx]
        # for i, flag in enumerate(self.cnn_flags):
        #     H[i+self.img_node_num] = self.H_CNN[flag]
        # H[self.cnn_node_num + self.img_node_num:] = self.H_MLP

        H = Tensor.cat([self.H_IMG, self.H_CNN, self.H_MLP])

        self.J = J
        self.H = H

    def create_m(self, images_batch=None, labels_batch=None):
        m = np.random.randint(0, 2, (images_batch.shape[0], self.all_node_num))
        if images_batch != None:
            m[:, :self.input_node_num] = images_batch

        if labels_batch != None:
            labels = ops.zeros((labels_batch.shape[0], self.label_node_num))
            for i, label in enumerate(labels_batch):
                labels[i][int(label)::self.label_class_num] = 1
            m[:, self.cnn_node_num + self.img_node_num:] = labels

        m = mindspore.numpy.where(m==0, -1., 1.)
        m = Tensor(m)
        return m

    def construct(self, m, group, sample_num=1e+3):
        J_group = [self.J[idxs] for idxs in group.values()]
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

    def updateParams(self, m_data, m_model, images, batch_size, lr=3e-3, momentum=0.6):
        batch_size = images.shape[0]
        m_data = m_data.cpu()
        m_model = m_model.cpu()
        images = mindspore.numpy.where(images==0., -1., 1.)

        # images = images.reshape(batch_size, self.inputsize, self.inputsize).cpu()
        # cnn_data = m_data[:, self.img_node_num:self.img_node_num+self.cnn_node_num].reshape(-1, self.kenrnal_num, 1, self.kenrnal_size, self.kenrnal_size)
        # cnn_model = m_model[:, self.img_node_num:self.img_node_num+self.cnn_node_num].reshape(-1, self.kenrnal_num, 1, self.kenrnal_size, self.kenrnal_size)
        # deta_J_cnn = ops.zeros((self.kenrnal_num, self.kenrnal_size, self.kenrnal_size))
        # for i in range(batch_size):
        #     deta_J_cnn += (F.conv2d(images[i].unsqueeze(0), weight=cnn_data[i], stride=self.stride)-F.conv2d(images[i].unsqueeze(0), weight=cnn_model[i], stride=self.stride))
        # deta_J_cnn /= batch_size
        # deta_J_cnn = deta_J_cnn.reshape((self.kenrnal_num, self.kenrnal_size * self.kenrnal_size))

        images = images.reshape(batch_size, self.inputsize*self.inputsize).cpu()
        cnn_data = m_data[:, :self.img_node_num+self.cnn_node_num]
        cnn_model = m_model[:, :self.img_node_num+self.cnn_node_num]
        cnn_data[:, :self.img_node_num] = images
        cnn_model[:, :self.img_node_num] = images
        deta = (mindspore.ops.mm(cnn_data.T, cnn_data) - mindspore.ops.mm(cnn_model.T, cnn_model)) / batch_size
        deta_J_cnn = ops.zeros((self.kenrnal_num, self.kenrnal_size*self.kenrnal_size))
        # deta_J_cnn_num = ops.zeros((self.kenrnal_num, self.kenrnal_size*self.kenrnal_size))
        num = 0
        for j, idxs in enumerate(self.cnn_idxs.reshape(self.kenrnal_num, -1, self.kenrnal_size*self.kenrnal_size)):
            numm = 0
            for _ , idx in enumerate(idxs):
                # j kenrnal_num
                for i, item in enumerate(idx):
                    # i p num
                    deta_J_cnn[j, i] += deta[self.img_node_num + num, item]
                    # deta_J_cnn_num[j, i] += 1
                    numm += 1
                num += 1
        # deta_J_cnn /= deta_J_cnn_num



        mlp_data = m_data[self.img_node_num:]
        mlp_model = m_model[self.img_node_num:]
        deta_J_mlp = (mindspore.ops.mm(mlp_data.T, mlp_data) - mindspore.ops.mm(mlp_model.T, mlp_model)) / batch_size
        deta_J_mlp = deta_J_mlp[self.img_node_num:self.img_node_num+self.cnn_node_num, -self.output_node_num:]

        deta_H = (m_data - m_model).mean(dim=0)
        deta_H_img = deta_H[:self.img_node_num]
        deta_H_cnn = deta_H[self.img_node_num:-self.output_node_num]
        deta_H_mlp = deta_H[self.img_node_num+self.cnn_node_num:]

        # # CNN
        # for i, idxs in enumerate(self.image_idxs_inv):
        #     deta_H_img[i] = deta_H[idxs].mean()

        # self.deta_J_CNN_all = self.deta_J_CNN_all.cpu()
        # self.deta_J_MLP_all = self.deta_J_MLP_all.cpu()
        # self.deta_H_IMG_all = self.deta_H_IMG_all.cpu()
        # self.deta_H_CNN_all = self.deta_H_CNN_all.cpu()
        # self.deta_H_MLP_all = self.deta_H_MLP_all.cpu()

        deta_J_cnn = deta_J_cnn.cpu()
        deta_J_mlp = deta_J_mlp.cpu()
        deta_H_img = deta_H_img.cpu()
        deta_H_cnn = deta_H_cnn.cpu()
        deta_H_mlp = deta_H_mlp.cpu()

        self.deta_J_CNN_all = momentum * self.deta_J_CNN_all + deta_J_cnn * lr
        self.deta_J_MLP_all = momentum * self.deta_J_MLP_all + deta_J_mlp * lr
        self.deta_H_IMG_all = momentum * self.deta_H_IMG_all + deta_H_img * lr
        self.deta_H_CNN_all = momentum * self.deta_H_CNN_all + deta_H_cnn * lr
        self.deta_H_MLP_all = momentum * self.deta_H_MLP_all + deta_H_mlp * lr

        self.J_CNN += self.deta_J_CNN_all
        self.J_MLP += self.deta_J_MLP_all
        self.H_IMG += self.deta_H_IMG_all
        # self.H_CNN += self.deta_H_CNN_all
        self.H_MLP += self.deta_H_MLP_all

        self.J_CNN = self.J_CNN.clip(-1,1)
        self.J_MLP = self.J_MLP.clip(-1,1)

        self.create_J_H()