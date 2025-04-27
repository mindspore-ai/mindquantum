'''
本文件用于对MNIST数据集进行预处理
包括使用 PCA 降维、抽样等操作
'''

from sklearn.decomposition import PCA
import numpy as np
from torchvision import datasets
from sklearn.model_selection import train_test_split
import torch

my_mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)


def sample_data(x, y, label, sample_ratio=0.2):
    label_mask = y == label
    x_label = x[label_mask]
    y_label = y[label_mask]
    sample_size = int(len(y_label) * sample_ratio)
    sample_indices = np.random.choice(len(y_label), sample_size, replace=False)
    return x_label[sample_indices], y_label[sample_indices]


def filter_3_and_6(data):
    images, labels = data
    mask = (labels == 3) | (labels == 6)
    return images[mask], labels[mask]


def pca_data_preprocessing(mnist_dataset: datasets.MNIST, pca_dim: int = 8, ratio: float = 0.1):
    '''
    将 28*28 的 MNIST 手写数字图像 基于PCA进行压缩
    mnist_dataset:datasets.MNIST MNIST数据集
    pca_dim:int=8 PCA降维后的维度
    ratio:float=0.1 采样比例
    '''
    np.random.seed(10)

    # mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))
    x_data, y = filtered_data  # X 图像数据 y 标签

    x_data_3, y_data_3 = sample_data(x_data, y, label=3, sample_ratio=ratio)
    x_data_6, y_data_6 = sample_data(x_data, y, label=6, sample_ratio=ratio)
    # 合并抽样后的数据
    x_sampled = torch.cat((x_data_3, x_data_6), dim=0)
    y_sampled = torch.cat((y_data_3, y_data_6), dim=0)

    n_samples = x_sampled.shape[0]
    x_flattened = x_sampled.view(n_samples, -1)  # 将图像展平为一维向量
    x_flattened = x_flattened / 255
    pca = PCA(n_components=pca_dim)
    x_pca = pca.fit_transform(x_flattened)

    # 将 PCA 处理后的值缩放到 [0, π] 之间
    x_pca_min = np.min(x_pca)
    x_pca_max = np.max(x_pca)
    x_pca_scaled = np.pi * (x_pca - x_pca_min) / (x_pca_max - x_pca_min)

    x_train, x_test, y_train, y_test = train_test_split(
        x_pca_scaled, y_sampled, test_size=0.2, random_state=0, shuffle=True
    )  # 将数据集划分为训练集和测试集
    y_train[y_train == 3] = 1
    y_train[y_train == 6] = 0
    y_test[y_test == 3] = 1
    y_test[y_test == 6] = 0
    y_train = y_train.numpy()
    y_test = y_test.numpy()

    return x_train, x_test, y_train, y_test


# x_train, x_te's't, y_train, y_test = pca_data_preprocessing(mnist_dataset,8)


def pca_data_preprocessing_micro(mnist_dataset, pca_dim: int = 8, ratio: float = 1.0):
    '''
    本函数用于在micro serach中进行数据预处理,抽取全部数据
    将 28*28 的 MNIST 手写数字图像 基于PCA进行压缩

    '''
    # mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
    filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))
    x_data, label = filtered_data  # X 图像数据 y 标签

    x_data_3, y_data_3 = sample_data(x_data, label, label=3, sample_ratio=ratio)
    x_data_6, y_data_6 = sample_data(x_data, label, label=6, sample_ratio=ratio)
    # 合并抽样后的数据
    x_sampled = torch.cat((x_data_3, x_data_6), dim=0)
    y_sampled = torch.cat((y_data_3, y_data_6), dim=0)

    n_samples = x_sampled.shape[0]
    x_flattened = x_sampled.view(n_samples, -1)  # 将图像展平为一维向量
    x_flattened = x_flattened / 255
    pca = PCA(n_components=pca_dim)
    x_pca = pca.fit_transform(x_flattened)

    # 将 PCA 处理后的值缩放到 [0, π] 之间
    x_pca_min = np.min(x_pca)
    x_pca_max = np.max(x_pca)
    x_pca_scaled = np.pi * (x_pca - x_pca_min) / (x_pca_max - x_pca_min)

    y_sampled[y_sampled == 3] = 1
    y_sampled[y_sampled == 6] = 0
    y_sampled[y_sampled == 3] = 1
    y_sampled[y_sampled == 6] = 0
    y_sampled = y_sampled.numpy()

    return x_pca_scaled, y_sampled


x_full, y_full = pca_data_preprocessing_micro(my_mnist_dataset, 8, 0.95)


def getfulldata(mnist_dataset: datasets.MNIST, pca_dim: int = 8):
    '''
    本函数用于获取完整的数据集，不进行抽样
    mnist_dataset:datasets.MNIST MNIST数据集
    pca_dim:int=8 PCA降维后的维度
    '''
    filtered_data = filter_3_and_6((mnist_dataset.data, mnist_dataset.targets))  # 过滤数据集
    x_data, y = filtered_data  # X 图像数据 y 标签
    x_flattened = x_data.view(x_data.shape[0], -1)  # 将图像展平为一维向量
    x_flattened = x_flattened / 255  # 归一化
    pca = PCA(n_components=pca_dim)  # PCA降维
    x_pca = pca.fit_transform(x_flattened)  # PCA降维
    x_pca_min = np.min(x_pca)  # 归一化
    x_pca_max = np.max(x_pca)
    x_pca_scaled = np.pi * (x_pca - x_pca_min) / (x_pca_max - x_pca_min)  # 归一化
    y[y == 3] = 1
    y[y == 6] = 0
    y[y == 3] = 1
    y[y == 6] = 0
    return x_pca_scaled, y.numpy()


x_fulldata, y_fulldata = getfulldata(my_mnist_dataset, 8)
