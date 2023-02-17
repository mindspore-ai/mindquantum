"""
The codes that reproduce the figures(Fig.4, Fig.5, Fig.7) in paper.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

from src.uitls import get_svd_norm, get_ansatz, matrix_distance
from src.uitls import run, reconstruct_by_svd, run_light


def plot_figure4():
    """复现论文中 Figure.4 结果."""
    print("It will cost about 26 minutes on 8u32G cpu.")

    n_qubit = 3           # 量子比特数
    m = 2**n_qubit        # 矩阵大小
    depths = [10, 20]     # 线路深度
    ranks = range(1, 9)   # 使用的秩数
    max_epoch = 500
    lr = 0.05
    # 随机生成矩阵
    in_mat = np.random.randn(m, m)
    dist_classic = get_svd_norm(in_mat, ranks)

    dist_list = []

    for depth in depths:
        ansatz_uv = get_ansatz(n_qubit, depth)
        sub_dist_list = []
        for rank in ranks:
            print(f"Depth = {depth}, rank = {rank}:")
            dy_lr = min(0.05, lr / (rank / 2))
            re_mat, _ = run(n_qubit, rank, ansatz_uv, in_mat,
                            epoch=max_epoch, lr=dy_lr)
            dist = matrix_distance(in_mat, re_mat)
            sub_dist_list.append(dist)
        dist_list.append(copy.deepcopy(sub_dist_list))

    # 绘图
    plt.plot(ranks, dist_classic, 'rd-', label='Classical SVD')
    plt.plot(ranks, dist_list[0], 'gs--', label='VQSVD D=10')
    plt.plot(ranks, dist_list[1], 'bo--', label='VQSVD D=20')
    plt.xlabel('Singular Value Used (Rank = T)')
    plt.ylabel('Norm Distance')
    plt.grid(visible=True)
    plt.legend(loc='lower left')
    plt.savefig('images/figure4.png')
    print("Image saves at: images/figure4.png")
    plt.show()


def plot_figure5(n_qubit=3):
    """复现论文中 Figure.5 结果. 图片大小取 2^n_qubit."""
    import cv2

    print("It will cost about 6 minutes for n_qubit=3 and 3 hours for "
          "n_qubit=5 on 8u32G cpu.")

    depths = [20, 40]     # 线路深度
    rank = 5              # 使用的秩数
    max_epoch = 500
    lr = 0.01
    # 读取图片
    m = 2**n_qubit
    in_mat = plt.imread('images/digit7.png')
    in_mat = cv2.resize(in_mat, (m, m))

    # 使用经典 SVD 重建图片
    re_mat_svd = reconstruct_by_svd(in_mat, rank)
    # 针对不同线路深度，使用量子 VQSVD 重建图片
    re_mats = []
    for depth in depths:
        ansatz_uv = get_ansatz(n_qubit, depth)
        print(f"Depth = {depth}, rank = {rank}:")
        re_mat, _ = run(n_qubit, rank, ansatz_uv, in_mat,
                        epoch=max_epoch, lr=lr)
        re_mats.append(re_mat)
    # 绘图
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(in_mat, cmap='gray')
    plt.title("(a) Original MNIST #7")
    plt.subplot(2, 2, 2)
    plt.imshow(re_mat_svd, cmap='gray')
    plt.title("(b) Reconstruction via SVD, T = 5")
    plt.subplot(2, 2, 3)
    plt.imshow(re_mats[0], cmap='gray')
    plt.title("(c) VQSVD Result, T = 5 and D = 20")
    plt.subplot(2, 2, 4)
    plt.imshow(re_mats[1], cmap='gray')
    plt.title("(d) VQSVD Result, T = 5 and D = 40")
    plt.savefig(f'images/figure5_reconstructed_{m}x{m}.png')
    print(f"Image saves at: images/figure5_reconstructed_{m}x{m}.png")
    plt.show()


def plot_figure5_light():
    """复现论文中 Figure.5 结果. 图片大小为 32x32，通过使用 BFGS 优化器加快收敛，
    另外通过获取量子态直接计算期望值，而不是通过将矩阵分解成哈密顿量再逐项计算，可
    以有效提高计算速度."""
    print("It will cost about 12 minutes on 8u32G cpu.")

    n_qubit = 5           # 量子比特数
    depths = [20, 40]     # 线路深度
    rank = 5              # 使用的秩数
    # 读取图片，图片大小为 32 x 32 的单通道（黑白）图片
    in_mat = plt.imread('images/digit7.png')

    # 使用经典 SVD 重建图片
    re_mat_svd = reconstruct_by_svd(in_mat, rank)
    # 针对不同线路深度，使用量子 VQSVD 重建图片
    re_mats = []
    for depth in depths:
        ansatz_uv = get_ansatz(n_qubit, depth)
        print(f"Depth = {depth}, rank = {rank}:")
        re_mat = run_light(n_qubit, rank, ansatz_uv, in_mat)
        re_mats.append(re_mat)

    # 绘图
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.imshow(in_mat, cmap='gray')
    plt.title("(a) Original MNIST #7")
    plt.subplot(2, 2, 2)
    plt.imshow(re_mat_svd, cmap='gray')
    plt.title("(b) Reconstruction via SVD, T = 5")
    plt.subplot(2, 2, 3)
    plt.imshow(re_mats[0], cmap='gray')
    plt.title("(c) VQSVD Result, T = 5 and D = 20")
    plt.subplot(2, 2, 4)
    plt.imshow(re_mats[1], cmap='gray')
    plt.title("(d) VQSVD Result, T = 5 and D = 40")
    plt.savefig('images/figure5_reconstructed.png')
    print(f"Image saves at: images/figure5_reconstructed.png")
    plt.show()


def plot_figure7():
    """复现论文中 Figure.7 结果."""
    print("It will cost about 60 minutes on 8u32G cpu.")

    n_qubit = 3                      # 量子比特数
    m = 2**n_qubit                   # 随机生成的方阵大小
    lr = 0.05                        # 学习率
    max_epoch = 500                  # 最大迭代次数
    ranks = range(1, 9)              # 使用的秩数
    in_mat = np.random.randn(m, m)   # 随机生成矩阵

    ansatz_a = get_ansatz(n_qubit=3, depth=8, kind='a')
    ansatz_b = get_ansatz(n_qubit=3, depth=3, kind='b')
    ansatz_c = get_ansatz(n_qubit=3, depth=8, kind='c')
    ansatz_d = get_ansatz(n_qubit=3, depth=4, kind='d')
    ansatz_names = ['a', 'b', 'c', 'd']

    dist_classic = get_svd_norm(in_mat, ranks)

    dist_list = []
    for i, ansatz_uv in enumerate([ansatz_a, ansatz_b, ansatz_c, ansatz_d]):
        sub_dist_list = []
        for rank in ranks:
            print(f"Ansatz type = {ansatz_names[i]}, rank = {rank}:")
            re_mat = run_light(n_qubit, rank, ansatz_uv, in_mat)
            dy_lr = lr / rank
            re_mat, _ = run(n_qubit, rank, ansatz_uv, in_mat,
                            epoch=max_epoch, lr=dy_lr)
            dist = matrix_distance(in_mat, re_mat)
            sub_dist_list.append(dist)
        dist_list.append(copy.deepcopy(sub_dist_list))
    # 绘图
    plt.plot(ranks, dist_classic, 'd-', color='purple', label='Classical SVD')
    plt.plot(ranks, dist_list[0], 'v--', color='plum', label='VQSVD Ansatz (a)')
    plt.plot(ranks, dist_list[1], 's--', color='green', label='VQSVD Ansatz (b)')
    plt.plot(ranks, dist_list[2], 'o--', color='cyan', label='VQSVD Ansatz (c)')
    plt.plot(ranks, dist_list[3], '^--', color='darkblue', label='VQSVD Ansatz (d)')
    plt.xlabel('Singular Value Used (Rank = T)')
    plt.ylabel('Norm Distance')
    plt.grid(visible=True)
    plt.legend(loc='upper right')
    plt.savefig('images/figure7.png')
    print("Image saves at: images/figure7.png")
    plt.show()


def plot_figure7_light():
    """复现论文中 Figure.7 结果."""
    print("It will cost about 30 minutes on 8u32G cpu.")

    n_qubit = 3                      # 量子比特数
    m = 2**n_qubit                   # 随机生成的方阵大小
    ranks = range(1, 9)              # 使用的秩数
    in_mat = np.random.randn(m, m)   # 随机生成矩阵

    ansatz_a = get_ansatz(n_qubit=3, depth=8, kind='a')
    ansatz_b = get_ansatz(n_qubit=3, depth=3, kind='b')
    ansatz_c = get_ansatz(n_qubit=3, depth=8, kind='c')
    ansatz_d = get_ansatz(n_qubit=3, depth=4, kind='d')
    ansatz_names = ['a', 'b', 'c', 'd']

    dist_classic = get_svd_norm(in_mat, ranks)

    dist_list = []
    for i, ansatz_uv in enumerate([ansatz_a, ansatz_b, ansatz_c, ansatz_d]):
        sub_dist_list = []
        for rank in ranks:
            print(f"Ansatz type = {ansatz_names[i]}, rank = {rank}:")
            re_mat = run_light(n_qubit, rank, ansatz_uv, in_mat)
            re_mat = run_light(n_qubit, rank, ansatz_uv, in_mat)
            dist = matrix_distance(in_mat, re_mat)
            sub_dist_list.append(dist)
        dist_list.append(copy.deepcopy(sub_dist_list))
    # 绘图
    plt.plot(ranks, dist_classic, 'd-', color='purple', label='Classical SVD')
    plt.plot(ranks, dist_list[0], 'v--', color='plum', label='VQSVD Ansatz (a)')
    plt.plot(ranks, dist_list[1], 's--', color='green', label='VQSVD Ansatz (b)')
    plt.plot(ranks, dist_list[2], 'o--', color='cyan', label='VQSVD Ansatz (c)')
    plt.plot(ranks, dist_list[3], '^--', color='darkblue', label='VQSVD Ansatz (d)')
    plt.xlabel('Singular Value Used (Rank = T)')
    plt.ylabel('Norm Distance')
    plt.grid(visible=True)
    plt.legend(loc='upper right')
    plt.savefig('images/figure7.png')
    print("Image saves at: images/figure7_light.png")
    plt.show()


if __name__ == "__main__":
    # 复现论文中 Fig.4
    plot_figure4()
    # 复现论文中 Fig.5，但仅取图像大小为 2^n_qubit次方，n_qubit=3运行较快
    # plot_figure5(n_qubit=3)
    # 使用优化后的接口复现论文 Fig.5
    plot_figure5_light()
    # 使用优化后的接口复现论文 Fig.7
    plot_figure7_light()
