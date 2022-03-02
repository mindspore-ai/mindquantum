#  论文复现 48: Grand Unification of Quantum Algorithms

## 项目介绍
我复现的是48号论文：Grand Unification of Quantum Algorithms。这篇论文主要介绍了量子奇异值变换算法（QSVT）及其在量子计算中的广泛应用。

## 主要结果
我复现的内容是论文Grand Unification of Quantum Algorithms中的第三部分"Search by QSVT"。主要设计出了基于QSVT算法的、可以用于小于10个量子比特数的量子搜索（Grover)算法的量子线路，并在mindquantum框架下举例列出了2、3、4、8个量子比特下的线路，用projectq模拟器进行仿真，能够100%成功地翻转被标记比特的相位。

## 创新点
用QSVT算法的思路设计了一个与经典Grover算法线路完全不同的线路，将经典Grover的二维旋转升级到了整个Bloch Sphere球面，解决了经典Grover算法容易overshoot的问题。QSVT已被证实还可以描述Shor算法、哈密顿量模拟算法等基础算法，有望实现量子算法的大统一。


邮箱：ZhangYR_linda@163.com