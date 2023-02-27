# 项目说明

## 论文说明

### 来源

题目： Koopman operator learning for accelerating quantum optimization and machine learning

链接：https://arxiv.org/abs/2211.01365

### 论文简要介绍

与经典机器学习类似，在量子机器学习的VQE方法中，我们同样需要借助梯度类算法（如梯度下降，Adam等）来调节参数以优化哈密顿量对应的能量值。然而与经典机器学习不同的是，在量子计算中，梯度的获取依赖于测量，而测量是一个相对昂贵的计算开销，因此，我们希望通过不依赖梯度的方法来加速VQE算法的学习过程。Koopman算子理论是动力系统领域的一种线性化方法，将原始的状态空间映射到一个由观测函数构成的无限维的函数空间中。对于映射后的动力系统，系统的状态的演化是线性的。本文将参数的下降过程视为动力系统中的演化轨迹，并希望借助Koopman算子理论对其进行加速。

本文在一维ising模型和MNIST数据集上做了实验，证明了本文提出的加速方法有一定效果。

### 复现要求

复现论文中的fig3-(b) 中的Full VQE、DMD和SW-DMD。

## python运行环境

本项目主要依赖以下package

- pydmd
- mindspore
- mindquantum

## 项目结构

- readme.md 本文件，项目说明文件
- src/experiment_class.py 控制实验的class的定义，参数设定，迭代控制均由该class完成。
- src/iteration_generators.py 迭代序列的生成器。给定初始点，ansatz，哈密顿量，优化器，生成器可返回对应的VQE优化过程的参数的序列。
- src/koopman_iteration_generators.py 使用Koopman理论的迭代初始点生成器。给定一个参数序列，生成器可以返回Koopman算子理论视角的最优的新的迭代初始点。
- src/vqe_utilities.py 其他函数。
- main.ipynb 实验开展和结果呈现的notebook，环境就绪后，直接运行可以得到实验结果。

## 创新点

原文中的Koopman理论的计算实际使用的是最小二乘法的显式解（尽管原文中声称使用dmd）；本项目中使用dmd算法和hankeldmd算法完成koopman理论的计算，而没有显式地求解方程，大大节省了koopman理论的计算开销。





