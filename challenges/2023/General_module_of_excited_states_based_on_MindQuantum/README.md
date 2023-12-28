# 利用MindQuantum实现VQE求解激发态通用模块

## 概述:
在Mindquantum中实现VQE求解激发态通用模块，包含Folded spectrum method、Orthogonally constrained VQE、Subspace expansion、Subspace-search VQE、Multistate contracted VQE、Orthogonal State Reduction Variational Eigensolver 六种方法。

## 代码目录说明:
算法实现集中在同一个文件，即`algorithm.py`，`report.ipynb`为用于演示的Jupyter Notebook。

## 算法简介:

Folded spectrum method: 使用大致的目标能量放入到VQE的cost function中成为惩罚项。

Orthogonally constrained VQE: 用一种迭代的方式，不断用swap test的方式测量量子态内积，惩罚已经得到的本征态，以得到下一个激发态。

Subspace expansion VQE: 先用传统的VQE算法得到基态，随后定义一组激发算子$\{O_i\}$。随后测量$H_{ij}^{'} =\langle\psi_{gs} |O_i^{\dagger} H O_j | \psi_{gs}\rangle$和$S_{ij}=\langle \psi_{gs} |O_i^\dagger O_j | \psi_{gs}\rangle$ 两个矩阵，求解$H^{'} C=SCE$这个广义本征值问题可得基态和激发态能量，其中E是本征值的对角矩阵。激发算子的选择很大程度上决定着算法的成功。

Subspace search VQE: 同时求解本征态能量。这是通过优化cost function: $\sum_{i} w_i \langle \psi_i |U^\dagger (\theta)|H|U(\theta)| \psi_i \rangle, w_0>w_1>\dots>w_k $ 实现的。

Multistate contracted VQE: 这个算法使用无权重的Subspace search VQE算法，这样子得到的$U(\theta)$可以展开k个最低能级本征态的子空间。随后，在训练结束的基础上，融入subspace expansion方法，测量$H_{ij}^{'}=\langle \phi_i |H| \phi_j \rangle$，其中$|\phi_i\rangle=U(θ)|\psi_i⟩$。再经典地对角化$H'$, 可得最低能量的k个本征态能量。

Orthogonal State Reduction Variational Eigensolver: 这个算法使用ancilla 量子比特，以去除已得到的本征态在cost function中的权重，从而得到只存在激发态能量的effective cost function。优化这个cost function即可得到下一个本征态。同时也意味着，制备k个本征态需要k个ancilla qubit。


## 训练
在算法演示中，使用的例子模型为海森堡模型。使用的所有变分电路为`mindquantum.algorithm.nisq.HardwareEfficientAnsatz`。本次用的是`scipy`里集成的`L-BFGS-B`优化器，使用的是默认的超参数，初始参数用`numpy.random.rand`函数随机生成。


## 参考论文:
[1] [MindQuantum](https://gitee.com/mindspore/mindquantum/tree/master)
[2] [Noisy intermediate-scale quantum (NISQ) algorithms](https://doi.org/10.1103/RevModPhys.94.015004)
[3] [Orthogonal State Reduction Variational Eigensolver for the Excited-State Calculations on Quantum Computers](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00159)