# Variational Quantum Singular Value Deposition

## 项目介绍

### SVD

**奇异值分解**(Singular Value Decomposition，简称**SVD**)是线性代数中一种重要的矩阵分解，它作为特征分解在任意维数矩阵上的推广，在机器学习领域中被广泛应用，常用于矩阵压缩、推荐系统以及自然语言处理等。

**定义**：给定一个复数矩阵 $M \in \mathbb{C}^{m \times n}$ ，则定义矩阵 $M$ 的**SVD**为： $M = UDV^\dagger$ 。其中 $U$ 是 $m \times m$ 的矩阵， $V$ 是 $n \times n$ 的矩阵， $U, V$ 都是酉矩阵，即满足 $UU^\dagger = I, VV^\dagger = I$ 。$D$ 是 $m \times n$ 的对角阵，主对角线上的的元素从大到小排列，每个元素都称为矩阵 $M$ 的奇异值。

### VQSVD

**变分量子奇异值分解**(Variational Quantum Singular Value Decomposition，简称**VQSVD**)将SVD转换成优化问题，并通过变分量子线路求解。

论文将矩阵奇异值分解分成4个步骤求解：

1. 输入待分解的矩阵 $M$ ，想压缩到的阶数 $T$ ，权重 $Weights$ ，测量基 $\{ | \psi_1\rangle,\cdots |\psi_T\rangle\}$ ，参数化的酉矩阵 $U(\theta)$ 和 $V(\phi)$ （即ansatz）；
2. 搭建量子神经网络估算奇异值 $m_j = \text{Re}\langle\psi_j|U(\theta)^{\dagger} M V(\phi)|\psi_j\rangle$ ，并最大化加权奇异值的和 $L(\theta,\phi) = \sum_{j=1}^T q_j\times \text{Re} \langle\psi_j|U(\theta)^{\dagger} M V(\phi)|\psi_j\rangle$ ，其中，加权是为了让计算出的奇异值从大到小排列；
3. 读出最大化时参数值 $\alpha^ \star$ 和 $\beta^\star$ ，计算出 $U(\alpha^\star)$ 和 $V(\beta^\star)$
4. 输出结果：奇异值 ${m_1, \cdots, m_r}$和奇异矩阵  $U(\alpha^\star)$ 和 $V(\beta^\star)$

![VQSVD steps](https://gitee.com/mindspore/mindquantum/raw/research/paper_recurrence/38_lolo1222/figure/QSVD.png)

## 主要结果

1. 对随机生成的 8×8 复数矩阵作**VQSVD**，进行矩阵压缩，作出误差分析，并与**SVD**的误差对比

   ![loss](https://gitee.com/mindspore/mindquantum/raw/research/paper_recurrence/38_lolo1222/figure/loss.png)![error](https://gitee.com/mindspore/mindquantum/raw/research/paper_recurrence/38_lolo1222/figure/error.png)

2. 使用**VQSVD**算法将MNIST中的一张图片大小从28\*28压缩至8*8

   <img src="https://gitee.com/mindspore/mindquantum/raw/research/paper_recurrence/38_lolo1222/figure/MNIST_32.png" alt="MNIST_32" style="zoom:750%;" />![compress](.\figure\compress.png)

## 创新点

虽然这篇论文里理论上是使用变分量子神经网络来近似优化，但在实际代码中，原作者使用的是将量子网络转换成经典张量网络优化，又转换回了经典计算机上，并未利用到量子计算机的特性。

考虑到**MindQuantum**是模拟真实物理上的量子计算机，基于测量给出结果，我学习了**MindQuantum**的用法后发现`get_expectation_with_grad` 方法可以用来计算如下表达式的值和线路中参数的梯度。

$$
E(\theta) = \langle\phi|U_l^{\dagger}(\theta) H U_r(\theta)|\psi\rangle
$$

我将待分解矩阵M嵌入哈密顿量H中，将 $U_l(\theta)$ 设置成 $U(\theta)$ ， $U_r(\theta)$ 设置成 $V(\phi)$ ，通过模拟器的`set_qs`设置模拟器的状态为我所取的测量基，从而获取到给定测量基下的测量结果，即对应位置的奇异值。再利用**MindQuantum**的 `MQAnsatzOnlyLayer` 搭建出基于各测量基下的量子网络层，其输出为 $\text{Re}\langle\psi_j|U(\theta)^{\dagger} M V(\phi)|\psi_j\rangle$ 。结合**MindSpore**提供的经典学习框架构建出量子经典混合网络，实现对量子网络层加权求和，从而构建出损失函数 $L(\theta,\phi) = \sum_{j=1}^T q_j\times \text{Re} \langle\psi_j|U(\theta)^{\dagger} M V(\phi)|\psi_j\rangle$ 并进行优化。

这种方法更符合量子计算的逻辑，更能发挥出量子ansatz的优势，从最终结果来看，这种方法的误差也要优于原作者转换成张量网络求解的方法。

邮箱地址：2265983842@qq.com
