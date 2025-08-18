# 基于单比特校准矩阵与局部 IBU 的测量误差校准方案

高荣浩(f.g.m.leonardo@gmail.com, y3lin@spinq.cn, chenronghang2020@outlook.com)

# 摘要

本文探讨了在含噪声的中等规模量子(NISQ[1])设备上，通过单比特校准矩阵与局部迭代贝叶斯展开(IBU[2])方法进行测量误差缓解的方案。首先，仅以两个基态  $|0\rangle^{\otimes n}$  和  $|1\rangle^{\otimes n}$  为标定电路，独立构建每个物理比特的  $2\times 2$  校准矩阵；随后按比特顺序张量积近似构建全局校准矩阵，并基于此引入局部IBU。通过逐比特张量收缩，将校准迭代复杂度从原始  $O(2^{3n})$  （显式三重循环）降至  $O(n\cdot 2^n)$ ，且无需显式存储或求逆  $2^n\times 2^n$  矩阵，极大节省了计算和存储资源。对于仅含  $|0\rangle^{\otimes n}$  和  $|1\rangle^{\otimes n}$  输出的GHZ态，我们设计了基于专用  $2\times 2$  校准矩阵的简化处理流程，以进一步提升其校准精度。实验结果表明：在9比特系统上，利用两条标定电路各10000次采样，仅5步局部IBU迭代便将平均1－TVD从0.93135544提升至0.95787941，总耗时不足  $2\mathrm{s}$  。此外，在对512个基态进行全面评估时，局部IBU的平均  $1 - \mathrm{TVD} = 0.99138$  略低于矩阵求逆法的0.99516，但其计算更高效。最后，我们还讨论了高阶串扰校准、子空间截断[3]、机器学习、自适应迭代停止等未来改进方向，为量子系统中实现高效测量误差校准提供了可行的思路。

# 目录

一、问题背景

二、问题分析
- 2.1矩阵求逆法
- 2.2迭代贝叶斯展开法 
- 2.3随机矩阵法 
- 2.4单比特张量积法
  
三、解决方案 
- 3.1向量化IBU
- 3.2局部IBU
- 3.3特殊量子态处理 
- 3.4算法步骤

四、结果

五、总结及展望 

参考文献 

A附录主要代码

# 一、问题背景

量子计算依托量子叠加与纠缠等本质特性，有望在组合优化、量子化学模拟和机器学习等领域实现对经典计算的显著加速。最著名的例子是Shor算法，它能在多项式时间内完成大整数因式分解，从而理论上破解基于RSA的公钥加密体系[4]；Grover算法可将无序数据库搜索的查询复杂度由经典的  $O(N)$  降至  $O(\sqrt{N})$  ，对加速优化和搜索问题具有重要意义[5]；HHL（Harrow- Hassidim- Lloyd）算法能够在对数时间内求解稀疏线性方程组，为量子线性代数和量子机器学习中的内核方法提供了新思路[6]。面向化学模拟的变分量子线性求解器（VQE,[7]）和面向组合优化的量子近似优化算法（QAOA,[8]）更是在NISQ（Noisy Intermediate- ScaleQuantum，[1]）阶段展示了可在十几到几十比特上运行的应用原型[9][10]。

尽管理论上具有指数级或平方级加速优势，现实量子硬件仍受制于噪声。超导量子比特的单量子门误差率一般为  $10^{- 4}$  至  $10^{- 3}$  ，双量子门误差率为  $10^{- 3}$  至  $10^{- 2}$  ，而读出误差往往高达  $1\%$  至  $10\%$  。读出过程将量子态通过谐振腔映射为微波信号，经过放大链、数字化处理并判决为经典比特，中间热噪声、信号失真和跨比特串扰会导致读出结果偏离理想概率分布。读出误差既降低了单次测量的置信度，也在多轮反馈或自适应算法中反复累积，严重制约量子算法的可用深度和结果可靠性。如何在有限比特数与显著读出噪声的双重约束下，通过噪声缓解技术恢复近似理想分布，并在NISQ设备上实现具有应用价值的量子计算算法，已成为当下的核心挑战。

# 二、问题分析

本赛题拟通过测量误差缓解（Measurement Error Mitigation, MEM）技术，将含噪声测量分布  $\bar{p}_{\mathrm{noisy}}$  校正为  $\bar{p}_{\mathrm{cal}}$  ，希望尽可能接近理想分布  $\bar{p}_{\mathrm{ideal}}$  ，以总变差距离（TVD）

$$
1 - \mathrm{TVD} = 1 - \frac{1}{2}\sum_{i}\left|p_{i}^{\mathrm{cal}} - p_{i}^{\mathrm{ideal}}\right| \tag{1}
$$

为主要评分指标。同时，为鼓励减少校准数据量，在训练样本消耗上给予额外奖励。针对这一目标，我们先在此归纳现有主要方法。

## 2.1矩阵求逆法

在最简单的读出误差模型中，假设系统的测量误差是线性且时间不变的，则理想分布  $\vec{p}_{\mathrm{ideal}}$  与含噪测量分布  $\vec{p}_{\mathrm{noisy}}$  满足

$$
\vec{p}_{\mathrm{noisy}} = M\vec{p}_{\mathrm{ideal}}, \tag{2}
$$

其中  $M\in \mathbb{R}^{2^n\times 2^n}$  称为校准矩阵，其元素

$$
M_{ij} = P(|i\rangle | |j\rangle) \tag{3}
$$

其中  $P(|i\rangle | |j\rangle)$  表示，测量基态  $|j\rangle$  得到基态  $|i\rangle$  的概率。可通过对所有  $2^{n}$  个计算基态分别制备并多次测量统计获得。若矩阵  $M$  非奇异，即可直接取逆

$$
\vec{p}_{\mathrm{ideal}} = M^{-1}\vec{p}_{\mathrm{noisy}}.
$$

然而，矩阵求逆方法[2]存在以下主要问题：
- 维度指数增长：当比特数  $n$  增大，矩阵维度为  $2^{n}\times 2^{n}$  ，其构造、存储与运算开销均呈指数增长，难以在实际NISQ设备上实现。

- 高维难解与伪逆近似：直接计算  $M^{- 1}$  在维度较大时往往数值病态且计算成本高昂，因而可以采用Moore- Penrose伪逆  $M^{+}$  来替代。然而，由于实验统计误差，伪逆得到的初步校正分布

$$
\vec{p}^{\prime} = M^{+}\vec{p}_{\mathrm{noisy}}
$$

中常含有负值或总和超过1，违背物理非负性与归一性，需要后续截断并重新归一化：

$$
\vec{p}_{\mathrm{cal}} = \max \big(\vec{p},0\big),\qquad \vec{p}_{\mathrm{cal}}\gets \frac{\vec{p}_{\mathrm{cal}}}{\|\vec{p}_{\mathrm{cal}}\|_1}.
$$

尽管矩阵求逆法直观且易于实现，对于小比特数和高保真度的器件仍有应用价值，但在大规模或严苛负概率约束下，其局限日益显现。

## 2.2选代贝叶斯展开法

选代贝叶斯展开（IterativeBayesianUnfolding,IBU，[2]）方法，将测量概率的校准视为在非负概率空间中执行期望最大化（ExpectationMaximization，EM）过程。其核心思想是通过选代优化逐步逼近观测数据的最可能真实分布。给定初始猜测分布  $\vec{p}^{(0)}\geq 0$  (通常为随机归一化向量或均匀分布)，其选代公式如下：

$$
p_j^{(t + 1)} = p_j^{(t)}\sum_{i = 0}^{2^n -1}\frac{M_{ij}p_i^{\mathrm{noisy}}}{\sum_{k = 0}^{2^n - 1}M_{ik}p_k^{(t)}},
$$

每次选代后需进行归一化，使  $\begin{array}{r}\sum_{j}p_{j}^{(t + 1)} = 1 \end{array}$，其代码见附录A。该方法具有以下优缺点：

- 非负性与归一性保障：IBU 中所有运算均基于非负加权和与归一化操作，天然满足物理概率的非负性与归一性，无需额外后处理；
- 收敛性与稳定性：IBU 迭代优化机制确保了算法在读出噪声校正中具有良好的稳定性，能够有效应对较大系统误差的影响，从而提供稳健的校正结果，同时最终迭代至局部极大似然估计（MLE）；
- 计算复杂度：IBU 的计算复杂度较高，尤其在量子比特数较多时，误差响应矩阵  $M$  的尺寸为  $2^n \times 2^n$ ，内存占用和计算时间呈指数增长；
- 收敛速度依赖初始猜测：IBU 的收敛速度在一定程度上依赖于初始分布的选择。如果初始分布与真实分布差异过大，可能需要更多迭代才能达到收敛；
- 统计涨落放大：在迭代过程中，尤其是迭代次数较多时，统计噪声可能被放大，导致校正结果的稳定性下降。因此，需控制迭代次数以平衡精度和稳定性。

## 2.3 随机矩阵法

随机矩阵法通过矩阵  $W \in \mathbb{R}^{2^n \times 2^n}$ ，利用所有  $2^n$  个基态的测量结果进行梯度优化，使得

$$
W p_{\mathrm{noisy},i} \approx \vec{p}_{\mathrm{ideal},i}, \quad i = 0,1,\ldots ,2^n -1,
$$

其中  $\vec{p}_{\mathrm{noisy},i}$  是对基态  $|i\rangle$  测量得到的含噪声的概率分布，  $\vec{p}_{\mathrm{ideal},i}$  是对应的理想分布。具体流程如下：

1. 随机初始化：令  $W^{(0)} \sim \mathcal{U}(-\alpha , \alpha)$  均匀分布初始化或其他方式随机初始化，保证初始矩阵数值稳定；

2. 构造训练集：对每个基态  $|i\rangle$  （共  $2^n$  个），采集足够多的测量样本，得到噪声分布  $\vec{p}_{\mathrm{noisy},i}$  和理想分布  $\vec{p}_{\mathrm{ideal},i}$ ；

3. 定义损失：以总变差距离（TVD）为损失函数  $\mathcal{L}(W)$ ；

4. 梯度优化：使用 Adam 等优化器，在全部  $2^n$  条训练样本上进行若干轮迭代，直至  $\mathcal{L}(W)$  收敛，获得矩阵  $W_{\mathrm{opt}}$ ；

5. 校正应用：对任意待校正噪声分布  $\vec{p}_{\mathrm{noisy}}$ ，直接计算  $\vec{p}_{\mathrm{cal}} = W_{\mathrm{opt}} \vec{p}_{\mathrm{noisy}}$ ，并做非负截断与归一化；

当然，如果在基态之外再加入纠缠态或随机电路态或其他复杂量子态的测量数据，能覆盖更多噪声模式，从而在有限标定样本下提升  $W$  对未知电路的校准效果，但需设计代表性训练态并承受额外制备复杂度。同时，随机矩阵法也存在以下优缺点：

- 灵活可学习：矩阵  $W$  可自动适应复杂多比特串扰与非线性误差分布，不依赖显式模型。

- 存储与训练开销：需存储并优化  $2^n \times 2^n$  参数，存储和计算开销与全矩阵求逆相当，难以直接扩展到极大  $n$ 。

- 样本消耗：训练需所有  $2^n$  个基态的标定数据，样本量与标定成本为指数级。

## 2.4 单比特张量积法

在假设各物理比特测量误差相互独立的前提下，整体校准矩阵  $M$  可近似分解为各比特的二维校准矩阵的张量积。注意张量积顺序要与基态一致，在本题中比特  $n - 1$  视作最左边（高位），比特0视作最右边（低位），则：

$$
p_{\mathrm{noisy}} = M p_{\mathrm{ideal}} \approx \left(R^{(n - 1)} \otimes R^{(n - 2)} \otimes \dots \otimes R^{(0)}\right) p_{\mathrm{ideal}}. \tag{4}
$$

其中第  $k$  个比特的局部校准矩阵

$$
R_{ij}^{(k)} = P\big(|i\rangle_k | |j\rangle_k\big), \quad i,j \in \{0,1\} , \tag{5}
$$

可仅通过制备并测量  $|0\rangle_k, |1\rangle_k$  两种态各若干次统计获得。基于此分解，我们可以按以下步骤进行误差缓解：

1. 逐比特标定：对每个比特  $k$  分别统计得到

$$
R^{(k)} = \begin{pmatrix} P(0|0)_k & P(0|1)_k \\ P(1|0)_k & P(1|1)_k \end{pmatrix} ,
$$

只需  $2n$  条标定电路，标定成本  $O(n)$

2. 张量积近似：构造近似全局校准矩阵

$$
\widetilde{M} = R^{(n - 1)} \otimes R^{(n - 2)} \otimes \dots \otimes R^{(0)}.
$$

避免直接存储和操作  $2^n \times 2^n$  的大矩阵。

3. 校准计算：若各  $R^{(k)}$  可逆，则

$$
\vec{p}_{\mathrm{cal}} = \left((R^{(n - 1)})^{-1} \otimes \dots \otimes (R^{(0)})^{-1}\right) \vec{p}_{\mathrm{noisy}}.
$$

若某些局部矩阵数值病态，可改用伪逆  $R^{(k) + }$

$$
\vec{p}_{\mathrm{cal}} \approx \left(R^{(n - 1) + } \otimes \dots \otimes R^{(0) + }\right) \vec{p}_{\mathrm{noisy}},
$$

并对负值做截断与归一化。

4. 优缺点：
   - 标定电路和内存消耗仅线性增长，比较适合大规模比特系统；
   - 张量积结构可并行化，运算高效；
   - 完全忽略比特间的测量串扰，高阶相关误差无法校正；
   - 在强耦合或串扰显著的体系中近似偏差较大。

根据上述优缺点，该单比特张量积方案可以作为大规模 NISQ 设备上测量误差缓解的首选，但在需要高精度捕捉多比特串扰时，应结合其他方法加以补充。

# 三、解决方案

在详细介绍我们的解决方案前，先查看问题分析中各方案的性能对比。为公平起见，各方案使用每个基态全部的测量结果(sample num = 50000)来构造校准矩阵  $R$ ，数值保留8位小数，代码在macmini  $\mathrm{M}_4$  芯片上运行：

表1各电路在不同校准方案下的1-TVD值  

<table><tr><td>电路序号</td><td>未校准</td><td>矩阵取逆法</td><td>迭代贝叶斯展开</td><td>单比特张量积法(1)</td><td>单比特张量积法(2)</td><td>随机矩阵法</td></tr><tr><td>1</td><td>0.95283571</td><td>0.95152587</td><td>0.95175418</td><td>0.95041508</td><td>0.95202852</td><td>0.951517365</td></tr><tr><td>2</td><td>0.93790416</td><td>0.95943812</td><td>0.95994963</td><td>0.95996494</td><td>0.95875831</td><td>0.959318854</td></tr><tr><td>3</td><td>0.94345259</td><td>0.94623040</td><td>0.94653334</td><td>0.94569124</td><td>0.94618743</td><td>0.946147078</td></tr><tr><td>4</td><td>0.95035142</td><td>0.95525639</td><td>0.95582499</td><td>0.95688618</td><td>0.95814065</td><td>0.955166596</td></tr><tr><td>5</td><td>0.91858845</td><td>0.92805325</td><td>0.92819141</td><td>0.92887428</td><td>0.92726943</td><td>0.928082227</td></tr><tr><td>6</td><td>0.88500000</td><td>0.92660935</td><td>0.92375393</td><td>0.92809608</td><td>0.91903434</td><td>0.925225245</td></tr><tr><td>均值</td><td>0.93135544</td><td>0.94451990</td><td>0.94433458</td><td>0.94498797</td><td>0.94357478</td><td>0.94424289</td></tr><tr><td>耗时(s)</td><td>0.0004</td><td>63.6</td><td>709.08</td><td>0.058</td><td>0.069</td><td>57.7</td></tr></table>

其中单比特张量积法(1)，仅使用  $|0\rangle^{\otimes 9}$  和  $|1\rangle ^{\otimes 9}$  来构造，单比特张量积法(2)使用  $|0\rangle^{\otimes 9},|000000001\rangle,...,|100000000\rangle$ 等10个基态来构造，随机矩阵法使用512个基态来构造训练集。由表1可见：

- (a) 单比特张量积法(1)以极低的标定电路数（2条）和毫秒级开销（0.058s），取得最高的平均分0.94499，且在大多数电路上取得最高或接近最高分；
- (b) 单比特张量积法(2)虽扩充先验测量态，但平均性能略逊于方案(1)，或因方案(1)的  $|0\rangle^{\otimes 9}$ ， $|1\rangle^{\otimes 9}$  测量已隐式捕获了部分比特串扰；
- (c) 迭代贝叶斯展开（IBU）尽管在平均值上低于矩阵取逆法，但在前五条测试电路中均超越伪逆，说明对一般电路表现更为鲁棒（电路6是GHZ态，强耦合），但耗时巨大，需要进行优化；
- (d) 只使用512个基态进行训练的随机矩阵法，其矩阵优化的结果是朝着矩阵取逆法的结果靠近的，训练集需要考虑加上更复杂的量子态；
- (e) 所有校准方案与未校准结果差异均在千分之几以内，这意味着校准带来的额外收益有限，需要考虑比特间窜扰的影响；

正如我们在表1中所示，原始Python实现的IBU在规模稍大的问题上（态空间维度  $m \gtrsim 100$ ）运算速度极慢，甚至无法在合理时间内完成。为了进一步加速，我们给出了两种方案：
- (i) 向量化 IBU: 利用矩阵运算, 尽可能减少显式的 Python 循环; 
- (ii) 局部 IBU: 针对单比特张量积法, 使得每个单比特校准矩阵各自校准概率分布对应的部分, 以此避免张量积后求逆, 减少了计算量和存储需求。

基于上述思考, 我们后续提出了一种结合单比特张量积法与局部 IBU 的校准方案。

## 3.1 向量化 IBU

在原始 IBU 实现中（见附录 A), 每次迭代都要做三重 for 循环, 开销巨大, 这导致时间复杂度为  $O(2^{3n})$  (n 为比特数), 如果考虑到迭代次数  $m$ , 则复杂度为  $O(m \cdot 2^{3n})$  。为了消除 Python 层面的循环开销, 我们将核心更新步骤改写为 NumPy 矩阵运算（见附录 A）。如令

$$
\mathbf{t}_n \in \mathbb{R}^{2^n}, \quad \mathbf{y}^{\mathrm{mes}} \in \mathbb{R}^{2^n}, \quad R \in \mathbb{R}^{2^n \times 2^n},
$$

则 IBU 的更新公式

$$
\mathbf{t}_{n + 1}[j] = \mathbf{t}_n[j] \sum_{i = 1}^{2^n} \frac{R_{i,j} \mathbf{y}_i^{\mathrm{mes}}}{\sum_{k = 1}^{2^n} R_{i,k} \mathbf{t}_n[k]}
$$

可重写为

$$
\mathbf{d} = R \mathbf{t}_n, \quad \rho = \frac{\mathbf{y}^{\mathrm{mes}}}{\mathbf{d}}, \quad \mathbf{t}_{n + 1} = \left(R^T \rho\right) \odot \mathbf{t}_n,
$$

其中“  $\odot$  ”表示逐元素相乘。

由于每次迭代只包含两次  $2^n \times 2^n$  矩阵与  $2^n$  向量乘法, 单次迭代复杂度降为  $O(2^{2n})$ , 总体复杂度为  $O(m \cdot 2^{2n})$ , 相较于  $O(m \cdot 2^{3n})$  实现了指数级的降维加速。与表 1 中的 709.08s 对比, 在相同实验条件下, 运行时间仅需 68.04s, 大约是 10 倍的提升, 且数值输出一致。如果每个基态只测量 1000 次, 则仅需 2s。

## 3.2 局部 IBU

对于大规模的量子系统, 完整的校准矩阵  $R \in \mathbb{R}^{2^n \times 2^n}$  的存储和计算代价随着量子比特数  $n$  的增长呈指数级增加。然而, 在许多实际设备中, 忽略比特间串扰后, 读出误差矩阵可以近似分解为单比特读出误差的张量积形式:

$$
R \approx R^{(n - 1)} \odot R^{(n - 2)} \odot \ldots \odot R^{(0)},
$$

其中每个  $R^{(k)} \in \mathbb{R}^{2 \times 2}$  表示单个量子比特的校准矩阵。通过这种分解, 可以避免显式构造  $2^n \times 2^n$  的矩阵  $R$ , 从而显著降低存储需求和计算复杂度。我们可以直接高效地计算  $R \mathbf{t}_n$  和  $R^T \rho$ , 从而加速 IBU 算法

为了具体实现这一高效计算过程, 我们将长度为  $2^n$  的向量（如当前概率分布  $\mathbf{t}_n$  或比率向量  $\rho$  ）视为大小为  $(2, 2, \ldots , 2)$  的  $n$  维张量（见附录 A）。此时, 计算

$$
\mathbf{d} = R \mathbf{t}_n
$$

可以通过逐比特收缩的方式完成。具体步骤如下：

(I) 初始化重塑

将一维向量  $t_n$  重塑为张量

$$
T^{(0)}[i_{n - 1},\ldots ,i_0] = t_n\big(i_{n - 1}2^{n - 1} + \dots +i_0\big).
$$

(II) 逐比特收缩

对每个比特  $k = 0,1,\ldots ,n - 1$  依次执行：

(a) 轴交换：用 moveaxis 将第  $k$  维移到最后一维，得到张量  $T^{\prime (k)}$

(b) 矩阵-向量乘法：对形状  $(2^{n - 1},2)$  的最后一维应用

$$
T^{\prime \prime (k)}[\ldots ,j_k] = \sum_{i_k = 0}^{1}R_{j_k,i_k}^{(k)}T^{\prime (k)}[\ldots ,i_k],
$$

这相当于对每个固定的其余索引，进行一次  $2\times 2$  的乘法。

(c) 恢复形状：将  $T^{\prime \prime (k)}$  reshape 回  $(2,\ldots ,2)$ ，再用 moveaxis 把最后一维移回第  $k$  维，得到下一步的输入张量  $T^{(k + 1)}$ 。

(III) 展平输出

完成  $n$  次收缩后得到

$$
D[i_{n - 1},\ldots ,i_0] = T^{(n)}[i_{n - 1},\ldots ,i_0],
$$

将其展平，即得目标向量  $\mathbf{d}$

上述过程中，数值运算是对每个比特一次  $2\times 2$  矩阵- 向量乘法，总共  $n$  次，处理  $2^n$  个元素，复杂度  $O(n\cdot 2^n)$ ，总体复杂度为  $O(m\cdot n\cdot 2^n)$ 。同理，将每个  $R^{(k)}$  替换为其转置  $R^{(k)T}$ ，即可以同样的方式计算

$$
R^T\pmb {\rho},\quad \pmb {\rho} = \frac{\pmb{y}^{\mathrm{mes}}}{d}. \tag{6}
$$

## 3.3 特殊量子态处理

由于第6条测试线路是GHZ态（见图1)，仅由  $|0\rangle^{\otimes 9}$  和  $|1\rangle^{\otimes 9}$  来构成。因此，我们针对GHZ态做特殊校准，仅构造一个  $2\times 2$  的校准矩阵  $R_{\mathrm{GHZ}}$ ，并基于测量统计进行简单校准。代码见附录A，主要流程如下：

1. 测量  $|0\rangle^{\otimes 9}$  和  $|1\rangle^{\otimes 9}$ ，并统计概率

$$
P_{0|0} = P(|0\rangle ||0\rangle),\quad P_{1|1} = P(|1\rangle ||1\rangle), \tag{7}
$$

2. 构造校准矩阵  $R_{\mathrm{GHZ}}$

$$
R_{\mathrm{GHZ}} = \left( \begin{array}{cc}P(0|0) & 1 - P(1|1) \\ 1 - P(0|0) & P(1|1) \end{array} \right), \tag{8}
$$

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/996f1dfd-2136-44d8-9813-5ea105639ceb/fe18bdba33c65db5ef7074a4ffc29603b5e1eeb455d4af3e485e11ec561fa10b.jpg)  
图1 GHZ态的电路形式

3. 从未经校准的GHZ态测量分布  $\vec{p}_{\mathrm{GHZ}}$  (有512项)，取  $|0\rangle^{\otimes 9}$  和  $|1\rangle^{\otimes 9}$  的概率出来，组成新的概率分布向量

$$
\vec{p}_{\mathrm{new}} = (\vec{p}_{\mathrm{GHZ}}[0],\vec{p}_{\mathrm{GHZ}}[511])^{\mathrm{T}}; \tag{9}
$$

4. 噪声校准，

$$
\vec{p}_{\mathrm{cal}} = R_{\mathrm{GHZ}}^{-1}\vec{p}_{\mathrm{new}}; \tag{10}
$$

并将负分量截为0、再整体归一化。其余510个分量置零，得到最终的修正分布。

## 3.4算法步骤

我们的整体校准与误差消除流程分为三个阶段：基准态采样、校准矩阵构建，以及测量概率校正。下文给出算法实施步骤：

#### 1. 基态采样

- 测量两个全比特同态：
  - $|0\rangle^{\otimes n}$  ，采样次数  $N$  
  - $|1\rangle^{\otimes n}$  ，采样次数  $N$  0
- 调用函数get_data获取测量结果

#### 2. 单比特校准矩阵构建

对每个比特  $k = 0,\ldots ,n - 1$

(a）从测量结果中，统计每个比特上“0测为  $0^{\gg}$  “1测为  $1^{\gg}$  的个数  $N_0,N_1$  (b）加入Beta先验  $(\alpha ,\beta)$  (给每种结果都额外加上  $(\alpha +\beta)$  次虚拟观测)，估计

$$
P(0|0) = \frac{N_0 + \alpha}{N + \alpha + \beta},\quad P(1|1) = \frac{N_1 + \alpha}{N + \alpha + \beta};
$$

(c) 构造单比特校准矩阵

$$
R^{(k)} = \left( \begin{array}{cc}P(0|0) & 1 - P(1|1) \\ 1 - P(0|0) & P(1|1) \end{array} \right).
$$

按比特顺序将  $\{R^{(k)}\}$  有入列表single_qubit_mats。在代码中，single_qubit_mats[0]是列表最左边高位比特的校准矩阵  $R^{(8)}$ 。

#### 3. 测量概率校正

对输入的  $M$  条电路测量分布  $\{\mathbf{p}^{(i)}\}_{i = 1}^{M}$

(a) 若为随机电路：

i. 随机生成归一化初始分布  $\mathbf{t}^{(0)}$ ；

ii. 迭代  $m$  步局部IBU（见3.2）；

iii. 取  $t^{(m)}$  作为校正后分布。

(b) 若电路为GHZ态（其他特殊态可类似处理），则

i. 取出  $\mathbf{p}^{(i)}[0]$  和  $\mathbf{p}^{(i)}[2^n -1]$ ；

ii. 构造  $2\times 2$  校准矩阵  $R_{\mathrm{GHZ}}$ ，见公式(8)；

iii. 求逆并校正：  $\bar{q} = R_{\mathrm{GHZ}}^{- 1}\left(p_0,p_1\right)^T$ ，截负、归一化，重组回  $2^n$  维，其他分量置零。

#### 4. 输出

返回所有电路的校正分布列表  $\{\hat{\mathbf{p}}^{(i)}\}$  以及总采样数 TRAIN_SAMPLE_NUM。

# 四、结果

接下来，给出我们的算法在不同参数下的性能，  $|0\rangle^{\otimes 9}$  和  $|1\rangle^{\otimes 9}$  均使用(sample num = 10000)来构造校准矩阵  $R$ ，数值保留8位小数，代码在macmini  $\mathbf{M}_4$  芯片上运行：
- 参数1：  $\alpha = \beta = 0$  ，single_qubit_mats  $= \{R^{(8)},R^{(8)},\dots ,R^{(0)}\}$  
- 参数2：  $\alpha = \beta = 0$  ，single_qubit_mats  $= \{R^{(0)},R^{(1)},\dots ,R^{(8)}\}$  
- 参数3：  $\alpha = \beta = 25$  ，single_qubit_mats  $= \{R^{(0)},R^{(1)},\dots ,R^{(8)}\}$  
- 参数4：  $\alpha = \beta = 25$  ，single_qubit_mats  $= \{R^{(8)},R^{(8)},\dots ,R^{(0)}\}$

由表2可以看到，

(a) 使用单比特校准矩阵 + 局部IBU的混校准方案，比未校准相比，基础分能提高  $240\sim 260$  分左右；

(b) 对比参数1和参数2，发现单比特校准矩阵存储的顺序更换一下，分数提高了13分，但参数1的存储顺序是正确的；

(c) 加入Beta先验  $(\alpha ,\beta)$  分数有所提高大约8分左右；

表2 算法在不同参数下的性能  

<table><tr><td>电路序号</td><td>未校准</td><td>参数1</td><td>参数2</td><td>参数3</td><td>参数4</td></tr><tr><td>1</td><td>0.95283571</td><td>0.95041575</td><td>0.95258610</td><td>0.95106677</td><td>0.94893369</td></tr><tr><td>2</td><td>0.93790416</td><td>0.96047973</td><td>0.96139625</td><td>0.96650620</td><td>0.96408786</td></tr><tr><td>3</td><td>0.94345259</td><td>0.94462053</td><td>0.94390579</td><td>0.94378003</td><td>0.94419276</td></tr><tr><td>4</td><td>0.95035142</td><td>0.95867538</td><td>0.96010489</td><td>0.95905725</td><td>0.95910534</td></tr><tr><td>5</td><td>0.91858845</td><td>0.92903442</td><td>0.93387779</td><td>0.93642271</td><td>0.93138787</td></tr><tr><td>6</td><td>0.88500000</td><td>0.99044348</td><td>0.99044348</td><td>0.99044348</td><td>0.99044348</td></tr><tr><td>均值</td><td>0.93135544</td><td>0.95561155</td><td>0.95705238</td><td>0.95787941</td><td>0.956358498</td></tr><tr><td>总分</td><td>9363.5539</td><td>9606.0764</td><td>9620.4848</td><td>9628.7550</td><td>9613.5459</td></tr></table>

接下来，我们对所有512个基态各自进行10000次采样，分别用全局矩阵求逆、单比特校准矩阵与局部IBU的混合方案构造校准矩阵，并用另一组10000次采样评估校准效果。图2显示：矩阵逆法平均  $1 - \mathrm{TVD} = 0.99516$  ，而局部IBU为0.99138。局部IBU略低的主要原因在于它仅基于单比特独立假设，近似忽略了比特间的高阶串扰，且迭代次数受限未完全收敛到全局最优解；但即便如此，其在绝大多数基态上均能保持 $1 - \mathrm{TVD} > 0.98$  ，且其计算与存储开销仅为全局矩阵逆法的一小部分，充分体现了在资源受限环境下的实用价值。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/996f1dfd-2136-44d8-9813-5ea105639ceb/041be96f5d24e4a41102af9b3590c988eb6fcfd75ca7258cf9d20b2c11157b37.jpg)  
图2各基态在矩阵取逆法和局部IBU法下的性能对比

此外，在第2.2节中我们提到，迭代次数过多可能会导致校准结果不稳定。图3使用向量化IBU，基于512个基态各50000次测量构造校准矩阵，展示了不同迭代次数下的  $1000\cdot (1 - \mathrm{TVD})$  得分随迭代步数的变化。可以看到，当迭代次数超过70次后，得分突然下降。因此，为了兼顾精度与稳定性，IBU迭代次数应控制在20次以内。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/996f1dfd-2136-44d8-9813-5ea105639ceb/1ebf8b9f7141dc8e939eb0f05854529dd6595f5418d4d133b3dc1465bcec7b82.jpg)  
图3 向量化IBU：不同迭代次数对基础的影响

# 五、总结及展望

本工作针对NISQ设备上的读出误差缓解问题，提出并实现了一种单比特校准矩阵与局部IBU混合校准的方案。主要贡献包括：

- 最小化标定代价：仅需制备并测量  $|0\rangle^{\otimes n}$  和  $|1\rangle^{\otimes n}$ ，即可构建  $n$  比特系统的单比特校准矩阵，样本开销远小于传统的  $2^{n}$  个基态标定；
- 高效校准算法：将校准矩阵张量积近似与迭代贝叶斯展开相结合，通过逐比特收缩实现  $Rt$ ， $R^{T}\rho$  的运算，将IBU单次迭代复杂度从  $O(2^{2n})$  降至  $O(n\cdot 2^{n})$ ，在9比特、5次迭代下耗时仅百毫秒级；
- 鲁棒性与精度：在6条典型测试电路（包括随机电路与GHZ态）上，平均1-TVD提升至0.956左右，接近或超过矩阵取逆法与全局IBU；
- 全面基态评估：对512个基态分别采样10000次进行校准评估，局部IBU平均  $1 - \mathrm{TVD} = 0.99138$ （略低于矩阵求逆法的0.99516），主要因忽略高阶串扰和迭代未完全收敛，但在绝大多数基态上仍保持  $1 - \mathrm{TVD} > 0.98$ ，且计算更高效；
- 特殊态处理：针对GHZ态仅含两种输出，设计专用  $2\times 2$  校准矩阵，有效捕获全局一致性误差，保证高保真率。
- 扩展性 在比特数进一步增长时，单比特校准矩阵与局部 IBU 的优势将更加明显，可快速的实现轻量级的校准。

尽管本方案在标定代价与运行速度上均表现优异，但仍有部分可改进或未完成的工作：

(1) 高阶串扰校准：本文假设各比特读出误差相互独立，忽略了多比特耦合效应。在误差严重耦合的体系中，可按物理或拓扑邻近关系将比特分组，分别构建每组的校准矩阵，再结合局部 IBU 进行校正，以捕捉组内串扰并提升整体精度。

(2) 子空间截断与稀疏化：借鉴 Nation 等人 [3] 的 M3 思路，将完整的  $2^{n} \times 2^{n}$  校准矩阵截断到仅包含测得噪声输出比特串的子矩阵  $\tilde{A}$ ，并可按 Hamming 距离阈值  $D$  保留近邻元素——既保证列重归一化，又大幅提升稀疏度。该方法尚未在本题环境中复现，后续可评估其对得分与性能的实际影响。

(3) 随机矩阵 / 机器学习：在表 1 中，随机矩阵只使用基态构造训练集，这里我们尝试在训练集中添加本题需要校准的量子态（随机采样），进行训练，结果如下：

表3 随机矩阵法在不同训练集下的1-TVD值  

<table><tr><td>电路序号</td><td>本校准</td><td>随机矩阵法(1)</td><td>随机矩阵法(2)</td><td>随机矩阵法(3)</td></tr><tr><td>1</td><td>0.95283571</td><td>0.951517365</td><td>0.95193698</td><td>0.98234604</td></tr><tr><td>2</td><td>0.93790416</td><td>0.959318854</td><td>0.96486278</td><td>0.98543302</td></tr><tr><td>3</td><td>0.94345259</td><td>0.946147078</td><td>0.97581739</td><td>0.98778582</td></tr><tr><td>4</td><td>0.95035142</td><td>0.955166596</td><td>0.97565136</td><td>0.98833539</td></tr><tr><td>5</td><td>0.91858845</td><td>0.928082227</td><td>0.92947498</td><td>0.98223997</td></tr><tr><td>6</td><td>0.88500000</td><td>0.925225245</td><td>0.99247086</td><td>0.99376344</td></tr><tr><td>均值</td><td>0.93135544</td><td>0.94424289</td><td>0.96503573</td><td>0.98665061</td></tr><tr><td>耗时(s)</td><td>0.0004</td><td>57.7</td><td>57.51</td><td>53.3</td></tr></table>

其中，随机矩阵法 (1) 训练集包含 512 个基态（使用所有测量结果），初始矩阵随机生成；随机矩阵法 (2) 在随机矩阵法 (1) 的基础上，训练集添加需要校准的量子态（从每个量子态 50000 个测量结果中采样 10000 个，重复 20 次）；随机矩阵法 (3) 训练集只包含需要校准的量子态，初始矩阵为矩阵取逆法的结果。可以看到，添加需要校准的量子态到训练集中时，获得 1- TVD 的均值有所提高的，但会影响到对基态的校准结果，如图 4，各方案基态 1- TVD 的均值分别为：0.99712350, 0.995783384, 0.487089374。后续可尝试其他机器学习或深度学习模型或，对测量噪声分布进行拟合与校正。

总之，本工作在保证极低标定开销的前提下，融合单比特校准矩阵与 IBU 方法，为 NISQ 设备上的读出误差缓解提供了一条高效可行的路径。未来可在更丰富的量子硬件与应用场景中检验与完善，以进一步推动实用量子计算的发展。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/996f1dfd-2136-44d8-9813-5ea105639ceb/81b12152c7438ea362db42059358dd013d8b92865885cfad44447f44050fdd14.jpg)  
图4 各基态在随机矩阵法下的1-TVD值

# 参考文献

[1] PRESKILL J. Quantum computing in the insq era and beyond[J]. Quantum, 2018, 2: 79.

[2] NACHMAN B, URBANEK M, DE JONG W A, et al. Unfolding quantum computer readout noise[J]. npj Quantum Information, 2020, 6(1): 84.

[3] NATION P D, KANG H, SUNDARESAN N, et al. Scalable mitigation of measurement errors on quantum computers[J]. PRX Quantum, 2021, 2(4): 040326.

[4] SHOR P W. Algorithms for quantum computation: discrete logarithms and factoring[C]// Proceedings 35th annual symposium on foundations of computer science. Ieee, 1994: 124- 134.

[5] GROVER L K. A fast quantum mechanical algorithm for database search[C]//Proceedings of the twenty- eighth annual ACM symposium on Theory of computing. 1996: 212- 219.

[6] HARROW A W, HASSIDIM A, LLOYD S. Quantum algorithm for linear systems of equations[J]. Physical review letters, 2009, 103(15): 150502.

[7] PERUZZO A, MCCLEAN J, SHADBOLT P, et al. A variational eigenvalue solver on a photonic quantum processor[J]. Nature communications, 2014, 5(1): 4213.

[8] FARHI E, GOLDSTONE J, GUTMANN S. A quantum approximate optimization algorithm[A]. 2014.

[9] GUO S, SUN J, QIAN H, et al. Experimental quantum computational chemistry with optimized unitary coupled cluster ansatz[J]. Nature Physics, 2024, 20(8): 1240- 1246.

[10] DECROSS M, CHERTKOV E, KOHAGEN M, et al. Qubit- reuse compilation with mid- circuit measurement and reset[J]. Physical Review X, 2023, 13(4): 041057.

# 附录A 主要代码

### 原始IBU

```python
def IBU(ymes: np.ndarray, t0: np.ndarray, Rin: np.ndarray, n: int) -> np.ndarray:
    """
    Args:
        ymes: Measured probability distribution
        t0: Initial guess for true distribution
        Rin: Response matrix (measured vs true)
        n: Number of iterations
    
    Returns:
        Mitigated probability distribution
    """
    tn = t0
    for _i in range(n):
        print(f'IBU iteration = {_i}')
        out = np.zeros(t0.shape)
        for j in range(len(t0)):
            mynum = 0.
            for i in range(len(ymes)):
                myden = sum(Rin[i][k] * tn[k] for k in range(len(t0)))
                if myden > 0:
                    mynum += Rin[i][j] * tn[j] * ymes[i] / myden
            out[j] = mynum
        tn = out
    return tn


```

### 向量化 IBU

```python
def IBU_vectorized(ymes: np.ndarray, t0: np.ndarray, Rin: np.ndarray, n: int) -> np.ndarray:
        """ 向量化实现的 IBU 算法 """
        tn = t0.copy()
        for i in range(n):
            # 计算分母 (对每个 i 计算 sum(Rin[i][k] * tn[k]))
            den = Rin @ tn  # 矩阵乘法计算所有分母
            # 避免除零
            den = np.where(den > 0, den, 1e-10)
            # 计算 ymes[i] / den[i] 的比率，测量噪声 ymes
            ratio = ymes / den
            # 计算更新后的 tn
            # tn = np.sum(Rin.T * ratio, axis=1) * tn
            tn = (Rin.T @ ratio) * tn            
        return tn    
```


### 局部 IBU

```python
# 单比特校准矩阵使用的 IBU 函数
def apply_response_matrix(vector, single_qubit_mats):
    """
    快速应用张量积结构的校准矩阵
    
    参数:
        vector: 输入向量，形状为(2^9,)
        single_qubit_mats: 9个单量子比特校准矩阵的列表
        
    返回:
        应用校准矩阵后的向量，形状与输入相同
    """
    # 将输入向量重塑为9维张量，每个维度大小为2
    tensor = vector.reshape([2]*9)
    # 应用每个量子比特的校准矩阵（按构造时的顺序）
    for k in range(9):
        mat = single_qubit_mats[k]
        # 将当前量子比特对应的轴移到最后一个位置
        tensor = np.moveaxis(tensor, k, -1)
        # 重塑为 (..., 2) 并进行矩阵乘法
        original_shape = tensor.shape
        tensor = tensor.reshape(-1, 2) @ mat.T
        # 恢复形状并移回轴位置
        tensor = tensor.reshape(original_shape)
        tensor = np.moveaxis(tensor, -1, k)
    return tensor.reshape(-1)

def IBU_optimized(ymes, t0, single_qubit_mats, n_iter=5):
    """
    优化后的IBU(迭代贝叶斯展开)算法
    
    参数:
        ymes: 测量结果向量
        t0: 初始概率分布
        single_qubit_mats: 9个单量子比特校准矩阵的列表
        n_iter: 迭代次数，默认为5
        
    返回:
        校准后的概率分布
    """
    # 预处理转置矩阵（用于后续计算）
    mats_T = [mat.T for mat in single_qubit_mats]
    tn = t0.copy()
    for _ in range(n_iter):
        # 计算分母：R @ tn
        den = apply_response_matrix(tn, single_qubit_mats)
        den = np.where(den > 0, den, 1e-10)
        # 计算比率：ymes / den
        ratio = ymes / den
        # 计算分子：R.T @ ratio
        rt_ratio = apply_response_matrix(ratio, mats_T)
        # 更新估计值
        tn = tn * rt_ratio
        tn /= tn.sum()  # 保持归一化
    return tn    
```

### GHZ

```python
# 第6个线路(GHZ态)特殊处理
ghz_states = [0, 511]  # 全0和全1
ghz_R = np.zeros((2,2))
for j, state in enumerate(ghz_states):
    bitstrings = measured_data[state]
    ghz_counts = np.zeros(2)
    for bs in bitstrings:
        # GHZ态的特殊处理 - 检查所有比特是否一致
        all_zero = np.all(bs == 0)
        all_one = np.all(bs == 1)
        if all_zero:
            ghz_counts[0] += 1
        elif all_one:
            ghz_counts[1] += 1
    if state == 0:
        ghz_R[0, j] = ghz_counts[0] / len(bitstrings)
        ghz_R[1, j] = 1 - ghz_R[0, j]
    else:
        ghz_R[1, j] = ghz_counts[1] / len(bitstrings)
        ghz_R[0, j] = 1 - ghz_R[1, j]
# 应用GHZ校准
ghz_p = np.array([measure_prob[0], measure_prob[-1]])
ghz_p_corrected = np.linalg.pinv(ghz_R) @ ghz_p
ghz_p_corrected = np.maximum(ghz_p_corrected, 0)
ghz_p_corrected /= ghz_p_corrected.sum()
_prob = np.zeros(dim)
_prob[0] = ghz_p_corrected[0]
_prob[-1] = ghz_p_corrected[1]



```