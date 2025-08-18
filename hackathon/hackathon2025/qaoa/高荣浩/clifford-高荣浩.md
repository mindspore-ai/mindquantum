# 基于Clifford和模拟退火的启发式量子算法求解

高荣浩(f.g.m.leonardo@gmail.com, y3lin@spinq.cn, chenronghang2020@outlook.com)

# 摘要

本研究探讨了一种基于Clifford线路与模拟退火优化的量子- 经典混合算法，用于在含噪声中等规模量子（NISQ）计算环境下高效求解Ising模型问题。该算法通过整合Clifford线路的经典可模拟特性与模拟退火算法的全局优化能力，构建了一个新型的量子- 经典协同计算框架。研究采用了四种量子态初始化策略（无翻转态、全翻转态、随机态和h导向态)，并设计了包含单比特翻转和双比特翻转的双层级状态转移机制。在优化过程中，算法实现了动态温度调控（  $T_{k + 1} = 0.993T_{k}$  ）和自适应马尔可夫链长度调整，同时引入热平衡判定准则和能量监测窗口（  $W = \lfloor 100 / \log (T + 1)\rfloor$  ）来保证收敛性。特别值得注意的是，通过推导增量梯度更新公式（  $\delta_{j}\leftarrow \delta_{j} + 4J_{ji}s_{j}s_{i}$  )，显著降低了计算复杂度。数值模拟结果表明，该算法能够有效制备Ising 哈密顿量（  $\begin{array}{r}H = \sum_{i}h_{i}Z_{i} + \sum_{i< j}J_{ij}Z_{i}Z_{j}) \end{array}$  的低能态，在本次赛题中获得179882.8976的高分，为解决NISQ时代的组合优化问题提供了新的理论视角和技术方案。

# 目录

一、问题背景

二、问题分析

三、解决方案
- 3.1 Ising 问题形式化 
- 3.2 算法流程
- 3.2.1 初始化策略
- 3.3 模拟退火优化
  - 3.3.1 参数初始化
  - 3.3.2 状态转移机制
  - 3.3.3 热平衡判定 
  - 3.3.4 收敛性保障
- 3.4 电路与状态更新机制
  - 3.4.1 单比特翻转操作
  - 3.4.2 双比特翻转操作
  - 3.4.3 终止条件判定

四、总结及展望

参考文献

# 一、问题背景

近年来，随着量子计算硬件的快速发展，含噪声中等规模量子（Noisy Intermediate- Scale Quantum, NISQ [1]）处理器为实现特定问题的量子优越性提供了新的可能性。在这一背景下，变分量子算法（Variational Quantum Algorithms, VQAs[2, 3]）因其对噪声的鲁棒性和灵活性，成为解决量子化学模拟和组合优化问题的重要方法。这类算法通过参数化量子电路构建目标函数，并利用经典优化器迭代调整参数，最终将初始量子态演化为包含问题近似解的叠加态。其中，量子近似优化算法（Quantum Approximate Optimization Algorithm, QAOA [4]）作为 VQA 的典型代表，已在组合优化领域展现出巨大潜力，特别是在 Ising 模型问题的求解上取得了显著进展。

Ising 模型最初是为描述磁性材料的相变行为而提出的统计物理模型，其哈密顿量可表示为自旋变量间的相互作用。该模型不仅能够刻画铁磁体、反铁磁体等物质的宏观性质，还被广泛映射到组合优化问题中，如最大割问题（MaxCut）、旅行商问题（TSP）和蛋白质折叠等。由于这些问题在金融、物流、生物信息学等领域具有重要应用价值，高效求解 Ising 模型成为学术界和工业界共同关注的课题。

在经典计算领域，Ising 模型的求解算法主要包括蒙特卡洛模拟、模拟退火、分支定界法等启发式方法，以及基于半正定规划（SDP）的近似算法。然而，随着问题规模的扩大，这些方法往往面临计算复杂度指数增长或求解精度不足的瓶颈。量子计算因其天然的并行性和纠缠特性，为突破这一瓶颈提供了新思路。QAOA 通过构造含参量子线路来逼近 Ising 模型的基态，理论上可在多项式时间内获得近似解。但受限于当前 NISQ 设备的噪声水平和量子比特数，QAOA 的电路深度和问题规模仍受到严格限制，且其性能受制于参数优化中的“贫瘠高原”（Barren Plateaus）[5]现象。

值得注意的是，Clifford 线路因其经典可模拟性而成为探索量子算法设计的重要工具。这类线路由 Hadamard 门、相位门、CNOT 门及泡利门等组成，虽然不能实现通用量子计算，但仍可生成多体纠缠态，并用于验证量子算法的基本特性。近期研究表明，浅层 Clifford 线路在 MaxCut 等组合优化问题中已展现出优于随机猜测的性能，这为开发新型混合量子- 经典算法提供了启示。通过结合经典算法的高效存储与更新机制（如贪心策略、动态规划）与 Clifford 线路的纠缠资源，有望在 NISQ 时代实现更鲁棒的优化求解方案。

本研究聚焦于Ising模型的量子算法求解，旨在系统探索基于Clifford线路的优化方法。具体而言，我们将分析Clifford线路在构造问题哈密顿量基态近似解时的表达能力，建立其与经典优化算法的协同框架，并通过数值模拟验证混合算法的性能优势。这项工作不仅有助于理解量子资源在组合优化中的核心作用，还将为未来容错量子计算机上的算法设计奠定基础。

# 二、问题分析

本赛题以基于Clifford门的量子组合优化算法设计为核心，目标是通过量子线路的设计与实现，解决普适的Ising问题及其特殊形式（如最大割问题）。赛题背景强调了当前量子算法（如变分量子算法（VQA）和量子近似优化算法（QAOA））在硬件规模受限、噪声干扰和经典模拟复杂性等方面的限制，而Clifford线路因其在经典计算机上的高效模拟能力以及纠缠量子态的处理能力，提供了一种新的研究思路。设计基于Clifford门的量子优化算法不仅能够探索新型量子算法的潜力，还可以揭示Clifford线路与非Clifford线路在性能上的差异，为解决大规模组合优化问题提供新的工具。

赛题的核心是如何设计量子线路来求解Ising问题，其哈密顿量形式为：

$$
H = \sum_{i}h_{i}Z_{i} + \sum_{i< j}J_{ij}Z_{i}Z_{j},
$$

其中  $h_i$  表示第  $i$  个变量的权重，  $J_{ij}$  表示变量间的相互作用权重，  $Z_{i}$  是作用在第  $i$  个比特上的Pauli- Z算符。这一问题的目标是通过量子线路找到使能量最小化的状态。

赛题提供了一个基于随机ADAPT- Clifford算法的框架，其基本流程为：初始时随机选择一个算符作用在某比特上，之后根据当前量子态的能量变化梯度，从算符库中选择最优算符进行更新。参赛者需要在该框架下，优化量子线路的设计与实现。

针对赛题的目标与问题定义，可以提出以下方法：

1. 算符库优化：探索不同Clifford门的组合，或者扩展算符库，选择更加适合特定问题的算符（如针对特定梯度变化的门操作）。
2. 梯度计算优化：设计高效的梯度计算策略，用于快速选择最优算符。可以结合经典优化器提升梯度计算的效率。
3. 并行化设计：利用并行计算技术（如多线程或向量化操作）加速问题求解，特别是对大规模问题的处理。
4. 量子与经典结合：充分利用经典优化算法的变量存储和更新能力，同时在量子线路中体现纠缠和叠加特性，实现两者的深度融合。
5. 贪心策略扩展：在随机ADAPT-Clifford算法框架下，改进贪心策略，探索更高效的局部优化路径，逐步逼近全局最优解。

通过以上优化策略，可以提升量子线路设计的效率与求解结果的精度，从而更好地解决复杂的Ising问题，并实现赛题目标。

# 三、解决方案

我们提出一种量子- 经典混合算法，利用Clifford电路求解Ising问题。该方法结合多种初始化策略和模拟退火优化，在每次迭代中同时探索单量子比特和双量子比特操作，最终输出可制备Ising哈密顿量低能态的Clifford电路。

## 3.1 Ising问题形式化

Ising哈密顿量定义为：

$$
H = \sum_{i}h_{i}Z_{i} + \sum_{i< j}J_{ij}Z_{i}Z_{j}
$$

其中：

-  $Z_{i}$  为泡利  $Z$  算符
-  $h_{i}$  （对角元素）和  $J_{ij}$  （非对角元素）来自输入矩阵  $Q\_ triu$

## 3.2 算法流程

### 3.2.1 初始化策略

我们采用了四种初始态生成策略：

1）无翻转态：所有量子比特初始化为 $|0\rangle$（经典态：$+1$ ）

2）全翻转态：所有量子比特初始化为 $|1\rangle$（经典态：$-1$ ）

3）随机态：随机选择量子比特子集施加X门翻转

4）h导向态：当  $h_{i} > 0$  时翻转量子比特（优化局域场）


## 3.3 模拟退火优化

采用经典模拟退火框架与量子电路操作相结合的方式，具体实现流程如下：

### 3.3.1 参数初始化

- 初始温度：  $T_{0} = 1.0$  （归一化单位）
- 降温系数：  $\alpha = 0.993$  （每迭代步按  $T_{k + 1} = \alpha T_{k}$  衰减）
- 马尔可夫链长度：  $L = 100$  （每个温度下最大尝试次数）
- 终止条件：  $T_{\mathrm{final}} = 10^{-3}$  或连续10次无改进

### 3.3.2 状态转移机制

#### 1. 邻域搜索：

- 单比特扰动：随机选取1个量子比特翻转

$$
\Delta E_{i} = 2s_{i}(h_{i} + \sum_{j\neq i}J_{ij}s_{j})
$$

- 双比特扰动（每3次迭代）：随机选取CNOT作用对

$$
\Delta E_{ij} = 2(s_ih_i + s_jh_j) + 4J_{ij}s_is_j + 2\sum_{k\neq i,j}(J_{ik}s_i + J_{jk}s_j)s_k
$$

#### 2. 接受概率计算：

$$
P_{\mathrm{accept}} = \min \left(1,e^{-\Delta E / T}\right) \tag{1}
$$

其中能量差计算考虑：

$$
\Delta E = E_{\mathrm{new}} - E_{\mathrm{current}} = \sum_{i\in \mathrm{flips}}\delta_{i} + \sum_{i< j\in \mathrm{pairs}}4J_{ij}s_{i}^{\prime}s_{j}^{\prime}
$$

#### 3. 量子电路更新：

- 单比特翻转：追加  $X$  门并更新经典寄存器
- 双比特翻转：构建子电路  $H_{i}CNOT_{ij}Z_{j}CNOT_{ij}H_{i}$ 
- 每次接受新状态后同步更新：

$$
\delta_{i}\leftarrow -2s_{i}^{\prime}(h_{i} + \sum_{j}J_{ij}s_{j}^{\prime})
$$

### 3.3.3 热平衡判定

采用动态调整策略：

$$
L_{k} = \left\{ \begin{array}{ll}L_{k - 1}\times 1.1 & \text{若接受率} >30\% \\ L_{k - 1}\times 0.9 & \text{若接受率} < 15\% \\ L_{k - 1} & \text{否则} \end{array} \right.
$$

### 3.3.4 收敛性保障

- 记录历史最优解，避免退火过程中的解退化
- 当  $T< 0.1T_{0}$  时启动局部搜索模式：
  - 仅接受  $\Delta E< 0$  的移动
  - 优先尝试梯度下降方向的操作
- 最终输出所有温度下的最优电路配置

## 3.4 电路与状态更新机制

### 3.4.1 单比特翻转操作

- 电路更新：
  - 追加量子门：  $\mathcal{C}\leftarrow \mathcal{C}\circ X_{i}$  
  - 矩阵表示：  $X_{i}=\binom{0}{1}\underset{i}{\times}I_{\{1,\ldots,n\}\backslash i}$

- 经典状态更新：
  - $\mathcal{s}_i \leftarrow \mathcal{s}_i$ (布尔值取反)
- 增量梯度更新：

$$
\delta_{j}\leftarrow \left\{ \begin{array}{ll} - \delta_{i} & j = i \\ \delta_{j} + 4J_{ji}s_{j}s_{i} & j\neq i \end{array} \right. \tag{2}
$$

### 3.4.2 双比特翻转操作

- 量子电路序列：
  - 1: $H_{i}$  {第i比特Hadamard门}
  - 2: $H_{j}$  {第j比特Hadamard门}
  - 3:  $CNOT_{j\rightarrow i}$  {控制j目标i的CNOT}
  - 4:  $Z_{j}$  {第j比特相位翻转}
  - 5:  $CNOT_{j\rightarrow i}$  {再次CNOT}
  - 6:  $H_{i}$  {恢复Hadamard门}
  - 7:  $H_{j}$  {恢复Hadamard门}

- 等效西矩阵：

$$
U_{ij} = (H_{i}\otimes H_{j})\cdot CNOT_{ji}\cdot (I_{i}\otimes Z_{j})\cdot CNOT_{ji}\cdot (H_{i}\otimes H_{j})
$$

- 状态更新规则：

$$
\begin{array}{l}{s_{i}\leftarrow -s_{i}}\\ {s_{j}\leftarrow -s_{j}}\\ {\delta_{k}\leftarrow -2s_{k}(h_{k} + \sum_{l = 1}^{n}J_{kl}s_{l})\quad \forall k\in \{1,\dots,n\}} \end{array}
$$

### 3.4.3 终止条件判定

- 能量监测窗口：
    $$ W = \lfloor 100/\log(T+1) \rfloor \quad \text{（动态窗口大小）} $$

- 停止准则：
    $$\text{终止条件} = \begin{cases}
    \text{True} & \text{当} \min_{t\in [k-W,k]}\delta^{(t)} \geq 0 \ \text{且} \ T < 10^{-3} \\
    \text{False} & \text{否则}
    \end{cases}
    $$


- 状态回滚机制：当触发终止时，回退到最近  $W$  步中能量最低的电路配置：

$$
\mathcal{C}_{\mathrm{final}} = \mathrm{argmin}_{\mathcal{C}(k - W),\ldots ,\mathcal{C}(k)}\langle \psi (\mathcal{C})|H|\psi (\mathcal{C})\rangle
$$

# 四、总结及展望

## 总结

本研究提出了一种基于Clifford线路与模拟退火的量子- 经典混合算法，用于高效求解Ising模型基态问题。其核心创新在于：
#### 1. 理论框架创新：
  结合Clifford线路的经典可模拟特性（多项式时间复杂度）与模拟退火的全局优化能力，构建了双层级状态转移机制（单/双比特翻转），突破了传统量子变分算法在NISQ时代的优化瓶颈。

#### 2.算法设计突破：
- 引入四种量子态初始化策略（无翻转、全翻转、随机、h导向态），提升初始解质量
- 设计动态温度调控（  $T_{k + 1} = 0.993T_k$  ）与自适应马尔可夫链长度，通过热平衡判定（接受率阈值）和能量监测窗口（  $W = \lfloor 100 / \log (T + 1)\rfloor)$  ）保障收敛性
- 推导增量梯度更新公式  $(\delta_{j}\leftarrow \delta_{j} + 4J_{ji}s_{j}s_{i})$，显著降低计算复杂度

## 未来展望

基于本研究的理论成果，未来工作可从以下方向深化：

#### 1.混合优化机制升级：

- 将模拟退火与量子启发的经典算法（如量子退火、并行回火）结合，设计多温度协同搜索策略，避免局部最优解
- 引入自适应扰动机制，根据能量梯度动态调整单/双比特翻转比例，提升优化效率

#### 2.问题泛化与理论分析：

- 扩展至非均匀Ising模型（如随机场、稀释耦合），分析Clifford线路在无序系统中的性能边界
- 建立算法收敛性严格证明，结合统计物理方法（如自由能分析）量化混合算法的近似比保证

#### 3.跨领域算法融合：

- 融合经典启发式方法（如禁忌搜索、遗传算法）与 Clifford 状态更新机制，构建分层优化框架
- 探索 Clifford 线路在量子-经典神经网络中的应用，实现端到端组合优化求解

# 参考文献

[1] PRESKILL J. Quantum computing in the nisq era and beyond[J]. Quantum, 2018, 2: 79. \
[2] CEREZO M, ARRASMITH A, BABBUSH R, et al. Variational quantum algorithms[J]. Nature Reviews Physics, 2021, 3(9): 625- 644. \
[3] BHARTI K, CERVERA- LIERTA A, KYAW T H, et al. Noisy intermediate- scale quantum algorithms[J]. Reviews of Modern Physics, 2022, 94(1): 015004. \
[4] FARHI E, GOLDSTONE J, GUTMANN S. A quantum approximate optimization algorithm[A]. 2014. \
[5] KARAMLOU A H, SIMON W A, KATABARWA A, et al. Analyzing the performance of variational quantum factoring on a superconducting quantum processor[J]. npj Quantum Information, 2021, 7(1): 156.