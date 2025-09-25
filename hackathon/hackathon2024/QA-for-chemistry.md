

[TOC]



# 1.引言

第六届·2024 MindSpore 量子计算黑客松大赛吸引了全球众多量子计算爱好者的报名参赛，累计报名人数1000多人。经过为期四个月的激烈角逐，本届赛事圆满收官。为系统梳理赛事成果、促进量子计算知识共享，技术团队从「量子模拟」「量子组合优化」「量子启发式算法」三大核心赛道切入，撰写了深度技术解析报告。本文为「量子模拟」赛道报告简要概览，报告全文已同步开源至MindSpore量子计算社区Gitee研究仓：https://gitee.com/mindspore/mindquantum/tree/research/hackathon/hackathon2024/qsim 相关团队的参赛代码也在此仓库。我们期待这些技术文档与代码能帮助开发者：全景式把握赛事技术脉络，启发新型量子-经典混合计算范式探索。



### 1.1 量子化学
求解分子体系的基态能量在量子化学中具有极其重要的意义。基态能量是分子在最低能量状态下的能量值，它直接反映了分子的稳定性和结构。具体的，基态能量的计算可以帮助化学家理解分子在不同键长和键角下的能量变化，从而确定分子的最佳几何构型。准确的基态能量计算对于新材料和药物的设计具有重要意义。通过计算分子的基态能量，可以预测材料的电子结构和光学性质，从而设计出具有特定性能的新材料。

目前，求解分子体系基态能量的经典方法主要包含Hartree-Fock (HF)、耦合簇理论（Coupled-Cluster, CC）、全配置相互作用方法（Full Configuration Interaction, FCI）等，通过近似处理电子相关效应来降低计算复杂度。然而，传统的方法仍然面临诸多挑战。Hartree-Fock通过自洽场方法求解单电子波函数，得到近似的基态能量。单电子近似使得计算速度较快，但由于忽略了电子相关作用，计算结果不够精确；耦合簇理论通过指数化算符来考虑电子的激发态，能够较精确地描述电子相关作用，其代价是计算复杂度的增加。全配置相互作用方法，虑所有可能的电子激发态，能够提供最精确的基态能量。但其计算量随电子数呈指数增长。



### 1.2 量子算法的引入

量子算法利用量子力学的特性，如量子叠加和量子纠缠，有望显著提升模拟复杂量子系统的计算效率。早期量子算法如量子相位估计算法（QPE）能够精确计算本征值，但对量子比特的高保真度和深度量子电路要求高，不适合当前噪声中等规模量子设备（NISQ）。2014 Peruzzo等人提出提出了一种替代量子相位估计算法的方法——变分量子本征求解器（Variational Quantum Eigensolver，VQE）**[1]**。VQE将量子处理器（QPU）与经典处理器（CPU）相结合，通过变分原理计算哈密顿量的本征值和本征矢。值得一提的是，该论文中实验验证了VQE算法的可行性就是通过通过计算$\mathrm{He}-\mathrm{H}^{+}$分子的基态能量。



# 2 相关技术概览&赛题分析

### 赛题说明

> 给定一个 $H_4$ 分子的结构数据（判题脚本中会采用新的键长），请利用变分量子算法，按照实际量子计算机的行为来得到该分子的基态能量。量子计算机的行为约束如下：
>
> 1. 只能采用规定的门集合：$\{X, CNOT, Y, Z, H, CZ, RX, RY, RZ, 测量, 栅栏\}$。
> 2. 期望值计算只能通过采样比特串来得到，你需要根据所需计算的哈密顿量，来合理的设计量子线路，并通过 `sampling` 得到的采样数据计算出不同 pauli 串哈密顿量的期望值。
> 3. 暂不约束芯片拓扑结构，默认为全联通，任意两比特之间可以相互作用。
> 4. 赛题会给定一个噪声模型，在比赛专用量子模拟器中，会自动在模拟器的 `apply_gate`、`apply_circuit`、`sampling` 等接口处自动添加噪声模型。噪声模型定义具体如下：单比特门会伴有一个极化率为 $0.001$ 的单比特去极化信道噪声和一个$t_1=100us, t_2=50us,t_{gate}=30ns$ 的热弛豫噪声，双比特门会伴有一个极化率为 $0.004$ 的双比特去极化信道噪声和一个 $t_1=100us, t_2=50us,t_{gate}=80ns$ 的热弛豫噪声，测量门会伴有一个翻转概率为 $0.05$ 的比特翻转信道噪声。

判题与评分标准

> 代码总得分为：$\frac{1}{|E−E_{fci} |}$，其中𝐸为选手得到的分子基态能量，𝐸_fci 为分子的FCI能量。



### 解题思路

目标：

> 在噪声干扰下，设计变分量子算法获得最接近 $H_4$ 分子基态能量的值。

解题思路

> 为了获得更接近 $H_4$ 分子基态能量的值，主要从三个方面进行设计：
>
> 1. **含参数线路的设计**：
>    - 在量子计算中，经常用“Ansatz”术语来描述参数化的量子电路。Ansatz 可以生成不同的量子态。
>    - 不同的Ansatz有不同的表达能力，因此Ansatz的设计是获取基态能量的关键步骤。
> 2. **误差缓解：**
>    - 该题通过噪声模型加入噪声，这也更接近实际中的量子线路遇到的情况。量子噪声是量子计算中不可避免的问题，而量子误差缓解是一系列技术，用于减少量子噪声对计算结果的影响。
>    - 量子误差缓解可以提升变分量子算法性能，因此选手可以选择合适的量子误差缓解方法来提高自己的得分。
> 3. **测量优化**
>    - 使用量子误差缓解后，会减小结果的偏差，但是会增大结果的方差，故需要增加测量的次数。
>    - 而增加采样的次数可能会导致超时，故需要对测量的方法进行优化。





# 3.比赛方案分析

在本章节，从解题思路的三个方面，结合比赛团队的方案进行分析

1. Ansatz设计 
2. 误差缓解
3. 测量优化+预处理

现在先给出参赛小组与方案号的对应，以及他们方案的概述：

| 参赛小组            | 设计使用的Ansatz                                             | 错误缓解方法                                                 | 测量优化方法                                    |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------- |
| 方案一：Hsm9789小组 | Qubit-ADAPT-VQE方法，HEA方法                                 | ZNE方法，噪声放大方法（FIIM,RIIM),REM方法（读取错误缓解）    | 剔除小量，分组测量(抽象为图论问题使用贪心算法） |
| 方案二：图南队      | Qubit-ADAPT-VQE方法,双电子激发拟设                           | ZNE方法，PEC技术，PualiTwirling(调整噪声），ZNE与PT组合，PEC与PT组合，REM方法 | 单重态子空间，分组测量                          |
| 方案三：高荣浩      | Givens Ansatz                                                | ZNE方法                                                      |                                                 |
| 方案四：量子动力    | UCCSD＋参数初始化                                            | ZNE方法                                                      | 根据泡利串构造联合测量                          |
| 方案五：name：str   | 费米子激发算符的直接构造方法，双激发算符的极浅构造法，活跃空间近似，分子点群对称性 | ZNE方法                                                      | 分组测量                                        |
| 方案六：QCB11112    | 通过构建Full CI矩阵的方法拆分激发态电路，量子线路分割，      | ZNE方法，PEC方法，REM方法，（未区分门噪声和测量噪声，最后统一使用读取错误缓解作为最好结果方法，没有同时使用ZNE和读取错误缓解） | 分组测量                                        |
| 方案七：Quiscus     | 基于HEA的X-Y合并近似&替换近似                                | ZNE方法，RSEM方法，REM方法                                   | 无                                              |



## 3.1 Ansatz设计 

报告全文介绍了4种参赛队伍使用的Ansatz以及改进设计

> 1. Qubit-ADAPT-VQE：通过自适应地从算符池中选择能量梯度下降最快的量子门，并且通过省略Z项缩减Pauli串长度。
> 2. HEA及其改进：与分子结构无关，硬件实现难度更低。HEA可以与线路剪枝结合。
> 3. Givens Ansatz：通过保持粒子数守恒的单双激发操作，构建浅层量子线路。
> 4. 费米子激发算符直接构造法：更低成本的激发算符构造方法，减少线路深度和量子门数量。

文本限于篇幅，介绍最具有代表性的Qubit-ADAPT-VQE

### Qubit-ADAPT-VQE

参赛团队中排名最高的两个团队（Hsm9789和图南队）都使用到`Qubit-ADAPT-VQE`，因此我们首先对`Qubit-ADAPT-VQE`进行介绍。



**Qubit-ADAPT-VQE**（量子比特自适应变分量子本征求解器）是 **ADAPT-VQE** 的改进版本，旨在解决传统UCCSD方法中线路过深、噪声敏感的问题**[12,13]**。**ADAPT-VQE** 的核心思想是：通过自适应地从算符池中选择能量梯度下降最快的量子门，将门添加到量子线路之中。这和传统方法中预先定义一个固定线路结构的方法不同，可以把对能量影响不大的量子门省略掉。



#### ADAPT-VQE流程

根据下图对**ADAPT-VQE** 的详细流程进行介绍：

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250206104318819.png" style="zoom:50%;" />

> **ADAPT-VQE**流程示意图。引用自参考文献[12]

1. **初始化**：初始状态，即从$\left|\psi^{(0)}\right\rangle=\left|\psi^{\mathrm{HF}}\right\rangle$（Hartree–Fock）波函数开始。
2. 构建**操作池（Operator Pool）** $\hat{A}_m$：含单双激发项
3. **扩展 ansatz**：每次迭代，从操作池中选择对能量梯度贡献最大的算符添加到线路中。

$$
\frac{\partial E^{(n)}}{\partial \theta_i}=\left\langle\psi^{(n)}\right|\left[H, A_i\right]\left|\psi^{(n)}\right\rangle,
$$

4. 重新优化所有参数 $\vec{\theta}^{(n+1)}$
5. 判断是否收敛



#### Qubit-ADAPT-VQE的改进

**Qubit-ADAPT-VQE**在 **ADAPT-VQE** 的基础上进行了改进：传统ADAPT－VQE的Pauli串包含X，Y，Z项，而Qubit－ADAPT-VQE通过省略Z项，使每个Pauli串长度不超过4项，每个算符对应的CNOT门数减少至6个以下。经数值模拟验证，省略Z 算符前后，得到的期望值差别很小，因此可以省略。

#### 效果对比

- CNOT门减少 $80 \%$ ：相比UCCSD，Qubit－ADAPT－VQE在H4分子中仅需 28 个CNOT门 （UCCSD需140＋）。
- 参数数量减少 $40 \%$ ：通过动态选择和剪枝，参数从 32 个减少至约 10 个。



## 3.2 错误缓解方法

当前量子计算硬件的发展阶段，噪声是制约计算精度和可靠性的核心挑战之一。受限于量子比特的退相干时间、门操作误差及测量不完美性，真实的量子设备难以实现理想的幺正演化。为贴近实际应用场景并在算法设计中充分考虑噪声影响，本次赛题特别在量子线路中引入模拟噪声。

为了克服噪声带来的影响，参赛团队通过错误缓解（Error Mitigation）技术提升计算结果的可信度。量子错误缓解（Quantum Error Mitigation, QEM）技术是面向噪声中等规模量子（NISQ）设备的关键方法，旨在通过后处理噪声电路的输出来降低噪声引起的系统偏差，而非完全消除噪声。其核心目标是通过有限的资源提升量子计算的可靠性，为短期内实现量子优势提供支持。在众多QEM方法中，**零噪声外推（Zero-noise Extrapolation, ZNE）**因其简洁性和广泛适用性成为最受关注的方案之一。经对参赛方案的统计分析，所有7个团队均采用了**零噪声外推（Zero-Noise Extrapolation, ZNE）**方法。此外，有4支团队进一步结合了**读取错误缓解（Readout Error Mitigation, REM）**技术。



### 零噪声外推 ZNE

#### 基本原理

零噪声外推的核心思想是通过主动调控噪声强度，在不同噪声水平下测量目标可观测量的期望值，并基于这些数据点外推至零噪声极限下的理想值。假设噪声电路的输出期望值 $\operatorname{Tr}\left[O \rho_\lambda\right]$ 是噪声强度 $\lambda$ 的函数，ZNE通过实验获取多个噪声放大后的数据点 $\left\{\left(\lambda_m, \operatorname{Tr}\left[O \rho_{\lambda_m}\right]\right)\right\}$ ，利用数学模型拟合该函数，最终外推至 $\lambda=0$ 时的无噪期望值 $\operatorname{Tr}\left[O \rho_0\right]$ 。

#### 噪声放大方法

为生成不同噪声水平下的数据点，需在硬件上实现噪声强度的可控放大，常用方法是门插入（Gate Insertion）：在电路中插入等效于恒等操作的冗余门序列（如 $U U^{\dagger}$ ），增加噪声累积次数。此方法适用于门错误主导的场景，但对非退极化噪声可能改变噪声模型。



#### 外推算法

1．**多项式外推（Richardson Extrapolation）**
假设 $\operatorname{Tr}\left[O \rho_\lambda\right]$ 可展开为 $\lambda$ 的泰勒级数，截断至 $M-1$ 阶多项式：
$$
f(\lambda ; \vec{\theta})=\sum_{\ell=0}^{M-1} \theta_{\ell} \frac{\lambda^{\ell}}{\ell!}
$$


通过最小二乘法拟合 $M$ 个数据点，零噪声估计值为 $\theta_0^*$ 。

2．**指数外推**
针对深电路 $(\lambda \gg 1)$ ，期望值可能随噪声呈指数衰减（如 $\operatorname{Tr}\left[O \rho_\lambda\right] \propto e^{-\beta \lambda}$​ ）。此时采用指数模型：
$$
f(\lambda ; \vec{\theta})=\theta_0 e^{-\theta_1 \lambda}
$$


通过拟合提取零噪声极限 $\theta_0$ 。此方法在高噪声下更为鲁棒，尤其适用于全局退极化等噪声模型。

3. **其他拟合函数**

参赛团队`高荣浩`对比了对数函数和多项式函数在比赛问题中的效果。其中对数函数的形式为
$$
f(\lambda ; a, b, c)=a \cdot \log (\lambda+b)+c,
$$
下图为测量次数为256 时，使用不同拟合函数的ZNE 外推结果

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250207145307338.png" style="zoom: 33%;" />

> 图片引用自参赛团队`高荣浩`





## 3.3 测量优化+预处理

测量优化旨在通过优化量子测量的次数（shots）分配策略，以平衡计算成本与算法收敛性。

- 参赛队伍中使用最多的测量优化方法是分组测量，有4队。
- 往往不是单纯使用分组测量，而是结合了其他方法，包含剔除小量，活性空间近似，泡利串构造联合测量。

预处理

- 预训练：基于参数迁移的优化范式，通过参数空间的连续性先验知识，规避了传统变分量子本征求解器（VQE）中随机初始化带来的收敛迟滞问题。
- 哈密顿量X-Y近似与小系数项的截断，Pauli 串的总数从初始的 184 个减少至 66 个，这一结果显著降低了哈密顿量的复杂度。



### 分组测量

1．问题背景

在量子变分算法（如VQE）中，系统的哈密顿量可以分解为多个Pauli项的线性组合（如 $\hat{H}=$ $\left.\sum \alpha_i P_i\right)$ 。直接逐项测量每个Pauli项需要大量的测量次数，尤其在大型分子中（如 $\mathrm{H}_4$的哈密顿量包含 184 个Pauli项），效率极低。

2. 分组测量的原理

若两个Pauli项 $P_i$ 和 $P_j$ 的对应量子比特算符对易（即 $\left[P_i, P_j\right]=0$ ），则它们可通过单次测量同时获取期望值。










# 4.数值结果

代码总得分为：$\frac{1}{|E−E_{fci} |}$，其中𝐸为选手得到的分子基态能量，𝐸_fci 为分子的FCI能量。这也是我们实验的评判依据。

​	在本次比赛中，各参赛团队提出了多种方法以改进变分量子算法在噪声环境求解分子基态能量的精度。

>  方案号与队伍名称的对应，以及他们方案的概述：
>
> 1. 方案一（Hsm9789小组）在设计量子线路中分别采用了 Qubit-ADAPT-VQE 和 Hardware-efficient ansatz 两种方法，其中在Qubit-ADAPT-VQE方法里省略了单激发项与双激发项在$Jordan-Wigner$变换后得到的$Pauli $ 算符中的 $Z$ 算符，在保证精度的同时加快了测量速度，在错误缓解中主要使用$ZNE$方法和读取错误纠正$REM$，在最后的测量过程中采用分组测量减少测量次数；
> 2. 方案二（图南队）分析出 $H4$ 分子中影响较大的为双电子激发，在 $UCCSD$ 拟设中只考虑双电子激发，大大减少了门的数量；
> 3. 方案三（高荣浩）线路设计采用 Givens  ansatz；
> 4. 方案四（量子动力）采用 $ZNE$ 方法与采样测量估计误差；
> 5. 方案五（mame：str）提出线路最左端激发算符的极简构造方法和激发算符排序方法。
> 6. 方案六（Quiscus）采用基预训练微调的参数优化以及读出误差缓解($REM$)。
> 7. 方案七（1+1）的方法与方案一大部分相同。

​	我们对各参赛团队的解决方案进行了实验测试，由于收集到的代码并非他们决赛答辩时的代码，所以无法复现出他们再决赛展示出的结果。得到的实验结果如下：

![image-20250224163946377](https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250224163946377.png)

结果说明：

1. 统一设置shots=5000
2. Box plot是运行10次结果的统计。
3. 就像一些选手的报告中说的，因为是变分算法，最优的运行结果不一定会再次运行得到。我们是根据gitee上提交的代码运行对比，因此和选手自己运行的结果存在差异。 









# 5. 结论与展望

​	本次Hackathon量子模拟算法赛道针对的是如何在噪声环境下，在合理的计算时间内利用 VQE 算法尽可能精确求解分子基态能量。各参赛团队提出的方法主要围绕量子线路设计（uccsd拟设及改进、Qubit-ADAPT-VQE、Hardware-efficient ansatz）、误差缓解方法（ZNE、Clifford 数据回归、机器学习）、测量优化与预处理（剔除小量、分组测量、shots分配优化、预训练-微调）。从这些方案的解读中可以获得以下感悟与结论：

> 1. 比赛结果显示，集成多维度技术方案的团队相较单一技术路径可获得更高得分。需注意不同技术的兼容性：
>    - 可兼容的串行技术（如误差缓解+测量优化+预处理）需多多使用。
>    - 并行技术（如不同测量分组策略）应建立量化评估机制择优选用。
> 2. 使用了Qubit-ADAPT-VQE的两个方案（方案一、二）从最后决赛结果来看结果最好。说明该方法的有效性，但是该方法训练成本较高。
> 3. 评判一个线路的好坏首要的还是看线路能否在无噪声的情况下得到接近基态能量的值，毕竟若这个结果太差的话，后面基于线路添加噪声后的通过**ZNE**方法得到的解就算很接近基态能量的值也没有说服力，因为**ZNE**方法的作用就是处理噪声，理论上最好的结果就是无噪声下的期望值。
> 4. 在保持求出解的精确性的同时，还要控制量子门的个数与线路深度。当量子门数量过多时，噪声的影响会增大，尤其是通过**ZNE**方法中常用到折叠线路扩大噪声的算法，进一步使门的数量增多，使我们通过测量方式求出的期望的方差增大。若要使结果稳定，必须要设置很大的shots，增加时间花费。
> 5. 有些方法有创新性，但是分数并不高，创新的技术往往需要时间去迭代提升性能。不过组织方通过复赛的形式，让评判标准不仅仅是看标准的计分方法。
> 6. 防止作弊，当赛题有准确解的时候。应防止与禁止参赛选手，利用准确解，因为在现实科研中，大部分情况下准确解是未知的。防止作弊的一些注意点，结合代码分析在附录中展示。
> 7. 未来，如果我们确实要在真实的量子计算机上进行量子化学模拟实验，噪声模型只会变得更加复杂且难以控制。错误缓解也要求量子错误率低于特定值。这些挑战当前是阻碍量子化学模拟进一步发展的因素。因此，提高量子门的保真度和发展量子纠错是克服这些问题的重要手段。
>







# 参考文献

> - 赛题链接https://competition.huaweicloud.com/information/1000042022/noise
> - 寇享-决赛视频https://m.koushare.com/live/details/36781
> - 比赛代码gitee仓库 https://gitee.com/mindspore/mindquantum/tree/research/hackathon/hackathon2024/qsim

[1] Peruzzo, A., McClean, J., Shadbolt, P., Yung, M. H., Zhou, X. Q., Love, P. J., ... & O’brien, J. L. (2014). A variational eigenvalue solver on a photonic quantum processor. *Nature communications*, *5*(1), 4213.

[2] Kandala, A., Mezzacapo, A., Temme, K., Takita, M., Brink, M., Chow, J. M., & Gambetta, J. M. (2017). Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. *nature*, *549*(7671), 242-246.

[3] Wang, X., Qi, B., Wang, Y., & Dong, D. (2024). Entanglement-variational hardware-efficient ansatz for eigensolvers. *Physical Review Applied*, *21*(3), 034059.

[4] Dave Wecker, Matthew B. Hastings, and Matthias Troye Towards Practical Quantum Variational Algorithms. Phys. Rev. A 92, 042303 (2015).

[5] Roeland Wiersema, Cunlu Zhou, Yvette de Sereville, Juan Felipe Carrasquilla, Yong Baek Kim, Henry Yuen Exploring entanglement and optimization within the Hamiltonian Variational Ansatz . PRX Quantum 1, 020319 (2020).

[6] Sim, S., Johnson, P. D., & Aspuru‐Guzik, A. (2019). Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum‐classical algorithms. *Advanced Quantum Technologies*, *2*(12), 1900070.

[7] McClean, J. R., Boixo, S., Smelyanskiy, V. N., Babbush, R., & Neven, H. (2018). Barren plateaus in quantum neural network training landscapes. *Nature communications*, *9*(1), 4812.

[8] Wang, S., Fontana, E., Cerezo, M., Sharma, K., Sone, A., Cincio, L., & Coles, P. J. (2021). Noise-induced barren plateaus in variational quantum algorithms. *Nature communications*, *12*(1), 6961.

[9] Cai, Z., Babbush, R., Benjamin, S. C., Endo, S., Huggins, W. J., Li, Y., ... & O’Brien, T. E. (2023). Quantum error mitigation. *Reviews of Modern Physics*, *95*(4), 045005.

[10] Giurgica-Tiron, T., Hindy, Y., LaRose, R., Mari, A., & Zeng, W. J. (2020, October). Digital zero noise extrapolation for quantum error mitigation. In *2020 IEEE International Conference on Quantum Computing and Engineering (QCE)* (pp. 306-316). IEEE.

[11] Zhu, L., Liang, S., Yang, C., & Li, X. (2024). Optimizing shot assignment in variational quantum eigensolver measurement. *Journal of Chemical Theory and Computation*, *20*(6), 2390-2403.

[12] GRIMSLEY H R, ECONOMOU S E, BARNES E, et al. An adaptive variational algorithm for exact molecular simulations on a quantum computer[J]. Nature communications, 2019, 10(1):3007.

[13] TANG H L, SHKOLNIKOV V, BARRON G S, et al. qubit-adapt-vqe: An adaptive algorithm for constructing hardware-efficient ansätze on a quantum processor[J]. PRX Quantum, 2021, 2(2):020310.

[14] Arrazola, J. M., Di Matteo, O., Quesada, N., Jahangiri, S., Delgado, A., & Killoran, N. (2022). Universal quantum circuits for quantum chemistry. *Quantum*, *6*, 742.

















