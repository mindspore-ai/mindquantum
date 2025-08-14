# 摘要 {#摘要 .unnumbered}

最大割问题是一个在组合优化领域著名的NP-hard问题，有关其多项式时间的近似优化算法已经达到了一定的数量。如果我们考虑量子算法，可以将其写成QUBO问题的形式，并转换为求解伊辛模型基态与基态能量的量子计算问题。本文中，我们将先回顾过往求解伊辛模型基态与基态能量的算法，并以其中迄今为止最高效的算法之一------Adapt-Clifford为蓝本，提出后选择Adapt-Clifford法，介绍它的理论与算法框架，而后通过数值实验比较它与Adapt-Clifford的数值结果并提供较优的一组后选择Adapt-Clifford算法参数。最后，我们对后选择Adapt-Clifford的优势与劣势进行分析，并对其优化能力给出总结。

# 问题背景与描述

## 问题背景：最大割问题

### 最大割问题定义与应用

给定一个图$\mathcal{G}=(\mathcal{V},\mathcal{E})$，其中$\mathcal{V}$是顶点集，$\mathcal{E}$是边的集合，且边权重$\omega_{i,j}\in\mathbb{R}$定义在$(i,j)\in\mathcal{E}$上。最大割（Maxcut）问题要求将$\mathcal{V}$划分为两个互补子集$\mathcal{A},\overline{\mathcal{A}}\subseteq\mathcal{V}$，使得$\mathcal{A}$和$\overline{\mathcal{A}}$之间所有边的总权重最大化。

如果我们要直接将其转变为数学表达式，我们可以使用二元变量$z_i\in\{0,1\}$（$i\in\mathcal{V}$）来标识每个顶点所属的子集：若顶点$i\in\mathcal{A}$，则$z_i = 1$；若$i\in\overline{\mathcal{A}}$，则$z_i = 0$。而后，我们的目标变为寻找使损失函数最大化的赋值$\mathbf{z}$：
$$C(\mathbf{z}) = \sum_{(i,j)\in\mathcal{E}} \omega_{i,j} z_{i} (1 - z_{j})$$
其中$\mathbf{z}=z_{1}\ldots z_{N}$是一个$N$位二进制字符串，
其中$N$是图的顶点数$|\mathcal{V}|$，且满足对称性条件$\omega_{i,j}=\omega_{j,i}$（$\forall(i,j)\in\mathcal{E}$）。

在网络设计中，最大割问题的权重常常代表特定基础设施的建造成本。由于现实环境中多重因素的影响，这些参数往往并非确定值，因此我们对最大割问题的权重没有一个条件较好的假设。除此以外，最大割问题与TSP（旅行商）问题，VLSI（超大规模集成电路）构建等重要组合优化问题之间也有紧密关联。因此，为一般的最大割问题提供一个尽量好的解决算法十分重要。

最大割问题是最经典的NP-Hard问题，其解空间随节点数$n$呈指数增长，因此，从现实角度考虑，我们一般不考虑其准确解法，而是考虑多项式时间内的逼近解，比如Goemans-Williamson
(GW) 近似算法，就可以对准确解达到约0.878的近似比。

### 最大割问题的量子计算变体

在介绍最大割问题的量子计算变体之前，我们先介绍哈密顿量。哈密顿量是描述物理系统状态的重要物理量，其在多种物理系统中都有广泛运用。一个最经典的例子是在量子力学中，薛定谔方程是系统演化的核心控制方程：
$$i\hbar \frac{\partial}{\partial t} \ket{\psi(t)} = H \ket{\psi(t)}$$

这其中就涉及到了哈密顿量的使用。而哈密顿量一个重要的用法是通过哈密顿量求解本征方程，形如
$$H \ket{\psi_n} = E_n \ket{\psi_n}$$

而若我们能找到能量最低的本征态：${H}|\psi_g\rangle = E_g|\psi_g\rangle$，其中$E_g$是所有$E_n$中的最小值，则我们称其为基态（ground
state）。基态满足变分原理：
$$E_g = \min_{|\psi\rangle}\frac{\langle\psi|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle}$$

而我们注意到，若将组合优化问题的损失函数编码为哈密顿量${H}_{\text{prob}}$，则两者存在如下对应关系：
$$\begin{array}{c|c}
\text{优化问题} & \text{量子系统} \\ 
\hline
\text{最小化代价函数 } C(\mathbf{z}) & \text{寻找最低能量本征态} \\
\text{解变量 } \mathbf{z} & \text{基态 } |\psi_g\rangle \\
\text{最优值 } \min C & \text{基态能量 } E_g \\
\end{array}$$

因此，我们可以组合优化问题转化为寻找对应哈密顿量基态与基态能量的问题。特别地，对于最大割问题这种二次无约束二元优化（QUBO）问题，即形式为
$$\min_{x \in \{0,1\}^n}\sum_{i=1}^n c_i x_i + \sum_{i=1}^n \sum_{j=1}^n q_{ij} x_i x_j$$
的问题，转化为寻找伊辛（Ising）模型基态问题
$${H}_\mathrm{Ising} = -\sum_{i} h_i \sigma_i^z - \sum_{i<j} J_{ij} \sigma_i^z\sigma_j^z$$
其中$\sigma_i^z$为泡利Z算符，$h_i,J_{ij}$编码优化问题参数。而后，我们只需要考虑对于${H}_\mathrm{Ising}$，如何尽量求取其基态与最低能量即可。

## 问题重述：基于Clifford门的量子组合优化算法

我们的目标是对于一个包含$N$个自旋变量[^1]的伊辛模型
$$H_c = \sum_i h_i \sigma_z^i + \sum_{i<j \in E} J_{ij}\sigma_z^i \sigma_z^j.$$
搭建一个对于较大的$N$能够在经典计算机上高效被模拟的Clifford线路，即由$H, S, X, Y, Z, CNOT,\\ CY, CZ, SWAP$等Clifford量子逻辑门构成的线路，来制备$H_c$的基态，并在合适的时间内得到一个尽量低的最终能量。

在这里，我们考虑Clifford电路是因为它可被经典计算机在​​多项式时间​​内高效模拟，因此，我们可以直接构建低深度Clifford电路，通过迭代添加Clifford门生成稳定子态，编码伊辛模型的近似解。

# 问题分析

## QUBO求解综述

### QUBO经典解法

QUBO的算法十分丰富，能够准确求解QUBO的经典算法有分支定界（Branch and
Bound）算法等。但基于QUBO已经被证明是NP-hard问题的事实，为了追求高效，我们也会使用近似求解的经典算法，如半正定松弛（SDP
Relaxation）算法。

而因为我们可以将QUBO问题转化为伊辛哈密顿量基态求解的问题，我们也有相应的量子启发式算法。

### QUBO优化函数转换为伊辛哈密顿量

一般的QUBO形式为
$$\min_{\mathbf{x}\in\{0,1\}^{n}}\left(\mathbf{x}^{T}\mathbf{Q}\mathbf{x}+\mathbf{c}^{T}\mathbf{x}\right)=\min_{x \in \{0,1\}^n}\sum_{i=1}^n c_i x_i + \sum_{i=1}^n \sum_{j=1}^n q_{ij} x_i x_j$$
其中$\mathbf{Q}$为对称系数矩阵。引入自旋变量$\sigma_{i}^{z}=2x_{i}-1\in\{-1,+1\}$后，我们可以将QUBO问题的优化函数$\sum_{i=1}^n c_i x_i + \sum_{i=1}^n \sum_{j=1}^n q_{ij} x_i x_j$，$\mathbf{x}\in\{0,1\}^{n}$转化为伊辛模型哈密顿量：
$$\begin{aligned}
        &H_\mathrm{Ising} = -\sum_i h_i \sigma_i^z - \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z \\
        \text{其中} \quad 
        &h_i = \frac{1}{2}c_i + \frac{1}{4}\sum_{j \neq i} (q_{ij} + q_{ji}), \quad 
        J_{ij} = \frac{1}{4}q_{ij} \\
        &\sigma_i^z = 2x_i - 1 \in \{-1, +1\}\end{aligned}$$

## 伊辛哈密顿量基态与基态能量求解综述

### 量子退火（Quantum Annealing）

量子退火是求解伊辛模型哈密顿量基态与基态能量的最重要的量子启发式算法之一。假设我们有了伊辛哈密顿量
$$\begin{aligned}
        H_\mathrm{Ising} = -\sum_i h_i \sigma_i^z - \sum_{i<j} J_{ij} \sigma_i^z \sigma_j^z \end{aligned}$$

我们由此构造时间演化哈密顿量 $$H(t)=A(t)H_0+B(t)H_\mathrm{Ising}$$ 其中

-   $H_0=-\sum\sigma_{x}^{(i)}$可以被理解为初始哈密顿量，其基态为均匀叠加态[^2]：
    $$|\psi(0)\rangle=|+\rangle^{\otimes n}=\frac{1}{\sqrt{2^{n}}}\sum_{z\in\{0,1\}^{n}}|z\rangle$$

-   $H_\mathrm{Ising}$是QUBO对应的伊辛模型哈密顿量

-   演化时间为$[0,T]$

-   时间演化调度函数
    $$A(t)=\left(1-\frac{t}{T}\right),\ B(t)=\frac{t}{T},\quad t\in[0, T]$$

当$t=0$，我们有$H(t)=H_0$。假设此时我们的演化初态为$|\psi(0)\rangle=|+\rangle^{\otimes n}$，即$H(t)=H_0$的基态，那么对于由$H(t)$表示，初态为$|\psi(0)\rangle$的量子系统，其在演化时间$[0,T]$内服从含时薛定谔方程
$$i\hbar \frac{\partial}{\partial t} \ket{\psi(t)} = H(t) \ket{\psi(t)}$$

而由绝热演化定理，如果一个量子系统的量子态初始处于本征态，那么随着哈密顿量缓慢随时间变化，那么量子态将持续处于瞬时本征态。数学上，当
$${
T \gg \frac{\max\limits_{t\in[0,T]} \lVert \partial H/\partial t \rVert}{\Delta_{\min}^2}
}$$
其中$\Delta_{\min} = \min\limits_{t\in[0,T]} \left[ E_1(t) - E_0(t) \right]$是演化过程中的最小能隙，$\lVert \partial H/\partial t \rVert$是哈密顿量变化速率的算子范数上界，我们即可运用时间演化定理。

因此，只要将$T$设计得符合绝热演化定理的要求，理论上我们便可以对量子退火的流程进行高度精密控制的物理实现。

### 量子退火的优势与局限性

我们可以很直观地看出，QUBO问题与解哈密顿量基态与基态能量的问题之间在数学形式上可以互相转换。但是与对QUBO使用经典模拟退火相比，量子退火是对微观粒子进行演化，而经典退火处理的是大量原子的集体行为而且温度很高，量子效应在大量级下被热运动"平均化"，在高温下被摧毁，表现出经典统计行为。

这就让量子退火比起经典退火有了"量子优势"。在退火演化过程中，物理态可以近似看成是在朝着能量更低的状态"发展"。如果这个过程是经典的，那么在"发展"的过程中，可能因为自身能量少于翻过势垒的势能而限于某个局部最优情况。而如果这个过程是量子的，那么即使自身能量少于翻过势垒的势能，量子态也有概率穿过这个势垒，更有益于寻找全局最优的解。这个"量子优势"就是量子隧穿。

但是，量子退火对物理实验平台的要求很高，需要在极低温（如15mK）下运行，以冻结热噪声、保护量子态。而且，其需维持量子相干性，且对于高势垒，演化时间$T$会指数级增长，在目前含噪中尺度量子（NISQ）硬件的情况下较难让高维伊辛模型哈密顿量有好的实验结果。

### QAOA（量子近似优化算法）

量子退火要求对哈密顿量的连续调控，但是对于目前的量子计算机，实现连续调控的成本非常高。因此，我们希望对量子退火离散化，并把它放到量子计算机中高效模拟。如果离散模拟的每次演化都是Clifford的，我们甚至可以试着在经典计算机上对其高效模拟。

注意到我们可以将量子退火的演化写成酉矩阵
$$U=\exp\left(-i\int_{0}^{T} H(t)dt\right)=\exp\left(-i\cdot \frac{T}{2}(H_0+H_\mathrm{Ising})\right)$$

取$\Delta t=T/p$，对足够大的$p$，我们假设在每区间$[t_{k},t_{k+1}]$内将哈密顿量都取为取$H(t_{k})$，其中$t_{k}=k\Delta t/T$，然后我们运用Trotter分解得到，对非对易项$[H_{B},H_{P}]\neq 0$使：
$$\begin{aligned}
    e^{-i H\left(t_{k}\right) \Delta t} & =e^{-i\left[\left(1-t_{k}\right) H_{0}+t_{k} H_\mathrm{Ising}\right] \Delta t} \\
    & \approx e^{-i\left(1-t_{k}\right) H_{0} \Delta t} e^{-i t_{k} H_\mathrm{Ising} \Delta t}+\mathcal{O}\left(\Delta t^{2}\right)
    \end{aligned}$$

由此，对于QAOA，我们将连续时间$t\in[0,T]$分割为$p$个区间，用Trotter分解公式对分层量子门序列近似量子退火：
$$U_{\mathrm{QA}}=\prod_{k=1}^{p} e^{-i\beta_{k} H_{0}}e^{-i\gamma_{k} H_\mathrm{Ising}}$$

其中$\gamma_{k}\sim t_{k} \Delta t,\ \beta_{k}\sim \left(1-t_{k}\right) \Delta t$为可调参数，对应第$k$步的演化时长，其本质即为用有限步的酉变换逼近连续演化（$p\rightarrow\infty$时等价于量子退火）。值得注意的是，由$H_\mathrm{Ising}$的形式，每次演化对应的酉矩阵都是Clifford的。

而后我们制备初始态$|\psi_{0}\rangle=|+\rangle^{\otimes n}$，经$p$层门操作生成变分态：

$$|\psi_{p}(\vec{\gamma},\vec{\beta})\rangle=\prod_{k=1}^{p}e^{-i\beta_{k} H_{0}}e^{-i\gamma_{k} H_\mathrm{Ising}}|\psi(0)\rangle$$

最后再测量期望值$F_{p}=\langle\psi_{p}|H_\mathrm{Ising}|\psi_{p}\rangle$，通过经典优化器调整$\vec{\gamma},\vec{\beta}$以最大化$F_{p}$，便完成了QAOA算法。

相对于量子退火，QAOA
的变分参数$(\vec{\gamma}, \vec{\beta})$不受线性调度约束，可通过经典优化器动态调整演化路径，突破绝热演化限，而量子退火则强制遵循$A(t)=\left(1-\frac{t}{T}\right),\ B(t)=\frac{t}{T},\quad t\in[0, T]$的固定调度函数。在QAOA电路中，已经被证明存在一个经典算法能够有效地计算量子期望值，而且在大多数情况下，该算法的运行时间与量子比特数成线性关系，可以被视为NISQ兼容的，这比起量子退火的效率和资源需求都有了极大的提升。因此，QAOA也是现在求解伊辛哈密顿量基态与基态能量的主流算法之一。

### Adapt-QAOA

在2023年，基于QAOA，Muñoz-Arias提出了改进性的Adapt-QAOA与Adapt-Clifford算法，提升了QAOA的变通性，为求解伊辛哈密顿量基态与基态能量提供了新的可能。QAOA虽然是为满足NISQ兼容性设计的算法，但是就目前的量子硬件条件而言，层数深的QAOA仍然需要较大的算力来达到好的优化效果。因此，我们希望能够从QAOA衍生出经典可模拟的算法。

启发于QAOA的形式，Adapt-QAOA将分层量子门改为了
$$U_{\mathrm{AQA}}=\prod_{k=1}^{p} e^{-i\beta_{k} A_k}e^{-i\gamma_{k} H_\mathrm{Ising}}$$

我们可以看到，唯一的改动是将$H_0$改为了$A_k$，其中
$$A_{k} = \max_{A\in P_{\mathrm{OP}}}\left[ -i \langle \psi_{k-1} | e^{i \gamma_{k} H_\mathrm{Ising}} [H_\mathrm{Ising}, {A}] e^{-i \gamma_{k} H_\mathrm{Ising}} | \psi_{k-1} \rangle \right]$$
是$P_{\mathrm{OP}}$中最大化能量梯度的哈密顿量，这里 $$\begin{aligned}
P_{\mathrm{OP}} = &\left\{\sum_{i}X_{i},\sum_{i}Y_{i}\right\}\cup\left\{X_{j},Y_{j}\right\}_{j=1,\ldots,N} \nonumber \\
&\cup\left\{X_{j}X_{k},Y_{j}Y_{k},Y_{j}Z_{k},Z_{j}Y_{k}\right\}_{j,k=1,\ldots,N,j\neq k}\end{aligned}$$
由此我们可以计算得到终态
$$|\psi_{p}(\vec{\gamma},\vec{\beta})\rangle^{\mathrm{ADAPT}} = 
\left[
    \prod_{k=1}^{p} e^{-i \beta_{k} A_{k}} e^{-i \gamma_{k} H_\mathrm{Ising}}
\right]|\psi_{0}\rangle$$

我们定义近似比
$$\alpha = \frac{\left\langle \phi \left| H_{\mathrm{Ising}} \right| \phi \right\rangle}{E_{\min}^{\mathrm{Ising}}}$$
其中$E_{\min}^{\mathrm{Ising}}$代表$H_{\mathrm{Ising}}$的实际最小能量，这在伊辛哈密顿量维度很小的时候很好计算。$\alpha$越接近1，就代表对基态和基态能量的逼近越成功。而根据数值实验的结果，对于服从不同分布的最大割问题边权重$\omega_{ij}$对应的伊辛哈密顿量基态于基态能量求解问题，可以得到数值结果如图1。

![Adapt-QAOA原论文数值实验结果](1.jpg){width="60%"}

对于较多的层数$p$，Adapt-QAOA的表现显著好于QAOA。由此可见，Adapt-QAOA拥有更好的潜力。

### Adapt-Clifford

为了实现经典可模拟的算法，我们希望
$$U_{\mathrm{AQA}}=\prod_{k=1}^{p} e^{-i\beta_{k} A_k}e^{-i\gamma_{k} H_\mathrm{Ising}}$$
是Clifford的，这就要求$(\vec{\gamma}, \vec{\beta}) \in \{k\pi/4|k\in\mathbb{Z}\}^p$，因为我们不能保证所有的$A_k$可交换。

而根据更进一步的数值结果，可以得出以下结论：

-   所有参数$\gamma_i,\ \beta_j$都只会是0或$-\pi/4$。给定层下的酉矩阵门指数上的哈密顿量具有形式$Y_{l}Z_{m}$，其中$l$和$m$是某对量子比特。

-   如果参数取0，只会与大多数酉矩阵门平凡作用。

-   只需$N$步即可找到近似解，其中$N$为伊辛哈密顿量的维度。因此，只需要$N$层如第一点所述形式的酉矩阵门。

由此，Adapt-Clifford求解最大割问题的步骤如下[^3]：

-   仿照Adapt-QAOA，我们定义能量梯度 $$\begin{aligned}
    g_{a,b}^{(r)}= -\sum_{l} \omega_{l, b}\left\langle Z_{l} X_{b} Z_{a}\right\rangle_{r-1}\end{aligned}$$

-   对层数$r=0$，通过Hadamard门$H^{\otimes N}$制备初态$|\psi(0)\rangle$，随机选取$k\in\{1,\cdots,N\}$，得到$|\psi_{0}\rangle=Z_k|\psi(0)\rangle$。我们将量子比特分为活跃量子比特和非活跃量子比特，分别记作$\mathbf{a}^{(0)}=\{k\}$和$\mathbf{b}^{(0)}=\{1,\ldots, N\}\backslash\{k\}$。

-   对层数$r=1$，给定$\mathbf{a}^{(0)}=\{k\}$，我们寻找$j$使得图$\mathcal{G}$中的边$(k,j)$满足
    $$j=\max_{\mathbf{b}^{(0)}}[g_{k,\mathbf{b}^{(0)}}^{(1)}]=\underset{\mathbf{b}^{(0)}}{\operatorname{argmax}}[\omega_{k, \mathbf{b}^{(0)}}]$$
    这会让我们的能量梯度达到最大。而后，根据数值结果的结论，应用门$e^{i\frac{\pi}{4}Y_{k}Z_{j}}$后，得到新的量子态为
    $$|\psi_{1}\rangle=e^{i\frac{\pi}{4}Z_{j}Y_{k}}|\psi_{0}\rangle$$
    而后，我们更新$\mathbf{a}^{(1)}=\{k, j\}$和$\mathbf{b}^{(1)}=\{1,\ldots, N\}\backslash\{k, j\}$。

-   对层数$r=2,\cdots,N-1$，我们寻找量子比特对$(\tilde{l}, b_r)$，其中$\tilde{l}\in\{k, j\}$，以最大化$g_{\tilde{l},b_r}^{(r)}$。

    而后，我们应用门$e^{i\frac{\pi}{4} Z_{\tilde{l}} Y_{b_r}}$得到新的量子态为
    $$|\psi_{r}\rangle=e^{i\frac{\pi}{4} Z_{\tilde{l}} Y_{b_r}}|\psi_{r-1}\rangle$$
    并更新$\mathbf{a}^{(r)}=\mathbf{a}^{(r-1)}\cup \{b_r\}$和$\mathbf{b}^{(r)}=\mathbf{b}^{(r-1)}\backslash\{b_r\}$。

-   最终，我们得到
    $$|\Psi\rangle=\left[\prod_{r=2}^{N-1} e^{i\frac{\pi}{4} Z_{\tilde{l}_r} Y_{b_r}}\right] e^{i\frac{\pi}{4} Z_{j}Y_{k}}Z_{k}|\psi(0)\rangle$$
    根据我们前面的结论，我们无需更多的优化即可将其确认为我们的最终基态解。

基于这篇论文所要解决的问题，我们更关心Adapt-Clifford在大规模（$N=200$）问题中的表现。在近似比方面，对于大规模的问题，Adapt-Clifford解能量期望显著优于经典现有最优的GW算法，其结果基本与GW舍入次数达到$10^5$时持平，能达到平均约0.8986的近似比。在计算效率上，随机化的Adapt-Clifford时间复杂度约为$O(N^{2.7})$，优于GW算法的$O(N^3)$。与此同时，Adapt-Clifford空间复杂度仅为$O(N^2)$。由此我们可见，Adapt-Clifford是一个十分出众的解决伊辛哈密顿量基态与基态能量求解问题的量子启发式算法。

# 方案描述

## 研究方法选择

我们的方法是基于Adapt-Clifford进行更细致的后选择处理，我们称其为后选择Adapt-Clifford（Post-selection
Adapt-Clifford）法。

### 算法设计思路

对于Adapt-Clifford法，我们注意到它已经是一个成型的求解伊辛哈密顿量基态与基态能量的方法，而且，基于数值实验的结果，它没有对参数进行优化，而是直接
在依靠梯度计算选择量子门作用的量子比特，并对这些量子比特作用已知的量子门（$e^{i\frac{\pi}{4}Z_{j}Y_{k}}$），构建量子电路后输出结果。

基于此，Adapt-Clifford在算法效率上体现出了显著优势。一些针对Adapt-Clifford本身的改进策略，如对量子门重新引入参数优化，即考虑量子门$e^{-i\beta_{k} A_k}e^{-i\gamma_{k} H_\mathrm{Ising}}$并将$\gamma,\beta,A_k$都当作参数看待，则其对最终结果的优化与额外引入的资源消耗相比，较为得不偿失。因此，我们考虑在不改变Adapt-Clifford本身的基础上，添加一些后选择的方法，既可以实现最终结果的优化，也不会消耗太多额外的算力资源。

### 后选择方法：后选择

设经过Adapt-Clifford流程后，得到初始优化电路$U_{\text{opt}}$，其输出的近似基态为$|\psi_{\text{approx}}\rangle$，对应能量期望值为$E_{\text{approx}}$。为进一步提升精度，我们引入后选择策略（post-selection）。下面假设我们需要一个最优的电路$U_{\text{best}}$。

-   首先，对近似基态$|\psi_{\text{approx}}\rangle$进行$M$次投影测量，获得$M$个测量基态（经典"01"比特串），记为$\mathcal{S} = \{ s_i \mid s_i \in \{0,1\}^N \}_{i=1}^M$。对每个$s_i$，我们为了减少量子资源的使用而启用经典方法计算能量：
    $$E(s_i) = \mathbf{h}^\top {s}_i + {s}_i^\top \mathbf{J} {s}_i$$

-   若存在$s_k \in \mathcal{S}$满足$E(s_k) < E_{\text{approx}}$，则选择其中能量最低的态$s_{\text{best}} = \underset{s_k}{\arg\min}\   E(s_k)$。

-   如果存在$s_{\text{best}}$，我们构造新量子电路$U_{\text{new}}$，该电路由一组$X$门组成，其中，$X$门作用且仅作用于于$s_{\text{best}}$中取值为1的比特位置，以达到$|0\rangle^{\otimes N}$在经过电路后能得到$s_{\text{best}}$的效果。如果不存在$s_{\text{best}}$，我们直接沿用最初Adapt-Clifford的结果$U_{\text{best}}=U_{\text{opt}}$。

-   如果得到了$U_{\text{new}}$，我们在$U_{\text{new}}$上执行单比特翻转贪心算法，即通过在电路中加入$X$门翻转每一个量子比特并寻找可以降低能量的翻转，直至所有翻转导致的能量差$\Delta E$都大于等于0，我们终止并得到最终的$U_{\text{best}}$。

## 理论依据

### 结果优势

比起Adapt-Clifford，PSAC因为加入了后选择方法，而且通过直接的能量对比在后选择中得到了能量结果优于或者等于Adapt-Clifford的量子电路，因此，PSAC理论上会选择优于Adapt-Clifford的电路，获得更低的能量结果。我们对此提供更详细的理论推导。

首先，对于任何$N$维量子比特$|\psi\rangle$，我们可以将其分解为
$$|\psi\rangle = \sum_{s \in \{0,1\}^N} c_s |s\rangle, \quad \text{其中} \sum_s |c_s|^2 = 1.$$
这里$|s\rangle = |s_1 s_2 \cdots s_N\rangle$是经典"01"比特串。

则此时，我们可以算得 $$\begin{aligned}
\langle H_{\text{Ising}} \rangle 
&= \langle \psi | H_{\text{Ising}} | \psi \rangle \\
&= \left( \sum_{s'} c_{s'}^* \langle s'| \right) H_{\text{Ising}} \left( \sum_{s} c_s |s\rangle \right) \\
&= \sum_{s', s} c_{s'}^* c_s \langle s'| H_{\text{Ising}} |s\rangle\end{aligned}$$
由$H_{\text{Ising}}= -\sum_{i=1}^N h_i \sigma_i^z - \sum_{1 \leq i < j \leq N} J_{ij} \sigma_i^z \sigma_j^z$的形式，$H_{\text{Ising}}$仅在对角线上有非零元素，因此
$$\begin{aligned}
\sum_{s', s} c_{s'}^* c_s \langle s'| H_{\text{Ising}} |s\rangle
=\sum_{s} |c_s|^2 \langle s| H_{\text{Ising}} |s\rangle=\sum_{s} |c_s|^2 E(s)\end{aligned}$$

由此我们有，对于任意一个$N$维量子比特$|\psi\rangle$，其能量是基态能量非负加权的和。这意味着
$$\begin{aligned}
\langle H_{\text{Ising}} \rangle =\sum_{s} |c_s|^2 E(s)=\frac{\sum_{s} |c_s|^2 E(s)}{\sum_s |c_s|^2}\leq \max_{s}E(s)\end{aligned}$$

而如果我们对$|\psi\rangle$进行测量，我们可以以$|c_s|^2$的概率得到$s$。此时我们可以计算得到，即使测量得到比$|\psi\rangle$能量更低的基态的概率只有$1\%$，经过500次测量，我们也可以有$99\%$以上的概率测量到这个基态，而经过数值实验，测量次数还有再次下降的可能。而对于测量来说，500次对算力的消耗并不算大。因此，我们可以通过对Adapt-Clifford的结果进行多次测量取能量最低的基态来达到对量子电路的优化效果。

在此基础上，贪心算法可以为最终结果锦上添花。为了防止贪心算法陷入局部优化陷阱，我们还可以进行重复测验，覆盖更多初始点，提高找到全局低能量解的概率。但是，Adapt-Clifford因为初始量子比特选择的随机性，可能导致最终结果不同，这意味着选取结果可能因为Adapt-Clifford初始量子比特的选择而产生变化。不过，这不与PSAC的理论优势冲突。

### 算力优势

在整个算法流程中，由于Adapt-Clifford优化后得到的量子态没有很好的性质，所以其能量计算为
$$\langle H_{\text{Ising}} \rangle = \sum_{s, s'} c_s^* c_{s'} \langle s' | H_{\text{Ising}} | s \rangle$$

其中$|\psi\rangle = \sum_{s \in \{0,1\}^N} c_s |s\rangle$。求解这个能量的算法有许多，如果直接计算复杂度约为$O(2^NN^2)$，而比如通过Metropolis方法等较为复杂的算法进行计算，可以勉强达到多项式的时间复杂度。但是相应地，如果是对测量基态进行能量计算，由于我们可以直接将能量计算写为
$$E(s) = -\frac{1}{2} \sum_{i \neq j} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i$$
直接计算的计算量只有$O(N^2)$，这意味着减少一般量子态的能量计算，而转为进行测量基态的能量计算，能为我们的算法节约许多时间。此外，我们算得的量子态测量优化后选择的01态$s_i$对应伊辛模型的经典构型，其能量$E(s_i)$为哈密顿量在该构型下的精确值，这对该算法适配NISQ系统也有显著的指导意义。

而对于贪心算法，测量并选择基态的流程会采样到能量更低的基态，比起随机起点，可以更快地完成计算。值得注意的是，Adapt-Clifford的平均近似比已经可以达到0.8986，而测量后选择的基态算得的能量近似比会比Adapt-Clifford更优，因此，贪心算法不会消耗太多的算力。

## 数据采集

我们使用黑客松比赛public数据集的数据"Q_Ising_triu_uniform_complete\_$n$\_$l$.npz"，其中$n=200,400,600,800,1000$为维度，$l=0,1,2,3,4$为序号，所有伊辛模型中$\textbf{J}$和$\textbf{h}$从均匀分布/正态分布中产生。

我们通过我们的算法得到优化的量子电路后，进行能量计算并与纯Adapt-Clifford算法与matlab\
的QUBO算法比较结果，即以matlab的QUBO算法为基准计算近似比并分析结果。除此之外，我们还会采集并分析后选择的额外时间，以及通过多次数值实验得到较优的基态测量次数与贪心算法防局部优化重复次数。

## 算法流程

首先，我们完成Adapt-Clifford的算法。\

::: algorithm
**初始化：** $|\psi_0\rangle \gets |+\rangle^{\otimes N}$
随机选取$k \in \{1,\cdots,N\}$\
$|\psi_0\rangle \gets Z_k |\psi_0\rangle$ $\mathbf{a}^{(0)} \gets \{k\}$
$\mathbf{b}^{(0)} \gets \{1,\ldots,N\}\backslash\{k\}$
:::

::: algorithm
$|\Psi\rangle \gets |\psi_{N-1}\rangle$\
$U\gets \left[\prod_{r=2}^{N-1} e^{i\frac{\pi}{4} Z_{\tilde{l}_r} Y_{b_r}}\right] e^{i\frac{\pi}{4} Z_{j}Y_{k}}Z_{k}H^{\otimes N}$\
:::

值得注意的是，在实际代码中，$k \in \{1,\cdots,N\}$的随机选取一般被替代为$k=1$。这里如果在遍历后选择能量最低的初始量子比特位置，会对计算哈密顿量的能量有较高要求，因此直接用给定$k$的取代原来的方法。

而后，我们通过\

::: algorithm
**初始化：** $M$ $\mathcal{S} \gets \emptyset$

$E_{\min} \gets E_{\text{approx}}$\
$s_{\text{best}} \gets \text{None}$\
:::

::: algorithm
:::

得到一个较优秀的低能量基态与算得基态需要的电路，而后我们再使用贪心算法\

::: algorithm
**初始化：**

$\text{bits} \gets \text{array}[N]$
:::

得到我们最终的结果。

## 代码结构

我们的代码以决赛要求的answer代码为例，我们需要构建一个可以求解伊辛模型基态与基态能量的solve()函数[^4]。

::: CJK
UTF8gbsn

    import ...

    def add_H_layer(nqubit):...
    #/* 添加Hadamard门层$H^{\otimes N}$制备$|+\rangle^{\otimes N}$*/

    def build_hamiltonian(nqubit, h, J):...
    #/* 将Q\_triu转为可以用于mindquantum计算的哈密顿量的形式*/

    def add_YZ_gate(q1, q2):...
    #/* 构造Adapt-Clifford每一步添加的YZ门*/

    def gradient(inaqubit, W, aqubits_k, aqubits_j):...

    def pos_max_grad(inaqubits, W, aqubits_k, aqubits_j):...
    #/* Adapt-Clifford所需梯度计算*/

    def x_gates_circuit(circuit, select_num, nqubit=None):...
    #/* 给出从纯0态演化至目标测量基态的X门序列*/

    def post_selection(circ, ham, nqubit, Q_triu):...
    #/* 对Adapt-Clifford的结果进行后选择得到更优测量基态*/

    def greedy(circ0, ham, nqubit, Q_triu, select_num, repeat):...
    #/* 对后选择的基态进行贪心优化*/

    def solve(nqubit, Q_triu):
        ham = build_hamiltonian(nqubit, np.diag(Q_triu).flatten(), 2 * (Q_triu - np.diag(np.diag(Q_triu))))
        h_0 = np.diag(Q_triu).flatten()
        J_0 = 2 * (Q_triu - np.diag(np.diag(Q_triu)))
        Q_sym = Q_triu + Q_triu.T - np.diag(np.diag(Q_triu))
        Q = nx.from_numpy_array(Q_sym)
        W = nx.adjacency_matrix(Q).toarray()
    #/* 初始化输入*/

        best_circ = add_H_layer(nqubit)
    #/* 为电路添加一层Hadamard门制备初态*/
        fqubit = 1
    #/* 随机选择初始量子比特位置（此处固定为1）*/
        best_circ += Z.on(1) 
    #/* 第1步，为在初始量子比特位置添加Z门*/

        sim = Simulator('stabilizer', nqubit)
        sim.apply_circuit(best_circ)

        active_qubits_k = []
        active_qubits_j = []
        inactive_qubits = list(range(nqubit))
        quantum_state = sim.get_qs()

        gate_positions = []
        for nn in range(nqubit - 1):
            if nn == 0:
                nonzero_indices = np.nonzero(W[:, fqubit])[0].astype(int).tolist()
                qpair = int(np.random.choice(nonzero_indices))
                active_qubits_j.append(qpair)
                active_qubits_k.append(fqubit)
                inactive_qubits.remove(qpair)
                inactive_qubits.remove(fqubit)
                qubits = (fqubit, qpair)
            else:
                aset, qpair, gra = pos_max_grad(inactive_qubits, W, active_qubits_k, active_qubits_j)
                qpair = int(qpair)
                if aset == "k":
                    qubits = (qpair, fqubit)
                    active_qubits_k.append(qpair)
                elif aset == "j":
                    qubits = (qpair, active_qubits_j[0])
                    active_qubits_j.append(qpair)

                inactive_qubits.remove(qpair)
            best_circ += add_YZ_gate(int(qubits[0]), int(qubits[1]))
            gate_positions.append(qubits)
    #/* 第2-$N$步，为取得最大梯度的量子比特位置添加YZ门*/
        sim.reset()
        best_circ2, select_num = post_selection(best_circ, ham, nqubit, Q_triu)
    #/* 对Adapt-Clifford得到的优化量子态进行后选择*/
        return greedy(best_circ2, ham, nqubit, Q_triu, select_num=select_num, repeat=2)
    #/* 对后选择的基态进行贪心优化*/
:::

# 结果与分析

## 公开数据集测试结果与分析

### 后选择优化结果

根据附录B.1的数据表，我们将我们的数据绘制成可视化的图表以供更明确的分析。

![后选择Adapt-Clifford与Adapt-Clifford结果对比图](2.png){width="80%"}

我们可以看到，后选择Adapt-Clifford的能量近似比（红色折线）无论在哪个维度都要优于Adapt-Clifford的能量近似比（蓝色折线），并且其与达成显著优势（后选择有效贡献）的次数并没有直接关联。除此以外，后选择无贡献的运行次数没有超过3次，这意味着后选择运行4次以内，理论上我们对所有公开数据集达到后选择优势。

### 后选择时间消耗

我们在每个维度各挑一个伊辛模型，并对比Adapt-Clifford与后选择Adapt-Clifford的主函数运行时间。

::: {#tab:time_performance}
   **维度**   **类别**   **A/P及比值测量结果**                                     **平均用时/比值**     
  ---------- ---------- ----------------------- -------- -------- ------- ------- ------------------- -- --
     200         A               1084             793      1610    1199     932          1124            
                 P               2762             1910     1768    3306    1874          2324            
               *A/P*             0.392           0.415    0.911    0.363   0.497        0.4836           
     400         A               6274             4512     4804    4339    4824          4951            
                 P               11071           11034    11145    12660   11594         11501           
               *A/P*             0.567           0.409    0.431    0.343   0.416        0.4305           
     600         A               21265           19669    15522    15859   15774         17618           
                 P               31047           29720    36743    33110   33325         32789           
               *A/P*             0.685           0.662    0.423    0.479   0.473        0.5373           
     800         A               40529           36240    51618      /       /           42796           
                 P              106622           90232    83166      /       /           93340           
               *A/P*             0.380           0.402    0.621      /       /          0.4585           
     1000        A               79134           97652    65405      /       /           80730           
                 P              174519           241255   208782     /       /          208185           
               *A/P*             0.454           0.405    0.313      /       /          0.3878           

  : Adapt-Clifford(A)/后选择Adapt-Clifford(P)在不同维度下的算法时间性能分析（单位：ms）
:::

由上表我们可以看出，后选择Adapt-Clifford的运行时间基本可以控制在Adapt-Clifford的3倍以内，并且基本没有表现出维度灾难。因此，考虑到后选择Adapt-Clifford对于Adapt-Clifford的最终结果提升情况，我们认为后选择Adapt-Clifford的额外时间成本是可以接受的。

## 隐藏数据集测试结果

根据隐藏数据集反馈的代码提交结果，基于Adapt-Clifford的代码一共提交8次，成功3次，平均成绩171086.67，最好成绩172231.3941；基于后选择Adapt-Clifford的代码一共提交28次，成功11次，平均成绩178141.19，最好成绩179720.298。由此我们可以看见后选择为Adapt-Clifford带来的显著优势。

## 测量次数与重复贪心算法次数数值测试结果与分析

基于附录B.2的数据，我们对以下情况进行分析以求得更优的测量次数与重复贪心算法次数。

### 不同测量次数下优化比率$\neq$`<!-- -->`{=html}1的占比

   测量次数$s$   总实验组数   $\neq$`<!-- -->`{=html}1组数   占比($\%$)
  ------------- ------------ ------------------------------ ------------
       250           12                    6                    50.0
       500           11                    6                    54.5
      1000           11                    6                    54.5

由此我们可见，测量次数从250增加至500可以较小程度提升优化概率，而从500提升至1000没有提升优化概率，表明后选择改善能量的概率与测量规模关系较小，我们可以优先选择250或500次测量。

### 不同测量次数下平均后选择时间

   测量次数$s$   平均时间(s)   时间增长倍数
  ------------- ------------- --------------
       250          2.06           1.0
       500          3.88           1.9
      1000          8.01           3.9

我们可以明显看到，后选择算法时间与$s$呈线性增长关系，$s=1000$时耗时达$s=250$的3.9倍，这印证了计算复杂度的$O(s)$特性。

### 重复次数$r$对优化比率的影响

因为优化比率为1的情况下，不会涉及到对基态进行贪心反转，因此，我们只考虑优化比率大于1的情况。

   $s$     $r$    优化比率均值   提升幅度
  ------ ------- -------------- -----------
   250      1        1.040         0.040
          **2**    **1.053**     **0.053**
            3        1.042         0.042
   500      1        1.052         0.052
          **2**    **1.062**     **0.062**
            3        1.049         0.049
   1000     1        1.043         0.043
          **2**    **1.053**     **0.053**
            3        1.050         0.050

所有$s$下$r=2$时优化效果最佳（平均提升5.1%--6.2%），$r=3$时均出现回落。这是因为量子门操作误差($\varepsilon_g$)与测量噪声($\varepsilon_m$)在$r>2$时显著累积：
$$\varepsilon_{\text{total}} = 1 - \prod_{k=1}^{r}(1-\varepsilon_k) \approx r\cdot\bar{\varepsilon}$$
导致优化比率在$r=3$时回落。

### 最优参数组合

基于以上数据分析，我们可以看到，选择测量次数/贪心重复次数为$s=500/r=2$对算法算力的提升幅度最高（6.2%），比$s=250/r=2$、$s=1000/r=2$高17%。并且，此时平均耗时适中，性价比为所有组合最高，也可以在硬件限制与噪声控制间取得平衡，避免$s=1000$的高退相干风险与$r=3$的噪声累积风险。因此，我们为我们的算法选择$s=500/r=2$的参数。

## 结论

基于Adapt-Clifford的后选择Adapt-Clifford方法能够在不消耗过多额外时间与算力的情况下达到比Adapt-Clifford显著更好的结果。可是，值得注意的是，后选择的测量会引入新的噪声，因此我们还需要在后选择的fault-tolerance下再进行额外考量。

未来，在时间允许的情况下，后选择Adapt-Clifford方法还可以通过多次重复后选择的方法达到更好的优化效果，也可以结合其他变分量子方法解决其他优化问题；而Adapt-Clifford方法本身也可以通过引入参数和更多Clifford门来得到更好的结果。最大割问题以及一系列QUBO问题都在各种领域有着广泛运用，我们希望能有更多的量子算法为更高效解决QUBO作出贡献。

# 参考文献

1.  Bennett D. Numerical Solutions to the Ising Model using the
    Metropolis Algorithm\[J\]. JS TP, 2016, 13323448.

2.  Farhi E, Goldstone J, Gutmann S. A quantum approximate optimization
    algorithm\[J\]. arXiv preprint arXiv:1411.4028, 2014.

3.  Ferreira Fialho dos Anjos M N. New Convex Relaxations for the
    Maximum Cut and VLSI Layout Problems\[J\]. 2001.

4.  Goto H, Tatsumura K, Dixon A R. Combinatorial optimization by
    simulating adiabatic bifurcations in nonlinear Hamiltonian
    systems\[J\]. Science advances, 2019, 5(4): eaav2372.

5.  Muñoz-Arias M H, Kourtis S, Blais A. Low-depth Clifford circuits
    approximately solve MaxCut\[J\]. Physical Review Research, 2024,
    6(2): 023294.

6.  Pawłowski J, Tuziemski J, Tarasiuk P, et al. VeloxQ: A Fast and
    Efficient QUBO Solver\[J\]. arXiv preprint arXiv:2501.19221, 2025.

7.  Wang R S, Wang L M. Maximum cut in fuzzy nature: Models and
    algorithms\[J\]. Journal of Computational and Applied Mathematics,
    2010, 234(1): 240-252.

8.  Zhou L, Wang S T, Choi S, et al. Quantum approximate optimization
    algorithm: Performance, mechanism, and implementation on near-term
    devices\[J\]. Physical Review X, 2020, 10(2): 021067.

9.  量子前哨. QAOA如何在NISQ处理器中展示应用级量子优势？\[EB/OL\].
    (2022-03-15)\[2025-06-15\]. https://zhuanlan.zhihu.com/p/481136805.

# 附录 A {#附录-a .unnumbered}

## A.1 完整比赛代码answer.py {#a.1-完整比赛代码answer.py .unnumbered}

    import numpy as np
    import networkx as nx
    from mindquantum import Circuit, X, Y, I, H, Z, S, CNOT, Hamiltonian, QubitOperator, Simulator
    from itertools import combinations
    import copy
    from mindquantum.simulator import get_stabilizer_string, get_tableau_string
    from collections import deque
    from mindquantum.core.circuit import Circuit, UN
    import random
    from mindquantum.core.gates import Measure
    from mindquantum.io import OpenQASM

    def add_H_layer(nqubit):
        circ = Circuit()
        for i in range(nqubit):
            circ += H.on(i)
        return circ


    def build_hamiltonian(nqubit, h, J):
        qubit_op = QubitOperator()
        for i in range(nqubit):
            if abs(h[i]) > 1e-10:
                qubit_op += QubitOperator(f'Z{i}', h[i])
        qubit_pairs = list(combinations(range(nqubit), 2))
        for i, j in qubit_pairs:
            if i >= j:
                continue
            if abs(J[i][j]) > 1e-10:
                qubit_op += QubitOperator(f'Z{i} Z{j}', J[i][j])
        return Hamiltonian(qubit_op)


    def add_YZ_gate(q1, q2):
        c = Circuit()
        c += Z.on(q1)
        c += S.on(q1)

        c += H.on(q2)
        c += CNOT.on(q2, q1)
        c += Z.on(q1)

        c += H.on(q1)
        c += S.on(q1)
        c += H.on(q1)
        c += S.on(q1)
        c += S.on(q1)

        c += CNOT.on(q2, q1)
        c += S.on(q1)
        c += H.on(q2)

        return c


    def gradient(inaqubit, W, aqubits_k, aqubits_j):
        lindex_k = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_k)
        lindex_j = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_j)
        sum_weights_k = np.sum(W[ll, inaqubit] for ll in lindex_k)
        sum_weights_j = np.sum(W[ll, inaqubit] for ll in lindex_j)
        return -sum_weights_k + sum_weights_j


    def pos_max_grad(inaqubits, W, aqubits_k, aqubits_j):
        all_grads_k = [gradient(inaqubit, W, aqubits_k, aqubits_j) for inaqubit in inaqubits]
        all_grads_j = -1.0 * np.array(all_grads_k)

        pos_max_k = np.argmax(all_grads_k)
        pos_max_j = np.argmax(all_grads_j)

        if all_grads_k[pos_max_k] > all_grads_j[pos_max_j]:
            return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
        elif all_grads_k[pos_max_k] < all_grads_j[pos_max_j]:
            return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j]
        else:
            return ("k", inaqubits[pos_max_k], all_grads_k[pos_max_k]) if np.random.choice([0, 1]) else \
                ("j", inaqubits[pos_max_j], all_grads_j[pos_max_j])



    def x_gates_circuit(circuit, select_num, nqubit=None):
        bits = np.zeros(nqubit)

        for i in range(nqubit):
            if i in select_num:
                circuit += X.on(i)
                bits[i] = -1
            else:
                bits[i] = 1

        return circuit, bits


    def greedy(circ0, ham, nqubit, Q_triu, select_num, repeat):
        J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
        h = np.diag(Q_triu)

        best_circ = copy.deepcopy(circ0)
        best_energy = 100
        best_depth = None
        sim = Simulator('stabilizer', nqubit)
        for ii in range(repeat):
            circ = Circuit(UN(I, nqubit))
            if select_num is None:
                break
            else:
                circ, bits = x_gates_circuit(circ, select_num, nqubit)

                while True:
                    delta_E = (2 * J @ bits + h) * -2 * bits
                    index = np.argmin(delta_E)
                    bits[index] = -bits[index]
                    if delta_E[index] < 0:
                        circ += X.on(int(index))
                    else:
                        break

            sim.reset()
            exp = sim.get_expectation(ham, circ)
            energy = exp.real

            if energy < best_energy:
                best_energy = energy
                best_circ = copy.deepcopy(circ)
        return best_circ


    def post_selection(circ, ham, nqubit, Q_triu):
        circ_b = copy.deepcopy(circ)
        expectation_b = Simulator('stabilizer', circ_b.n_qubits).get_expectation(ham, circ_b).real
        print(expectation_b)

        measure_circuit = Circuit()
        for index in range(0, circ.n_qubits):
            measure_circuit += Measure().on(index)

        sim = Simulator('stabilizer', circ.n_qubits)
        sim.apply_circuit(circ)
        result = sim.sampling(measure_circuit, shots=200)
        output_dic = result.data
        select_num_best = []
        for key in output_dic:
            select_num = []
            circ0 = Circuit()
            circ0 += Circuit(UN(I, nqubit))
            x0 = np.ones(nqubit)
            for index in range(circ.n_qubits):
                if key[index] == '1':
                    circ0 += X.on(circ.n_qubits - index - 1)
                    x0[circ.n_qubits - index - 1] = -1
                    select_num.append(circ.n_qubits - index - 1)

            output0 = 0
            nq = Q_triu.shape[0]
            for i in range(nq):
                output0 += Q_triu[i, i] * x0[i]

            for j in range(nq):
                for k in range(j + 1, nq):
                    if Q_triu[j, k] != 0:
                        output0 += 2 * Q_triu[j, k] * x0[j] * x0[k]
            if output0 <= expectation_b:
                circ_b = copy.deepcopy(circ0)
                expectation_b = output0
                select_num_best = copy.deepcopy(select_num)
        if len(select_num_best) == 0:
            select_num_best = copy.deepcopy(select_num)

        return circ_b, select_num_best


    def solve(nqubit, Q_triu):
        ham = build_hamiltonian(nqubit, np.diag(Q_triu).flatten(), 2 * (Q_triu - np.diag(np.diag(Q_triu))))

        h_0 = np.diag(Q_triu).flatten()
        J_0 = 2 * (Q_triu - np.diag(np.diag(Q_triu)))

        Q_sym = Q_triu + Q_triu.T - np.diag(np.diag(Q_triu))
        Q = nx.from_numpy_array(Q_sym)
        W = nx.adjacency_matrix(Q).toarray()

        best_circ = add_H_layer(nqubit)
        fqubit = 1 
        best_circ += Z.on(1)

        sim = Simulator('stabilizer', nqubit)
        sim.apply_circuit(best_circ)

        active_qubits_k = []
        active_qubits_j = []
        inactive_qubits = list(range(nqubit))
        quantum_state = sim.get_qs()

        gate_positions = []
        for nn in range(nqubit - 1):
            if nn == 0:
                nonzero_indices = np.nonzero(W[:, fqubit])[0].astype(int).tolist()
                qpair = int(np.random.choice(nonzero_indices))
                active_qubits_j.append(qpair)
                active_qubits_k.append(fqubit)
                inactive_qubits.remove(qpair)
                inactive_qubits.remove(fqubit)
                qubits = (fqubit, qpair)
            else:
                aset, qpair, gra = pos_max_grad(inactive_qubits, W, active_qubits_k, active_qubits_j)
                qpair = int(qpair)
                if aset == "k":
                    qubits = (qpair, fqubit)
                    active_qubits_k.append(qpair)
                elif aset == "j":
                    qubits = (qpair, active_qubits_j[0])
                    active_qubits_j.append(qpair)

                inactive_qubits.remove(qpair)
            best_circ += add_YZ_gate(int(qubits[0]), int(qubits[1]))
            gate_positions.append(qubits)
        sim.reset()
        best_circ2, select_num = post_selection(best_circ, ham, nqubit, Q_triu)
        return greedy(best_circ2, ham, nqubit, Q_triu, select_num=select_num, repeat=2)

## A.2 完整能量计算代码 {#a.2-完整能量计算代码 .unnumbered}

### A.2.1 能量计算比较代码final_comparison.py {#a.2.1-能量计算比较代码final_comparison.py .unnumbered}

    import numpy as np
    import stim
    import itertools
    import networkx as nx
    import time

    start_time = time.time()
    def add_H_layer(nbit, c: stim.TableauSimulator):
        c.h(*range(nbit))
    def hamil_terms(nbit, h, J, combis, term):
        terms = []
        for i in range(nbit):
            if abs(h[i]) > 1e-10:
                pstring = stim.PauliString(nbit)
                pstring[i] = term
                terms.append((h[i], pstring))

        for (i, j) in combis:
            if i >= j:
                continue
            if abs(J[i, j]) > 1e-10:
                pstring = stim.PauliString(nbit)
                pstring[i] = term
                pstring[j] = term
                terms.append((J[i, j], pstring))
        return terms

    def weights_vector(nbit, h, J, combis):
        weights = []
        for i in range(nbit):
            if abs(h[i]) > 1e-10:
                weights.append(h[i])

        for (i, j) in combis:
            if i < j and abs(J[i, j]) > 1e-10:
                weights.append(J[i, j])
        return weights

    def hamil_expectation_vals(terms, c: stim.TableauSimulator):
        vals = []
        for coeff, term in terms:
            val = c.peek_observable_expectation(term)
            vals.append(val)
        return vals

    def current_energy(weights, hterms, c: stim.TableauSimulator):
        expects = hamil_expectation_vals(hterms, c)
        return np.dot(weights, expects)

    def add_YZ_gate(q1, q2, c: stim.TableauSimulator):
        c.s_dag(q1)
        c.h(q2)
        c.cnot(q1, q2)
        c.z(q1)
        c.h_yz(q1)
        c.cnot(q1, q2)
        c.s(q1)
        c.h(q2)

    def gradient(inaqubit, W, aqubits_k, aqubits_j, c: stim.TableauSimulator):

        lindex_k = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_k)
        lindex_j = np.intersect1d(np.nonzero(W[:, inaqubit])[0], aqubits_j)

        sum_weights_k = np.sum(W[ll, inaqubit] for ll in lindex_k)
        sum_weights_j = np.sum(W[ll, inaqubit] for ll in lindex_j)

        grad_k = -sum_weights_k + sum_weights_j
        return grad_k

    def pos_max_grad(inaqubits, W, aqubits_k, aqubits_j, c: stim.TableauSimulator):

        all_grads_k = [gradient(inaqubit, W, aqubits_k, aqubits_j, c) for inaqubit in inaqubits]
        all_grads_j = -1.0 * np.array(all_grads_k)

        pos_max_k = np.argmax(all_grads_k)
        pos_max_j = np.argmax(all_grads_j)

        if all_grads_k[pos_max_k] > all_grads_j[pos_max_j]:
            return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
        elif all_grads_k[pos_max_k] < all_grads_j[pos_max_j]:
            return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j]
        else:
            char = np.random.choice([1, 2])
            if char == 1:
                return "k", inaqubits[pos_max_k], all_grads_k[pos_max_k]
            elif char == 2:
                return "j", inaqubits[pos_max_j], all_grads_j[pos_max_j]
    def post_selection(sim: stim.TableauSimulator, wvec, hterms, nqubit, Q_triu, s):
        circuit_0 = stim.Circuit()
        expectation_b = current_energy(wvec, hterms, sim).real
        x1 = np.zeros(nqubit)
        select_num_best = []
        best_energy = expectation_b
        best_circ = None
        pp = 1

        for ss in range(s):
            for i in range(nqubit):
                x1[i] = sim.measure(i)
            circ0 = stim.Circuit()
            x0 = np.ones(nqubit)
            select_num = []

            for i in range(nqubit):
                if x1[i]: 
                    circ0.append("X", [i])
                    x0[i] = -1
                    select_num.append(i)

            output0 = 0
            for i in range(nqubit):
                output0 += Q_triu[i, i] * x0[i]
            for j in range(nqubit):
                for k in range(j + 1, nqubit):
                    if Q_triu[j, k] != 0:
                        output0 += 2 * Q_triu[j, k] * x0[j] * x0[k]

            if output0 < best_energy:
                best_energy = output0
                best_circ = circ0.copy()
                select_num_best = select_num.copy()

        if best_circ is None:
            best_circ = circuit_0.copy()
            select_num_best = []
            pp = 0
            print("No change!")

        return best_circ, select_num_best, pp


    def greedy(circ0: stim.Circuit, wvec, hterms, nqubit, Q_triu, select_num, repeat, pp, ener):
        J = np.triu(Q_triu, k=1) + np.triu(Q_triu, k=1).T
        h = np.diag(Q_triu)
        if pp == 0:
            ener_ps = ener
        else:
            best_circ = circ0.copy()
            best_energy = float('inf')

            for _ in range(repeat):
                circ = stim.Circuit()
                bits = np.zeros(nqubit)  

                if select_num is not None:
                    for idx in select_num:
                        circ.append("X", [idx])
                        bits[idx] = 1  

                while True:
                    delta_E = np.zeros(nqubit)
                    for i in range(nqubit):
                        s_i = bits[i]

                        delta = 0
                        delta += h[i] * (-2 * (1 if s_i == 0 else -1))

                        for j in range(nqubit):
                            if i != j:
                                s_j = bits[j]
                                delta += 2 * J[i, j] * (-2 * (1 if s_i == 0 else -1)) * (1 if s_j == 0 else -1)

                        delta_E[i] = delta

                    min_delta = np.min(delta_E)
                    if min_delta >= 0:
                        break

                    flip_index = np.argmin(delta_E)
                    bits[flip_index] = 1 - bits[flip_index]
                    circ.append("X", [flip_index])

                sim = stim.TableauSimulator()
                sim.do_circuit(circ)
                energy = current_energy(wvec, hterms, sim).real

                if energy < best_energy:
                    best_energy = energy
                    best_circ = circ.copy()

            sim_ps = stim.TableauSimulator()
            sim_ps.do_circuit(best_circ)
            ener_ps = current_energy(wvec, Hterms, sim_ps).real

        return ener_ps

    data_0 = np.load('Q_ising_triu_uniform_complete_200_0.npz')
    data_1 = np.load('Q_ising_triu_uniform_complete_200_1.npz')
    data_2 = np.load('Q_ising_triu_uniform_complete_200_2.npz')
    data_3 = np.load('Q_ising_triu_uniform_complete_200_3.npz')
    data_4 = np.load('Q_ising_triu_uniform_complete_200_4.npz')
    data_5 = np.load('Q_ising_triu_uniform_complete_400_0.npz')
    data_6 = np.load('Q_ising_triu_uniform_complete_400_1.npz')
    data_7 = np.load('Q_ising_triu_uniform_complete_400_2.npz')
    data_8 = np.load('Q_ising_triu_uniform_complete_400_3.npz')
    data_9 = np.load('Q_ising_triu_uniform_complete_400_4.npz')
    data_10 = np.load('Q_ising_triu_uniform_complete_600_0.npz')
    data_11 = np.load('Q_ising_triu_uniform_complete_600_1.npz')
    data_12 = np.load('Q_ising_triu_uniform_complete_600_2.npz')
    data_13 = np.load('Q_ising_triu_uniform_complete_600_3.npz')
    data_14 = np.load('Q_ising_triu_uniform_complete_600_4.npz')
    data_15 = np.load('Q_ising_triu_uniform_complete_800_0.npz')
    data_16 = np.load('Q_ising_triu_uniform_complete_800_1.npz')
    data_17 = np.load('Q_ising_triu_uniform_complete_800_2.npz')
    data_18 = np.load('Q_ising_triu_uniform_complete_800_3.npz')
    data_19 = np.load('Q_ising_triu_uniform_complete_800_4.npz')
    data_20 = np.load('Q_ising_triu_uniform_complete_1000_0.npz')
    data_21 = np.load('Q_ising_triu_uniform_complete_1000_1.npz')
    data_22 = np.load('Q_ising_triu_uniform_complete_1000_2.npz')
    data_23 = np.load('Q_ising_triu_uniform_complete_1000_3.npz')
    data_24 = np.load('Q_ising_triu_uniform_complete_1000_4.npz')

    Q_triu = data_0['arr_0']
    data_0.close()

    h_0 = np.diag(Q_triu).flatten()  
    J_0 = 2 * (Q_triu - np.diag(np.diag(Q_triu)))  

    Q_sym = Q_triu + Q_triu.T - np.diag(np.diag(Q_triu))
    Q = nx.from_numpy_array(Q_sym)
    nbit = Q.number_of_nodes()
    W = nx.adjacency_matrix(Q).toarray()
    nzcombis = list(itertools.combinations(range(nbit), 2))

    s = 500
    r = 2

    wvec = weights_vector(nbit, h_0, J_0, nzcombis)
    Hterms = hamil_terms(nbit, h_0, J_0, nzcombis, "Z")

    simulator = stim.TableauSimulator()
    add_H_layer(nbit, simulator)

    fqubit = 1
    simulator.z(fqubit)

    active_qubits_k = []
    active_qubits_j = []
    inactive_qubits = list(range(nbit))

    gate_posis = []
    aratio = np.zeros(nbit)
    for nn in range(nbit - 1):
        if nn == 0:
            qpair = np.random.choice(W[:, fqubit].nonzero()[0])
            gra = W[qpair, fqubit]
            qubits = (fqubit, qpair)

            active_qubits_j.append(qpair)
            active_qubits_k.append(fqubit)

            inactive_qubits.remove(qpair)
            inactive_qubits.remove(fqubit)
        else:
            aset, qpair, gra = pos_max_grad(inactive_qubits, W, active_qubits_k, active_qubits_j, simulator)

            if aset == "k":
                qubits = (qpair, fqubit)
                active_qubits_k.append(qpair)
            elif aset == "j":
                qubits = (qpair, active_qubits_j[0])
                active_qubits_j.append(qpair)

            inactive_qubits.remove(qpair)

        add_YZ_gate(qubits[0], qubits[1], simulator)
        gate_posis.append(qubits)

    ener = current_energy(wvec, Hterms, simulator).real
    elapsed_1 = time.time() - start_time

    best_circ2, select_num, pp = post_selection(simulator, wvec, Hterms, nbit, Q_triu, s)
    ener_ps = greedy(best_circ2,  wvec, Hterms, nbit, Q_triu, select_num, r, pp, ener)
    elapsed_2 = time.time() - start_time


    print(f"Energy = {ener}, Post-selected Energy = {ener_ps}, time1={elapsed_1}, time2={elapsed_2}")

### A.2.2 数据绘图代码final_data.py {#a.2.2-数据绘图代码final_data.py .unnumbered}

    import numpy as np
    import matplotlib.pyplot as plt

    data = np.array([
        [-290.910, -301.048, -316.628, 0],
        [-153.575, -166.171, -176.071, 0],
        [-2354.801, -2447.171, -2471.3, 2],
        [-242.493, -249.768, -281.375, 0],
        [-313.123, -319.978, -356.019, 3],
        [-792.670, -814.145, -886.967, 0],
        [-326.982, -345.210, -368.741, 0],
        [-6477.687, -6802.420, -7134.9, 0],
        [-687.736, -691.668, -785.583, 2],
        [-915.269, -952.392, -1019.4, 0],
        [-1485.690, -1541.162, -1594.1, 0],
        [-478.786, -506.927, -544.967, 0],
        [-11329.841, -11903.961, -12781, 2],
        [-1255.517, -1316.355, -1468.5, 0],
        [-1744.413, -1823.275, -1893.3, 2],
        [-2236.158, -2343.601, -2499.1, 0],
        [-651.450, -676.167, -733.797, 1],
        [-17669.001, -18298.299, -20070, 3],
        [-1997.416, -2073.509, -2249.9, 1],
        [-2609.977, -2711.277, -2851.8, 0],
        [-3201.850, -3317.467, -3342.7, 0],
        [-835.852, -861.065, -922.993, 0],
        [-25805.954, -26623.258, -28086, 1],
        [-2834.459, -2949.677, -3109, 0],
        [-3722.137, -3896.329, -4045.1, 0]
    ])


    col3 = data[:, 0]  
    col4 = data[:, 1]  
    col5 = data[:, 2]  
    col6 = data[:, 3]  

    a_1 = col3 / col5  
    a_2 = col4 / col5  
    f_1 = col6         

    f_1_mapped = 0.85 + f_1 * 0.03

    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    plt.subplots_adjust(hspace=0.4)  

    for i in range(5):
        start_idx = i * 5
        end_idx = start_idx + 5
        x = np.arange(1, 6)  

        axs[i].plot(x, a_1[start_idx:end_idx],
                    'o-', color='blue', linewidth=2, markersize=8,
                    label='Adapt-Clifford')  

        axs[i].plot(x, a_2[start_idx:end_idx],
                    's-', color='red', linewidth=2, markersize=8,
                    label='Post-selection Adapt-Clifford')  

        axs[i].scatter(x, f_1_mapped[start_idx:end_idx],
                       color='orange', s=100, zorder=5,
                       label='optimization run counts')  

        for j, (a1_val, a2_val, f_val) in enumerate(zip(a_1[start_idx:end_idx],
                                                        a_2[start_idx:end_idx],
                                                        f_1[start_idx:end_idx])):
            axs[i].text(j + 1, a1_val + 0.005, f'{a1_val:.3f}',
                        ha='center', va='bottom', fontsize=9, color='blue')
            axs[i].text(j + 1, a2_val - 0.005, f'{a2_val:.3f}',
                        ha='center', va='top', fontsize=9, color='red')
            if f_val > 0:
                axs[i].text(j + 1, f_1_mapped[start_idx:end_idx][j] + 0.005,
                            f'f={int(f_val)}', ha='center', fontsize=9, color='orange')

        axs[i].set_title(f'dimension {200*(i + 1)} (label {start_idx + 1}-{end_idx})', fontsize=12)
        axs[i].set_ylabel('Approximation Ratio', fontsize=10)
        axs[i].set_ylim(0.85, 1.0)  
        axs[i].set_xlim(0.5, 5.5)  
        axs[i].set_xticks(x) 

        if i == 0:
            axs[i].legend(loc='upper right', fontsize=10)

    axs[4].set_xlabel('data label', fontsize=12)

    plt.suptitle('Visualization of optimization results', fontsize=16, y=0.95)
    plt.savefig('2.png', dpi=300, bbox_inches='tight')
    plt.show()

## A.3 时间验证代码 {#a.3-时间验证代码 .unnumbered}

### A.3.1 Adapt-Clifford时间验证final_adcl.ipynb {#a.3.1-adapt-clifford时间验证final_adcl.ipynb .unnumbered}

    import...

    def...

    def solve_ac(nqubit, Q_triu):
        ham = ...
        for nn in range(nqubit - 1): ...
        return best_circ

    if __name__ == "__main__":
        data_0 = ...
        Q_triu = data_0['arr_0']
        data_0.close()
        nqubit = 200
        circ_0_test = solve_ac(nqubit, Q_triu)

### A.3.2 后选择Adapt-Clifford时间验证final_circuit.ipynb {#a.3.2-后选择adapt-clifford时间验证final_circuit.ipynb .unnumbered}

    import...

    def...

    def solve(nqubit, Q_triu):
        ham = ...
        for nn in range(nqubit - 1): ...
        best_circ2, select_num = post_selection(best_circ, ham, nqubit, Q_triu)
        return greedy(best_circ2, ham, nqubit, Q_triu, select_num=select_num, repeat=2)

    if __name__ == "__main__":
        data_0 = ...
        Q_triu = data_0['arr_0']
        data_0.close()
        nqubit = 200
        circ_0_test = solve_ac(nqubit, Q_triu)

## A.4 Matlab QUBO验证代码qubo_sol.m {#a.4-matlab-qubo验证代码qubo_sol.m .unnumbered}

``` {.matlab language="Matlab"}
clear all
load('Q_0.mat');
n = size(Q, 1);      
h = diag(Q); 

J = 2 * triu(Q, 1) + 2 * triu(Q, 1)';

A = 2 * J;

row_sums = sum(J, 2);
col_sums = sum(J, 1);

b = 2 * h - 2 * row_sums;

d = sum(J, 'all')/2 - sum(h);
qprob=qubo(A,b,d);
result=solve(qprob)
```

# 附录 B {#附录-b .unnumbered}

## B.1 Adapt-Clifford与后选择Adapt-Clifford算法对比总结数据 {#b.1-adapt-clifford与后选择adapt-clifford算法对比总结数据 .unnumbered}

这里的数据第一列是序号，第二列是数据编号，其中前面的数字$N$（200，600等）表示这是$N$维的伊辛模型哈密顿量，其数据生成服从均匀分布或正态分布。

第三列和第四列是同一个能量计算代码下，Adapt-Clifford与后选择Adapt-Clifford的结果对比。即，后选择Adapt-Clifford的结果是对Adapt-Clifford结果的直接优化。第五列是用matlab中的QUBO算法得到的能量结果，我们将其用于替换近似比中的最小能量值并与Adapt-Clifford与后选择Adapt-Clifford的结果计算近似比。

第六列是在后选择基态测量次数为500次，贪心算法重复2次的情况下，后选择没有测量到比Adapt-Clifford结果更优基态的代码运行次数。

::: {#tab:result}
   序号   数据编号   Adapt-Clifford   后选择Adapt-Clifford   MATLAB QUBO   后选择无效次数  
  ------ ---------- ---------------- ---------------------- ------------- ---------------- --
    1      200_0        -290.910            -301.048          -316.628           0         
    2      200_1        -153.575            -166.171          -176.071           0         
    3      200_2       -2354.801           -2447.171           -2471.3           2         
    4      200_3        -242.493            -249.768          -281.375           0         
    5      200_4        -313.123            -319.978          -356.019           3         
    6      400_0        -792.670            -814.145          -886.967           0         
    7      400_1        -326.982            -345.210          -368.741           0         
    8      400_2       -6477.687           -6802.420           -7134.9           0         
    9      400_3        -687.736            -691.668          -785.583           2         
    10     400_4        -915.269            -952.392           -1019.4           0         
    11     600_0       -1485.690           -1541.162           -1594.1           0         
    12     600_1        -478.786            -506.927          -544.967           0         
    13     600_2       -11329.841          -11903.961          -12781            2         
    14     600_3       -1255.517           -1316.355           -1468.5           0         
    15     600_4       -1744.413           -1823.275           -1893.3           2         
    16     800_0       -2236.158           -2343.601           -2499.1           0         
    17     800_1        -651.450            -676.167          -733.797           1         
    18     800_2       -17669.001          -18298.299          -20070            3         
    19     800_3       -1997.416           -2073.509           -2249.9           1         
    20     800_4       -2609.977           -2711.277           -2851.8           0         
    21     1000_0      -3201.850           -3317.467           -3342.7           0         
    22     1000_1       -835.852            -861.065          -922.993           0         
    23     1000_2      -25805.954          -26623.258          -28086            1         
    24     1000_3      -2834.459           -2949.677            -3109            0         
    25     1000_4      -3722.137           -3896.329           -4045.1           0         

  : 三种算法结果对比与后选择有效优运行次数
:::

## B.2 测量次数与重复贪心算法次数数值测试数据据 {#b.2-测量次数与重复贪心算法次数数值测试数据据 .unnumbered}

::: {#tab:quantum_optimization}
   测量次数$s$   重复次数$r$   优化比率   后选择时间   优化比率=1
  ------------- ------------- ---------- ------------ ------------
       250            1         1.0000      1.9197         是
       250            1         1.0362      1.8833         否
       250            1         1.0439      1.8856         否
       250            2         1.0000      1.6492         是
       250            2         1.0000      1.5086         是
       250            2         1.0000      1.5916         是
       250            2         1.0411      2.3247         否
       250            2         1.0000      1.9737         是
       250            2         1.0000      1.9370         是
       250            2         1.0654      2.5535         否
       250            3         1.0573      2.6483         否
       250            3         1.0264      3.1687         否
       500            1         1.0000      3.1452         是
       500            1         1.0000      3.1802         是
       500            1         1.0718      3.6235         否
       500            1         1.0319      4.9631         否
       500            2         1.0620      4.5079         否
       500            2         1.0000      3.3198         是
       500            2         1.0488      3.7771         否
       500            3         1.0000      3.7855         是
       500            3         1.0000      3.0644         是
       500            3         1.0411      2.9722         否
       500            3         1.0565      5.3127         否
      1000            1         1.0000      8.8501         是
      1000            1         1.0000      6.6177         是
      1000            1         1.0000      7.1254         是
      1000            1         1.0594      6.4049         否
      1000            1         1.0000      6.1150         是
      1000            1         1.0263      6.2686         否
      1000            2         1.0000      7.4312         是
      1000            2         1.0865     12.4655         否
      1000            2         1.0196      9.0502         否
      1000            3         1.0411      6.6339         否
      1000            3         1.0588      7.5438         否

  : 不同测量次数与重复次数运行结果
:::

这里的数据中，优化比率是后选择的Adapt-Clifford能量与Adapt-Clifford能量的比率，反映了后选择是否贡献了优化以及贡献程度。后选择时间则由进行后选择后的耗时减去后选择前的得到。

[^1]: 后简称为其维度。

[^2]: $|\psi(0)\rangle$的记号后同。

[^3]: 我们考虑最大割问题的边权重$\omega_{ij}$，这与伊辛哈密顿量的转化是trivial的，所以不多作计算。

[^4]: 完整代码见附录A.1。
