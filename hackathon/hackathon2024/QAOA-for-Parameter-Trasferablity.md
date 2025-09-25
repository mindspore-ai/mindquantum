# Hackthon - QAOA for Parameter Trasferablity

[TOC]



## 1 引言
### 1.1 初参设置的重要性和参数迁移
$\qquad$初始参数设置在许多领域和应用中都至关重要，尤其是在数据分析、机器学习、工程控制、科学研究等领域。在机器学习中，初始参数可以显著影响模型的收敛速度和最终性能。一个好的初始参数设置可以帮助模型更快地收敛到全局最优解，而不是陷入局部最优。正确的初始参数可以减少计算资源的需求，加快计算过程，特别是在处理大规模数据集或复杂模型时。在一个工程中，优秀的初始参数设置可以优化资源的使用，减小时间、空间开销，从而提高效率和降低成本。寻找最优QAOA参数的一种有前途的方法在于最优参数在不同问题实例之间的可迁移性和可重用性。这个概念基于这样的观察：最优参数倾向于集中在特定的值附近，并且这些值可以根据它们的局部特征从一个问题实例转移到另一个问题实例。

$\qquad$在量子机器学习中，BP（Barren Plateau）现象会导致梯度消失，使得量子算法的优化变得困难。研究表明，采用不同的参数初始化策略可以显著影响梯度方差的衰减，从而影响量子算法的优化效率<sup><a href="#ref1">1</a></sup>。通过优化初始参数设置，可以提高量子电路的训练效率和性能。例如，通过将一组固定的电路参数调整为随机参数，可以发现一个电路具有表现力但不受贫瘠高原影响的区域，这暗示了一种初始化电路的好方法<sup><a href="#ref2">2</a></sup>。

$\qquad$基于大量MaxCut问题实验数据的观察，最优初始参数倾向于集中在特定的值附近，并且这些值可以根据它们的局部特征从一个问题实例转移到另一个问题实例，即最优初始参数在不同问题实例之间的可迁移性和可重用性。因此，我们志在寻找一个在一类近似图中均可以提高量子电路求解精度的初始参数，方便后续进行参数迁移以求解一个全新的近似图。
### 1.2 现有方法和局限
$\qquad$参数迁移是迁移学习中的一个重要概念，它涉及将从一个领域学到的知识应用到另一个领域的过程。  
$\qquad$现有的参数迁移方法中，主要有这么几种参数迁移的方法，分别是：基于特征的迁移学习方法、基于实例的迁移学习方法、基于模型的迁移学习方法、基于关系/基于对抗性的方法。这些方法都有各自的优势和局限性。  

| **方法名称**       | **简要概述**                                                 | **缺点**                                                     |
| ------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 基于特征的迁移学习 | 通过特征变换将源域和目标域映射到同一特征空间，使得模型可以在新的特征空间中学习。 | 特征变换可能难以求解，且容易发生过适配，无法让模型泛化到目标域数据。 |
| 基于实例的迁移学习 | 从实际数据中选择与目标域相似的部分数据，然后直接在这些数据上进行学习。 | 比较依赖源域和目标域之间的相似性，如果两个域的差异很大，迁移效果可能不佳。 |
| 基于模型的迁移学习 | 利用模型之间存在的相似性，迁移模型参数以适应新的任务。       | 模型参数可能不易收敛，且需要大量的计算资源。                 |

$\qquad$除了以上三种方法外，其他的一些方法也有在被使用。深度学习使用预训练的深度网络来初始化自己的网络，对网络进行微调以适应新任务，缺点在需要大量计算资源，对于小数据集可能存在过拟合风险；参数高效的迁移学习通过固定预训练模型的大部分参数，仅调整模型的一小部分参数来达到与全参数微调接近的效果，局限在于对于不同的任务需要重新调整参数迁移策略。

### 1.3 参数迁移之于QAOA的重要性
$\qquad$量子近似优化算法(Quantum Approximate Optimization Algorithm, QAOA)是一种变分算法，需要在初始输入的线路参数的基础上根据目标函数不断调整参数，直到目标函数达到极小值。因此初始参数的选取非常重要，直接影响到算法最终的收敛效果和时间开销。



## 2 相关技术

### 2.1 经典参数迁移方法
#### 2.1.1 迁移学习
1. 基于实例的迁移学习方法基于使用源数据中实例的选定部分（或全部）并应用不同的加权策略来与目标数据一起使用;
2. 基于特征的方法将源数据和目标数据中的实例（或某些特征）映射到更同质的数据中;
3. 基于模型（基于参数）的方法是将模型（网络）中获得的知识与预训练层的不同组合一起使用：冻结、微调、添加一些新层;基于关系/对抗性的方法侧重于使用在源域中学习的逻辑关系或规则或通过应用受生成对抗网络（GAN）启发的方法从源数据和目标数据中提取可转移的特征<sup><a href="#ref4">4</a></sup>。

#### 2.1.2 深度迁移学习
$\qquad$深度迁移学习是指利用从另一个任务和数据集（即使是与源任务或数据集不密切相关的任务和数据集）获得的知识来降低学习成本。深度迁移学习大体可以分为以下四种方式:

- (i)微调：针对目标数据微调预训练模型；
- (ii)冻结CNN层：冻结早期的CNN层，仅对横向全连接层进行微调；
- (iii)渐进学习：选择预训练模型的部分或全部层并冻结使用，并将一些新层添加到模型中以对目标数据进行训练；
- (iv)基于对抗性：使用对抗性或关系方法从源数据和目标数据中提取可转移特征<sup><a href="#ref4">4</a></sup>。

### 2.2 Max-Cut问题
$\qquad$ Max-Cut问题是最著名的优化问题之一，也是大多数QAOA变体的基石，是图论中的一个NP-complete问题。

> 给定一个无向图$\mathcal{G} = (\mathcal{V}、\mathcal{E})$，其中$\mathcal{V}$为顶点集合，$\mathcal{E}$为边集合，$\omega_{ij}$为连接顶点$i$和$j$的边$( i , j)\in \mathcal{E}$对应的权值。Max-Cut的目标是将图顶点$x_i$ ，$( i = 1,\dots ,\left | \mathcal{V} \right | )$划分为两个以0和1标记的互补子集，使得不同划分中连接顶点的边的加权和最大，这个加权和定义为：
> $$
>  C(x)=\sum_{i,j=1}^{\left | \mathcal{V} \right |}\omega_{ij}x_i(1-x_j)
> $$
> 其中$\omega_{ij}>0$，$\omega_{ij}=\omega_{ji}$，$\forall (i,j)\in\mathcal{E}$，且$x_i\in \{0,1\}$。

$\qquad$当图中顶点较少时，我们可以在较短时间内通过穷举法找到最大的切割边数，但当图中顶点增多时，我们很难找到有效的经典算法来解决Max-Cut问题，因为这类NP-complete问题很有可能不存在多项式时间算法。但尽管精确解不容易得到，我们却可以想办法在多项式时间内找到问题的一个近似解来代替最优解，这就是后面介绍到的近似优化算法QAOA。



### 2.3 量子近似优化算法（Quantum Approximate Optimization Algorithm）
$\qquad$ QAOA是由Farhi<sup><a href="#ref3">3</a></sup>等人首先提出的一种VQA（Variational Quantum Algorithm），它能够找到Max-Cut问题最大割的近似解，适合在NISQ（Noisy Intermidiate-Scale Quantum）设备上运行。QAOA被设计为具有重复的成本层和混合层的变分算法，被变分训练。QAOA的核心思想是将优化问题的目标函数编码到成本哈密顿量$\hat {H_c}$中，以搜索一个最优的比特串$x^*$，从而以很高的概率给出一个好的近似比$\alpha$。事实上，成本函数$C(x)$可以映射为一个成本哈密顿量$\hat {H_c}$，使得
$$
\hat {H_c}\left | x  \right \rangle = C(x)\left | x  \right \rangle  
$$
$\qquad$式中：$x$为编码比特串$x$的量子态。

> QAOA遵循以下步骤：
>
> 1.**定义成本哈密顿量$\hat {H_c}$**：使得其最高能量状态编码优化问题的解。还定义了一个与$\hat {H_c}$不交换的混频哈密顿量$\hat {H_M}$ .通常地，对于图$\mathcal{G}=(\mathcal{V},\mathcal{E})$的最大割，$\hat {H_c}$和$\hat {H_M}$如下给出：
> $$
>  \hat {H_c}=\frac{1}{2}\sum_{(i,j)\in \mathcal{E}}\omega_{ij}(I-Z_iZ_j)
> $$
>
> $$
> \hat {H_M}=\sum_{j\in\mathcal{V}}X_j
> $$
>
> $\qquad$式中：$I$为单位算子，$Z_j(X_j)$为作用在第$j$个量子比特上的泡利Z（X）算子。
>
> 
>
> 2.在状态$\left | s  \right \rangle $下对电路进行**初始化**:
> $$
> \left | s  \right \rangle =\left | +  \right \rangle ^{\otimes n}=\frac{1}{\sqrt{2^n}}\sum_{x\in\left \{ 0,1 \right \}^n}\left | x  \right \rangle,
> $$
> $\qquad$其中$n$是量子比特的个数，$n = \left |  V\right | $。态$\left | s  \right \rangle$对应于Pauli - X基的最高能态，即对应于混合哈密顿量$\hat {H_M}$的最高能态。
>
> 
>
> 3.通过定义和应用酉矩阵，构造了MaxCut的**电路拟设**：
> $$
> \begin{aligned}
> & \hat{U}_C(\gamma)=e^{-i \gamma \hat{H}_C}=\prod_{i=1, j<i}^n R_{Z_i Z_j}\left(-2 \omega_{i j} \gamma\right) \\
> & \hat{U}_M(\beta)=e^{-i \beta \hat{H}_M}=\prod_{i=1}^n R_{X_i}(-2 \beta)
> \end{aligned}
> $$
> 称为成本层(cost layers)和混合器层(mixer layers)。同样，对于一般的QUBO问题，成本层也可以定义为成本哈密顿量乘以$\gamma$参数的幂，其中现在，从$\hat H_C$的更复杂结构开始，$\hat U_C(\gamma)$也将需要更多的门实施的。单个QAOA层包含一个成本层和一个混合器层，可以进一步堆叠以构建具有更多层的更深电路。
>
> 
>
> 4.定义**QAOA层总数**，$p\ge 1$。初始化$2p$个变分参数$\gamma =(\gamma_1,\gamma_2,\dots ,\gamma_p)$和$\beta =(\beta_1,\beta_2,\dots ,\beta_p)$，其中$\gamma_k\in[0,2\pi)$,$\beta_k\in[0,\pi)$，对于$k=1,\dots,p$。因此，电路的最终状态输出由下式给出：
> $$
> \left | \psi_p(\gamma,\beta)  \right \rangle=e^{-i\beta_p \hat H_M}e^{-i\gamma_p \hat H_C}\dots e^{-i\beta_p \hat H_M}e^{-i\gamma_p \hat H_C} \left | s  \right \rangle 
> $$
> 5.哈密​​顿量$\hat H_C$关于拟设(ansatz状态)$\left | \psi_p(\gamma,\beta)  \right \rangle$的期望值，是量子算法针对目标问题的**成本函数**的值，通过对最终状态的重复测量，最小化下面这个式子：
> $$
> F_p(\gamma,\beta)=\left \langle \psi_p(\gamma,\beta)  \right | \hat H_C\left | \psi_p(\gamma,\beta)  \right \rangle
> $$
> 6.采用经典优化器迭代更新参数$\gamma$和$\beta$。上述例程的目标是找到最佳参数集$(\gamma^*,\beta^*)$，以使期望值$F_p(\gamma,\beta)$最大化：
> $$
>  (\gamma^*,\beta^*)=arg \max_{\gamma,\beta}F_p(\gamma,\beta)
> $$
> 



### 2.4 QAOA的初参设置与参数迁移 
$\qquad$ QAOA中的初参设置就是对$2p$个变分参数$\gamma =(\gamma_1,\gamma_2,\dots ,\gamma_p)$和$\beta =(\beta_1,\beta_2,\dots ,\beta_p)$进行初始化设置。研究QAOA参数迁移的学者提出了一些参数迁移方案来提升QAOA的收敛效果并减少算法的时间开销，这些参数迁移方案的具体内容将在第3部分分别介绍。

Galda等人<sup><a href="#ref7">7</a></sup>为QAOA中的参数可迁移性提供了理论基础。他们的工作表明最优QAOA参数收敛于特定值附近，并且这些参数在不同QAOA实例之间的可迁移性可以根据构成原图的子图的局部特征进行预测和描述。这种观察为识别组合优化问题的类别提供了一种方法，其中QAOA和其他VQAs可以提供显著的加速比。

$\qquad$基于这一思想，Shaydulin等人<sup><a href="#ref8">8</a></sup>提出将给定问题的最优QAOA参数作为相似问题实例的初始点。他们证明了这样做不仅可以通过避免局部最优来提高解的质量，而且可以减少达到该解所需的评估次数。

​	在此基础上，Shaydulin等人<sup><a href="#ref9">9</a></sup>利用最优参数传递在相似图上的性质，提出了QAOAKit，它是一个Python框架，包含一组用于QAOA的预优化参数和电路模板。给定一个输入图，通过一个图同构器获得输入图的准最优参数，然后将其作为密钥从QAOAKit的数据库中提取角度。如果对于特定的图实例，数据库中不存在最优角度，系统将提供与该特定图最相近的固定角度。



## 3 基准方法
$\qquad$ QAOA最优参数在不同问题实例之间有可迁移性和可重用性。这个现象基于这样的观察：最优参数倾向于集中在特定的值附近，并且这些值可以根据它们的局部特征从一个问题实例转移到另一个问题实例。

###  JP Morgan方案
$\qquad$不同于经典计算机中系统的参数迁移学习方法，QAOA由于提出至今也只过了十余年，其中大部分高效的参数迁移方案还是**基于数学直觉与实验验证**得到的。下面本文将对其中一种目前比较高效的QAOA参数迁移方案进行一个介绍，这个参数迁移方案是JP Morgan所做的工作<sup><a href="#ref5">5</a></sup></sup><sup><a href="#ref6">6</a></sup>。
####  无三角形图参数迁移
$\qquad$无三角形图就是图中不存在包含三个顶点或三个顶点以上的超边。考虑**p = 1的无三角形加权图**的简单情况，其中QAOA优化目标可以表示为：
$$
\left \langle C(\beta_1,\gamma_1) \right \rangle=\frac{W}{2}+\frac{\sin{4\beta_1}}{4}\sum_{(i,j)\in E}\omega_{i,j}\sin(\omega_{i,j}\gamma_1)(\prod_{l\in \mathcal{N}_i\setminus \{j\}}\cos(\omega_{i,l}\gamma_1)+\prod_{k\in \mathcal{N}_j\setminus \{i\}}\cos(\omega_{j,k}\gamma_1))
$$
$\qquad$其中$W=\sum_{(i,j)\in E}\omega_{i,j} $且$\mathcal{N}_i\setminus \{j\} $是顶点 $i $不包括顶点 $j $的邻域。
$\qquad$Ruslan Shaydulin等人的工作<sup><a href="#ref5">5</a></sup>提出当上式中八项相位近似对齐时，QAOA能产生较高的近似比，相位对齐时$\gamma$接近于0。

#### 通用图参数迁移
$\qquad$更普遍的情况下，第 $l$ 步可以用参数和权重的三角函数来表示：
$$
 e^{-i\beta_iB}=\prod_{j=1}^n[\cos \beta_l-ix_j\sin \beta_l],\\
 e^{-i\gamma_lC}=\prod_{(i,j)\in E}[\cos\frac{\gamma_l\omega{i,j}}{2}+iz_iz_j\sin\frac{\gamma_l\omega{i,j}}{2}] 
$$
$\qquad$和无三角图一样，在相位近似对齐时，QAOA能产生较高的近似比。具体来说，从实验中可以观察到 $\gamma $的最佳值随着边缘权重的平均绝对值的变化而变化，即$ \gamma \approx  O(1/\bar{\left | \omega \right | })$，与深度 $p $无关。  
$\qquad$考虑将 QAOA 应用于两个通用目标函数 $C_w$ 和 $C$，它们在某个因子 $w > 0$ 上等效，换句话说，$C_\omega = \omega C$。如果 QAOA 使用参数 $(\beta, \gamma)$ 实现 $C$ 对 $r$ 的近似比，则 QAOA 将使用 $(\beta, \gamma/\omega)$ 对 $C_\omega$ 实现相同的近似比。

  

####  参数传递的缩放规则
$\qquad$根据一些数学经验，通过 $\arctan( \sqrt{d-1} )$ 缩放 $\gamma $，因为在$p$值较低时，提供了更好的性能。  
$\qquad$最后，为了确定新实例的转移角度，预先计算的中值参数被重新调整为：
$$
\begin{aligned}
\beta_\omega & =\beta_{\text {median }}^S \\
\gamma_\omega & =\gamma_{\text {median }}^S \frac{\arctan \left(\frac{1}{\sqrt{d_\omega-1}}\right)}{|\bar{\omega}|}
\end{aligned}
$$
$\qquad$这里的$\beta_{median}^S$和$\gamma_{median}^S$均来自于QAOAkit<sup><a href="#ref9">9</a></sup>。

$\qquad$利用重缩放因子缩放带权重图的权重:
$$
\omega_{uv}\to \frac{\omega_{uv}}{\sqrt{\frac{1}{| E |}\sum_{\{u,v \}\in E}\omega_{uv}^2}}
$$


## 4 赛题说明与方案概览
### 4.1 赛题说明+方案概览
> 对于一个包含$N$个自旋变量$\{ z_i | z_i\in \{+1,-1\},i \in \{0,1,2,\dots ,N-1 \}  \}$的一般情况高阶Ising模型可表示为：
> $$
>  H_C=\sum_i J_i^{(1)}z_i+\sum_{i,j} J_{ij}^{(2)}z_iz_j+\sum_{i,j,k} J_{ijk}^{(3)}z_iz_jz_k+\dots
> $$
> $\qquad$其中每一项都是多个自旋变量的乘积，从图的角度可以看作是自选变量所组成的超边，每项的系数$J_{ijk\dots}^{(m)}$则看作是这条超边的权重。
>
> $\qquad$这样的高阶ising模型所对应的标准$p$层QAOA线路如下，含有$\gamma=(\gamma_1,\dots ,\gamma_p),\beta=(\beta_1,\dots ,\beta_p)$共$2p$个参数，对线路末态$|\Psi\rangle$测量ising模型对应的哈密顿量均值$C(\gamma,\beta)=\left \langle \Psi | H_C |\Psi \right \rangle$作为目标函数。
>
> $\qquad$请给出$(\gamma,\beta)$的初参设置方案，对于各种不同的Ising模型的输入，初始QAOA线路所产生量子态的目标函数尽可能小。

> 样例代码里面主要的方法：
>
> 1．提前准备好案例无关的**无穷顶点图的最优参数 $\gamma^{\infty}, \beta^{\infty}$**
>
> 2．对输入的ising模型计算缩放系数 $\alpha$ ：
> $$
> \alpha=\sqrt{\frac{1}{\left|E_k\right|} \sum_{\left\{u_1, \cdots, u_k\right\}}\left(J_{u_1, \cdots, u_k}^{(k)}\right)^2+\cdots+\frac{1}{\left|E_1\right|} \sum_{\{u\}}\left(J_u^{(1)}\right)^2}
> $$
>
> 4．计算图的平均度（degree）$D$ ，给出最后的 $\gamma, \beta$ ：
> $$
> \begin{gathered}
> \gamma=\alpha \times \gamma^{\infty} \times \arctan \frac{1}{\sqrt{D-1}} \\
> \beta=\beta^{\infty}
> \end{gathered}
> $$
> 

方案概览

| 团队           | ising模型计算缩放系数 $\alpha$                               | 图的平均度（degree）D相关 $\arctan \frac{1}{\sqrt{D-1}}$     | $\gamma^{\infty}$和$\beta^{\infty}$                          | 其他                                                         |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| New            | 优化因子$o_0=\frac{\sum_{i=1}^k \operatorname{dist}_M\left(g_0, g_i\right) \cdot o_i}{\sum_{i=1}^k \operatorname{dist}_M\left(g_0, g_i\right)}$ |                                                              |                                                              | （1）云端加速器架构（2）精确匹配 +近似图匹配                 |
| 深瞳           | $\tilde{\alpha}_D=\frac{1}{\sqrt{d}-0.1 \cdot e}$ 其中，$ d $为度数，$ e $为基于图结构（如三角形数量）的动态调整因子 | $\arctan \frac{1}{\sqrt{D}} \quad$ or $\quad \frac{1}{\sqrt{D}}$ | $\gamma$ 优化条件推导 $\cot ^2 \gamma-d=...$                 |                                                              |
| 宇宙探索编辑部 | ![image-20250225144658515](https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250225144658515.png) |                                                              | 通过分析无穷顶点图的最优参数规律，结合平均度数与权重统计量，得到$\gamma^g$和$\beta^g$ | 利用跳层神经网络，对初始参数进行修正                         |
| biubiu         |                                                              |                                                              | 超图神经网络回归模型，接预测QAOA参数$\gamma, \beta$,         | 引入截断奇异值分解（TSVD）降维，降低计算复杂度               |
| Qucius         | 重缩放因子rescaler $\gamma^*=\operatorname{rescaler} \cdot \frac{\gamma_{\mathrm{inf}}}{\sqrt{\frac{1}{|E|} \sum w_{u v}^2}} \cdot \arctan \left(\frac{1}{\sqrt{D-1}}\right)$ |                                                              | 同时优化 $\gamma$ 和 $\beta$ 参数（传统方法仅调整 $\gamma$ ） | 线性插值外推：将已知层数（如p=4,8）的最优参数通过线性拉伸适配到任意层数 |
| 1+1            | 不再区别不同的阶 $q=k$，任意阶缩放因子 。$\alpha=\sqrt{\frac{1}{\left|E_{\text {all }}\right|} \sum_{\{u\}}\left(J_u^{(a l l)}\right)^2}$ | 针对不同阶数k的Ising项，提出动态度数修正公式 $D_{\text {new }}=\frac{\sum_1^n N_k 2^{k-1} / k}{N_q}$ |                                                              | 根据Ising模型的权值方差阈值：分离**无权图**（低方差）与**带权图**（高方差）参数库 |
|                |                                                              |                                                              |                                                              |                                                              |





### 4.2 队伍New程序报告

参数生成模块总体流程图（包含并行过程）图片引用来自` New队伍 `

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250225095400482.png" style="zoom:50%;" />

我们将简要介绍流程图中的4个模块

1. 精确匹配
2. 基于参数的近似图匹配
3. 基于因子的近似图匹配
4. 公式生成算法



#### **（1） 精确匹配**

$\qquad$精确匹配是最直接的算法，它将输入图与预计算的图库进行直接对比。若图库中存在完全一致的图，则输出对应参数。此方法优点在于简单高效、结果精准。

- 其**缺点**也十分明显：匹配概率低，泛化性弱，性能高度依赖图库规模。

#### **（2） 基于参数的近似图匹配**

基于参数的近似图匹配算法的核心思想是：相似的图具有相似的参数。该算法巧妙地结合并应用多种机器学习技术，算法复杂度主要体现在度量学习训练部分。

- 当图的样本数量为$n$，图的属性数为$m$，则算法复杂度为$O(nm^2C_n^2)$。
- 当$n$和$m$较小时可以直接求出$M$。
- 当$n $和$m $较大时可以利用蒙特卡洛等方法近似求得$M$。

$\qquad$已计算参数图库规模越大对算法性能提升越明显（近似图越靠近输入图，参数预测越准确）

**数据溯源**

- 算法首先提取图结构的基础属性（阶数、节点/边数）并推断边权分布。
- 通过极大似然估计确定边权分布：在候选分布集合F中寻找最大化似然函数$\prod_{i=1}^{|W|} P\left(w_i \mid f\right)$的分布。
- 采用对数似然$\sum_{i=1}^{|W|} \log P\left(w_i \mid f\right)$避免数值溢出，同时基于独立同分布假设推导生成边概率。
- 最终输出包含边权分布、生成概率等特征向量，供后续分析使用。

**距离度量（度量学习）**

- 图的属性（阶数、边权分布、节点数等）构成图空间G的坐标系统。欧氏距离可初步度量图间差异：$dist_2(g_i,g_j)=\sqrt{(g_i-g_j)^T(g_i-g_j)}$

- 此方法存在两缺陷：属性权重不均（如边数比阶数重要）、属性间存在相关性（如边数与生成概率相关）。故引入马氏距离：$dist_M(g_i,g_j)=\sqrt{(g_i-g_j)^TM(g_i-g_j)}$

- 矩阵M通过优化确定：$\mathop{\arg\min}\limits_{M}\sum_{g_i,g_j}\frac{1}{2}(dist_M-dist_{true})^2$ 其中真实距离由参数迁移得分差定义：$dist_{true}=|s_i-s_j^i|+|s_j-s_i^j|$（$s_i$为图g_i最优参数得分，$s_j^i$为$g_i$参数在$g_j$的得分）

**近似图匹配（最近邻算法）**

- 邻居选择：在参数库中检索输入图$g_0$的k个最近邻图$\{g_1,...,g_k\}$，计算$dist_M(g_0,g_i)$
- 参数融合：通过距离倒数加权平均计算输入图参数：$p_0 = \frac{\sum_{i=1}^k p_i/dist_M(g_0,g_i)}{\sum_{i=1}^k 1/dist_M(g_0,g_i)}$
  

#### **（3） 基于因子的近似图匹配**

$\qquad$​核心公式改进：

$$
\gamma = \alpha \cdot \gamma^\infty \cdot \arctan(1/\sqrt{D-1}), \rightarrow \gamma = \alpha \cdot \gamma^\infty \cdot o_0
$$
其中加权平均计算优化因子
$$
o_0 = \frac{\sum_{i=1}^k dist_M(g_0,g_i) \cdot o_i}{\sum_{i=1}^k dist_M(g_0,g_i)}
$$
其中

- 使用COBYLA/BFGS优化典型图的缩放因子$o$
- 通过马氏距离检索其k近邻图$\{g_1,...,g_k\}$

优势对比：

- 精度：基于参数的方法（直接迁移参数）精度更高，但依赖密集图库；
- 泛化性：基于因子的方法通过公式计算，在稀疏图库区域表现更优。



####    **（4） 公式生成算法**

$\qquad$该算法是通过实验和经验性的方法对文献<sup><a href="#ref10">10</a></sup> 算法的部分系数值作了一些调整，然后直接利用公式生成参数。该方法输出参数精度相较前三种方法比较一遍，但适用范围最广，主要用于前面三种算法都不太奏效时的兜底。

####  小结
$\qquad$总的来说，四种所提子算法各有优缺点和适用场景。最终参数生成模块设置了性能优选过程进行参数对比。针对具体的输入图数据和应用场景，性能优选过程分别评估四种子算法输出的参数得分，找到最高质量的参数进行输出。



### 4.3 深瞳程序报告



1. 第1个改进：通过数学推导，图中无三角形时 $\gamma=\arctan \frac{1}{\sqrt{d}}$
2. 第2个改进：虑图中三角形的影响，修正后的缩放因子：$\tilde{\alpha}_D=\frac{1}{\sqrt{d}-0.1 \cdot e}$
   - 要调节的参数 $e$ 为正整数。赛题所给的伊辛模型数据集，取 $e=10$ 即可获得一个不错的提升。
3. 将伊辛模型数据集做一些划分，针对不同的情况设置不同的参数。根据$\left\lfloor\frac{4 m}{n}-n\right\rfloor>0$是否成立，将伊辛模型数据集分为两个不同的类别。（$|V|=n,|E|=m$）
4. 参数 $e$ 随着层数 $p$ 的增大而增大
5. 不同阶的伊辛模型分为不同类别，寻找不同的 $e$ 值以达到更好的效果



####  小结

$\qquad$该代码通过以上一系列修正，从最后得分来看，较大地提高了该题目解的准确性；从运行时间上来看，与样例代码相近，并没有增大运行时间的开销。

$\qquad$该程序主要对平均度缩放因子$\alpha_D$进行了修改，修正过程中用到的常数是根据图超边的最大阶数和深度p求得的。



### 4.4 宇宙探索编辑部程序报告

#### (1) **基于图论的无标度参数缩放规则**
针对Ising模型的超图特性，提出"85%权重分位数比例缩放"规则：
$$
D_{rescale} = ({\frac{2}{\left| E_1 \right |}\sum_{u_1}J_{u_1}^{(1)}}+\dots + {\frac{2}{\left | E_k \right |}\sum_{u_1\dots u_k}J_{u_1\dots u_k}^{(k)}})/J^{85\%}
$$
其中$ J_{85\%} $为权重分布的85%分位数，有效平衡不同规模问题的参数范围。

原本的系数为
$$
\bar{J}=\sqrt{\frac{1}{\left|E_k\right|} \sum_{\left\{u_1, \cdots, u_k\right\}}\left(J_{u_1, \cdots, u_k}^{(k)}\right)^2+\cdots+\frac{1}{\left|E_1\right|} \sum_{\{u\}}\left(J_u^{(1)}\right)^2}
$$



#### (2)  **无穷顶点图参数迁移框架**

通过分析无穷顶点图的最优参数规律（如 $\varphi^{\infty}$ 和 $\beta \infty$ ），结合平均度数与权重统计量，构建参数迁移公式：

$$
\gamma_g=\bar{J} \cdot \gamma_{\infty} \cdot \arctan \left(\frac{1}{\sqrt{D_{\text {rescale }}-1}}\right)
$$
实现从理论模型到实际问题的参数泛化。



#### (3)  结合ML的 量子参数初始化（StatQInit） 

- 提出结合经典统计特征与量子优化的混合方法，通过提取 1 sing模型的超图统计信息（节点聚类系数，边权重分布，节点度数分布等），构建多维统计表征向量。
- 利用跳层神经网络（Skip－Connection Neural Network）对初始参数进行修正，通过L1损失函数优化参数偏差 $(\Delta \gamma, \Delta \beta)$ ，显著提升参数初始猜测的准确性。



#### 数值实验：

在7,853个Ising实例对神经网络进行训练

测试了

- 不同分布（了指数分布、均一分布、均一偏分布、二项分布、同一分布、伽马分布、泊松分布）
- 不同QAOA深度 $p=2,3,4,5$,

1,008个测试案例中，StatQInit相比传统启发式缩放方法等效期望值指标上有更好的表现





### 4.5 biubiu程序报告

#### (1) **超图神经网络（HGNN）驱动的参数预测模型**

- 提出基于超图结构的深度神经网络，通过构建超图关联矩阵 $H \in \mathbb{R}^{n \times m}$ 捕捉高阶相互作用（ $k \geq 2$ 的 1 sing项），突破传统GNN仅处理二元边的限制。
- 设计两层级联的HyperGCN卷积层，融合节点度数矩阵 $D_v$ 和超边权重矩阵 $D_e$ ，实现高阶统计特征的自动提取：

$$
h^{(l+1)}=D_v^{-1 / 2} H D_e^{-1} H^T D_v^{-1 / 2} h^{(l)} W^{(l)}
$$

其中，$h^{(l)}$ 是第$l$层的节点表示，$W $是线性变换矩阵，$D_v$ 和$D_e $分别是节点和超边的度数矩阵，$H$ 是关联矩阵。

输出层通过线性回归直接预测QAOA参数 $\gamma, \beta$ ，支持端到端训练。

#### (2) **TSVD+kNN混合迁移学习框架**

- 引入截断奇异值分解（TSVD）对超图关联矩阵 $H$ 降维，提取主成分特征向量 $\Sigma_w \in \mathbb{R}^w$ 。
- k 近邻（k－Nearest Neighbors，kNN）的量子参数回归模型，通过加权融合历史最优参数（权重系数 $0.7: 0.3$ ），实现跨问题规模的参数泛化。



##### 4.5.1.1 超图神经网络介绍

$\qquad$超图可以看作是一般图的推广。一般图的边是顶点的成对2元组合。超图的边也被称为超边，是顶点的无序k 元组合。确切的说，超图是有序对$(V,E)$，其中$V $是一组顶点/节点，$E $是一组$V $的无序$k$ 元组合（其中，$k \in \{1, 2, ..., |V |\}$）。

$\qquad$对于一个有n个顶点和m个超边的超图，$H\in \mathbb{R}^{n\times m}$称为关联矩阵。它的定义如下：

$$h_{i,j}=\left\{\begin{matrix}
1\quad if \ v_i\in e_j  
\\
0\quad else
\end{matrix}\right. $$

$\qquad$​在超图上进行神经网络的构建和学习涉及多个步骤。首先，我们需要构建每个超边的表示。通常，通过对所有包含在超边中的节点表示进行求和或平均来实现。然后，对于每个节点，我们通常会对包含该节点的所有超边的表示进行求和或平均，并将其与原始节点表示结合。

数学上，这个过程可以表示为：

$$h_i^{(l+1)}=\sigma(W_1^{(l)}h_i^{(l)}+W_2^{(l)}\sum_{j\in\mathcal{N}_i}\frac{h_j^{(l)}}{|\mathcal{N}_i|}) $$​

$\qquad$其中，$h^{(l)}_i $表示第l 层中节点i 的表示，$W^{(l)}_1$ 和$W^{(l)}_2$ 是第$l $层的可学习线性变换矩阵，$\sigma$是激活函数，$\mathcal{N}_i$ 是包含节点$i $的所有超边的集合，$|N_i|$ 是包含节点$i$​的超边的数量。

$\qquad$具体计算步骤如下：
1. 计算节点和超边的度数：节点的度数矩阵$D_v $是关联矩阵的列和的对角矩阵，表示
    每个节点连接的超边数量。超边的度数矩阵$D_e$ 是关联矩阵的行和的对角矩阵，表
    示每个超边包含的节点数量。

2. 归一化关联矩阵：使用节点和超边的度数矩阵，对关联矩阵$H $进行归一化，得到一
    个归一化的关联矩阵。具体操作是先计算$D_v^{-\frac{1}{2}}HD_e^{-1} $，然后计算$H^T $的归一化。

3. 线性变换：将归一化后的节点特征矩阵乘以一个可学习的线性投影矩阵$W$​，得到新
    的节点表示。

  上述过程可以公式表示为：

  $$h^{l+1}=D_v^{-\frac{1}{2}}HD_e^{-1}D_v^{-\frac{1}{2}}h^{(l)}W $$

  其中，$h^{(l)}$ 是第$l$层的节点表示，$W $是线性变换矩阵，$D_v$ 和$D_e $分别是节点和超边的度数矩阵，$H$ 是关联矩阵。

##### 4.5.1.2 基于超图神经网络的量子线路参数回归模型
$\qquad$基于超图神经网络的量子线路回归模型如下所示，

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250210140749683.png" style="zoom:50%;" />

$\qquad$该模型的具体结构如下:

```python
class HGNN(nn.Module):
	def __init__(
		self, in_channels, hid_channels, out_channels, use_bn = False, drop_rate = 0.5):
		super().__init__()
		self.layers = nn.ModuleList()
		self.layers.append(
		HyperGCNConv(
		in_channels, hid_channels, False, use_bn=use_bn, drop_rate=drop_rate ))
		self.layers.append(
		HyperGCNConv(
		hid_channels, hid_channels, False, use_bn=use_bn, is_last=True ))
		self.regression_layer1 = nn.Linear(hid_channels, hid_channels)
		self.regression_layer = nn.Linear(hid_channels, out_channels)
    def forward(self, X: torch.Tensor, hg: Hypergraph) -> torch.Tensor:
		for layer in self.layers:
    		X = layer(X, hg)
		X = X.mean(dim=0, keepdim=True)
		X = self.regression_layer1(X)
		prediction = self.regression_layer(X)
		return prediction
    
p = 4 # 对应量子线路层数
model_p4 = HGNN(in_channels=12, hid_channels=10, out_channels=p*2, use_bn=False,
drop_rate=0.1)
```

$\qquad$在该模型中，我们将节点特征定义为节点编号经过one-hot 编码成的向量$x\in R^N$,
其中$N=12 $​表示图中节点个数。

```python
def to_one_hot(indices, num_classes):
	indices = indices.view(-1, 1) # 将索引展开为二维
	one_hot = torch.zeros(indices.size(0), num_classes) # 初始化one-hot编码矩阵
	one_hot.scatter_(1, indices, 1) # 将索引位置设置为1
	return one_hot

indices = torch.tensor([_ for _ in range(12)], dtype=torch.long) # 整数索引
X = to_one_hot(indices, num_classes)
```

$\qquad$输入超图神经网络的超边和权重为数据集中定义的边和权重，将其组合起来得到一个超图输入$HG$。

```python
from dhg import Hypergraph
num_nodes = 12
HG = Hypergraph(num_nodes, e_list = data['J'], e_weight = data['c'])
```

$\qquad$接着，将节点特征$X $和超图实例$HG$​ 一起输入到HGNN 中进行训练，训练采用SGD 优化器和MAE 损失函数。

```python
def train(net, X, A, lbls, optimizer, epoch):
	net.train()
	optimizer.zero_grad()
	outs = net(X, A)
	outs = outs.type(torch.float64)
	lbls = lbls.type(torch.float64)
	loss = F.l1_loss(outs, lbls)
	loss.backward()
	optimizer.step()
	return loss.item()

optimizer = optim.SGD(model_p4.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
loss += train(model_p4, X, G, lable, optimizer, epoch)
```

$\qquad$训练好的模型可以在输入任意大小的超图进行预测，从而得到相应的量子线路参数
预测结果。

```python
@torch.no_grad()
def infer(net, X, A):
	net.eval()
	outs = net(X, A)
	return outs

data = read_json(file_path) # 测试数据
el = [list(t) for t in data['J']]
ew = [float(t) for t in data['c']]
hg = Hypergraph(12, e_list = el, e_weight = ew)
X = to_one_hot(indices, num_classes)
pred_param = infer(model_p4, X, G)
```



#### 4.5.2 基于kNN 的量子线路参数回归模型
$\qquad$由于超图可以表示为关联矩阵，为了实现不同阶数问题之间的学习，还可以采用TSVD+kNN 的方法进行学习。首先，对于关联矩阵$H \in R^{n\times m}$(含权重)，我们采用TSVD的方法将其分解为三个矩阵:

$$H = U\Sigma V^T$$​

其中，$U \in R^{n\times n},V \in R^{m\times m},\Sigma \in R^{n\times m}$，$\Sigma$对角线上的值即为特征值，我们选取其中前$w $个特征值并组成一维向量$\lambda \in R^w$。$\lambda$​ 在一定程度上反映了原始超图的信息。进一步地，我们可以将数据集中的所有超图都转换为这种向量表示，每个向量对应的标签（即，最优或近似最优的量子线路参数）通过2.3 节的QAOA 算法得到，向量及其标签组成了kNN 的查询数据库。



$\qquad$当我们需要进行测试时，将新样本按照同样的方法转换为向量，并在查询数据库中进行检索，并选取前k 个标签的均值作为新样本的量子线路初始值。


#### 4.5.3 小结
$\qquad$选手提出了一种基于超图神经网络的量子线路参数回归模型，以解决量子线路参数初始化的问题。由于数据集和时间的限制，我们并没有完全发挥该方法的全部潜力。因此，可以进一步结合小样本学习与超图神经网络，从而实现超图神经网络更好的性能。此外，超图神经网络依然为监督学习方法，这种方法只能达到其数据的上限，而我们无法得知QAOA 训练得到的量子线路参数是否是最优，导致超图神经网络学习的标签可能是有偏的。因此，可以进一步将图神经网络结合强化学习进行量子线路参数的寻找，利用超图神经网络得到超图的特征表示，强化学习进行后续的线路参数优化；也可以采用对比学习等策略，识别拉开劣解和好解的距离，让超图神经网络学到更好的规律。



### 4.6 Qucius程序报告
#### (1) $\gamma$重缩放因子修正
 在现有参数迁移公式基础上引入全局乘性常量因子（公式3），动态调控 $\gamma$ 参数的缩放幅度：

$$
\gamma^*=\operatorname{rescaler} \cdot \frac{\gamma_{\mathrm{inf}}}{\sqrt{\frac{1}{|E|} \sum w_{u v}^2}} \cdot \arctan \left(\frac{1}{\sqrt{D-1}}\right)
$$

通过实验发现调整 rescaler＝1．275 可使初始参数相比baseline提升 $8.87 \%$（表1方法 ＃3 vs ＃2）。





4.6.2 多策略微调预制表

$\qquad$采用**微调训练**的方式进一步优化预制表中的参数$\gamma^{inf}$ 和$\beta^{inf}$，即使用各具体样本的最优参数来调整全局平均最优参数。最朴素的更新策略如下式，

$$
\theta_{i+1}^*=(1-\Delta x)*\theta_i^*+\Delta x*\hat \theta_i
$$
其中$\theta_i^*$为第$i $次训练迭代后的全局平均最优参数，而$\hat\theta_i$ 为第$i $次训练时所用样本的最优参数，$\Delta x$ 可理解为学习率。

$\qquad$为了获得更好的训练效果，我们进一步引入了学习率衰减、自适应损失、差分动量的更新策略。

$\qquad$学习率衰减：保证学习呈现收敛趋势
$$ \Delta E=E_{before}-E_{after}$$
$\qquad$自适应损失：动态样例权重
$$ \Delta x=\Delta x*(\omega^{\frac{cur\_ iter}{decay\_ per\_ iter}})*log(1+\Delta E)$$
$\qquad$动量：变相加大epoch
$$ p_{i+1}^*=(1-m)*\mu_i^*+m*\hat \theta_i$$
$\qquad$均权-加权分离：观察到uniform情况下的衰退
$$ \theta_{i+1}^*=(1-\bar{\Delta x})*\theta_i^*+\bar{\Delta x}*\mu_{i+1}^*$$



#### 4.6.3 小结

$\qquad$针对QAOA 线路最优初参选取问题，该选手提出了一种基于预制表进行多策略微调的方法，进一步降低了线路初始损失值，为后续迭代优化创造有利起点。提出了一种线性插值外推的方式，可在保持参数数值比例一致的条件下进行内外插值，以扩展训练优化后预制表的应用范围。



### 4.7 1+1程序报告
#### 4.7.1 缩放系数修正
##### 4.7.1.1 缩放因子修正

 参考文献在 $\mathrm{q}=2$ 阶时的缩放系数公式，并将该缩放系数拓展到了任意 $\mathrm{q}=\mathrm{k}$ 阶：

$$
\alpha=\sqrt{\frac{1}{\left|E_k\right|} \sum_{\left\{u_1, \ldots, u_k\right\}}\left(J_{u_1, \ldots, u_k}^{(k)}\right)^2+\ldots+\frac{1}{\left|E_1\right|} \sum_{\{u\}}\left(J_u^{(1)}\right)^2}
$$


但这种拓展到了任意 $\mathrm{q}=\mathrm{k}$ 阶的缩放系数公式实际上是错误的。这里`1+1`团队给出了修正方案：首先，$\alpha$ 修改为以下式子，不再区别不同的阶 $q=k$ 。

$$
\alpha=\sqrt{\frac{1}{\left|E_{\text {all }}\right|} \sum_{\{u\}}\left(J_u^{(a l l)}\right)^2}
$$


##### 4.7.1.2 平均度修正

$$
D = \frac{2N}{N_q}\to D_{new} = \frac{\sum_{1}^{n}N_k2^{k-1}/k }{N_q}
$$

$\qquad$对于无环图，该修正计算结果极其接近网格搜索最优结果



#### 4.7.2 权重分布与非无环正规图修正
$\qquad$对于高阶Ising 模型对应的标准P 层QAOA 线路的最优初始参数，当顶点数固定，层数p 固定，阶数q 固定，且超边项数固定时，最优初始参数只与数据权重分布的均方根和均值的比值有关，而与权重分布的其他具体细节无关。  

$\qquad$根据超边权重分布优化上面得到一个权重系数，修正之前得到的缩放因子。



#### 4.7.3 无穷顶点正规图修正
$\qquad$参数和被定义为在无穷顶点图的最优参数，它们是在以下条件下成立的:

$\bullet \quad$顶点数量趋向于无穷大。

$\bullet \quad$图的平均度D 也趋向于无穷大。

$\bullet \quad$数据图为无环正规图。

$\qquad$方案的数据集并不满足这些条件，它们具有固定的顶点数量（12 个），并且平均度$D$ 有限。需要对和的值进行修正，以适应我们特定的数据集特性。  

$\qquad$因此需要修正原题给出的在无穷顶点得到的$\gamma^{\infty}$和$\beta^{\infty}$为$\gamma^x$和$\beta^x$

#### 4.7.4 平均度D相关函数修正
$\qquad$当p=4,8时，难以获得简洁的数学解，可通过模拟数据，拟合一个合适的修正函数。
$$ \arctan{\frac{1}{\sqrt{D-1}}} \to f(D,p,k)=g(D,p,k)*\arctan{\frac{1}{\sqrt{D-1}}}$$



#### 4.7.5 小结

$\qquad$选手对缩放系数进行了准确的修正，还对平均度$D$ 进行了恰当的等价换算，从而在数据集为无环正规图的情况下很好地解决了问题。即使在数据集为非无环正规图的情况下，我们也提出了一套有效的解决方案来寻找最优的初始参数。  
$\qquad$针对不同权重分布的情况，我们得出了一个重要结论：对于高阶Ising 模型对应的标准$P$层QAOA 线路的最优初始参数，当顶点数、层数$p$、阶数$q$ 以及超边项数固定时，最优初始参数仅与数据权重分布的均方根和均值的比值有关，而与权重分布的其他具体细节无关。这一发现对于未来在该领域的研究具有重要的启示作用。









## 5 实验结果与结论

### 5.1 各参赛队伍模型在大赛数据集上的得分

![](score.png)

得分计算规则如下：

1. **总评分** (`score` 函数)：
   - 初始化总分 `score` 为0。
   - 记录开始时间。
   - 遍历隐藏数据文件夹中的所有文件，加载数据并调用 `single_score` 函数计算分数。
   - 遍历不同参数设置的数据文件，加载数据并调用 `single_score` 函数计算分数。
   - 记录结束时间并打印总分。
2. **单个模型评分** (`single_score` 函数)：
   - 使用 `build_ham_high` 函数构建哈密顿量。
   - 对于每个深度（4和8），运行QAOA算法，生成量子电路。
   - 使用MindQuantum的模拟器计算哈密顿量的期望值。
   - 将期望值的负值累加到总分 `s` 中。

`single_score` 函数的详细公式
$$
\text { single score }\left(J c \_d i c t\right)=\sum_{\text {depth } \in\{4,8\}}-E(\text { depth })
$$
其中，$E(depth)$ 是在给定深度下，通过QAOA算法生成的量子电路的哈密顿量期望值。

更详细计分规则，请见附录“计分规则的代码”

### 5.2 结论

$\qquad$各位选手对于本次大赛赛题展现出了极高的热情，各自展现自己的聪明才智，查阅文献，做出自己独特的创新。纵观各位选手提交的代码，近似图匹配、神经网络学习、基于权重分布的初参设置、调整重缩放因子和平均度等方法被用在了提交的代码中，极大地提高了该题目解的准确性。在今后的工作中，以上提及的各种高效的初参设置方案相信会被更多地运用，推动这个领域的发展。  
$\qquad$相信各位选手通过对本次题目的解答会对QAOA初参设置有一些更加深入的思考，在后续工作中能够提出更为高效、泛用性更强的QAOA参数迁移方案。那么，本次大赛也就达到了一个“他山之石，可以攻玉”的目的了。



## 参考文献

1. <a name = "ref1" href="https://arxiv.org/abs/2311.13218">Muhammad Kashif, Muhammad Rashid, Saif Al-Kuwari, Muhammad Shafique. Alleviating Barren Plateaus in Parameterized Quantum Machine Learning Circuits: Investigating Advanced Parameter Initialization Strategies</a>
2. <a name = "ref2" href="https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.2.040309">Tobias Haug, Kishor Bharti, and M.S. Kim. Capacity and Quantum Geometry of Parametrized Quantum Circuits</a>
3. <a name = "ref3" href="https://arxiv.org/abs/1411.4028">Edward Farhi, Jeffrey Goldstone, and Sam Gutmann. A Quantum Approximate Optimization Algorithm</a>
4. <a name = "ref4" href="https://doi.org/10.3390/technologies11020040">Iman, M.; Arabnia, H.R.; Rasheed, K. A Review of Deep Transfer Learning and Recent Advancements. Technologies 2023, 11, 40. https://doi.org/10.3390/technologies11020040</a>
5. <a name = "ref5" href="https://arxiv.org/abs/2201.11785">Ruslan Shaydulin, Phillip C. Lotshaw, Jeffrey Larson, James Ostrowski, Travis S. Humble. Parameter Transfer for Quantum Approximate Optimization of Weighted MaxCut</a>
6. <a name = "ref6" href="http://arxiv.org/abs/2305.15201">Shree Hari Sureshbabu, Dylan Herman, Ruslan Shaydulin, Joao Basso, Shouvanik Chakrabarti, Yue Sun, Marco Pistoia. Parameter Setting in Quantum Approximate Optimization of Weighted Problems</a>
7. <a name = "ref7" href="https://ieeexplore.ieee.org/document/9605328">Alexey Galda, Xiaoyuan Liu, Danylo Lykov, Yuri Alexeev, Ilya Safro, Transferability of optimal QAOA parameters between random graphs, 2021</a>
8. <a name = "ref8" href="http://dx.doi.org/10.1109/HPEC.2019.8916288">Ruslan Shaydulin, Ilya Safro, Jeffrey Larson, Multistart methods for quantum approximate optimization, in: 2019 IEEE High Performance Extreme Computing Conference, HPEC, IEEE, Waltham, MA, USA, ISBN: 978-1-72815-020-8, 2019, pp. 1–8</a>
9. <a name = "ref9" href="http://dx.doi.org/ 10.1109/QCS54837.2021.00011">Ruslan Shaydulin, Kunal Marwaha, Jonathan Wurtz, Phillip C. Lotshaw, QAOAKit: A toolkit for reproducible study, application, and verification of the QAOA, in: 2021 IEEE/ACM Second International Workshop on Quantum Computing Software, QCS, 2021, pp. 64–71</a>
10. <a name = "ref10" href="[[2305.15201\] Parameter Setting in Quantum Approximate Optimization of Weighted Problems](https://arxiv.org/abs/2305.15201)">SURESHBABU S H, HERMAN D, SHAYDULIN R, et al. Parameter setting in quantum
    approximate optimization of weighted problems[J/OL]. Quantum, 2024, 8:1231. http:
    //dx.doi.org/10.22331/q-2024-01-18-1231.</a>





## 附件

计分规则的代码

```python
from mindquantum.core.operators import Hamiltonian
from mindquantum.simulator import Simulator
import json 
from utils.qcirc import qaoa_hubo, build_ham_high
from main import main
import time
import os 


def load_data(filename):
    '''
    Load the data for scoring.
    Args:
        filename (str): the name of ising model.
    Returns:
        Jc_dict (dict): new form of ising model for simplicity like {(0,): 1, (1, 2, 3): -1.}
    '''
    data = json.load(open(filename, 'r'))
    Jc_dict = {}
    for item in range(len(data['c'])):
        Jc_dict[tuple(data['J'][item])] = data['c'][item]
    return Jc_dict
    
def single_score(Jc_dict):    
    hamop = build_ham_high(Jc_dict)
    ham=Hamiltonian(hamop)
    s=0
    for depth in [4,8]:
        gamma_List, beta_List= main(Jc_dict, depth, Nq=Nq)
        circ= qaoa_hubo(Jc_dict, Nq, gamma_List,beta_List ,p=depth)
        sim=Simulator('mqvector',n_qubits=Nq)
        E = sim.get_expectation(ham, circ).real   
        s += -E
    return s
    
def score():
    score=0
    start=time.time()
    files = os.listdir('data/_hidden')
    for file in files:
        Jc_dict = load_data('data/_hidden/'+file)
        score+=single_score(Jc_dict)
    end=time.time()
    for propotion in [0.3,0.9]:
        for k in range(2,5):
            for coef in ['std', 'uni', 'bimodal']:
                for r in range(5):
                    Jc_dict = load_data(f"data/k{k}/{coef}_p{propotion}_{r}.json")
                    score+=single_score(Jc_dict)
                end=time.time()
	print(f'score: {score}')  
	return score

    
            
            
if __name__ == '__main__':
    Nq=12
    score()
```

