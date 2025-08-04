# 量子启发式算法在线性阵列波束赋形中的组合优化方案

李子雨 佘俊逸 郭强、重庆邮电大学

# 摘要

在雷达探测、无线通信等场景中，波束赋形需兼顾主瓣宽度窄化和旁瓣强度压制，这属于典型的组合优化问题。传统方法在大规模阵列中存在计算复杂度高、易陷入局部最优解、优化效率低的挑战。为此，本文将波束赋形优化问题转化为二次无约束二进制优化问题(Quadratic Unconstrained Binary Optimization, QUBO)，采用基于MindQuantum提供的量子启发式算法(Quantum Inspired Algorithms, QIA)模块并结合Torch微分方法进行组合优化，重点对QIA方法进行优化，致力于实现波束赋形问题的高效自动寻优。

具体而言，针对32个天线阵子组成的等间距半波长线性阵列，本文的策略为引入评分函数以量化优化效果，对QIA算法中的批量大小(batch_size)参数进行合理选取，综合多种QIA子类算法(如BSB、DSB、CFC)等求解，对候选解去重，优化相位编码方式，还对Torch微分方法和QIA方法进行择优，并优化了演化步长(dt)、迭代步数(n_iter)、损失函数控制系数(xi)等超参数。

实验确定了各参数的最优取值：演化步长dt为0.8，迭代步数n_iter为2750，损失函数控制系数xi为0.2，综合QIA子算法方案下求解批量大小batch_size为65。最终，该方案在指定资源下，优化任意单个目标波束的时间不超过90秒，实现了主瓣方向与目标方向偏差不超过  $1^{\circ}$ ，主瓣宽度尽可能小，且在相应范围内旁瓣强度低于主瓣强度的优化目标，最终求解分数相比样例提高约  $99.03\%$ ，验证了方案在提高优化效率和求解质量方面的有效性。

# 1 问题背景与描述

# 1.1 问题背景

波束赋形是通过调整天线阵列中各阵子振幅和相位，实现特定方向(主瓣)的信号增强和非目标方向(旁瓣)的信号抑制的技术。在雷达探测、无线通信等应用场景中，需兼顾主瓣宽度窄化和旁瓣强度压制，此问题属于典型的组合优化问题。传统方法在大规模阵列中面临计算复杂度高、易陷入局部最优解的挑战，优化效率偏低。因此，本文拟采用量子退火启发式算法(QIA)，结合其他优化算法，实现波束赋形问题的高效自动寻优。

# 1.2 问题描述

# 1.2.1 天线阵列与信号强度模型

$N$  个沿  $z$  轴等间距分布的半波长线性排列的天线阵子组成天线阵列，采用式(1)所示极坐标系描述空间位置，其中  $\theta$  为方向向量与  $z$  轴的夹角，  $\phi$  为方向向量投影在  $xy$  平面后与  $x$  轴的夹角。对于一维线阵，取  $\phi = 90^{\circ}$  进行研究。

$$
(\theta ,\phi), \theta \in [0,\pi ], \phi \in [0,2\pi ] \tag{1}
$$

以  $|F(\theta ,\phi)|^2$  度量天线阵列在  $(\theta ,\phi)$  方向上的信号强度，对于一维线阵，省略  $\phi$  ，记作  $|F(\theta)|^2$  。采用式(2)计算信号强度，其中，  $A_{n}(\theta)$  为阵因子，其表达式如式(3)所示， $\alpha_{n}$  为第  $n$  个阵子的相位角，在  $[0,2\pi)$  范围内按指定比特数离散调节，  $\beta_{n}$  为第  $n$  个阵子的振幅，在[0,1]范围内连续变化；  $E(\theta)$  为天线单元因子，其表达式如式(4)所示。

$$
|F(\theta)|^2 = E(\theta)\left[\sum_{n = 1}^{N}A_n(\theta)\right]\left[\sum_{n = 1}^{N}A_n^*\left(\theta\right)\right] \tag{2}
$$

$$
\left\{ \begin{array}{l}A_{n}(\theta) = I_{n}\exp \{\pi i n\cos \theta \} , \\ I_{n} = \beta_{n}\exp \{i a_{n}\} \end{array} \right. \tag{3}
$$

$$
\left\{ \begin{array}{l}E(\theta) = 10^{E_{dB}(\theta) / 10}, \\ E_{dB}(\theta) = -\min \left\{12\left(\frac{\theta - 90^{\circ}}{90^{\circ}}\right)^{2}, 30\right\} , \theta \in [0^{\circ}, 180^{\circ}] \end{array} \right. \tag{4}
$$

# 1.2.2 优化目标

结合QIA算法与其他优化算法，为由32个天线阵子组成的等间距半波长线性阵列自动搜寻主瓣方向  $\theta_{0} \in [45^{\circ}, 135^{\circ}]$  范围内的最优相位- 振幅序列  $(\{\alpha_{n}, \beta_{n}\})$  ，并实现以下目标：

1）成形方向  $[\theta_{0} - 30^{\circ}, \theta_{0} + 30^{\circ}]$  范围内旁瓣强度至少低于主瓣强度的  $-30dB$

2）$[0, \theta_{0} - 30^{\circ}]$  和  $[\theta_{0} + 30^{\circ}, 180^{\circ}]$  范围内旁瓣强度至少低于主瓣强度的  $-15dB$

3）主瓣宽度  $W$  应尽可能小，最好小于  $6^{\circ}$

4）主瓣实际方向  $\theta_{\mathrm{max}}$  与目标波束成形方向  $\theta_{0}$  相差不超过  $1^{\circ}$

5）在指定资源下，优化任意单个目标波束的时间不超过90秒。

# 1.2.3 评分规则

以  $y_{i}$  表示优化函数最终得分，评分函数如式(5)所示，各项指标计算方式如式(6)所示，其中，指标  $a$  表示非成形区间的旁瓣压制；指标  $b$  表示主瓣宽度惩罚；指标  $c$  表示成形区

间的旁瓣压制。

$$
y_{i} = 1000 - 100a - 80b - 20c \tag{5}
$$

$$
\left\{ \begin{array}{l}a = \max \left\{15 + \max \left\{10\lg \frac{|F(\theta)|^{2}}{\max |F(\theta)|^{2}}\right\} ,0\right\} \\ b = \max \left\{W - 6^{\circ},0^{\circ}\right\} \\ c = \max \left\{10\lg \frac{|F(\theta)|^{2}}{\max |F(\theta)|^{2}} +30\right\} \end{array} \right. \tag{6}
$$

特别地，如果主瓣实际方向  $\theta_{\mathrm{max}}$  与目标方向  $\theta_{0}$  的偏差超过  $1^{\circ}$  、主瓣两侧第一个极小值点的强度下降不足  $30dB$  或由公式计算出的  $y_{i}< 0$  ，则强制赋值  $y_{i} = 0$  ，公式表示如式(7)所示。

$$
y_{i} = \left\{ \begin{array}{l l}{0,|\theta_{\mathrm{max}} - \theta_{0}| > 1^{\circ}}\\ {0,10\lg \frac{|F(\theta_{1})^{2}|}{\max |F(\theta)|^{2}}\geqslant -30 o r 10\lg \frac{|F(\theta_{2})^{2}|}{\max |F(\theta)|^{2}}\geqslant -30}\\ {0,y_{i}< 0} \end{array} \right. \tag{7}
$$

# 2 问题分析

波束赋形技术在雷达探测和无线通信领域中得到了广泛的研究和应用。近年来，许多研究者针对波束赋形问题提出了多种优化算法，以提高主瓣宽度的窄化和旁瓣强度的压制效果。

文献[1]中，提出使用遗传算法(Genetic Algorithms, GA)对天线阵列的相位和振幅进行优化，GA是一种基于自然选择和遗传学原理的优化方法，它通过选择、交叉和变异等操作来逐步优化解，可取得良好的波束成形效果。但GA在处理大规模阵列时，可能会面临收敛速度慢和计算复杂度高的问题。

文献[2]中提出使用改进粒子群优化算法(Culled Fuzzy Adaptive Particle Swarm Optimization, CFAPSO)对无线传感器网络协作波束成形的节点相位和振幅权重进行优化。CFAPSO是一种融合模糊逻辑参数自适应和粒子筛选机制的混合优化方法，它通过模糊推理系统动态调整惯性权重和置信参数，并采用离线查找表机制降低计算复杂度；同时在  $50\%$  迭代次数时执行粒子筛，可有效提升收敛速度并避免早熟收敛问题。但CFAPSO在实现上需要设计精细的模糊规则和粒子筛选机制，增加了算法设计复杂度。

文献[3]提出一种概率约束联合鲁棒波束成形方法，通过联合优化发射端与接收端波束形成向量解决雷达系统中模型失配问题。其核心创新在于引入导向矢量误差的统计特

性构建无失真响应概率约束，确保目标方向增益达标，并将非凸优化问题分解为发射端与接收端两个二阶锥规划(SOCP)子问题，通过序列优化算法迭代求。但该方法需迭代求解SOCP导致实时性受，且大规模阵列计算下效率骤降。

文献[4]提出两种混合量子- 经典神经网络架构用于多用户MIMO下行波束成形优化。第一种(QNN)在经典卷积神经网络后引入小型参数化量子电路替代全连接层，通过角度编码将经典特征映射至量子态，经纠缠处理后测量输出，显著减少训练参数；第二种(QCNN)采用幅度编码的量子卷积层进行特征提取，利用量子态空间增强特征表达能力，再输入经典CNN处理。但这两种方法都存在大规模阵列量子电路计算效率骤降的问题。其突破性在于首次将变分量子电路引入波束成形优化，实现参数量线性增长与特征提取能力提升的协同优势。

综合来看，现有算法在处理大规模阵列波束赋形时仍存在计算复杂度高、易陷入局部最优、参数调优依赖经验等问题。因此，本文提出采用量子启发式算法(QIA)结合多算法协同优化策略，通过量子启发式算法的优势增强全局搜索能力，同时优化批量大小、演化步长等关键参数，以突破现有方法在大规模阵列波束赋形中的效率瓶颈，实现主瓣宽度窄化、旁瓣压制和实时优化的协同提升。

# 3 方案描述

# 3.1 核心方法描述

# 3.1.1 引入打分函数

在优化过程中，需要一个量化的指标来判断不同参数组合或优化算法的效果，以确保answer中optimized函数返回值为所用方法的最优解。

本方案将样例代码判题脚本run.py中的get_score打分函数引入answer中，将相位角、振幅以及波束成形方向、相位量化比特数作为输入，保持振幅优化选项打开，返回单个角度对应优化参数的分数(完整代码块见附录A.1.1)。

# 3.1.2 Torch微分方法

该方法基于PyTorch自动微分机制和经典的梯度下降算法。在波束赋形中，定义损失函数为主瓣信号强度与旁瓣功率之比，通过反向传播计算梯度，结合动量优化器Adam进行参数更新。在相位角固定的情况下，振幅作为连续变量，通过PyTorch的autograd机

制求导对振幅进行优化；借助模拟分叉机制，将离散的相位表示为连续变量，在优化过程中逼近离散值对相位进行优化。Torch微分方法对连续变量优化效果稳定，适合振幅优化，但对于离散变量优化能力有限，易于陷入局部最优解。

# 3.1.3 QIA方法

该方法基于量子退火启发式优化机制(QAIA)，通过模拟物理系统中能量演化与最小化的过程，解决组合优化问题。其核心思想是将优化任务映射为伊辛模型形式的二进制优化问题(QUBO)，进而利用多种模拟量子演化机制(如BSB、DSB、LQA等)求解系统的最低能量态，从而获得最优解。

在波束赋形问题中，目标是优化天线阵列的相位设置，以在指定方向形成主瓣并压制旁瓣。该任务本质上属于离散变量优化问题，适合使用QAIA处理，为实现这一目标，需先将波束增益函数转化为QUBO的形式。令相位通过k- bit二进制向量编码，并引入一组复数系数  $c_{1},c_{2},\ldots ,c_{n}$ ，将每个阵元的复数相位表示为式(8)，整个阵列在方向  $\theta$  上的波束输出表示为式(9)，其中，  $a_{n}$  为第  $n$  个天线阵元的幅度系数，  $A_{n}(\theta)$  为第  $n$  个天线阵元在方向  $\theta$  的方向响应，矩阵表示形式中，  $x$  为所有二进制编码变量组成的向量，  $C$  为复系数构成的矩阵，  $e_{\theta}[n] = a_{n}A_{n}(\theta)$  为各天线在角度  $\theta$  上的响应。

$$
z_{n} = \sum_{i = 1}^{k}c_{i}x_{ni},x_{ni}\in \{-1, + 1\} \tag{8}
$$

$$
z(\theta) = \sum_{n = 1}^{N}a_{n}z_{n}A_{n}(\theta)\Rightarrow F(\theta) = x^{T}C^{T}e_{\theta} \tag{9}
$$

通过构造如式(10)所示目标函数，实现主瓣增强与旁瓣抑制的统一优化，其中J矩阵如式(11)所示，最终的J矩阵如式(12)所示，是QUBO问题的核心输入。将构造好的J矩阵输入MindQuantum提供的QAIA模块，以模拟量子物理系统的能量演化过程，对目标函数式(10)进行全局搜索最小化。由于这些算法具有强非凸优化能力，能有效跳出局部最优，特别适合处理相位离散空间这种非连续、高维组合问题。

$$
L(x) = x^{T}[-\lambda \cdot J_{main} + (1 - \lambda)\cdot J_{side}]x,\lambda \in (0,1) \tag{10}
$$

$$
\left\{ \begin{array}{ll}J_{main} = C^{\dagger}e_{\theta_0}e_{\theta_0}^{\dagger}C\\ J_{side} = \sum w_i\cdot C^{\dagger}e_{\theta_i}e_{\theta_i}^{\dagger}C \end{array} \right. \tag{11}
$$

$$
J = \mathrm{Re}\left(\lambda J_{main} - (1 - \lambda)J_{side}\right) \tag{12}
$$

通过QIA方法，可高效地在大规模天线阵列中实现高质量的波束赋形，包括精确聚焦主瓣方向、压制特定角度的干扰波束。

# 3.1.4 优化相位比特编码方式

在波束赋形中，天线阵元的复数加权表示如式(13)所示，其中  $a_{n}$  为振幅，  $\phi_{n}$  为相位。为了进行组合优化，需要将优化目标转化为二次型表达式以兼容QUBO问题的求解方式，通常将相位  $\phi_{n}$  离散化为有限集合，即进行相位比特编码。1个k- bit相位编码意味着相位取值为  $\phi_{n}\in \{2\pi /2^{k}m|m = 0,1,\dots,2^{k} - 1\}$  。

$$
w_{n} = a_{n}e^{j\phi_{n}} \tag{13}
$$

针对2bit编码方式，相位可取  $\phi \in \{0,\pi /2,\pi ,3\pi /2\}$ ，等价为  $e^{j\phi}\in \{1,j, - 1, - j\}$ ，这些点均匀分布在复平面单位圆上，形成4象限对称结构。为了将离散相位映射到优化变量上，根据文献[5]采用的方法，使用如式(14)所示复数线性组合方式表示相位，令编码字典为矩阵  $X$ ，对应目标为  $p$ ，如式(15)所示，解得  $c_{1} = \frac{1 + j}{2},c_{2} = \frac{1 - j}{2}$ 。

$$
e^{j\phi} = c_{1}s_{1} + c_{2}s_{2},s_{n}\in \{-1, + 1\} \tag{14}
$$

$$
X = \left[ \begin{array}{ll} + 1 & +1\\ -1 & +1\\ +1 & -1\\ -1 & -1 \end{array} \right],t = \left[ \begin{array}{l}1\\ -j\\ j\\ -1 \end{array} \right] \tag{15}
$$

针对3bit编码方式，相位可取  $\phi \in \{0,\pi /4,\pi /2,3\pi /4,\pi ,5\pi /4,3\pi /2,7\pi /4\}$ ，共8个相位点，构建表达式如式(16)所示，这些点均匀分布在复平面单位圆上，形成4象限对称结构。为了将离散相位映射到优化变量上，采用与2bit编码相同的方法，得到复系数近似结果如式(17)和表1所示。

$$
e^{j\phi} = c_{1}s_{1} + c_{2}s_{2} + c_{3}s_{3} + c_{4}s_{1}s_{2}s_{3},s_{n}\in \{-1, + 1\} \tag{16}
$$

$$
\frac{1}{4}\left\{ \begin{array}{l l}{\sqrt{4 + 2\sqrt{2}}\cdot e^{-j\frac{2\pi}{8}}}\\ {\sqrt{4 + 2\sqrt{2}}\cdot e^{-j\frac{\pi}{8}}}\\ {\sqrt{4 + 2\sqrt{2}}\cdot e^{-j\frac{\pi}{8}}}\\ {\sqrt{4 + 2\sqrt{2}}\cdot e^{-j\frac{5\pi}{8}}}\\ {\sqrt{4 + 2\sqrt{2}}\cdot e^{-j\frac{5\pi}{8}}} \end{array} \right.\Rightarrow \left\{ \begin{array}{l l}{c_{1}\approx 0.25 + 0.6036j}\\ {c_{2}\approx 0.6036 - 0.25j}\\ {c_{3}\approx 0.25 - 0.1036j}\\ {c_{4}\approx -0.1036 - 0.25j} \end{array} \right. \tag{17}
$$

但该编码方式会不可避免地引入非线性组合项  $s_{1}s_{2}s_{3}$ ，这会导致原本的QUBO问题

变成高阶无约束二进制优化(Higher- Order Unconstrained Binary Optimization, HUBO)问题，但在量子退火机或QIA算法中，高阶项不易直接建模或求解，需要额外的变量或替代结构，求解复杂度显著上升。为了使问题回到QUBO可处理的二次形式，本方案引入冗余自旋变量  $s_4$ ，来替代高阶项，如式(18)所示，加入冗余变量后各变量的复系数如表2所示。此策略相当于将原本高阶多项式问题投影到了更高维空间中的二次形式，通过冗余变量编码来避免不可处理的高阶优化问题。

$$
s_4 = s_1s_2s_3 \Rightarrow e^{j\phi} = c_1s_1 + c_2s_2 + c_3s_3 + c_4s_4 \tag{18}
$$

表13bit编码自旋变量取值组合与相位角关系及复系数  

<table><tr><td colspan="2">s1</td><td colspan="2">s2</td><td colspan="2">s3</td><td colspan="2">s1s2s3</td></tr><tr><td>ak</td><td>c1=0.25+0.6036j</td><td>c2=0.6036-0.25j</td><td>c3=0.25-0.1036j</td><td colspan="4">c4=-0.1036-0.25j</td></tr><tr><td>0</td><td>+1</td><td>+1</td><td>+1</td><td colspan="4">+1</td></tr><tr><td>π/4</td><td>+1</td><td>+1</td><td>-1</td><td colspan="4">-1</td></tr><tr><td>π/2</td><td>+1</td><td>-1</td><td>+1</td><td colspan="4">-1</td></tr><tr><td>3π/4</td><td>+1</td><td>-1</td><td>-1</td><td colspan="4">+1</td></tr><tr><td>π</td><td>-1</td><td>-1</td><td>-1</td><td colspan="4">-1</td></tr><tr><td>5π/4</td><td>-1</td><td>-1</td><td>+1</td><td colspan="4">+1</td></tr><tr><td>3π/2</td><td>-1</td><td>+1</td><td>-1</td><td colspan="4">+1</td></tr><tr><td>7π/4</td><td>-1</td><td>+1</td><td>+1</td><td colspan="4">-1</td></tr></table>

表2引入冗余变量后3bit编码自旋变量取值组合与相位角关系及复系数  

<table><tr><td></td><td>s1</td><td>s2</td><td>s3</td><td>s4</td></tr><tr><td>ak</td><td>c1=0.5</td><td>c2=0.5j</td><td>c3=0.5</td><td>c4=0.5j</td></tr><tr><td>0</td><td>+1</td><td>±1</td><td>+1</td><td>±1</td></tr><tr><td>π/4</td><td>+1</td><td>+1</td><td>+1</td><td>+1</td></tr><tr><td>π/2</td><td>±1</td><td>+1</td><td>±1</td><td>+1</td></tr><tr><td>3π/4</td><td>-1</td><td>+1</td><td>-1</td><td>+1</td></tr><tr><td>π</td><td>-1</td><td>±1</td><td>-1</td><td>±1</td></tr><tr><td>5π/4</td><td>-1</td><td>-1</td><td>-1</td><td>-1</td></tr><tr><td>3π/2</td><td>±1</td><td>-1</td><td>±1</td><td>-1</td></tr><tr><td>7π/4</td><td>+1</td><td>-1</td><td>+1</td><td>-1</td></tr></table>

# 3.2 求解流程优化

# 3.2.1 综合QIA子类算法求解

3.2.1 综合QIA子类算法求解因不同的QIA子类算法具有不同的搜索策略和收敛特性，使用多种QIA子类算法，可以扩大搜索空间，增加找到全局最优解的可能性。本方案分别运行BSB、DSB、CFC、SimCIM、SFC、NMFA、LQA算法并存储每个子类算法的结果和对应的得分，最后选择分数最高的结果作为最终的优化结果并返回(完整代码块见附录A.1.3)。

# 3.2.2 QIA求解结果去重

在QIA优化求解过程中，因存在取符号操作，可能会导致出现多个相同的二进制最优解向量，这些重复的解会增加计算量，并且可能会影响最终结果的选择。

本方案中，调用numpy库中np.unique函数对包含所有候选解的结果矩阵进行去重，从而提高QIA算法的运行效率，在得到相同结果的前提下节省计算时间，这一思想在3.2.1所示伪代码中有所体现。

以QIA下BSB算法求解f矩阵为例，2bit相位编码、batch_size=70组合下，候选解数量从  $n_{QIA}\cdot batch_{size} = 490$  组减少到了394组，如图1所示；3bit相位编码、batch_size=70组合下，候选解数量从490组减少到了418组，如图2所示。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/df4b868b4d21b5109b3402729c702005c37098b1d04ea6a39550f19c5f6d7ff0.jpg)  
图1 2bit相位编码、batch_size=70组合下候选解去重效果

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/51d1650bc22d83b3bbcf335a947a3f05cdf110d3a17b59c73b5043d533e90a56.jpg)  
图2 3bit相位编码、batch_size=70组合下候选解去重效果

# 3.2.3 QIA与Torch组合求解择优

在3.1核心方法描述中分别介绍了QIA和Torch方法各自的特点与优势，本方案期望组合QIA与Torch方法进行求解，探索可能的最佳求解组合。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/89d00f9562043100cebf60d4d888f07662b9574b8f90e5f35f76a86ec274d63d.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/316926e411c73c09d1999ee466b3d317d95d9189691d670f8961ab6904d91cd4.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/97600ad4f1447b76b72cafd6b10db4dbbd0433b0c2dd21412e405c289e032a18.jpg)  
图3 Torch微分优化相位振幅(上)、QIA优化相位(中)、Torch与QIA组合优化(下)结果对比

本方案中，在solve函数中分别使用仅Torch方法优化相位与振幅、仅QIA优化相位、Torch与QIA组合优化相位与振幅三种方法，三种方法在目标角度为  $90^{\circ}$  度下的波束图输出样例如图3所示，通过评分函数对优化结果进行评价，对应得分如表3所示，选择分数更高的解作为最终结果。通过择优选择Torch方法和QIA的组合方法进行优化，可以充分发挥各自的优势，提高优化的效果和效率(完整代码块见附录A.1.4)。

表3 不同优化方法下的得分  

<table><tr><td>优化方法</td><td>得分(score)</td><td>W</td><td>a</td><td>b</td><td>c</td></tr><tr><td>Torch优化相位振幅</td><td>703.67</td><td>11.81</td><td>0</td><td>5.81</td><td>-8.43</td></tr><tr><td>QIA优化相位</td><td>589.59</td><td>6.94</td><td>0</td><td>0.94</td><td>16.72</td></tr><tr><td>Torch与QIA组合优化</td><td>730.19</td><td>11.78</td><td>0</td><td>5.78</td><td>-9.63</td></tr></table>

# 3.3 超参数优化

# 3.3.1 QIA方法优化批量大小(batch_size)参数

在QIA算法中，批量大小(batch_size)直接影响算法的搜索效率和求解质量，较大的batch_size能并行处理更多候选解，扩大搜索空间，提升找到全局最优解的概率，但会增加计算资源消耗和单次迭代时间；较小的batch_size则收敛更快，但可能陷入局部最优。因此，需要通过实验确定batch_size的最优取值，在搜索精度和效率间取得平衡。

此外，本方案针对不同batch_size取值开展实验时，会借助评分函数对每batch内的候

选解进行评估，从所有batch中筛选出最优批次，充分挖掘大batch_size带来的更广泛搜索空间价值，既利用其扩大搜索范围的优势，又通过评分机制精准捕捉优质解。具体而言，会针对不同batch_size取值进行实验，先依据评分函数选出各batch里的最优解，再综合对比不同batch_size实验下的整体分数表现，分数最高实验下对应的batch_size取值即为最优取值(完整代码块见附录A.1.2)。因batch_size取值与求解时间有强关联，且题目有明确的时间限制，在得到batch_size取值与分数之间的关系后，应对不同batch_size取值下的求解时间进行统计分析，以确保方案在严格求解时间约束下的可行性。

# 3.3.2 优化演化步长(dt)与迭代步数(n_iter)

演化步长(dt)和迭代步数(n_iter)是模拟分叉(SB)算法中的两个重要超参数，其中，dt决定了每次参数更新的幅度，如果dt取值过大，可能会导致优化过程不稳定，无法收敛到最优解；如果dt取值过小，优化速度可能会变慢。n_iter决定了迭代的次数，如果n_iter取值过大，会增加计算时间；如果n_iter取值过小，可能无法达到最优解。因此，二者需协同分析：dt取值小时需更多n_iter保证收敛，dt取值大时若n_iter不足则无法稳定，组合分析能找到效率与精度的平衡。

本方案拟通过不同dt与n_iter取值组合实验下的分数来确定最优的取值组合，分数最高实验下对应的dt与n_iter取值组合即为最优取值组合(完整代码块见附录A.1.5)。

# 3.3.3 优化损失函数控制系数(xi)

在模拟分叉(SB)算法中，优化损失函数控制系数(xi)用于调节损失函数梯度的相对大小。在参数更新过程中，梯度会与xi相乘后再参与后续计算，通过调整xi的值，可以控制梯度对参数更新的影响程度。如果xi取值过大，梯度的影响会增加，可能会导致优化过程不稳定，出现震荡或发散的情况；如果xi取值过小，梯度的影响会减弱，优化速度会变慢甚至可能陷入局部最优解。因此，合理选择xi的取值，可以平衡梯度的影响，使优化过程更加稳定高效。

本方案拟通过不同xi取值实验的分数确定最优的取值，分数最高实验下对应的xi取值即为最优取值(完整代码块见附录A.1.6)。

# 3.4 方案总体优化思路

QIA方法与Torch微分方法中的关键参数(超参数)是优化算法结果的重点，应先对各关键参数(超参数)的最优取值进行系统实验，以确保在对各子算法进行组合优化或流程优化前，各子算法的求解状态均为最优，使最后的优化结果更具说服力。方案总体优化流程如图4所示。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/a7b67ec12b2db40bae8caa9196cc94650b16d0b119df7875560505f030e7207e.jpg)  
图4 方案总体优化流程图

# 4 结果与分析

# 4.1 参数(超参数)最优取值实验

# 4.1.1 QIA方法参数批量大小(batch_size)最优取值实验

本实验以样例代码answer_1. py为基础，仅将BSB算法中的batch_size参数作为变量，取值列表为[1,2,3,4,5,10,20,50,100,200,300,400,500,700,1000]，计算不同batch_size取值下的分数，绘制折线图展示batch_size取值与分数的关系，如图5所示。

根据图5，分数与batch_size总体呈正相关，batch_size达到  $10^{2}$  数量级后分数增幅有限，这是因为batch_size取值越大，QIA算法中取符号后的二进制最优解向量候选解重复的数量也就越多，难以得到更优解。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/b1cfd8a6b9c783766cde01388a88eb39d77661986dd20350f6d52a2ab94e6105.jpg)  
图5 batch_size取值与对应得分关系折线图

此外，因题目有明确时间限制，且batch_size的取值是影响求解时间的重要因素，应对候选batch_size取值区间再进行计算求解时间的相关实验。为贴合最终优化方案的求解时间，本实验在运用综合QIA子算法求解、候选解去重、3bit相位编码、QIA方法与Torch方法组合择优等方案上进行，batch_size取值列表为[1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]，计算不同batch_size取值下的求解时间，绘制折线图展示batch_size取值与求解时间的关系，如图6所示。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/cbe8a60ef2925ede2b322153fc833cc43b63144b7526ecb42f71e292a5baa75d.jpg)  
图6 batch_size取值与对应求解时间关系折线图

根据图6，batch_size取值大于70时，有较高几率导致大部分求解目标求解时超时，导致多个求解目标分数置0，从而严重影响最后得分，为综合方案整体效率，batch_size

取值为70时最为保险，即batch_size=70是适合本方案的，在题目限定求解时间下的理想取值。

# 4.1.2 Torch微分方法超参数演化步长(dt)与迭代步数(n_iter)最优组合取值实验

本实验以样例代码answer.py为基础，仅将SB算法中的超参数n_iter、dt作为变量，取值列表分别为[500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]和[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]，以计算10个优化目标方向作为一组实验，计算不同n_iter与dt组合取值下的分数，共进行2组实验，分别绘制热图展示n_iter与dt组合取值与分数的关系，如图7所示。

根据图7，两组实验展示的n_iter与dt组合取值与分数的关系并无明显特征，两组实验的唯一共性为dt=0.8时，n_iter=2750左右均出现最高分，证明dt=0.8，n_iter=2750是优化分数的理想取值组合。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/790cd2e0127fc5209a001c742789bed4a144cafc07e7d38f8501cbf39f04fceb.jpg)  
图7 n_iter、dt取值组合与对应得分关系热图

# 4.1.3 Torch微分方法超参数损失函数控制系数(xi)最优取值实验

本实验以样例代码answer.py为基础，仅将SB算法中的超参数xi作为变量，取值列表为[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]，以计算10个优化目标方向作为一组实验，计算不同n_iter与dt组合取值下的分数，共进行7组实验，分别绘制折线图展示每组实验xi取值与分数的关系，并绘制每个取值下7组实验得分的平均值折线体现总体分数变化趋势，如图8所示。

根据图8，多数实验在xi取值区间[0.10, 0.20]出现峰值，表明该区间是优化分数的潜力区间，应关注此取值区间进行取值优化。由分数平均值折线可知，分数平均值在xi取值为0.20时出现峰值，证明xi取0.20是优化分数的理想取值。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/36b3fb69d8838b785a6f875f3c515019c3812a0609be63f4b6ce57227b9edcd6.jpg)  
图8 xi取值组合与对应得分关系折线图

# 4.1.4 参数(超参数)最优取值实验总结

综合上述实验结果，得到两种计算方法中四个参数(超参数)的最优取值，如表4所示。

表4参数（超参数）试验后最优取值  

<table><tr><td>参数(超参数)</td><td>取值</td></tr><tr><td>演化步长(dt)</td><td>0.8</td></tr><tr><td>迭代步数(n_iter)</td><td>2750</td></tr><tr><td>损失函数控制系数(xi)</td><td>0.2</td></tr><tr><td>批量大小(batch_size)</td><td>70</td></tr></table>

# 4.2 方案整体优化结果

在4.1中参数选择的基础上，使用QIA子算法综合求解方案、QIA候选解去重方案、QIA方法与Torch方法择优方案进行求解，并进行在线测试评分，得到优化方案与在线提交得分区间关系如图9所示，其中优化方案从下至上为包含关系，在线测试最高分为401.28分，相比样例方案提升约  $37.4\%$  (决赛阶段前)。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/7fcee509abce660fcb5484d9e8e05d25fec2a5e3db3f682f8ea8a553b88e04fd.jpg)  
图9 优化方案与在线提交得分区间关系(决赛前在线测试数据)

因决赛阶段后在线测试通道关闭，且在线测试数据集为隐藏数据集，所以本方案自定义测试数据集在本地测试各阶段优化方案的优化效果进行横向对比。定义测试数据集参数组合为[[50.0, 2, True], [60.0, 2, True], [70.0, 2, True], [80.0, 2, True], [90.0, 2, True], [100.0, 2, True], [110.0, 2, True], [120.0, 2, True], [130.0, 2, True]]（如果优化方法支持3bits相位编码，则测试数据集中编码位数改为3），覆盖目标角度可取值范围，以确保得分的说服力。得到的测试结果得分如表5所示，最优方案下测试集各参数下得分如表6所示，对应输出波束图如图10所示。

表5 本方案组合优化下的最优得分  

<table><tr><td rowspan="2">优化方法</td><td colspan="2">在线测试</td><td colspan="2">本地测试</td></tr><tr><td>最高得分</td><td>增幅</td><td>最高得分</td><td>增幅</td></tr><tr><td>仅QIA-BSB(样例代码)</td><td>179.62</td><td>--</td><td>173.87</td><td>--</td></tr><tr><td>仅Torch微分(样例代码)</td><td>261.59</td><td>--</td><td>218.28</td><td>--</td></tr><tr><td>QIA-BSB增大batch_size为70</td><td>372.78</td><td>42.49%</td><td>332.44</td><td>52.32%</td></tr><tr><td>增加其他QIA子算法求解</td><td>390.63</td><td>49.34%</td><td>405.89</td><td>85.94%</td></tr><tr><td>QIA相位与Torch振幅相位择优</td><td>401.28</td><td>53.40%</td><td>418.62</td><td>91.78%</td></tr><tr><td>改为3bit相位编码</td><td>--</td><td>--</td><td>425.99</td><td>95.16%</td></tr><tr><td>QIA&amp;amp;Torch组合优化择优</td><td>--</td><td>--</td><td>434.45</td><td>99.03%</td></tr></table>

表6 本方案组合优化下的最优得分  

<table><tr><td>测试目标角度θ0</td><td>W</td><td>a</td><td>b</td><td>c</td><td>得分</td><td>总分</td></tr><tr><td>50°</td><td>9.15</td><td>8.08</td><td>3.15</td><td>17.9</td><td>0</td><td></td></tr><tr><td>60°</td><td>12.51</td><td>0</td><td>6.51</td><td>-3.15</td><td>542.35</td><td></td></tr><tr><td>70°</td><td>7.56</td><td>0</td><td>1.56</td><td>16.49</td><td>545.75</td><td></td></tr><tr><td>80°</td><td>7.04</td><td>0</td><td>1.04</td><td>20.11</td><td>514.37</td><td></td></tr><tr><td>90°</td><td>11.78</td><td>0</td><td>5.78</td><td>-9.63</td><td>730.19</td><td>434.45</td></tr><tr><td>100°</td><td>7.62</td><td>0</td><td>1.62</td><td>18.76</td><td>495.45</td><td></td></tr><tr><td>110°</td><td>7.56</td><td>0</td><td>1.56</td><td>16.49</td><td>545.75</td><td></td></tr><tr><td>120°</td><td>12.48</td><td>0</td><td>6.48</td><td>-2.74</td><td>536.21</td><td></td></tr><tr><td>130°</td><td>10.39</td><td>8.28</td><td>4.39</td><td>17.43</td><td>0</td><td></td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/116389f1294261d46975bb1c52511658f283d55201a91110774451f689c11f2c.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-04/54af1165-3c0d-4e4f-a941-51a311e704f3/fcc756d74fe605240fb3381a2cf2dede3192157d9b9544cea7116d57a5e6afbf.jpg)  
图10 本方案组合优化下的最优得分对应波束图

# 5 参考文献

[1] Wang S, Li S, Sun H. Genetic algorithm- based beamforming using power pattern function[C]//Communications, Signal Processing, and Systems: Proceedings of the 2018 CSPS Volume III: Systems 7th. Springer Singapore, 2020: 159- 167.

[2] Maina R M, Lang'at P K, Kihato P K. Collaborative beamforming in wireless sensor networks using a novel particle swarm optimization algorithm variant[J]. Heliyon, 2021, 7(10). [3] SubbaRaju P S, Mohamed H, Susaritha M, et al. Beamforming Optimization in Massive MIMO Networks Using Improved Subset Optimization Algorithm with Hybrid Beamforming[C]//2025 3rd International Conference on Integrated Circuits and Communication Systems (ICICACS). IEEE, 2025: 1- 7.

[4] Zhang J, Zheng G, Koike- Akino T, et al. Hybrid quantum- classical neural networks for downlink beamforming optimization[J]. IEEE Transactions on Wireless Communications, 2024.

[5] Jiang Y, Ge H, Wang B Y, et al. Quantum- inspired beamforming optimization for quantized phase- only massive MIMO arrays[J]. arXiv preprint arXiv:2409.19938, 2024.
