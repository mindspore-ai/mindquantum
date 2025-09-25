# 黑客松量子启发式算法赛道：使用量子启发式算法求解 MIMO detection 问题

[TOC]



# 一、问题背景

## 1.1 MIMO 检测问题

多输入多输出（Multiple－Input Multiple－Output，MIMO）是满足当前和未来无线系统对大量数据流量需求的关键技术。在这种技术中，发射端和接收端都配备了多个天线，以改善基于多径传播的无线链路容量。

对于 MIMO 系统，$N_t$ 个用户使用单天线发送符号 $\boldsymbol{x}=\left[x_1, x_2, \ldots, x_{N_t}\right]^T \in \mathbb{C}^{N_t}$ 来自星座 $\Omega$ ，然后由基站用 $N_r$ 条天线接收符号，接收到的符号可以表示为 $\boldsymbol{y}=\left[y_1, y_2, \ldots, y_{N_r}\right]^T \in \mathbb{C}^{N_r}=\boldsymbol{y}^R+j \boldsymbol{y}^I$ 。 $N_t$ 个用户天线和 $N_r$ 个接收天线之间的传输过程可以用信道矩阵 $\mathbf{H} \in \mathbb{C}^{N_r \times N_t}=\mathbf{H}^R+j \mathbf{H}^I$ 来描述。最后，这个过程可以表述为

$$
\mathbf{y}=\mathbf{H} \mathbf{x}+\mathbf{n}
$$


其中，$  \mathbf{n} \in \mathbb{C}^{N_r}$ 表示加性高斯白噪声。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925145442909.png" style="zoom:67%;" />

图 1：MIMO 系统。左边为用户端发射天线，右边为基站端接收天线，中间为信道矩阵。

在 MIMO 系统中，调制用于将待传输的信号转换成适合在通信信道传播中传输的形式。每个天线都可以传输经过调制的数据流，这些数据流通过空间复用可以在相同的频谱资源上实现并发传输，从而显著提高系统的吞吐量。在此过程中，信号通常被映射到称为星座图的特定点上，这些点代表了不同调制阶数（如 BPSK、QPSK、16-QAM、64-QAM 等）中的符号。

 <img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925145523648.png" style="zoom:50%;" />

图 2：QPSK（4－QAM）星座图。图中包含 4 个点，分别表示复平面上的四个方向，表示两比特信息。

MIMO 检测问题是在给定信道矩阵 $\mathbf{H}$ 和接收符号 $\boldsymbol{y}$ 的情况下尽可能重现传输符号 $\boldsymbol{x}$ 。然而，对于接收端来说，在存在噪声和干扰的情况下，根据信道状态信息和接收到的信号来重建传输的符号是一项具有挑战性的任务，寻找准确的解决方案，如最大似然检测器，已被证明是 NP 难问题。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925145609678.png" alt="image-20250925145609678" style="zoom:67%;" />

图 3：MIMO 检测问题一般天线配置与检测方法 [2]

举个例子，最大似然检测器（Maximal Likelihood Detector，MLD）是在存在噪声情况下的最小化检测误差的 MIMO 检测方案［3］，表示如下：

$$
\hat{\mathbf{x}}_{M L}=\arg \min _{\mathbf{x} \in \Omega^{N_t}}\|\mathbf{y}-\mathbf{H} \mathbf{x}\|_2^2
$$
近年来，研究人员开始尝试使用量子退火或量子启发式算法（Quantum Annealing Inspired Algorithms，QAIA）来解决 MIMO 检测问题，即将 MIMO 检测问题转化为量子启发式算法擅长的 Ising 问题，并使用 QAIA 算法求解。

## 1.2 Ising 建模过程

量子启发式算法通过引入 variable－to－symbol 转化函数，将 MLD 模型转化为 Ising 模型，Ising问题一般格式如下：
$$
\min _{\mathbf{s} \in\{-1,+1\}^n} H(\mathbf{s})=\sum_{1 \leq i \leq j \leq n} J_{i j} s_i s_j+\sum_{i=1}^n h_i s_i
$$

其中，$J_{i j}$ 是耦合矩阵，$h_i$ 是外场项。

### 1．ML－to－QUBO 问题简化

ML－MIMO 检测问题转化为 QUBO 形式的关键是用 0 和 1 的变量来表示可能传输的符号。如果这是一个线性转换，那么范数的展开也将产生线性项和二次项。因此，引入一个线性的变量 $q$ 和信号 $v$的转换 $T$［1］，其中 $q$ 是 $\{0,1\}$ 的变量，$v$ 是复数。举个例子，QPSK 的转换为 $00 \leftrightarrow-1-j, 01 \leftrightarrow-1+j, 10 \leftrightarrow 1-j, 11 \leftrightarrow 1+j$ 。最直接的从 QUBO 转换为 Ising 模型的方式是执行 $q_i \leftrightarrow \frac{1}{2}\left(s_i+1\right)$ 。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925145819571.png" style="zoom:80%;" />

图 4：ML－to－QUBO 问题转化过程［1］及举例。图中 16－QAM 应为 $4 q_{4 i-3}+2 q_{4 i-2}-3+j\left(4 q_{4 i-1}+2 q_{4 i}-3\right)$ ，图引用自［13］

### 2．ML－to－Ising 一般形式［3］

Ising 模型的转换过程如下［3］，对于 $y=H x+n$ ，其等价于实值形式
$$
\hat{x}=\left[\begin{array}{c}
\Re(x) \\
\Im(x)
\end{array}\right], \hat{y}=\left[\begin{array}{c}
\Re(y) \\
\Im(y)
\end{array}\right], \hat{H}=\left[\begin{array}{cc}
\Re(H) & -\Im(H) \\
\Im(H) & \Re(H)
\end{array}\right]
$$


其中，$\Re(\cdot) 、 \Im(\cdot)$ 分别表示实部和虚部，$x$ 是发射信号，$y$ 是接收信号，$H$ 是信道矩阵，$n$ 是加性高斯白噪声。

使用 QuAMax transform 编码后，发射信号公式为［3］

$$
\begin{aligned}
\mathrm{T} & =\left\lfloor 2^{r_b-1} \mathbb{I}_N \quad 2^{r_b-2} \mathbb{I}_N \ldots \mathbb{I}_N\right\rfloor \\
\hat{\mathrm{x}} & =\mathrm{T}\left(\mathrm{~s}+\overline{\mathbb{I}}_{\mathrm{N} * \mathrm{r}_{\mathrm{b}}}\right)-(\sqrt{\mathrm{M}}-1) \overline{\mathbb{I}}_{\mathrm{N}}
\end{aligned}
$$


其中， T 为转换矩阵，执行变量－信号转换操作。
令 $z=\hat{y}-\hat{H} \mathrm{~T} \bar{\Pi}_{N * r_b}+(\sqrt{M}-1) \hat{H} \bar{\Pi}_N$ ，在高阶调制下，原本的耦合矩阵 J 和外场项 h 计算公式为［3］

$$
\begin{aligned}
& \mathrm{h}=2 * z(\hat{H} T)^{\mathrm{T}} \\
& \mathrm{~J}=-z \operatorname{eroDiag}\left[(\hat{H} T)^T \hat{H} T\right]
\end{aligned}
$$

其中，zeroDiag（ W ）是将矩阵 W 的对角线元素设 0 。
另外，由于使用［1］中提到的转换 $T$ ，接收端为 QuAMax transform 编码［1］，而发送端为 Gray code 编码，因此在得到结果后需要对编码格式做转换才能比较，从而得到误码率。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925145922757.png" alt="image-20250925145922757" style="zoom:80%;" />

图 5：16-QAM，QuAMax transform-to-Gray code 转换过程 [1]。

接收端的转换过程大致为：
Ising spins $\{-1,1\}$－＞Qubo bits＿hat $\{0,1\}$ ，QuAMax transform－＞bits＿hat $\{0,1\}$ ，Intermediate code－＞bits＿final $\{0,1\}$ ，Gray code

# 二、赛题说明

## 2.1 数据集

比赛提供 150 个训练数据（＊。pickle），每个数据包含：

- H ：信道矩阵， $\mathrm{Nt}=\mathrm{Nr}, 64^* 64 、 128^* 128$
- y：接收信号
- bits：用二进制｛0，1\}表示的传输信号
- num_bits_per_symbol：每个信号所包含的比特数目（调制阶数为 $2^{n u m \_b i t s \_p e r \_s y m b o l}$ ）， $\{4,6,8\}$
- SNR：信噪比，$\{10,15,20\}$
- ZF_ber：基线算法 zero_forcing 的误码率结果

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925155139406.png" style="zoom:50%;" />

图 5：数据集可视化。以第一个数据文件为例，左上角的图为 bits 映射到星座图上的信号表示 x ，右上角的图为传输过程加噪声后接收端收到的信号表示 y＿hat，左下角的图为忽略噪声，直接逆向计算得到发射信号 x_hat，右下角为实际接收信号 y。

## 2.2 评分标准

对于给定的 MLD 实例，计算其比特错误率： $\mathrm{BER}=\frac{|\mathrm{s}-\hat{\mathbf{s}}|}{2 N_t \cdot \log _2|\Omega|}$ 。
设各个实例 BER 算数平均值为 $\overline{\mathrm{BER}}$ ，得分公式为 score $=(1-\overline{\mathrm{BER}}) \times \frac{\text { baseline＿time }}{\text { running＿time }} 。$ 其中，baseline＿time 为基线方法运行时间，running＿time 为选手代码运行时间。

# 三、Hackathon－QAIA 赛道各团队方案

## 3.1 方案一（灿言团队）

#### 3.1.1 现有方法及问题

- 球形求解器：计算复杂度指数级增加
- 线性求解器：需要接收天线数量大于用户数量
- 最大似然求解器：$\hat{\mathbf{x}}_{M L}=\arg \min _{\mathbf{x} \in \Omega^{N_t}}\|\mathbf{y}-\mathbf{H} \mathbf{x}\|_2^2$ ，NP－hard
- Ising 模型求解：转化为 Ising 问题，用 Ising 机或者量子启发式算法求解

这种方法的大致步骤为，将信号检测问题转化为 Ising 问题，对 Ising 问题求解，对原始解符号和求解得到的符号的编码格式进行统一并进行比较计算误码率。

这个方法的问题在于，在高信噪比区域会出现误码平层（error floor），即误码率（BER）不再随着信噪比（SNR）的增加而显著下降，而是趋于平缓。

### 3.1.2 改进：基于 bSB（LMMSE－like 矩阵、参数优化）

#### 1．LMMSE－like 正则项：克服误码平层，提高求解精度

- **发射符号归一化**

为了更适合真实环境且不影响性能，将其进行归一化操作。计算发射符号的方差，即
$$
\operatorname{var}_{q a m}=\frac{\left[\sum_{i=1}^{2^{\left(r_b-1\right)}}(2 * i-1)^2\right]}{2^{\left(r_b-2\right)}}
$$


用标准差对公式（5）进行归—化

$$
\begin{aligned}
\mathrm{T} & =\left\lfloor 2^{r_b-1} \mathbb{I}_N \quad 2^{r_b-2} \mathbb{I}_N \ldots \mathbb{I}_N\right\rfloor / \sqrt{\operatorname{var}_{q a m}} \\
\hat{\mathrm{x}} & =\mathrm{T}\left(\mathrm{~s}+\overline{\mathbb{I}}_{\mathrm{N} * \mathrm{r}_{\mathrm{b}}}\right)-(\sqrt{\mathrm{M}}-1) \overline{\mathbb{I}}_{\mathrm{N}} / \sqrt{\operatorname{var}_{q a m}} \\
z & =\hat{y}-\hat{H} \mathrm{~T} \overline{\mathbb{I}}_{N * r_b}+(\sqrt{M}-1) \hat{H} \overline{\mathbb{I}}_N / \sqrt{\operatorname{var}_{q a m}}
\end{aligned}
$$

- **加入 LMMSE－like 正则项**

论文［4］中加入了类似 $x_{L M M S E}=H^T\left(H H^T+\lambda I\right)^{-1} y$ 形式的正则项，方案一同样在公式
（6）中加入类似 $\left(H H^T+\lambda I\right)^{-1}$ 形式的部分，即
$$
\left[(\hat{H} T)(\hat{H} T)^T+\lambda I\right]^{-1}
$$


令 $\mathbf{U}=(\hat{H} T)^T\left[(\hat{H} T)(\hat{H} T)^T+\lambda I\right]^{-1}$ ，公式（6）转化为：

$$
\begin{aligned}
& \mathrm{h}=2 * z(\hat{H} T)^T\left[(\hat{H} T)(\hat{H} T)^T+\lambda I\right]^{-1}=2 * z \mathrm{U} \\
& \mathrm{~J}=-z \operatorname{eroDiag}\left[(\hat{H} T)^T\left[(\hat{H} T)(\hat{H} T)^T+\lambda I\right]^{-1}(\hat{H} T)\right]=-z \operatorname{eroDiag}[\mathrm{U}(\hat{H} T)]
\end{aligned}
$$
在该方案中，当 SNR＝10，$\lambda=30$ ；当 SNR＝15，$\lambda=10$ ；当 SNR＝20，$\lambda=5$ ；否找， $\lambda=200 / S N R$ 。

另外，与其他方案不同，该方案最终的结果是计算每个自旋的平均值的符号得到的，而不是最低能量的自旋配置的符号。

#### 2．网格搜索优化 bSB 算法参数：提高小迭代轮次下信号检测性能和检测速度

bSB 的更新规则为［7］
$$
\begin{aligned}
& y_i\left(t_{k+1}\right)=y_i\left(t_k\right)+\left\{-\left[a_0-a\left(t_k\right)\right] x_i\left(t_k\right)+c_0 \sum_{j=1}^N J_{i, j} x_j\left(t_k\right)\right\} \Delta_t \\
& x_i\left(t_{k+1}\right)=x_i\left(t_k\right)+a_0 y_i\left(t_{k+1}\right) \Delta_t
\end{aligned}
$$

如果 $\left|x_i\right|>1$ ，则令 $x_i=\operatorname{sign}\left(x_i\right), y_i=0$ 。

另外， bSB 的原始论文中给出了 $c_0$ 的计算方法［7］，即
$$
c_0=\frac{0.5}{\langle J\rangle \sqrt{N}},\langle J\rangle=\sqrt{\frac{\sum_{i, j} J_{i, j}^2}{N(N-1)}}
$$


原本 bSB 需要调整的参数为 $\Delta_t, a_0, c_0$ ，这里对 $c_0$ 进行缩放，令

$$
c=w * c_0
$$


因此，需要调整的参数为 $\Delta_t, a_0, w$ 。
优势：使用网格搜索进行参数优化，虽然网络搜索计算量随参数个数指数级增加，但是此处只有 3 个参数，且可以使用 GPU 加速计算。

#### 3．bSB 更新规则研究

- 旧：更新 $x_i$ 和 $y_i$ ，检查是否 $\left|x_i\right|>1$ ，如是，则替换 $x_i$ 为 $\operatorname{sgn}\left(x_i\right)$ ，并设置 $y_i$ 为 0 。
- 新：更新 $x_i$ ，检查是否 $\left|x_i\right|>1$ ，如是，则替换 $x_i$ 为 $\operatorname{sgn}\left(x_i\right)$ ；更新 $y_i$ ，检查是否 $\left|x_i\right|>1$ ，如是，设置 $y_i$ 为 0 。



### 3.1.3 结果

**1．LMMSE－like 建模方式优化后性能与 baseline 对比情况**

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925155630479.png" alt="image-20250925155630479" style="zoom:67%;" />

图 6：使用 LMMSE-like 矩阵前后，误码率随信噪比变化情况

![image-20250925155738760](https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925155738760.png)

从图中可以看出，在各个条件下，误码率相对于 baseline 都是有所下降的。对于误码平层问题，只有在 16－QAM 的条件下有所改进，在 64－QAM 和 256－QAM 条件下，仍然存在该问题。

**2．参数优化前后，误码率随总迭代轮次变化情况**

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925160157570.png" alt="image-20250925160157570" style="zoom:50%;" />

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925160208781.png" alt="image-20250925160208781" style="zoom:50%;" />

图 7：在信噪比为 20 的条件下，bSB 算法参数优化前后，误码率随总迭代次数变化情况

网格搜索进行参数优化后，误码率在小迭代次数下确实更优，随着迭代次数增大，误码率反而就比不过参数优化前的。

**3．参数优化前后，一次求解过程误码率随迭代步数变化情况**

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925160327993.png" alt="image-20250925160327993" style="zoom:80%;" />

图 8：在信噪比为 20 的条件下，bSB 算法参数优化前后，在一次求解过程中，误码率随迭代步数变化情况

参数优化后，收玫速度更快，但是最终结果并没有比未优化要好。

综合结果2和3可以得到，使用网格搜索进行参数优化只对小迭代轮次有效，而且虽然提高了收敛速度但是损失了一定的检测精度。

### 3.1.4 总结

使用 LMMSE－like 矩阵改进建模后，误码率在各信噪比条件下均有所下降，但在高阶调制（如 64－ QAM 和 256－QAM）下仍存在误码平层问题。在参数优化方面，通过网格搜索在小迭代次数下提升了性能并加快了收玫速度，但随着迭代次数增加，优化后的误码率表现反而不如优化前，表明优化在精度上有所牺牲。

## 3.2 方案二（ $1+1$ 团队）

### 3.2.1 现有方法

- 模拟分叉算法：aSB、bSB、dSB
- 相干伊辛机算法：CIM－CAC（混沌振幅控制）、CIM－CFC（混沌反馈控制）、CIM－SFC（分离反馈控制）
－局部量子退火：LQA
考虑一个系统在时间依赖的哈密顿量下演化［11］，如

$$
H(t)=t \gamma H_z-(1-t) H_x
$$


其中，$\gamma$ 控制目标哈密顿量 $H_z$ 在总哈密顿量中的能量比例，$H_z=\sum_{i j} J_{i j} \sigma_z^{(i)} \sigma_z^{(j)}, H_x=\sum_i \sigma_x^{(i)}$

限制量子态为直积态 $|\theta\rangle=\left|\theta_1\right\rangle \otimes\left|\theta_2\right\rangle \otimes \cdots \otimes\left|\theta_n\right\rangle$ ，其中 $\left|\theta_i\right\rangle=\cos \frac{\theta_i}{2}|+\rangle+\sin \frac{\theta_i}{2}|-\rangle$ 。因此，$\left\langle\theta_i\right| \sigma_z\left|\theta_i\right\rangle=\sin \theta_i,\left\langle\theta_i\right| \sigma_x\left|\theta_i\right\rangle=\cos \theta_i$ 。

LQA 的代价函数定义为：

$$
\mathrm{C}(t, \theta)=\langle\theta| H(t)|\theta\rangle=t \gamma \sum_{i, j} J_{i j} \sin \theta_i \sin \theta_j-(1-t) \sum_i \cos \theta_i
$$


令 $\theta_i=\frac{\pi}{2} \tanh w_i$ ，使得 $w_i \rightarrow \pm \infty \Longrightarrow\left|\theta_i\right\rangle \rightarrow|0\rangle,|1\rangle$ 。则代价函数为

$$
\begin{aligned}
C(w, t) & =t \gamma z^T J z-(1-t) x^T \cdot 1 \\
z & =\left(\sin \left(\frac{\pi}{2} \tanh w_1\right), \ldots, \sin \left(\frac{\pi}{2} \tanh w_n\right)\right)^T \\
x & =\left(\cos \left(\frac{\pi}{2} \tanh w_1\right), \ldots, \cos \left(\frac{\pi}{2} \tanh w_n\right)\right)^T
\end{aligned}
$$


其中，$t$ 是退火过程中从 0 到 1 变化的时间参数，$J_{i j}$ 描述了系统中第 $i$ 和第 $j$ 个元素之间的相互作用强度。

对参数 $w$ 的梯度为：
$$
\nabla_w C(w, t)=\left[\frac{\pi}{2} t \gamma(2 J z) \circ x+(1-t) \frac{\pi}{2} z\right] \circ a(w)
$$


其中，$\circ$ 表示向量的元素乘法，$a(\cdot)=1-\tanh ^2(\cdot)$ 是 $\tanh$ 函数的导数，且其作用于元素。
所以，参数更新公式为：

$$
\begin{aligned}
& \nu \leftarrow \mu \nu-\eta \nabla_w C(w, t) \\
& w \leftarrow w+\nu
\end{aligned}
$$

其中，$\mu \in[0,1]$ 是应用于动量变量 $\nu$ 的动量参数，$\eta$ 是梯度下降的步长。

### 3.2.2 改进：基于 LQA（参数优化、初始状态优化、代码细节优化、稀疏矩阵优化）

**1．超参数优化**

对于给定数据集，设置超参数 $d t=0.4 、 \gamma=0.05 、 \beta_1=0.6 、 \beta_2=0.999$ ，迭代步数从 100步减少到 20 步，求解效果提升，求解时间缩短。

其中，$d t$ 是时间步长，$\gamma$ 是 J 参数的权重比例，$\beta_1 、 \beta_2$ 是 Adam 的超参数，$\beta_1$ 决定过去梯度信息在当前的权重，$\beta_2$ 决定过去梯度平方信息在当前的权重。

**2．初始状态优化**

- 随机分布初始解

从均匀分布中随机抽样，为每个维度赋予一个值，实现解的随机初始化。

- 线性间隔点广播高维空间初始化解

在 $[-1, ~ 1]$ 范围内生成一系列线性间隔点，通过广播扩展到更高维度。

- Sobol 序列初始化解

利用 Sobol 序列，在高维空间中生成更为均匀分布的点。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925160901126.png" alt="image-20250925160901126" style="zoom:80%;" />

图 9：不同初始化方案实验结果。蓝色为随机初始化，绿色为线性间隔点广播初始化，红色为 Sobol序列初始化

**3．代码细节优化**

代码重构、并行计算、重复计算合并。

**4．稀疏矩阵优化**

在处理大规模的 Ising 问题时，可以使用稀疏矩阵编码（CSR）。但是对于 MIMO 问题，规模相对较小，矩阵本身非稀疏矩阵，所以稀疏矩阵编码是不必要的，因此将算法中的稀疏矩阵操作 csr＿matrix（•）删除，提升计算速度。

**5．展望：AI 去噪声优化**

存在求解得到的能量更低，但是误码率更高的问题，认为当噪声较大时，单纯提高求解器的性能并不能提升求解效果上限。

因此，考虑利用 AI 去噪优化
- 扩散模型（Diffusion Models）：使用 Diffusion Models 学习噪声分布，实现去噪功能。
- 多模态匹配：对于 MIMO 问题，发送的信息不是无序的，而是符合语言逻辑的，考虑在得到求解结果后，利用多模态匹配方法，修正发送的信息，降低误码率。

### 3.2.3 结果

（前面介绍的是 LQA，此处应该是基于 LQA 的优化）

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925161033049.png" alt="image-20250925161033049" style="zoom:67%;" />

### 3．2．4 总结

该方案主要通过超参数优化（如调整时间步长 $d t$ 、权重比例 $\gamma$ 及 Adam 参数）提升了求解效率，并将迭代步数从 100 步减少到 20 步，同时使用 Sobol 序列初始化方法替代了原本的随机均匀初始化，进一步提高了求解精度。此外，针对问题规模较小的特点，去除了不必要的稀疏矩阵操作，并通过代码重构、并行计算和重复计算合并等细节优化，进一步提升了计算效率。

## 3.3 方案三（BatchLions 团队）

### 3．3．1 现有方法及问题

- Sphere Detector 等最大似然方法：检测性能优秀，但是复杂性呈指数增长，且只适用于小规模的 MIMO 检测问题。
- Zero Forcing（ZF），Minimal Mean Square Error（MMSE）等线性检测器：复杂性低，但在大规模 MIMO 检测问题上性能较差。
- 量子退火启发式算法（QAIA）：CIM 类算法、SB 类算法、LQA

### 3．3．2 改进：基于 LQA（确定性初始条件）

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925165824484.png" alt="image-20250925165824484" style="zoom:50%;" />

图 11：QAIA 算法实验，确定基线算法

**1．包含 SNR 信息的初始条件：确定的初始单一条件替代随机生成的批量初始条件**

信噪比的表达式为
$$
\mathrm{SNR}=10 \log _{10} \frac{n}{\sigma_w^2}
$$
其中，$n$ 表示信号功率，$\sigma_w^2$ 表示噪声强度。
该方案通过近似和推导得到

$$
\mathbf{s}_0=\frac{1}{r_b}\left[\begin{array}{c}
\left(\tilde{\mathbf{x}}+(\sqrt{M}-1) \mathbf{1}_N\right) / 2^{r_b-1} \\
\left(\tilde{\mathbf{x}}+(\sqrt{M}-1) \mathbf{1}_N\right) / 2^{r_b-2} \\
\vdots \\
\left(\tilde{\mathbf{x}}+(\sqrt{M}-1) \mathbf{1}_N\right) / 2^0
\end{array}\right]
$$


其中，$\tilde{\mathbf{x}}=\frac{\tilde{\mathbf{y}}}{\sqrt{1+10^{-\frac{\text { SNR }}{10}}}}$ 。该方法针对的是比赛数据集发射天线规模等于接收天线规模的情况。引入确定性初始条件的作用：
- 初始条件选择合理，则可以避免陷入局部最优解，而且 BER 也可以降低
- 可以使得运行时间缩短
- 确定性替代随机性：初始条件不再包含 batch＿size 信息，即对该初始条件，只进行一次计算，另外，确定性的初始条件避免了耗时的随机数计算，都缩短了运行时间。

其他可能初始条件形式：
－从 $y=H x+n$ 中忽略噪声 w ，反解出 $x=H^{-1} y$ 。但是这个需要进行耗时的矩阵求逆和矩阵乘操作。
－应用机器学习确定初始条件：最好的初始条件是原始传输信号 $\times$ 对应的自旋向量 s ，训练集是包含 x 的，那么，可以从训练集中的原始传输信号 x 对应的自旋向量 s 出发，对 s 进行回归，确定一个近似的初始条件表达式。如果数据量够大，也可以尝试直接从 x 中学习出一个表法式。

难点：不同的数据点 s 的长度不同，回归函数的输出维度需要随着 s 的长度而变化，函数可选择的范围受到一定的限制。

### 3．3．3 结果

#### 1．参数初始调整

**a．Adam 优化器**

在同个batch＿size下，通过固定三个参数的其中两个，调整剩下的一个参数，观测得分结果。最终，当batch＿size＝10，n＿iter＝65，dt＝1，$\gamma=0.03$ 时，测试集分数为 6.1651

**b．Momentum 优化器**

同理，当batch＿size $=10$ ，momentum $=0.85, \mathrm{n} \_$iter $=60, \mathrm{dt}=1, \gamma=0.01$ 时，测试集分数接近 7

#### 2．Momentum 优化器＋确定性初始条件

momentum $=0.85, \mathrm{n} \_$iter $=72, \mathrm{dt}=1$ 时，测试集分数超过 17 ，相对于随机生成的初始化条件有了明显提高

#### 3．Adam 优化器 + 确定性初始条件

n ＿iter $=72, \mathrm{dt}=1, ~ \beta_1=0.9, ~ \beta_2=0.999, ~ \epsilon=10 e^{-8}, ~ g a m m a=0.026$ 时，测试集分数超过
了 21 ，相对于 Momentum 优化器提高 $23 \%$ 。
因此，改变优化器可以缩短收玫时间，降低迭代次数，从而提高分数。

#### 4．Adam 优化器 + batch＿size $=1$ 的随机初始条件

相较于结果 4，令 batch＿size $=1$ ，n＿iter $=49$ 时，单个随机生成的初始条件的测试集得分与确定初始条件的测试集得分相近。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925170136019.png" alt="image-20250925170136019" style="zoom:50%;" />

图 12：Adam 优化器 + 确定／随机初始条件计算结果比较

综合结果4和5，确定性初始条件的优化方法并没有使得 BER 有太大变化。

### 3．3．4 总结

确定性初始条件优化通过引入包含SNR信息的确定性初始条件，提升了测试集分数（Adam优化器下分数超过21），同时缩短了运行时间。实验表明，Adam优化器在确定性初始条件下得分优于批量随机初始化，但是其对于单个随机生成的初始条件，改进有限，可以得出产生较高得分的原因是降低了大batch＿size情况下生成随机数消耗的时间。另外，根据其初始化公式，该方案只适用于发射天线数等于接收天线数的规模。

## 3.4 方案四（淡黄的长群团队）

### 3．4．1 问题分析

经过原代码测试，发现存在一些问题：

- 数据集中包含的 SNR 信息未被利用。
- 不同 QAIA 的效果有显著差异。
- 在求解器函数中，并没有做最优解的选择，因为传入的解只有一个。

### 3．4．2 改进：基于 LQA 和 bSB（考虑噪声信息、正则化输入、LMMSE－like 输入、 MLD 后处理）

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925170339312.png" alt="image-20250925170339312" style="zoom:50%;" />

图 13：方案四改进流程

- 输入优化：类 MMSE 输入优化，相较于 baseline，运行时间提增加 30 倍，但 BER 值降低 $25 \%$ 。而1和2的优化并没有特别好的提升。
- 算法优化：基于少样本和低迭代情形的表现，该输入优化方法可以作为子过程在预处理过程中起到加速作用，快速获得较为准确的解。

**1．考虑噪声信息：baseline 代码测试（无显著影响）**

蓝色的是时间，绿色的是 BER，都是越低越好。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925170442505.png" alt="image-20250925170442505" style="zoom:67%;" />

图 14：在不同信噪比下，不同 QAIA 实验结果。固定采样次数为 1 ，循环次数为 10 ，蓝色柱状表示时间，绿色柱状表示误码率 BER

结论：在信噪比不同的情况下，各个算法的表现趋势几乎没差别。

**2．正则化输入模型［1］（无显著影响）**

原始的 Ising 模型为
$$
\hat{s}=\min _{s \in\{-1,1\}^N}-h^T s-s^T J s
$$


加入惩罚项后，Ising 模型变为［1］

$$
\hat{s}=\min _{s \in\{-1,1\}^N}-\left(h+2 \cdot r \cdot s_p\right)^T s-s^T J s
$$

其中，$r$ 是可调的正则化参数，$s_p$ 是惩罚项。
设置采样次数为 1 ，循环次数为 5 ，通过 LQA 或 bSB 得到 $s_p$ ，将其加入输入矩阵 $h$ 。然后，固定采样次数为 1 ，循环次数为 10 ，运行 LQA 和 bSB 得到结果，图中蓝色为运行时间，绿色为 BER，左边为正则化结果，右边为 baseline 结果。从图中可以看出，在付出一定时间代价后，并未显著降低 BER。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925170543700.png" alt="image-20250925170543700" style="zoom:67%;" />

图 15：加入正则化项前后，bSB、LQA 分别与 baseline 结果。蓝色为运行时间，绿色为 BER结论：加入正则化项，并没有显著改进。

3．LMMSE－like 输入的优化［4］
原始的耦合矩阵 $J$ 和外场项 $h$ 为

$$
h=2 \cdot z^T \widetilde{H} T, \quad J=-2 \cdot T^T \widetilde{H}^T \widetilde{H} T
$$


加入 LMMSE－like 矩阵 $U_\lambda=\widetilde{H} \widetilde{H}^T+\lambda I, \lambda$ 是可调的非负参数，耦合矩阵 $J$ 和外场项 $h$ 修改为

$$
h=2 \cdot z^T U_\lambda^{-1} \widetilde{H} T, \quad J=-2 \cdot T^T \widetilde{H}^T U_\lambda^{-1} \widetilde{H} T
$$
固定采样次数为 1 ，循环次数为 $10, ~ \lambda=10$ ，运行 LQA 和 bSB 得到结果，图中蓝色为运行时间，绿色为 BER，左边为类 MMSE 结果，右边为 baseline 结果，可以发现，LQA 的效果并没有得到提升，而 bSB 虽然增加了求逆运算的时间，但是 BER 降低了 $25 \%$ 。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925170635284.png" alt="image-20250925170635284" style="zoom:67%;" />

图 16：加入 LMMSE－like 矩阵前后，bSB、LQA 分别与 baseline 结果。蓝色为运行时间，绿色为 BER

- 参数选择：参数 $\lambda$ 、采样次数和循环次数。参数 $\lambda=10$ 是在固定采样次数为 1 ，循环次数为 10 的条件下得到的最优值，然后改变采样次数和循环次数，付出了更多的时间，但并未显著降低 BER。

**4．少样本和低迭代条件下，量子启发式算法预处理＋MLD**

固定采样次数为 1 ，循环次数为 10 ，搜索距离为 1 ，对量子启发式算法求解得到的结果，使用 MLD 求解器进行后处理，BER 值得到了降低。图中蓝色为无 MLD 后处理，绿色为有 MLD 后处理。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925170715721.png" alt="image-20250925170715721" style="zoom:50%;" />

图 17：有无 MLD 后处理结果对比。蓝色为无 MLD 后处理 BER，绿色为有 MLD 后处理 BER

### 3．4．3 展望

- 深度学习自动调参
- 如何有效结合噪声信息
- 采用不同启发式算法的结果，通过选择算法结果获得最优解

### 3．4．4 总结

首先，比赛数据集中，噪声信息的考虑对求解几乎没用影响。其次，正则项的加入未带来显著改进，可能是LQA与bSB算法能以较少的迭代次数到达它们能给出的最优解附近，所以即使加入了估计的解作为惩罚项也难以获得更大的提升。而LMMSE－like矩阵优化建模虽然付出了一部分时间代价，但是显著降低了 bSB的误码率，对于LQA则不起作用。此外，在少样本和低迭代条件下，结合量子启发式算法预处理和MLD后处理进一步降低了 BER。总体而言，LMMSE－like建模优化和MLD后处理在提升性能方面效果显著，但需权衡时间成本。

## 3.5 方案五（孤电子队伍）

### 3．5．1 现有方法及问题

- 极大似然检测器：NP－hard
- QAIA：
  - SB：计算资源所需较大；容易陷入局部最优
  - CIM：实际体系耗损强，无法大规模并行
  - SimCIM：大规模并行，损耗较小且不容易陷入局部最优

相干伊辛机（CIM）是一种光学量子计算的模型，它通过光脉冲模拟伊辛自旋模型，并利用系统的物理特性（如脉冲间的相互作用）来找到问题的最优解。SIMCIM 则通过数值方法模拟这一过程。

在 SimCIM 中，每个脉冲 $a_i=\frac{1}{\sqrt{2}}\left(x_i+i p_i\right)$ 服从以下随机微分方程［8］：

$$
\begin{aligned}
& \Delta x_i=w x_i-\gamma x_i-s\left(x_i^2+p_i^2\right) x_i+\zeta \sum_j J_{i j} x_j+\operatorname{Re}\left(f_i\right) \\
& \Delta p_i=-w p_i-\gamma p_i-s\left(x_i^2+p_i^2\right) p_i+\operatorname{Im}\left(f_i\right)
\end{aligned}
$$


为了简化计算，系统忽略了非线性损失项，将分析限制在实数范围内：

$$
\begin{aligned}
\Delta x_i & =v x_i+\zeta \sum_j J_{i j} x_j+f_i \\
v & =w-\gamma
\end{aligned}
$$
其中，$v$ 是泵浦增益－损失因子差值，取决于 $\left[-v_{\text {bound }}, v_{\text {bound }}\right] ; \zeta$ 是耦合强度系数，$J_{i j}$ 是耦合矩阵，描述了系统中自旋之间的相互作用；$f_i$ 是外部噪声项，可选用高斯噪声、均匀噪声、泊松噪声、拉普拉斯噪声和卡方噪声进行模拟随机干扰。。

为了解决自旋系统中的振幅增长问题，防止系统不稳定，引入饱和机制来限制自旋值：

$$
\begin{aligned}
x_i & \leftarrow \phi\left(x_i+\Delta x_i\right) \\
\phi(x) & = \begin{cases}x, & \text { if }|x| \leq x_{\text {sat }} \\
x_{\text {sat }}, & \text { otherwise }\end{cases}
\end{aligned}
$$


使用 Momentum，在梯度更新时，不仅考虑当前梯度，还考虑之前梯度的累积效果，加快收敛速度，减少震荡现象：

$$
\begin{aligned}
v_t & =\beta v_{t-1}+(1-\beta) \nabla_\theta L\left(\theta_t\right) \\
\theta_{t+1} & =\theta_t-\alpha v_t
\end{aligned}
$$

### 3．5．2 改进：基于 SimCIM（带位置编码的自适应矩估计）

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925170910856.png" style="zoom:67%;" />

图 18：方案五，带位置编码的自适应矩估计的 SimCIM 流程

1．位置编码
高斯噪声对于 Ising 问题的转化可能存在对齐问题，因此引入先验知识，改进去噪过程。在实际无线传输过程中，发送和接收天线相对位置不会发生巨大变化，因此在转化 Ising 模型阶段加入天线位置进行编码，从而区分来自不同天线端的信号，加速收敛。误码率下降 10％。

2．自适应矩估计（Adam）
3．SimCIM 调参：PSO 优化

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171043012.png" alt="image-20250925171043012" style="zoom:67%;" />

图 19：PSO 可视化解释

粒子群优化（Particle Swarm Optimization，PSO）通过模拟鸟群觅食行为，利用个体间的信息共享，实现对解空间的全局搜索。通过惯性部分、自我认知部分和社会认知部分共同作用于粒子的速度更新，使得粒子能够在搜索空间中以一种动态的方式移动。通过调整各个成分的权重和随机因素， PSO 算法能够在全局搜索和局部细化之间取得平衡，从而有效地寻找最优解。

具体来说，每个优化问题的解都是搜索空间中的一只鸟，称之为＂粒子＂。所有的粒子都有一个由被优化的函数决定的适应值，每个粒子还有一个速度决定他们飞翔的方向和距离。然后粒子们就追随当前的最优粒子在解空间中搜索。在每一次迭代中，粒子通过跟踪两个＂极值＂来更新自己。第一个就是粒子本身所找到的最优解，这个极值叫做个体极值 pbest。另一个极值是整个种群目前找到的最优解，这个极值是全局极值 gbest。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171121263.png" alt="image-20250925171121263" style="zoom:50%;" />

图 20：改进后结果。时间更短，误码率也更低

**4. FPGA 产业尝试**

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171148519.png" alt="image-20250925171148519" style="zoom:67%;" />

### 3．5．3 总结

基于SimCIM的改进方案通过引入位置编码和自适应矩估计（Adam），误码率降低了 $10 \%$ 。位置编码通过利用天线位置的先验知识，加速了收敛过程；自适应矩阵估计结合Adam优化进一步优化了矩阵估计精度。此外，采用粒子群优化（PSO）进行参数调优，提升了算法效率。实验结果表明，改进后的方案在时间和误码率上均优于原方法。同时，结合FPGA的尝试展现了硬件加速的潜力，为未来实际应用提供了方向。

## 3.6 方案六（请再努力一些团队）

### 3．6．1 改进：基于 SimCIM（改进贝叶斯优化超参数调优）

1．贝叶斯优化超参数调优

- 参数设置
- 改进贝叶斯优化超参数调优流程

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171233466.png" alt="image-20250925171233466" style="zoom:67%;" />

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171254163.png" alt="image-20250925171254163" style="zoom:80%;" />

在该方案中，应用帕尔逊树结构估计器（TPE）来建立代理模型，其算法基于贝叶斯模型展开，并且可以与期望改进（Expected Improvement，EI）结合使用进行超参数选取。

在TPE贝叶斯优化过程中，算法通过概率代理模型对问题进行拟合，减少了目标函数的调用次数。然而，每次迭代过程中，算法可能将无用点作为信息点进行评估，产生额外计算并干扰TPE代理模型的学习过程，影响算法性能。此外，不同模型参数的取值范围不同，例如学习率等参数在 $[0,1]$ 区间内，均匀抽样可能导致参数集中在较高区间（如 $[0.1,1]$ ），这种采样方式存在缺陷。

针对上述问题，该方案引入Optuna框架对TPE贝叶斯优化过程进行优化。Optuna是一种机器学习参数优化框架，能够动态构建参数搜索空间，并通过剪枝策略提升优化效率。

经过此步处理，对于学习率等参数值的取样过程将更加符合模型的需求范围。在此之后构建 Optuna 的相对搜索空间，在相对搜索空间内的参数进行相对采样，并将参数返回试验对象。在目标函数执行过程中，如果目标参数在相对搜索空间内，则通过试验对象返回参数，并通过 TPE 采样产生下一组评估参数，否则对搜索空间外的参数进行独立采样，并返回抽取的参数。最后通过目标函数计算对应的目标值，得到此次试验的结果。为了保留TPE优化过程更有效的历史信息点，减少无用的评估消耗，该方案引入剪枝策略对无望的试验过程进行裁剪。

### 3．6．2 结果

1．基准结果

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171337940.png" alt="image-20250925171337940" style="zoom:80%;" />

实验次数为 $30, \mathrm{bSB}$ 在最差平均误码率上是最好的，但是运行时间较长，LQA 各指标比较均衡， SimCIM 在几类 CIM 算法中表现最佳。

后续基于 LQA 和 SimCIM 进行实验。

**2．超参数调优结果**

实验次数为 70。分别对使用改进贝叶斯优化和随机搜索的 LQA 和 SimCIM 进行对比。

- 改进贝叶斯优化超参数调优

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171426302.png" alt="image-20250925171426302" style="zoom:80%;" />

结果：改进贝叶斯优化与随机搜索在 70 次实验中寻优效果接近，但改进贝叶斯优化算法用时更少。

**3．最优超参数下算法结果**

实验次数为 50 。

- 默认参数下算法运行结果

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171527585.png" alt="image-20250925171527585" style="zoom:50%;" />

- 最优参数下算法运行结果

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171536707.png" alt="image-20250925171536707" style="zoom:50%;" />

 结果：使用改进贝叶斯优化算法超参数调优后，SimCIM 算法表现效果改善较大，且整体表现优于 LQA 算法。

**4．$v_{\text {bound }}$（泵浦－损耗因子列表 $v$ 的上下界）、 $x_{s a t}$（振幅阈值）进行基于独立采样的超参数调优结果**

- $v_{\text {bound }}$ 结果， $\mathbf{5 0}$ 次实验 

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171647447.png" alt="image-20250925171647447" style="zoom:50%;" />

- $x_{s a t}$ 结果， 50 次实验，与默认方案无显著差异

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171659329.png" alt="image-20250925171659329" style="zoom:50%;" />

- 噪声类型选择结果，50 次实验，均匀噪声表现最佳

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171721999.png" alt="image-20250925171721999" style="zoom:50%;" />

### 3．6．3 展望

- 在 SimCIM 中加入前面省略的非线性项，希望在提高算法精度的同时，效率不会减少
- 引入树搜索算法对 ML－MIMO 问题降维后转化为 Ising 模型，并使用 SimCIM 求解

### 3．6．4 总结

基于SimCIM的改进方案通过TPE贝叶斯优化方法结合Optuna实现进行超参数调优，结合剪枝策略和尽早终止机制，显著减少了调优时间，同时保持了与随机搜索相近的优化效果。实验表明，改进贝叶斯优化在 70 次实验中用时更少，且在最优参数下，SimCIM算法的性能显著提升，整体表现优于LQA算法。此外，通过对 $v_{\text {bound }}$（泵浦－损耗因子列表 $v$ 的上下界）、 $x_{\text {sat }}$（振幅阈值）的独立采样调优，进一步优化了参数选择，其中均匀噪声表现最佳。

## 3.7 方案七（Quiscus 团队）

### 3．7．1 改进：基于 DUSB（LMMSE－like 矩阵、正则项可学习程度、效率优化）

在高阶调制下，原本的耦合矩阵 J 和外场项 h 计算公式为
$$
\begin{aligned}
& \mathrm{h}=2 * z^T \hat{H} T \\
& \mathrm{~J}=-z \operatorname{eroDiag}\left[(\hat{H} T)^T \hat{H} T\right]
\end{aligned}
$$

**1．引入正则项 $U_\lambda$（LMMSE－like 矩阵）**

- 和方案一类似，引入 $\mathrm{U}_\lambda=\left(\mathrm{HH}^T+\lambda I\right)^{-1}$ ，但是不起作用
- 改为 $\mathrm{U}_\lambda=\left(\mathrm{HH}^T+\lambda I\right)^{-1} / \lambda$

$$
\begin{aligned}
& \mathrm{h}=2 * z^T U_\lambda \hat{H} T \\
& \mathrm{~J}=-z \operatorname{eroDiag}\left[(\hat{H} T)^T U_\lambda \hat{H} T\right]
\end{aligned}
$$

**2．深度展开自动调参**

可调参数：正则化 $U_\lambda$ 中的系数 $\lambda, \mathrm{SB}$ 算法中的步长参数 $\Delta_k, c_0$ 的系数 $\eta$

**3．pReg－LM－SB 系列方法**

基于 DU－LM－SB，扩展参数正则项的可学习程度：
－DU－LM－SB： $\mathrm{U}_\lambda=\left(\mathrm{HH}^T+\lambda I\right)^{-1} / \lambda$
－pReg－LM－SB： $\mathrm{U}_\lambda=\left(\mathrm{HH}^T+A A^T\right)^{-1} / \lambda, A$ 为对角阵，且针对每个规模都训练一个。比如，给定数据集中，天线数为 64、128，所以 pReg－LM－SB 代码实现中设定了可训练参数 $A \_64$ ，$A \_128$
- ppReg－LM－SB： $\mathrm{U}_\lambda=A, A$ 为对角阵，且针对每个 Nr 规模都训练一个
- pppReg－LM－SB： $\mathrm{U}_\lambda=A, A$ 为对角阵，且训练数量为不同的 Nr 数量＊不同的 snr 数量

**4．运行效率优化**

- 归一化操作使用了更简洁的形式。对于 M－QAM，$q a m=\frac{2}{3}(M-1)$
- 使用深度展开技术减少 SB 算法迭代轮次，训练完成后，在 SB 算法中使用训练出的参数时设置 batch＿size＝1

- 使用稠密矩阵表示，因为信道矩阵 H 并不稀疏，和方案二的优化一致
- 缓存频繁访问的中间结果和辅助数据
- 借助矩阵乘法结合律，调整矩阵运算顺序以最小化计算量
- 使用近似运算求矩阵的逆，借助诺伊曼级数递归近似

![image-20250925171938858](https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171938858.png)

### 3．7．2 结果

1．传统方法和量子退火启发式算法结果

- 基于传统方法的对比算法

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925171951661.png" alt="image-20250925171951661" style="zoom:50%;" />

- 基于量子退火启发式算法的对比算法

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925172011276.png" alt="image-20250925172011276" style="zoom:50%;" />

2. 优化后结果

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925172053943.png" alt="image-20250925172053943" style="zoom:67%;" />

### 3．7．3 总结

基于DUSB的改进方案通过引入LMMSE－like矩阵改进建模和深度展开自动调参，优化了性能。具体改进包括：1）引入LMMSE－like正则项并调整其形式，使其在给定数据集下有效；2）通过深度展开技术自动调参，优化了正则化系数；3）提出pReg－LM－SB系列方法，扩展了正则项的可学习程度，针对不同天线数和信噪比（SNR）规模训练特定参数；4）通过归一化简化、稠密矩阵表示、矩阵运算顺序调整和诺伊曼级数递归近似求逆等方法，显著提升了运行效率。



# 四、总结



## 4.1 基准算法总结

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925172146164.png" alt="image-20250925172146164" style="zoom:50%;" />

在以上 QAIA 算法中，LQA、bSB、SimCIM 表现相对较好，上述七个方案基于这三个算法进行改进。

## 4.2 改进点总结

1．预处理：位置编码
2．建模：LMMSE－like 矩阵、正则项
3．计算效率：归一化、求逆近似、代码优化
4．初始化：Sobol 初始化、包含 SNR 的确定性初始条件
5．算法：深度展开
6．调参：网格搜索、改进贝叶斯优化调参、PSO调参
7．后处理：MLD

## 4.3 数值结果

比赛中提供的数据为发射天线和接收天线数量一致，而目前这种配置在低阶建模时存在误码平层问题，即误码率在一个特定值附近波动，难以进一步降低，因此，为了更准确地评估各方案的性能，我们在实验时选择使用发射天线小于接收天线的配置。另外，我们选择分别测试多个SNR下的误码率，而不是一次性测量所有不同数据并取平均值，这样可以体现这些方法在不同SNR条件下的适应性。比赛的成绩评判主要基于求解时间和误码率，而本文我们只考虑误码率。

在本次比赛中，各参赛团队提出了多种方法以改进量子退火启发式算法从而解决MIMO检测中的挑战。方案一（灿言团队）引入了结合LMMSE－like矩阵和网格搜索参数优化的bSB算法；方案二（ $1+1$团队）则采用了经过参数和代码优化，并Sobol序列初始化的LQA算法；方案三（BatchLions团队）开发了一种基于SNR的确定性初始条件的LQA算法；方案四（淡黄的长裙团队）提出了融合正则化输入、 LMMSE－like矩阵以及MLD后处理的bSB和LQA算法；方案五（孤电子队伍）设计了基于位置编码的自适应矩阵估计的SimCIM算法；方案六（请再努力一些团队）采用基于改进贝叶斯调参的SimCIM算法；方案七（Quiscus团队）则是选择使用DUSB算法，同样加入LMMSE－like矩阵并对其学习程度进行优化，同时引入近似运算求解矩阵的逆的方法，借助诺伊曼级数递归技术，减少计算资源消耗。

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925172253727.png" alt="image-20250925172253727" style="zoom:50%;" />

<img src="https://jptanjing.oss-cn-beijing.aliyuncs.com/img/image-20250925172315202.png" alt="image-20250925172315202" style="zoom:50%;" />

我们对各参赛团队的解决方案进行了实验测试。设置迭代次数为 10 ，并考虑了天线规模为 $1: 2$ 或 $1: 1.5$ 的配置，具体选择的是 $16 \times 32$ 和 $32 \times 48$ 的 16 －QAM调制方式，以及 $16 \times 32$ 的 64 －QAM调制方式，信噪比（SNR）设定为 $10 、 15 、 20 、 25$ 和 30 。需要注意的是，方案三基于包含 SNR 的确定性初始条件，其针对的是发射天线等于接收天线的场景，因此未包含在这组特定条件的测试中。

从上述结果中可以看出，在16－QAM条件下，方案一（引入了结合LMMSE－like矩阵和网格搜索参数优化的bSB算法）、方案四（提出了融合正则化输入、LMMSE－like矩阵以及MLD后处理的bSB和 LQA算法）和方案七（使用DUSB算法，同样加入LMMSE－like矩阵并对其学习程度进行优化，同时引入近似运算求解矩阵的逆的方法，借助诺伊曼级数递归技术）的BER较低，结果较好，而其他几个方案均陷入误码平层（Error floor）；在64－QAM条件下，方案一、方案七的BER整体较低。

参考文献
[1] M．Kim，D．Venturelli，and K．Jamieson，＂Leveraging quantum annealing for large mimo processing in centralized radio access networks，＂in Proceedings of the ACM Special Interest Group on Data Communication，2019，pp．241－255．
[2] M．Kim，S．Mandr｀a，D．Venturelli，and K．Jamieson，＂Physics－inspired heuristics for soft mimo detection in 5G new radio and beyond，＂in Proceedings of the 27th Annual International Conference on Mobile Computing and Networking，ser．MobiCom＇21．New York，NY，USA：Association for Computing Machinery, 2021, p. 42-55. [Online]. Available: https://doi.org/10.1145/ 3447993.3448619
[3] A. K. Singh, K. Jamieson, P. L. McMahon, and D. Venturelli, "Ising machines' dynamics and regularization for near-optimal mimo detection," IEEE Transactions on Wireless Communications, vol. 21, no. 12, pp. 11 080-11 094, 2022.
[4] Takabe S. "Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection," arXiv preprint arXiv:2306.16264, 2023.
[5] High-order Modulation Large MIMO Detector based on Physics-inspired Methods. (under review)
[6] Goto H, Tatsumura K, Dixon A R. Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems. Science advances, 2019, 5(4): eaav2372.
[7] Goto H, Endo K, Suzuki M, et al. High-performance combinatorial optimization based on classical mechanics. Science Advances, 2021, 7(6): eabe7953.
[8] Tiunov E S, Ulanov A E, Lvovsky A I. Annealing by simulating the coherent Ising machine. Optics express, 2019, 27(7): 10288-10295.
[9] Reifenstein S, Kako S, Khoyratee F, et al. Coherent Ising machines with optical error correction circuits. Advanced Quantum Technologies, 2021, 4(11): 2100077.
[10] Kanao T, Goto H. Simulated bifurcation assisted by thermal fluctuation. Communications Physics, 2022, 5(1): 153.
[11] Bowles J, Dauphin A, Huembeli P, et al. Quadratic unconstrained binary optimization via quantum-inspired annealing. Physical Review Applied, 2022, 18(3): 034016
[12] https://dsplog.com/2007/09/23/scaling-factor-in-qam/
[13] https://www.cs.princeton.edu/~kylej/talks/ieee-comsoc-2020a.pdf