# 使用量子启发式算法求解MIMO detection问题

## 问题背景

多输入多输出（MIMO）是满足当前和未来无线系统对大量数据流量需求的关键技术。在这种技术中，发射端和接收端都配备了多个天线，以改善基于多径传播的无线链路容量。然而，对于接收端来说，在存在噪声和干扰的情况下，根据信道状态信息和接收到的信号来重建传输的符号是一项具有挑战性的任务，被称为MIMO检测问题。寻找准确的解决方案，如最大似然检测器，已被证明是NP难问题。

近年来，研究人员开始尝试使用量子退火或量子启发式算法(Quantum Annealing Inspired Algorithms, QAIA)来解决MIMO检测问题(参见[1,2,3,4，MLD在投论文])。我们邀请选手在参考文献的基础上,将MIMO检测问题转化为量子启发式算法擅长的Ising自旋玻璃问题，并在建模和算法方面进行探索,设计基于量子启发的MIMO检测算法。

## 赛题说明

MIMO 检测中，$N_t$ 个用户使用单天线发送符号 $\mathbf{x}=[x_1,x_2,\dots, x_{N_t}]^\top\in\mathbb{C}^{ N_t}$ 来自星座 $\Omega$，然后由基站用 $N_r$ 天线接收符号。 接收到的符号可以表示为 $\mathbf{y}=[y_1, y_2,\dots, y_{N_r}]^\top\in\mathbb{C}^{N_r}=\mathbf{y}^R+j \mathbf{y}^I$。 $N_t$个用户天线和$N_r$个接收天线之间的传输特性总结为信道矩阵$\mathbf{H}\in\mathbb{C}^{N_r\times N_t}=\mathbf{H}^R+ j\mathbf{H}^I$。 最后，这个过程可以表述为
$$
\mathbf{y}=\mathbf{H}\mathbf{x}+\mathbf{n},
$$
其中$\mathbf{n}\in\mathbb{C}^{N_r}$表示加性高斯白噪声。 MIMO检测问题是在给定信道矩阵和接收符号的情况下尽可能重现传输符号。

![mimo_detection](graphs/mimo_detection.jpg)

极大似然检测器（Maximal Likelihood Detector）是在存在噪声情况下的最小化检测误差的MIMO检测方案，原理如下所示
$$
\hat{\mathbf{x}}_{\text{ML}}=\arg\min_{\mathbf{x}\in\Omega^{N_t}}\Vert\mathbf{y}-\mathbf{H} \mathbf{x}\Vert^2,
$$

量子启发式算法通过引入variable-to-symbol转化函数（详情参见文献[1]），将MLD问题转化为Ising模型，格式如下：
$$
   \min_{\mathbf{s}\in {\pm 1}^n} H(\mathbf{s}) = \sum_{1\leq i\leq j\leq N} J_{ij} s_i s_j.
$$

我们要求选手设计或参考编码方案，将MLD问题应建模为Ising问题，并用包括但不限于量子启发式算法（参考qaia）的技术求解。

## 实现

参考qaia中提供的算法, 从QAIA基类中继承, 并提出新的算法, 或者任意设计修改main.py文件中的ising_generator和qaia_mld_solver函数以实现求解, 同时juger.py中的代码无需修改.

## 评分指标

对于给定的MLD实例，计算其比特错误率：
$$
\text{BER} = \frac{|\mathbf{s}-\mathbf{\hat{s}}|}{2N_t\cdot\log_2|\Omega|}
$$
设各个实例BER算数平均值为$\overline{\text{BER}}$, 选手得分公式大致为
$$
\text{score} = (1 - \overline{\text{BER}})\times\frac{\text{baseline\_time}}{\text{running\_time}}.
$$

$\text{baseline\_time}$ 为基线方法运行时间, $\text{running\_time}$为选手代码运行时间.

## 数据集

比赛会提供150个训练数据(MLD_data/), 每个数据包含: 

* H: 信道矩阵;
* y: 接受信号;
* bits: 用二进制$\{-1, 1\}$表示的传输信号;
* num_bits_per_symbol: 每个信号所包含的比特数目(调制阶数);
* SNR: 信噪比;
* ZF_ber: 基线算法zero_forcing的结果, 选手开发的算法的结果应该好于这个结果;

选手可在训练集上构建Ising问题转化代码, 优化量子启发式算法, 甚至构建基于机器学习的调参算法等; 但最终会在测试集上测试选手的代码. 因此要求构建的求解算法有一定的泛化能力.

## 样例代码

- main.py: 顶层演示程序, 也包含了将MLD问题转化为Ising问题的示例代码;
- qaia/: 包含了QAIA基类, 以及继承自QAIA的各种量子启发式算法;
- judger.py：判题程序

## 判题程序

请按照指定的数据格式进行输出，判题系统会使用同样的判题程序juger.py


## 要求

1. 选手仅能将MIMO detection转化为Ising问题来进行求解, 即继承基类QAIA;
2. 程序总运行时间在1h内;
3. 所使用库可通过pip或者conda安装，不使用收费库

## 参考材料

[1]  M. Kim, D. Venturelli, and K. Jamieson, “Leveraging quantum annealing for large mimo processing in centralized radio access networks,” in Proceedings of the ACM Special Interest Group on Data Communication, 2019, pp. 241–255.

[2]  M. Kim, S. Mandr`a, D. Venturelli, and K. Jamieson, “Physics-inspired heuristics for soft mimo detection in 5G new radio and beyond,” in Proceedings of the 27th Annual International Conference on Mobile Computing and Networking, ser. MobiCom ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 42–55. [Online]. Available: https://doi.org/10.1145/3447993.3448619

[3]  A. K. Singh, K. Jamieson, P. L. McMahon, and D. Venturelli, “QUBO machines’ dynamics and regularization for near-optimal mimo detection,” IEEE Transactions on Wireless Communications, vol. 21, no. 12, pp. 11 080–11 094, 2022.

[4]  Takabe S. “Deep Unfolded Simulated Bifurcation for Massive MIMO Signal Detection,” arXiv preprint arXiv:2306.16264, 2023.

[5] High-order Modulation Large MIMO Detector based on Physics-inspired Methods. (under review)

[6] Goto H, Tatsumura K, Dixon A R. Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems. Science advances, 2019, 5(4): eaav2372.

[7] Goto H, Endo K, Suzuki M, et al. High-performance combinatorial optimization based on classical mechanics. Science Advances, 2021, 7(6): eabe7953.

[8] Tiunov E S, Ulanov A E, Lvovsky A I. Annealing by simulating the coherent Ising machine. Optics express, 2019, 27(7): 10288-10295.

[9] Reifenstein S, Kako S, Khoyratee F, et al. Coherent Ising machines with optical error correction circuits. Advanced Quantum Technologies, 2021, 4(11): 2100077.

[10] Kanao T, Goto H. Simulated bifurcation assisted by thermal fluctuation. Communications Physics, 2022, 5(1): 153.

[11] Bowles J, Dauphin A, Huembeli P, et al. Quadratic unconstrained binary optimization via quantum-inspired annealing. Physical Review Applied, 2022, 18(3): 034016.
