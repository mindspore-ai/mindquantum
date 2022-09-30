文件说明：

221cb0174.
- `vqa_for_quantum_dynamics.ipynb`. 此文档利用`mindquantum`完美复现了Yuan X, Endo S, Zhao Q, et al. Theory of variational quantum simulation[J]. Quantum, 2019, 3: 191.的结果，[link](https://doi.org/10.22331/q-2019-10-07-191)。给出了密度矩阵的基本概念介绍、利用纯化方案得到混合态的密度矩阵表示，详细介绍了用量子变分算法求解主方程的方案，给出了Ising模型量子动力学演化的实例。建议放入`tutorials`中，作为一种新的应用实例介绍。
- `dynamics.py`. 密度矩阵主方程含时演化的量子电路源程序。
- `densitymatrix.py`. 将输入的Circuit、array转化为密度矩阵，密度矩阵在算符作用以后的演化
- `get_probs.py`. 在不做sampling的情况下，直接通过张量乘法，快速得到circuit在某个qubit上计算基下的测量结果。jupyter notebook中介绍的量子变分方法的关键在于设计量子线路去求解$\textrm{Tr}\left[\rho_1\rho_2\right]$，最后我们需要调用`mindquantum`内置的`Simulator`做测量来拿到q$_0$量子比特在$|0\rangle$与$|1\rangle$取到的概率，而采样的数目`shots`动辄需要$10^5\sim 10^6$次，这使得对于每个电路的测量时间非常长：当设置`shots`$=100000$时，每求一次所有$\theta$对$t$的导数需要约10分钟，而对于FIG.6我们需要演化1000步，因此这使得计算时长非常久（$\sim 10^4$ min）。于是增加了新的功能，即不经过采样，通过直接做张量乘积得到精确的测量结果，借鉴了IBM的Qiskit的功能，可以在秒量级输出测量结果。建议之后放入`mindquantum.core.circuit`中，与`get_qs()`配合使用。
