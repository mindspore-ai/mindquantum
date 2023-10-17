论文题目：Improving the Performance of Deep Quantum Optimization Algorithms with Continuous Gate Sets

项目介绍：本论文使用QAOA算法解决精确覆盖问题。
论文中提出了一种量子线路的优化，论文图1(a)(b)分别显示了使用QAOA算法解决精确覆盖问题的两种量子线路。
(b)是一般线路，UC部分应用了多个H门、CZ门和旋转Z门；
(a)是本文采取的改进线路，UC部分每层只应用了两个旋转Z门和一个受控旋转Z门；
相比之下，优化线路每一层都减少了一定的量子门数量，考虑到UC部分会根据完整线路的层数p，不断倍增，最终优化线路就能减少很多量子门，从而提高性能。



复现目标：
1、利用MindQuantum复现论文中提到的QAOA算法；
2、复现图3中的结果。



主要结果：
完整代码见main.ipynb
1、生成初态（均匀叠加态），通过在每个qubit上应用H门来实现。代码如下：
init_state_circ = UN(H, k.shape[1])

2、生成哈密顿量HC，本论文中，HC = Sigma(Jij * Zi * Zj) + Sigma(hi * Zi)
辅助函数get_Jij用于计算系数Jij，函数get_hi用于计算系数hi，实际上，本论文中，所有hi全为零。代码如下：
def build_HC(k):
    HC = QubitOperator()
    for j in range(k.shape[1]):
        for i in range(j):
            HC += get_Jij(i, j, k) * QubitOperator(f'Z{i} Z{j}')
    for i in range(k.shape[1]):
        HC += get_hi(i, k) * QubitOperator(f'Z{i}')
    return HC

3、构建ansatz线路，由p层UC+UB交替组合而成，
UCij = Phase(i,2*gamma*Jij) + Phase(j,2*gamma*Jij) + Phase(i,j,4*gamma*Jij)
UB就是给每个量子位应用一次RX门，参数为2*beta。代码如下：
def build_ansatz(k, p):
    circ = Circuit()
    for i in range(p):
        circ += build_UC(k, f'g{i}')
        circ += build_UB(k, f'b{i}')
    return circ
def build_UC(k, para):
    UC = Circuit()
    for j in range(k.shape[1]):
        for i in range(j):
            Jij = get_Jij(i, j, k)
            UC += PhaseShift({para: 2*Jij }).on(i)
            UC += PhaseShift({para: 2*Jij }).on(j)
            UC += PhaseShift({para: -4 * Jij }).on(j, i)
    UC.barrier()
    return UC
def build_UB(k, para):
    UB = Circuit()
    for i in range(k.shape[1]):
        UB += RX({para:2}).on(i)
    UB.barrier()
    return UB

4、组合成完整线路之后，借助其他模块，构建模拟器和带训练模型，对模型进行训练后就可得到gamma和beta的最优参数，带入量子线路中就可解决精确覆盖问题。
例如，本论文中，对矩阵（其中，每一列代表一个子集，每一行代表一个元素，第i行第j列取值为1，代表第i个元素在第j个子集中）
[[1 0 1]
 [0 1 1]
 [0 1 1]]
计算其精确覆盖，结果应为前两列或最后一列，用qubit的形式来表示就是|110>或|001>，详细结果见main.ipynb最末尾处。

5、要复现的几张图像，都是层数为1时，三量子比特的运行情况（即第4点中的矩阵），具体图像见main.ipynb靠近末尾处。
①②本论文中，使用低温超导量子设备进行实验，原文第一张heatmap图应该是实验数据，第二张是模拟数据。复现时给出的两张图都是模拟数据，计算方式相同。
该heatmap图是在(gamma，beta)的取值下，反映cost函数的变化情况。本论文中，Cost(gamma,beta)=<gamma,beta| HC |gamma,beta>
可以根据模拟器simulator基于当前量子态的求解算子grad_ops进行计算。
图上可以观察到，图像被分为了左侧一上一下两个分界明显的区域，以及右侧田字形的四个区域。
③⑤第三张和第五张分别是，从随机初始参数开始，对gamma和beta进行优化的收敛迹，可以根据模型训练过程中的数据进行计算。
图上可以观察到，gamma收敛于pi/4，beta收敛于pi/8和pi*3/8，与heatmap图相对应。
④第四张的两条曲线分别是，在beta=pi/8或beta=pi*3/8的条件下，gamma与cost的对应关系。
图上可以观察到，两条曲线关于cost=0的横轴上下对称。
⑥第六张的十条彩色线条分别是，对应于图2和图4的条件下，cost的收敛迹，其中，黑色线条是平均cost。
图上可以观察到，所有线条都逐渐收敛于cost=-1.



项目总结：
整个复现过程都是基于本人自己的理解进行的，所以所有内容也仅仅是我个人的观点，学有不足，可能存在错误。
复现目标一，利用MindQuantum复现论文中提到的QAOA算法已基本完成；
复现目标二，图3中共有6张小图，第一张小图（heatmap图）应该是在低温超导量子设备上的结果，第二张小图是无噪模拟的结果，
所以复现中只给出了无噪模拟的结果，其他几张小图都复现了。



一些简单的思考：
1、本论文的改进之处主要位于UC部分的线路，但是观察本论文中绘制的UB部分的线路（UBi = Y(-pi/2) + Z(2*beta）+ Y(pi/2)），
发现UB部分可以直接用一个RX门来实现，即，此处的Y+Z+Y的功能与RX相同。
如果对量子门数量的减少可以有效提高算法性能，那么为什么论文中仍然将UB部分绘制成三个量子门的叠加？
2、其实对本论文的内容有一个小小的看法，文中给出了改进门的矩阵表示，但并没有给出改进门与一般线路的相等性证明，
感觉给出数学证明会好一点。只是一个非常细节的小想法。
