# 自适应的混合误差缓解

刘展欧 华东师范大学

# 1 摘要

在含噪声中等规模量子（NISQ）计算领域，测量读出误差是影响量子算法计算结果准确性的重要因素。本研究针对量子测量误差缓解（MEM）问题，提出一种三阶段自适应混合误差缓解方法，该方法整合智能校准、迭代条件精炼（ICR）和启发式后处理（HPE）三个核心模块，旨在实现资源消耗与模型精度的合理平衡。方案通过数据驱动的Louvain社区发现算法对量子比特进行分组，结合张量积分解构建局部噪声模型，并基于物理先验知识设计后处理策略，有效解决了传统全局模型计算复杂度随比特数指数增长以及局部模型精度不足的问题。

在赛题设定的线性误差模型  $\vec{p}_{\mathrm{noisy}} = M\vec{p}_{\mathrm{ideal}}$  框架下，该方法采用重要性加权采样策略优化校准数据的利用效率，在9比特量子系统中通过选择性采集部分基态测量数据实现噪声模型的高精度构建。消融实验结果显示，智能分组处理使评分从1145.91提升至2982.06，引入条件串扰建模后评分进一步提高至9576.44，启发式后处理步骤将总变差距离（TVD）降至93.21

与现有研究相比，本方案克服了全局校准方法资源消耗过大和张量积分解模型精度受限的不足，通过混合架构实现了算法可扩展性与结果准确性的协调。研究表明，NISQ设备上的测量误差缓解需要结合数据驱动的噪声拓扑分析、物理机制建模和特定问题优化策略，为大规模量子计算的误差管理提供了可行路径。

# 2 问题背景与描述

# 2.1 赛题描述

本次竞赛的核心任务是设计一种高效的算法，用于缓解量子计算中的读出误差（Readout Error）。参赛者需要根据提供的含噪声的测量数据，还原出量子系统在测量前的理想概率分布。算法的性能将根据其输出结果与理论理想值的接近程度以及所消耗的校准数据量综合评定。

# 核心目标

给定一个由量子测量产生的、包含噪声的概率分布向量  $\vec{p}_{\mathrm{noisy}}$  ，参赛者需要设计一个误差缓解算法，计算出一个校准后的概率分布向量  $\vec{p}_{\mathrm{calibrated}}$  ，使得该向量尽可能地接近真实的理想概率分布  $\vec{p}_{\mathrm{ideal}}$  。

# 核心模型

问题背景建立在一个线性的误差模型之上，由Maciejewski,Zimboras,and Oszmaniec (2020)提出：

$$
\vec{p}_{\mathrm{noisy}} = M\vec{p}_{\mathrm{ideal}}
$$

其中  $M$  是一个  $2^{n}\times 2^{n}$  （对于本赛题  $n = 9$  ）的校准矩阵（Calibration Matrix）。矩阵元素 $M_{ij}$  代表了当系统被制备在理想的计算基态  $|j\rangle$  时，经过测量后得到结果为  $|i\rangle$  的概率。理论上，通过求解该方程，如矩阵求逆  $\vec{p}_{\mathrm{calibrated}} = M^{- 1}\vec{p}_{\mathrm{noisy}}$  ，即可恢复理想分布。

# 数据提供

- 基础线路数据（校准数据）：提供了一个9比特系统在所有  $2^{9} = 512$  个计算基态（从 $|00\dots 0\rangle$  到|11. ..1)）上进行制备和测量的结果。该数据用于学习和构建误差模型，即估计校准矩阵  $M$  。参赛者通过get_data函数获取此数据，其使用量将被记录并影响最终得分。

- 目标线路数据(测试数据)：提供了6个具体的量子线路（包括GHZ态制备线路和随机线路）的测量结果。这些是含有未知噪声的分布，是算法需要去校正的最终目标。

# - 评分标准

- 基础分：基于校正后分布与理想分布的总变差距离（Total Variation Distance，TVD）来计算。TVD越小，基础分越高。

$$
\mathrm{score}_{\mathrm{base}} = 1000\times (1 - \mathrm{TVD}) = 1000\times \left(1 - \frac{1}{2}\sum_{i = 1}^{2^n}|p_{\mathrm{calibrated},i} - p_{\mathrm{ideal},i}|\right)
$$

- 奖励分：与算法在训练/校准阶段所使用的“基础线路数据”量线性负相关。使用的数据量‘train_sample_num'越少，奖励分数  $\alpha$  越高。

$$
\mathrm{final~score} = \mathrm{score}_{\mathrm{base}} + 1000\times \alpha
$$

这鼓励参赛者设计出数据高效（data- efficient）的算法。

# 2.2 赛题分析

这个赛题深刻地触及了当前含噪声中等规模量子（NISQ）计算时代的一个核心问题：如何在有限的量子资源和不可避免的噪声影响下，提取出有用的计算结果。赛题的难点和关键点可以从以下几个方面进行分析。

# - 主要挑战：指数复杂度 (The Exponential Challenge)

对于一个  $n$  比特的系统，其状态空间维度为  $2^n$  。在本赛题中  $n = 9$  ，意味着校准矩阵  $M$  是一个  $512\times 512$  的矩阵。

- 数据量挑战：完整地、精确地刻画这个  $512\times 512$  矩阵，理论上需要对512个不同的初始态进行制备和测量，并且每个初始态都需要大量的测量次数（shots）来获得统计上可靠的概率。这正是评分中“奖励分”机制所针对的问题：如何用远少于“全部”的数据来构建一个足够好的误差模型。

- 计算资源挑战：存储和求逆一个  $512\times 512$  的矩阵在经典计算机上是完全可行的，但赛题的设置是在为更大规模的问题（例如  $n > 20$  ）做铺垫。参赛者需要思考的不仅仅是如何解决  $n = 9$  的问题，更是要探索具有良好可扩展性（scalable）的方案。

# - 核心权衡 (Key Trade-offs)

优秀的解决方案必须在以下几个方面做出精妙的权衡：

- 精度 vs. 资源 (Accuracy vs. Resources)：这是最直接的权衡。使用更多的校准数据可以构建更精确的全局校准矩阵  $M$  ，从而可能获得更高的基础分，但这会牺牲奖励分。

反之，使用少量数据构建的模型可能不准，导致基础分降低。寻找“性价比”最高的数据使用策略是关键。

- 全局 vs. 局部 (Global vs. Local Models): 这是应对指数挑战的核心思路。

* 全局模型: 直接构建完整的  $512 \times 512$  矩阵。优点是能捕捉到所有比特间的关联噪声（correlated noise）和串扰（crosstalk），模型最精确。缺点是资源消耗巨大。* 局部模型: 假设各个比特的读出噪声是独立的。那么总的校准矩阵可以近似为每个单比特校准矩阵的张量积（Kronecker Product):  $M_{\mathrm{total}} \approx M_8 \otimes M_7 \otimes \dots \otimes M_0$ 。这种方法的资源消耗极小（只需校准9个  $2 \times 2$  的小矩阵），但它完全忽略了比特间的关联性，会导致模型失真，精度受限。

- 分区策略 (Partitioning Strategy): 介于全局和局部之间的最佳路径。将9个量子比特智能地划分为若干个小组（例如， $\{q_0, q_1\}$ ， $\{q_2, q_3\}$ ， $\{q_4\}$ ，...），然后为每个小组构建一个局部的校准矩阵，总的误差模型是这些小组矩阵的张量积。这里的核心挑战在于：如何找到最优的分区？最优分区应该将相互之间噪声关联最强的比特划分到同一个组内。正如样例代码‘answer.py’和参考文献[2,3]所揭示的，可以利用校准数据本身来计算比特间的噪声相关性，从而指导分区。

# - 潜在的解法路径

- 基线方法 (Baseline): 使用足量数据构建完整的  $512 \times 512$  校准矩阵  $M$ ，然后通过矩阵求逆或更稳健的迭代贝叶斯展开（IBU）方法来求解。IBU等方法可以保证结果的物理意义（概率非负且归一）。

- 张量积分解法 (Tensor Product Decomposition): 实施分区策略。首先确定一个分区方案（可以是固定的，也可以是数据驱动的），然后为每个子集获取校准数据，构建子矩阵，最后通过张量积重构近似的全局校准矩阵  $M$  或其逆矩阵  $M^{-1}$ 。

- 迭代式分区优化 (Iterative Partitioning and Mitigation): 这是基于分治法做出的改进。可以先用一个初步的分区方案进行一次误差缓解，然后分析缓解后的结果与理想值的偏差，利用这个偏差信息来指导下一轮更优的分区，如此迭代，不断优化误差模型和分区方案。

- 电路感知分区 (Circuit-Aware Partitioning): 可以分析目标线路本身的结构。如果两个比特在目标线路中有两比特门（如CNOT）直接相连，它们在物理上更有可能产生关联噪声。因此，可以将线路结构作为分区的一个重要先验信息。

# 3 相关工作与文献

在含噪声中等规模量子（NISQ）计算时代，器件的物理不完美性引入了多种噪声，其中读出误差（Readout Error）是影响计算结果保真度的主要障碍之一。它导致测量结果的概率分布偏离理论上的理想分布，从而可能掩盖量子算法的潜在优势。因此，发展高效、可扩展且对物理实现友好的测量误差缓解（Measurement Error Mitigation, MEM）技术，是推动量子计算从理论走向应用的关键一环。这一章将梳理该领域从基础理论到前沿方案的技术演进脉络，并对关键工作进行深度分析，涵盖了从全局校准、分解近似、到机器学习和噪声不可知论等多种技术范式。

量子测量误差缓解技术的发展，可以视为一场在“模型精度”、“资源开销”和“算法鲁棒性”三者之间不断寻求更优平衡的探索。其演进路径大致遵循了从全局、高耗散的方法，向可扩展、精巧且稳健的混合框架发展的趋势。

# 3.1阶段一：全局校准与选代修正的奠基

该阶段方法论的核心，是试图构建一个能完整描述整个系统读出噪声的全局模型，并在此基础上进行精确校正。

·理论基础与挑战：线性误差模型  $\vec{p}_{\mathrm{noisy}} = M\vec{p}_{\mathrm{ideal}}$  构成了该阶段的理论基石。这里，  $M$  是一个维度为  $2^{n}\times 2^{n}$  的校准矩阵，原则上包含了所有比特间任意复杂的关联噪声信息。早期的工作，如Kandalaetal.(2017)在真实硬件上的演示，就已经认识到缓解此类误差的必要性。最直观的缓解方案便是通过实验标定出  $M$  ，然后对其求逆以恢复理想分布。然而，Bravyiet al.(2021)的深入分析指出，由于有限测量次数带来的统计涨落，直接求逆会放大噪声，特别是当  $M$  矩阵条件数较大时，结果极易出现非物理的负概率。

·从直接求逆到选代展开为解决此问题，Nachmanetal.(2020）将源自高能物理的选代贝叶斯展开（IterativeBayesianUnfolding，IBU）方法引人量子计算领域。IBU并非粗暴地求逆，而是基于贝叶斯定理进行选代修正，从含噪分布出发，逐步“剥离”  $M$  矩阵所代表的噪声效应。其内在的选代机制保证了每一步的输出都满足概率的非负性和归一性，因此具有优越的数值稳定性和对统计噪声的鲁棒性。

·核心局限：无论是直接求逆还是IBU，它们都共享一个致命的“阿喀琉斯之睡”——指数可扩展性问题。标定一个完整的  $2^{n}\times 2^{n}$  矩阵需要  $2^{n}$  次不同的校准实验，这使得该类方法的资源开销随比特数指数增长，对于超过约20个量子比特的系统而言，这在实践中是不可行的。

# 3.2阶段二：可扩展分解策略的兴起

为了突破指数墙，研究重心转向了如何对噪声模型进行简化和分解，以实现可扩展的误差缓解。

·路径一：张量积分解(TensorProductDecomposition)：Nationetal.(2021）的工作是这一思路的典型代表。该方法做出一个强假设：各个比特的读出噪声是相互独立的。在此假设下，全局校全矩阵可以被近似为所有单比特校准矩阵  $(2\times 2)$  的张量积：  $M\approx \otimes_{k = 0}^{n - 1}M_k$  。这种方法的校准开销仅随比特数  $n$  线性增长，展现了极佳的可扩展性。然而，这是一种以牺牲精度为代价的实用主义策略，它完全忽略了物理系统中普遍存在的串扰和关联噪声，其精度存在一个由模型失真决定的理论上限。

·路径二：智能分区与子空间模型(IntelligentPartitioning)：为了在可扩展性与模型精度之间找到更优的平衡点，研究者们提出了“分而治之”的策略。其核心思想是将量子比特划分为若干小组，假设组内比特的噪声是强关联的，而组间噪声则相对独立。

Tanetal.(2024）提出的QuFEM方法将这一思想推向了新的高度。该工作创造性地借鉴了有限元分析的理念，将比特分组视为物理仿真中的“有限元”。它通过数据驱动的方式分析比特间的噪声相关性，从而进行智能分区，确保将噪声耦合最强的比特划分到同一个“元”内进行联合校准。相似地，Maciejewski，Baccari，etal.(2021）等工作也探索了基于相关性度量来对量子比特进行分组的策略，共同推动了这一方向的发展。

·路径三：基于Clifford电路的校准(Clifford- basedCalibration)：这是一个功能强大且高效的类别，它利用了Clifford电路的特殊性质——可在经典计算机上高效模拟。通过在大量随机选择的Clifford 电路上运行并对比实验结果与理论模拟值，可以有效地学习噪声模型。IBM的M3(Matrix- FreeMeasurementMitigation）方法，如VanDenBergetal.(2023）所述，是这一方向的工业级实现。它避免了显式构造校准矩阵  $M$  ，而是直接求解一个在校准数

据集上优化的修正分布，极具可扩展性。另一重要技术是 Clifford 数据回归（Clifford Data Regression, CDR），如 Hamilton et al. (2020) 和 Chen et al. (2019) 的工作所示，它通过对 Clifford 电路测量结果的线性回归来学习一个简化的噪声模型，并将其应用于任意电路的误差缓解。

# 3.1 阶段三：高级、混合与不可知论框架

当前的研究前沿，是将前两个阶段的优势进行有机结合，或另辟蹊径，发展不依赖于精确噪声模型的全新范式。

- 混合框架：可扩展性与鲁棒性的融合：Pokharel et al. (2024) 的工作完美诠释了这一融合思想。他们将 IBU 算法与可扩展的分解噪声模型相结合，证明了即便底层的噪声模型是近似的，IBU 的稳健性依然能够确保最终产出物理上合理的、且精度得到显著改善的概率分布。这代表了当前解决大规模系统 MEM 问题的最有力途径之一。

- 机器学习赋能：机器学习，特别是深度学习模型，因其强大的非线性函数拟合能力，被用于学习复杂的噪声信道。例如，García-Pérez et al. (2021) 探索了使用生成模型来学习和逆转噪声过程。而 Zheng et al. (2023) 则将贝叶斯推断与神经网络相结合，提出了一种混合的缓解方案。这些方法为捕捉传统模型难以描述的复杂关联噪声提供了新的可能性，尽管其可解释性和训练开销仍是挑战。

- 噪声不可知论：零噪声外推（Zero-Noise Extrapolation, ZNE）：与上述所有试图“表征-逆转”噪声的思路不同，ZNE 是一种“黑箱”式的缓解方法。正如 Temme, Bravyi, and Gambetta (2017) 和 Li and Benjamin (2017) 的开创性工作所展示的，ZNE 的核心思想是：通过某种方式（如拉伸门的作用时间）主动地放大电路中的噪声，并在几个不同的噪声水平下运行同一电路；然后，将测得的期望值作为噪声强度的函数进行拟合，并外推到噪声为零的理想情况。尽管 ZNE 主要针对门噪声，但其思想对缓解测量噪声也有启发，并代表了一类不依赖于精确噪声模型的通用 mitigation 范式。

量子测量误差缓解技术在过去数年间经历了深刻的演进：从概念上完美但实践中不可扩展的全局模型，到以牺牲部分精度为代价的实用主义分解模型，再到如今结合了智能分区、Clifford 校准、机器学习和鲁棒迭代算法的复杂混合框架。这一历程不仅反映了领域内对物理噪声认识的不断深化，也体现了在面对 NISQ 时代计算资源限制时，研究者们在算法设计上的智慧与创新。未来的突破将可能来自于更深层次的噪声物理建模、与机器学习等数据科学方法的交叉融合，以及软硬件协同设计的进一步发展。

# 4 方案描述

# 4.1 方案设计

该解决方案并非采用单一的传统纠错方法，而是构建了一个三阶段、自适应的混合误差缓解流水线（Three- Stage Adaptive Hybrid Error Mitigation Pipeline）。其核心哲学在于深刻认识到，在资源受限的 NISQ 时代，最优策略并非追求一个完美的、全局的噪声模型，而是在资源效率与模型精度之间取得动态平衡。算法的设计精妙地融合了多种前沿思想，包括可扩展的张量网络分解、数据驱动的智能分区、以及针对特定目标线路的启发式后处理。

- 阶段一：智能校准 (Intelligent Calibration): 以最少的量子资源消耗，最高效地学习噪声的关键特征，特别是比特间的关联噪声 (Crosstalk)。

- 阶段二：混合式误差缓解 (Hybrid Error Mitigation): 基于校准数据，动态地构建一个近似但高度精确的噪声模型，并应用该模型对含噪数据进行纠正。

- 阶段三：启发式后处理 (Heuristic Post-Processing): 对纠正后的结果进行最终的“润色”，利用对目标理想态（如GHZ态）的先验知识，进一步消除残余误差和非物理伪影。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-05/f30d7b91-d33e-4909-bac8-492e5828c431/0f1bb7e51ad0b1065b4af4ede81ff404e73181f7734ff1efe68cc7f03a7b6b60.jpg)  
图1算法三阶段流水线总览

# 4.2 算法流程

- 阶段一：校准方案设计与数据获取

考虑到评分标准中对训练数据量的激励机制，一个高效的校准方案是获得高分的先决条件。本方案采用了一种重要性加权（Importance- Weighted）的校准状态选择策略，旨在通过有限的测量次数最大化地获取关于系统噪声，特别是关联噪声的信息。

- 校准状态的选择依据：校准状态的选取并非随机或全局遍历，而是基于对噪声来源的物理理解，策略性地覆盖了以下几类关键的计算基态：

1. 全局基准态： $|00\dots 0\rangle$  和  $|11\dots 1\rangle$  这两个状态用于标定系统在无激发和全激发情况下的基础错误率，是所有模型的基准。2. 单比特错误率标定：对每个量子比特  $i$  制备单比特激发态  $|0\dots 1_{i}\dots 0\rangle$  。这组测量能够精确地刻画每个比特在隔离状态下的主要错误概率  $P(1|0)_i$  和  $P(0|1)_i$ 。3. 本地串扰（Crosstalk）探测：针对物理上相邻或在目标线路中存在两比特门操作的比特对  $(i,j)$ ，制备激发态  $|0\dots 1_{i}\dots 1_{j}\dots 0\rangle$  。通过对比单比特激发和双比特激发时的错误率变化，可以定量地分析测量串扰的强度。4. 多体关联噪声探测：为了探测更复杂的噪声模式，方案引入了如GHZ态、W态、以及棋盘格（Checkerboard）等具有高度纠缠或特定空间关联性的多体激发态。这些状态的测量结果对于构建高精度的多体关联噪声模型至关重要。

- 自适应采样策略：在 correct 函数的实现中，通过 importance_weights 字典为不同类别的校准状态分配了不同的测量次数。例如，用于探测串扰的邻近比特激发态被赋予了比单比特激发态更高的权重，这意味着算法会投入更多的测量资源（shots）来更精确地学习关联噪声的参数。这种自适应采样策略确保了有限的“TRAIN_SAMPLE_NUM”预算被高效地利用。

# - 阶段二：迭代条件精炼（ICR）缓解引擎

这是本方案的核心技术。它将数据驱动的图论算法与基于张量积分解的物理模型相结合，并通过一个两步迭代过程来执行缓解。

![](https://cdn-mineru.openxlab.org.cn/result/2025-08-05/f30d7b91-d33e-4909-bac8-492e5828c431/ad53e5e2b5be47d6939aa1293aa66d858f288fa828cd73216b062491568eb06e.jpg)  
图2阶段二核心：数据驱动的智能分区与模型构建流程

- 数据驱动的噪声拓扑发现: 在 _build_correlation_graph 函数中, 算法首先量化任意两比特  $(q_{i}, q_{j})$  间的噪声相关性。这通过计算  $q_{i}$  的错误率在  $q_{j}$  的理想状态为  $|0\rangle$  和  $|1\rangle$  两种情况下的差异来实现。随后, 结合目标线路的物理连接图 ('connectivity_graph'), 构建一个加权图, 其边的权重反映了噪声关联的强度。最后, 在 _find_clusters_and_neighbors 函数中, 利用如图论中的 Louvain 社区发现算法对该图进行社群划分, 从而自动地将关联最紧密的比特 “聚类” 到同一个分区中。

# - ICR 缓解过程:

1. 第一阶段（鲁棒预校正）：首先，算法应用一个平均噪声模型。该模型为每个簇  $c_{i}$  构建一个单一的、平均化的校准矩阵  $M_{c_{i}}$  。这个过程通过在构建矩阵时对所有邻居状态进行边缘化（即不加区分地使用所有相关校准数据）来实现。该模型的逆矩阵  $(\otimes M_{c_{i}})^{-1}$  被应用于原始含噪数据  $\vec{p}_{\mathrm{noisy}}$  ，得到一个噪声水平显著降低但仍可能包含高阶误差的中间分布  $\vec{p}_{\mathrm{refined}}$  。

2. 第二阶段（条件精细校正）：随后，算法启用完整的条件噪声模型。它将  $\vec{p}_{\mathrm{refined}}$  作为新的输入。在处理该分布中的每一个比特串时，对于需要校正的簇  $c_{i}$  ，算法会从比特串中读取其邻居  $N_{i}$  的状态  $y_{N_{i}}$  。然后，它精确地从预先构建的条件矩阵族中查找到对应的逆矩阵  $\left(M_{c_{i}|y_{N_{i}}}\right)^{-1}$  并加以应用。这一步能够精确地修正由邻居状态引起的串扰效应。由于输入数据  $\vec{p}_{\mathrm{refined}}$  已经过预校正，此阶段应用更复杂的条件模型时，因统计噪声而被放大的风险大大降低，从而保证了数值稳定性。

# - 阶段三：启发式后处理

考虑到线性模型和近似分解可能引入的残余误差，或导致非物理的负概率，算法在最后阶段加入了一个基于物理先验知识的后处理模块。

- 基于模式识别的增强: 在 heuristic_post_processing 函数中, 算法首先分析校正后概率分布的宏观特征（如峰值位置、熵、汉明权重分布等), 以识别其最可能对应的理想态类型（例如, “GHZ 态”、“W 态”或“稀疏峰值分布”)。

- 专用非线性滤波器: 针对识别出的不同模式, 算法会应用不同的、经过经验优化的非线性变换。

* 对于 GHZ 态, 它会有选择地增强  $|00\dots 0\rangle$  和  $|11\dots 1\rangle$  的概率幅值, 同时以保持总概率守恒的方式抑制其他态的幅值。* 对于 W 态, 它会增强所有汉明权重为 1 的态的概率。

这种基于先验知识的增强, 能有效平滑伪影, 强制结果符合物理约束（概率非负), 从而在 TVD 评分上获得最终的优势。

# 4.3 代码结构

该解决方案的代码结构清晰, 体现了良好的面向对象设计思想, 将复杂的算法逻辑解耦到不同的类中。

- correct(): 总指挥。作为算法的唯一入口, 负责调用各个模块, orchestrating the entire pipeline from calibration to final output.

- Mitigator Class: 战略规划师。管理整个缓解流程, 核心职责是执行数据驱动的智能分区策略 ('correlation_based_partition'), 并根据评估结果选择最优的迭代次数。

- Iteration Class: 战术执行官。封装了一次具有固定分区方案的完整缓解过程。它负责根据当前的分区方案，向‘TPEngine’请求并构建相应的校准矩阵。- TPEngine Class: 数学计算核心。这是最底层的“引擎”，负责执行具体的矩阵运算，包括对局部校准矩阵求逆，以及通过张量积运算来重构和应用缓解算子。- 辅助函数: 大量的工具函数（如‘to_int’，‘statuscnt_to_npformat’等）负责数据的格式转换和基础数学运算，保证了主流程代码的简洁与可读性。

# 5 结果与分析

为了提升在含噪声中等规模量子（NISQ）设备上量子算法的保真度，我们实现并评估了一种先进的混合式测量误差缓解（MEM）算法。该算法集成了基于数据驱动的噪声拓扑发现、迭代式条件精炼（ICR）以及启发式后处理（HPE）三个核心阶段。本报告通过一次全面的消融研究（Ablation Study），定量地剖析了算法中各个关键组件——包括比特分区、条件串扰建模以及启发式后处理——对最终缓解效果的独立贡献。研究结果表明，尽管每个组件均能带来性能提升，但只有将它们协同地整合在一个多阶段流水线中，才能达到最优的性能，这凸显了在NISQ时代构建复合型、多层次误差缓解框架的必要性与优越性。

# 5.1 消融实验设计

我们的全功能算法是一个三阶段流水线。为了量化每个阶段的贡献，我们设计了四种递进的算法配置进行比较。

- Config 1: Uncorrelated + HPE (非关联模型 + HPE): 此为基线配置。它采用了最简化的噪声模型，假设所有量子比特的噪声完全独立（即每个比特自成一簇）。在此基础上，应用了启发式后处理（HPE）阶段。此配置用于衡量在不考虑任何比特间关联的情况下，仅凭HPE所能达到的性能。

- Config 2: Partitioned (Averaged) + HPE (分区平均模型 + HPE): 此配置引入了智能分区。算法首先根据校准数据和电路拓扑进行数据驱动的分区，但对每个分区内的噪声采用“平均化”处理，即忽略邻居状态依赖（串扰）。与Config 1对比，可以衡量仅“聚类”这一行为带来的性能提升。

- Config 3: ICR (Conditional) only (仅ICR模型): 此配置采用了完整的两阶段迭代条件精炼（ICR）模型，它既包括分区，也包括对串扰的条件建模。然而，此配置禁用了最后的启发式后处理（HPE）阶段。其目的是为了独立评估纯粹基于物理模型的、最先进的线性校正方法所能达到的性能极限。

- Config 4: Full Model (ICR + HPE) (完整模型): 此为我们最终的、全功能的算法。它顺序执行了ICR和HPE两个核心模块。与Config 2和Config 3的对比，可以分别量化出条件串扰建模和启发式后处理的独立贡献。

# 5.2 研究结果

我们对上述四种配置在全部6个目标电路上进行了评估，并计算了平均的最终得分、总变差距离（TVD）以及执行时间。

表1不同算法配置下的性能消融研究结果  

<table><tr><td>算法配置</td><td>得分</td><td>平均 TVD (%)</td><td>执行时间 (s)</td></tr><tr><td>Config 1: Uncorrelated + HPE</td><td>1145.91</td><td>88.54</td><td>103.37</td></tr><tr><td>Config 2: Partitioned (Avg) + HPE</td><td>2982.06</td><td>91.40</td><td>83.56</td></tr><tr><td>Config 3: ICR (Conditional) only</td><td>8391.27</td><td>91.17</td><td>154.53</td></tr><tr><td>Config 4: Full Model (ICR + HPE)</td><td>9576.44</td><td>93.21</td><td>157.29</td></tr></table>

# 5.3 性能分析

根据表1和图7的结果，我们可以得出以下结论：

- 分区的重要性 (Config 2 vs. Config 1): 从 Config 1 到 Config 2，我们看到分数的显著提升。这清晰地证明，即使只是将强相关的比特聚类并使用一个平均化的噪声模型，其性能也远超完全不相关的模型。这验证了噪声的局域性是其最重要的结构特征之一，仅仅是识别并独立处理这些局域的“错误中心”，就能带来巨大的性能收益。

- 串扰建模的贡献 (Config 4 vs. Config 2): 对比 Config 4 和 Config 2（两者都使用了分区和 HPE，但 Config 4 额外使用了条件模型），我们能观察到分数的进一步提升。这个增量完全归功于 ICR 的第二阶段，即对测量串扰的精确建模。这表明，在初步校正之后，对邻居依赖的噪声进行精细调节是提升保真度的关键一步，单纯的平均模型会掩盖这一部分重要的物理效应。

- 启发式后处理的使用价值 (Config 4 vs. Config 3): 这是本次研究中一个有价值的发现。对比 Config 3（纯 ICR 模型）和 Config 4（ICR+HPE），我们观察到分数的显著提升。这揭示了一个核心事实：对于具有特定结构（如 GHZ 态）的基准测试问题，一个纯粹的、基于线性矩阵的物理模型，即便再复杂，也难以完全消除所有非物理伪影和统计噪声。而一个简单的、基于物理先验知识的非线性后处理步骤，能够有效地优化现有结果，使其更接近理想的稀疏分布，从而在 TVD 度量上获得更理想的结果。

# 5.4 结论

本次消融研究定量地证实了我们所设计的三阶段混合式误差缓解算法的有效性。研究表明，没有任何一个单一组件可以独立地实现最优性能。算法的成功来源于各个阶段的协同作用：

1. 智能分区是基础，它通过利用噪声的局域性来克服指数复杂度墙。2. 迭代条件精炼 (ICR) 是核心，它通过两步迭代，稳定地校正了包括串扰在内的主要物理噪声。3. 启发式后处理 (HPE) 是进一步优化，它利用对问题本身的先验知识，完成了线性模型无法企及的“最后一公里”优化。

这一研究为未来 NISQ 设备上的误差缓解算法设计提供了清晰的范式：即结合可扩展的物理噪声建模与针对特定问题的启发式优化，是通往高精度量子计算的有效路径。

#

[Bra+21] Sergey Bravyi et al. "Mitigating measurement errors in multiqubit experiments". In: Physical Review A 103.4 (2021), p. 042605. [Che+19] Yanzhu Chen et al. "Detector tomography on IBM quantum computers and mitigation of an imperfect measurement". In: Physical Review A 100.5 (2019), p. 052315. [Gar+21] Guillermo Garcia- Perez et al. "Learning to measure: Adaptive informationally complete generalized measurements for quantum algorithms". In: Prx quantum 2.4 (2021), p. 040342. [Ham+20] Kathleen E Hamilton et al. "Scalable quantum processor noise characterization". In: 2020 IEEE International Conference on Quantum Computing and Engineering (QCE). IEEE. 2020, pp. 430- 440. [Kan+17] Abhinav Kandala et al. "Hardware- efficient variational quantum eigensolver for small molecules and quantum magnets". In: nature 549.7671 (2017), pp. 242- 246. [LB17] Ying Li and Simon C Benjamin. "Efficient variational quantum simulator incorporating active error minimization". In: Physical Review X 7.2 (2017), p. 021050. [Mac+21] Filip B Maciejewski, Flavio Baccari, et al. "Modeling and mitigation of cross- talk effects in readout noise with applications to the quantum approximate optimization algorithm". In: Quantum 5 (2021), p. 464. [MZO20] Filip B Maciejewski, Zoltan Zimboras, and Michal Oszmaniec. "Mitigation of readout noise in near- term quantum devices by classical post- processing based on detector tomography". In: Quantum 4 (2020), p. 257. [Nac+20] Benjamin Nachman et al. "Unfolding quantum computer readout noise". In: npj Quantum Information 6.1 (2020), p. 84. [Nat+21] Paul D Nation et al. "Scalable mitigation of measurement errors on quantum computers". In: PRX Quantum 2.4 (2021), p. 040326. [Pok+24] Bibek Pokharel et al. "Scalable measurement error mitigation via iterative bayesian unfolding". In: Physical Review Research 6.1 (2024), p. 013187. [Tan+24] Siwei Tan et al. "QaFEM: Fast and Accurate Quantum Readout Calibration Using the Finite Element Method". In: Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2. 2024, pp. 948- 963. [TBG17] Kristan Temme, Sergey Bravyi, and Jay M Gambetta. "Error mitigation for short- depth quantum circuits". In: Physical review letters 119.18 (2017), p. 180509. [Van+23] Ewout Van Den Berg et al. "Probabilistic error cancellation with sparse Pauli- Lindblad models on noisy quantum processors". In: Nature physics 19.8 (2023), pp. 1116- 1121. [Zhe+23] Muqing Zheng et al. "A bayesian approach for characterizing and mitigating gate and measurement errors". In: ACM Transactions on Quantum Computing 4.2 (2023), pp. 1- 21.