# 论文复现大赛

- [论文复现大赛](#论文复现大赛)
  - [已完成复现论文](#已完成复现论文)
  - [量子计算论文复现大赛开发指南](#量子计算论文复现大赛开发指南)
  - [可选论文](#可选论文)
  - [代码要求](#代码要求)
  - [代码提交格式](#代码提交格式)
  - [代码提交路径](#代码提交路径)
  - [评分标准](#评分标准)
    - [复现结果审查阶段](#复现结果审查阶段)
    - [入围决赛评选阶段](#入围决赛评选阶段)

## 已完成复现论文

|论文名称|选手ID|
|-|-|
|02: [Qubit-ADAPT-VQE: An Adaptive Algorithm for Constructing Hardware-Efficient Ansätze on a Quantum Processor](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/2_hw86909202/main.ipynb)|@xie-qingxing|
|04: [Reachability Deficits in Quantum Approximate Optimization](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/4_Magi/main.ipynb)|@Magi_karp|
|05: [Microcanonical and finite-temperature ab initio molecular dynamics simulations on quantum computers](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/05_hw_008615959957849_01/main.ipynb)|@Franke_cdk|
|06: [Accelerated variational algorithms for digital quantum simulation of the many-body ground states](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/6_hw08624896/main.ipynb)|@he-tianshen|
|09: [Quantum generative adversarial networks](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/09_big91987/main.ipynb)|@big91987|
|10: [An aritifical neuron implemented on an actual quantum processor](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/%E8%AE%BA%E6%96%8710_Mr_Tang754/main.ipynb)|@tang-love-coke|
|12: [Barren Plateaus in Quantum Neural Network Training Landscape](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/12_hid_b4uryzmyfxuzzn1/main.ipynb)|@zishen-li|
|13: [Universal quantum state preparation via revised greedy algorithm](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/13_waikikilck/main.ipynb)|@herunhong|
|15: [When does reinforcement learning stand out in quantum control? A comparative study on state preparation](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/15_firing_feather/main.ipynb)|@yuanjzhang|
|19: [Capacity and quantum geometry of parametrized quantum circuits](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/19_Rebecca/main.ipynb)|@Rebecca666|
|20: [Improving the Performance of Deep Quantum Optimization Algorithms with Continuous Gate Sets](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/20_faketrue/main.ipynb)|@fake-true|
|21: [Deep reinforcement learning for universal quantum state preparation via dynamic pulse control](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/21_waikikilick/main.ipynb)|@herunhong|
|22: [Blind quantum machine learning based on quantum circuit model](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/22_hw33393305/main.ipynb)|@zhou-xu3|
|24: [Quantum simulation with hybrid tensor networks](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/24_hw_008613816232674_01/main.ipynb)|@overshiki|
|28: [Introduction to Quantum Reinforcement Learning: Theory and PennyLane-based Implementation](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/28_hid_r3jndb66c0zbhr9/main.ipynb)|@xianyu256|
|30: [Stochastic gradient descent for hybrid quantum-classical optimization](https://gitee.com/mindspore/mindquantum/tree/research/paper_recurrence/30_hw_008613571866975_01)|@weifuchuan123|
|35: [A hybrid classical-quantum approach for multi-class classification](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/%E8%AE%BA%E6%96%8735_Mr_Tang754/main.ipynb)|@tang-love-coke|
|48: [Grand Unification of Quantum Algorithms](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/48_kyrrego/main.ipynb)|@yr_zhang|
|50: [Variational ansatz-based quantum simulation of imaginary time evolution](https://gitee.com/mindspore/mindquantum/blob/research/paper_recurrence/50_hw_008613816232674_01/main.ipynb)|@overshiki|

## 量子计算论文复现大赛开发指南

[量子计算论文复现大赛开发指南](https://gitee.com/mindspore/mindquantum/tree/research/paper_recurrence/developers_guide)

## 可选论文

[https://competition.huaweicloud.com/information/1000041627/circumstance](https://competition.huaweicloud.com/information/1000041627/circumstance)

## 代码要求

请使用[**MindQuantum**](https://gitee.com/mindspore/mindquantum)量子计算库和[**MindSpore**](https://www.mindspore.cn/install)机器学习框架（如有需求）来复现您选择的论文，如有部分功能MindQuantum中尚未实现，请使用Numpy或者Scipy科学计算包来实现。

## 代码提交格式

请参考[**paperid_username_for_example**](https://gitee.com/mindspore/mindquantum/tree/research/paper_recurrence/paperid_username_for_example)来组织和提交您的代码，主要包含如下三大块：`src`、`main.ipynb`和`readme.md`。`src`存放您论文复现的代码，`main.ipynb`是用来介绍和展示您复现结果的jupyter notebook文件，`readme.md`是用来对项目进行介绍的简要文档。

## 代码提交路径

- 初赛作品提交：将代码和文档按照**paperid_username_for_example**格式要求提交到mindquantum仓research分支的**paper_recurrence/paperid_username**。**paperid**为论文的序号，**username**为参赛者华为云账号名。

## 评分标准

分赛事分为两个阶段：1、复现结果审查阶段；2、入围决赛评选阶段；

### 复现结果审查阶段

只需达到相应论文的**复现最低要求**即可完成复现结果审查阶段。

### 入围决赛评选阶段

在入围决赛评选阶段，评委会根据您复现的结果在如下五个维度进行打分：

|评分维度|解释|
|-|-|
|复现完成度|在最低要求的基础之上，是否复现更多的论文内容|
|复现精度|复现模型在你的调教之下，是否比原论文中精度更高，如已达到100%等特殊情况，本单项直接获得满分|
|代码质量|根据代码符合编程规范的程度来评判，可参考MindQuantum的源代码编码规范|
|性能|根据代码复现过程中CPU占用量、内存占用量和计算时长等来判定|
|创新性|在原论文的基础上，是否有更多的自己的思考，是否对模型有优化等|

> 评选规则解释权归本次大赛组委会所有