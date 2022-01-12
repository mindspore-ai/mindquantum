# Blind quantum machine learning based on quantum circuit model

复现作者：周旭

华为云账号：hw33393305

邮箱地址：zhouxu39@huawei.com

## 项目介绍

### 复现文章介绍

我要复现的是2021年11月1日在《Quantum Information Processing》上发表的论文，文章题目为“Blind quantum machine learning based on quantum circuit model”，文章作者为Xu Zhou和Daowen Qiu。

### 文章摘要

Blind quantum machine learning (BQML) is a novel secure quantum computation protocol that enables a client (Alice), who has limited quantum technology at her disposal, to delegate her quantum machine learning to a remote quantum server (Bob) who owns a fully-fledged quantum computer and promises to execute the learning task honestly, in such a way that Bob cannot obtain Alice’s private information. In this paper, we first propose the concept of BQML that combines blind quantum computation and quantum machine learning and we devise two BQML protocols based on quantum circuit model in which Alice can classify vectors of any dimension to different clusters. The first protocol is half-blind, while the second is blind. It means Alice’s privacy can be protected. On the other hand, Alice is only required to possess a quantum random access memory (QRAM), apply the Pauli operators ($X$ and $Z$), store, send, receive qubits and perform measurements in our protocols. Finally, we analyze the security, blindness and correctness of our protocols, and give a brief conclusion.

## 主要结果

### 复现目标

我将通过MindQuantum验证：

（1）论文中的Fig1可以计算2个2维向量$\vec{u}$和$\vec{v}$的欧几里得距离；

（2）论文中的Fig2-Fig6和Fig9（`H` gate, `Fredkin` gate, `CNOT` gate, `CZ` gate, `SWAP` gate，`Toffoli` gate）的线路正确性；

（3）论文中的Fig7可以计算2个16维向量$\vec{u}$和$\vec{v}$的欧几里得距离（即$n=4$的情况）；

（4）论文中的Fig8和Fig10的线路等价性；

（5）论文中的Fig11的线路正确性；

（6）论文中的Fig12（$n=4$的情况），即在Fig7的基础上保密计算2个16维向量$\vec{u}$和$\vec{v}$的欧几里得距离。

## 创新点

在复现过程中，我的创新点如下：

（1）利用MindQuantum首次复现盲量子计算相关的论文；

（2）利用到了MindQuantum最新的功能，输出量子线路矩阵的matrix()功能，并把发现的问题在主仓中提了issue；

（3）利用到了np.kron()计算向量外积，再在模拟器上set_qs()的功能来制备线路的初态，而一般我们量子线路的初态为全|0>态；

（4）利用MindQuantum实现输出量子线路真值表的功能；

（5）华为bbs论坛的复现要求：验证论文中的Fig2，Fig3，Fig4，Fig5，Fig6，Fig9，Fig11中线路的正确性，复现论文中最终的Fig12中n=4的情况。在完成复现要求的基础上，我把剩下的Fig1，Fig7，Fig8，Fig10也复现了，并给出了证明，使得整个复现工作比较完整。