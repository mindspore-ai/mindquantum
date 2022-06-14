# 黑客松赛题-量子化学模拟:基态能求解的浅层线路设计

Variational Quantum Eigensolver（VQE）算法是一种用于计算分子基态能量的量子化学模拟方法，它可以结合经典计算机，利用当前含噪的量子计算机解决一些化学问题。现有常用的线路设计模板通常难以平衡计算效率和计算精度的问题，而线路设计方案是这种算法领域中被研究的一个主要问题。

## 赛题要求：

- 我们对于不同的测试分子案例，选择基矢为sto-3g。根据给出的分子体系设计量子线路并试进行各种优化，使计算的基态能量尽可能达到化学精度(0.0016Ha)。
- 要求设计线路需要有可扩展性，能根据不同的化学分子系统给出相应的量子线路，测试案例包括两个给出的分子；举办方会再评测一个分子。
- 程序要求基于mindquantum，可基于现有的ansatz进行修改，也可以自行创作新的ansatz。
- 参数优化过程可以基于梯度的方法或非梯度的方法。
- 计算结果利用模板中的`main.py`中的Plot作图。


## 测试数据：

- 分子1（40%）：LiH，比特数：12，键长点：[0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]
- 分子2（30%）：CH4，比特数：18，键长点：[0.4, 0.8]
- 分子3（30%）：举办方测试，不对外公布

## 评分标准：

与FCI方法计算结果进行比较，各个键长点的计算结果均需达到化学精度；程序运行时间越短，评分越高。每个键长点根据下列公式得分，再按照比例加权得到总分（如分子1的每个键长点得分为40% * 1/10；分子2的每个键长点得分为30% * 1/2），没有达到化学精度的点不计分。
- 每个点的得分 Score_per_point = 1000 / (Rumtime_in_sec)

## 提交说明：

- 将所有代码归档到`src`文件夹内。
- 开发的程序可以基于`main.py`文件，或自行定义并将其类名命名为`Main`，放置于`main.py`文件内。最终结果需要经过`eval.py`的测试。
- 作品打成压缩包`hackathon02_队长姓名_队长联系电话.zip`提交。
- 文件请按照项目结构要求保存所有项目源代码和输出文件（通过`eval.py`输出）。
- 举办方会调用`eval.py`文件来对开发的模型进行验证，请参考`eval.py`代码保证开发的模型能够被顺利验证。

## 项目结构

```bash
.   				<--根目录，提交时将整个项目打包成`hackathon02_队长姓名_队长联系电话.zip`
├── eval.py			<--举办方用来测试选手的模型的脚本,可以作为个人测试使用
├── readme.md		<--说明文档
├── src				<--源代码目录，选手的所有开发源代码都应该放入该文件夹类
│   ├── main.py		<--参考程序模型范例，
│   ├── LiH.hdf5	<--分子1数据文件
│   └── CH4.hdf5	<--分子2数据文件
```

## 参考文献

[1] McArdle, S., Endo, S., Aspuru-Guzik, A., Benjamin, S. C., & Yuan, X. (2020). Quantum computational chemistry. Reviews of Modern Physics, 92(1), 015003.
[2] Li, Y., Hu, J., Zhang, X., Song, Z., & Yung, M. (2019). Variational Quantum Simulation for Quantum Chemistry. Advanced Theory and Simulations, 2(4), 1800182.
[3] Kandala, A., Mezzacapo, A., Temme, K. et al. Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. Nature 549, 242–246 (2017).
