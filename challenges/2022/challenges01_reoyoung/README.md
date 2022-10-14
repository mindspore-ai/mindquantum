# 黑客松赛题-量子化学模拟:基态能求解的浅层线路设计

Variational Quantum Eigensolver (VQE) 算法是一种用于计算分子基态能量的量子化学模拟方法，它可以结合经典计算机，利用当前含噪的量子计算机解决一些化学问题。现有常用的线路设计和优化方法通常难以平衡计算效率和计算精度的问题，本次挑战赛旨在让参赛者通过设计线路或改良优化方法，用最短的时间优化得到对应分子的符合化学精度的基态能量。

## 赛题要求：

（1）对于不同的测试分子案例，选择基矢为sto-3g。根据给出的分子体系设计量子线路并尝试进行各种优化，使计算的基态能量尽可能达到化学精度(0.0016Ha)。
（2）要求设计线路需要有可扩展性，能根据不同的化学分子系统给出相应的量子线路。
（3）提交代码必须经过HiQ量子计算云平台Jupyter Notebook环境测试验证。


## 测试数据：

初赛第一轮测试数据集
（1）分子1：H6，比特数：12， 键长点：[0.8, 0.9, 1.0]
（2）分子2：LiH，比特数：12，键长点：[0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.6, 4.0]

## 评分标准：

（1）与FCI方法计算结果进行比较：各个键长点的计算结果均需达到化学精度，没有达到化学精度的点不得分，达到化学精度的键长点越多，评分越高；程序运行时间越短，评分越高；总得分如下，按照从小到大排名：
    time cost = Σ未达到精度的键长点*3600/(Baseline_in_sec) + Σ达到化学精度的键长点(Rumtime_in_sec)/(Baseline_in_sec)

（2）判题系统验证时间超过1小时判为0。

## 提交说明：

- 将所有代码归档到`src`文件夹内。
- 开发的程序可以基于`main.py`文件，或自行定义并将其类名命名为`Main`，放置于`main.py`文件内。最终结果需要经过`eval.py`的测试。
- 作品打成压缩包`challenges01_姓名_联系电话.zip`提交。
- 文件请按照项目结构要求保存所有项目源代码和输出文件（通过`eval.py`输出）。
- 举办方会调用`eval.py`文件来对开发的模型进行验证，请参考`eval.py`代码保证开发的模型能够被顺利验证。

## 项目结构

```bash
.   				<--根目录，提交时将整个项目打包成`challenges01_姓名_联系电话.zip`
├── eval.py			<--举办方用来测试选手的模型的脚本,可以作为个人测试使用，提交作品请勿修改
├── readme.md		<--说明文档
├── src				<--源代码目录，选手的所有开发源代码都应该放入该文件夹类
│   ├── hdf5files   <--相关分子文件储存
│   └── main.py		<--`eval.py`调用程序

```

## 参考文献

[1] McArdle, S., Endo, S., Aspuru-Guzik, A., Benjamin, S. C., & Yuan, X. (2020). Quantum computational chemistry. Reviews of Modern Physics, 92(1), 015003.
[2] Li, Y., Hu, J., Zhang, X., Song, Z., & Yung, M. (2019). Variational Quantum Simulation for Quantum Chemistry. Advanced Theory and Simulations, 2(4), 1800182.
[3] Kandala, A., Mezzacapo, A., Temme, K. et al. Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. Nature 549, 242–246 (2017).
