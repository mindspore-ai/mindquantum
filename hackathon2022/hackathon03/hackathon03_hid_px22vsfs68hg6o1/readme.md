# 3比特幺正算符重构

## 赛题说明

- 在本题中，我们给定一个包含18个参数的3比特的量子线路编码器，位于`encoder_circuit.py`中。
- `train_x.npy`中包含一个二维的numpy数组，其维度为(800, 18)，为800个经典数据。每一个数据中的18个特征与编码线路中的18个参数一一对应。
- 每一个数据经过编码线路后会得到一个量子态$\left|\psi\right>$。该量子态经过一个未知的幺正算符$U$操作后得到目标量子态$\left|\varphi\right>=U\left|\psi\right>$。我们将所有800个目标量子态存放于`train_y.npy`中。
- `test_x.npy`中包含500个用于测试的经典数据，其维度为(500, 18)。

## 赛题要求

试利用MindQuantum，训练出一个量子线路，该量子线路能够重构出幺正算符$U$。

- `test_x.npy`中的500个经典数据经过`encoder_circuit.py`中编码线路编码后的量子态为$\left|\psi\right>$，请利用你重构出来的量子线路，将$\left|\psi\right>$经过$U$演化出来的量子态$\left|\varphi\right>=U\left|\psi\right>$保存到`test_y.npy`中。

## 测试

评分系统将会调用`eval.py`来测试你的预测量子态与理论量子态之间的保真度平均值，并以此为最终得分。请将包含预测量子态`test_y.npy`的整个项目打包提交。

## 注意

当前给的`test_y.npy`和`real_test_y.npy`均为全零样本，仅供参考。评分系统评分时将覆盖`encoder_circuit.py`和`eval.py`，并载入理论量子态数据集`real_test_y.npy`，结合你生成的`test_y.npy`来给出得分。