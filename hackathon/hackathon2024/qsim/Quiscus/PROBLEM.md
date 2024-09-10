# 黑客松大赛--含噪变分量子算法求解

## 赛题说明

给定一个 $H_4$ 分子的结构数据（判题脚本中会采用新的键长），请利用变分量子算法，按照实际量子计算机的行为来得到该分子的基态能量。量子计算机的行为约束如下：

1. 只能采用规定的门集合：$\{X, CNOT, Y, Z, H, CZ, RX, RY, RZ, 测量, 栅栏\}$。
2. 期望值计算只能通过采样比特串来得到，你需要根据所需计算的哈密顿量，来合理的设计量子线路，并通过 `sampling` 得到的采样数据计算出不同 pauli 串哈密顿量的期望值。例如，计算如下期望值时 $\left<\psi\right|X_0\left|\psi\right>$，我们需要对零比特添加一个 $RY(-\pi/2)$ 旋转，再对 $Z_0$ 求期望值：$\left<\psi\right|X_0\left|\psi\right> = \left<\psi\right|RY(\pi/2)Z_0RY(-\pi/2)\left|\psi\right>$。
3. 暂不约束芯片拓扑结构，默认为全联通，任意两比特之间可以相互作用。
4. 赛题会给定一个噪声模型，在比赛专用量子模拟器中，会自动在模拟器的 `apply_gate`、`apply_circuit`、`sampling` 等接口处自动添加噪声模型，请勿绕过，**一经发现将取消比赛成绩**。噪声模型定义具体如下：单比特门会伴有一个极化率为 $0.001$ 的单比特去极化信道噪声和一个$t_1=100us, t_2=50us,t_{gate}=30ns$ 的热弛豫噪声，双比特门会伴有一个极化率为 $0.004$ 的双比特去极化信道噪声和一个 $t_1=100us, t_2=50us,t_{gate}=80ns$ 的热弛豫噪声，测量门会伴有一个翻转概率为 $0.05$ 的比特翻转信道噪声。

## 判题与评分标准：

赛题提供了一个专用的量子模拟器，该模拟器会自动添加赛题要求的噪声模型，并禁用一些只能在量子模拟器中才有的API逻辑，并且会统计整个赛题过程中 `sampling` 的总次数。比赛总时长要求为2小时。
总得分为：

$$\frac{10000}{|E−E_{fci} |* n_{shots}}$$

其中𝐸为选手得到的分子基态能量，𝐸_fci 为分子的FCI能量，𝑛_shots为比赛过程中总的采样次数。

## 准备

安装依赖：

```bash
pip install -r requirements.txt
```

## 脚本使用说明

请在 `solution.py` 中完成函数

```python
def solution(molecule, Simulator: HKSSimulator) -> float:
    ...
```

判题系统会通过 `solver.py` 来调用你的求解方法来进行判题。判题系统只会上传你的 `solution.py`文件，修改其他文件将不会生效。

## 脚本参数说明

```bash
python solver.py --help

usage: solver.py [-h] [-f FILE] [--demo]

测试你的求解算法。

optional arguments:
  -h, --help            show this help message and exit
  -f FILE, --file FILE  输入分子文件，默认值为 mol.csv
  --demo                运行官方提供的demo方案
```

运行官方提供的求解方案：

```bash
python solver.py --file mol.csv --demo
```