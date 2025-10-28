# HUBO求解器使用说明

本文用于展示HUBO求解器的详细使用说明。

------

HUBO求解器包括python版和C++版两个部分：

1. CUDA_code中为C++版，包括CPU版和GPU版。
2. tensorflow_code中为python版，CPU版和GPU版通用同一份代码（可切换）。

以下是相关代码文件列表和使用说明。

## 相关代码文件列表



```
.
├── CUDA_code //C++版HUBO求解器
│   ├── BSB_CPU.cu //BSB算法的HUBO求解器，CPU版
│   ├── BSB_GPU.cu //BSB算法的HUBO求解器，GPU版
│   ├── CAC_CPU.cu //CAC算法的HUBO求解器，CPU版
│   ├── CAC_GPU.cu //CAC算法的HUBO求解器，GPU版
│   ├── CFC_CPU.cu //CFC算法的HUBO求解器，CPU版
│   ├── CFC_GPU.cu //CFC算法的HUBO求解器，GPU版
│   ├── DSB_CPU.cu //DSB算法的HUBO求解器，CPU版
│   ├── DSB_GPU.cu //DSB算法的HUBO求解器，GPU版
│   ├── json.hpp //json头文件
├── tensorflow_code //python版HUBO求解器
│   ├── BSB.py //BSB算法的HUBO求解器
│   ├── CAC.py //CAC算法的HUBO求解器
│   ├── CFC.py //CFC算法的HUBO求解器
│   ├── DSB.py //DSB算法的HUBO求解器
│   ├── HUBO.py //代码结构化的HUBO求解器
├── 数据集格式模板.json //json数据集的内容及格式
```

## 数据集格式说明

**一、数据集格式说明** 

本数据集格式旨在为HUBO（高阶无约束二元优化）问题提供一个标准化的输入接口。HUBO问题的目标是最小化一个定义在二元变量（自旋）上的多项式函数，该函数通常被称为哈密顿量 (Hamiltonian) 。

整个数据集是一个json对象，包含两个核心数据：`J` 和`coeff` 。

`J` ：定义了哈密顿量中的每一个“项”，即哪些自旋之间存在相互作用。`coeff` ：定义了与 J 中每个项一一对应的系数（或称为权重、强度）。这两个列表通过索引严格对应，即 coeff[i] 是 J[i] 所定义项的系数。

**1. `J`（相互作用关系） **

`J` 是一个二维列表，`List[List[int]]`。它的每一个子列表代表哈密顿量中的一个数学项。

- **`J` 中子列表的元素**: 子列表中的每个整数都是一个**自旋变量的索引**，通常从0开始编号。
- **项的类型**:
  - **相互作用项 (Interaction Term)**: 当子列表包含多个整数时，它定义了一个多体相互作用。例如，`[29, 33, 34, 35]` 表示第29、33、34和35号自旋之间存在一个四体相互作用。
  - **自作用项 (On-site Term / Bias Term)**: 当子列表只包含一个整数时，它定义了一个单体项，也称为“偏置项”或“局部场”。例如，`[0]` 代表只与第0号自旋相关的项。这对应于哈密顿量中的线性项。

##### **2. `coeff` (Coefficients / 系数)**

`coeff` 是一个一维的浮点数列表，`List[float]`。

- **与 `J` 的关系**: `coeff` 列表的长度必须与 `J` 列表的长度完全相等。
- **值的含义**: `coeff` 中的每一个元素 `coeff[i]` 是 `J` 中对应项 `J[i]` 的数值系数。它决定了该项对系统总能量的贡献大小和符号。

**3. 文件命名格式**

数据集文件为json格式，命名格式为“0.json”、“1.json”……

具体例子可见数据集格式模板.json

## 使用说明

先准备好所需求解问题的json数据集

**一、python版HUBO求解器** 

1.若使用 tensorflow_code中的.py代码，需安装tensorflow库。

2.修改代码中的数据集读取路径：代码中的base_path = r'demo_data\\{}.json'修改为base_path = r'你的数据集存放路径\\{}.json'

3.修改BSB.py、DSB.py、CAC.py、CFC.py代码中的with tf.device('/CPU:0'):为with tf.device('/GPU:0'):或with tf.device('/CPU:0'):来切换计算设备为CPU或GPU。

4.HUBO.py中可直接通过以下代码来进行HUBO求解器使用

```
solver = DSB(J, coeff, backend='CPU')
solver.initialize()
solver.update()
```

**二、C++版HUBO求解器**

1.CUDA_code中的.cu代码，调用了json.hpp头文件。

2.修改代码中的数据集读取路径：代码中的 string filePath = "demo_data/" + to_string(time_i) + ".json";修改为 string filePath = "你的数据集存放路径/" + to_string(time_i) + ".json";