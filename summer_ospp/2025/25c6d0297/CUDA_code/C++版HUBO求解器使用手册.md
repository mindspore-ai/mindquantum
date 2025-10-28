# HUBO_Solver C++/CUDA 使用手册



这是一个使用 C++/CUDA 加速的 HUBO (Higher-Order Unconstrained Binary Optimization) 求解器的 Python 接口。

该项目使用 pybind11 将 C++ 类绑定到一个名为 `hubo_solver` 的 Python 模块中。`HUBO.h` 声明了类结构，`HUBO.cu` 提供了 CUDA/C++ 实现，`bindings.cpp` 负责 Python 绑定。



## 1. 依赖要求



编译本项目需要以下依赖：

- `CMake >= 3.18` 
- `CUDAToolkit` (NVIDIA CUDA 工具包) 
- `pybind11`
- 支持 C++17 和 CUDA 的编译器 (例如 `g++`, `nvcc`)

`pybind11` 可以通过 `pip` 安装：

Bash

```
pip install pybind11
```



## 2. 编译指南



您可以按照以下步骤编译生成 Python 模块 `hubo_solver.so`：

1. **创建构建目录**

   Bash

   ```
   mkdir build
   ```

2. **进入构建目录**

   Bash

   ```
   cd build
   ```

3. **运行 CMake**

   Bash

   ```
   cmake ..
   ```

   注意：如果 CMake 找不到您的 Python 环境，您可能需要指定 `pybind11` 的路径。`CMakeLists.txt` 3文件中包含了查找 `pybind11` 的逻辑。

4. **编译**

   使用 make 进行编译。-j8 参数会使用 8 个核心并行编译，您可以根据需要调整：

   Bash

   ```
   make -j8
   ```

## 3. Python API 用法



编译完成后，您可以直接在 Python 中 `import hubo_solver`。



### 3.1. 模块和类



- `import hubo_solver`：导入模块。
- `hubo_solver.BSB`: BSB 求解器类。
- `hubo_solver.DSB`: DSB 求解器类。



### 3.2. 构造函数



`BSB` 和 `DSB` 类共享相同的构造函数：

Python

```
solver = hubo_solver.BSB(J, coeff, iter, b_size, be="CPU")
solver = hubo_solver.DSB(J, coeff, iter, b_size, be="GPU", xi=0.5, dt=1.0, delta=1.0)
```

**参数:**

- `J` (`vecto<vector<int>>`): 定义 HUBO 问题的耦合项。每个子列表代表一个项，包含参与该项的自旋索引。
- `coeff` (`vector<float>`): `J` 中各项对应的系数。`coeff` 列表的顺序必须与 `J` 列表的顺序严格对应。
- `iter` (`int`): 求解器迭代的总次数。
- `b_size` (`int`): 批处理大小 (batch size)，即同时求解多少个独立的实例。
- `be` (`str`, 可选): 后端选择。`"CPU"` 或 `"GPU"`。 默认为 `"CPU"`。
- `xi` (`float`, 可选): 求解器参数，默认为 `0.5`。
- `dt` (`float`, 可选): 求解器参数，默认为 `1.0`。
- `delta` (`float`, 可选): 求解器参数，默认为 `1.0`。



### 3.3. 成员方法



`BSB` 和 `DSB` 实例共享以下方法：

- `solver.initial()`: 初始化求解器的自旋和辅助变量 `y`。
- `solver.update()`: 运行 `iter` 次迭代求解。
- `solution = solver.get_solution()`: 返回求解器找到的最佳解（一个自旋列表 `list[float]`）。
- `energy = solver.get_energy(solution)`: 计算并返回**单个解**（`list[float]`）的能量（`float`）。
- `energies = solver.get_energy(solutions)`: 计算并返回**一批解**（`list[list[float]]`）的能量（`list[float]`）。



### 能量计算公式 (Energy Formula)



`get_energy` 方法计算的是 HUBO 模型的总能量 `E` 。该模型是伊辛 (Ising) 模型的推广。

假设$$s_i \in \{-1, 1\}$$是第 i 个自旋。

- `J` 列表 ( `vector<vector<int>>` ) 定义了耦合项。
- `coeff` 列表 ( `vector<double>` ) 定义了对应的系数。

总能量 `E`的计算公式为：
$$
E = \sum_{k} \left( \text{coeff}[k] \times \prod_{i \in J[k]} s_i \right)
$$


**示例：**

- 如果 `J[k] = [i, j]` (二阶项)，则该项为 $\text{coeff}[k] \times s_i s_j$。
- 如果 `J[k] = [i]` (一阶项)，则该项为 $\text{coeff}[k] \times s_i$。
- 如果 `J[k] = [i, j, k]` (高阶项)，则该项为 $\text{coeff}[k] \times s_i s_j s_k$。

当所有 `J` 中的项最多只包含两个自旋时（即只含一阶和二阶项），该模型就退化为标准的**伊辛问题**。



## 4. Python 示例 (`HUBO.py`)



这是一个基于 `HUBO.py`  文件的完整示例。



*注意：此示例假定您有一个名为 `0.json` 的数据文件，其格式为 `{"J": [[...], ...], "coeff": [..., ...]}`。*

Python

```
import hubo_solver
import numpy as np
import json

# --- 1. 准备数据 ---
# 假设数据文件位于 0.json
# 您需要将其替换为您的实际路径
try:
    base_path = r'{}.json'
    file_path = base_path.format(0)
    data = json.load(open(file_path, 'r'))
    J = data['J']
    coeff = data['coeff']
    
# --- 2. 设置参数 ---
n_iterations = 1000
batch_size = 10
backend = "GPU"  # 您可以更改为 "CPU" 来测试 BSB 的 CPU 版本

# --- 3. 初始化 BSB 求解器 ---
print(f"Initializing BSB solver on {backend}...")
# 使用 BSB 类，并指定 GPU 后端
bsb_solver = hubo_solver.BSB(J, coeff, n_iterations, batch_size, be = backend)

# --- 4. 运行求解器 ---
print("Initializing spins")
bsb_solver.initial()

print("Running solver...")
bsb_solver.update() 


# --- 5. 获取并打印结果 ---
solution = bsb_solver.get_solution()
energy = bsb_solver.get_energy(solution)

print(f"\n--- 结果 ---")
print(f"找到的最佳解:\n{solution}")
print(f"对应的能量: {energy}")
```