# Gset 数据集 MAX-CUT 问题求解文档

## 项目概述

本项目使用模拟退火算法（SB 算法）求解 Gset 数据集中的 MAX-CUT 问题。项目包含 Python 接口和 CUDA C++ 实现，支持多种数据精度配置，旨在提供高效的组合优化问题求解方案。

### 项目目标
- 使用 BSB（Ballistic SB）和 DSB（Discrete SB）算法求解 MAX-CUT 问题
- 支持 FP8、FP16、INT8 三种数据精度
- 提供统一的 Python 接口
- 实现批处理并行加速，提高求解效率

---

## 项目结构

```
Gset_SB/
├── SB_solve_Gset.py              # 基础 Python 实现（BSB + DSB）
├── SB_solve_Gset_GPU.py          # GPU 加速 Python 接口（SB_CUDA版本）
├── SB_solve_Gset_GPU_pure.py     # GPU 加速 Python 接口（SB_CUDA_pure版本）
├── SB_solve_Gset.md              # 本文档
├── data/                         # Gset 数据目录
│   ├── G1.txt                   # 示例数据文件
│   ├── G21.txt                  # 示例数据文件
│   └── G22.txt                  # 示例数据文件
├── SB_CUDA/                      # CUDA 实现目录（基础版本）
│   ├── sb_cuda_gset_reader.h     # Gset数据读取器
│   ├── BSB_GPU_fp8_gset.cu       # BSB FP8 实现
│   ├── BSB_GPU_fp16_gset.cu      # BSB FP16 实现
│   ├── BSB_GPU_purefp16_gset.cu  # BSB 纯FP16实现
│   ├── BSB_GPU_int8_gset.cu      # BSB INT8 实现
│   ├── DSB_GPU_fp8_gset.cu       # DSB FP8 实现
│   ├── DSB_GPU_fp16_gset.cu      # DSB FP16 实现
│   ├── DSB_GPU_int8_gset.cu      # DSB INT8 实现
│   ├── compile_all_cuda.sh       # Linux编译脚本
│   └── compile_all_cuda.bat      # Windows编译脚本
├── SB_CUDA_pure/                 # CUDA 实现目录（优化版本）
│   ├── sb_cuda_gset_reader.h     # Gset数据读取器
│   ├── BSB_GPU_purefp8_improved.cu     # BSB FP8 优化实现
│   ├── BSB_GPU_purefp16_improved.cu    # BSB FP16 优化实现
│   ├── BSB_GPU_pureint8_improved.cu    # BSB INT8 优化实现
│   ├── DSB_GPU_purefp8_improved.cu     # DSB FP8 优化实现
│   ├── DSB_GPU_purefp16_improved.cu    # DSB FP16 优化实现
│   ├── DSB_GPU_pureint8_improved.cu    # DSB INT8 优化实现
│   ├── compile_all_cuda.sh              # Linux编译脚本
│   └── compile_all_cuda.bat             # Windows编译脚本
└── png/                          # 结果图片目录
    ├── G22_python_cut_convergence.png              # Python版本收敛图
    ├── G22_python_energy_time_convergence.png     # Python版本能量-时间图
    ├── G22_cuda_all_cut_convergence.png           # 6种算法收敛图(SB_CUDA版本)
    ├── G22_cuda_all_energy_with_time.png          # 6种算法能量-时间图(SB_CUDA版本)
    ├── G22_cuda_all_cut_convergence_pure.png      # 6种算法收敛图(SB_CUDA_pure版本)
    └── G22_cuda_all_energy_with_time_pure.png     # 6种算法能量-时间图(SB_CUDA_pure版本)
```

---

## 核心算法

### 1. 问题建模：MAX-CUT 到 Ising 模型的转换

#### 1.1 MAX-CUT 问题定义

给定一个无向图 $G = (V, E)$，其中 $V$ 是顶点集合，$E$ 是边集合，每条边 $(i, j) \in E$ 具有权重 $w_{ij} \geq 0$。MAX-CUT 问题的目标是找到一个顶点分割，使得跨越分割的边权重之和最大：

$$\max_{x \in \{-1, +1\}^n} \sum_{(i,j) \in E} w_{ij} \cdot \frac{1 - x_i x_j}{2}$$

其中 $x_i \in \{-1, +1\}$ 表示顶点 $i$ 的分割选择。

#### 1.2 Ising 模型转换

为了使用模拟退火算法求解，我们将 MAX-CUT 问题转换为 Ising 能量最小化问题。定义自旋变量 $s_i \in \{0, 1\}$：

$$E(s) = -\sum_{(i,j) \in E} J_{ij} s_i s_j - \sum_{i \in V} h_i s_i$$

其中：
- $J_{ij} = -\frac{w_{ij}}{2}$ 是耦合强度
- $h_i$ 是外场（对于 MAX-CUT 问题，$h_i = 0$）

#### 1.3 切割值计算公式

给定离散自旋配置 $\boldsymbol{\sigma} = (\sigma_1, \sigma_2, \dots, \sigma_n) \in \{-1, +1\}^n$，切割值计算为：

$$\text{Cut}(\boldsymbol{\sigma}) = \sum_{(i,j) \in E} w_{ij} \cdot \frac{1 - \sigma_i \sigma_j}{2}$$

展开得到：
$$\text{Cut}(\boldsymbol{\sigma}) = \frac{1}{2} \sum_{(i,j) \in E} w_{ij} - \frac{1}{2} \sum_{(i,j) \in E} w_{ij} \sigma_i \sigma_j$$

### 2. BSB 算法（Ballistic Simulated Bifurcation）

BSB 算法基于 Ising 模型的分叉动力学，通过模拟量子退火过程求解组合优化问题。

#### 2.1 BSB 动力学方程

BSB 算法的核心是以下耦合微分方程组：

**动量更新方程：**
$$\frac{dy_i}{dt} = -(1 - p(t)) \cdot x_i + \xi \cdot \frac{\partial E}{\partial x_i}$$

**位置更新方程：**
$$\frac{dx_i}{dt} = y_i$$

**能量梯度计算：**
$$\frac{\partial E}{\partial x_i} = h_i + \sum_{j \in N(i)} J_{ij} x_j$$

其中：
- $p(t) = \frac{t}{T}$ 是泵浦振幅，从 0 线性增加到 1
- $\xi$ 是耦合强度
- $dt$ 是时间步长
- $N(i)$ 是顶点 $i$ 的邻居集合

#### 2.2 BSB 与 DSB 的区别

| 特性 | BSB | DSB |
|-----|-----|-----|
| 变量表示 | 连续变量 $x_i \in [-1, 1]$ | 连续变量 $x_i \in [-1, 1]$ |
| 梯度计算 | $J_{ij} x_j$（连续） | $J_{ij} \text{sign}(x_j)$（离散） |
| 能量计算 | $J_{ij} x_i x_j$（连续） | $J_{ij} \text{sign}(x_i)\text{sign}(x_j)$（离散） |
| 切割值计算 | $\text{sign}(x_i)$ | $\text{sign}(x_i)$（两者相同） |

**关键区别说明：**
- **BSB 算法**：梯度计算和能量计算使用连续变量 $x$，**仅在边界约束和计算切割值时**使用 `sign(x)`
- **DSB 算法**：梯度计算和能量计算使用离散化变量 $\text{sign}(x)$，**在边界约束和计算切割值时**也使用 `sign(x)`

---

## Python 实现说明

### 1. SB_solve_Gset.py - 基础 Python 实现

该脚本提供使用 MindQuantum 库的纯 Python 实现，支持 BSB 和 DSB 两种算法。

#### 1.1 核心函数

**`solve_gset_bsb_dsb(gset_file, sample_size, n_iter_list, save_plot)` - 统一求解接口**

参数说明：
- `gset_file`: Gset数据文件路径
- `sample_size`: 采样次数 (默认100)
- `n_iter_list`: 迭代次数列表 (默认[10, 25, 50, 75, 100, 250, 500, 750, 1000])
- `save_plot`: 是否保存切割值收敛图 (默认True)

#### 1.2 可视化函数

**`plot_cut_convergence(n_iter_list, bsb_cut_list, dsb_cut_list, gset_file, show)` - 切割值收敛图**

图表特性：
- X 轴：对数刻度的迭代次数（log10）
- Y 轴：切割值（Cut Value）
- BSB 算法：蓝色实线（平均值）+ 虚线圆圈（最大值）
- DSB 算法：红色实线（平均值）+ 虚线方块（最大值）

### 2. SB_solve_Gset_GPU.py - GPU 加速实现

该脚本提供了 6 种 CUDA 实现的统一 Python 接口，支持多种数据精度。

#### 2.1 算法类型

| 算法类型 | 说明 | 精度 |
|---------|------|------|
| BSB_FP8 | BSB算法 | FP8 |
| BSB_FP16 | BSB算法 | FP16 |
| BSB_INT8 | BSB算法 | INT8 |
| DSB_FP8 | DSB算法 | FP8 |
| DSB_FP16 | DSB算法 | FP16 |
| DSB_INT8 | DSB算法 | INT8 |

### 3. SB_solve_Gset_GPU_pure.py - GPU 加速实现（SB_CUDA_pure版本）

该脚本提供了 SB_CUDA_pure 目录下 6 种优化 CUDA 实现的统一 Python 接口。

#### 3.1 算法类型

| 算法类型 | 说明 | 精度 | 实现文件 |
|---------|------|------|---------|
| BSB_PURE_FP8 | BSB算法优化版本 | FP8 | BSB_GPU_purefp8_improved.cu |
| BSB_PURE_FP16 | BSB算法优化版本 | FP16 | BSB_GPU_purefp16_improved.cu |
| BSB_PURE_INT8 | BSB算法优化版本 | INT8 | BSB_GPU_pureint8_improved.cu |
| DSB_PURE_FP8 | DSB算法优化版本 | FP8 | DSB_GPU_purefp8_improved.cu |
| DSB_PURE_FP16 | DSB算法优化版本 | FP16 | DSB_GPU_purefp16_improved.cu |
| DSB_PURE_INT8 | DSB算法优化版本 | INT8 | DSB_GPU_pureint8_improved.cu |

#### 3.2 与 SB_solve_Gset_GPU.py 的区别

| 特性 | SB_solve_Gset_GPU.py | SB_solve_Gset_GPU_pure.py |
|-----|---------------------|--------------------------|
| CUDA目录 | SB_CUDA | SB_CUDA_pure |
| 实现文件 | BSB_GPU_fp8_gset.cu 等 | BSB_GPU_purefp8_improved.cu 等 |
| 优化程度 | 基础实现 | 优化实现 |
| 数据读取 | 内嵌Python读取 | 调用sb_cuda_gset_reader.h |
| 编译脚本 | SB_CUDA/compile_all_cuda.* | SB_CUDA_pure/compile_all_cuda.* |

---

## CUDA 实现详情

### 1. 混合精度策略

所有 CUDA 实现采用混合精度策略：

| 数据类型 | 存储格式 | 计算格式 | 说明 |
|---------|---------|---------|------|
| 自旋状态 x | FP8/FP16/INT8 | Float32 | 量化存储，计算时反量化 |
| 动量 y | FP8/FP16/INT8 | Float32 | 量化存储，计算时反量化 |
| 梯度 gradient_term | Float32 | Float32 | 始终使用 Float32 保持精度 |
| 系数 | FP8/FP16/INT8 | Float32 | 量化存储，计算时反量化 |

### 2. SB_CUDA 目录下的基础实现

#### 2.1 BSB 基础实现

**BSB_GPU_fp8_gset.cu** - BSB FP8 实现
- 使用 FP8 E4M3FN 格式（PyTorch 兼容）
- 范围：[-448, 448]
- 精确的 round-to-nearest-even 舍入策略

**BSB_GPU_fp16_gset.cu** - BSB FP16 实现
- 使用 `__half` 类型存储
- PyTorch 风格的 FP16 实现

**BSB_GPU_int8_gset.cu** - BSB INT8 实现
- 使用 INT8 对称量化
- 范围：[-128, 127]
- INT8_SCALE=100.0f 将 [-1.27, 1.27] 映射到 INT8 范围

#### 2.2 DSB 基础实现

**DSB_GPU_fp8_gset.cu** - DSB FP8 实现
- 在梯度计算中使用 sign(x) 而非 x
- 与 BSB 的核心区别

**DSB_GPU_fp16_gset.cu** - DSB FP16 实现
- 与 BSB_FP16 类似，但梯度计算使用离散化变量

**DSB_GPU_int8_gset.cu** - DSB INT8 实现
- 应用了与 BSB_INT8 相同的优化

### 3. SB_CUDA_pure 目录下的优化实现

#### 3.1 Gset 数据读取器

**sb_cuda_gset_reader.h** - Gset 数据文件读取函数
- 提供 C/C++ 接口读取 Gset 格式文件
- 支持 COO（Coordinate）格式的稀疏矩阵表示
- 包含以下功能：
  - `read_gset_file()`: 读取 Gset 文件
  - `convert_gset_to_solver_format()`: 转换为求解器格式
  - `free_gset_data()`: 释放内存
  - `print_gset_info()`: 打印数据信息

数据结构定义：
```cpp
typedef struct {
    int num_vertices;          // 顶点数量
    int num_edges;             // 边数量
    int* row_indices;          // 行索引 [num_edges]
    int* col_indices;          // 列索引 [num_edges]
    float* weights;           // 边权重 [num_edges]
} GsetData;
```

#### 3.2 BSB 优化实现

**BSB_GPU_purefp8_improved.cu** - BSB FP8 优化实现
- 使用 FP8 E4M3FN 格式（PyTorch 兼容）
- 范围：[-448, 448]
- 精确的 round-to-nearest-even 舍入策略
- 与 sb_cuda_gset_reader.h 集成
- 编译输出：BSB_GPU_purefp8_improved.exe

**BSB_GPU_purefp16_improved.cu** - BSB FP16 优化实现
- 使用 `__half` 类型存储
- PyTorch 风格的 FP16 实现
- 纯 FP16 实现（无混合精度）
- 与 sb_cuda_gset_reader.h 集成
- 编译输出：BSB_GPU_purefp16_improved.exe

**BSB_GPU_pureint8_improved.cu** - BSB INT8 优化实现
- 使用 INT8 对称量化
- 范围：[-128, 127]
- INT8_SCALE=100.0f 将 [-1.27, 1.27] 映射到 INT8 范围
- 与 sb_cuda_gset_reader.h 集成
- 编译输出：BSB_GPU_pureint8_improved.exe

#### 3.3 DSB 优化实现

**DSB_GPU_purefp8_improved.cu** - DSB FP8 优化实现
- 在梯度计算中使用 sign(x) 而非 x
- 与 BSB 的核心区别
- 与 sb_cuda_gset_reader.h 集成
- 编译输出：DSB_GPU_purefp8_improved.exe

**DSB_GPU_purefp16_improved.cu** - DSB FP16 优化实现
- 与 BSB_PURE_FP16 类似，但梯度计算使用离散化变量
- 与 sb_cuda_gset_reader.h 集成
- 编译输出：DSB_GPU_purefp16_improved.exe

**DSB_GPU_pureint8_improved.cu** - DSB INT8 优化实现
- 应用了与 BSB_PURE_INT8 相同的优化
- 与 sb_cuda_gset_reader.h 集成
- 编译输出：DSB_GPU_pureint8_improved.exe

### 4. GPU 优化特性

1. **循环展开**：4次循环展开提高指令级并行
2. **__ldg()**：使用只读缓存优化内存访问
3. **Stream 异步执行**：减少同步点，提高效率
4. **异步内存传输**：使用 cudaMemcpyAsync

---

## 编译和运行

### 1. 编译 SB_CUDA 目录下的 CUDA 文件

**Windows:**
```bash
cd SB_CUDA
compile_all_cuda.bat
```

**Linux:**
```bash
cd SB_CUDA
chmod +x compile_all_cuda.sh
./compile_all_cuda.sh
```

### 2. 编译 SB_CUDA_pure 目录下的 CUDA 文件

**Windows:**
```bash
cd SB_CUDA_pure
compile_all_cuda.bat
```

**Linux:**
```bash
cd SB_CUDA_pure
chmod +x compile_all_cuda.sh
./compile_all_cuda.sh
```

编译脚本会编译以下 6 个可执行文件：
- BSB_GPU_purefp8_improved.exe
- BSB_GPU_purefp16_improved.exe
- BSB_GPU_pureint8_improved.exe
- DSB_GPU_purefp8_improved.exe
- DSB_GPU_purefp16_improved.exe
- DSB_GPU_pureint8_improved.exe

### 2. 运行 Python 程序

**运行基础实现：**
```bash
cd Gset_SB
python SB_solve_Gset.py
```

**运行 GPU 加速实现：**
```bash
cd Gset_SB
python SB_solve_Gset_GPU.py
```

**运行 GPU 加速实现（SB_CUDA_pure 版本）：**
```bash
cd Gset_SB
python SB_solve_Gset_GPU_pure.py
```

### 3. 运行 CUDA 独立程序

**SB_CUDA 基础版本：**
```bash
# BSB FP16
cd SB_CUDA
./BSB_GPU_fp16_gset ../data/G22.txt 1000 200 0.1 0.9 0.1

# 参数说明: <gset_file> <num_iter> <batch_size> <dt> <delta> <xi>
```

**SB_CUDA_pure 优化版本：**
```bash
# BSB FP16 (优化版本)
cd SB_CUDA_pure
./BSB_GPU_purefp16_improved.exe ../data/G22.txt 1000 200 0.1 0.9 0.1

# 参数说明: <gset_file> <num_iter> <batch_size> <dt> <delta> <xi>
```

---

## 结果图片说明

### 1. Python 版本结果图

#### 1.1 G22_python_cut_convergence.png - Python 版本切割值收敛图

![G22_python_cut_convergence](png/G22_python_cut_convergence.png)

- **生成脚本**: `SB_solve_Gset.py`
- **图表类型**: 收敛曲线图
- **内容**:
  - BSB 算法：蓝色实线（平均值）+ 虚线圆圈（最大值）
  - DSB 算法：红色实线（平均值）+ 虚线方块（最大值）
- **X 轴**: 对数刻度迭代次数（$10^1$ 到 $10^3$）
- **Y 轴**: 切割值
- **特点**: 展示 Python 版本 BSB 和 DSB 算法在不同迭代次数下的收敛情况，同时显示平均值和最大值

#### 1.2 G22_python_energy_time_convergence.png - Python 版本能量-时间收敛图

![G22_python_energy_time_convergence](png/G22_python_energy_time_convergence.png)

- **生成脚本**: `SB_solve_Gset.py`
- **图表类型**: 能量-时间曲线图
- **内容**:
  - BSB 算法：蓝色实线（平均值）+ 虚线圆圈（最大值）
  - DSB 算法：红色实线（平均值）+ 虚线方块（最大值）
- **X 轴**: 累计时间（毫秒）
- **Y 轴**: 切割值（能量）
- **用途**: 分析 BSB 和 DSB 算法的时间-性能权衡，比较 CPU 实现（Python）的计算效率
- **特点**: 展示 Python 版本 BSB 和 DSB 算法随时间变化的切割值收敛曲线

### 2. CUDA 版本结果图

#### 2.1 G22_cuda_all_cut_convergence.png - 6种算法切割值收敛图 (SB_CUDA版本)

![G22_cuda_all_cut_convergence](png/G22_cuda_all_cut_convergence.png)

- **生成脚本**: `SB_solve_Gset_GPU.py`
- **图表类型**: 多算法收敛曲线图
- **内容**:
  - BSB_FP8: 蓝色实线 + 圆圈
  - BSB_FP16: 青色实线 + 方块
  - BSB_INT8: 绿色实线 + 三角形
  - DSB_FP8: 红色虚线 + 圆圈
  - DSB_FP16: 橙色虚线 + 方块
  - DSB_INT8: 紫色虚线 + 三角形
- **X 轴**: 对数刻度迭代次数（$10^1$ 到 $10^3$）
- **Y 轴**: 切割值
- **用途**: 全面对比 6 种算法在不同精度下的收敛性能

#### 2.2 G22_cuda_all_energy_with_time.png - 6种算法能量-时间收敛图 (SB_CUDA版本)

![G22_cuda_all_energy_with_time](png/G22_cuda_all_energy_with_time.png)

- **生成脚本**: `SB_solve_Gset_GPU.py`
- **图表类型**: 能量-时间曲线图
- **内容**:
  - 所有 6 种算法的收敛曲线
  - 图例中显示每个算法的总执行时间（毫秒）
- **X 轴**: 累计时间（毫秒）
- **Y 轴**: 切割值（能量），从 12500 开始
- **用途**: 分析算法的时间-性能权衡，评估计算效率

### 3. CUDA Pure 版本结果图 (SB_CUDA_pure)

#### 3.1 G22_cuda_all_cut_convergence_pure.png - 6种算法切割值收敛图 (优化版本)

![G22_cuda_all_cut_convergence_pure](png/G22_cuda_all_cut_convergence_pure.png)

- **生成脚本**: `SB_solve_Gset_GPU_pure.py`
- **图表类型**: 多算法收敛曲线图（优化版本）
- **内容**:
  - BSB_PURE_FP8: 蓝色实线 + 圆圈
  - BSB_PURE_FP16: 青色实线 + 方块
  - BSB_PURE_INT8: 绿色实线 + 三角形
  - DSB_PURE_FP8: 红色虚线 + 圆圈
  - DSB_PURE_FP16: 橙色虚线 + 方块
  - DSB_PURE_INT8: 紫色虚线 + 三角形
- **X 轴**: 对数刻度迭代次数（$10^1$ 到 $10^3$）
- **Y 轴**: 切割值
- **用途**: 展示优化版本 CUDA 实现的收敛性能

#### 3.2 G22_cuda_all_energy_with_time_pure.png - 6种算法能量-时间收敛图 (优化版本)

![G22_cuda_all_energy_with_time_pure](png/G22_cuda_all_energy_with_time_pure.png)

- **生成脚本**: `SB_solve_Gset_GPU_pure.py`
- **图表类型**: 能量-时间曲线图（优化版本）
- **内容**:
  - 所有 6 种优化算法的收敛曲线
  - 图例中显示每个算法的总执行时间（毫秒）
- **X 轴**: 累计时间（毫秒）
- **Y 轴**: 切割值（能量）
- **用途**: 分析优化版本算法的时间-性能权衡，评估计算效率提升

---

## 结果对比与分析

### 1. Python 版本 vs CUDA 版本

| 版本 | 实现方式 | 加速方式 | 主要特点 |
|-----|---------|---------|---------|
| Python 版本 | MindQuantum 库 | CPU 串行计算 | 基准实现，便于算法理解和调试 |
| CUDA 版本 | C++ CUDA | GPU 并行计算 | 高性能实现，支持批处理加速 |
| CUDA Pure 版本 | C++ CUDA 优化 | GPU 并行计算 + 算法优化 | 最优性能实现，集成数据读取器 |

### 2. 算法性能对比

#### 2.1 收敛性能

从结果图片可以看出：
- **BSB vs DSB**: BSB 算法通常具有更好的收敛性，能够达到更高的切割值
- **精度影响**:
  - FP16: 平衡精度和性能的最佳选择
  - INT8: 速度最快，但精度略低
  - FP8: 在某些情况下表现优异，但需要精心设计量化策略

#### 2.2 计算效率

从能量-时间图中可以看出：
- **CUDA 版本**相比 Python 版本速度提升显著（约 10-50 倍）
- **Pure 版本**相比基础 CUDA 版本进一步优化性能（约 5-20% 提升）
- **批处理效果**: 批处理大小越大，GPU 利用率越高，计算效率越好

### 3. 不同数据格式的适用场景

| 数据格式 | 精度范围 | 内存占用 | 计算速度 | 适用场景 |
|---------|---------|---------|---------|---------|
| FP32 | 高 | 大 | 慢 | 基准实现，精度要求极高 |
| FP16 | 中 | 中 | 快 | 平衡选择，推荐使用 |
| FP8 | 低 | 小 | 较快 | 内存受限场景 |
| INT8 | 低 | 小 | 最快 | 超大规模问题，速度优先 |

---

## Gset 数据格式

### 文件格式说明

```
<节点数> <边数>
<节点1> <节点2> <边权重>
...
```

### 示例（G22.txt）
```
2000 9994
1 2 1.0
1 3 1.0
...
```

---

## 依赖项

### Python 依赖
```
numpy >= 1.19.0
pandas >= 1.2.0
matplotlib >= 3.3.0
scipy >= 1.5.0
torch >= 1.8.0
mindquantum >= 0.9.0
```

### CUDA 依赖
- CUDA Toolkit >= 10.0
- GPU 架构 >= Volta (sm_70)

---

## 联系方式

- 项目路径: `example/mindquantum_Quantitative_SB/Gset_SB/`
- CUDA 代码（SB_CUDA）: `example/mindquantum_Quantitative_SB/Gset_SB/SB_CUDA/`
- CUDA 代码（SB_CUDA_pure）: `example/mindquantum_Quantitative_SB/Gset_SB/SB_CUDA_pure/`
- 主接口: `example/mindquantum_Quantitative_SB/Gset_SB/SB_solve_Gset_GPU_pure.py`
- 结果图片: `example/mindquantum_Quantitative_SB/Gset_SB/png/`

---

## 结果图片索引

| 图片文件名 | 描述 | 生成脚本 | 版本 |
|-----------|------|---------|------|
| G22_python_cut_convergence.png | Python 版本切割值收敛图 | SB_solve_Gset.py | Python |
| G22_python_energy_time_convergence.png | Python 版本能量-时间图 | SB_solve_Gset.py | Python |
| G22_cuda_all_cut_convergence.png | 6种算法切割值收敛图 | SB_solve_Gset_GPU.py | CUDA (SB_CUDA) |
| G22_cuda_all_energy_with_time.png | 6种算法能量-时间图 | SB_solve_Gset_GPU.py | CUDA (SB_CUDA) |
| G22_cuda_all_cut_convergence_pure.png | 6种算法切割值收敛图（优化） | SB_solve_Gset_GPU_pure.py | CUDA (SB_CUDA_pure) |
| G22_cuda_all_energy_with_time_pure.png | 6种算法能量-时间图（优化） | SB_solve_Gset_GPU_pure.py | CUDA (SB_CUDA_pure) |


