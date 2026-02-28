# 量化量子启发式算法

> 基于 Simulated Bifurcation 的低比特量化组合优化求解器

本项目实现了基于模拟分岔算法（Simulated Bifurcation, SB）的二次二元优化问题低比特量化求解方案，支持 CPU 和 GPU 平台，实现了 BSB 和 DSB 算法的多种精度变体。

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [快速开始](#快速开始)
- [性能对比](#性能对比)
- [编译指南](#编译指南)
- [使用文档](#使用文档)
- [故障排除](#故障排除)
- [参考文献](#参考文献)

---

## 项目概述

本项目基于 Zhang 等人提出的低比特量化模拟分岔算法理论，实现了 BSB（Ballistic Simulated Bifurcation）和 DSB（Discrete Simulated Bifurcation）算法的 INT8、FP8、FP16 三种精度版本，并在 QPLIB 基准测试集上验证了算法的有效性。

### 应用场景

本算法适用于解决以下类型的组合优化问题：

- 二次二元规划（QUBO）
- 最大割问题（MAX-CUT）
- 伊辛模型优化
- 其他可转化为二次二元优化的问题

### 核心成果

- **24种算法变体**：BSB/DSB × CPU/GPU × FP16/FP8/INT8 × 混合精度/纯精度
- **零精度损失**：混合精度版本求解质量达到 99.29%-100%（Q≥0.9929）
- **显著性能提升**：INT8版本TTT性能比FP16快29%，GPU加速比达14.4-81.9倍
- **完整测试验证**：在23个QPLIB问题上测试，成功率100%

## 核心特性

### 多精度支持

| 精度类型 | 存储格式 | 计算精度 | 内存占用 | 适用场景 |
|---------|---------|---------|---------|---------|
| **FP16** | IEEE 754 half | FP32 | 16-bit | 通用场景，速度与精度平衡 |
| **FP8** | PyTorch float8_e4m3fn | FP32 | 8-bit | 大规模问题，内存敏感 |
| **INT8** | 量化存储 | FP32 | 8-bit | 超大规模问题，最佳速度 |

### 算法对比

| 特性 | BSB (Ballistic) | DSB (Discrete) |
|------|----------------|----------------|
| 变量类型 | 连续值 [-1, 1] | 离散 {0, 1} |
| 梯度计算 | 使用连续 x | 使用 sign(x) |
| 收敛速度 | 更快 | 稍慢 |
| 解质量 | 良好 | 更优 |
| 推荐场景 | 大规模问题 | 中等规模，追求最优解 |

```
mindquantum_Quantitative_SB/
├── BSB/                        # BSB算法实现
│   ├── BSB_CPU_fp16.cu        # CPU FP16版本
│   ├── BSB_CPU_fp8.cu         # CPU FP8版本
│   ├── BSB_CPU_int8.cu        # CPU INT8版本
│   ├── BSB_GPU_fp16.cu        # GPU FP16版本
│   ├── BSB_GPU_fp8.cu         # GPU FP8版本
│   ├── BSB_GPU_int8.cu        # GPU INT8版本
│   └── read_BSB.md            # BSB算法详细文档
├── DSB/                        # DSB算法实现
│   ├── DSB_CPU_fp16.cu
│   ├── DSB_CPU_fp8.cu
│   ├── DSB_CPU_int8.cu
│   ├── DSB_GPU_fp16.cu
│   ├── DSB_GPU_fp8.cu
│   ├── DSB_GPU_int8.cu
│   └── read_DSB.md            # DSB算法详细文档
├── Gset_SB/SB_CUDA/           # Gset数据集GPU实现
│   ├── BSB_GPU_*_gset.cu      # BSB GPU实现 (6种精度)
│   ├── DSB_GPU_*_gset.cu      # DSB GPU实现 (6种精度)
│   ├── sb_cuda_gset_reader.*  # Gset数据读取库
│   └── compile_*.sh/.bat      # 编译脚本
├── data/                      # 最优解数据
├── qplib/                     # QPLIB测试问题文件
├── save_pic/                  # 性能分析图表
├── qplib_parser.h             # QPLIB解析器头文件
├── qplib_parser.py            # QPLIB解析器Python实现
├── SB_basic_run.py            # 单次运行脚本
├── SB_basic_run_23qplibs.py   # 批量测试脚本
├── SB_optuna_tuner.py         # 超参数优化
├── plot_*.py                  # 绘图脚本
├── build_all_linux_cpu.sh     # Linux CPU编译脚本
├── build_all_linux_gpu.sh     # Linux GPU编译脚本
├── compile_single.sh          # 单个算法编译脚本
├── Makefile                   # Makefile构建文件
├── LINUX_BUILD_GUIDE.md       # Linux编译接口汇总
└── README.md                  # 本文件
```

## 环境要求

### 系统要求

| 组件 | CPU版本 | GPU版本 |
|------|---------|---------|
| **操作系统** | Windows/Linux/macOS | Linux (推荐) / Windows |
| **编译器** | MSVC 2019/2022 或 GCC 7+/Clang 5+ | nvcc (CUDA Toolkit) |
| **C++标准** | C++17 | C++17 |
| **Python** | ≥3.7 | ≥3.7 |

### CUDA要求 (GPU版本)

- **CUDA Toolkit**: 10.0 或更高版本
- **NVIDIA驱动**: 支持CUDA的驱动
- **GPU**: NVIDIA GPU (Pascal架构或更高)
  - GTX 10系列: sm_60, sm_61
  - Titan V, V100: sm_70
  - RTX 20系列: sm_75
  - RTX 30系列: sm_80, sm_86
  - RTX 40系列: sm_89

### Python依赖

```bash
# 核心依赖
pip install numpy matplotlib pandas pyyaml

# 可选依赖（超参数优化）
pip install optuna
```

---

## 快速开始

### 5分钟快速指南

#### Linux用户（有GPU）

```bash
# 1. 进入项目目录
cd example/mindquantum_Quantitative_SB

# 2. 编译所有GPU版本
chmod +x build_all_linux_gpu.sh
./build_all_linux_gpu.sh

# 3. 运行测试
./BSB/BSB_GPU_int8 qplib/QPLIB_3565.qplib
```

#### Linux用户（无GPU）

```bash
# 1. 进入项目目录
cd example/mindquantum_Quantitative_SB

# 2. 编译所有CPU版本
chmod +x build_all_linux_cpu.sh
./build_all_linux_cpu.sh

# 3. 运行测试
./BSB/BSB_CPU_fp16 qplib/QPLIB_3565.qplib
```

#### Windows用户 (PowerShell)

```powershell
# 1. 进入BSB目录
cd BSB

# 2. 编译FP16版本
nvcc -O3 -arch=native BSB_CPU_fp16.cu -o BSB_CPU_fp16.exe

# 3. 运行
.\BSB_CPU_fp16.exe ..\qplib\qplib_3506.qplib
```

### 使用Makefile编译（Linux）

```bash
# 查看所有可用命令
make help

# 编译所有CPU版本
make cpu

# 编译所有版本（CPU+GPU）
make gpu

# 清理编译产物
make clean

# 重新编译
make rebuild

# 并行编译（4个线程）
make -j4 gpu
```

### 编译单个算法

```bash
# 使用脚本
./compile_single.sh BSB_CPU_fp16
./compile_single.sh DSB_GPU_fp8

# 使用Makefile
make compile-BSB_CPU_fp16
make compile-DSB_GPU_int8
```

### 批量测试

```python
# 批量测试23个QPLIB问题
python SB_basic_run_23qplibs.py --algorithm BSB --precision int8 --device GPU

# 查看结果
python plot_batch_results.py
```

---

## 性能对比

### 核心算法推荐

| 场景 | 推荐实现 | TTT时间 | Q值 | 加速比 |
|------|---------|---------|-----|--------|
| **最快速度** | BSB_GPU_int8 | 78ms | 0.9929 | - |
| **最高精度** | DSB_GPU_int8 | 378ms | 1.0000 | - |
| **CPU环境** | BSB_CPU_int8 | 7501ms | 0.9929 | - |
| **内存受限** | BSB_GPU_fp8 | 79ms | 0.9929 | 4x压缩 |

### 性能对比表

| 算法版本 | 求解质量Q | TTT时间 | 精度损失 | vs CPU加速比 |
|---------|----------|---------|---------|-------------|
| BSB_GPU_int8 | 0.9929 | 78ms | 0% | 81.9× |
| BSB_GPU_fp8 | 0.9929 | 79ms | 0% | 80.7× |
| BSB_GPU_fp16 | 0.9929 | 101ms | 0% | 63.1× |
| DSB_GPU_fp16 | 1.0000 | 284ms | 0% | 14.4× |
| BSB_CPU_fp16 | 0.9716 | 7054ms | 0% | 1× |

### 关键发现

- **INT8优势**: TTT性能比FP16快29%，零精度损失
- **混合精度**: 所有混合精度版本零精度损失
- **GPU加速**: 最高达81.9倍加速比
- **内存优化**: FP8/INT8实现4倍内存压缩

### TTT指标说明

TTT（Time to Target）衡量算法达到目标解所需的时间，考虑了智能重启策略的效果。

**TTT性能排名**:
1. BSB_GPU_int8: 78ms
2. BSB_GPU_fp8: 79ms
3. BSB_GPU_fp16: 101ms
4. DSB_GPU_fp16: 284ms

---

## 编译指南

### 编译选项说明

#### CPU版本

```bash
# 使用nvcc（推荐）
nvcc -O3 -arch=native filename.cu -o filename

# 使用g++
g++ -O3 -std=c++17 -march=native -fopenmp filename.cu -o filename
```

**编译参数**:
- `-O3`: 最高优化级别
- `-std=c++17`: C++17标准
- `-arch=native`: 使用本地CPU指令集优化
- `-fopenmp`: OpenMP并行计算支持

#### GPU版本

```bash
nvcc -O3 -arch=sm_70 filename.cu -o filename
```

**编译参数**:
- `-arch=sm_70`: 针对Volta架构（V100）
- 其他架构: `sm_60` (Pascal), `sm_75` (Turing), `sm_80` (Ampere), `sm_86` (Ampere), `sm_89` (Ada)

### 查询GPU架构

```bash
# 查询GPU计算能力
nvidia-smi --query-gpu=compute_cap --format=csv

# 查询完整GPU信息
nvidia-smi
```

### 修改GPU架构

编辑编译脚本中的 `GPU_OPTS` 变量：

```bash
# 例如，针对RTX 3090 (sm_86)
GPU_OPTS="-O3 -std=c++17 -arch=sm_86"
```

### 清理编译产物

```bash
# 删除BSB可执行文件
rm -f BSB/BSB_*

# 删除DSB可执行文件
rm -f DSB/DSB_*

# 使用Makefile清理
make clean
```

---

## 使用文档

### 运行参数说明

```bash
# 基本用法
./BSB_GPU_fp16 <qplib_file>

# 示例
./BSB_GPU_fp16 qplib/QPLIB_3565.qplib
```

**程序参数**:
- 第1个参数: QPLIB数据文件路径（必需）

### 输出说明

程序会输出以下信息：
- 读取的问题规模（变量数、约束数）
- 算法配置（精度、迭代次数、批次大小等）
- 求解结果（目标函数值、计算时间）
- 求解质量（Q值，相对于已知最优解的比例）

### Gset数据集支持

项目同时支持Gset数据集的MAX-CUT问题求解。详见 `Gset_SB/SB_CUDA` 目录。

**编译Gset版本**:

```bash
cd Gset_SB/SB_CUDA
chmod +x compile_all_cuda.sh
./compile_all_cuda.sh
```

**运行Gset版本**:

```bash
# 基本用法
./BSB_GPU_fp8_gset <gset_file> [n_iter] [batch_size] [dt] [delta]

# 示例
./BSB_GPU_fp8_gset ../data/G22.txt 1000 100 0.5 0.9
```

**参数说明**:
- 第1个参数: Gset数据文件路径（必需）
- 第2个参数: 迭代次数 n_iter（默认: 1000）
- 第3个参数: 批处理大小 batch_size（默认: 100）
- 第4个参数: 时间步长 dt（默认: 0.5）
- 第5个参数: 阻尼系数 delta（默认: 0.9）
- 第6个参数: 耦合强度 xi（默认: 自动计算）

### Python脚本集成

使用Python脚本调用所有实现：

```bash
cd Gset_SB
python SB_solve_Gset_GPU.py
```

Python脚本会：
1. 自动检测并调用CUDA可执行文件
2. 如果可执行文件不存在，则回退到Python实现
3. 绘制收敛曲线和性能对比图
4. 保存结果到`pig/`目录

---

## 故障排除

### 编译问题

#### CUDA版本不匹配

**症状**: 编译时提示CUDA版本相关错误

**解决方案**:
```bash
# 检查nvcc版本
nvcc --version

# 检查GPU驱动版本
nvidia-smi

# 确保nvcc版本与GPU驱动兼容
# 通常需要升级CUDA Toolkit或降级GPU驱动
```

#### 找不到头文件

**症状**: `fatal error: qplib_parser.h: No such file or directory`

**解决方案**:
- 确保在项目根目录下执行编译命令
- 检查 `qplib_parser.h` 是否存在于项目根目录

#### GCC版本不兼容

**症状**: `no supported gcc/g++ host compiler found`

**解决方案**:
- 降低g++版本（如GCC 7.x）或升级CUDA版本
- Ubuntu: `sudo apt-get install gcc-7 g++-7`

#### command not found

**症状**: `command not found: g++` 或 `command not found: nvcc`

**解决方案**:
```bash
# Ubuntu/Debian - 安装g++
sudo apt-get update
sudo apt-get install build-essential g++

# CentOS/RHEL - 安装g++
sudo yum groupinstall "Development Tools"
sudo yum install gcc-c++

# 安装CUDA Toolkit
# 访问 https://developer.nvidia.com/cuda-downloads
```

### 运行问题

#### CUDA out of memory

**症状**: 运行时提示CUDA显存不足

**解决方案**:
- 减小 `batch_size` 参数
- 减少问题规模
- 使用INT8/FP8精度降低内存占用

#### no CUDA-capable device is detected

**症状**: 程序无法检测到GPU

**解决方案**:
```bash
# 检查GPU状态
nvidia-smi

# 检查nvcc版本
nvcc --version

# 确保GPU驱动已正确安装
```

#### Python脚本找不到可执行文件

**症状**: Python运行时提示找不到CUDA可执行文件

**解决方案**:
- 确认已编译所有CUDA文件
- 检查可执行文件是否在正确的目录下
- Windows系统需要`.exe`后缀

### 其他资源

- **[BSB/read_BSB.md](./BSB/read_BSB.md)**: BSB算法详细文档
- **[DSB/read_DSB.md](./DSB/read_DSB.md)**: DSB算法详细文档
- **[项目报告.md](./项目报告.md)**: 完整技术报告
- **[LINUX_BUILD_GUIDE.md](./LINUX_BUILD_GUIDE.md)**: Linux编译详细指南

---

## 技术细节

### GPU优化特性

所有GPU实现包含以下优化：

1. **CUDA Stream异步执行**: 使用异步CUDA流减少同步开销
2. **循环展开4次**: 提高指令级并行
3. **__ldg()缓存优化**: 利用GPU只读缓存
4. **FMA融合乘加**: 使用硬件融合乘加指令
5. **异步内存传输**: `cudaMemcpyAsync`减少CPU-GPU同步
6. **减少同步点**: 只在必要时同步

---

## 参考文献

1. Zhang, Y., & Han, L. (2023). Low-bit quantization of simulated bifurcation algorithms for combinatorial optimization. *arXiv preprint*.

2. Zeng, Y., et al. (2022). Dynamical bifurcation in quantum-inspired optimization algorithms. *Nature Communications*, 13, 1-12.

3. Goto, E., et al. (2019). Combinatorial optimization by simulating adiabatic bifurcations in nonlinear Hamiltonian systems. *Science Advances*, 5(4), eaav2372.

4. Goto, E., et al. (2021). High-performance combinatorial optimization based on classical mechanics. *Science Advances*, 7(9), eabe7953.

5. QPLIB. (2019). QPLIB: A Library of Quadratic Programming Instances. *Mathematical Programming Computation*, 11(1), 1-36.

<<<<<<< HEAD
## 许可证

本项目遵循MindQuantum项目的许可证。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue
- 发送邮件至项目维护者

---

## Linux编译快速参考

### 无GPU用户（仅CPU版本）

| 方法 | 命令 | 说明 |
|------|------|------|
| Bash脚本 | `./build_all_linux_cpu.sh` | 编译所有6个CPU版本 |
| Makefile | `make cpu` | 使用Make编译CPU版本 |
| 单个编译 | `./compile_single.sh BSB_CPU_fp16` | 编译单个算法 |

### 有GPU用户（CPU+GPU版本）

| 方法 | 命令 | 说明 |
|------|------|------|
| Bash脚本 | `./build_all_linux_gpu.sh` | 编译所有12个版本 |
| Makefile | `make gpu` | 使用Make编译所有版本 |
| 单个编译 | `./compile_single.sh BSB_GPU_fp8` | 编译单个GPU算法 |

### 编译方法对比

**方法1: Bash Shell脚本（推荐）**
- 文件：`build_all_linux_cpu.sh`、`build_all_linux_gpu.sh`、`compile_single.sh`
- 优点：简单易用，详细输出，自动检测环境
- 适用：初学者，不熟悉Makefile的用户

**方法2: Makefile**
- 文件：`Makefile`
- 优点：增量编译，支持并行编译，灵活依赖管理
- 适用：熟悉Makefile的用户，频繁重新编译

**方法3: 手动编译**
```bash
# CPU版本
g++ -O3 -std=c++17 -march=native -fopenmp -I.. BSB/BSB_CPU_fp16.cu -o BSB/BSB_CPU_fp16

# GPU版本
nvcc -O3 -std=c++17 -arch=sm_70 -I.. BSB/BSB_GPU_fp16.cu -o BSB/BSB_GPU_fp16
```

### Makefile常用命令

```bash
make help              # 查看帮助
make                   # 编译所有CPU版本（默认）
make cpu              # 编译CPU版本
make gpu              # 编译所有版本（CPU+GPU）
make clean            # 清理编译产物
make rebuild          # 清理并重新编译
make compile-BSB_CPU_fp16  # 编译单个算法
make -j4 gpu         # 使用4个并行任务编译
```

### 测试编译结果

编译完成后，可以直接运行程序测试：

```bash
# 测试CPU版本
BSB/BSB_CPU_fp16 qplib/QPLIB_3565.qplib
DSB/DSB_CPU_int8 qplib/QPLIB_3565.qplib

# 测试GPU版本
BSB/BSB_GPU_fp8 qplib/QPLIB_3565.qplib
DSB/DSB_GPU_fp16 qplib/QPLIB_3565.qplib
```

---

## Gset数据集GPU实现（SB_CUDA）

本项目的 `Gset_SB/SB_CUDA` 子目录包含针对Gset数据集的6种GPU精度实现。

### CUDA实现文件（6种精度）

1. **BSB_GPU_fp8_gset.cu** - Ballistic SB算法，FP8精度（PyTorch float8_e4m3fn）
2. **BSB_GPU_fp16_gset.cu** - Ballistic SB算法，FP16精度（PyTorch torch.float16）
3. **BSB_GPU_int8_gset.cu** - Ballistic SB算法，INT8精度（量化）
4. **DSB_GPU_fp8_gset.cu** - Discrete SB算法，FP8精度（PyTorch float8_e4m3fn）
5. **DSB_GPU_fp16_gset.cu** - Discrete SB算法，FP16精度（PyTorch torch.float16）
6. **DSB_GPU_int8_gset.cu** - Discrete SB算法，INT8精度（量化）

### 头文件和辅助文件

- **sb_cuda_gset_reader.h / sb_cuda_gset_reader.c** - Gset数据读取库

### 编译脚本

- **compile_all_cuda.bat** - Windows批处理编译脚本
- **compile_all_cuda.sh** - Linux/Mac Shell编译脚本

### 编译方法

#### Windows系统
```bash
cd Gset_SB/SB_CUDA
compile_all_cuda.bat
```

#### Linux/Mac系统
```bash
cd Gset_SB/SB_CUDA
chmod +x compile_all_cuda.sh
./compile_all_cuda.sh
```

#### 手动编译（单独编译某个文件）
```bash
# FP8实现
nvcc BSB_GPU_fp8_gset.cu -o BSB_GPU_fp8_gset -arch=sm_70
nvcc DSB_GPU_fp8_gset.cu -o DSB_GPU_fp8_gset -arch=sm_70

# FP16实现
nvcc BSB_GPU_fp16_gset.cu -o BSB_GPU_fp16_gset -arch=sm_70
nvcc DSB_GPU_fp16_gset.cu -o DSB_GPU_fp16_gset -arch=sm_70

# INT8实现
nvcc BSB_GPU_int8_gset.cu -o BSB_GPU_int8_gset -arch=sm_70
nvcc DSB_GPU_int8_gset.cu -o DSB_GPU_int8_gset -arch=sm_70
```

**注意**:
- Windows系统生成的可执行文件会自动添加`.exe`后缀
- `-arch=sm_70`指定GPU计算能力，可根据实际GPU调整（如sm_52, sm_60, sm_75, sm_80等）
- 编译需要NVIDIA CUDA Toolkit和兼容的GPU

### 使用方法

#### 单独运行某个可执行文件
```bash
# Windows
BSB_GPU_fp8_gset.exe ..\data\G22.txt 1000 100 0.5 0.9

# Linux/Mac
./BSB_GPU_fp8_gset ../data/G22.txt 1000 100 0.5 0.9
```

#### 参数说明
- 第1个参数: Gset数据文件路径（必需）
- 第2个参数: 迭代次数 n_iter（默认: 1000）
- 第3个参数: 批处理大小 batch_size（默认: 100）
- 第4个参数: 时间步长 dt（默认: 0.5）
- 第5个参数: 阻尼系数 delta（默认: 0.9）
- 第6个参数: 耦合强度 xi（默认: 自动计算）

#### 输出示例
```
Reading Gset file: ../data/G22.txt
  Number of vertices: 2000
  Number of edges: 9998
  Negate weights: True

Auto-calculated xi: 0.123456

Solving MAX-CUT problem...
  - Variables: 2000, Edges: 9998
  - DSB Algorithm (INT8, discrete variables via sign)
  - INT8 Format: Scaled quantization with 100x scaling factor
  - Storage: INT8, Computation: FP32 (high precision)
  - Initialization: normal distribution, sigma=1.0
  - Using xi=0.123456, dt=0.5
  - Iterations: 1000, Batch size: 100
DSB INT8 CUDA Solver Results:
  Time: 1234 ms
  Max cut: 13567.89
  Iterations: 1000
  Batch size: 100
  xi: 0.123456
  dt: 0.5
  delta: 0.9
```

#### Python脚本集成

使用`SB_solve_Gset_GPU.py`调用所有6种实现：

```python
cd Gset_SB
python SB_solve_Gset_GPU.py
```

Python脚本会：
1. 自动检测并调用SB_CUDA目录中的可执行文件
2. 如果可执行文件不存在，则回退到Python实现
3. 绘制收敛曲线和性能对比图
4. 保存结果到`pig/`目录

### 精度说明

#### FP8 (float8_e4m3fn)
- 格式：1位符号 + 4位指数 + 3位尾数
- 范围：约[-448, 448]
- 特点：PyTorch原生支持，推理优化

#### FP16 (IEEE 754 half)
- 格式：1位符号 + 5位指数 + 10位尾数
- 范围：[-65504, 65504]
- 特点：IEEE 754标准，广泛支持

#### INT8 (量化)
- 格式：8位整数，范围[-127, 127]
- 特点：量化存储，FP32计算，平衡精度和速度

### GPU优化特性

所有6种实现都包含以下优化：
1. **CUDA Stream异步执行**: 使用异步CUDA流减少同步开销
2. **循环展开4次**: 提高指令级并行
3. **__ldg()缓存优化**: 利用GPU只读缓存
4. **fma融合乘加**: 使用硬件融合乘加指令
5. **异步内存传输**: `cudaMemcpyAsync`减少CPU-GPU同步
6. **减少同步点**: 只在必要时同步

### BSB vs DSB算法区别

| 特性 | BSB | DSB |
|------|-----|-----|
| 梯度计算 | 使用连续x | 使用sign(x) |
| 自旋更新 | 连续值 [-1,1] | 连续值 [-1,1] |
| 切割计算 | 使用连续x | 使用sign(x) |
| 泵浦项 | -(δ-p)*x | -(δ-p)*x |

#### BSB (Ballistic Simulated Bifurcation)
- 使用连续的自旋变量
- 适合大规模问题
- 收敛速度较快

#### DSB (Discrete Simulated Bifurcation)
- 使用离散的{0,1}二值变量
- 适合中等规模问题
- 通常能获得更好的解

### 算法对比

| 算法 | 精度 | 存储格式 | 计算精度 | 内存占用 | 速度 | 适用场景 |
|------|------|----------|----------|----------|------|----------|
| BSB_FP8 | 8-bit | PyTorch FP8 | FP32 | 最小 | 最快 | 大规模稀疏图 |
| BSB_FP16 | 16-bit | FP16 | FP32 | 小 | 快 | 中等规模图 |
| BSB_INT8 | 8-bit | INT8量化 | FP32 | 最小 | 最快 | 超大规模图 |
| DSB_FP8 | 8-bit | PyTorch FP8 | FP32 | 最小 | 最快 | 大规模稀疏图 |
| DSB_FP16 | 16-bit | FP16 | FP32 | 小 | 快 | 中等规模图 |
| DSB_INT8 | 8-bit | INT8量化 | FP32 | 最小 | 最快 | 超大规模图 |

### 性能对比

不同精度的性能特点：
- **FP8**: 最快速度，但精度最低，适合大规模问题
- **FP16**: 速度和精度的良好平衡，推荐使用
- **INT8**: 速度与FP16相近，整型计算可能有特殊优势

### 性能优化建议

1. **小规模图**（N < 1000）：使用FP16精度
2. **中等规模图**（1000 < N < 10000）：使用FP8或FP16精度
3. **大规模图**（N > 10000）：使用INT8精度
4. **超大规模图**（N > 100000）：使用INT8精度并增加批次大小

### 故障排除

#### 编译错误
- 检查CUDA Toolkit是否正确安装: `nvcc --version`
- 检查GPU驱动是否支持CUDA: `nvidia-smi`
- 确认GPU计算能力，调整`-arch`参数

#### 运行时错误
- 检查Gset数据文件路径是否正确
- 确认GPU显存足够（batch_size * num_spins）
- 尝试减小batch_size参数

#### Python脚本找不到可执行文件
- 确认已编译所有CUDA文件
- 检查可执行文件在SB_CUDA目录下
- Windows系统需要.exe后缀

**最后更新**: 2026年2月6日
**版本**: v1.3
=======
>>>>>>> feature/quantitative-sb
