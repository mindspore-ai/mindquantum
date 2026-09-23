# 基准与性能

[首页](../README.md) · [使用指南](usage.md) · 基准与性能 · [实现原理](design.md)

## 重复求解的收益与首次使用的成本

在 Ascend 910B2 上，新版 DSB 的主要实测收益如下。两组问题均采用 FP32、100 个副本、1000 步；
时间为预热后完整同步调用的中位数，包含初态上传和结果下载。

| 问题 | 旧 DSB | 新 DSB | DSB 加速比 | 新 BSB |
| --- | ---: | ---: | ---: | ---: |
| QPLIB_3506，496 变量 | 228.50 ms | 15.96 ms | 14.3× | 15.04 ms |
| 稠密 Ising，1024 变量 | 528.91 ms | 25.48 ms | 20.8× | 25.02 ms |

这些数字衡量的是**反复求解同一个问题的成本**。旧版 `solve_cached(force_copy=True)` 每次上传 J/h/p，
新版 `Solver.run` 复用构造时的快照，因此加速既来自计算路径，也包含复用问题数据的收益。
新 DSB 的纯设备中位数分别为 15.18 ms、24.64 ms。旧 BSB 存在未初始化缓冲区问题，未将其作为加速比基线。

首次使用还需要准备 Solver。所测 QPLIB 实例的构造中位数约 232 ms，进程首次构造约 1.5–1.6 s。
只求解一次时，这笔成本需要完整承担；反复更换初态时，才能逐渐分摊它。

测量日期为 2026-09-23，环境为 CANN 9.1、Python 3.11.13、NumPy 1.24.0、SciPy 1.15.3。
新旧 DSB 使用相同的 J/h/p、参数和初态；旧版测量 3 次，新版测量 5 次。
结果适用于上述工作负载。小问题可能受调度开销主导，大而极稀疏的问题则可能更适合稀疏 CPU 实现。

## 解质量与精度选择

速度需要和搜索结果一起看。全量 QPLIB 测试覆盖 23 个实例、两种算法、两种精度，共 92 条配置记录。
每条记录包含三个随机种子的试验，每轮 CPU 与 NPU 使用相同的初态、参数和压力序列。

FP32 相对本仓库**单线程 SciPy 稀疏 CPU**基线的结果如下。加速比取各实例的中位数；
目标值比较的是每轮 batch 中最好的候选，而不是每个副本的轨迹。

| 算法 | 重复求解加速比 | 与同初态 CPU 最好目标值一致的试验 |
| --- | ---: | ---: |
| DSB | 52.1× | 69 / 69 |
| BSB | 37.8× | 69 / 69 |

目标值一致说明这些试验得到了同样好的候选，不代表中间状态逐元素相同，也不构成全局最优证明。

HF32 相对 FP32 的加速比中位数仅为 DSB 1.02×、BSB 1.00×，且分别改变了 3 个、4 个实例的部分目标值。
在这组问题上，降低矩阵乘精度几乎没有带来速度收益，却改变了搜索结果，所以 FP32 保持为默认模式。

Gset 测试进一步覆盖了 G22、G30、G50、G60。最大的 G60 有 7000 个变量，FP32 下重复求解中位数为
DSB 195.01 ms、BSB 197.96 ms。四个实例中，FP32 DSB 的最好割值都与同初态 CPU 一致；
BSB 在 G30、G60 的部分试验中不同。短轨迹已经通过独立数值校验，但长时间迭代可能放大浮点计算的微小差异，
因此仍需单独检查最终解质量。

## 复现实验

先完成[构建](usage.md#构建)，再在仓库根目录执行以下命令。
`test_qplib.py`、`test_gset.py`、`test.py` 是性能基准入口；正确性测试另放在 `tests/` 中。

### 先用一个 QPLIB 实例试跑

[QPLIB](https://qplib.zib.de/) 是二次规划问题库，本仓库选取其中 23 个无约束、全二元实例。
先下载 QPLIB_3506，并各运行一轮 DSB、BSB：

```bash
python3 qplib.py download 3506
OPENBLAS_NUM_THREADS=1 python3 test_qplib.py --instances 3506 --repeat 1
```

输出为两行 JSON，分别对应两种算法。每行既有耗时，也有该轮最好的原始目标值。
试跑成功后，可下载全部所选实例，比较两种精度并保存结果：

```bash
python3 qplib.py download
OPENBLAS_NUM_THREADS=1 python3 test_qplib.py \
  --precision fp32 hf32 > data/qplib-current.jsonl
```

下载文件保存在 `data/`。不带编号时下载全部，也可传入多个编号，例如 `python3 qplib.py download 3506 3850`。
下载器校验并复用已有文件，新文件验证通过后才写入目标路径。无效的已有文件会保留并报错；
下载遇错立即停止，不重试、跟随重定向或切换镜像。

### 比较 Gset 与合成稠密问题

Gset 用于最大割测试。从 [Gset 官网](https://web.stanford.edu/~yyye/yyye/Gset/) 获取 G22、G30、G50 或 G60，
放入 `data/gset/`；脚本不自动下载这些文件。以 G22 为例：

```bash
OPENBLAS_NUM_THREADS=1 python3 test_gset.py \
  --graph data/gset/G22 --precision fp32 hf32 > data/gset-G22-current.jsonl
```

合成稠密问题直接生成矩阵，不依赖下载数据：

```bash
mkdir -p data
OPENBLAS_NUM_THREADS=1 python3 test.py \
  --size 1024 --cpu dense > data/dense-current.jsonl
```

这些命令把新结果写入 `*-current.jsonl`，与历史实测档案分开。再次运行同一条重定向命令会替换该结果文件。

### 选择比较范围

三个入口共用以下选项。默认的三轮试验使用种子 42、43、44，每轮 100 个副本、1000 步。

| 选项 | 默认值 | 说明 |
| --- | --- | --- |
| `--algorithm` | `dsb bsb` | 运行一种或两种算法 |
| `--precision` | `fp32` | 可指定 `fp32 hf32` 做精度对比 |
| `--batch` / `--niter` | `100` / `1000` | 并行副本数 / 迭代步数 |
| `--repeat` / `--seed` | `3` / `42` | 试验次数 / 起始随机种子 |
| `--cpu` | 按入口选择 | `sparse`、`dense` 或 `none` |
| `--device-id` / `--dt` | `0` / `1.0` | ACL 逻辑设备号 / 积分步长 |

QPLIB 和 Gset 默认对照 SciPy 稀疏 CPU 实现，合成问题默认对照 NumPy 稠密 CPU 实现。
`--cpu none` 跳过 CPU 对照，仍在 NPU 上求解。其他参数见各入口的 `--help`。

## 一条结果记录包含什么

输出采用 JSONL 格式，每行对应一个“问题 × 算法 × 精度”配置。同一条记录将准备、首次执行和后续调用分开计时：

| 字段 | 测量范围 |
| --- | --- |
| `setup_ms` | 构造 Solver，包括校验、上传、算子预热和执行图捕获 |
| `first_run_ms` | 构造后的第一次完整 `run` |
| `wall_ms` | 后续各轮完整 `run`，包括状态校验、分配、上传和下载 |
| `device_ms` | 各轮执行图的设备时间，不含状态传输 |
| `best_objectives` | 各轮 batch 中最好的原始目标值 |

`wall_median_ms` 和 `device_median_ms` 是相应数组的中位数。
启用 CPU 对照时，`cpu_speedup` 等于 CPU 耗时中位数除以 NPU 的 `wall_median_ms`。
后者反映复用 Solver 的调用延迟；首次使用的成本则由 `setup_ms` 与 `first_run_ms` 共同记录。

下载、文件解析、初态生成、目标值评分和 Solver 资源释放不在上述计时区间内。
将求解器接入应用后，端到端计时还需覆盖实际请求中的这些工作。

### QPLIB：从原始目标到求解，再回到评分

QPLIB 的二元变量取 0/1，Solver 的自旋取 ±1，两者通过 \(z=(s+1)/2\) 转换。
`to_ising()` 将最大化或最小化方向统一到低能量搜索，并归一化系数；
`problem.objective` 则使用原始系数，以 FP64 计算候选解的分数。归一化能量用于求解，原始目标值用于报告。

下载 QPLIB_3506 后，可以用以下代码单独走完转换、求解和评分：

```python
import numpy as np
from kernel_sb import kernel_dsb
from qplib import load_qplib

problem = load_qplib("data/QPLIB_3506.qplib")
J, h, _ = problem.to_ising()
x, y = kernel_dsb(J, h, batch_size=100, seed=42)
values = problem.objective((x >= 0).astype(np.float64))
best = values.max() if problem.sense == "maximize" else values.min()
print("原始目标值：", best)
```

QPLIB_3506 的官网可行解目标值为 478，可作为评分实现的核对点。Gset 则直接报告割值，越大越好。

### QPLIB 文件中的系数约定

官网文件存储下三角系数 T，目标函数为：

\[
f(z)=\frac12\sum_{i\ge j}T_{ij}z_i z_j+b^Tz+c.
\]

记录 `2 1 4.0` 乘以前面的 1/2 系数后，贡献为 \(2z_2z_1\)。
若把原值镜像到对称矩阵，再代入带 1/2 系数的二次型，就会把这一项计算两次。
转换还需要把二元对角项移入外场与常数，保留孤立变量，并处理优化方向。

这些工作由纯 CPU 模块 `qplib.py` 完成。解析器支持默认线性项及覆盖值、对角项、孤立变量和常数项，
只接受无约束、全二元问题；无效记录或不支持的类型会直接报错。
原始目标评分与 Ising 转换分开实现，测试既核对转换关系，也使用官网给出的解验证评分。

## 进一步分析设备耗时

使用 CANN 的 `msprof` 可以区分矩阵乘与状态更新的成本。采样时采用较短的 100 步序列，减少 trace 数据量：

```bash
OPENBLAS_NUM_THREADS=1 msprof \
  --output=data/npu-profile-current \
  --task-time=on --aic-metrics=PipeUtilization \
  --application="$(command -v python3) $PWD/test.py \
    --size 1024 --algorithm dsb --precision fp32 \
    --niter 100 --repeat 1 --cpu none"
```

在已记录的 1024 变量 FP32 DSB 采样中，矩阵乘任务耗时中位数为 16.44 µs，融合更新为 8.22 µs，
分别使用 AI Core 和 Vector Core。这些数值只统计执行图内的任务；构造时的预热任务单独识别，
其 `Model ID` 在 `op_summary` 中为 `4294967295`。

## 实测档案

以下档案保留在测量工作区的 `data/` 中，未纳入 Git。
克隆仓库后，可用本页命令生成当前实现的结果；旧版基线需要对应的旧版实现与匹配输入。

| 档案 | 内容 |
| --- | --- |
| `data/baseline.json`、`data/baseline_*.npz` | 旧版计时及匹配输入 |
| `data/optimized-comparison.json` | 新版五次重复调用的计时 |
| `data/qplib-results.jsonl` | 全量 QPLIB 结果 |
| `data/gset-G*-results.jsonl` | 四个 Gset 实例的结果 |
| `data/npu-profile-final/` | 算子级 profiling 数据 |

## 数据来源与许可

QPLIB 数据来自 [qplib.zib.de](https://qplib.zib.de/)，采用 [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/) 许可。
发布使用这些数据的结果时，请注明来源并引用 [Furini et al., QPLIB](https://doi.org/10.1007/s12532-018-0147-4)。
Gset 数据来自 [Stanford Gset](https://web.stanford.edu/~yyye/yyye/Gset/)。
