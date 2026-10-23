# 使用指南

[首页](../README.md) · 使用指南 · [基准与性能](benchmarks.md) · [实现原理](design.md)

## 构建

需要 Ascend 910B 系列 NPU、CANN 9.1、Python 3.10 及以上、CMake 3.22 及以上，以及支持 C++17 的编译器。
以下命令均在仓库根目录执行；CANN 路径和 `SOC_VERSION` 按实际环境填写。
旧版用户先按[迁移说明](#从旧版迁移)检查残留扩展。

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -m pip install -r requirement
SOC_VERSION=Ascend910B2 bash run.sh
```

构建成功后，可在仓库根目录导入 `kernel_sb`。下面的示例使用 ACL 逻辑设备 0；
这个编号可能与 `npu-smi` 显示的物理卡号不同。设备不可用时，求解器直接报错。

## 求解一个可以核对答案的问题

考虑三个只能取 −1 或 +1 的变量，其能量为：

$$
E(s)=-s_0s_1+0.5s_0s_2-0.25s_1s_2.
$$

要降低能量，第一项希望 $s_0$ 与 $s_1$ 同号，第二项希望 $s_0$ 与 $s_2$ 异号，第三项又希望 $s_1$ 与 $s_2$ 同号。
这三种偏好不能同时满足，求解需要在它们之间取舍。

把系数写成对称矩阵 `J`，便可交给 Solver。每次搜索从一组小幅随机位置 `x0` 和动量 `y0` 出发；
将 32 组初态放在同一个数组的 32 列中，就能并行进行 32 次搜索。本例没有外场，因此省略 `h`。

```python
import numpy as np
from kernel_sb import Solver

J = np.array([
    [0.0, 1.0, -0.5],
    [1.0, 0.0, 0.25],
    [-0.5, 0.25, 0.0],
], dtype=np.float32)

batch_size = 32
shape = (J.shape[0], batch_size)
rng = np.random.default_rng(42)
x0 = rng.uniform(-0.01, 0.01, shape).astype(np.float32)
y0 = rng.uniform(-0.01, 0.01, shape).astype(np.float32)

with Solver(J, algorithm="dsb", batch_size=batch_size, device_id=0) as solver:
    x, y = solver.run(x0, y0)

spins = np.where(x >= 0, 1, -1)
energies = -0.5 * np.sum(spins * (J @ spins), axis=0)
best = np.argmin(energies)
print("候选自旋：", spins[:, best])
print("候选能量：", energies[best])
```

一次运行的结果是：

```text
候选自旋： [-1 -1  1]
候选能量： -1.25
```

结果里的 `x` 仍是连续位置，按正负转换后才得到自旋；这里将零位置归到 +1。
每一列是一个候选解，`argmin` 选出能量最低的一列。`y` 是最终动量，用于描述迭代状态，不参与这一步评分。

三个变量只有 8 种组合，逐一代入可确认最低能量确实为 −1.25。
`[1, 1, -1]` 也是等价解：零外场下，自旋整体取反不改变能量。

## 保留问题，更换初态

更大的问题通常无法枚举答案，可以通过尝试更多初态寻找更好的候选。问题本身没变时，
只需继续调用同一个 Solver。构造时上传的问题数据和准备好的设备资源，都会被后续调用复用。

沿用上例的 `J`、`batch_size` 和 `shape`，用三个随机种子分别搜索：

```python
with Solver(J, algorithm="dsb", batch_size=batch_size) as solver:
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed)
        x0 = rng.uniform(-0.01, 0.01, shape).astype(np.float32)
        y0 = rng.uniform(-0.01, 0.01, shape).astype(np.float32)
        x, y = solver.run(x0, y0)

        spins = np.where(x >= 0, 1, -1)
        energies = -0.5 * np.sum(spins * (J @ spins), axis=0)
        print(f"seed={seed}, 候选能量={energies.min()}")
```

每次 `run` 都从传入初态执行完整迭代，不会续接上一次搜索。它不修改输入，返回独立的结果数组；
离开 `with` 后释放设备资源。也可以显式调用 `close()`，关闭后不能继续求解。

Solver 在构造时保存 `J`、`h` 和压力序列 `p` 的快照。之后修改原数组不会改变已经准备好的问题；
更换系数、算法或迭代参数时，应重新构造 Solver。改用 BSB 只需把 `algorithm` 设为 `"bsb"`。

如果只求解一次，也可以沿用上例的 `J`，让便捷函数生成初态并管理资源：

```python
from kernel_sb import kernel_bsb

x, y = kernel_bsb(J, batch_size=32, n_iter=1000, seed=42)
```

`kernel_dsb` 的用法相同。两个函数每次都会构造并释放 Solver，因此重复求解时仍应使用前面的复用方式。
显式传入 `x` 或 `y` 时，副本数由初态形状确定，缺失的状态按相同形状随机生成；`seed` 只控制自动生成的部分。

## 接入自己的模型

上例是零外场的特殊情形。一般情况下，Solver 接受耦合矩阵 `J` 和外场 `h`，搜索以下能量函数的低能量自旋：

$$
E(s)=-\frac12 s^TJs-h^Ts,\qquad s_i\in\{-1,+1\}.
$$

对于 `N` 个变量、`B = batch_size` 个副本，输入形状如下：

| 输入 | 形状 | 说明 |
| --- | --- | --- |
| `J` | `(N, N)` | NumPy 数组或 SciPy 稀疏矩阵 |
| `h` | `(N,)` 或 `(N, 1)` | 省略时为零外场 |
| `x0`、`y0` | `(N, B)` | 每列是一组初态；`B=1` 时也接受 `(N,)` |

输入须为有限实数，计算前转换为 FP32。硬件对齐在内部完成，调用者按实际变量数准备数组即可。
返回的 `x`、`y` 始终为 `(N, B)`，是独立、按列连续存储的 FP32 数组。

Ising 模型通常使用对称、对角为零的 `J`。求解器按传入的系数执行更新，不会自动对称化或删除对角项。
SciPy 稀疏输入在构造时转为稠密矩阵，因此不能靠稀疏输入减少设备上的矩阵存储。

有外场时，评分也要包含外场项：`-0.5 * np.sum(spins * (J @ spins), axis=0) - h.reshape(-1) @ spins`。
若原问题采用 0/1 变量或经过系数归一化，最终应回到原始目标函数评分；[QPLIB 示例](benchmarks.md#qplib从原始目标到求解再回到评分)展示了这条完整路径。

### Solver 参数

前面的例子使用默认的 1000 步 FP32 迭代。接入问题后，可从以下参数调整搜索过程。
比较算法或精度时，固定问题与初态，并保持其余参数不变。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `algorithm` | `"dsb"` | `"dsb"` 或 `"bsb"` |
| `batch_size` | `1` | 并行副本数，正整数 |
| `n_iter` | `1000` | 每次 `run` 的迭代步数，正整数 |
| `p` | `None` | 压力序列，形状为 `(n_iter,)`；默认从 0 到 1 线性取值 |
| `delta` | `1.0` | 正数；位置与动量更新中的尺度参数 |
| `dt` | `1.0` | 正数；积分步长 |
| `xi` | 自动计算 | 有限实数；耦合项与外场的缩放系数 |
| `precision` | `"fp32"` | `"fp32"` 或 `"hf32"` |
| `device_id` | `0` | ACL 逻辑设备号 |

默认 $\xi=0.5\sqrt{N-1}/\|J\|_F$，其中 $\|J\|_F$ 是矩阵元素平方和的平方根。
`J` 全为零时必须显式指定 `xi`。这些参数在[更新方程](design.md#一步迭代计算什么)中的位置决定了它们的作用。

HF32 只改变矩阵乘模式，状态存储和更新仍使用 FP32。它可能改变最终解，也不保证更快；
选择前可参考[精度与解质量的实测](benchmarks.md#解质量与精度选择)。

### 调用、计时与线程

`run` 是同步调用，返回时结果已经可用。`solver.last_device_ms` 只记录最近一次执行图的设备耗时，单位为毫秒；
完整调用还包含状态上传、下载和主机处理，测量应用延迟时应计入这些成本。

同一 Solver 的多线程调用会串行执行。不同 Solver 各自持有设备资源；原生求解期间释放 Python GIL，
允许其他 Python 线程继续运行。

## 验证与排错

构建后，可在真机上运行全部正确性测试。`SB_TEST_DEVICE` 同样使用 ACL 逻辑设备号：

```bash
SB_TEST_DEVICE=0 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests
```

只检查 QPLIB 数据处理时不需要 NPU；其中真实数据测试需先[下载对应实例](benchmarks.md#复现实验)：

```bash
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests/test_qplib_data.py
```

| 现象 | 优先检查 |
| --- | --- |
| 构建提示缺少 `ASCEND_HOME_PATH` | 是否在当前终端加载了 CANN 9.1 的 `set_env.sh` |
| 导入不到 `_kernel_sb` | 是否成功运行 `run.sh`，以及是否从仓库根目录启动 Python |
| 设备初始化失败 | CANN 环境、设备可见性、逻辑设备号与编译时的 `SOC_VERSION` |
| 初态形状报错 | 是否把 `(N, B)` 写成了 `(B, N)`，或遗漏 `batch_size` |

`run.sh` 使用 Release 配置并支持增量编译，可通过环境变量设置 `PYTHON_BIN`（默认 `python3`）、
`JOBS`（默认 `8`）和 `SOC_VERSION`（默认 `Ascend910B2`）。脚本只负责构建，不处理数据；
遇到错误时立即停止，保留原有构建目录。

## 从旧版迁移

1. 检查仓库根目录是否残留旧的 `kernel_sb*.so`。确认后移除它，避免遮蔽新的 `kernel_sb.py`；
   **保留带前导下划线的新扩展 `_kernel_sb*.so`**。
2. 将 `solve`、`solve_bsb`、`solve_cached`、`solve_cached_bsb` 调用迁移到 `Solver.run`。
3. 接收 `(x, y)` 返回值。新接口不再原地更新初态；旧的主机指针缓存接口不再保留。
