# Ascend 上的批量 DSB / BSB

面向 Ascend 910B 的模拟分岔求解器。**CANN 9.1、Python ≥ 3.10、CMake ≥ 3.22、C++17**。
默认保留 FP32 计算；可显式选择 HF32。没有 CPU 回退、按主机指针推测的缓存，或自动降精度。

## 计算架构

对整个 batch 做一次矩阵乘，而不是每个副本分别逐行点积：

- **Cube**：`(J @ X).T = X.T @ J.T`；DSB 将 X 换为 `sign(X)`。调用 CANN
  `aclnnMatmul`，由算子库负责设备相关 tiling，而不是再维护一套通用 GEMM。
- **Vector**：一个 AscendC 内核融合动量更新、位置更新、越界裁剪，以及下一轮 DSB 的符号生成。
- **执行图**：构造时捕获完整迭代，求解时一次 `aclmdlRIExecuteAsync` 重放。
  每次求解只有初态上传与最终状态下载，迭代之间没有主机同步或数据往返。
- **资源**：每个 `Solver` 拥有独立 context、stream、J/h/p 快照、状态、工作区、页锁定主机缓冲和执行图。
  `close()`/上下文管理器按依赖顺序释放资源。同一 Solver 的并发调用串行化，并释放 Python GIL。

状态对外为 `(N, batch)`，对内按副本连续存储。N 统一补零到 64：同时满足
DMA、Cube 和 910B 比较指令的完整 256-byte FP32 repeat 契约。h 在构造时按 batch 展开，
使 Vector 内核能跨副本均匀分工，每核按最多 4096 元素分块。显式 UB 分配约 112.5 KiB，
另保留 CANN 要求的 8 KiB 临时区，**不再要求整个问题能放进 UB**。密集 J 仍占用
\(O(N^2)\) 设备内存；稀疏输入只在构造时转为密集矩阵，当前不是稀疏 NPU 求解器。

两种算法使用同一组方程，只有 \(u_t\) 不同：DSB 为 \(\operatorname{sign}(x_t)\)，BSB 为 \(x_t\)。

\[
v = y_t + dt\big[(p_t-\delta)x_t + \xi(Ju_t+h)\big],\qquad
z = x_t + dt\,\delta v.
\]

若 \(|z|>1\)，令 \(x_{t+1}=\operatorname{sign}(z), y_{t+1}=0\)；否则取 \(z,v\)。
**恰好等于 ±1 时不清零动量；sign(0)=0。** 默认
\(\xi=0.5\sqrt{N-1}/\|J\|_F\)，零 J 必须显式指定 xi。

## 构建和验证

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 -m pip install -r requirement
SOC_VERSION=Ascend910B2 bash run.sh

# 无需设备的 QPLIB 解析、映射和下载器测试
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests/test_qplib_data.py
# 真机数值、边界、执行图重放、资源隔离和线程测试
SB_TEST_DEVICE=0 OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests
```

`SOC_VERSION` 必须与硬件匹配；`ASCEND_HOME_PATH` 来自 CANN 环境。
`device_id` 是 **ACL 逻辑设备号**，不一定等于 `npu-smi` 的物理卡号。
可用 `PYTHON_BIN` 和 `JOBS` 设置构建用 Python 和并行度。
构建使用 Release，支持增量构建，不自动删除目录、下载数据或运行基准。
从旧版迁移时，先移除旧的根目录 `kernel_sb*.so`，以免它遮蔽新的 Python API。

## 可复用 Solver

```python
import numpy as np
from kernel_sb import Solver

rng = np.random.default_rng(42)
J = rng.normal(size=(1025, 1025)).astype(np.float32)
J = (J + J.T) / 2
np.fill_diagonal(J, 0)
x0 = rng.uniform(-.01, .01, (1025, 100)).astype(np.float32)
y0 = rng.uniform(-.01, .01, (1025, 100)).astype(np.float32)

with Solver(J, algorithm="dsb", batch_size=100, n_iter=1000,
            precision="fp32", device_id=0) as solver:
    x, y = solver.run(x0, y0)
    print(solver.last_device_ms)  # 仅设备执行图，不含传输
    x_again, y_again = solver.run(x0, y0)  # 重用 J、工作区和执行图
```

J 可为 NumPy 或 SciPy 稀疏矩阵；h 可为 `(N,)` 或 `(N,1)`；p 必须为 `(n_iter,)`。
输入必须是有限实数，计算前明确转换到 FP32；无效形状、参数立即报错。
`run` 不修改调用者初态，返回独立的 Fortran-contiguous FP32 数组。
修改原始 J/h/p 不会改变已经构造的 Solver；新问题需要新 Solver。

一次性使用可调用 `kernel_dsb(J, x=x0, y=y0, ...)` 或 `kernel_bsb(...)`。
省略初态时可传 `seed`，不会读取其他模块的全局随机初态。
旧版 `solve/solve_bsb/solve_cached/solve_cached_bsb` 原地更新接口已由 `Solver.run` 取代；
迁移时请接收 `(x, y)` 返回值。没有保留不安全的指针缓存兼容层。

`fp32` 使用 `cubeMathType=0 (KEEP_DTYPE)`；`hf32` 使用 `3 (USE_HF32)`，状态更新仍是 FP32。
HF32 可能改变求解轨迹和解质量；它是调用者的显式选择，不宣称与 FP32 等价。

## 下载数据并运行公平基准

```bash
# 官网逐个下载全部 23 个实例；验证后原子写入，首个失败即停止，不重试/换源。
python3 qplib.py download
# 也可以只下载指定实例
python3 qplib.py download 3506 3850

# 两种算法，同一初态、schedule、batch、迭代数，比较 FP32/HF32 和稀疏 CPU
OPENBLAS_NUM_THREADS=1 python3 test_qplib.py --precision fp32 hf32 > data/qplib-results.jsonl
OPENBLAS_NUM_THREADS=1 python3 test_gset.py --graph data/gset/G22 > data/gset-results.jsonl
# 稠密场景；--cpu none 可以只测 NPU
OPENBLAS_NUM_THREADS=1 python3 test.py --size 1024 --cpu dense > data/dense-results.jsonl
```

默认 batch=100、niter=1000、repeat=3、seed=42。每轮使用 seed+轮数，CPU 和各精度使用完全相同
的初态。JSONL 分别记录 `setup_ms`、`first_run_ms`、各轮 `wall_ms`（含状态传输/验证/分配）、
`device_ms`（ACL 事件时间）、中位数，以及每轮最优原始目标值。CPU 基线是本仓库
NumPy/SciPy FP32 方程实现，可用 `--cpu sparse/dense/none` 指定。
计时排除目标值计算，但不隐去构造开销；重复求解与一次性调用必须分别比较。

QPLIB 官网文件使用下三角多项式
\(f(z)=\frac12\sum_{i\ge j}T_{ij}z_i z_j+b^Tz+c\)，不能把非对角条目原值镜像成 Hessian。
解析器只接受无约束、全二元实例，支持默认线性项、覆盖值、对角项、孤立变量、常数和 min/max。
将 \(z=(s+1)/2\) 映射到最小化 \(E=-s^TJs/2-h^Ts\)，去除自耦合，并归一化。
**报告目标值时直接代回原始 FP64 多项式**，不从归一化能量反推，以避免消去误差。
QPLIB_3506 的官网模型项 `+2*b2*b3` 对应文件 `2 1 4.0`；官网可行解目标值为 478。

数据保存在忽略的 `data/` 中。QPLIB：<https://qplib.zib.de/>，许可
[CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)，引用
[Furini et al., QPLIB](https://doi.org/10.1007/s12532-018-0147-4)。
Gset：<https://web.stanford.edu/~yyye/yyye/Gset/>，使用 G22、G30、G50、G60，放入 `data/gset/`。

## 910B2 实测（2026-09-23）

Ascend 910B2、CANN 9.1、Python 3.11.13、NumPy 1.24.0、SciPy 1.15.3。
以下均为 FP32、batch=100、1000 步，预热后的同步调用中位数，**包含状态传输**。
新旧 DSB 使用保存的同一组 J/h/p、参数和初态；旧版测 3 次，新版测 5 次。

| 问题 | 旧 DSB | 新 DSB | DSB 加速比 | 新 BSB |
| --- | ---: | ---: | ---: | ---: |
| QPLIB_3506，N=496 | 228.50 ms | 15.96 ms | 14.3× | 15.04 ms |
| 稠密 Ising，N=1024 | 528.91 ms | 25.48 ms | 20.8× | 25.02 ms |

旧版使用 `solve_cached(force_copy=True)`，计时包含 J/h/p 上传；新版为复用快照的 `Solver.run`，
所以该比值衡量完整重复求解路径的改进，**不是同一传输边界下的纯内核加速比**。
旧 BSB 存在未初始化缓冲区问题，不提供不可信的旧版 BSB 加速比。
原始计时在 `data/baseline.json`、`data/optimized-comparison.json`，匹配输入在
`data/baseline_*.npz`。新 DSB 的纯设备中位数分别为 15.18 ms、24.64 ms。

全量 QPLIB 使用 seeds=42/43/44，23 个实例 × 两种算法 × 两种精度，共 92 条记录。
FP32 相对本仓库**单线程 SciPy 稀疏 CPU**基线的重复求解加速比中位数为
**DSB 52.1×、BSB 37.8×**；每种算法各 69 次试验的最优原始目标值均与 CPU 一致。
这不表示轨迹逐元素相等，也不证明达到问题的全局最优。
HF32 的加速比中位数相对 FP32 仅为 DSB 1.02×、BSB 1.00×，且分别改变了 3 个、4 个实例的
部分试验目标值，因此仍默认 FP32。完整结果在 `data/qplib-results.jsonl`。

额外运行 G22/G30/G50/G60，最大 N=7000。FP32 的 G60 重复求解为 DSB 195.01 ms、BSB 197.96 ms。
四个实例的 DSB 最优割值均与同初态 CPU 一致；BSB 在 G30/G60 的部分试验存在差异，
因此不承诺跨设备长轨迹或解质量完全等价。逐次结果在 `data/gset-G*-results.jsonl`。

**构造成本不能忽略**：QPLIB 的 Solver 构造中位数约 232 ms，进程首次构造约 1.5–1.6 s，
包含 ACL 初始化、算子预热和执行图捕获。上述加速比适用于复用 Solver 的场景，不能套用于冷启动。
每条基准记录同时保留构造、首次执行、重复执行和设备时间，便于按实际调用次数评估摊销成本。

另用 msprof 验证 N=1024、batch=100、FP32 DSB 的设备任务：排除构造预热后，矩阵乘耗时中位数
16.44 µs，融合更新 8.22 µs，分别使用 AI Core 和 Vector Core。采样保存在 `data/npu-profile-final/`。
可用较短 schedule 重现算子级分析，避免采集完整基准的大量 trace：

```bash
OPENBLAS_NUM_THREADS=1 msprof --output=data/npu-profile --task-time=on --aic-metrics=PipeUtilization \
  --application="$(command -v python3) $PWD/test.py --size 1024 --algorithm dsb --precision fp32 --niter 100 --repeat 1 --cpu none"
```

## 性能判断的边界

优先优化 batch 级数据复用、Cube 计算和设备调度，再考虑单条 Vector 指令。
不假设“更少的 kernel launch 就一定更快”：旧常驻内核减少了 launch，却让 Cube 闲置并重复读取 J。
也不把降低精度、CPU/NPU 初态不同或忽略初始化成本当作加速。

测试用 FP64 独立方程验证短轨迹和非对称 J，用边界样例检查严格裁剪条件，用重复/并发执行检查
快照和资源隔离；长轨迹比较可行性与原始目标值，因为浮点归约顺序的微小变化会被非线性动力学放大。
小 batch、小问题可能由执行图调度而不是 GEMM 吞吐主导；超稀疏大图也可能更适合稀疏 CPU 路径。
当前实现不声称对所有尺寸达到硬件理论峰值，超长 schedule 的捕获资源受 CANN 运行时限制，失败直接报错。
