# Ascend 上的批量模拟分叉求解器

MindQuantum 的 CANN 后端在 Ascend 910B 上执行 BSB（弹道模拟分叉）和 DSB（离散模拟分叉）。
同一个问题的多个初态合并为一次批量矩阵乘，完整迭代序列捕获为执行图，后续求解复用设备资源与执行图。
它不依赖 PyTorch，不自动降精度、不回退到 CPU、不重试设备错误。

## 构建与设备选择

与 [mqvector_npu](mqvector_npu.md) 共用 `ENABLE_ASCEND`、CANN 安装和精确的 SoC 目标。
默认 CPU 构建不查找或链接 CANN。启用时，配置阶段检查图捕获与可重复执行的 Matmul API，
运行时检查设备型号与编译目标是否一致。源码分支的原始测试环境是 CANN 9.1；
集成使用现有 Ascend C 工具链，不保留独立的两阶段构建或 `libkernels.so`。

安装项目开发依赖并加载 CANN 的 `set_env.sh` 后，在仓库根目录执行：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DIN_PLACE_BUILD=ON \
  -DPython_EXECUTABLE="$(command -v python3)" \
  -DENABLE_ASCEND=ON -DMQ_ASCEND_SOC=Ascend910B2 \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build --target python -j8
```

无 SVE 的 ARM 主机需要附加 `-DMQ_ARM_SVE=OFF`。`ASCEND_CANN_PACKAGE_PATH` 必须指向
`ASCEND_HOME_PATH` 对应的同一安装，避免头文件、库与编译器来自不同 CANN 版本。
切换 CANN 安装必须使用新的构建目录；只更新库路径不足以清除 CMake 缓存的 ASC 编译器与编译规则。
wheel 构建沿用项目入口，例如：

```bash
MQ_CIBW_BUILD_ARGS='--set,ENABLE_ASCEND,--var,MQ_ASCEND_SOC,Ascend910B2' python3 -m build .
```

CANN 与驱动是运行时依赖，不打入 wheel。`device_id` 是 ACL **逻辑设备号**，不是 `npu-smi` 的物理卡号。
容器中的设备映射不应通过试错修改可见设备环境变量解决。
导入 QAIA 或使用 CPU 后端不会加载该 CANN 扩展。
显式构造 `ascend-float32` 算法对象时加载扩展并校验原生布局，但不初始化 ACL 或创建设备资源；
首次 `update()` 才初始化设备并捕获执行图。
现有 `npu-float32` 仍表示 PyTorch/torch_npu 后端，不是本后端的别名。

## 在 BSB/DSB 上选择后端

```python
import numpy as np
from mindquantum.algorithm.qaia import BSB

J = np.array([[0., -1., .5], [-1., 0., -2.], [.5, -2., 0.]])
solver = BSB(J, batch_size=100, backend='ascend-float32', device_id=0)
solver.update()  # 第一次准备执行图并求解
print(solver.calc_cut())
solver.x = None
solver.initialize()  # 重新生成 x、y，不丢弃执行计划
solver.update()      # 问题与参数不变时复用执行图
```

`DSB` 使用相同方式。`ascend-float32` 的 Cube 矩阵乘、Vector 更新和状态存储均使用 FP32，
不启用 HF32 或自动降精度。

所有后端遵循同一套 QAIA 数学定义：J 对称、对角为零，默认耦合强度包含外场修正：

\[
\xi = \frac{0.5\sqrt{N-1}}{\sqrt{\lVert J\rVert_F^2+2\lVert h/2\rVert_2^2}}.
\]

构造时生成主机初态，首次 `update()` 才捕获执行图。每次更新检查 J 的 CSR 数据与索引、h、p 的**内容**，
以及迭代数、批量大小、积分参数和设备号；替换数组或原地修改导致内容变化时，旧计划失效。
这需要一次与系数存储量成正比的主机扫描，不重复上传或捕获未变的问题。
J 的类型、形状和本次初态始终检查；对称性与零对角校验仅在构造对象或重建计划时执行。
对称性沿用 QAIA 的规则：稠密矩阵用 `allclose`，稀疏矩阵精确比较。
J 在稠密与稀疏表示之间切换也会使计划失效，不能用相同的 CSR 内容绕过不同的校验规则。
xi 是构造时求出的参数：修改 J/h 后若希望重新归一化，应显式更新 xi 或创建新算法对象。
改变 N 需要新建算法对象；改变 `n_iter` 时应同时设置相应长度的 p。
归一化分母为零时必须显式指定 xi；J 为零但外场非零时仍按上述公式计算。
状态必须为二维 `(N, batch_size)`，系数和状态必须能表示为有限 FP32 值。
主机数值与逻辑维度错误在加载扩展前拒绝；原生布局校验在分配初态前完成。
重建时所有校验均先于旧图释放。
有效问题需要重建时先释放旧图，避免同时持有两份设备资源；若此后 CANN 构造失败，错误直接上报，不回退到旧问题。

`update()` 将 FP32 结果赋给 `self.x/self.y`，不修改传入的初态数组；外部引用不自动指向新数组。
`initialize()` 仅在 `x is None` 时重建 x，总会重新生成 y，与已有 SB 行为一致。
设备资源由算法对象持有，通过原生 RAII 在对象销毁时释放，不另设关闭协议。
长生命周期脚本不再使用某个实例时，可删除其所有引用（例如 `del solver`）释放资源。
算法对象的属性可变，不支持同一对象上的并发修改或更新；并发求解使用独立实例。
设备执行失败不承诺状态回滚，应丢弃失败的对象。

需要可复现的初态或重复相同的求解时，直接设置算法的二维状态：

```python
from mindquantum.algorithm.qaia import DSB

rng = np.random.default_rng(42)
x0, y0 = (rng.uniform(-.01, .01, (3, 100)).astype(np.float32) for _ in range(2))
solver = DSB(J, x=x0, batch_size=100, n_iter=1000, backend='ascend-float32')
solver.y = y0
solver.update()
x, y = solver.x, solver.y
solver.x, solver.y = x0, y0
solver.update()
np.testing.assert_array_equal(solver.x, x)
np.testing.assert_array_equal(solver.y, y)
```

默认参数、随机初始化、数学校验、状态与计划缓存均由算法层管理。
内部执行计划只接收已确定且校验通过的系数与积分参数，处理布局、padding 和原生调用；
不另设公开求解器、默认归一化公式或一维状态接口。

## 执行方程与资源生命周期

对每个压力值 p，DSB 令 \(u=\operatorname{sign}(x)\)，BSB 令 \(u=x\)，随后执行：

\[
v=y+dt[(p-\delta)x+\xi(Ju+h)],\qquad z=x+dt\,\delta v.
\]

若 \(|z_i|>1\)，位置截到 \(\operatorname{sign}(z_i)\)，动量归零；否则保留 z 和 v。
恰好等于 ±1 时保留动量。迭代内 `sign(0)=0`，最终二元评分时的零值约定独立于该动力学。

Cube 通过 `aclnnMatmul` 计算整批耦合，融合的 Ascend C Vector 内核完成其余更新及 DSB 下一步符号计算。
设备布局使用 `(batch,padded_n)`，借助 `(JX).T = X.T @ J.T` 计算整批耦合。
原生计划将 N 补齐到 64 的倍数，补入的耦合与外场为零；补齐计算及其越界校验只在原生层定义。
J 通过二维拷贝直接上传到补齐的设备矩阵，不构造补齐的主机矩阵。
状态按 NumPy stride 直接打包到页锁定缓冲，每次求解重置尾部，返回未补齐的 Fortran 连续 FP32 数组；
Python 层不分配补齐数组或再次裁剪结果。
Vector 按整批元素数均衡划分，
每次处理最多 4096 个元素，固定使用约 112.5 KiB UB；外场在构造时展开，不在每步重新广播。
稀疏输入也会在设备上稠密化，J 占用 \(O(N^2)\) 内存。

每个计划独占 context、stream、Matmul 执行器、张量、页锁定主机缓冲和设备缓冲。
预热后捕获完整 `aclmdlRI` 图；未失效的后续 `update()` 在主机检查后只上传初态、重放图、下载终态。
销毁时先结束设备工作，再销毁执行图与执行器，最后释放内存和 context。
原生调用先释放 GIL，再等待计划锁，避免与 Python 线程形成死锁。
与 mqvector_npu 共用上下文守卫，恢复调用方已有的非空 context，不 reset 设备或 finalize 他人拥有的 ACL。
CANN 不允许恢复空 context，因此此前未绑定 context 的线程不能恢复为空。
已报告的运行时错误同时从 CANN 线程错误槽中取出，避免后续其他组件的成功发射再次读到同一个错误。

## 验证与性能测量

```bash
python3 -m pytest -q tests/st/test_algorithm/test_qaia/test_qaia_ascend_api.py
OPENBLAS_NUM_THREADS=1 python3 -m pytest -q tests/st/test_algorithm/test_qaia/test_qaia_ascend.py
```

硬件测试通过 BSB/DSB 对照独立 FP64 方程，验证短轨迹、64-lane/4096 元素边界、严格截断、
长迭代重启、外场归一化、稀疏格式、系数原地修改、参数修改、布局越界拒绝和 G43 解质量。
原生计划测试使用非对称矩阵检测 GEMM 转置错误，并验证非连续状态、快照隔离与并发安全。
与 mqvector_npu 混用的测试验证正常执行、析构与失败构造后的调用方 context 恢复。
未构建扩展时只跳过硬件测试，不把设备执行错误当作跳过条件。

性能测量直接使用公开 API，区分首次 `update()`（含设备初始化、上传、预热和建图）
与后续 `update()`（含主机校验、状态传输和图重放）。CANN 后端的 `update()` 同步返回，
因此可以使用 `perf_counter()` 测量完整更新，无需额外设备同步：

```python
from time import perf_counter

import numpy as np
from mindquantum.algorithm.qaia import BSB

J = np.array([[0., -1., .5], [-1., 0., -2.], [.5, -2., 0.]], dtype=np.float32)
batch_size = 100
rng = np.random.default_rng(42)
x0, y0 = (rng.uniform(-.01, .01, (J.shape[0], batch_size)).astype(np.float32) for _ in range(2))
solver = BSB(J, x=x0, batch_size=batch_size, n_iter=1000, backend='ascend-float32')
solver.y = y0
start = perf_counter()
solver.update()
first_ms = (perf_counter() - start) * 1000

update_ms = []
for _ in range(5):
    solver.x, solver.y = x0, y0  # 恢复相同初态，不把前一次终态继续演化
    start = perf_counter()
    solver.update()
    update_ms.append((perf_counter() - start) * 1000)

print(f"首次更新：{first_ms:.3f} ms；后续更新中位数：{np.median(update_ms):.3f} ms")
print(solver.calc_cut())  # 评分不计入更新时间
```

`DSB` 使用相同方式。构造算法对象、准备初态、评分和析构不计入上述时间；
每次更新均从相同的 x/y 开始，后端不会修改这两个输入数组。
这个小问题仅演示计时方法，实际评估应使用目标工作负载，并记录硬件、CANN 版本、
问题规模、批量大小、迭代数、初态、数据类型及内存布局；不能将纯设备时间当作端到端加速比。
算子或纯设备时间使用 CANN `msprof` 分析。
正确性回归不能替代性能评测；比较不同实现的启发式长轨迹时，还应检查可行性和目标值分布。
