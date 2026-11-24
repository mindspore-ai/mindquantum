# MindQuantum Release Notes

[View English](./RELEASE.md)

## MindQuantum 0.13.0 Release Notes

MindQuantum **0.13.0** 正式支持昇腾 NPU 原生加速，新增态矢量模拟器后端 `mqvector_npu` 与 QAIA `BSB`/`DSB` 求解器的原生 CANN 后端 `ascend-float32`；官方发布的 ARM Linux `.whl` 安装包已同时内置昇腾 910 系列与 950 系列内核，安装即用、无需用户自行编译。此外，本版本通过局部线路分块（Circuit Tiling）、Arm SVE 向量化、重构 `mqmatrix` 密度矩阵模拟器以及按工作量自适应的 OpenMP 调度大幅提升 CPU 模拟性能，新增 `GBSB` 求解器、前沿算法示例与 AI 编程助手技能包，并将支持环境升级至 Python 3.9–3.12。

### 主要特性和增强

#### Simulator

- [BETA] **昇腾 NPU 态矢量模拟器后端 [`mqvector_npu`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html)**（[`!3172`](https://atomgit.com/mindspore/mindquantum/pull/3172)、[`!3181`](https://atomgit.com/mindspore/mindquantum/pull/3181)、[`!3196`](https://atomgit.com/mindspore/mindquantum/pull/3196)）：新增面向昇腾 NPU 的单精度（`mindquantum.complex64`）全振幅态矢量模拟器后端（`Simulator('mqvector_npu', n_qubits, device_id=0)`）。
    - **全流程设备端执行**：量子态常驻 NPU 显存，以严格单精度（FP32，禁用低精度近似）在设备端直接完成线路演化、期望值计算、伴随梯度与精确参数平移（Parameter-Shift）梯度、含噪轨迹模拟、测量采样及约化密度矩阵求解，仅将最终结果传回主机（在 Ascend 910B2 上，24 比特 4 层 MaxCut QAOA 单步伴随梯度约 **54 ms**）。
    - **开箱即用与使用说明**：**官方 ARM Linux `.whl` 安装包已同时内置昇腾 910 系列（如 910B）与 950 系列（如 950PR/950DT）设备内核，无需从源码构建**；在主机激活匹配的 CANN 运行时环境（CANN Toolkit 及对应设备 Ops 算子包）后，MindQuantum 会自动识别 NPU 架构并加载对应内核（无 NPU/CANN 环境时，同一安装包中的 CPU 后端仍可正常使用）。使用时传入的 `Hamiltonian` 需指定为单精度（如 `Hamiltonian(op, mq.float32)` 或 `ham.astype(mq.float32)`），详见 [`docs/mqvector_npu.md`](./docs/mqvector_npu.md)。
- [STABLE] **CPU 线路分块执行引擎与 Arm SVE 加速**（[`!3146`](https://atomgit.com/mindspore/mindquantum/pull/3146)、[`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)、[`!3175`](https://atomgit.com/mindspore/mindquantum/pull/3175)）：
    - **缓存友好的线路分块（Circuit Tiling）**：相比将多个量子门合并为稠密矩阵而损失稀疏性的传统门融合，新增的线路分块引擎将局部量子态分块加载至 CPU L1 缓存，在保留单门稀疏结构的同时在缓存内连续作用多个门并合并相邻对角门，大幅降低内存带宽开销。`mqvector` 在 `n_qubits >= 13` 时对正向演化与伴随梯度自动启用。
    - **Arm SVE 向量化与自动派发**：Linux AArch64 平台支持利用 Arm SVE 指令集并行演化多个状态分块，并在运行时根据 CPU 硬件能力自动切换最优内核（无需手动配置，同时兼容不支持 SVE 的通用 ARM 机器）。在 Kunpeng 920F 上开启 NUMA 绑核（`OMP_PLACES=cores OMP_PROC_BIND=close`）后，20–28 比特随机线路与 MaxCut QAOA 任务较 0.12.0 提速 **3–5×**。
- [STABLE] **重构 [`mqmatrix`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html) 密度矩阵模拟器**（[`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)）：基于全矩阵存储与线路分块引擎全面重构 `mqmatrix`，利用密度矩阵的厄米对称性减少一半冗余计算，支持噪声信道的原生超算子执行与含噪线路 C++ 伴随梯度，并将末端测量的含噪采样优化为单次混合态演化后直接抽样。10 比特下 `apply_circuit` 提速 **3×**、梯度计算提速 **2×**，1000 次含噪采样耗时从 **278 s 降至 0.12 s**。
- [STABLE] **OpenMP 线程数按工作量自适应调度**（[`!3168`](https://atomgit.com/mindspore/mindquantum/pull/3168)）：CPU 模拟器与算符稀疏化现根据量子态规模自动分配线程数（每 8,192 个振幅分配 1 个线程，上限为 `OMP_NUM_THREADS`），批处理梯度则在总线程预算内动态平衡批间与态内并行度，消除了多核服务器运行小比特线路时的线程同步开销，中等规模批处理梯度提速 **1.6×–1.8×**。

#### Algorithm, Examples & AI Skills

- [BETA] **QAIA `BSB` / `DSB` 原生昇腾 CANN 后端 `ascend-float32`**（[`!3162`](https://atomgit.com/mindspore/mindquantum/pull/3162)、[`!3178`](https://atomgit.com/mindspore/mindquantum/pull/3178)、[`!3181`](https://atomgit.com/mindspore/mindquantum/pull/3181)、[`!3196`](https://atomgit.com/mindspore/mindquantum/pull/3196)）：为 [`BSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.BSB.html) 与 [`DSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.DSB.html) 新增不依赖 PyTorch 的原生昇腾 NPU 后端 `backend='ascend-float32'`（支持关键字参数 `device_id=0`；原有的 `'npu-float32'` 仍为 PyTorch/`torch_npu` 后端），并在 [`qaia-cann` 分支](https://atomgit.com/mindspore/mindquantum/tree/qaia-cann)提供 QPLIB/Gset 基准评测套件。
    - **批量加速与执行图复用**：将多组随机初态（batch）的耦合计算合并为严格 FP32 批量矩阵乘，并在首次 `update()` 时将完整多步迭代捕获为设备执行图；后续对同一问题重复求解时直接重放执行图以消除主机逐步调度开销（修改问题矩阵或积分参数时会自动重建执行图）。在 Ascend 910B2（100 副本、1000 步）上，23 个 QPLIB 实例较单线程 SciPy 稀疏 CPU 基线中位加速达 **DSB 52.1×**、**BSB 37.8×**。
    - **开箱即用**：**官方 ARM Linux `.whl` 安装包已同时内置昇腾 910 系列与 950 系列内核，无需用户自行编译**，仅需在运行时环境中安装并激活匹配的 CANN Toolkit 及对应设备 Ops 算子包（详见 [`docs/qaia_ascend.md`](./docs/qaia_ascend.md)）。
- [BETA] **QAIA 通用弹道模拟分岔算法 `GBSB`**（[`!3159`](https://atomgit.com/mindspore/mindquantum/pull/3159)）：在 `mindquantum.algorithm.qaia` 下新增 [`GBSB`](https://arxiv.org/abs/2508.17655) 求解器，支持对个体分岔参数进行非线性控制，提供 `'cpu-float32'`、`'gpu-float32'` 和 `'npu-float32'` 后端。
- [STABLE] **解码量子干涉（DQI）与量子回声（OTOC）示例**（[`!3128`](https://atomgit.com/mindspore/mindquantum/pull/3128)、[`!3142`](https://atomgit.com/mindspore/mindquantum/pull/3142)）：在 `example/dqi_mindquantum/` 下新增基于解码量子干涉（DQI）的组合优化完整示例（[arXiv:2509.08328](https://arxiv.org/abs/2509.08328)）；在 `example/otoc_simulation/` 下新增基于 `mqmps` 的多体 OTOC 光锥扩散模拟（[arXiv:2506.10191](https://arxiv.org/abs/2506.10191)）与 $\text{H}_2$ 分子键长反演示例（[arXiv:2510.19550](https://arxiv.org/abs/2510.19550)）。
- [STABLE] **MindQuantum AI 编程助手技能包（Agent Skills）**（[`!3125`](https://atomgit.com/mindspore/mindquantum/pull/3125)、[`!3126`](https://atomgit.com/mindspore/mindquantum/pull/3126)）：在 `skills/` 下新增 6 个面向 AI 编程助手的领域技能包（涵盖核心 API、变分算法训练、含噪模拟、QAIA 求解、线路编译与量子化学），可通过 `npx skills add https://atomgit.com/mindspore/mindquantum.git` 安装。

### 问题修复

- **QAIA `LSB` 与 `USB` 求解精度修复**（[`!3124`](https://atomgit.com/mindspore/mindquantum/pull/3124)）：修复 `LSB` 对数缩放因子指数符号反转及 `x = 0` 除零问题，并将 `USB` 默认初始化范围从 `[-0.01, 0.01]` 扩大至 `[-0.1, 0.1]` 以避免初始驱动力被均匀量化归零；修复后两者在 MaxCut 基准上恢复论文级性能（求解速度较 `DSB` 提升 2.0×–2.7×）。
- **QAIA 输入转换、耦合归一化与参数校验修复**（[`!3176`](https://atomgit.com/mindspore/mindquantum/pull/3176)、[`!3178`](https://atomgit.com/mindspore/mindquantum/pull/3178)、[`!3184`](https://atomgit.com/mindspore/mindquantum/pull/3184)、[`!3195`](https://atomgit.com/mindspore/mindquantum/pull/3195)）：支持在 GPU/NPU 后端直接传入稠密 `numpy.ndarray` 耦合矩阵 `J`，修复 `calc_cut`/`calc_energy` 的输入转换及 `SB` 在 NPU 下带外场 `h` 时的初始化报错；修正 CPU 后端 `CAC`/`CFC`/`SFC` 对稀疏矩阵误用矩阵乘法代替逐元素平方的归一化错误；对含 `NaN`/`Inf` 的 `SB` 输入参数（`J`、`h`、`x`、`dt`、`xi`、`A`）、全零 `J`/`h` 下缺失的 `xi` 以及不支持的 `backend` 名称统一抛出明确的 `ValueError`。
- **模拟器异常数值拦截与多线程异常传播修复**（[`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)、[`!3185`](https://atomgit.com/mindspore/mindquantum/pull/3185)、[`!3186`](https://atomgit.com/mindspore/mindquantum/pull/3186)、[`!3191`](https://atomgit.com/mindspore/mindquantum/pull/3191)）：CPU 模拟器（`mqvector`/`mqmatrix`）在 `set_qs` 接收含 `NaN`/`Inf` 或范数溢出的量子态，以及 `apply_circuit`/`apply_gate` 接收非有限门参数时，现统一抛出 `ValueError` 并保持原量子态不变；`get_reduced_density_matrix` 对空比特列表或负数比特索引显式报错；修复批处理梯度工作线程抛异常导致 Python 进程直接崩溃的问题。
- **Windows 多线程稳定性与编译兼容性修复**（[`!3181`](https://atomgit.com/mindspore/mindquantum/pull/3181)、[`!3190`](https://atomgit.com/mindspore/mindquantum/pull/3190)、[`!3191`](https://atomgit.com/mindspore/mindquantum/pull/3191)、[`!3192`](https://atomgit.com/mindspore/mindquantum/pull/3192)）：修复 Windows 平台下 `mqmatrix` 开启 OpenMP 多线程时的堆内存损坏崩溃问题，以及 `mqmps` 在 MinGW-w64 下的向量指令栈对齐崩溃问题；修复 MSVC/MinGW 编译兼容性与 Windows Python 3.9 打包路径问题。
- **`mqmatrix` 导数门、Pauli 信道与测量精度修复**（[`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)）：修复 `apply_gate(..., diff=True)` 在导数算符作用子空间之外未置零的问题，消除 Pauli 信道因浮点舍入误差偶发产生 `NaN` 的问题，并完善测量时的迹与分支权重校验。
- **OpenQASM 多比特 `barrier` 解析修复**（[`!3156`](https://atomgit.com/mindspore/mindquantum/pull/3156)）：修复 `OpenQASM().from_string()` 解析作用于整个量子寄存器的 `barrier` 指令（如 `qreg q[4]; barrier q;`）时仅保留首个比特的问题。
- **MindSpore 设备接口更新与文档修复**（[`!3127`](https://atomgit.com/mindspore/mindquantum/pull/3127)、[`!3130`](https://atomgit.com/mindspore/mindquantum/pull/3130)、[`!3166`](https://atomgit.com/mindspore/mindquantum/pull/3166)、[`!3180`](https://atomgit.com/mindspore/mindquantum/pull/3180)）：将 `mindquantum.framework` 中已弃用的 `ms.set_context(device_target="CPU")` 更新为 `ms.set_device("CPU")`，并修复多处 API 文档错别字与公式渲染问题。

### 构建与打包改进

- **昇腾多架构单包构建与多核扩展**（[`!3181`](https://atomgit.com/mindspore/mindquantum/pull/3181)、[`!3196`](https://atomgit.com/mindspore/mindquantum/pull/3196)）：`build.sh` 与 `build_locally.sh` 新增 `--npu` 及 `--npu-arch=<arch,...>`（如 `dav-2201,dav-3510`）或 `--npu-soc=<soc>` 选项，支持一次构建将多代昇腾微架构内核打包进同一个 `.whl` 文件，并移除 `mqvector_npu` 观测算子原有的 48 核上限以适配昇腾 950 系列硬件（源码构建指南见 [`docs/ascend_build.md`](./docs/ascend_build.md)）。
- **模块化构建与工程工具链升级**（[`!3111`](https://atomgit.com/mindspore/mindquantum/pull/3111)、[`!3137`](https://atomgit.com/mindspore/mindquantum/pull/3137)、[`!3153`](https://atomgit.com/mindspore/mindquantum/pull/3153)、[`!3154`](https://atomgit.com/mindspore/mindquantum/pull/3154)、[`!3174`](https://atomgit.com/mindspore/mindquantum/pull/3174)、[`!3181`](https://atomgit.com/mindspore/mindquantum/pull/3181)）：新增 `-DENABLE_MPS=ON/OFF` 开关并在配置阶段校验 `OpenBLAS` 符号（解决 macOS 误选 Accelerate 导致链接失败的问题）；隔离各 C++ 扩展模块的 `pybind11` 符号表，`mqrt` 支持 x86_64 与 AArch64 构建；Linux `.whl` 修包升级至 `auditwheel >= 6.4` 自动匹配 `manylinux` 标签，Python 代码检查统一迁移至 Ruff。

### 不兼容变更与行为变化

- **Python 版本要求**（[`!3111`](https://atomgit.com/mindspore/mindquantum/pull/3111)）：支持版本调整为 **Python 3.9–3.12**，不再支持 Python 3.7 与 3.8。
- **API 移除与异常修正**（[`!3168`](https://atomgit.com/mindspore/mindquantum/pull/3168)）：移除长期未实现的 `Simulator.set_threads_number()`（请使用环境变量 `OMP_NUM_THREADS` 控制线程上限）；访问 `mindquantum.framework` 不存在的属性时由误抛 `ImportError` 修正为 `AttributeError`。
- **`mqmatrix` 内存占用、采样序列与校验变化**（[`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)）：改为全矩阵存储后单个密度矩阵内存翻倍（相同内存下最大可模拟比特数减 1），$\le 6$ 比特极小线路耗时增加约 10%–15%；末端测量含噪采样在固定 `seed` 下的具体采样序列与 0.12.0 不同（统计概率分布不变）；传入不支持的 `circ_left`/`simulator_left` 或非法噪声信道参数现在会在执行前立即报错。
- **源码构建与 CMake 选项变更（仅影响自行编译源码的开发者）**（[`!3127`](https://atomgit.com/mindspore/mindquantum/pull/3127)、[`!3148`](https://atomgit.com/mindspore/mindquantum/pull/3148)、[`!3169`](https://atomgit.com/mindspore/mindquantum/pull/3169)、[`!3174`](https://atomgit.com/mindspore/mindquantum/pull/3174)、[`!3181`](https://atomgit.com/mindspore/mindquantum/pull/3181)）：
    - 要求 **CMake $\ge$ 3.20**、**`pybind11 >= 3.1`**（Linux `.whl` 修包需 **`auditwheel >= 6.4`**）；
    - GPU 与 cuQuantum 构建须显式传入 `--gpu` 及 `--cuquantum`（或 `-DENABLE_CUQUANTUM=ON`），不再隐式启用或在缺少 CUDA $\ge$ 11.0 时静默回退；Linux AArch64 默认开启 `MQ_ARM_SVE=ON`（编译器不支持 SVE 时需传 `-DMQ_ARM_SVE=OFF`）；
    - CMake 的 BLAS 变量统一为 `OpenBLAS_ROOT`、`OpenBLAS_LIBRARY`、`OpenBLAS_SYMBOL_PREFIX`（移除 `MQ_BLAS_LIBRARIES` 等旧变量）；构建脚本默认安装运行与测试依赖，移除 `--only-pytest`（可用 `--no-python-deps` 跳过）。

### 贡献者

感谢以下开发者做出的贡献：

dsdsdshe, huangzhe-2026, liqingyu_2026, lllrt, mangrovecoder, Tsingpy.

欢迎以任何形式对项目提供贡献！

## MindQuantum 0.12.0 Release Notes

### 主要特性和增强

#### Device

- [BETA] [`QPU`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/device/mindquantum.device.QPU.html)：新增用于真实量子硬件执行的抽象基类。该类标准化了基于 `sampling` 的接口，与 `mindquantum.simulator.Simulator` 保持一致，实现模拟器与真实 QPU 之间的无缝后端切换。核心能力包括（[`!2842`](https://atomgit.com/mindspore/mindquantum/pull/2842)）：
    - **最小化厂商接入成本**：硬件厂商仅需实现 `sampling()` 抽象方法。
    - **基于采样的期望值估计**：`get_expectation()` 自动执行 Pauli 基旋转并从测量结果进行统计估计。
    - **基于采样的梯度估计**：`get_expectation_with_grad()` 返回 `GradOpsWrapper`，支持参数平移规则（适用于 RX、RY、RZ、U3、Rxx、Ryy、Rzz 等 10+ 种门族）和中心有限差分回退策略，并具备智能的逐参数策略选择。
    - **批量优化接口**：厂商可选择性重写 `_sampling_batch()` 以实现批量任务提交。

#### Simulator

- [BETA] [`mqmps`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html) 后端：新增矩阵乘积态（MPS）模拟器，以张量网络而非全态矢量方式表示量子态。当线路纠缠度适中时，该后端可以用远少于态矢量方法的内存模拟更多量子比特（[`!3088`](https://atomgit.com/mindspore/mindquantum/pull/3088)）。
    - 支持单比特和双比特门（非相邻比特自动插入 SWAP 门）、测量、采样、约化密度矩阵计算和 Pauli 期望值。
    - 高性能 C++ 内核，基于 BLAS/LAPACK 后端确保数值稳定性。
- [STABLE] [`SequentialUCCAnsatz`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.mqchem.SequentialUCCAnsatz.html)：在 `mindquantum.simulator.mqchem` 下新增顺序 UCC 拟设构建器，保留激发算符的插入顺序。与标准 UCC 不同，相同的激发算符可以独立参数多次出现，支持分层/Trotter 风格的拟设构建，有助于改善 VQE 工作流的收敛性（[`!3096`](https://atomgit.com/mindspore/mindquantum/pull/3096)、[`!3103`](https://atomgit.com/mindspore/mindquantum/pull/3103)）。

#### Algorithm

- [BETA] QAIA 算法族扩展：在 `mindquantum.algorithm.qaia` 下新增量化模拟分叉变体 [`TSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.TSB.html)（三元）、[`USB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.USB.html)（一元）和 [`LSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.LSB.html)（线性），支持 CPU 和 GPU 后端（[`!2820`](https://atomgit.com/mindspore/mindquantum/pull/2820)）。

### 改进

- 改进了子模块导入失败时的错误提示，现在显示实际原因（[`!2828`](https://atomgit.com/mindspore/mindquantum/pull/2828)）。
- 完善了 SABRE 和 MQSABRE 映射算法的返回值文档说明（[`!3099`](https://atomgit.com/mindspore/mindquantum/pull/3099)）。

### 问题修复

- **MQSABRE 映射**：`InitialMapping` 在某些拓扑下可能产生无效比特分配，导致线路执行时段错误。已修复（[`!2793`](https://atomgit.com/mindspore/mindquantum/pull/2793)、[`!2794`](https://atomgit.com/mindspore/mindquantum/pull/2794)）。
- **OpenQASM 导出**：对同一量子比特重复测量的线路导出为无效 OpenQASM。现在每次测量会正确映射到独立经典比特（[`!2826`](https://atomgit.com/mindspore/mindquantum/pull/2826)）。
- **OpenQASM 导入**：解析器无法处理门参数中的算术表达式（如 `pi/4 + 0.5`）。已替换为通用表达式求值器（[`!2825`](https://atomgit.com/mindspore/mindquantum/pull/2825)）。
- **`Circuit.apply_value`**：参数替换时自定义门（如 `UCCExcitationGate`）的内部状态被破坏。门状态现已正确保留（[`!3095`](https://atomgit.com/mindspore/mindquantum/pull/3095)）。
- **NoiseBackend 梯度接口缺失**：修复 `NoiseBackend.get_expectation_with_grad` 支持缺失的问题。（[`!3097`](https://atomgit.com/mindspore/mindquantum/pull/3097)）
- **QAIA SB 初始化**：提供外场 `h` 时 `xi` 初始化错误，导致优化结果不正确（[`!2818`](https://atomgit.com/mindspore/mindquantum/pull/2818)）。
- **QAIA 错误提示**：改进了 SB 求解器的 GPU 导入错误信息，清楚说明所需条件（[`!2788`](https://atomgit.com/mindspore/mindquantum/pull/2788)）。

### 贡献者

感谢以下开发者做出的贡献：

dsdsdshe, Waikikilick, Yaocheng Chen, MangroveCoder, liudongmei2025, wangyin888, zhoushijie2025, 宦晓玲.

欢迎以任何形式对项目提供贡献！

## MindQuantum 0.11.0 Release Notes

### 主要特性和增强

#### Simulator

- [STABLE] [`mqchem`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.mqchem.MQChemSimulator.html): 新增 `mqchem` 模拟器后端。该模拟器基于配置相互作用（CI）方法，通过在固定电子数的子空间中进行计算，为量子化学问题提供了内存和计算效率更高的解决方案。其核心组件包括 ([`!2717`](https://gitee.com/mindspore/mindquantum/pulls/2717))：
    - [`MQChemSimulator`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.mqchem.MQChemSimulator.html): CI空间模拟器，默认初始化为Hartree-Fock态。
    - [`UCCExcitationGate`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.mqchem.UCCExcitationGate.html): 用于构建UCC拟设线路的专用激发门。
    - [`CIHamiltonian`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.mqchem.CIHamiltonian.html): 用于在CI空间中高效计算期望值的哈密顿量包装器。
    - [`prepare_uccsd_vqe`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.mqchem.prepare_uccsd_vqe.html): 自动化UCCSD-VQE实验准备流程的高级函数。
- [STABLE] [`mqvector_cq`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html): 新增 `mqvector_cq` 模拟器后端，利用 NVIDIA cuQuantum SDK 进一步加速在 NVIDIA GPU 上的量子线路模拟 ([`!2724`](https://gitee.com/mindspore/mindquantum/pulls/2724))。该后端依赖于cuquantum，因此需要在环境中正确安装cuqauntum后方可使用，详情请参考cuquantum官网上的安装指南。

#### Algorithm

- [STABLE] **QAIA 算法后端扩展**: 为量子退火启发式算法（QAIA）家族（包括 [`ASB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.ASB.html), [`BSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.BSB.html), [`DSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.DSB.html), [`LQA`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.LQA.html), [`CAC`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.CAC.html), [`CFC`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.CFC.html), [`SFC`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.SFC.html), [`NMFA`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.NMFA.html), [`SimCIM`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.SimCIM.html)）新增了基于 PyTorch 的 GPU 和 NPU 后端支持，显著提升了在相应硬件上的计算性能。用户现在可以通过 `backend` 参数选择 `'gpu-float32'` 或 `'npu-float32'` ([`!2669`](https://gitee.com/mindspore/mindquantum/pulls/2669), [`!2678`](https://gitee.com/mindspore/mindquantum/pulls/2678))。

### 问题修复

- [`PR2727`](https://gitee.com/mindspore/mindquantum/pulls/2727): 修复了线路图的 SVG 导出功能在处理多控 CNOT 门时只显示一个控制比特的问题。
- [`PR2717`](https://gitee.com/mindspore/mindquantum/pulls/2717): 修复了 `uccsd_singlet_get_packed_amplitudes` 中双激发振幅索引错误的问题，确保了从CCSD计算中提取的振幅的正确性。
- [`PR2716`](https://gitee.com/mindspore/mindquantum/pulls/2716): 修复了 SABRE 和 MQSABRE 映射算法在处理非连续物理比特ID（例如，`[12, 13, 15]`）时会崩溃的问题。
- [`PR2713`](https://gitee.com/mindspore/mindquantum/pulls/2713): 修复了 `TimeEvolution` 在处理含常数项（单位算符）的哈密顿量时的问题，现在会正确地将其转换为全局相位门。
- [`PR2679`](https://gitee.com/mindspore/mindquantum/pulls/2679): 修复了 NMFA 算法在包含外场 `h` 时 `J_norm` 计算不正确的问题。
- [`PR2714`](https://gitee.com/mindspore/mindquantum/pulls/2714): 修复了在 Windows CI 环境下，当 Python 安装在带空格的路径中时可能出现的链接器错误。
- [`PR2650`](https://gitee.com/mindspore/mindquantum/pulls/2650): 修复了 `Rn` 门在某些情况下可能出现的除零错误。
- [`PR2648`](https://gitee.com/mindspore/mindquantum/pulls/2648): 修复了 `sabre` 映射算法中 `barrier` 门处理不当的问题。

### 其他更新

- **QAIA 算法**:
    - 增强了对输入参数的检查，确保耦合矩阵 `J` 是对称且对角线为零，为用户提供更明确的指引 ([`!2710`](https://gitee.com/mindspore/mindquantum/pulls/2710))。
    - 优化了 [`LQA`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.LQA.html) 算法，通过缓存中间计算结果减少了冗余计算 ([`!2656`](https://gitee.com/mindspore/mindquantum/pulls/2656))。
    - 改进了部分算法中数值不稳定时的错误提示信息，使其更具指导性 ([`!2710`](https://gitee.com/mindspore/mindquantum/pulls/2710))。
- **依赖项**: 更新了依赖项，现在要求 `scipy>=1.13.1`。
- 为 `ParameterResolver` 中出现复数以及 `SB` 模拟器使用 `int8` 精度时增加了警告提示 ([`!2651`](https://gitee.com/mindspore/mindquantum/pulls/2651))。

### 贡献者

感谢以下开发者做出的贡献：

beastsenior, dsdsdshe, GhostArtyom, liushiwei2024, lyq, MangroveCoder, YangleiSHAO, zengqg, ZhuangJP, 葛宇非, 满成, 肖阳, 左博伟.

欢迎以任何形式对项目提供贡献！

## MindQuantum 0.10.0 Release Notes

### 主要特性和增强

#### Algorithm

- [BETA] [`virtual_distillation`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/error_mitigation/mindquantum.algorithm.error_mitigation.virtual_distillation.html): 新增基于虚拟蒸馏的误差缓解算法，通过创建量子态的虚拟副本并在纠缠系统上进行测量来减少量子噪声。
- [BETA] [`QuantumNeuron`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/nisq/mindquantum.algorithm.nisq.QuantumNeuron.html): 新增基于重复直到成功（RUS）策略的量子神经元实现，通过量子电路模拟经典神经元行为，应用非线性函数旋转。
- [STABLE] [`SGAnsatz`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/nisq/mindquantum.algorithm.nisq.SGAnsatz.html): 新增序列生成变分量子线路，可高效生成具有固定键维度的矩阵乘积态。该ansatz通过在相邻量子比特上应用参数化量子线路块，自然适应一维量子多体问题。
- [STABLE] [`SGAnsatz2D`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/nisq/mindquantum.algorithm.nisq.SGAnsatz2D.html): 新增二维序列生成变分量子线路，可生成字符串键态。支持通过指定二维网格尺寸自动生成遍历路径，或通过自定义线路集合构建特定类型的string-bond态。
- [STABLE] [`qjpeg`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/library/mindquantum.algorithm.library.qjpeg.html): 新增基于量子傅里叶变换的量子图像压缩算法，可以通过减少量子比特数量来压缩量子图像，同时保留频域中的关键信息。
- [STABLE] [`cnry_decompose`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/compiler/mindquantum.algorithm.compiler.cnry_decompose.html): 新增对CnRY门的分解。
- [STABLE] [`cnrz_decompose`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/compiler/mindquantum.algorithm.compiler.cnrz_decompose.html): 新增对CnRZ门的分解。
- [STABLE] [`BSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.BSB.html): 为弹道模拟分叉算法添加GPU加速支持，支持`'cpu-float32'`, `'gpu-float16'`, `'gpu-int8'`三种精度选项。
- [STABLE] [`DSB`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/qaia/mindquantum.algorithm.qaia.DSB.html): 为离散模拟分叉算法添加GPU加速支持，支持`'cpu-float32'`, `'gpu-float16'`, `'gpu-int8'`三种精度选项。
- [STABLE] [`qudit_symmetric_encoding`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/library/mindquantum.algorithm.library.qudit_symmetric_encoding.html): 新增qudit编码功能，将d级量子态映射到量子比特态，通过对称编码实现，在标准量子比特量子计算机上高效模拟高维量子系统。
- [STABLE] [`qudit_symmetric_decoding`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/library/mindquantum.algorithm.library.qudit_symmetric_decoding.html): 新增解码功能，将量子比特对称态或矩阵解码为qudit态或矩阵，增强对多能级量子系统的支持。解码过程涉及将对称量子比特态转换为相应的qudit态，便于在标准量子比特量子计算机上高效模拟高维量子系统。
- [STABLE] [`qutrit_symmetric_ansatz`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/library/mindquantum.algorithm.library.qutrit_symmetric_ansatz.html): 引入qutrit对称ansatz，构建保持任意qutrit门编码对称性的量子比特ansatz。该功能通过利用对称性保持变换，允许在标准量子比特量子计算机上高效模拟高维量子系统。ansatz支持分解为`"zyz"`或`"u3"`基，并可选择性地包含全局相位。

#### Measure

- [STABLE] [`MeasureResult.to_json`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.MeasureResult.html#mindquantum.core.gates.MeasureResult.to_json): 支持测量结果的序列化和存储。
- [STABLE] [`MeasureResult.reverse_endian`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.MeasureResult.html#mindquantum.core.gates.MeasureResult.reverse_endian): 支持反转测量结果中比特串和测量键的字节序。

#### Operator

- [STABLE] [`mat_to_op`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/library/mindquantum.algorithm.library.mat_to_op.html): 新增从矩阵转换为`QubitOperator`的函数，支持小端和大端量子比特排序，以便与不同的量子计算框架无缝集成。

#### Circuit

- [STABLE] 新增[`Circuit.from_qcis()`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.from_qcis)和[`Circuit.to_qcis()`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.to_qcis)函数，支持与QCIS格式互转。
- [STABLE] 新增`__eq__`和`__ne__`方法，支持电路对象比较。
- [STABLE] [`Circuit.depth()`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.depth): 新增获取量子线路深度的功能，支持考虑单比特门和栅栏门对电路深度的影响，帮助用户更好地评估和优化量子线路的复杂度。

#### Simulator

- [STABLE] [`get_reduced_density_matrix`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_reduced_density_matrix): 新增获取指定量子比特约化密度矩阵的功能，通过对其他量子比特执行部分迹运算来实现。
- [STABLE] [`get_qs_of_qubits`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_qs_of_qubits): 新增获取指定量子比特量子态的功能。如果结果态是纯态，则返回态矢量；如果是混态，则返回密度矩阵。支持以 ket 记号（狄拉克记号）格式返回量子态。
- [STABLE] 模拟器后端选择"stabilizer"时，支持使用[`reset`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.reset)重置量子态。
- [STABLE] 模拟器后端选择"stabilizer"时，支持使用[`get_expectation`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation)计算给定哈密顿量在当前量子态下的期望值。

#### Compiler

- [STABLE] [`U3Fusion`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/compiler/mindquantum.algorithm.compiler.U3Fusion.html): 新增将连续的单量子比特门融合为一个U3门的编译规则。该规则扫描电路并将作用在同一量子比特上的连续单量子比特门组合成单个U3门。对于独立的单量子比特门，也会被转换为U3形式。可选择是否跟踪和包含全局相位。
- [STABLE] [`u3_decompose`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/compiler/mindquantum.algorithm.compiler.u3_decompose.html): 新增将U3门分解为Z-X-Z-X-Z旋转序列的功能。支持标准分解（U3(θ,φ,λ) = Rz(φ)Rx(-π/2)Rz(θ)Rx(π/2)Rz(λ)）和替代分解（U3(θ,φ,λ) = Rz(φ)Rx(π/2)Rz(π-θ)Rx(π/2)Rz(λ-π)）两种方法。当任何旋转角度为常数且等于0时，相应的RZ门将被省略。
- [STABLE] [`DecomposeU3`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/compiler/mindquantum.algorithm.compiler.DecomposeU3.html): 新增U3门分解的编译规则，将U3门分解为Z-X-Z-X-Z旋转序列。支持标准和替代两种分解方法。

#### IO

- [STABLE] [`QCIS`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/io/mindquantum.io.QCIS.html): 新增量子电路与QCIS格式转换类。

#### Utilities

- [STABLE] [`random_hamiltonian`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/utils/mindquantum.utils.random_hamiltonian.html): 新增随机泡利哈密顿量生成功能。支持指定量子比特数量和泡利项数量，可设置随机种子以保证结果可重现。生成的哈密顿量可用于量子算法测试和基准测试。

### 破坏性改动

- [重要] `MeasureResult` 中的 `keys`、`samples` 的字节序被统一为小端序（little-endian）。如果您的代码使用了这两个属性，请小心检查并使用新增的 `reverse_endian` 方法进行调整。

### 问题修复

- [`PR2497`](https://gitee.com/mindspore/mindquantum/pulls/2497)：修复了 **Amplitude Encoder** 中参数名可能重复的问题。
- [`PR2410`](https://gitee.com/mindspore/mindquantum/pulls/2410)：修复了 `is_measure_end` 的错误，该错误会导致即使没有测量操作也返回 `True`。
- [`PR2410`](https://gitee.com/mindspore/mindquantum/pulls/2410)：修复了在双量子比特门中颠倒量子比特顺序后计算结果不正确的问题。
- [`PR2377`](https://gitee.com/mindspore/mindquantum/pulls/2377)：修复了 `DAGCircuit` 在处理深层线路时会出现递归错误的问题，现在支持对任意深度线路的处理。
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345)：修复了 `mqmatrix` 的 `get_expectation_with_grad` 方法在处理批量哈顿量时计算错误的问题，并添加了测试用例。
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345)：修复了未按指定顺序添加门并使用 `reverse_qubits` 时出现的错误。
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345)：修正了 `FermionOperator.hermitian()` 示例代码中的错误。
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319)：修复了 Stabilizer 模拟器的测量错误。
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319)：修复了 Stabilizer 模拟器中种子未正确应用的问题。
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319)：增加了对 Stabilizer 模拟器输出比特串正确性的检测。
- [`PR2315`](https://gitee.com/mindspore/mindquantum/pulls/2315)：使 **MQSim** 和 **Hamiltonian** 支持序列化，支持python多进程`multiprocessing`。
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309)：修复了 **QAOA** 的一些 ansatz 中缺失虚数项和系数的问题。
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309)：修复了 `QAOAAnsatz` 示例无法正常运行的问题。
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309)：修改了 ansatz 电路中的参数名称，使其与公式对应。
- [`PR2296`](https://gitee.com/mindspore/mindquantum/pulls/2296)：修复了 `kron_factor_4x4_to_2x2s()` 返回值的索引错误，确保了双比特门分解函数 `kak_decompose` 的正确性。
- [`PR2285`](https://gitee.com/mindspore/mindquantum/pulls/2285)：移除了计算梯度时不必要的输出。

### 其他更新

- 优化了量子线路第一次运行时的速度，提升了性能。
- 提高了 `params_zyz()` 函数的精度，提升了 **ZYZ** 分解的计算精度。
- 移除了未安装 `mqvector_gpu` 的警告信息，仅在使用时提示。
- 移除了未安装mindspore时的警告信息，仅在使用时提示。
- 当哈密顿量包含虚部时，增加了警告提示，提醒用户注意可能的计算结果异常。
- 提升了未安装 **MindSpore** 时警告信息的清晰度。
- 将 `pip` 源更改为清华镜像源。

### 贡献者

感谢以下开发者做出的贡献：

Arapat Ablimit, Chufan Lyu, GhostArtyom, LuoJianing, Mr1G, Waikikilick, donghufeng, dsdsdshe, xuxusheng, yuhan, zengqg, zhouyuanyang2024, 王上, 杨金元, 糖醋排骨.

欢迎以任何形式对项目提供贡献！

## MindQuantum 0.9.11 Release Notes

### 主要特性和增强

#### Gates

- [STABLE] [`任意轴旋转门`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.Rn.html#mindquantum.core.gates.Rn): 新增绕布洛赫球上任意轴旋转的单比特门[`Rn`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.Rn.html#mindquantum.core.gates.Rn)。
- [STABLE] [`matrix`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.Rxx.html#mindquantum.core.gates.Rxx.matrix): 量子门支持通过该接口并指定参数`full=True`来获取量子门完整的矩阵形式（受作用位比特和控制位比特影响）。
- [STABLE] [`热弛豫信道`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.ThermalRelaxationChannel.html#mindquantum.core.gates.ThermalRelaxationChannel): 新增 ThermalRelaxationChannel 热弛豫信道。
- [Alpha] [`量子测量`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.Measure.html#mindquantum.core.gates.Measure): 测量门现支持比特重置功能，可将测量后的量子态重置为|0⟩态或者|1⟩态。优化测量门执行速度。
- [STABLE] [`RotPauliString`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.RotPauliString.html#mindquantum.core.gates.RotPauliString): 新增任意泡利串旋转门。
- [STABLE] [`GroupedPauli`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.GroupedPauli.html#mindquantum.core.gates.GroupedPauli): 新增泡利组合门，该门比逐个执行单个泡利门会更加快速。
- [STABLE] [`GroupedPauliChannel`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.GroupedPauliChannel.html#mindquantum.core.gates.GroupedPauliChannel): 新增泡利信道组合信道，该组合信道比逐一执行泡利信道更快。
- [STABLE] [`SX`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.SXGate.html): 新增根号X门。
- [STABLE] [`Givens`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/gates/mindquantum.core.gates.Givens.html): 新增Givens旋转门。

#### Circuit

- [STABLE] [`summary`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.summary): 通过该接口展示的量子线路汇总信息会以表格形式呈现，更加美观直接。
- [STABLE] [`svg`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.svg): 现在可以通过控制参数`scale`来对量子线路图进行缩放。
- [STABLE] [`openqasm`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit): 量子线路直接支持转化为[`openqasm`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.to_openqasm)或者从[`openqasm`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.from_openqasm)转化为mindquantum线路。

#### ParameterResolver

- [STABLE] [`PRGenerator`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/parameterresolver/mindquantum.core.parameterresolver.PRGenerator.html#mindquantum.core.parameterresolver.PRGenerator): [`new`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/core/parameterresolver/mindquantum.core.parameterresolver.PRGenerator.html#mindquantum.core.parameterresolver.PRGenerator.new)接口支持配置临时的前缀和后缀。

#### Ansatz

- [STABLE] [`硬件友好型量子线路`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mindquantum.algorithm.nisq.html#ansatz): 新增多种硬件友好型量子线路，请参考论文[Physics-Constrained Hardware-Efficient Ansatz on Quantum Computers that is Universal, Systematically Improvable, and Size-consistent](https://arxiv.org/abs/2307.03563)。

#### Device

- [STABLE] [`QubitsTopology`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/device/mindquantum.device.QubitsTopology.html#mindquantum.device.QubitsTopology): 支持通过[set_edge_color](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/device/mindquantum.device.QubitsTopology.html#mindquantum.device.QubitsTopology.set_edge_color)设置不同边的颜色。支持通过`show`来直接展示拓扑结构图。

#### Simulator

- [STABLE] [`sampling`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.sampling): 加速量子模拟器在对不含噪声且测量门全部在线路末端的量子线路的采样。

#### utils

- [STABLE] [`进度条`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.utils.html#progress-bar): 新增两个基于rich构建的简单易用的进度条，分别为支持单层循环的[`SingleLoopProgress`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/utils/mindquantum.utils.SingleLoopProgress.html#mindquantum.utils.SingleLoopProgress)和支持两层循环的[`TwoLoopsProgress`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/utils/mindquantum.utils.TwoLoopsProgress.html#mindquantum.utils.TwoLoopsProgress)。
- [Alpha] [`random_insert_gates`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/utils/mindquantum.utils.random_insert_gates.html): 支持在量子线路中随机插入量子门。

#### Algorithm

- [Alpha] [`MQSABRE`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/mapping/mindquantum.algorithm.mapping.MQSABRE.html#mindquantum.algorithm.mapping.MQSABRE): 新增支持设置量子门保真度的比特映射算法。

### Bug Fix

- [`PR1971`](https://gitee.com/mindspore/mindquantum/pulls/1971): 修复[`amplitude_encoder`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/algorithm/library/mindquantum.algorithm.library.amplitude_encoder.html#mindquantum.algorithm.library.amplitude_encoder)中符号错误问题。
- [`PR2094`](https://gitee.com/mindspore/mindquantum/pulls/2094): 修复[`get_expectation_with_grad`](https://www.mindspore.cn/mindquantum/docs/zh-CN/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation_with_grad)在使用parameter shift规则时随机数种子单一性问题。
- [`PR2164`](https://gitee.com/mindspore/mindquantum/pulls/2164): 修复windows系统下的构建脚本传入参数问题。
- [`PR2171`](https://gitee.com/mindspore/mindquantum/pulls/2171): 修复密度矩阵模拟器在量子态复制时可能遇到的空指针问题。
- [`PR2175`](https://gitee.com/mindspore/mindquantum/pulls/2175): 修复泡利信道的概率可以为负数的问题。
- [`PR2176`](https://gitee.com/mindspore/mindquantum/pulls/2176): 修复parameter shift规则在处理含控制位量子门时的问题。
- [`PR2210`](https://gitee.com/mindspore/mindquantum/pulls/2210): 修复parameter shift规则在处理多参数门且部分参数为常数时的问题。

### 贡献者

感谢以下开发者做出的贡献：

yufan, wengwenkang, xuxusheng, Damien Ngyuen, zhouxu, wangzidong, yangkang, lujiale, zhangzhenghai, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng, 朱祎康, dorothy20212021, dsdsdshe, buyulin, norl-corxilea, herunhong, Arapat Ablimit, NoE, panshijie, longhanlin.

欢迎以任何形式对项目提供贡献！
