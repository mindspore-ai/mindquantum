# MindQuantum Release Notes

[View English](./RELEASE.md)


## MindQuantum 0.10.0 Release Notes

### 新特性

- 新增对 **qudit** 编码和解码的支持，以及 **qutrit** 对称 **ansatz**，丰富了对多能级量子系统的支持。

### 主要特性和增强

#### Simulator

- 使 **MQSim** 支持序列化，支持多进程（`multiprocessing`）。

#### Measure

- 为 `MeasureResult` 添加了 `to_json` 函数，方便测量结果的序列化和存储。
- 统一了 `MeasureResult` 中 `keys`、`samples` 和 `data` 的字节序，并添加了 `reverse_endian` 函数，方便用户根据需求调整字节序。

#### Operator

- 使 **Hamiltonian** 支持序列化，支持多进程（`multiprocessing`）。
- 新增了从矩阵转换为 `QubitOperator` 的函数 `mat_to_op`。

#### Circuit

- 为 `Circuit` 类添加了 `from_qcis()` 和 `to_qcis()` 函数，支持量子电路与 **QCIS** 格式的相互转换。
- 增加了 `__eq__` 和 `__ne__` 方法，支持电路对象的比较操作。
- 添加了新的函数 `depth()`，用于获取电路的深度信息。

#### IO

- 新增了用于在量子电路和 **QCIS** 之间转换的类，方便不同格式之间的兼容应用。

#### Compiler

- `DAGCircuit` 现在支持对任意深度线路的处理，不再会出现递归错误。

### 问题修复

- [`PR2497`](https://gitee.com/mindspore/mindquantum/pulls/2497)：修复了 **Amplitude Encoder** 中参数名可能重复的问题。
- [`PR2410`](https://gitee.com/mindspore/mindquantum/pulls/2410)：修复了 `is_measure_end` 的错误，该错误会导致即使没有测量操作也返回 `True`。
- [`PR2410`](https://gitee.com/mindspore/mindquantum/pulls/2410)：修复了在双量子比特门中颠倒量子比特顺序后计算结果不正确的问题。
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345)：修复了 `mqmatrix` 的 `get_expectation_with_grad` 方法在处理批量哈密顿量时计算错误的问题，并添加了测试用例。
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345)：修复了未按指定顺序添加门并使用 `reverse_qubits` 时出现的错误。
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345)：修正了 `FermionOperator.hermitian()` 示例代码中的错误。
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319)：修复了 Stabilizer 模拟器的测量错误。
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319)：修复了 Stabilizer 模拟器中种子未正确应用的问题。
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319)：增加了对 Stabilizer 模拟器输出比特串正确性的检测。
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309)：修复了 **QAOA** 的一些 ansatz 中缺失虚数项和系数的问题。
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309)：修复了 `QAOAAnsatz` 示例无法正常运行的问题。
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309)：修改了 ansatz 电路中的参数名称，使其与公式对应。
- [`PR2296`](https://gitee.com/mindspore/mindquantum/pulls/2296)：修复了 `kron_factor_4x4_to_2x2s()` 返回值的索引错误，确保了双比特门分解函数 `kak_decompose` 的正确性。
- [`PR2285`](https://gitee.com/mindspore/mindquantum/pulls/2285)：移除了计算梯度时不必要的输出。

### 其他更新

- 优化了量子线路第一次运行时的速度，提升了性能。
- 提高了 `params_zyz()` 函数的精度，提升了 **ZYZ** 分解的计算精度。
- 移除了 `mqvector_gpu` 的警告信息。
- 当哈密顿量包含虚部时，增加了警告提示，提醒用户注意可能的计算结果异常。
- 提升了未安装 **MindSpore** 时警告信息的清晰度。
- 将 `pip` 源更改为清华镜像源。


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
