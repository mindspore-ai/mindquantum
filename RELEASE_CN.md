# MindQuantum Release Notes

[View English](./RELEASE.md)

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
