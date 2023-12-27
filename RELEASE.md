# MindQuantum Release Notes

[查看中文](./RELEASE_CN.md)

## MindQuantum 0.9.0 Release Notes

### Major Feature and Improvements

#### Gates

- [STABLE] [`Arbitry axis rotation`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Rn.html#mindquantum.core.gates.Rn): New single-qubit gates for arbitrary axis rotation on the Bloch sphere[`Rn`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Rn.html#mindquantum.core.gates.Rn)。
- [STABLE] [`matrix`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Rxx.html#mindquantum.core.gates.Rxx.matrix): The quantum gate supports retrieving its complete matrix representation by using the interface and specifying the parameter `full=True`. This matrix representation is influenced by the target qubit and the control qubit, if applicable.
- [STABLE] [`热弛豫信道`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.ThermalRelaxationChannel.html#mindquantum.core.gates.ThermalRelaxationChannel): Add ThermalRelaxationChannel.
- [Alpha] [`量子测量`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Measure.html#mindquantum.core.gates.Measure): The measurement gate now supports qubit reset functionality, allowing the measured quantum state to be reset to the |0⟩ state or the |1⟩ state. The execution speed of the measurement gate has been optimized for improved performance.
- [STABLE] [`RotPauliString`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.RotPauliString.html#mindquantum.core.gates.RotPauliString): Add arbitrary pauli string rotation gate.
- [STABLE] [`GroupedPauli`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.GroupedPauli.html#mindquantum.core.gates.GroupedPauli): Add Pauli combination gate. This gate allows for faster execution compared to individually applying single Pauli gates.
- [STABLE] [`GroupedPauliChannel`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.GroupedPauliChannel.html#mindquantum.core.gates.GroupedPauliChannel): Add Pauli combination channel. This channel allows for faster execution compared to applying Pauli channels individually.
- [STABLE] [`SX`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.RX.html#mindquantum.core.gates.RX): Add sqrt X gate.
- [STABLE] [Givens]: Add Givens rotation gate.

#### Circuit

- [STABLE] [`summary`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.summary): The summary information of the quantum circuit displayed through this interface will be presented in a table format, making it more visually appealing and straightforward.
- [STABLE] [`svg`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.svg): 现在可以通过控制参数`scale`来对量子线路图进行缩放。
- [STABLE] [`openqasm`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit): 量子线路直接支持转化为[`openqasm`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.to_openqasm)或者从[`openqasm`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.from_openqasm)转化为mindquantum线路。

#### ParameterResolver

- [STABLE] [`PRGenerator`](https://www.mindspore.cn/mindquantum/docs/en/master/core/parameterresolver/mindquantum.core.parameterresolver.PRGenerator.html#mindquantum.core.parameterresolver.PRGenerator): [`new`](https://www.mindspore.cn/mindquantum/docs/en/master/core/parameterresolver/mindquantum.core.parameterresolver.PRGenerator.html#mindquantum.core.parameterresolver.PRGenerator.new)接口支持配置临时的前缀和后缀。

#### Ansatz

- [STABLE] [`硬件友好型量子线路`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/mindquantum.algorithm.nisq.html#ansatz): 新增多种硬件友好型量子线路，请参考论文[Physics-Constrained Hardware-Efficient Ansatz on Quantum Computers that is Universal, Systematically Improvable, and Size-consistent](https://arxiv.org/abs/2307.03563)。

#### Device

- [STABLE] [`QubitsTopology`](https://www.mindspore.cn/mindquantum/docs/en/master/device/mindquantum.device.QubitsTopology.html#mindquantum.device.QubitsTopology): 支持通过[set_edge_color](https://www.mindspore.cn/mindquantum/docs/en/master/device/mindquantum.device.QubitsTopology.html#mindquantum.device.QubitsTopology.set_edge_color)设置不同边的颜色。支持通过`show`来直接展示拓扑结构图。

#### Simulator

- [STABLE] [`sampling`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.sampling): 加速量子模拟器在对不含噪声且测量门全部在线路末端的量子线路的采样。

#### utils

- [STABLE] [`进度条`](https://www.mindspore.cn/mindquantum/docs/en/master/mindquantum.utils.html#progress-bar): 新增两个基于rich构建的简单易用的进度条，分别为支持单层循环的[`SingleLoopProgress`](https://www.mindspore.cn/mindquantum/docs/en/master/utils/mindquantum.utils.SingleLoopProgress.html#mindquantum.utils.SingleLoopProgress)和支持两层循环的[`TwoLoopsProgress`](https://www.mindspore.cn/mindquantum/docs/en/master/utils/mindquantum.utils.TwoLoopsProgress.html#mindquantum.utils.TwoLoopsProgress)。
- [Alpha] [random_insert_gates]: 支持在量子线路中随机插入量子门。

#### Algorithm

- [Alpha] [`MQSABRE`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/mapping/mindquantum.algorithm.mapping.MQSABRE.html#mindquantum.algorithm.mapping.MQSABRE): 新增支持设置量子门保真度的比特映射算法。

### Bug Fix

- [`PR1971`](https://gitee.com/mindspore/mindquantum/pulls/1971): 修复[`amplitude_encoder`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/library/mindquantum.algorithm.library.amplitude_encoder.html#mindquantum.algorithm.library.amplitude_encoder)中符号错误问题。
- [`PR2094`](https://gitee.com/mindspore/mindquantum/pulls/2094): 修复[`get_expectation_with_grad`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation_with_grad)在使用parameter shift规则时随机数种子单一性问题。
- [`PR2164`](https://gitee.com/mindspore/mindquantum/pulls/2164): 修复windows系统下的构建脚本传入参数问题。
- [`PR2171`](https://gitee.com/mindspore/mindquantum/pulls/2171): 修复密度矩阵模拟器在量子态复制时可能遇到的空指针问题。
- [`PR2175`](https://gitee.com/mindspore/mindquantum/pulls/2175): 修复泡利信道的概率可以为负数的问题。
- [`PR2176`](https://gitee.com/mindspore/mindquantum/pulls/2176): 修复parameter shift规则在处理含控制位量子门时的问题。
- [`PR2210`](https://gitee.com/mindspore/mindquantum/pulls/2210): 修复parameter shift规则在处理多参数门且部分参数为常数时的问题。

### Contributor

Thanks to the following developers for their contributions:

yufan, wengwenkang, xuxusheng, Damien Ngyuen, zhouxu, wangzidong, yangkang, lujiale, zhangzhenghai, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng, 朱祎康, dorothy20212021, dsdsdshe, buyulin, norl-corxilea, herunhong, Arapat Ablimit, NoE, panshijie, longhanlin.

Welcome contributions to the project in any form!
