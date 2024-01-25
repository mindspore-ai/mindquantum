# MindQuantum Release Notes

[View English](./RELEASE.md)

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
