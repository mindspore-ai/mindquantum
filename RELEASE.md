# MindQuantum Release Notes

[查看中文](./RELEASE_CN.md)

## MindQuantum 0.9.11 Release Notes

### Major Feature and Improvements

#### Gates

- [STABLE] [`Arbitrary axis rotation`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Rn.html#mindquantum.core.gates.Rn): New single-qubit gates for arbitrary axis rotation on the Bloch sphere[`Rn`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Rn.html#mindquantum.core.gates.Rn)。
- [STABLE] [`matrix`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Rxx.html#mindquantum.core.gates.Rxx.matrix): The quantum gate supports retrieving its complete matrix representation by using the interface and specifying the parameter `full=True`. This matrix representation is influenced by the target qubit and the control qubit, if applicable.
- [STABLE] [`Terminal relaxation channel`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.ThermalRelaxationChannel.html#mindquantum.core.gates.ThermalRelaxationChannel): Add ThermalRelaxationChannel.
- [Alpha] [`Quantum measurement`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Measure.html#mindquantum.core.gates.Measure): The measurement gate now supports qubit reset functionality, allowing the measured quantum state to be reset to the |0⟩ state or the |1⟩ state. The execution speed of the measurement gate has been optimized for improved performance.
- [STABLE] [`RotPauliString`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.RotPauliString.html#mindquantum.core.gates.RotPauliString): Add arbitrary pauli string rotation gate.
- [STABLE] [`GroupedPauli`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.GroupedPauli.html#mindquantum.core.gates.GroupedPauli): Add Pauli combination gate. This gate allows for faster execution compared to individually applying single Pauli gates.
- [STABLE] [`GroupedPauliChannel`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.GroupedPauliChannel.html#mindquantum.core.gates.GroupedPauliChannel): Add Pauli combination channel. This channel allows for faster execution compared to applying Pauli channels individually.
- [STABLE] [`SX`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.SXGate.html#mindquantum.core.gates.SXGate): Add sqrt X gate.
- [STABLE] [`Givens`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.Givens.html): Add Givens rotation gate.

#### Circuit

- [STABLE] [`summary`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.summary): The summary information of the quantum circuit displayed through this interface will be presented in a table format, making it more visually appealing and straightforward.
- [STABLE] [`svg`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.svg): Scaling the svg of quantum circuit by setting `scale` of this API.
- [STABLE] [`openqasm`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit): Directly convert quantum circuit to [`openqasm`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.to_openqasm) or convert [`openqasm`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.from_openqasm) to a mindquantum circuit.

#### ParameterResolver

- [STABLE] [`PRGenerator`](https://www.mindspore.cn/mindquantum/docs/en/master/core/parameterresolver/mindquantum.core.parameterresolver.PRGenerator.html#mindquantum.core.parameterresolver.PRGenerator): [`new`](https://www.mindspore.cn/mindquantum/docs/en/master/core/parameterresolver/mindquantum.core.parameterresolver.PRGenerator.html#mindquantum.core.parameterresolver.PRGenerator.new) is able to set temporary prefix of suffix.

#### Ansatz

- [STABLE] [`Hardware efficient ansatz`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/mindquantum.algorithm.nisq.html#ansatz): Add more hardware efficient ansatz, please refers to [Physics-Constrained Hardware-Efficient Ansatz on Quantum Computers that is Universal, Systematically Improvable, and Size-consistent](https://arxiv.org/abs/2307.03563)。

#### Device

- [STABLE] [`QubitsTopology`](https://www.mindspore.cn/mindquantum/docs/en/master/device/mindquantum.device.QubitsTopology.html#mindquantum.device.QubitsTopology): Enable to set color of edge by [set_edge_color](https://www.mindspore.cn/mindquantum/docs/en/master/device/mindquantum.device.QubitsTopology.html#mindquantum.device.QubitsTopology.set_edge_color). Enable to display the topology with `show`.

#### Simulator

- [STABLE] [`sampling`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.sampling): Improve the sampling speed for quantum circuit without noise and all measurement only at end.

#### utils

- [STABLE] [`Progress bar`](https://www.mindspore.cn/mindquantum/docs/en/master/mindquantum.utils.html#progress-bar): Add two easily used progress bar based on rich, which are [`SingleLoopProgress`](https://www.mindspore.cn/mindquantum/docs/en/master/utils/mindquantum.utils.SingleLoopProgress.html#mindquantum.utils.SingleLoopProgress) and [`TwoLoopsProgress`](https://www.mindspore.cn/mindquantum/docs/en/master/utils/mindquantum.utils.TwoLoopsProgress.html#mindquantum.utils.TwoLoopsProgress)。
- [Alpha] [random_insert_gates]: Enable to randomly insert quantum gate into a quantum circuit.

#### Algorithm

- [Alpha] [`MQSABRE`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/mapping/mindquantum.algorithm.mapping.MQSABRE.html#mindquantum.algorithm.mapping.MQSABRE): A new qubit mapping algorithm that enable to set fidelity of quantum gate.

### Bug Fix

- [`PR1971`](https://gitee.com/mindspore/mindquantum/pulls/1971): Fix sign bug in [`amplitude_encoder`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/library/mindquantum.algorithm.library.amplitude_encoder.html#mindquantum.algorithm.library.amplitude_encoder).
- [`PR2094`](https://gitee.com/mindspore/mindquantum/pulls/2094): Fix the issue of randomness in the [`get_expectation_with_grad`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation_with_grad) method when using the parameter shift rule.
- [`PR2164`](https://gitee.com/mindspore/mindquantum/pulls/2164): Fixed an issue with passing parameters in the build script on Windows systems.
- [`PR2171`](https://gitee.com/mindspore/mindquantum/pulls/2171): Fixed a potential null pointer issue with the density matrix simulator when copying quantum states.
- [`PR2175`](https://gitee.com/mindspore/mindquantum/pulls/2175): Fixed an issue with negative probabilities for Pauli channels.
- [`PR2176`](https://gitee.com/mindspore/mindquantum/pulls/2176): Fixed an issue with the parameter shift rule when dealing with controlled quantum gates.
- [`PR2210`](https://gitee.com/mindspore/mindquantum/pulls/2210): Fixed an issue with the parameter shift rule when dealing with multi-parameter gates with some of them are constant.

### Contributor

Thanks to the following developers for their contributions:

yufan, wengwenkang, xuxusheng, Damien Ngyuen, zhouxu, wangzidong, yangkang, lujiale, zhangzhenghai, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng, 朱祎康, dorothy20212021, dsdsdshe, buyulin, norl-corxilea, herunhong, Arapat Ablimit, NoE, panshijie, longhanlin.

Welcome contributions to the project in any form!
