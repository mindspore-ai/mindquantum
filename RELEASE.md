# MindQuantum Release Notes

[查看中文](./RELEASE_CN.md)

## MindQuantum 0.9.0 Release Notes

### Major Feature and Improvements

#### Data precision

- [STABLE] [`Data Precision`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.dtype.html): MindQuantum now supports `float32`, `float64`, `complex64` and `complex128` four types of precision, and can set different precision types for operators, parameter resolvers and simulators.

#### Gates

- [STABLE] [`Quantum Gates`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.html#quantum-gate): Added multiple two-qubit Pauli rotation gate, including: [`Rxx`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.Rxx.html#mindquantum.core.gates.Rxx), [`Rxy`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.Rxy.html#mindquantum.core.gates.Rxy), [`Rxz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.Rxz.html#mindquantum.core.gates.Rxz), [`Ryy`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.Ryy.html#mindquantum.core.gates.Ryy), [`Ryz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.Ryz.html#mindquantum.core.gates.Ryz) and [`Rzz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.Rzz.html#mindquantum.core.gates.Rzz).
- [STABLE] [`Quantum Channel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.html#quantum-channel): Noise channels now support returning kraus operators via the `.matrix()` method.

#### Operator

- [STABLE] [`QubitOperator`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.operators.QubitOperator.html#mindquantum.core.operators.QubitOperator): Added [ `relabel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.operators.QubitOperator.html#mindquantum.core.operators.QubitOperator.relabel) interface, supports according to new qubit number to rearrange operators. [`FermionOperator`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.operators.FermionOperator.html#mindquantum.core.operators.FermionOperator.relabel) also supports this function.
- [STABLE] [`Ground state calculation`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.operators.ground_state_of_sum_zz.html#mindquantum.core.operators.ground_state_of_sum_zz): New interface supports the calculation of the ground state energy of the Hamiltonian containing only the direct product of the Pauli-Z operator and the Pauli-Z operator.

#### Ansatz

- [STABLE] [`Ansatz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.nisq.html#ansatz): Add 19 ansatz mentioned in Arxiv:[`1905.10876`](https://arxiv.org/abs/1905.10876), all have been implemented.

#### Circuit

- [STABLE] [`ChannelAdder`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#channel-adder): Add `ChannelAdder` module, support customized adding various quantum channels into the quantum circuit to construct a noise model. For more details, please refer to: [`ChannelAdder`](https://mindspore.cn/mindquantum/docs/en/master/noise_simulator.html).

#### Simulator

- [STABLE] [`Density Matrix Simulator`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator): Add density matrix simulator, named `mqmatrix`. Support variational quantum algorithms, noise simulation, etc. Its functionality is basically aligned with the existing `mqvector` full-amplitude simulator.
- [BETA] [`parameter shift`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation_with_grad): The quantum simulator gradient operator now supports the parameter shift rule algorithm, which is closer to the experiment.
- [STABLE] [`Expectation Calculation`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation): interface is basically aligned with [`get_expectation_with_grad`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation_with_grad), but does not calculate the gradient, saving time.

#### Device

- [STABLE] [`QubitNode`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.device.QubitNode.html#mindquantum.device.QubitNode): Added the qubit node object in the qubit topology interface, which supports the configuration of qubit position, color and connectivity.
- [STABLE] [`QubitsTopology`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.device.QubitsTopology.html#mindquantum.device.QubitsTopology): Qubit topology, supports custom topology. Also available with preset structures: linear qubit topology [`LinearQubits`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.device.LinearQubits.html#mindquantum.device.LinearQubits) and grid qubit topology [`GridQubits`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.device.GridQubits.html#mindquantum.device.GridQubits).

#### Algorithm

- [STABLE] [`Bit Mapping`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.mapping.html): Added Bit mapping algorithm [`SABRE`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.mapping.SABRE.html#mindquantum.algorithm.mapping.SABRE), please refer to Arxiv [`1809.02573`](https://arxiv.org/abs/1809.02573).
- [BETA] [`Error Mitigation`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.error_mitigation.zne.html): Added zero noise extrapolation algorithm for quantum error mitigation.
- [STABLE] [`Circuit folding`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.error_mitigation.fold_at_random.html): The quantum circuit folding function is added to support the growth of quantum circuits while ensuring the equivalence of quantum circuits.
- [BETA] [`Quantum circuit compilation`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.compiler.html): A new quantum circuit compilation module is added, which uses [`DAG`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.compiler.DAGCircuit.html) graphs to compile quantum circuits, and supports quantum compilation algorithms such as gate replacement, gate fusion, and gate decomposition.
- [STABLE] [`ansatz_variance`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.nisq.ansatz_variance.html): Added an interface to calculate the variance of the gradient of a certain parameter in the variable quantum circuit, which can be used to verify the [`barren plateau`](https://www.nature.com/articles/s41467-018-07090-4) phenomenon of the variable quantum circuit.

#### Framework

- [STABLE] [`QRamVecLayer`](https://mindspore.cn/mindquantum/docs/en/master/layer/mindquantum.framework.QRamVecLayer.html): The QRam quantum encoding layer has been added to support direct encoding of classical data into full-amplitude quantum states. The corresponding operator is [`QRamVecOps`](https://mindspore.cn/mindquantum/docs/en/master/operations/mindquantum.framework.QRamVecOps.html#mindquantum.framework.QRamVecOps).

#### IO

- [STABLE] [`OpenQASM`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.io.OpenQASM.html): OpenQASM has added the [`from_string`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.io.OpenQASM.html#mindquantum.io.OpenQASM.from_string) interface, which supports converting OpenQASM from string format to quantum circuits in MindQuantum.

### Bug fix

- [`PR1757`](https://gitee.com/mindspore/mindquantum/pulls/1757): Fixed the bug of [`StronglyEntangling`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.algorithm.nisq.StronglyEntangling.html) when the depth is greater than 2.
- [`PR1700`](https://gitee.com/mindspore/mindquantum/pulls/1700): Fixed matrix expression of [`CNOT`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.CNOTGate.html) gates and logic errors of [`AmplitudeDampingChannel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.AmplitudeDampingChannel.html).
- [`PR1523`](https://gitee.com/mindspore/mindquantum/pulls/1523): Fix logic errors of [`PhaseDampingChannel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.PhaseDampingChannel.html).

### Contributor

Thanks to the following developers for their contributions:

yufan, wengwenkang, xuxusheng, Damien Ngyuen, zhouxu, wangzidong, yangkang, lujiale, zhangzhenghai, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng, 朱祎康, dorothy20212021, dsdsdshe, buyulin, norl-corxilea, herunhong, Arapat Ablimit, NoE, panshijie, longhanlin.

Welcome contributions to the project in any form!
