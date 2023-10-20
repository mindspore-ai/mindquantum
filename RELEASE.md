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

## MindQuantum  0.8.0 Release Notes

### Major Feature and Improvements

#### Gates

- [STABLE] [`FSim`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.FSim): Fermionic simulation gate supported, and this gate also works in variational quantum algorithm.
- [STABLE] [`U3`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.U3): The single qubit arbitrary gate U3 now will work as a single gate but not a piece of quantum circuit. U3 gate also works in variational quantum algorithm.
- [STABLE] [`Customed parameterized gate`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantum.core.gates.gene_univ_parameterized_gate). Customed parameterized gate now will compiled to machine code by jit compiler [numba](https://numba.pydata.org), and the simulator backend can call customed parameterized gate in parallel thread.
- [STABLE] [`BarrierGate`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.BarrierGate): BarrierGate now can be acted on certain qubits.
- [STABLE] [`KrausChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.KrausChannel): Design a customed kraus channel for quantum simulator.

#### Circuit

- [STABLE] [`svg`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.svg): Now you can set the `width` to split the svg circuit, so that you can copy it into your paper.

#### Simulator

- [STABLE] **New simulator supported**. `mqvector` and `mqvector_gpu` are two mindquantum simulate that prepared for cpu and gpu. And `projectq` simulator will be deprecated. The new simulator is total compatible with old one, what you only to do is to change the backend name when you initialize the simulator.

## MindQuantum 0.7.0 Release Notes

### Major Features and Improvements

#### Circuit

- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.as_encoder): Method of `Circuit` to mark this circuit as an encoder circuit.
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.as_ansatz): Method of `Circuit` to mark this circuit as an ansatz circuit.
- [STABLE] [`encoder_params_name`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.encoder_params_name): Method of `Circuit` to return all encoder parameters in the circuit.
- [STABLE] [`ansatz_params_name`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.ansatz_params_name): Method of `Circuit` to return all ansatz parameters in the circuit.
- [STABLE] [`remove_noise`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.remove_noise): Method of `Circuit` to remove all noise channels.
- [STABLE] [`with_noise`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.with_noise): Method of `Circuit` to add a given noise channel after every non-noise gate.
- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.as_encoder): A decorator to mark the circuit returned by the decorator function as an encoder circuit.
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.as_ansatz): A decorator to mark the circuit returned by the decorator function as an ansatz circuit.
- [STABLE] [`qfi`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.qfi): A method that can calculate the quantum fisher information of a given parameterized quantum circuit.
- [STABLE] [`partial_psi_partial_psi`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.partial_psi_partial_psi): A method that can calculate the first part of quantum fisher information.
- [STABLE] [`partial_psi_psi`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.partial_psi_psi): A method that can calculate the second part of quantum fisher information.

#### Gates

- [STABLE] [`AmplitudeDampingChannel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.html#mindquantum.core.gates.AmplitudeDampingChannel): Amplitude damping channel, caused by the energy dissipation.
- [STABLE] [`PhaseDampingChannel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.html#mindquantum.core.gates.PhaseDampingChannel): Phase damping channel express error that qubit loses quantum information without exchanging energy with environment.

#### FermionOperator and QubitOperator

- [STABLE] [`split`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.operators.html#mindquantum.core.operators.FermionOperator.split): A method of `FermionOperator` and `QubitOperator` that can split the coefficient with the operator.

#### ParameterResolver

- [STABLE] [`astype`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.astype): Convert the ParameterResolver to a given type.
- [STABLE] [`const`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.const): Get the constant part of this ParameterResolver.
- [STABLE] [`is_const`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_const): Check whether this ParameterResolver is constant.
- [STABLE] [`encoder_part`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.encoder_part): Set a part of parameters to be encoder parameters.
- [STABLE] [`ansatz_part`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.ansatz_part): Set a part of parameters to be ansatz parameters.
- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.as_encoder): Set all parameters to encoder parameters.
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.as_ansatz): Set all parameters to ansatz parameters.
- [STABLE] [`encoder_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.encoder_parameters): Return all encoder parameters.
- [STABLE] [`ansatz_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.ansatz_parameters): Return all ansatz parameters.
- [STABLE] [`is_hermitian`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_hermitian): Check whether this ParameterResolver is hermitian conjugate.
- [STABLE] [`is_anti_hermitian`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_anti_hermitian): Check whether this ParameterResolver is anti hermitian conjugate.
- [STABLE] [`no_grad_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.no_grad_parameters): Return all parameters that do no require gradient.
- [STABLE] [`requires_grad_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.requires_grad_parameters): Return all parameters that require gradient.

#### Simulator

- [STABLE] [`copy`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.html#mindquantum.simulator.Simulator.copy): The simulator now supports replication.
- [STABLE] [`apply_gate`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.html#mindquantum.simulator.Simulator.apply_gate): In this version, you can apply a gate in differential version.
- [BETA] [`inner_product`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.html#mindquantum.simulator.inner_product): Calculate the inner product of two states in two given simulators.

#### IO

- [STABLE] [`BlochScene`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.io.html): We support the establishment of a Bloch sphere mapping scene, which can draw quantum states on it, and can also dynamically demonstrate the change of quantum states.

### Contributors

Thanks goes to these wonderful people:

yufan, wengwenkang, xuxusheng, Damien Ngyuen, zhouxu, wangzidong, yangkang, lujiale, zhangzhenghai, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng, 朱祎康, dorothy20212021, dsdsdshe, buyulin, norl-corxilea, herunhong, Arapat Ablimit, NoE, panshijie, longhanlin.

Contributions of any kind are welcome!

## MindQuantum 0.6.0 Release Notes

### Major Features and Improvements

#### Better iteration supported for `QubitOperator` and `FermionOperator`

- Iterate over a multinomial fermion or boson operator and yield each term
- When the operator has only one item, each fermion or boson can be obtained through `singlet`

### Add line module

- [**general_w_state**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-w-state): prepare w-state quantum circuits.
- [**general_ghz_state**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-ghz-state): prepare ghz-state quantum circuits
- [**bitphaseflip_operator**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarybitphaseflip-operator): bit-flip quantum circuits
- [**amplitude_encoder**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibraryamplitude-encoder): amplitude-encoded quantum circuits

### Richer circuit operation supported

- `shift`: translation qubit

- `reverse_qubits`: flip circuit bit

### Feature enhancement

- `MaxCutAnsatz`: [**get_partition**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-partition), get the max-cut cutting solution
- `MaxCutAnsatz`: [**get_cut_value**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-cut-value), get the number of cuts for a cutting solution
- `Circuit`: [**is_measure_end**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.circuit.html#mindquantumcorecircuitcircuitis-measure-end), determine whether the quantum circuit is the end of the measurement gate

### SVG drawing mode that supports quantum circuits

- The quantum circuit build by mindquantum now can be showd by svg in jupyter notebook, just call `svg()` of any quantum circuits.

### Noise simulator supported

MindQuantum adds the following quantum channels for quantum noise simulation

- [`PauliChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatespaulichannel): Pauli channel
- [`BitFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesbitflipchannel): bit-flip channel
- [`PhaseFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesphaseflipchannel): phase-flip channel
- [`BitPhaseFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesbitphaseflipchannel): bit-phase flip channel
- [`DepolarizingChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesdepolarizingchannel): depolarized channel

## MindQuantum 0.5.0 Release Notes

### Major Features and Improvements

### API Change

#### Backwards Incompatible Change

We unified the abbreviations of some nouns in MindQuantum.

- `isparameter` property of gate changes to `parameterized`

<table>
<tr>
<td style="text-align:center"> 0.3.1 </td> <td style="text-align:center"> 0.5.0 </td>
</tr>
<tr>
<td>

```bash
>>> from mindquantum import RX
>>> gate = RX('a').on(0)
>>> gate.isparameter
True
```

</td>
<td>

```bash
>>> from mindquantum import RX
>>> gate = RX('a').on(0)
>>> gate.parameterized
True
```

</td>
</tr>
</table>

- `para_name` of a quantum circuit changes to `params_name`

<table>
<tr>
<td style="text-align:center"> 0.3.1 </td> <td style="text-align:center"> 0.5.0 </td>
</tr>
<tr>
<td>

```bash
>>> from mindquantum import Circuit
>>> circ = Circuit().rx('a', 0)
>>> circ.para_name
['a']
```

</td>
<td>

```bash
>>> from mindquantum import Circuit
>>> circ = Circuit().rx('a', 0)
>>> circ.params_name
['a']
```

</td>
</tr>
</table>

The quantum neural network API was redesigned in this version. From now on, we can easily build a hybrid quantum neural network with the help of `Simulator` in `PYNATIVE_MODE`.

The following API was removed.

1. `generate_pqc_operator`
2. `PQC`
3. `MindQuantumLayer`
4. `generate_evolution_operator`
5. `Evolution`
6. `MindQuantumAnsatzOnlyLayer`
7. `MindQuantumAnsatzOnlyOperator`

The new API was shown as below.

1. `MQOps`
2. `MQN2Ops`
3. `MQAnsatzOnlyOps`
4. `MQN2AnsatzOnlyOps`
5. `MQEncoderOnlyOps`
6. `MQN2EncoderOnlyOps`
7. `MQLayer`
8. `MQN2Layer`
9. `MQAnsatzOnlyLayer`
10. `MQN2AnsatzOnlyLayer`

The above modules are placed in `mindquantum.framework`.

#### Removed

Due to the duplication of functions, we deleted some APIs.

- `mindquantum.circuit.StateEvolution`

The following APIs have been remoted.

- `mindquantum.core.operators.Hamiltonian.mindspore_data`
- `mindquantum.core.operators.Projector.mindspore_data`
- `mindquantum.core.circuit.Circuit.mindspore_data`
- `mindquantum.core.parameterresolver.ParameterResolver.mindspore_data`

#### New feature

New gates are shown as below.

- `mindquantum.core.gates.SGate`
- `mindquantum.core.gates.TGate`

Measurement on certain qubits are now supported. The related APIs are shown as below.

- `mindquantum.core.gates.Measure`
- `mindquantum.core.gates.MeasureResult`

QASM is now supported.

- `mindquantum.io.OpenQASM`
- `mindquantum.io.random_hiqasm`
- `mindquantum.io.HiQASM`

Simulator is now separated from MindSpore backend. Now you can easily to use a simulator.

- `mindquantum.simulator.Simulator`

### Refactoring

For improving MindQuantum's package structure, we did some refactoring on MindQuantum.

<table>
<tr>
<td style="text-align:center"> old </td> <td style="text-align:center"> new </td>
</tr>
<tr><td>

`mindquantum.gate.Hamiltonian`
</td><td>

`mindquantum.core.operators.Hamiltonian`
</td></tr>
<tr><td>

`mindquantum.gate.Projector`
</td><td>

`mindquantum.core.operators.Projector`
</td></tr>
<tr><td>

`mindquantum.circuit.qft`
</td><td>

`mindquantum.algorithm.library.qft`
</td></tr>
<tr><td>

`mindquantum.circuit.generate_uccsd`
</td><td>

`mindquantum.algorithm.nisq.chem.generate_uccsd`
</td></tr>
<tr><td>

`mindquantum.circuit.TimeEvolution`
</td><td>

`mindquantum.core.operators.TimeEvolution`
</td></tr>
<tr><td>

`mindquantum.utils.count_qubits`
</td><td>

`mindquantum.core.operators.count_qubits`
</td></tr>
<tr><td>

`mindquantum.utils.commutator`
</td><td>

`mindquantum.core.operators.commutator`
</td></tr><tr><td>

`mindquantum.utils.normal_ordered`
</td><td>

`mindquantum.core.operators.normal_ordered`
</td></tr><tr><td>

`mindquantum.utils.number_operator`
</td><td>

`mindquantum.core.operators.number_operator`
</td></tr><tr><td>

`mindquantum.utils.hermitian_conjugated`
</td><td>

`mindquantum.core.operators.hermitian_conjugated`
</td></tr><tr><td>

`mindquantum.utils.up_index`
</td><td>

`mindquantum.core.operators.up_index`
</td></tr><tr><td>

`mindquantum.utils.down_index`
</td><td>

`mindquantum.core.operators.down_index`
</td></tr><tr><td>

`mindquantum.utils.sz_operator`
</td><td>

`mindquantum.core.operators.sz_operator`
</td></tr>
<tr><td>

`mindquantum.ansatz.Ansatz`</td><td>

`mindquantum.algorithm.nisq.Ansatz`
</td></tr>
<tr><td>

`mindquantum.ansatz.MaxCutAnsatz`
</td><td>

`mindquantum.algorithm.nisq.qaoa.MaxCutAnsatz`
</td></tr>
<tr><td>

`mindquantum.ansatz.Max2SATAnsatz`
</td><td>

`mindquantum.algorithm.nisq.qaoa.Max2SATAnsatz`
</td></tr>
<tr><td>

`mindquantum.ansatz.HardwareEfficientAnsatz`
</td><td>

`mindquantum.algorithm.nisq.chem.HardwareEfficientAnsatz`
</td></tr>
<tr><td>

`mindquantum.ansatz.QubitUCCAnsatz`
</td><td>

`mindquantum.algorithm.nisq.chem.QubitUCCAnsatz`
</td></tr>
<tr><td>

`mindquantum.ansatz.UCCAnsatz`
</td><td>

`mindquantum.algorithm.nisq.chem.UCCAnsatz`
</td></tr>
<tr><td>

`mindquantum.hiqfermion.Transform`
</td><td>

`mindquantum.algorithm.nisq.chem.Transform`
</td></tr>
<tr><td>

`mindquantum.hiqfermion.get_qubit_hamiltonian`
</td><td>

`mindquantum.algorithm.nisq.chem.get_qubit_hamiltonian`
</td></tr>
<tr><td>

`mindquantum.hiqfermion.uccsd_singlet_generator`
</td><td>

`mindquantum.algorithm.nisq.chem.uccsd_singlet_generator`
</td></tr>
<tr><td>

`mindquantum.hiqfermion.uccsd_singlet_get_packed_amplitudes`
</td><td>

`mindquantum.algorithm.nisq.chem.uccsd_singlet_get_packed_amplitudes`
</td></tr>
<tr><td>

`mindquantum.hiqfermion.uccsd0_singlet_generator`
</td><td>

`mindquantum.algorithm.nisq.chem.uccsd0_singlet_generator`
</td></tr>
<tr><td>

`mindquantum.hiqfermion.quccsd_generator`
</td><td>

`mindquantum.algorithm.nisq.chem.quccsd_generator`
</td></tr>
<tr><td>

`mindquantum.utils.bprint`
</td><td>

`mindquantum.io.bprint`
</td></tr>
<tr><td>

`mindquantum.circuit`
</td><td>

`mindquantum.core.circuit`
</td></tr>
<tr><td>

`mindquantum.gate`
</td><td>

`mindquantum.core.gates`
</td></tr>
<tr><td>

`mindquantum.ops`
</td><td>

`mindquantum.core.operators`
</td></tr>
<tr><td>

`mindquantum.parameterresolver`
</td><td>

`mindquantum.core.parameterresolver`
</td></tr>
<tr><td></td><td></td></tr>
</table>

### Contributors

Thanks goes to these wonderful people:

yufan, wengwenkang, xuxusheng, Damien Ngyuen, zhouxu, wangzidong, yangkang, lujiale, zhangzhenghai, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng.

Contributions of any kind are welcome!

## MindQuantum 0.3.1 Release Notes

### Major Features and Improvements

- Three tutorials have been rewritten to make them easier to read
- Circuit information such as qubit number, parameters will update immediately after you add gate
- The UN operator now support parameterized gate
- New ansatz that solving max 2 sat problem now are supported

### Contributors

Thanks goes to these wonderful people:

yufan, wengwenkang, xuxusheng, wangzidong, yangkang, lujiale, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Damien Ngyuen, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng.

Contributions of any kind are welcome!

## MindQuantum 0.2.0 Release Notes

### Major Features and Improvements

1. Parameterized FermionOperator and QubitOperator for quantum chemistry
2. Different kinds of transformation between FermionOperator and QubitOperator
3. UCCSD, QAOA and hardware efficient ansatz supported
4. MindQuantumAnsatzOnlyLayer for simulating circuit with ansatz only circuit
5. TimeEvolution with first order Trotter decomposition
6. High level operations for modifying quantum circuit

### Contributors

Thanks goes to these wonderful people:

yufan, wengwenkang, xuxusheng, wangzidong, yangkang, lujiale, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Damien Ngyuen, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng.

Contributions of any kind are welcome!

## MindQuantum 0.1.0 Release Notes

Initial release of MindQuantum.

### Major Features and Improvements

1. Easily build parameterized quantum circuit.
2. Effectively simulate quantum circuit.
3. Calculating the gradient of parameters of quantum circuit.
4. PQC (parameterized quantum circuit) operator that naturally compatible with other operators in mindspore framework.
5. Evolution operator that evaluate a quantum circuit and return the quantum state.
6. Data parallelization for PQC operator.

### Contributors

Thanks goes to these wonderful people:

yufan, wengwenkang, xuxusheng, wangzidong, yangkang, lujiale, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Damien Ngyuen, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng.
MindQuantum adds the following quantum channels for quantum noise simulation

Contributions of any kind are welcome!
