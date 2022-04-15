# MindQuantum Release Notes

[查看中文](./RELEASE_CN.md)

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

```python
>>> from mindquantum import RX
>>> gate = RX('a').on(0)
>>> gate.isparameter
True
```

</td>
<td>

```python
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

```python
>>> from mindquantum import Circuit
>>> circ = Circuit().rx('a', 0)
>>> circ.para_name
['a']
```

</td>
<td>

```python
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

`mindquantum.utils.get_fermion_operator`
</td><td>

`mindquantum.core.operators.get_fermion_operator`
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