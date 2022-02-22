# MindQuantum 0.6.0

## MindQuantum 0.6.0 Release Notes

### Major Features and Improvements

#### Better iteration supported for `QubitOperator` and `FermionOperator`

> The following example will be demonstrated with `QubitOperator`

- Iter multiple terms `QubitOperator`

```python
>>> ops = QubitOperator('X0 Y1', 1) + QubitOperator('Z2 X3', {'a': 3})

>>> for idx, o in enumerate(ops):
>>>     print(f'Term {idx}: {o}')
```

You will get each term of this operator,

```bash
Term 0: 1 [X0 Y1]
Term 1: 3*a [Z2 X3]
```

- Iter single term `QubitOperator`

```python
>>> ops = QubitOperator('X0 Y1', 2)

>>> for idx, o in enumerate(ops):
>>>     print(f'Word {idx}: {o}')
```

You will get each word of this operator with coefficient set to identity,

```bash
Word 0: 1 [X0]
Word 1: 1 [Y1]
```

### More built-in circuit supported

- [**general_w_state**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-w-state): circuit that can prepare a w state.
- [**general_ghz_state**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-ghz-state): circuit that can prepare a ghz state.
- [**bitphaseflip_operator**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarybitphaseflip-operator): circuit that can flip the sign of one or multiple calculation base.
- [**amplitude_encoder**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibraryamplitude-encoder): circuit that can encode classical number into quantum amplitude.

### Richer circuit operation supported

For origin circuit,

```python
>>> circuit = Circuit().z(0).rx('a', 1, 0).y(1)
```

```bash
q0: ──Z──────●─────────
             │
q1: ───────RX(a)────Y──
```

- Add a integer to a circuit will shift the qubit index.

```python
>>> circuit + 2
```

```bash
q2: ──Z──────●─────────
             │
q3: ───────RX(a)────Y──
```

```python
>>> 1 - circuit
```

```bash
q0: ───────RX(a)────Y──
             │
q1: ──Z──────●─────────
```

- Add a string to a circuit will add prefix to every parameters in this circuit.

```python
>>> 'l1' + circuit
```

```bash
q0: ──Z───────●───────────
              │
q1: ───────RX(l1_a)────Y──
```

- Reverse circuit qubits, the circuit will be flipped upside down.

```python
>>> circuit.reverse_qubits()
```

```bash
q0: ───────RX(a)────Y──
             │
q1: ──Z──────●─────────
```

### Feature enhancement

- `MaxCutAnsatz`: [**get_partition**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-partition)
- `MaxCutAnsatz`: [**get_cut_value**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-cut-value)
- `Circuit`: [**is_measure_end**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.circuit.html#mindquantumcorecircuitcircuitis-measure-end)

### Contributors

Thanks goes to these wonderful people:

yufan, wengwenkang, xuxusheng, wangzidong, yangkang, lujiale, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Damien Ngyuen, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng.

Contributions of any kind are welcome!

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

Contributions of any kind are welcome!
