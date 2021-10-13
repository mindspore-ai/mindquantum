# MindQuantum 0.3.0

## MindQuantum 0.3.0 Release Notes

### Major Features and Improvements

### API Change

#### Backwards Incompatible Change

We unified the abbreviations of some nouns in MindQuantum.

- `isparameter` property of gate changes to `parameterized`

<table>
<tr>
<td style="text-align:center"> 0.2.0 </td> <td style="text-align:center"> 0.3.0 </td>
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
<td style="text-align:center"> 0.2.0 </td> <td style="text-align:center"> 0.3.0 </td>
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

yufan, wengwenkang, xuxusheng, wanzidong, yankang, lujiale, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Damien Ngyuen, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng.

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

yufan, wengwenkang, xuxusheng, wanzidong, yankang, lujiale, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Damien Ngyuen, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng.

Contributions of any kind are welcome!
