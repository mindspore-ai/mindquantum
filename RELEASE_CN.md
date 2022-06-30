# MindQuantum Release Notes

[View English](./RELEASE.md)

## MindQuantum 0.7.0 Release Notes

### 主要特性和增强

#### Circuit

- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.as_encoder): `Circuit` 中的方法，将量子线路标记为编码量子线路。
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.as_ansatz): `Circuit` 中的方法，将量子线路标记为训练量子线路。
- [STABLE] [`encoder_params_name`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.encoder_params_name): `Circuit` 中的方法，返回量子线路中所有编码量子线路的参数名。
- [STABLE] [`ansatz_params_name`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.ansatz_params_name): `Circuit` 中的方法，返回量子线路中所有训练量子线路的参数名。
- [STABLE] [`remove_noise`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.remove_noise): `Circuit` 中的方法，用于将所有噪声信道移除。
- [STABLE] [`with_noise`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.with_noise): `Circuit` 中的方法，用于在每个非噪声门后面添加一个噪声信道。
- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.as_encoder): 一个装饰器，将所装饰函数返回的量子线路标记为编码量子线路。
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.circuit.html#mindquantum.core.circuit.as_ansatz): 一个装饰其，将所装饰函数返回的量子线路标记为训练量子线路。

#### Gates

- [STABLE] [`AmplitudeDampingChannel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.html#mindquantum.core.gates.AmplitudeDampingChannel): 振幅阻尼信道，由能量耗散所引起。
- [STABLE] [`PhaseDampingChannel`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.gates.html#mindquantum.core.gates.PhaseDampingChannel): 相位阻尼信道，量子比特没有于外界发生能量交换，但损失了量子信息。

#### FermionOperator and QubitOperator

- [STABLE] [`split`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.operators.html#mindquantum.core.operators.FermionOperator.split): `FermionOperator` 或者 `QubitOperator` 的方法，用于将系数和算符本身分开。

#### ParameterResolver

- [STABLE] [`astype`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.astype): 将参数解析器转化为指定的类型。
- [STABLE] [`const`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.const): 获取参数解析器的常数部分。
- [STABLE] [`is_const`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_const): 判断参数解析器是不是只有常数部分。
- [STABLE] [`encoder_part`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.encoder_part): 将部分参数设置为encoder参数。
- [STABLE] [`ansatz_part`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.ansatz_part): 将部分参数设置为ansatz参数。
- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.as_encoder): 将所有参数设置为encoder参数。
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.as_ansatz): 将所有参数设置为ansatz参数。
- [STABLE] [`encoder_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.encoder_parameters): 返回所有encoder参数。
- [STABLE] [`ansatz_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.ansatz_parameters): 返回所有ansatz参数。
- [STABLE] [`is_hermitian`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_hermitian): 检查参数解析器是不是厄米共轭。
- [STABLE] [`is_anti_hermitian`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_anti_hermitian): 检查参数解析器是不是反厄米共轭
- [STABLE] [`no_grad_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.no_grad_parameters): 返回所有不需要更新梯度的参数。
- [STABLE] [`requires_grad_parameters`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.requires_grad_parameters): 返回所有需要更新梯度的参数。

#### Simulator

- [STABLE] [`copy`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.html#mindquantum.simulator.Simulator.copy): 模拟器现在支持复制操作。
- [STABLE] [`apply_gate`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.html#mindquantum.simulator.Simulator.apply_gate): 在此次更新中，可以以导数的形式来作用一个参数化量子门。
- [BETA] [`inner_product`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.simulator.html#mindquantum.simulator.inner_product): 计算两个给定模拟器中量子态的内积。

#### IO

- [STABLE] [`BlochScene`](https://mindspore.cn/mindquantum/docs/en/master/mindquantum.io.html): 此次更新，我们支持搭建布洛赫球绘图场景，可在其上绘制量子态，也可以动态演示量子态的变化。

## MindQuantum 0.6.0 Release Notes

### 主要特性和增强

#### `QubitOperator`和`FermionOperator`的迭代器功能增强

- 对多项费米子或玻色子算符迭代，可得到每一项

- 当算符只有一项时，可通过`singlet`来获取每一个费米子或者玻色子

### 新增线路模块

- [**general_w_state**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-w-state): 制备w态量子线路
- [**general_ghz_state**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-ghz-state): 制备ghz态量子线路
- [**bitphaseflip_operator**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarybitphaseflip-operator): 比特翻转量子线路
- [**amplitude_encoder**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.library.html#mindquantumalgorithmlibraryamplitude-encoder): 振幅编码量子线路

### 高效线路操作

- `shift` ：平移量子比特

- `reverse_qubits`：翻转线路比特

### 特性增强

- `MaxCutAnsatz`: [**get_partition**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-partition)，获取max-cut切割方案
- `MaxCutAnsatz`: [**get_cut_value**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-cut-value)，获取某个切割方案的切割数
- `Circuit`: [**is_measure_end**](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.circuit.html#mindquantumcorecircuitcircuitis-measure-end)，判断量子线路是否是测量门结尾

### 支持量子线路的SVG绘图模式

- 在jupyter notebook模式下，调用量子线路的`svg()`接口能够绘制出svg格式线路图

### 新增量子噪声模拟

MindQuantum新增如下量子信道进行量子噪声模拟

- [`PauliChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatespaulichannel)：泡利信道
- [`BitFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesbitflipchannel)：比特翻转信道
- [`PhaseFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesphaseflipchannel)：相位翻转信道
- [`BitPhaseFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesbitphaseflipchannel)：比特相位翻转信道
- [`DepolarizingChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/master/mindquantum.core.gates.html#mindquantumcoregatesdepolarizingchannel)：去极化信道
