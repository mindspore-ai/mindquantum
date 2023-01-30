# MindQuantum Release Notes

[View English](./RELEASE.md)

## MindQuantum 0.8.0 Release Notes

### 主要特性和增强

#### Gates

- [STABLE] [`FSim`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.FSim): 支持费米子算符模拟门 fSim，fSim 门在变分量子算法中可以有效的运行。
- [STABLE] [`U3`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.U3): 单比特的任何量子门 U3 将会以一个单独的量子门存在，而不是一段量子线路。且 U3 门在变分量子算法中可以有效的运行。
- [STABLE] [`自定义量子门`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantum.core.gates.gene_univ_parameterized_gate)。自定义量子门现在会被即时编译器 [numba](https://numba.pydata.org) 编译成机器码，以提高运行效率。且编译后的自定义量子门可以在模拟器后端的多线程场景中运行。
- [STABLE] [`BarrierGate`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.BarrierGate): BarrierGate 现在可以只作用在某些特定比特上，而不是全部比特。
- [STABLE] [`KrausChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html?highlight=fsim#mindquantum.core.gates.KrausChannel): 用户可自定义 kraus 量子信道。

#### Circuit

- [STABLE] [`svg`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.svg): 现在可以通过设置 `width` 参数来将量子线路分段，从而可以将量子线路图以更美观的方式复制到论文中。

#### Simulator

- [STABLE] **全新量子模拟器**. 新版本中我们推出了全新的 cpu 和 gpu 模拟器： `mqvector` 和 `mqvector_gpu`. 旧版本中的 `projectq` 模拟器将会在下个版本中被弃用。全新一代模拟器与旧模拟器完全兼容，只需在模拟器声明时修改后端名称即可。

## MindQuantum 0.7.0 Release Notes

### 主要特性和增强

#### Circuit

- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.as_encoder)：`Circuit` 中的方法，将量子线路标记为编码量子线路。
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.as_ansatz)：`Circuit` 中的方法，将量子线路标记为训练量子线路。
- [STABLE] [`encoder_params_name`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.encoder_params_name)：`Circuit` 中的方法，返回量子线路中所有编码量子线路的参数名。
- [STABLE] [`ansatz_params_name`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.ansatz_params_name)：`Circuit` 中的方法，返回量子线路中所有训练量子线路的参数名。
- [STABLE] [`remove_noise`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.remove_noise)：`Circuit` 中的方法，用于将所有噪声信道移除。
- [STABLE] [`with_noise`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.Circuit.with_noise)：`Circuit` 中的方法，用于在每个非噪声门后面添加一个噪声信道。
- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.as_encoder)：一个装饰器，将所装饰函数返回的量子线路标记为编码量子线路。
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.as_ansatz)：一个装饰器，将所装饰函数返回的量子线路标记为训练量子线路。
- [STABLE] [`qfi`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.qfi)：用于计算给定参数化量子线路的量子fisher信息的方法。
- [STABLE] [`partial_psi_partial_psi`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.partial_psi_partial_psi)：计算量子fisher信息第一部分的方法。
- [STABLE] [`partial_psi_psi`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantum.core.circuit.partial_psi_psi)：计算量子fisher信息第二部分的方法。

#### Gates

- [STABLE] [`AmplitudeDampingChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantum.core.gates.AmplitudeDampingChannel)：振幅阻尼信道，由能量耗散所引起。
- [STABLE] [`PhaseDampingChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantum.core.gates.PhaseDampingChannel)：相位阻尼信道，量子比特没有与外界发生能量交换，但损失了量子信息。

#### FermionOperator and QubitOperator

- [STABLE] [`split`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.operators.html#mindquantum.core.operators.FermionOperator.split)：`FermionOperator` 或者 `QubitOperator` 的方法，用于将系数和算符本身分开。

#### ParameterResolver

- [STABLE] [`astype`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.astype)：将参数解析器转化为指定的类型。
- [STABLE] [`const`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.const)：获取参数解析器的常数部分。
- [STABLE] [`is_const`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_const)：判断参数解析器是不是只有常数部分。
- [STABLE] [`encoder_part`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.encoder_part)：将部分参数设置为encoder参数。
- [STABLE] [`ansatz_part`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.ansatz_part)：将部分参数设置为ansatz参数。
- [STABLE] [`as_encoder`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.as_encoder)：将所有参数设置为encoder参数。
- [STABLE] [`as_ansatz`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.as_ansatz)：将所有参数设置为ansatz参数。
- [STABLE] [`encoder_parameters`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.encoder_parameters)：返回所有encoder参数。
- [STABLE] [`ansatz_parameters`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.ansatz_parameters)：返回所有ansatz参数。
- [STABLE] [`is_hermitian`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_hermitian)：检查参数解析器是不是厄米共轭。
- [STABLE] [`is_anti_hermitian`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.is_anti_hermitian)：检查参数解析器是不是反厄米共轭。
- [STABLE] [`no_grad_parameters`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.no_grad_parameters)：返回所有不需要更新梯度的参数。
- [STABLE] [`requires_grad_parameters`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.parameterresolver.html#mindquantum.core.parameterresolver.ParameterResolver.requires_grad_parameters)：返回所有需要更新梯度的参数。

#### Simulator

- [STABLE] [`copy`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.simulator.html#mindquantum.simulator.Simulator.copy)：模拟器现在支持复制操作。
- [STABLE] [`apply_gate`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.simulator.html#mindquantum.simulator.Simulator.apply_gate)：在此次更新中，可以以导数的形式来作用一个参数化量子门。
- [BETA] [`inner_product`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.simulator.html#mindquantum.simulator.inner_product)：计算两个给定模拟器中量子态的内积。

#### IO

- [STABLE] [`BlochScene`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.io.html)：此次更新，我们支持搭建布洛赫球绘图场景，可在其上绘制量子态，也可以动态演示量子态的变化。

### 贡献者

感谢以下开发者做出的贡献：

yufan, wengwenkang, xuxusheng, Damien Ngyuen, zhouxu, wangzidong, yangkang, lujiale, zhangzhenghai, fanyi, zhangwengang, wangkaisheng, zhoufeng, wangsiyuan, gongxiaoqing, chengxianbin, sunxiyin, wenwenkang, lvdingshun, cuijiangyu, chendiqing, zhangkai, Zotov Yuriy, liqin, zengjinglin, cuixiaopeng, 朱祎康, dorothy20212021, dsdsdshe, buyulin, norl-corxilea, herunhong, Arapat Ablimit, NoE, panshijie, longhanlin.

欢迎以任何形式对项目提供贡献！

## MindQuantum 0.6.0 Release Notes

### 主要特性和增强

#### `QubitOperator`和`FermionOperator`的迭代器功能增强

- 对多项费米子或玻色子算符迭代，可得到每一项

- 当算符只有一项时，可通过`singlet`来获取每一个费米子或者玻色子

### 新增线路模块

- [**general_w_state**](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-w-state): 制备w态量子线路
- [**general_ghz_state**](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarygeneral-ghz-state): 制备ghz态量子线路
- [**bitphaseflip_operator**](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.algorithm.library.html#mindquantumalgorithmlibrarybitphaseflip-operator): 比特翻转量子线路
- [**amplitude_encoder**](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.algorithm.library.html#mindquantumalgorithmlibraryamplitude-encoder): 振幅编码量子线路

### 高效线路操作

- `shift` ：平移量子比特

- `reverse_qubits`：翻转线路比特

### 特性增强

- `MaxCutAnsatz`: [**get_partition**](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-partition)，获取max-cut切割方案
- `MaxCutAnsatz`: [**get_cut_value**](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.algorithm.nisq.html#mindquantumalgorithmnisqmaxcutansatzget-cut-value)，获取某个切割方案的切割数
- `Circuit`: [**is_measure_end**](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.circuit.html#mindquantumcorecircuitcircuitis-measure-end)，判断量子线路是否是测量门结尾

### 支持量子线路的SVG绘图模式

- 在jupyter notebook模式下，调用量子线路的`svg()`接口能够绘制出svg格式线路图

### 新增量子噪声模拟

MindQuantum新增如下量子信道进行量子噪声模拟

- [`PauliChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantumcoregatespaulichannel)：泡利信道
- [`BitFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantumcoregatesbitflipchannel)：比特翻转信道
- [`PhaseFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantumcoregatesphaseflipchannel)：相位翻转信道
- [`BitPhaseFlipChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantumcoregatesbitphaseflipchannel)：比特相位翻转信道
- [`DepolarizingChannel`](https://mindspore.cn/mindquantum/docs/zh-CN/r0.8/mindquantum.core.gates.html#mindquantumcoregatesdepolarizingchannel)：去极化信道
