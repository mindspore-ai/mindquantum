# MindQuantum Release Notes

[View English](./RELEASE.md)

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
