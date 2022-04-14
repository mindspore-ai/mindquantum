# MindQuantum 0.6.0

## MindQuantum 0.6.0 Release Notes

### Major Features and Improvements

#### Better iteration supported for `QubitOperator` and `FermionOperator`

-Iterate over a multinomial fermion or boson operator and yield each term

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