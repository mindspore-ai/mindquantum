# MindQuantum Release Notes

[查看中文](./RELEASE_CN.md)

## MindQuantum 0.10.0 Release Notes

### Major Features and Improvements

#### Algorithm

- [BETA] [`virtual_distillation`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/error_mitigation/mindquantum.algorithm.error_mitigation.virtual_distillation.html): Added error mitigation algorithm based on virtual distillation, which reduces quantum noise by creating virtual copies of quantum states and performing measurements on an entangled system.
- [BETA] [`QuantumNeuron`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/nisq/mindquantum.algorithm.nisq.QuantumNeuron.html): Added quantum neuron implementation based on Repeat-Until-Success (RUS) strategy, which simulates classical neuron behavior through quantum circuits by applying non-linear function rotations.
- [STABLE] [`qjpeg`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/library/mindquantum.algorithm.library.qjpeg.html): Added quantum image compression algorithm based on quantum Fourier transform, which can compress quantum images by reducing the number of qubits while preserving key information in the frequency domain.
- [STABLE] [`cnry_decompose`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/compiler/mindquantum.algorithm.compiler.cnry_decompose.html): Added decomposition for CnRY gate.
- [STABLE] [`cnrz_decompose`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/compiler/mindquantum.algorithm.compiler.cnrz_decompose.html): Added decomposition for CnRZ gate.
- [STABLE] [`BSB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.BSB.html): Added GPU acceleration support for Ballistic Simulated Bifurcation algorithm with the three precision options: `'cpu-float32'`, `'gpu-float16'`, `'gpu-int8'`.
- [STABLE] [`DSB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.DSB.html): Added GPU acceleration support for Discrete Simulated Bifurcation algorithm with the three precision options: `'cpu-float32'`, `'gpu-float16'`, `'gpu-int8'`.
- [STABLE] [`qudit_symmetric_encoding`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/library/mindquantum.algorithm.library.qudit_symmetric_encoding.html): Added qudit encoding functionality that maps d-level quantum states to qubit states through symmetric encoding, enabling efficient simulation of higher-dimensional quantum systems on standard qubit-based quantum computers.
- [STABLE] [`qudit_symmetric_decoding`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/library/mindquantum.algorithm.library.qudit_symmetric_decoding.html): This feature introduces the ability to decode qubit symmetric states or matrices into qudit states or matrices, thereby enhancing the support for multi-level quantum systems. The decoding process involves transforming symmetric qubit states into corresponding qudit states, which facilitates efficient simulation of higher-dimensional quantum systems on standard qubit-based quantum computers.
- [STABLE] [`qutrit_symmetric_ansatz`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/library/mindquantum.algorithm.library.qutrit_symmetric_ansatz.html): Introduced a qutrit symmetric ansatz that constructs a qubit ansatz preserving the symmetry of encoding for arbitrary qutrit gates. This feature allows for efficient simulation of higher-dimensional quantum systems on standard qubit-based quantum computers by leveraging symmetry-preserving transformations. The ansatz supports decomposition into `"zyz"` or `"u3"` basis and can optionally include a global phase.

#### Measure

- [STABLE] [`MeasureResult.to_json`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.MeasureResult.html#mindquantum.core.gates.MeasureResult.to_json): for measurement result serialization and storage
- [STABLE] [`MeasureResult.reverse_endian`](https://www.mindspore.cn/mindquantum/docs/en/master/core/gates/mindquantum.core.gates.MeasureResult.html#mindquantum.core.gates.MeasureResult.reverse_endian): reverse bit order in measurement results, enabling flexible endianness handling for quantum state readouts

#### Operator

- [STABLE] [`mat_to_op`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/library/mindquantum.algorithm.library.mat_to_op.html): Added function to convert matrix to `QubitOperator`, supporting both little-endian and big-endian qubit ordering for seamless integration with different quantum computing frameworks.

#### Circuit

- [STABLE] Added [`Circuit.from_qcis()`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.from_qcis) and [`Circuit.to_qcis()`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.to_qcis) functions for QCIS format conversion
- [STABLE] Added `__eq__` and `__ne__` methods for circuit comparison
- [STABLE] [`Circuit.depth()`](https://www.mindspore.cn/mindquantum/docs/en/master/core/circuit/mindquantum.core.circuit.Circuit.html#mindquantum.core.circuit.Circuit.depth): Added function to calculate quantum circuit depth with options to include single-qubit gates and align gates to barriers.

#### IO

- [STABLE] [`QCIS`](https://www.mindspore.cn/mindquantum/docs/en/master/io/mindquantum.io.QCIS.html): Added quantum circuit and QCIS format conversion class

### Breaking Changes

- [IMPORTANT] The byte order of `keys` and `samples` in `MeasureResult` has been unified to little-endian. If your code uses these attributes, please carefully check and use the newly added `reverse_endian` method to adjust if needed.

### Bug Fixes

- [`PR2497`](https://gitee.com/mindspore/mindquantum/pulls/2497): Fixed potential parameter name duplication in **Amplitude Encoder**.
- [`PR2410`](https://gitee.com/mindspore/mindquantum/pulls/2410): Fixed `is_measure_end` error that returned `True` even without measurement operations.
- [`PR2410`](https://gitee.com/mindspore/mindquantum/pulls/2410): Fixed incorrect calculation results after reversing qubit order in two-qubit gates.
- [`PR2377`](https://gitee.com/mindspore/mindquantum/pulls/2377): Fixed recursive error in `DAGCircuit` when processing deep circuits, now supporting arbitrary depth circuits.
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345): Fixed calculation error in `mqmatrix`'s `get_expectation_with_grad` method when processing batch Hamiltonians and added test cases.
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345): Fixed error when using `reverse_qubits` with gates not added in specified order.
- [`PR2345`](https://gitee.com/mindspore/mindquantum/pulls/2345): Fixed error in `FermionOperator.hermitian()` example code.
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319): Fixed measurement error in Stabilizer simulator.
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319): Fixed seed not properly applied in Stabilizer simulator.
- [`PR2319`](https://gitee.com/mindspore/mindquantum/pulls/2319): Added verification for bit string correctness in Stabilizer simulator output.
- [`PR2315`](https://gitee.com/mindspore/mindquantum/pulls/2315): Made **MQSim** and **Hamiltonian** serializable, supporting Python multiprocessing.
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309): Fixed missing imaginary terms and coefficients in some **QAOA** ansatzes.
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309): Fixed non-working `QAOAAnsatz` example.
- [`PR2309`](https://gitee.com/mindspore/mindquantum/pulls/2309): Modified parameter names in ansatz circuits to match formulas.
- [`PR2296`](https://gitee.com/mindspore/mindquantum/pulls/2296): Fixed index error in `kron_factor_4x4_to_2x2s()` return values, ensuring correctness of two-qubit gate decomposition function `kak_decompose`.
- [`PR2285`](https://gitee.com/mindspore/mindquantum/pulls/2285): Removed unnecessary output during gradient computation.

### Other Updates

- Optimized first-time quantum circuit execution speed for improved performance.
- Improved precision of `params_zyz()` function, enhancing **ZYZ** decomposition accuracy.
- Removed warning for uninstalled `mqvector_gpu`, now only prompting when used.
- Removed warning for uninstalled MindSpore, now only prompting when used.
- Added warning when Hamiltonian contains imaginary parts, alerting users to potential calculation anomalies.
- Enhanced clarity of warning messages when MindSpore is not installed.
- Changed `pip` source to Tsinghua mirror.

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
