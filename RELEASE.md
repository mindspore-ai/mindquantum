# MindQuantum Release Notes

[查看中文](./RELEASE_CN.md)

## MindQuantum 0.13.0 Release Notes

MindQuantum **0.13.0** introduces `mqvector_npu`, a device-resident state-vector simulator backend for Ascend 910B NPUs, along with native Ascend C QAIA simulated bifurcation solvers on the `qaia-cann` branch; extracts a shared circuit-tiling execution engine (`circuit_tiling`) with runtime-dispatched Arm SVE vectorization on Linux AArch64; rebuilds the `mqmatrix` density-matrix simulator on full row-major storage; replaces binary OpenMP thread switching with work-proportional team sizing; and modernizes the build system and Python 3.9–3.12 toolchain.

### Major Features and Enhancements

#### Simulator

- [BETA] **Device-resident Ascend 910B state-vector backend [`mqvector_npu`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html)** ([`!3172`](https://atomgit.com/mindspore/mindquantum/pull/3172)): Added a single-precision (`mindquantum.complex64`) full-amplitude state-vector simulator backend for Ascend 910B NPUs.
    - **End-to-end device execution**: Quantum states reside in NPU HBM, combining Vector-core local circuit tiling with Cube-core strict-FP32 dense operator fusion (disabling `HF32`/`FP16`/`BF16` precision downgrades). Circuit evolution, Pauli/CSR expectations, checkpoint-free hybrid adjoint gradients, exact multi-frequency parameter-shift rules, pure-state noise trajectories, tree-based sampling, and reduced density-matrix contractions run entirely on device, returning only scalar or target results to the host (~54 ms per adjoint gradient step on 24-qubit 4-layer MaxCut QAOA on Ascend 910B2).
    - **Precision and build-from-source requirements**: Supports `complex64` only, and input `Hamiltonian` objects must be single-precision (e.g., `Hamiltonian(op, mq.float32)` or `ham.astype(mq.float32)`). Because Ascend C kernels are compiled for a specific target SoC (`Ascend910B[1-4]`) against a local CANN installation, **pre-built release `.whl` packages do not include `mqvector_npu`**; users must source their local CANN environment and build from source with `ENABLE_ASCEND=ON` and the matching `MQ_ASCEND_SOC` (see [`docs/mqvector_npu.md`](./docs/mqvector_npu.md)).
- [STABLE] **Shared circuit-tiling engine (`circuit_tiling`) and Arm SVE vectorization** ([`!3146`](https://atomgit.com/mindspore/mindquantum/pull/3146), [`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165), [`!3175`](https://atomgit.com/mindspore/mindquantum/pull/3175)):
    - **Circuit tiling**: Unlike conventional gate fusion (which multiplies consecutive gates into a dense matrix, losing single-gate sparsity and fragmenting on parameterized gates), `mqsim_circuit_tiling` gathers a local 8-qubit state tile (256 amplitudes) into L1 cache once, applies multiple gates in place while preserving structured sparsity and folding adjacent diagonal runs, and scatters once. **Forward evolution and adjoint gradients share the same tiled execution path**, enabled automatically on `mqvector` when `n_qubits >= 13`.
    - **Cross-tile Arm SVE execution with runtime dispatch**: On Linux AArch64, dedicated SVE + FCMA kernels use Gather/Scatter, `FCMLA` complex multiply-accumulate, and predicate masks to evolve multiple independent state tiles in parallel, with automatic runtime dispatch based on host CPU capabilities. On Kunpeng 920F with NUMA affinity (`OMP_PLACES=cores OMP_PROC_BIND=close`), 20–28 qubit random circuits and MaxCut QAOA workloads run **3–5× faster** than 0.12.0.
- [STABLE] **[`mqmatrix`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html) density-matrix simulator rebuilt on full storage and circuit tiling** ([`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)): Treats an $n$-qubit row-major density matrix as a state vector over $2n$ index bits on top of `circuit_tiling`, exploiting Hermitian symmetry at runtime by evolving only the lower block triangle.
    - Noise channels lower directly to superoperators (diagonal channels such as phase damping and dephasing fold into adjacent diagonal gates), noisy and noiseless circuits share one C++ adjoint-differentiation path, and measurement-ending noisy circuits sample from a single exact mixed-state evolution. At 10 qubits, `apply_circuit` is **3× faster**, gradients are **2× faster**, and 1000-shot noisy sampling drops from **278 s to 0.12 s**.
- [STABLE] **Work-proportional OpenMP thread scheduling** ([`!3168`](https://atomgit.com/mindspore/mindquantum/pull/3168)): CPU simulators and operator sparsification now allocate 1 thread per 8,192 amplitudes (capped by `OMP_NUM_THREADS`), and gradient batches allocate workers within `workers * team <= MaxThreads`, eliminating many-core fork/join slowdowns on small states and speeding up mid-size batches by **1.6×–1.8×**.

#### Algorithm, Examples & AI Skills

- [BETA] **Native CANN / Ascend C DSB and BSB solvers ([`qaia-cann` branch](https://atomgit.com/mindspore/mindquantum/tree/qaia-cann))** ([`!3162`](https://atomgit.com/mindspore/mindquantum/pull/3162)): Added `kernel_sb.Solver` on the standalone `qaia-cann` branch for Discrete (DSB) and Ballistic (BSB) Simulated Bifurcation on Ascend 910B NPUs (built from source with CANN 9.1).
    - Combines batch-wide Cube matrix multiplication (`aclnnMatmul`) with a fused Ascend C Vector state-update kernel, and captures the full multi-step iteration schedule into a device execution graph (Graph Replay) for repeated solves without per-step host launches. On Ascend 910B2 (FP32, 100 replicas, 1000 steps), repeated solves across 23 QPLIB binary instances achieve median speedups of **52.1× (DSB)** and **37.8× (BSB)** over the single-threaded SciPy sparse CPU baseline (~195 ms per solve on the 7000-variable Gset G60 benchmark).
- [BETA] **QAIA Generalized Ballistic Simulated Bifurcation (`GBSB`)** ([`!3159`](https://atomgit.com/mindspore/mindquantum/pull/3159)): Added [`GBSB`](https://arxiv.org/abs/2508.17655) under `mindquantum.algorithm.qaia` with nonlinear bifurcation control, supporting `'cpu-float32'`, `'gpu-float32'`, and `'npu-float32'` backends.
- [STABLE] **Decoded Quantum Interferometry (DQI) and OTOC examples** ([`!3128`](https://atomgit.com/mindspore/mindquantum/pull/3128), [`!3142`](https://atomgit.com/mindspore/mindquantum/pull/3142)): Added a complete DQI combinatorial optimization implementation ([arXiv:2509.08328](https://arxiv.org/abs/2509.08328)) under `example/dqi_mindquantum/`, and `mqmps`-based many-body OTOC light-cone spreading ([arXiv:2506.10191](https://arxiv.org/abs/2506.10191)) and $\text{H}_2$ bond-length inversion ([arXiv:2510.19550](https://arxiv.org/abs/2510.19550)) studies under `example/otoc_simulation/`.
- [STABLE] **MindQuantum AI Agent Skills** ([`!3125`](https://atomgit.com/mindspore/mindquantum/pull/3125), [`!3126`](https://atomgit.com/mindspore/mindquantum/pull/3126)): Added 6 domain skill packages under `skills/` (covering core APIs, variational training, noisy simulation, QAIA solvers, circuit compilation, and quantum chemistry) for AI coding assistants, installable via `npx skills add https://atomgit.com/mindspore/mindquantum.git`.

### Bug Fixes

- **QAIA `LSB` and `USB` fixes** ([`!3124`](https://atomgit.com/mindspore/mindquantum/pull/3124)): Fixed the inverted logarithmic scale exponent ($2^{-l_x} \to 2^{l_x}$) and `x = 0` guard in `LSB`, and widened default `x` initialization in `USB` from `[-0.01, 0.01]` to `[-0.1, 0.1]` so uniform quantization does not zero out the driving force (restoring 2.0×–2.7× TTS speedups over `DSB` on MaxCut benchmarks).
- **QAIA backend conversions and coupling normalization fixes**: Accepted dense `numpy.ndarray` coupling matrices `J` on GPU/NPU backends; converted external spin inputs `x` in `calc_cut`/`calc_energy` and NPU external field `h` in `SB` properly; fixed CPU `CAC`/`CFC`/`SFC` coupling normalization to use element-wise sparse squares `self.J.power(2).sum()`.
- **Worker thread exception propagation** ([`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)): Caught exceptions thrown inside `ParallelBatch` worker threads and rethrew them on the calling thread instead of triggering `std::terminate` and crashing the Python interpreter.
- **`mqmatrix` derivative gates, Pauli channels, and measurement accuracy** ([`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)): Fixed `apply_gate(..., diff=True)` to zero local basis states outside the derivative's support; accumulated Pauli channels as $\sum_k p_k P_k \rho P_k$ directly to avoid square-root `NaN`s from rounding; validated finite positive trace and both branch weights during measurement.
- **OpenQASM `barrier` parsing** ([`!3156`](https://atomgit.com/mindspore/mindquantum/pull/3156)): Fixed `OpenQASM().from_string()` retaining only the first qubit when parsing `barrier` instructions over multi-qubit registers (e.g., `qreg q[4]; barrier q;`).
- **MindSpore device context and documentation fixes** ([`!3127`](https://atomgit.com/mindspore/mindquantum/pull/3127), [`!3130`](https://atomgit.com/mindspore/mindquantum/pull/3130), [`!3166`](https://atomgit.com/mindspore/mindquantum/pull/3166)): Replaced deprecated `ms.set_context(device_target="CPU")` calls with `ms.set_device("CPU")` in `mindquantum.framework`, and fixed typos and math rendering across API docstrings.

### Build System and Project Modernization

- **Configure-time OpenBLAS verification and `ENABLE_MPS` option** ([`!3174`](https://atomgit.com/mindspore/mindquantum/pull/3174)): Added `FindOpenBLAS.cmake` to verify all required `mqmps` symbols at configure time (preventing macOS from selecting Accelerate and failing at link time) and gated `mqmps` behind `-DENABLE_MPS=ON/OFF`.
- **Isolated pybind11 ABI and multi-arch `mqrt`** ([`!3137`](https://atomgit.com/mindspore/mindquantum/pull/3137), [`!3174`](https://atomgit.com/mindspore/mindquantum/pull/3174)): Enabled `MQ_ISOLATE_PYBIND11_INTERNALS` by default to isolate extension symbol tables; enabled `mqrt` builds on both AArch64 and x86_64.
- **Build scripts and toolchain improvements** ([`!3111`](https://atomgit.com/mindspore/mindquantum/pull/3111), [`!3153`](https://atomgit.com/mindspore/mindquantum/pull/3153), [`!3154`](https://atomgit.com/mindspore/mindquantum/pull/3154), [`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)): Fixed stale cache errors on consecutive `build.sh` runs, `--clean-all` directory resolution, and GCC 13+ `<cstdint>` headers for Catch2; migrated Python linting/formatting to Ruff and aligned local hooks with the CI gate.

### Breaking and Behavioral Changes

- **Python version requirement** ([`!3111`](https://atomgit.com/mindspore/mindquantum/pull/3111)): Supported versions are now **Python 3.9–3.12** (dropped Python 3.7 and 3.8).
- **API removal and exception fix** ([`!3168`](https://atomgit.com/mindspore/mindquantum/pull/3168)): Removed `Simulator.set_threads_number()` (use `OMP_NUM_THREADS` instead); accessing non-existent attributes on `mindquantum.framework` now raises `AttributeError` instead of `ImportError`.
- **`mqmatrix` storage, sampling, and validation changes** ([`!3165`](https://atomgit.com/mindspore/mindquantum/pull/3165)): Full-matrix storage doubles single-state memory (reducing max qubit count by 1 for a fixed memory budget) and adds 10%–15% overhead on $\le 6$-qubit circuits; terminal noisy sampling consumes random draws differently under a fixed `seed` (distribution unchanged); passing `circ_left`/`simulator_left` or invalid noise channel parameters now raises an error before execution.
- **Build and CMake option changes** ([`!3127`](https://atomgit.com/mindspore/mindquantum/pull/3127), [`!3148`](https://atomgit.com/mindspore/mindquantum/pull/3148), [`!3169`](https://atomgit.com/mindspore/mindquantum/pull/3169), [`!3174`](https://atomgit.com/mindspore/mindquantum/pull/3174)):
    - Requires **CMake $\ge$ 3.20** and **`pybind11 >= 3.1`**; building `mqvector_cq` now requires `-DENABLE_CUQUANTUM=ON` (or `build.sh --gpu --cuquantum`); Linux AArch64 defaults to `MQ_ARM_SVE=ON` (pass `-DMQ_ARM_SVE=OFF` if the compiler lacks SVE support);
    - Removed `MQ_BLAS_LIBRARIES`, `MQ_LAPACK_LIBRARIES`, and `BLAS_SYMBOL_PREFIX` in favor of `OpenBLAS_ROOT`, `OpenBLAS_LIBRARY`, and `OpenBLAS_SYMBOL_PREFIX`; build scripts install runtime and test dependencies by default and removed `--only-pytest` (use `--no-python-deps` to skip).

### Contributors

Thanks to the following developers for their contributions:

dsdsdshe, huangzhe-2026, liqingyu_2026, lllrt, mangrovecoder, Tsingpy.

Welcome contributions to the project in any form!

## MindQuantum 0.12.0 Release Notes

### Major Features and Enhancements

#### Device

- [BETA] [`QPU`](https://www.mindspore.cn/mindquantum/docs/en/master/device/mindquantum.device.QPU.html): Added an abstract base class for real quantum hardware execution. This class standardizes a `sampling`-driven interface that mirrors `mindquantum.simulator.Simulator`, enabling seamless backend swapping between simulators and real QPUs. Key capabilities include ([`!2842`](https://atomgit.com/mindspore/mindquantum/pull/2842)):
    - **Minimal vendor burden**: Hardware providers only need to implement the abstract `sampling()` method.
    - **Shot-based expectation estimation**: `get_expectation()` automatically performs Pauli basis rotations and statistical estimation from measurement shots.
    - **Shot-based Gradient estimation**: `get_expectation_with_grad()` returns a `GradOpsWrapper` supporting both parameter-shift rules (for RX, RY, RZ, U3, Rxx, Ryy, Rzz, and 10+ other gate families) and central finite-difference fallback, with intelligent per-parameter strategy selection.
    - **Batch optimization hook**: Vendors can optionally override `_sampling_batch()` for batched job submission.

#### Simulator

- [BETA] [`mqmps`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html) backend: Added a Matrix Product State (MPS) simulator that represents quantum states as tensor networks instead of full state vectors. When circuit entanglement is moderate, this backend can simulate many more qubits than state-vector methods with significantly less memory ([`!3088`](https://atomgit.com/mindspore/mindquantum/pull/3088)).
    - Supports single- and two-qubit gates (automatic SWAP insertion for non-adjacent qubits), measurement, sampling, reduced density matrix computation, and Pauli expectation values.
    - High-performance C++ core with BLAS/LAPACK backend for numerical stability.
- [STABLE] [`SequentialUCCAnsatz`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.mqchem.SequentialUCCAnsatz.html): Added a sequential UCC ansatz builder under `mindquantum.simulator.mqchem` that preserves excitation insertion order. Unlike standard UCC, the same excitation operator can appear multiple times with independent parameters, enabling layered/Trotter-style ansatz constructions for VQE workflows ([`!3096`](https://atomgit.com/mindspore/mindquantum/pull/3096), [`!3103`](https://atomgit.com/mindspore/mindquantum/pull/3103)).

#### Algorithm

- [BETA] QAIA family extension: Added quantized simulated bifurcation variants [`TSB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.TSB.html) (Ternary), [`USB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.USB.html) (Unary), and [`LSB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.LSB.html) (Linear) under `mindquantum.algorithm.qaia`, supporting both CPU and GPU backends ([`!2820`](https://atomgit.com/mindspore/mindquantum/pull/2820)).

### Improvements

- Improved error messages when submodules fail to import, now showing the actual cause ([`!2828`](https://atomgit.com/mindspore/mindquantum/pull/2828)).
- Clarified return value documentation for SABRE and MQSABRE mapping algorithms ([`!3099`](https://atomgit.com/mindspore/mindquantum/pull/3099)).

### Bug Fixes

- **MQSABRE mapping**: `InitialMapping` could produce invalid qubit assignments on certain topologies, potentially causing segfaults. Fixed ([`!2793`](https://atomgit.com/mindspore/mindquantum/pull/2793), [`!2794`](https://atomgit.com/mindspore/mindquantum/pull/2794)).
- **OpenQASM export**: Circuits with repeated measurements on the same qubit produced invalid OpenQASM output. Each measurement now maps to a distinct classical bit ([`!2826`](https://atomgit.com/mindspore/mindquantum/pull/2826)).
- **OpenQASM import**: The parser could not handle arithmetic expressions in gate parameters (e.g., `pi/4 + 0.5`). Replaced with a general expression evaluator ([`!2825`](https://atomgit.com/mindspore/mindquantum/pull/2825)).
- **`Circuit.apply_value`**: Custom gate internal state (e.g., `UCCExcitationGate`) was corrupted during parameter substitution. Gate state is now preserved ([`!3095`](https://atomgit.com/mindspore/mindquantum/pull/3095)).
- **Noise backend gradients API missing**: Fixed missing `NoiseBackend.get_expectation_with_grad` support. ([`!3097`](https://atomgit.com/mindspore/mindquantum/pull/3097)).
- **QAIA SB initialization**: `xi` was initialized incorrectly when external field `h` was provided, causing wrong optimization results ([`!2818`](https://atomgit.com/mindspore/mindquantum/pull/2818)).
- **QAIA error messaging**: Improved GPU import error message for SB solver to clearly indicate what is needed ([`!2788`](https://atomgit.com/mindspore/mindquantum/pull/2788)).

### Contributors

Thanks to the following developers for their contributions:

dsdsdshe, Waikikilick, Yaocheng Chen, MangroveCoder, liudongmei2025, wangyin888, zhoushijie2025, 宦晓玲.

Welcome contributions to the project in any form!

## MindQuantum 0.11.0 Release Notes

### Major Features and Enhancements

#### Simulator

- [STABLE] [`mqchem`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.mqchem.MQChemSimulator.html): Introduced the `mqchem` simulator backend. Based on the Configuration Interaction (CI) method, this simulator operates within a subspace of a fixed electron count, offering a more memory and computationally efficient solution for quantum chemistry problems. Its core components include ([`!2717`](https://gitee.com/mindspore/mindquantum/pulls/2717)):
    - [`MQChemSimulator`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.mqchem.MQChemSimulator.html): A simulator for the CI space, initialized to the Hartree-Fock state by default.
    - [`UCCExcitationGate`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.mqchem.UCCExcitationGate.html): A dedicated excitation gate for constructing UCC ansatz.
    - [`CIHamiltonian`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.mqchem.CIHamiltonian.html): A Hamiltonian wrapper for efficient expectation value calculations in the CI space.
    - [`prepare_uccsd_vqe`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.mqchem.prepare_uccsd_vqe.html): A high-level function to automate the preparation process for UCCSD-VQE experiments.
- [STABLE] [`mqvector_cq`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html): Added the `mqvector_cq` simulator backend, which leverages the NVIDIA cuQuantum SDK to further accelerate quantum circuit simulations on NVIDIA GPUs ([`!2724`](https://gitee.com/mindspore/mindquantum/pulls/2724)). This backend depends on cuQuantum, which must be properly installed in your environment. For details, please refer to the official cuQuantum installation guide.

#### Algorithm

- [STABLE] **QAIA Algorithm Backend Extension**: Added PyTorch-based GPU and NPU backend support for the Quantum Annealing Inspired Algorithm (QAIA) family (including [`ASB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.ASB.html), [`BSB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.BSB.html), [`DSB`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.DSB.html), [`LQA`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.LQA.html), [`CAC`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.CAC.html), [`CFC`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.CFC.html), [`SFC`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.SFC.html), [`NMFA`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.NMFA.html), and [`SimCIM`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.SimCIM.html)). This significantly boosts computational performance on the respective hardware. Users can now select `'gpu-float32'` or `'npu-float32'` via the `backend` parameter ([`!2669`](https://gitee.com/mindspore/mindquantum/pulls/2669), [`!2678`](https://gitee.com/mindspore/mindquantum/pulls/2678)).

### Bug Fixes

- [`PR2727`](https://gitee.com/mindspore/mindquantum/pulls/2727): Fixed an issue where the SVG export for circuit diagrams only displayed a single control qubit for multi-controlled CNOT gates.
- [`PR2717`](https://gitee.com/mindspore/mindquantum/pulls/2717): Fixed an indexing error for double-excitation amplitudes in `uccsd_singlet_get_packed_amplitudes`, ensuring the correctness of amplitudes extracted from CCSD calculations.
- [`PR2716`](https://gitee.com/mindspore/mindquantum/pulls/2716): Fixed a crash in the SABRE and MQSABRE mapping algorithms when processing non-consecutive physical qubit IDs (e.g., `[12, 13, 15]`).
- [`PR2713`](https://gitee.com/mindspore/mindquantum/pulls/2713): Fixed an issue with `TimeEvolution` when handling Hamiltonians containing a constant term (identity operator), which is now correctly converted to a global phase gate.
- [`PR2679`](https://gitee.com/mindspore/mindquantum/pulls/2679): Fixed an incorrect `J_norm` calculation in the NMFA algorithm when an external field `h` is included.
- [`PR2714`](https://gitee.com/mindspore/mindquantum/pulls/2714): Fixed a potential linker error in the Windows CI environment when Python is installed in a path containing spaces.
- [`PR2650`](https://gitee.com/mindspore/mindquantum/pulls/2650): Fixed a potential division-by-zero error in the `Rn` gate under certain conditions.
- [`PR2648`](https://gitee.com/mindspore/mindquantum/pulls/2648): Fixed the improper handling of `barrier` gates in the `sabre` mapping algorithm.

### Other Updates

- **QAIA Algorithm**:
    - Enhanced input parameter validation to ensure the coupling matrix `J` is symmetric with a zero diagonal, providing clearer guidance to users ([`!2710`](https://gitee.com/mindspore/mindquantum/pulls/2710)).
    - Optimized the [`LQA`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/qaia/mindquantum.algorithm.qaia.LQA.html) algorithm by caching intermediate results to reduce redundant computations ([`!2656`](https://gitee.com/mindspore/mindquantum/pulls/2656)).
    - Improved error messages for numerical instability in certain algorithms to be more instructive ([`!2710`](https://gitee.com/mindspore/mindquantum/pulls/2710)).
- **Dependencies**: Updated dependencies: `scipy>=1.13.1` is now required.
- Added warnings for the presence of complex numbers in `ParameterResolver` and for the use of `int8` precision with the `SB` simulator ([`!2651`](https://gitee.com/mindspore/mindquantum/pulls/2651)).

### Contributor

Thanks to the following developers for their contributions:

beastsenior, dsdsdshe, GhostArtyom, liushiwei2024, lyq, MangroveCoder, YangleiSHAO, zengqg, ZhuangJP, 葛宇非, 满成, 肖阳, 左博伟.

Welcome contributions to the project in any form!

## MindQuantum 0.10.0 Release Notes

### Major Features and Improvements

#### Algorithm

- [BETA] [`virtual_distillation`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/error_mitigation/mindquantum.algorithm.error_mitigation.virtual_distillation.html): Added error mitigation algorithm based on virtual distillation, which reduces quantum noise by creating virtual copies of quantum states and performing measurements on an entangled system.
- [BETA] [`QuantumNeuron`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/nisq/mindquantum.algorithm.nisq.QuantumNeuron.html): Added quantum neuron implementation based on Repeat-Until-Success (RUS) strategy, which simulates classical neuron behavior through quantum circuits by applying non-linear function rotations.
- [STABLE] [`SGAnsatz`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/nisq/mindquantum.algorithm.nisq.SGAnsatz.html): Added sequentially generated variational quantum circuit that can efficiently generate matrix product states with fixed bond dimension. The ansatz naturally adapts to one-dimensional quantum many-body problems by applying parameterized quantum circuit blocks to adjacent qubits.
- [STABLE] [`SGAnsatz2D`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/nisq/mindquantum.algorithm.nisq.SGAnsatz2D.html): Added two-dimensional sequentially generated variational quantum circuit that can generate string-bond states. Supports automatic generation of traversal paths by specifying 2D grid dimensions or construction of specific types of string-bond states through custom line sets.
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

#### Simulator

- [STABLE] [`get_reduced_density_matrix`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_reduced_density_matrix): Added functionality to obtain the reduced density matrix of specified qubits by performing partial trace over other qubits.
- [STABLE] [`get_qs_of_qubits`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_qs_of_qubits): Added functionality to obtain the quantum state of specified qubits. Returns state vector for pure states and density matrix for mixed states. Supports returning quantum states in ket (Dirac) notation.
- [STABLE] When using "stabilizer" as simulator backend, added support for [`reset`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.reset) to reset quantum states.
- [STABLE] When using "stabilizer" as simulator backend, added support for [`get_expectation`](https://www.mindspore.cn/mindquantum/docs/en/master/simulator/mindquantum.simulator.Simulator.html#mindquantum.simulator.Simulator.get_expectation) to calculate the expectation value of a given Hamiltonian in the current quantum state.

#### Compiler

- [STABLE] [`U3Fusion`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/compiler/mindquantum.algorithm.compiler.U3Fusion.html): Added compilation rule for fusing consecutive single-qubit gates into a single U3 gate. This rule scans the circuit and combines consecutive single-qubit gates acting on the same qubit into a single U3 gate. For standalone single-qubit gates, they are also converted to U3 form. Global phase tracking and inclusion is optional.
- [STABLE] [`u3_decompose`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/compiler/mindquantum.algorithm.compiler.u3_decompose.html): Added functionality to decompose U3 gates into Z-X-Z-X-Z rotation sequences. Supports both standard decomposition (U3(θ,φ,λ) = Rz(φ)Rx(-π/2)Rz(θ)Rx(π/2)Rz(λ)) and alternative decomposition (U3(θ,φ,λ) = Rz(φ)Rx(π/2)Rz(π-θ)Rx(π/2)Rz(λ-π)). When any rotation angle is constant and equals zero, the corresponding RZ gate will be omitted.
- [STABLE] [`DecomposeU3`](https://www.mindspore.cn/mindquantum/docs/en/master/algorithm/compiler/mindquantum.algorithm.compiler.DecomposeU3.html): Added compilation rule for U3 gate decomposition, breaking down U3 gates into Z-X-Z-X-Z rotation sequences. Supports both standard and alternative decomposition methods.

#### IO

- [STABLE] [`QCIS`](https://www.mindspore.cn/mindquantum/docs/en/master/io/mindquantum.io.QCIS.html): Added quantum circuit and QCIS format conversion class

#### Utilities

- [STABLE] [`random_hamiltonian`](https://www.mindspore.cn/mindquantum/docs/en/master/utils/mindquantum.utils.random_hamiltonian.html): Added random Pauli Hamiltonian generation functionality. Supports specifying the number of qubits and Pauli terms, with optional random seed for reproducibility. Generated Hamiltonians can be used for quantum algorithm testing and benchmarking.

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

### Contributor

Thanks to the following developers for their contributions:

Arapat Ablimit, Chufan Lyu, GhostArtyom, LuoJianing, Mr1G, Waikikilick, donghufeng, dsdsdshe, xuxusheng, yuhan, zengqg, zhouyuanyang2024, 王上, 杨金元, 糖醋排骨.

Welcome contributions to the project in any form!

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
