# Ascend state-vector simulator

`mqvector_npu` keeps a full complex64 state vector on a single Ascend 910B NPU.
It uses Vector cores for structured operators and reductions and Cube cores for
sufficiently large fused dense operators. It never substitutes CPU simulation,
complex128 emulation, FP16, BF16 or HF32 arithmetic.

## Build and select the device

The implemented toolchain is Linux, CANN 8.5 and the Ascend C compiler. The exact
SoC must be specified at build time and must match the device at runtime.
Hardware verification uses Ascend910B2; other 910B variants require their own
build and hardware qualification. CPU-only builds remain the default and do not
discover or link CANN.

After installing MindQuantum's normal development dependencies and sourcing
your CANN installation's `set_env.sh`:

```bash
cmake -S . -B build-npu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DIN_PLACE_BUILD=ON \
  -DPython_EXECUTABLE="$(command -v python3)" \
  -DMQ_PYTHON_PACKAGE_NAME=mindquantum \
  -DENABLE_ASCEND=ON -DMQ_ASCEND_SOC=Ascend910B2 \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH"
cmake --build build-npu --target python -j8
```

For a Python wheel, use the repository's normal build frontend with the same
explicit target selection:

```bash
MQ_CIBW_BUILD_ARGS='--set,ENABLE_ASCEND,--var,MQ_ASCEND_SOC,Ascend910B2' \
  python3 -m build .
```

On AArch64 hosts without SVE, also set `-DMQ_ARM_SVE=OFF`. CANN and the driver
must remain available at runtime; they are not vendored into MindQuantum. For
the wheel command, append `,--unset,MQ_ARM_SVE` to `MQ_CIBW_BUILD_ARGS` on such hosts.
`device_id` is the **logical ACL ordinal** after container/device visibility
mapping, not a physical host card identifier. Do not change visibility merely
because a container lacks a working `npu-smi` command.

```python
import mindquantum as mq
from mindquantum.core.circuit import Circuit
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.simulator import Simulator

sim = Simulator('mqvector_npu', 20, device_id=0, seed=831)
circuit = Circuit().h(0).ry('theta', 1).x(2, 1)
ham = Hamiltonian(QubitOperator('Z1') + QubitOperator('X0', 0.3), mq.float32)
operator = sim.get_expectation_with_grad(ham, circuit)
expectation, gradient = operator([0.37])
sim.apply_circuit(circuit, {'theta': 0.37})
value = sim.get_expectation(ham)
samples = sim.sampling(Circuit().measure_all(up_to=20), shots=1000, seed=9)
```

Import and backend discovery do not initialize ACL. Construction defaults to
`mq.complex64`; explicitly requesting another dtype fails. Hamiltonians must
have matching single precision, for example `ham.astype(mq.float32)` for a
Pauli Hamiltonian. Device, allocation, compilation and execution errors
propagate; there is no retry or simulator fallback.

## Supported contracts

- Built-in deterministic gates, controls, dagger gates, Pauli strings and dense
  custom matrices on up to six targets. A gate's mixing support plus the three
  aligned low index bits must fit the selected Vector tile. Large Pauli strings
  can retain external Z parity and controls instead of expanding a dense matrix.
- `apply_circuit`, `apply_gate` (including intrinsic `diff=True`), `set_qs`,
  `get_qs`, reset, independent copies, same-precision `astype` and circuit matrices.
- Pauli and full complex CSR Hamiltonians, including non-Hermitian CSR and
  distinct left/right circuits and states; `apply_hamiltonian` and inner products.
- Adjoint gradients, shared parameters, `no_grad`, encoder/ansatz batches and
  multiple observables. Plans snapshot circuits and Hamiltonians, while each
  invocation reads the simulator's current input state without changing it.
  The wrapper's circuit/Hamiltonian attributes are informational references,
  as on the other backends; mutating them cannot change the native execution
  snapshot, captured parameter order or privately retained custom callbacks.
  Custom gates must be unitary for adjoint reconstruction. Constant matrices
  are checked when creating the plan; parameter-dependent custom matrices are
  checked at every binding. Nonunitary evolution and non-Hermitian observables
  are separate, supported operations.
- Exact, **gate-local** parameter-shift gradients. Controlled rotations use
  their full frequency spectrum; U3, FSim, Givens and SWAPalpha have their own
  exact rules. Shared input parameters are differentiated through every gate's
  intrinsic-angle Jacobian, not by shifting the shared parameter globally.
  Parameterized custom gates and Rn have adjoint support but no parameter-shift
  rule; the backend rejects them instead of using finite differences.
- Measurement/reset, terminal and mid-circuit sampling, Pauli/grouped Pauli,
  multi-qubit depolarizing, amplitude/phase damping, single-qubit Kraus and
  thermal-relaxation channels. Noise uses normalized pure-state trajectories.
  Noisy gradients automatically use exact parameter-shift and return stochastic
  estimates: average independent evaluations to estimate the channel derivative.
  As with the CPU trajectory API, noisy gradients do not accept a distinct bra.
- Reduced density matrices and `get_qs_of_qubits`, contracted directly from the
  device state. Only the requested matrix is downloaded. A pure reduced matrix
  is diagonalized on the host to return its state vector. Pure-state entropy,
  purity and `get_pure_state_vector` follow the other vector backends.

Seeds reproduce results within this backend, not CPU/CUDA random-number streams.
Sampling and expectation/gradient circuit arguments do not mutate the input
state. Invalid circuit structure is checked before evolution. A device failure
after an in-place launch does not promise rollback; discard the failed state.
Density-matrix-only methods such as `partial_trace` are not implemented.

## Execution architecture

The shared `circuit_tiling::Builder` is the single source of quantum gate and
derivative arithmetic. The NPU compiler lowers its semantic operators into a
small device instruction set, rather than reimplementing each gate formula.
`Planner::Capacity::FullSupport` includes every operand/control when selecting
dense operators; the CPU planner's default mixing-only policy is unchanged.

Compilation, binding, placement and execution have separate lifetimes:

1. A gradient plan validates and snapshots each input circuit once, sharing the
   immutable semantic input between `ForwardPlan` and `AdjointPlan`. Pure
   constant blocks are compiled immediately. Ordered physical support determines
   shared packet-routing tables; sharing a table still updates each block's own
   physical layout. The scheduler reuses a segment's dependency workspace across
   candidate evaluations and tile commits instead of copying the entire graph
   for every tile.
2. Binding synthesizes parameter-dependent operators/derivatives in host double
   precision. It shares all other program storage. Symbolic `no_grad` expressions
   remain dynamic, and custom callbacks are never constant-folded. Dynamic
   Vector instructions may change shape at zero angles, so they are encoded from
   the actual parameters, never from a dummy-zero template. Independent built-in
   Cube blocks are synthesized in parallel within the `OMP_NUM_THREADS` budget;
   custom callbacks retain serial execution. Dense tangents begin evolving only
   after their parameter's first contribution to the block.
3. Placement assembles each complete bound program into one self-contained
   image, deduplicating shared routing and coefficient pools within that image.
   Matmul tiling is computed once per distinct shape in the assembly. Pinned and
   device buffers reuse their capacity, growing only when needed; no address
   refers into another program's image and no dummy binding is needed.
4. Execution uploads the complete image once per binding and reuses it across
   observables. Uploads and kernels share one stream; transaction guards drain
   it before releasing any call-local storage, including on exceptions.

Forward execution combines:

1. FP32 real/imaginary planes, aligned packet DMA and double-buffered Vector
   tiles; diagonal folding, coalesced pair strides and indexed gathers.
2. Complex operators of at most 64 by 64, realified into strict FP32 Cube
   matrix products, with eight-packet batches and bounded two-slot staging.
3. Internal qubit-layout changes retained between fused operators. The final
   executable restores canonical little-endian order, so observables, sampling
   and gradients share one public state representation.

Adjoint execution uses the same structured-Vector/dense-Cube split. For a dense
block B, the compiler applies the product rule to Builder's gate/derivative
images in complex128, combining all occurrences of a shared parameter into one
dB-dagger matrix. Packet routing is planned once. Cube reconstructs both states
with B-dagger and contracts each dB-dagger against the original costate and the
reconstructed state. Each derivative uses one Matmul over up to eight packets,
followed by a compensated batch sum and one partial-record update. Staging is
reused sequentially across derivatives, without grouped-M output decoding or a
second derivative pipeline. Local MTE3-to-scalar fences obtain free TPipe event
IDs rather than reusing the event reserved by Matmul's asynchronous message
protocol. Other packet fences retain literal IDs for their paired local events.
Both states retain the same physical layout across Cube and Vector blocks:
their inner product is invariant under that common basis
permutation. These disposable sweep states need no canonical restoration.
The Vector path evolves two spatial tiles resident in UB together, combining
each intrinsic derivative's contractions before applying its Jacobian targets
and updating device partials. Scratch is independent of target count; a final
single tile uses the same loop. Each AIV accumulates the initial overlap across
all its tiles and writes that partial once. Evolution-only runs fold adjacent
diagonals and load instruction metadata once for both states; derivative
boundaries remain explicit. Both paths zero-extend derivatives outside their
controls/support.

Cube staging is bounded per core and reused across parameters; Vector derivative
scratch stays in UB. There are no per-gate or per-parameter full-state checkpoints. Complex two-sided
gradients use a left sweep with H and a conjugated right sweep with H dagger;
only identical sides with known Hermitian observables use twice the real part.
Device partials use balanced reductions and compensated accumulation, followed
by a small host double-precision reduction. Each AIV owns its output records;
updates preserve both the sum and its correction, without atomics, a UB hash
table or a fixed parameter-count limit. The two complex components share a balanced
reduction tree rather than separately synchronizing scalar `ReduceSum` results.

Pauli observables group terms by flip mask to share state reads. A gradient plan
may precompute one real I/Z Hamiltonian diagonal on the device, choosing the
eligible observable with the most terms. The immutable cache is shared across
evaluations and both sweeps when H and H-dagger are the same Pauli operator, so
applying that Hamiltonian becomes a single elementwise product.
Expectation-only execution never allocates a full H|psi> state. Sampling builds a device probability
tree and transfers only bin masses and sampled indices. Reduced-state contraction
pairs only matching traced indices, without constructing a full density matrix.

ACL contexts are shared by device with explicit ownership. Executables and
observables are immutable. Each active gradient evaluation exclusively owns its
stream, state scratch and dynamic metadata. A plan retains at most one completed
workspace for reuse; concurrent calls acquire separate workspaces, and excess
workspaces are released on completion. Aliased state pairs are locked once.
Exception paths drain outstanding work and discard the failed workspace instead
of returning it to the plan. Destruction does not reset the device or finalize
another library's ACL runtime.

## Memory and performance boundaries

A state requires `8 * 2**n` bytes (2 GiB at 28 qubits), plus bounded workspace.
The index ABI supports 0–32 qubits, **subject to available device memory**.
A Hermitian gradient requires two additional full states for one observable,
three for multiple observables; a general two-sided gradient requires four.
Scratch is reused across samples and calls, not multiplied by parameter count or
depth. The one idle workspace remains allocated until its gradient plan is
destroyed; retaining several gradient plans therefore retains several working
sets. No allocation retry or memory-pressure fallback is performed.
The optional diagonal cache adds at most `4 * max(8, 2**n)` bytes per gradient plan
(64 MiB at 24 qubits), independent of the number of observables. Dense adjoint
staging is bounded independently of state size and total parameter count:
60 MiB on a 48-Vector-core 910B2, plus CANN's mixed-core workspace. It holds
the input and reconstructed state pair and one derivative result for each
eight-packet batch, not one buffer per circuit parameter.
Compiled operator metadata is separate from state scratch: the plan retains
immutable host data proportional to circuit depth and distinct routing tables.
Each concurrent evaluation owns its complete bound device metadata and
staging; repeated evaluations do not grow a cache of parameter values/results.
`apply_hamiltonian` requires a separate output state. State download and explicit
circuit/reduced matrices require host memory proportional to their output.

Small matrices synthesized on the host are compilation metadata, not host state
evolution. Parameter batches currently execute sequentially on the NPU;
`parallel_worker` does not create CPU worker pools. Noisy sampling executes
individual trajectories and synchronizes at random branch decisions. These paths
are correct but are not throughput-optimized for large batches of tiny circuits.
One Kraus candidate is reused within a trajectory, and parameter-shift state
buffers are reused within each invocation, never shared across active calls.
Arbitrary CSR uses scalar indexed loads and is not expected to match grouped
Pauli throughput. Gradient timings include coefficient/derivative synthesis,
observable application and reverse contractions; forward Cube performance must
not be reported as gradient performance.

The defaults (`tile_qubits=12`, `fusion_qubits=6`, `cube_min_gates=4`) target
large deterministic states. Both directions require `4 <= tile_qubits <= 12` and
`1 <= fusion_qubits <= tile_qubits - 3`, checked without compiling an empty
circuit. These options are upper budgets, not exact tile sizes: the state size
and execution engine can lower them. The Vector adjoint tile is capped at 11 bits
and its fusion budget at eight to fit its additional live states and aligned DMA;
Cube support remains capped at six. `cube_min_gates=0` selects Vector-only execution; it is an
explicit execution policy, not a runtime recovery path. Defaults are not a claim
of optimality for every circuit or every 910B model.

## Hardware verification and benchmarking

```bash
python -m pytest -q tests/st/test_simulator/test_mqvector_npu.py \
  tests/st/test_simulator/test_mqvector_npu_api.py
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 \
  python tests/benchmark_mqvector_npu.py --qubits 24 28 --gradient-qubits 20
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 \
  python tests/benchmark_mqvector_npu.py --maxcut 24 --seed 42 --repeats 20 --optimize
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=1 \
  python tests/benchmark_mqvector_npu.py --uccsd path/to/h10_sto3g.hdf5 --seed 7281 --repeats 4
```

The tests use independent NumPy embeddings and complex128 CPU references,
asymmetric complex matrices, external controls/parities, sparse operators,
measurement boundaries, trajectory averages and finite-difference checks of
gradients. They also check no-state-download paths, source snapshots, independent
contexts/states and invalid-input atomicity. Without the compiled extension,
hardware tests skip; with it, device/runtime errors fail rather than skip.

The benchmark emits JSON lines with every measured repeat, device/compiler
context, full-state relative error, expectation error and gradient error. Its
first `apply_circuit` includes conversion, planning, matrix synthesis, upload,
launch and synchronization. Construction, download, explicit compiled replay,
gradient-plan construction and gradient invocation are reported separately.
There is no hidden warm-up or subtraction of compilation from first-call timing.
CPU complex128 references are correctness references, not equal-precision
hardware comparisons. Run without other tests/jobs on the device and retain the
output with the build configuration; do not extrapolate one circuit's timing to
all simulation workloads.

The MaxCut command above uses a seed-42 four-regular graph, four QAOA layers and
the legacy NumPy seed-42 initial point. `--optimize` runs SciPy's unmodified default
BFGS and records every gradient call, accepted-iteration call counts, termination
status and a complex128 check at the final point. FP32 loss of precision can make
line searches repeatedly evaluate candidates without accepting another iteration;
iteration count alone is not a measure of total gradient work. `--parameters`
can replay a saved point without changing the workload.

The UCCSD command requires OpenFermion and a saved `MolecularData` file containing
the chemistry calculation. It loads that file rather than rerunning CCSD/FCI,
prepends the Hartree–Fock X gates, and reports circuit generation and gradient
construction separately. It evaluates the returned `x0` directly, followed by
seeded perturbations, and checks the first result against CPU complex128. It
does not run BFGS. Both generated and circuit parameter orders are recorded:
they can differ, and silently reordering `x0` would change the benchmark point.
