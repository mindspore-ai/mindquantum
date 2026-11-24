# Building and distributing MindQuantum for Ascend

The Linux Ascend build includes both `mqvector_npu` and QAIA's `ascend-float32`
backend, alongside the CPU backends. CUDA is an independent, explicit option.
Neither CMake configuration nor wheel construction requires access to an NPU.

## Select the environment, then build

Install CANN Toolkit **and the matching device Ops package** in the same prefix.
QAIA needs the ACLNN Matmul headers, `libopapi_nn` and `libnnopbase`; Toolkit alone
is not sufficient. Source that installation's `set_env.sh` before configuring
or building. Do not combine headers, libraries or compilers from different
installations.

The release environment selects and pins the CANN version, host compiler,
Python and host ABI. MindQuantum does not have a `--cann-version` switch or
require an exact SDK version string: CMake checks the APIs and libraries it
uses, including graph capture and repeatable Matmul executors. Architecture
targets require `ACL_DEV_ATTR_NPU_ARCH` and native ASC architecture selection,
introduced in CANN 9.1. SoC targets require `aclrtCheckArchCompatibility`
instead; they do not depend on the 9.1 architecture-query API. CANN 9.1 remains
a candidate release baseline, not a claim of compatibility with every runtime
or every device it supports.

For example, with an activated CANN environment and a Python supported by
MindQuantum:

```bash
bash build.sh --no-config --npu --npu-arch=dav-2201,dav-3510 -B build-npu -o output-npu -j8
```

Wheel repair selects a manylinux platform automatically, just as for CPU/GPU
builds. To enforce an agreed release host ABI, optionally add
`--plat-name=manylinux_2_34_aarch64`, for example; this is not a project-wide
minimum. See [Package boundary](#package-boundary) for what the tag guarantees.

`--npu` maps to `ENABLE_ASCEND=ON`. Select a compiler target, not the build
machine's device:

- `--npu-arch=<arch,...>` sets native `CMAKE_ASC_ARCHITECTURES`. With CANN 9.1,
  `dav-2201` targets 910B and `dav-3510` targets 950PR/950DT. A comma-separated
  list builds both sets of kernels into **one wheel**, using one toolchain.
  Native CMake callers use a quoted list: `-DCMAKE_ASC_ARCHITECTURES='dav-2201;dav-3510'`.
- Alternatively, `--npu-soc=<soc>` sets `MQ_ASCEND_SOC`, passed directly to
  the compiler's `--npu-soc`. Accepted names depend on the installed SDK:
  CANN 9.1 accepts `Ascend910B2` but requires architecture selection for 950.

Select architectures or a SoC, not both. MindQuantum has no product-model or
architecture whitelist and does not translate marketing names into compiler
targets. The installed CANN compiler validates the target. Use its architecture
selector for later hardware; do not assume a binary compiled for one architecture
runs on another.
Each compiler invocation still targets one architecture. ASC is enabled once
in the common kernel directory, isolated from Host compiler flags; each kernel
target passes its architecture to both compilation and linking. All variants
belong to one build graph. CPU and Ascend Host code compile only once; there
are no external sub-builds or wheel-merging steps. Target identifiers contain
letters, digits, `.`, `_` or `-`; empty list entries and quoting characters fail
before SDK discovery. Repeated targets are compiled only once. SoC selection
remains a single target.

The SDK has no default target. For repeated builds, add a target default to
the `[common]` section of the existing `build.conf`:

```ini
[common]
npu_arch = dav-3510
```

Then `bash build.sh --npu` uses that configuration without requiring
`--plat-name`. The target default does not enable NPU support: CPU/GPU builds
and `--no-npu` ignore it. An explicit `--npu-arch` or `--npu-soc` command-line
option still requires NPU support to be enabled.

Targets can also come from native CMake configuration or, for wheels,
`MQ_CIBW_BUILD_ARGS`. CMake validates the final selection, including a target
already configured in the build directory. Missing or conflicting targets fail
configuration; there is no device autodetection or implicit 910B default, so
release builds also work without a visible NPU.

Before loading any kernel library, architecture builds select the exact target
reported by `ACL_DEV_ATTR_NPU_ARCH`. Only that backend's matching library is
loaded, using its path relative to the Host extension, not an environment search
path. Missing targets or broken libraries fail immediately; there is no attempt
to launch another architecture's kernels. Selected libraries remain mapped
until process exit (`RTLD_NODELETE`), because CANN registers process-wide
profiling callbacks into their code. This retains code and static tables, not
MindQuantum's contexts, streams or device buffers: those keep their existing
explicit owners and cleanup. Dispatch adds a function-pointer call, not a device
query, to a launch; captured graph replay is unchanged. SoC builds use CANN's
`aclrtCheckArchCompatibility`, not exact model-name equality. The [CANN 9.1 implementation](https://gitcode.com/cann/runtime/blob/v9.1.0/src/runtime/core/src/api_impl/api_impl.cc)
of that API resolves both SoC names to NPU architectures and compares them for
equality; it does not accept `dav-*` inputs. MindQuantum does not maintain a
SoC-to-architecture mapping.

These preflight checks must not be replaced by waiting for a kernel-launch error.
On CANN 9.1 with a 910B2, the ASC launch path accepted MindQuantum's `dav-3510`
reset kernel and reported a Vector Core illegal-instruction error only at stream
synchronization. Loading or submitting a binary is not a compatibility verdict.

Hardware qualification is separate from target selection: the backends have
been tested on 910B2, and their kernels compile for `dav-3510` with CANN 9.1.
A 950DT release still needs numerical, graph-replay and performance tests on
950DT hardware with its matching device Ops package. Compilation is not
hardware qualification.

For in-source development:

```bash
bash build_locally.sh --no-config --npu --npu-arch=dav-2201 -B build-npu -j8
```

For AArch64 hosts without SVE, append `-- -DMQ_ARM_SVE=OFF` to the local
command. For wheels, set `MQ_CIBW_BUILD_ARGS='--unset,MQ_ARM_SVE'`. The host
instruction-set baseline must match the intended users, not just the build
machine; `--unset,USE_NATIVE_INTRINSICS` can be added to `MQ_CIBW_BUILD_ARGS`
when native CPU tuning is inappropriate.

The underlying PEP 517 entry point remains available:

```bash
MQ_CIBW_BUILD_ARGS='--set,ENABLE_ASCEND,--var,CMAKE_ASC_ARCHITECTURES,dav-2201;dav-3510' \
MQ_DELOCATE_WHEEL=1 python3 -m build --wheel .
```

`MQ_CIBW_BUILD_ARGS` separates arguments with commas, so its architecture value
must use CMake's **semicolon** separator. Only the `--npu-arch` shell option
translates comma-separated architectures to a CMake list.

Use a clean workspace for release jobs. Use a new build directory when changing
CANN, the compiler or the target. Ordinary incremental builds of the same
configuration are supported; automatic migration between toolchains is not.
`--fast-build` is a developer shortcut using an already configured in-source
build, not a release recipe. GPU jobs must explicitly pass `--gpu`; being in a
CI environment no longer selects CUDA implicitly. Requested CUDA support fails
configuration if its compiler is missing or too old, or if CMake cannot find
the requested CUDA toolkit; it does not produce a CPU-only wheel instead.

## Package boundary

CMake determines which native extensions exist. Packaging builds that graph
once, then stages its outputs. A disabled backend is absent; a compilation
failure in an enabled backend fails the build, including light wheels. Native
outputs belong to the CMake build directory, not the wheel staging directory.
For full wheels, including `--fast-build`, CMake installs only its `Development`
component (headers, development libraries and CMake files). Python extensions
and runtime libraries follow the same build-to-staging-to-source copy path;
CMake's native Python/SDK install components are not installed into a wheel.
This avoids nested `site-packages` copies without reconfiguring the development
build. A native `cmake --install` without a component still installs everything.
`--fast-build` reuses the in-place graph's linker settings, including development
RPATHs in Host extensions; use a normal clean build for release wheels.

The wheel contains MindQuantum's own kernel code. CANN libraries and their
driver dependencies stay external; ordinary dependencies are still repaired
by auditwheel. Linux repair requires auditwheel **6.4 or newer**: libraries are
excluded by their resolved location under the activated `ASCEND_HOME_PATH`,
not by an SDK-wide list of filenames that could also match CPU dependencies.
Keep the same CANN environment active throughout configuration, compilation
and repair.

CPU, GPU and NPU wheels use auditwheel's native `auto` platform selection by
default, based on the wheel's CPU architecture and ABI requirements, including
bundled dependencies. MindQuantum does not infer a tag from the host glibc
version or retry a list of platforms. Optionally set `--plat-name` (or
`MQ_DELOCATE_WHEEL_PLAT` for PEP 517) to enforce the release baseline: repair
fails if that ABI cannot be satisfied, and does not add older policy tags
(equivalent aliases are still allowed). Repair failures propagate; they do not
fall back to a generic `linux_*` wheel. `--no-delocate` explicitly skips repair
and produces an unrepaired wheel instead.

Neither automatic nor explicit manylinux tags validate external CANN/CUDA
libraries or driver requirements, or describe the NPU target. Qualify and
document those separately in the release environment; an explicit tag is an
ABI constraint on the wheel, not a substitute for SDK/runtime validation.

Standard wheel tags do not select an Ascend architecture. Publish one wheel
containing the required architectures rather than competing single-target wheels
with identical version/Python/platform tags. Each backend has one Host extension
and one small kernel library per target under `mindquantum/lib/ascend/`; the
architecture is also part of each library's filename/SONAME. Runtime libraries
are required package contents, including in light wheels and in-place builds.
The private `mindquantum/_build_info.json` records the architecture list (or
single SoC), Python targets and runtime files. It contains no build-machine
paths or SDK inventory and does not replace hardware qualification. All variants
in a wheel share the build's Host ABI and CANN API baseline; this is not a way
to combine arbitrary CANN versions or independently built wheels.

Release validation must install the repaired wheel outside the source tree,
run both backends on the target hardware, and check CPU operation without an
activated CANN environment. Archive the source revision, build image and CANN
components, driver/firmware, host ABI and wheel checksums. See the
[state-vector tests](mqvector_npu.md) and [QAIA tests](qaia_ascend.md) for the
backend-specific numerical and performance checks.

`tests/build` validates the source build and packaging tools, not the installed
Ascend backend. Run it in the development environment prepared by `build.sh`
or `build_locally.sh`: it needs the `[build-system].requires` dependencies from
`pyproject.toml` (including `wheel` and `wheel-filename`), pytest, CMake and a host
compiler; Linux wheel-repair tests also need auditwheel and patchelf. Installing
a MindQuantum wheel, even with the `test` extra, does not install that build
toolchain. For installed-wheel hardware acceptance, run the backend-specific
tests above rather than treating packaging-test dependency failures as device
failures.
