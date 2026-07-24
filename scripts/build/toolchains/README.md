# Managed release toolchains

MindQuantum selects a compiler before CMake's first configure step. A compiler
is build infrastructure, not a project dependency, so provisioning belongs at
the shell build-entry boundary.

Linux AArch64 MindSpore release builds use the GCC 10.4.0 environment pinned in
`gcc-10.4.0-linux-aarch64.lock`. It is a single-platform explicit conda lock:
there is no dependency resolution during a build, and every package URL carries
its SHA-256 digest. All artifacts come from the Tsinghua TUNA conda-forge
mirror, and the environment includes a glibc 2.17 sysroot.

`managed_gnu_toolchain.sh` bootstraps a pinned, authenticated micromamba binary.
Micromamba installs the explicit lock without solving dependencies and verifies
the locked packages while applying prefix relocation. Compiler entry points
select GCC's packaged `builtin.specs`, leaving runtime-path policy to CMake
instead of embedding the temporary conda prefix in every linked artifact.
Finally, the bootstrap compiles and links an SVE/FCMA probe and rejects any
leaked toolchain path before exporting `CC`, `CXX`, `AR`, and `RANLIB`. A
failed download, integrity check, platform check, version check, or capability
probe stops the build.

`build.sh` creates the environment in a unique temporary directory and removes
it when the build exits. This lifecycle matches the ephemeral MindSpore worker:
the repository contains no toolchain cache or partially reusable installation.

To update this toolchain:

1. Resolve all compiler, binutils, sysroot, and runtime packages for
   `linux-aarch64`, pinning versions and build strings.
2. Regenerate the explicit lock with SHA-256 hashes; never replace it with an
   environment YAML or unconstrained package specifications.
3. Update the bootstrap constants and compiler assertions together when their
   versions change.
4. Run the bootstrap on Linux AArch64 and verify the SVE/FCMA probe and a wheel
   build before submitting the lock change.
