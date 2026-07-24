#!/bin/bash
# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

[ "${_sourced_managed_gnu_toolchain}" != "" ] && return || _sourced_managed_gnu_toolchain=.

# The compiler must be selected before CMake configures the project. This module
# therefore provisions and validates the release toolchain at the build-entry
# boundary instead of treating a compiler as a CMake dependency.

function mq_toolchain_require_command() {
    command -v "$1" > /dev/null 2>&1 || die "The managed GNU toolchain requires '$1'"
}

function mq_toolchain_sha256() {
    local checksum_output
    checksum_output=$(sha256sum "$1") || return 1
    printf '%s\n' "${checksum_output%% *}"
}

function mq_toolchain_verify_sha256() {
    local file_path expected_hash actual_hash
    file_path=$1
    expected_hash=$2
    actual_hash=$(mq_toolchain_sha256 "$file_path") || return 1
    [[ "$actual_hash" == "$expected_hash" ]]
}

function mq_toolchain_download() {
    local archive_url archive_hash archive_path
    archive_url=$1
    archive_hash=$2
    archive_path=$3

    echo "Downloading ${archive_url##*/}"
    curl --proto '=https' --tlsv1.2 --fail --location --silent --show-error --retry 3 \
        --output "$archive_path" "$archive_url" \
        || die "Unable to download authenticated toolchain archive: $archive_url"
    mq_toolchain_verify_sha256 "$archive_path" "$archive_hash" \
        || die "Toolchain archive failed its SHA-256 check: $archive_url"
}

function mq_toolchain_write_compiler_wrapper() {
    local wrapper compiler specs
    wrapper=$1
    compiler=$2
    specs=$3

    printf '#!/bin/bash\nexec %q -specs=%q "$@"\n' "$compiler" "$specs" > "$wrapper" || return 1
    chmod 0755 "$wrapper"
}

function mq_toolchain_validate_gcc_10_4() {
    local toolchain_prefix cxx probe_source probe_dir readelf compiler_version compiler_target runtime_path dynamic_section
    toolchain_prefix=$1
    cxx=$2
    probe_source=$3
    probe_dir=$4
    readelf="$toolchain_prefix/bin/aarch64-conda-linux-gnu-readelf"
    runtime_path="$toolchain_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    compiler_version=$(PATH="$toolchain_prefix/bin:$PATH" LD_LIBRARY_PATH="$runtime_path" "$cxx" -dumpfullversion) \
        || die "Unable to read the managed GCC version"
    compiler_target=$(PATH="$toolchain_prefix/bin:$PATH" LD_LIBRARY_PATH="$runtime_path" "$cxx" -dumpmachine) \
        || die "Unable to read the managed GCC target"
    [[ "$compiler_version" == "10.4.0" ]] \
        || die "Managed GCC version mismatch: expected 10.4.0, found $compiler_version"
    [[ "$compiler_target" == "aarch64-conda-linux-gnu" ]] \
        || die "Managed GCC target mismatch: expected aarch64-conda-linux-gnu, found $compiler_target"
    [[ -x "$readelf" ]] || die "Managed readelf not found: $readelf"

    mkdir -p "$probe_dir" || die "Unable to create the managed GCC validation directory"
    PATH="$toolchain_prefix/bin:$PATH" LD_LIBRARY_PATH="$runtime_path" \
        "$cxx" -std=c++17 -march=armv8-a+sve "$probe_source" -o "$probe_dir/sve_fcma" \
        || die "Managed GCC cannot compile and link the required SVE/FCMA intrinsics"
    dynamic_section=$("$readelf" -Wd "$probe_dir/sve_fcma") \
        || die "Unable to inspect the managed GCC validation artifact"
    [[ "$dynamic_section" != *"$toolchain_prefix"* ]] \
        || die "Managed GCC embedded its temporary prefix in the validation artifact"
}

function setup_managed_gnu_toolchain() {
    local workspace machine toolchain_dir lock_file probe_source toolchain_prefix
    local micromamba_version micromamba_archive_url micromamba_archive_hash micromamba_binary_hash
    local bootstrap_dir micromamba_archive micromamba mamba_root
    local builtin_specs wrapper_dir cc cxx wrapper

    [[ "${ROOTDIR:-}" != "" ]] || die "ROOTDIR must be defined before setting up the managed GNU toolchain"
    [[ "$#" -eq 1 ]] || die "The managed GNU toolchain requires one temporary workspace"
    workspace=$1
    [[ "$workspace" == /* && -d "$workspace" ]] \
        || die "Managed GNU toolchain workspace must be an existing absolute directory: $workspace"
    [[ "$(uname -s)" == "Linux" ]] || die "The managed GNU toolchain is only available on Linux"
    machine=$(uname -m)
    [[ "$machine" =~ ^(aarch64|arm64)$ ]] \
        || die "The managed GNU toolchain requires AArch64 (detected '$machine')"

    mq_toolchain_require_command bzip2
    mq_toolchain_require_command curl
    mq_toolchain_require_command sha256sum
    mq_toolchain_require_command tar

    toolchain_dir="$ROOTDIR/scripts/build/toolchains"
    lock_file="$toolchain_dir/gcc-10.4.0-linux-aarch64.lock"
    [[ -f "$lock_file" ]] || die "Managed GNU toolchain lock file not found: $lock_file"
    probe_source="$ROOTDIR/cmake/probes/arm_sve_fcma.cpp"
    [[ -f "$probe_source" ]] || die "Managed GNU toolchain probe not found: $probe_source"

    micromamba_version=2.8.1
    micromamba_archive_url=https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/linux-aarch64/micromamba-2.8.1-0.tar.bz2
    micromamba_archive_hash=70c60a36609ee8bcc07a3a1a66b2c3a65cff7c1053466963437f1bda72e5210f
    micromamba_binary_hash=e5ba23b5945aa49dfd11022e592a510d2686a8feee810e00140b73c9fdf0ba2a
    bootstrap_dir="$workspace/bootstrap"
    micromamba_archive="$bootstrap_dir/micromamba.tar.bz2"
    micromamba="$bootstrap_dir/bin/micromamba"
    mamba_root="$workspace/micromamba-root"
    toolchain_prefix="$workspace/gcc-10.4.0"

    mkdir -p "$bootstrap_dir" "$mamba_root" || die "Unable to initialize the managed toolchain workspace"
    mq_toolchain_download "$micromamba_archive_url" "$micromamba_archive_hash" "$micromamba_archive"
    tar -xjf "$micromamba_archive" -C "$bootstrap_dir" bin/micromamba \
        || die "Unable to extract the authenticated micromamba bootstrap"
    [[ -x "$micromamba" ]] || die "Managed micromamba bootstrap is not executable: $micromamba"
    mq_toolchain_verify_sha256 "$micromamba" "$micromamba_binary_hash" \
        || die "Managed micromamba bootstrap failed its SHA-256 check"
    [[ "$("$micromamba" --version)" == "$micromamba_version" ]] \
        || die "Managed micromamba bootstrap has an unexpected version"

    echo "Installing managed GCC 10.4.0 toolchain from the explicit lock file"
    "$micromamba" create --yes --no-rc --extra-safety-checks \
        --root-prefix "$mamba_root" --prefix "$toolchain_prefix" --file "$lock_file" \
        || die "Unable to install the managed GCC 10.4.0 toolchain"

    # Conda's default specs add the environment prefix to every linked
    # artifact. Select the package's unmodified specs so runtime-path policy
    # stays in CMake and the temporary prefix cannot leak into release binaries.
    builtin_specs="$toolchain_prefix/lib/gcc/aarch64-conda-linux-gnu/10.4.0/builtin.specs"
    [[ -f "$builtin_specs" ]] || die "Managed GCC builtin specs not found: $builtin_specs"
    wrapper_dir="$workspace/bin"
    cc="$wrapper_dir/aarch64-conda-linux-gnu-gcc"
    cxx="$wrapper_dir/aarch64-conda-linux-gnu-g++"
    mkdir -p "$wrapper_dir" || die "Unable to create managed compiler entry points"
    for wrapper in "$cc" "$wrapper_dir/gcc" "$wrapper_dir/cc"; do
        mq_toolchain_write_compiler_wrapper \
            "$wrapper" "$toolchain_prefix/bin/aarch64-conda-linux-gnu-gcc" "$builtin_specs" \
            || die "Unable to create the managed C compiler entry point"
    done
    for wrapper in "$cxx" "$wrapper_dir/g++" "$wrapper_dir/c++"; do
        mq_toolchain_write_compiler_wrapper \
            "$wrapper" "$toolchain_prefix/bin/aarch64-conda-linux-gnu-g++" "$builtin_specs" \
            || die "Unable to create the managed C++ compiler entry point"
    done

    mq_toolchain_validate_gcc_10_4 "$toolchain_prefix" "$cxx" "$probe_source" "$workspace/probe"

    export PATH="$wrapper_dir:$toolchain_prefix/bin:$PATH"
    export LD_LIBRARY_PATH="$toolchain_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export CC="$cc"
    export CXX="$cxx"
    export AR="$toolchain_prefix/bin/aarch64-conda-linux-gnu-ar"
    export RANLIB="$toolchain_prefix/bin/aarch64-conda-linux-gnu-ranlib"

    echo "Using managed GNU toolchain: GCC $("$CXX" -dumpfullversion) ($toolchain_prefix)"
}
