#!/bin/bash
set -e

CURRENT_DIR=$(cd $(dirname $0); pwd)
SOC_VERSION="Ascend910B1"
# ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
ASCEND_INSTALL_PATH=/home/ma-user/anaconda3/envs/npu/Ascend/ascend-toolkit/latest/

if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
    ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
fi

export ASCEND_HOME_PATH=${ASCEND_INSTALL_PATH}
export ASCEND_TOOLKIT_HOME=${ASCEND_INSTALL_PATH}
export LD_LIBRARY_PATH=${ASCEND_INSTALL_PATH}/lib64:${CURRENT_DIR}/out/lib64:${CURRENT_DIR}/out/lib:$LD_LIBRARY_PATH
export PYTHONPATH=${CURRENT_DIR}:$PYTHONPATH
PYTHON_BIN=python3
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN=python
fi


LD_WRAPPER_DIR="${CURRENT_DIR}/.toolchain"
mkdir -p "${LD_WRAPPER_DIR}"
if [ -x "${ASCEND_INSTALL_PATH}/compiler/ccec_compiler/bin/ld.lld" ]; then
    ln -sf "${ASCEND_INSTALL_PATH}/compiler/ccec_compiler/bin/ld.lld" "${LD_WRAPPER_DIR}/ld"
    export PATH="${LD_WRAPPER_DIR}:$PATH"
fi

echo "Using SOC_VERSION=${SOC_VERSION}"
echo "Using ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH}"

cd ${CURRENT_DIR}

# 1. Build Kernel
echo ">>> Building Kernel..."
rm -rf kernel/build
mkdir -p kernel/build
(cd kernel/build && cmake .. \
    -DSOC_VERSION=${SOC_VERSION} \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_INSTALL_PATH} \
    -DCMAKE_INSTALL_PREFIX=${CURRENT_DIR}/out \
    -DLIBRARY_TYPE=SHARED)

cmake --build kernel/build -j
cmake --install kernel/build
if [ -f "${CURRENT_DIR}/out/lib64/libkernels.so" ]; then
    cp -f "${CURRENT_DIR}/out/lib64/libkernels.so" "${CURRENT_DIR}/out/lib/libkernels.so"
fi

# 2. Build Host
echo ">>> Building Host..."
rm -rf host/build
mkdir -p host/build
(cd host/build && env -u CONDA_PREFIX -u Python_EXECUTABLE -u Python_ROOT_DIR cmake .. \
    -DASCEND_CANN_PACKAGE_PATH=${ASCEND_INSTALL_PATH} \
    -DPython_EXECUTABLE=$(command -v ${PYTHON_BIN}))

cmake --build host/build -j

# Copy generated .so to root for easy import
find host/build -name "kernel_sb*.so" -exec cp {} . \;

# 3. Run C++ Main
echo ">>> Running C++ Main..."
./host/build/kernel_sb_main

# 4. Run Python Test
echo ">>> Running Python Test..."
${PYTHON_BIN} test.py
