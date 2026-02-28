#!/bin/bash
# ============================================
# Linux GPU编译脚本 - 编译所有BSB和DSB实现
# 适用于有GPU的Linux系统
# ============================================

echo "============================================"
echo "BSB/DSB Linux GPU编译脚本"
echo "============================================"
echo

# 检查nvcc是否安装
if ! command -v nvcc &> /dev/null; then
    echo "错误: nvcc 未安装！"
    echo "请安装CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# 显示CUDA版本
CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
echo "检测到CUDA版本: $CUDA_VERSION"
echo

# 设置编译选项
CPU_OPTS="-O3 -std=c++17 -march=native"
GPU_OPTS="-O3 -std=c++17 -arch=sm_70"
INCLUDE_DIR=".."

echo "编译选项 (CPU): $CPU_OPTS"
echo "编译选项 (GPU): $GPU_OPTS"
echo "包含目录: $INCLUDE_DIR"
echo

# BSB编译
echo "============================================"
echo "编译 BSB 算法..."
echo "============================================"
echo

echo "[1/6] 编译 BSB_CPU_fp16.cu ..."
nvcc $CPU_OPTS -I$INCLUDE_DIR BSB/BSB_CPU_fp16.cu -o BSB/BSB_CPU_fp16
if [ $? -eq 0 ]; then
    echo "  [成功] BSB_CPU_fp16 编译完成"
else
    echo "  [失败] BSB_CPU_fp16 编译失败"
fi

echo "[2/6] 编译 BSB_CPU_int8.cu ..."
nvcc $CPU_OPTS -I$INCLUDE_DIR BSB/BSB_CPU_int8.cu -o BSB/BSB_CPU_int8
if [ $? -eq 0 ]; then
    echo "  [成功] BSB_CPU_int8 编译完成"
else
    echo "  [失败] BSB_CPU_int8 编译失败"
fi

echo "[3/6] 编译 BSB_CPU_fp8.cu ..."
nvcc $CPU_OPTS -I$INCLUDE_DIR BSB/BSB_CPU_fp8.cu -o BSB/BSB_CPU_fp8
if [ $? -eq 0 ]; then
    echo "  [成功] BSB_CPU_fp8 编译完成"
else
    echo "  [失败] BSB_CPU_fp8 编译失败"
fi

echo "[4/6] 编译 BSB_GPU_fp16.cu ..."
nvcc $GPU_OPTS -I$INCLUDE_DIR BSB/BSB_GPU_fp16.cu -o BSB/BSB_GPU_fp16
if [ $? -eq 0 ]; then
    echo "  [成功] BSB_GPU_fp16 编译完成"
else
    echo "  [失败] BSB_GPU_fp16 编译失败"
    echo "  提示: 如果GPU架构不匹配，请修改 GPU_OPTS 中的 -arch 参数"
fi

echo "[5/6] 编译 BSB_GPU_int8.cu ..."
nvcc $GPU_OPTS -I$INCLUDE_DIR BSB/BSB_GPU_int8.cu -o BSB/BSB_GPU_int8
if [ $? -eq 0 ]; then
    echo "  [成功] BSB_GPU_int8 编译完成"
else
    echo "  [失败] BSB_GPU_int8 编译失败"
    echo "  提示: 如果GPU架构不匹配，请修改 GPU_OPTS 中的 -arch 参数"
fi

echo "[6/6] 编译 BSB_GPU_fp8.cu ..."
nvcc $GPU_OPTS -I$INCLUDE_DIR BSB/BSB_GPU_fp8.cu -o BSB/BSB_GPU_fp8
if [ $? -eq 0 ]; then
    echo "  [成功] BSB_GPU_fp8 编译完成"
else
    echo "  [失败] BSB_GPU_fp8 编译失败"
    echo "  提示: 如果GPU架构不匹配，请修改 GPU_OPTS 中的 -arch 参数"
fi

# DSB编译
echo
echo "============================================"
echo "编译 DSB 算法..."
echo "============================================"
echo

echo "[7/12] 编译 DSB_CPU_fp16.cu ..."
nvcc $CPU_OPTS -I$INCLUDE_DIR DSB/DSB_CPU_fp16.cu -o DSB/DSB_CPU_fp16
if [ $? -eq 0 ]; then
    echo "  [成功] DSB_CPU_fp16 编译完成"
else
    echo "  [失败] DSB_CPU_fp16 编译失败"
fi

echo "[8/12] 编译 DSB_CPU_int8.cu ..."
nvcc $CPU_OPTS -I$INCLUDE_DIR DSB/DSB_CPU_int8.cu -o DSB/DSB_CPU_int8
if [ $? -eq 0 ]; then
    echo "  [成功] DSB_CPU_int8 编译完成"
else
    echo "  [失败] DSB_CPU_int8 编译失败"
fi

echo "[9/12] 编译 DSB_CPU_fp8.cu ..."
nvcc $CPU_OPTS -I$INCLUDE_DIR DSB/DSB_CPU_fp8.cu -o DSB/DSB_CPU_fp8
if [ $? -eq 0 ]; then
    echo "  [成功] DSB_CPU_fp8 编译完成"
else
    echo "  [失败] DSB_CPU_fp8 编译失败"
fi

echo "[10/12] 编译 DSB_GPU_fp16.cu ..."
nvcc $GPU_OPTS -I$INCLUDE_DIR DSB/DSB_GPU_fp16.cu -o DSB/DSB_GPU_fp16
if [ $? -eq 0 ]; then
    echo "  [成功] DSB_GPU_fp16 编译完成"
else
    echo "  [失败] DSB_GPU_fp16 编译失败"
    echo "  提示: 如果GPU架构不匹配，请修改 GPU_OPTS 中的 -arch 参数"
fi

echo "[11/12] 编译 DSB_GPU_int8.cu ..."
nvcc $GPU_OPTS -I$INCLUDE_DIR DSB/DSB_GPU_int8.cu -o DSB/DSB_GPU_int8
if [ $? -eq 0 ]; then
    echo "  [成功] DSB_GPU_int8 编译完成"
else
    echo "  [失败] DSB_GPU_int8 编译失败"
    echo "  提示: 如果GPU架构不匹配，请修改 GPU_OPTS 中的 -arch 参数"
fi

echo "[12/12] 编译 DSB_GPU_fp8.cu ..."
nvcc $GPU_OPTS -I$INCLUDE_DIR DSB/DSB_GPU_fp8.cu -o DSB/DSB_GPU_fp8
if [ $? -eq 0 ]; then
    echo "  [成功] DSB_GPU_fp8 编译完成"
else
    echo "  [失败] DSB_GPU_fp8 编译失败"
    echo "  提示: 如果GPU架构不匹配，请修改 GPU_OPTS 中的 -arch 参数"
fi

echo
echo "============================================"
echo "所有编译完成！"
echo "============================================"
echo
echo "使用方法："
echo "  BSB/BSB_CPU_fp16 qplib/QPLIB_3565.qplib"
echo "  BSB/BSB_GPU_fp8 qplib/QPLIB_3565.qplib"
echo "  DSB/DSB_CPU_fp16 qplib/QPLIB_3565.qplib"
echo "  DSB/DSB_GPU_fp8 qplib/QPLIB_3565.qplib"
echo
echo "GPU架构提示："
echo "  - 当前使用 -arch=sm_70 (Volta架构)"
echo "  - 其他选项：sm_50(Maxwell), sm_60(Pascal), sm_75(Turing), sm_80(Ampere), sm_86(Ampere RTX 30xx)"
echo "  - 查询GPU架构: nvidia-smi --query-gpu=compute_cap --format=csv"
echo
