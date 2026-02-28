#!/bin/bash
# ============================================
# 单个文件编译脚本 - BSB/DSB
# 用法: ./compile_single.sh <algorithm_name>
# 示例: ./compile_single.sh BSB_CPU_fp16
# ============================================

if [ $# -eq 0 ]; then
    echo "错误: 请指定要编译的算法名称"
    echo
    echo "用法: $0 <algorithm_name>"
    echo
    echo "支持的算法:"
    echo "  BSB:"
    echo "    - BSB_CPU_fp16"
    echo "    - BSB_CPU_int8"
    echo "    - BSB_CPU_fp8"
    echo "    - BSB_GPU_fp16 (需要CUDA)"
    echo "    - BSB_GPU_int8 (需要CUDA)"
    echo "    - BSB_GPU_fp8 (需要CUDA)"
    echo
    echo "  DSB:"
    echo "    - DSB_CPU_fp16"
    echo "    - DSB_CPU_int8"
    echo "    - DSB_CPU_fp8"
    echo "    - DSB_GPU_fp16 (需要CUDA)"
    echo "    - DSB_GPU_int8 (需要CUDA)"
    echo "    - DSB_GPU_fp8 (需要CUDA)"
    exit 1
fi

ALGORITHM=$1
INCLUDE_DIR=".."

# 检测算法类型
if [[ $ALGORITHM == BSB* ]]; then
    DIR="BSB"
    SOURCE="${DIR}/${ALGORITHM}.cu"
    OUTPUT="${DIR}/${ALGORITHM}"
elif [[ $ALGORITHM == DSB* ]]; then
    DIR="DSB"
    SOURCE="${DIR}/${ALGORITHM}.cu"
    OUTPUT="${DIR}/${ALGORITHM}"
else
    echo "错误: 未知的算法名称 '$ALGORITHM'"
    echo "算法名称必须以 BSB 或 DSB 开头"
    exit 1
fi

# 检查源文件是否存在
if [ ! -f "$SOURCE" ]; then
    echo "错误: 源文件不存在: $SOURCE"
    exit 1
fi

echo "============================================"
echo "编译 $ALGORITHM"
echo "============================================"
echo "源文件: $SOURCE"
echo "输出: $OUTPUT"
echo

# 根据算法类型选择编译器
if [[ $ALGORITHM == *_GPU_* ]]; then
    # GPU版本 - 使用nvcc
    if ! command -v nvcc &> /dev/null; then
        echo "错误: nvcc 未安装！GPU版本需要CUDA Toolkit"
        exit 1
    fi

    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "检测到CUDA版本: $CUDA_VERSION"
    echo

    # 设置GPU编译选项
    GPU_OPTS="-O3 -std=c++17 -arch=sm_70"

    echo "编译命令: nvcc $GPU_OPTS -I$INCLUDE_DIR $SOURCE -o $OUTPUT"
    echo
    nvcc $GPU_OPTS -I$INCLUDE_DIR $SOURCE -o $OUTPUT

else
    # CPU版本 - 使用g++
    if ! command -v g++ &> /dev/null; then
        echo "错误: g++ 未安装！"
        echo "请安装: sudo apt-get install g++ (Ubuntu/Debian)"
        exit 1
    fi

    # 设置CPU编译选项
    CPU_OPTS="-O3 -std=c++17 -march=native -fopenmp"

    echo "编译命令: g++ $CPU_OPTS -I$INCLUDE_DIR $SOURCE -o $OUTPUT"
    echo
    g++ $CPU_OPTS -I$INCLUDE_DIR $SOURCE -o $OUTPUT
fi

# 检查编译结果
if [ $? -eq 0 ]; then
    echo
    echo "============================================"
    echo "✓ 编译成功！"
    echo "============================================"
    echo
    echo "运行命令:"
    echo "  $OUTPUT qplib/QPLIB_3565.qplib"
    echo
else
    echo
    echo "============================================"
    echo "✗ 编译失败！"
    echo "============================================"
    exit 1
fi
