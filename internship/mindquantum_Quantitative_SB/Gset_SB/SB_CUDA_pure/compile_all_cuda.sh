#!/bin/bash
# 编译所有CUDA可执行文件 (Linux/Mac)

echo "========================================"
echo "编译BSB和DSB的纯精度GPU实现 (SB_CUDA_pure)"
echo "========================================"
echo

echo "[1/6] 编译 BSB_GPU_purefp8_improved.cu..."
nvcc BSB_GPU_purefp8_improved.cu sb_cuda_gset_reader.cu -o BSB_GPU_purefp8_improved -O3 -arch=sm_75 --use_fast_math
if [ $? -ne 0 ]; then
    echo "错误: BSB_GPU_purefp8_improved.cu 编译失败"
    exit 1
fi
echo "成功: BSB_GPU_purefp8_improved"

echo "[2/6] 编译 BSB_GPU_purefp16_improved.cu..."
nvcc BSB_GPU_purefp16_improved.cu sb_cuda_gset_reader.cu -o BSB_GPU_purefp16_improved -O3 -arch=sm_75 --use_fast_math
if [ $? -ne 0 ]; then
    echo "错误: BSB_GPU_purefp16_improved.cu 编译失败"
    exit 1
fi
echo "成功: BSB_GPU_purefp16_improved"

echo "[3/6] 编译 BSB_GPU_pureint8_improved.cu..."
nvcc BSB_GPU_pureint8_improved.cu sb_cuda_gset_reader.cu -o BSB_GPU_pureint8_improved -O3 -arch=sm_75 --use_fast_math
if [ $? -ne 0 ]; then
    echo "错误: BSB_GPU_pureint8_improved.cu 编译失败"
    exit 1
fi
echo "成功: BSB_GPU_pureint8_improved"

echo "[4/6] 编译 DSB_GPU_purefp8_improved.cu..."
nvcc DSB_GPU_purefp8_improved.cu sb_cuda_gset_reader.cu -o DSB_GPU_purefp8_improved -O3 -arch=sm_75 --use_fast_math
if [ $? -ne 0 ]; then
    echo "错误: DSB_GPU_purefp8_improved.cu 编译失败"
    exit 1
fi
echo "成功: DSB_GPU_purefp8_improved"

echo "[5/6] 编译 DSB_GPU_purefp16_improved.cu..."
nvcc DSB_GPU_purefp16_improved.cu sb_cuda_gset_reader.cu -o DSB_GPU_purefp16_improved -O3 -arch=sm_75 --use_fast_math
if [ $? -ne 0 ]; then
    echo "错误: DSB_GPU_purefp16_improved.cu 编译失败"
    exit 1
fi
echo "成功: DSB_GPU_purefp16_improved"

echo "[6/6] 编译 DSB_GPU_pureint8_improved.cu..."
nvcc DSB_GPU_pureint8_improved.cu sb_cuda_gset_reader.cu -o DSB_GPU_pureint8_improved -O3 -arch=sm_75 --use_fast_math
if [ $? -ne 0 ]; then
    echo "错误: DSB_GPU_pureint8_improved.cu 编译失败"
    exit 1
fi
echo "成功: DSB_GPU_pureint8_improved"

echo
echo "========================================"
echo "所有CUDA可执行文件编译完成!"
echo "========================================"
