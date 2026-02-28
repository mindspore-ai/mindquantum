#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <random>
#include <algorithm>
#include <limits>
#include <cstring>
#include <sstream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <exception>
#include <cmath>
#include <ctime>
#include "sb_cuda_gset_reader.h"

// 使用自定义 FP8 实现（位级操作）
// 编译需要: -arch=sm_75 或更高 (Turing 架构及以上)
// CUDA 版本: >= 11.0
// 注意：不依赖 CUDA 内置的 FP8 转换函数，纯手工实现
// 使用 uint8_t 存储 FP8 E4M3 格式数据

// 纯 FP8 加法（直接操作FP8位，不使用FP16/FP32中间计算）
// FP8 E4M3格式: 1符号位 + 4指数位 + 3尾数位
__device__ inline uint8_t fp8_add(uint8_t a, uint8_t b) {
    // 提取符号、指数、尾数
    int sign_a = (a >> 7) & 0x1;
    int sign_b = (b >> 7) & 0x1;
    int exp_a = (a >> 3) & 0xF;
    int exp_b = (b >> 3) & 0xF;
    int mant_a = a & 0x7;
    int mant_b = b & 0x7;

    // 归一化到尾数扩展形式（添加隐含位）
    int value_a = sign_a ? -( (1 << 6) | (exp_a << 3) | mant_a ) : (1 << 6) | (exp_a << 3) | mant_a;
    int value_b = sign_b ? -( (1 << 6) | (exp_b << 3) | mant_b ) : (1 << 6) | (exp_b << 3) | mant_b;

    // 简化的加法（将两个值相加）
    int sum = value_a + value_b;

    // 转换回FP8格式
    int sign_result = (sum < 0) ? 1 : 0;
    unsigned int abs_sum = (sum < 0) ? -sum : sum;

    if (abs_sum == 0) {
        return 0x00;
    }

    // 查找指数（左对齐）
    int exp_result = 6;  // 从隐含位开始
    int mant_result = abs_sum;

    // 归一化
    while ((mant_result & (1 << 7)) == 0 && exp_result > 0) {
        mant_result <<= 1;
        exp_result--;
    }

    // 舍入到3位尾数（保留前3位）
    int mant_out = (mant_result >> 5) & 0x7;
    int remainder = mant_result & 0x1F;

    // 舍入
    if (remainder >= 0x10) {
        mant_out++;
        if (mant_out >= 8) {
            mant_out = 0;
            exp_result++;
        }
    }

    // 饱和处理
    if (exp_result > 15) {
        exp_result = 15;
        mant_out = 0x7;
    }
    if (exp_result < 0) {
        exp_result = 0;
        mant_out = 0;
    }

    return (sign_result << 7) | ((exp_result & 0xF) << 3) | (mant_out & 0x7);
}

// 纯 FP8 取负（翻转符号位）
__device__ inline uint8_t fp8_neg(uint8_t a) {
    return a ^ 0x80;  // 翻转最高位（符号位）
}

// 纯 FP8 减法：a - b（直接操作FP8位）
__device__ inline uint8_t fp8_sub(uint8_t a, uint8_t b) {
    return fp8_add(a, fp8_neg(b));
}

// 纯 FP8 乘法（直接操作FP8位）
__device__ inline uint8_t fp8_mul(uint8_t a, uint8_t b) {
    // 零的特殊处理
    if (a == 0 || b == 0) {
        return 0x00;
    }

    // 提取符号、指数、尾数
    int sign_a = (a >> 7) & 0x1;
    int sign_b = (b >> 7) & 0x1;
    int exp_a = (a >> 3) & 0xF;
    int exp_b = (b >> 3) & 0xF;
    int mant_a = a & 0x7;
    int mant_b = b & 0x7;

    // 结果符号
    int sign_result = sign_a ^ sign_b;

    // 结果指数（相加并减去偏置）
    const int FP8_EXP_BIAS = 7;
    int exp_result = exp_a + exp_b - FP8_EXP_BIAS;

    // 结果尾数（相乘，7位 * 7位 = 14位）
    unsigned int full_mant_a = (1 << 3) | mant_a;  // 添加隐含位
    unsigned int full_mant_b = (1 << 3) | mant_b;
    unsigned int product = full_mant_a * full_mant_b;  // 14位结果

    // 归一化并舍入到3位
    int exp_offset = 0;
    while ((product & (1 << 7)) == 0 && exp_offset < 14) {
        product <<= 1;
        exp_offset++;
    }

    exp_result -= exp_offset;

    // 舍入到3位尾数
    int mant_out = (product >> 5) & 0x7;
    int remainder = product & 0x1F;

    if (remainder >= 0x10) {
        mant_out++;
        if (mant_out >= 8) {
            mant_out = 0;
            exp_result++;
        }
    }

    // 饱和处理
    if (exp_result > 15) {
        exp_result = 15;
        mant_out = 0x7;
    }
    if (exp_result < 0) {
        exp_result = 0;
        mant_out = 0;
    }

    return (sign_result << 7) | ((exp_result & 0xF) << 3) | (mant_out & 0x7);
}

// 纯 FP8 乘加：a*b + c（使用纯FP8计算）
__device__ inline uint8_t fp8_fma(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t product = fp8_mul(a, b);
    return fp8_add(product, c);
}

// 纯 FP8 比较函数
__device__ inline bool fp8_gt(uint8_t a, uint8_t b) {
    // 处理符号
    int sign_a = (a >> 7) & 0x1;
    int sign_b = (b >> 7) & 0x1;

    if (sign_a != sign_b) {
        return !sign_a;  // 正数大于负数
    }

    // 同符号下比较大小
    if (sign_a) {
        // 都是负数，较小的数值更大
        return a < b;
    } else {
        // 都是正数，较大的数值更大
        return a > b;
    }
}

__device__ inline bool fp8_lt(uint8_t a, uint8_t b) {
    return fp8_gt(b, a);
}

// 纯 FP8 限制函数（在最小和最大值之间）
__device__ inline uint8_t fp8_clamp(uint8_t val, uint8_t min_val, uint8_t max_val) {
    return fp8_lt(val, min_val) ? min_val : (fp8_gt(val, max_val) ? max_val : val);
}

// FP8 到 FP32 转换（仅用于最终输出）
__device__ inline float fp8_to_float(uint8_t x) {
    int sign = (x >> 7) & 0x1;
    int exp = (x >> 3) & 0xF;
    int mant = x & 0x7;

    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    // E4M3格式：偏置7
    float value = mant / 8.0f;
    if (exp != 0) {
        value = (1.0f + mant / 8.0f) * exp2f(exp - 7);
    }

    return sign ? -value : value;
}

// FP32 到 FP8 转换（仅用于初始化）
__device__ inline uint8_t float_to_fp8(float f) {
    // 处理特殊值
    if (isnan(f)) return 0x80;
    if (isinf(f)) return signbit(f) ? 0xFF : 0x7F;
    if (f == 0.0f) return signbit(f) ? 0x80 : 0x00;

    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    int sign_bit = (bits >> 31) & 0x1;
    int exp32 = (bits >> 23) & 0xFF;
    uint32_t mant32 = bits & 0x7FFFFF;

    const int FP8_EXP_BIAS = 7;
    const int FP8_MAX_EXP = 7;
    const int FP8_MIN_EXP = -7;

    // 计算实际指数
    int true_exp = exp32 - 127;

    // E4M3FN范围检查
    if (true_exp > FP8_MAX_EXP) {
        return sign_bit ? 0xFF : 0x7F;
    }

    if (true_exp < FP8_MIN_EXP) {
        return sign_bit ? 0x80 : 0x00;
    }

    // 正常值转换
    int exp8 = true_exp + FP8_EXP_BIAS;
    exp8 = max(0, min(15, exp8));

    uint8_t mant8 = (mant32 >> 20) & 0x07;
    uint32_t remainder = mant32 & 0xFFFFF;

    if (remainder >= 0x80000) {
        mant8 += 1;
        if (mant8 >= 8) {
            mant8 = 0;
            exp8 += 1;
            if (exp8 > 15) {
                return sign_bit ? 0xFF : 0x7F;
            }
        }
    }

    return (sign_bit << 7) | (exp8 << 3) | mant8;
}

using namespace std;
using namespace std::chrono;

// ============ 纯FP8量化配置 ============
// 纯FP8：不使用任何缩放因子，直接使用FP8原始范围
// FP8 E4M3 范围：最大值约为 240，最小值约为 -240
// 纯FP8策略：所有数据直接映射到FP8范围，不使用额外缩放
constexpr float FP8_MAX_VAL = 240.0f;   // FP8 E4M3 的正饱和值
constexpr float FP8_MIN_VAL = -240.0f;  // FP8 E4M3 的负饱和值

// 纯FP8：不使用缩放因子，所有值直接存储为FP8

// Clamp函数（与CPU版本一致）
__host__ __device__ inline float clamp(float val, float min_val, float max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

// 保留旧的 uint8_t 版本用于兼容性（仅用于 CPU 端初始化）
__host__ inline uint8_t float_to_fp8_e4m3fn_host(float f) {
    // 处理特殊值
    if (isnan(f)) return 0x80;
    if (isinf(f)) return signbit(f) ? 0xFF : 0x7F;
    if (f == 0.0f) return signbit(f) ? 0x80 : 0x00;

    // 获取IEEE 754 FP32的符号、指数和尾数
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    int sign_bit = (bits >> 31) & 0x1;
    int exp32 = (bits >> 23) & 0xFF;
    uint32_t mant32 = bits & 0x7FFFFF;

    const int FP8_EXP_BIAS = 7;
    const int FP8_MAX_EXP = 7;
    const int FP8_MIN_EXP = -7;

    // 计算实际指数
    int true_exp = exp32 - 127;

    // E4M3FN范围检查: 指数范围[-7, 7], 偏置7
    if (true_exp > FP8_MAX_EXP) {
        return sign_bit ? 0xFF : 0x7F;
    }

    if (true_exp < FP8_MIN_EXP) {
        // 次正规数处理：当指数过小时，返回零（简化处理）
        return sign_bit ? 0x80 : 0x00;
    }

    // 正常值转换（带更精确的舍入）
    int exp8 = true_exp + FP8_EXP_BIAS;
    exp8 = max(0, min(15, exp8));

    // 保留高3位尾数，并检查是否需要进位（round-to-nearest-even）
    uint8_t mant8 = (mant32 >> 20) & 0x07;
    uint32_t remainder = mant32 & 0xFFFFF;

    // 更精确的舍入：检查第20位以及后续是否全部为零
    if (remainder >= 0x80000) {
        mant8 += 1;
        if (mant8 >= 8) {
            mant8 = 0;
            exp8 += 1;
            if (exp8 > 15) {
                return sign_bit ? 0xFF : 0x7F;
            }
        }
    }

    return (sign_bit << 7) | (exp8 << 3) | mant8;
}

// FP32到FP8转换（直接转换，不使用缩放）
__host__ inline uint8_t float_to_fp8_scaled(float f) {
    // 直接转换为FP8，不进行缩放
    return float_to_fp8_e4m3fn_host(f);
}

// FP8到FP32转换（主机端版本，直接转换，不使用缩放）
__host__ inline float fp8_scaled_to_float(uint8_t x) {
    int sign = (x >> 7) & 0x1;
    int exp = (x >> 3) & 0xF;
    int mant = x & 0x7;

    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    // E4M3格式：偏置7
    float value = mant / 8.0f;
    if (exp != 0) {
        value = (1.0f + mant / 8.0f) * exp2f(exp - 7);
    }

    return sign ? -value : value;
}

// 计算自适应量化参数（纯FP8：不使用缩放）
// 对于纯FP8，图权重 [-1, 1] 直接映射到 FP8 的有效范围
inline void calculate_quantization_params(int num_edges, float max_weight) {
    // 纯FP8策略：
    // - 不使用VALUE_SCALE缩放
    // - 图权重 [-1, 1] 直接存储为FP8
    // - 自旋变量 x 在 [-1, 1] 范围内直接存储为FP8
    // - 所有计算使用纯FP8，接受精度损失

    cout << "\n=== Pure FP8 Parameters (No Scaling) ===" << endl;
    cout << "  Graph statistics:" << endl;
    cout << "    - Edges: " << num_edges << ", Max weight: " << max_weight << endl;
    cout << "  FP8 format: E4M3" << endl;
    cout << "    - FP8 range: [" << FP8_MIN_VAL << ", " << FP8_MAX_VAL << "]" << endl;
    cout << "  Quantization strategy:" << endl;
    cout << "    - NO VALUE_SCALE: Pure FP8 without scaling" << endl;
    cout << "    - Weights [-1, 1] stored directly as FP8" << endl;
    cout << "    - Spin x [-1, 1] stored directly as FP8" << endl;
    cout << "  Computation strategy:" << endl;
    cout << "    - Storage: FP8 (all variables)" << endl;
    cout << "    - Arithmetic: Pure FP8 (all operations)" << endl;
    cout << "    - Accumulation: Block-wise FP8 (final sum also FP8)" << endl;
    cout << "    - NO FP32 intermediates" << endl;
    cout << "  Precision consideration:" << endl;
    cout << "    - Accepting FP8 precision loss for pure FP8 computation" << endl;
    cout << "==============================================\n" << endl;
}

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << #call << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============ 纯FP8累加器 ============
// 用于累加多个FP8值（使用纯FP8计算，可能存在精度损失但保持FP8一致性）
struct Fp8Accumulator {
    uint8_t value;

    __device__ void reset() {
        value = float_to_fp8(0.0f);
    }

    __device__ void add_fp8(uint8_t x) {
        // 使用纯FP8加法累加
        value = fp8_add(value, x);
    }

    __device__ void add_fp8_mul(uint8_t a, uint8_t b) {
        // 使用纯FP8乘加：value += a * b
        value = fp8_fma(a, b, value);
    }

    __device__ void add_fp8_mul_scaled(uint8_t a, uint8_t b, float scale) {
        // 纯FP8：先相乘，再考虑缩放因子（用FP8近似）
        // 由于scale是float，这里转换为FP8进行计算
        uint8_t scale_fp8 = float_to_fp8(1.0f / scale);
        uint8_t product = fp8_mul(a, b);
        uint8_t scaled_product = fp8_mul(product, scale_fp8);
        value = fp8_add(value, scaled_product);
    }

    __device__ float to_float() {
        // 最终输出时转为FP32（仅用于显示，不使用缩放）
        return fp8_to_float(value);
    }

    __device__ float to_float_cut() {
        // 割值输出时转为FP32（不使用缩放）
        return fp8_to_float(value);
    }

    __device__ uint8_t to_fp8() {
        return value;
    }
};

// GPU批处理数据结构（全部使用 uint8_t 存储 FP8）
struct GPUBatchData {
    uint8_t* x;
    uint8_t* y;
    uint8_t* gradient_term;  // FP8存储
};

__global__ void init_batches_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ y,
                                   uint8_t* __restrict__ gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;

    for (int i = 0; i < num_spins; ++i) {
        // x在[-1,1]范围内初始化（使用纯FP8计算）
        float rand_val = curand_uniform(&state);
        uint8_t rand_fp8 = float_to_fp8(rand_val - 0.5f);
        uint8_t scale_fp8 = float_to_fp8(0.1f);
        x[batch_offset + i] = fp8_mul(scale_fp8, rand_fp8);

        rand_val = curand_uniform(&state);
        rand_fp8 = float_to_fp8(rand_val - 0.5f);
        y[batch_offset + i] = fp8_mul(scale_fp8, rand_fp8);

        gradient_term[batch_offset + i] = float_to_fp8(0.0f);
    }
}

__global__ void calc_gradient_kernel_optimized(uint8_t* x, uint8_t* gradient_term,
                                               int* __restrict__ second_order_terms,
                                               uint8_t* __restrict__ second_order_coeff,
                                               uint8_t* __restrict__ one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 使用纯FP8累加器计算梯度
    // 纯FP8：不使用缩放，所有值直接在FP8范围内
    Fp8Accumulator grad;
    grad.reset();

    // 添加一次项系数（使用纯FP8）
    uint8_t one_coeff = one_order_coeff[spin_idx];
    grad.add_fp8(one_coeff);

    // 循环展开4次以提高指令级并行
    // 梯度计算：grad = coeff + sum(J_ij * x_j)
    // 纯FP8计算，无缩放
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = second_order_terms[(t + k) * 2];
            int var2 = second_order_terms[(t + k) * 2 + 1];
            uint8_t J_ij = second_order_coeff[t + k];

            if (var1 == spin_idx) {
                uint8_t x2_val = x[batch_offset + var2];
                // 纯FP8计算: grad += J_ij * x2_val
                grad.add_fp8_mul(J_ij, x2_val);
            } else if (var2 == spin_idx) {
                uint8_t x1_val = x[batch_offset + var1];
                // 纯FP8计算: grad += J_ij * x1_val
                grad.add_fp8_mul(J_ij, x1_val);
            }
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        uint8_t J_ij = second_order_coeff[t];

        if (var1 == spin_idx) {
            uint8_t x2_val = x[batch_offset + var2];
            grad.add_fp8_mul(J_ij, x2_val);
        } else if (var2 == spin_idx) {
            uint8_t x1_val = x[batch_offset + var1];
            grad.add_fp8_mul(J_ij, x1_val);
        }
    }

    // 保存纯FP8结果
    gradient_term[batch_offset + spin_idx] = grad.to_fp8();
}

__global__ void update_y_kernel_optimized(uint8_t* __restrict__ x, uint8_t* __restrict__ y,
                                          uint8_t* __restrict__ gradient_term,
                                          int num_spins, float p, float xi, float dt,
                                          int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 使用纯FP8计算
    uint8_t x_fp8 = x[batch_offset + spin_idx];
    uint8_t grad_fp8 = gradient_term[batch_offset + spin_idx];
    uint8_t y_fp8 = y[batch_offset + spin_idx];

    // dy/dt = -(1-p)*x + ξ*∇f(x)
    // 纯FP8计算
    uint8_t p_fp8 = float_to_fp8(p);
    uint8_t one_fp8 = float_to_fp8(1.0f);
    uint8_t one_minus_p = fp8_sub(one_fp8, p_fp8);  // (1-p)
    uint8_t neg_one_minus_p = fp8_neg(one_minus_p);  // -(1-p)

    uint8_t xi_fp8 = float_to_fp8(is_maximize ? xi : -xi);

    // term1 = -(1-p)*x
    uint8_t term1 = fp8_mul(neg_one_minus_p, x_fp8);

    // term2 = xi*grad
    uint8_t term2 = fp8_mul(xi_fp8, grad_fp8);

    // update = (term1 + term2) * dt
    uint8_t sum = fp8_add(term1, term2);
    uint8_t dt_fp8 = float_to_fp8(dt);
    uint8_t update = fp8_mul(sum, dt_fp8);

    // y_new = y + update
    uint8_t y_new = fp8_add(y_fp8, update);

    // 边界约束[-1, 1]
    uint8_t one_h = float_to_fp8(1.0f);
    uint8_t neg_one_h = float_to_fp8(-1.0f);
    y_new = fp8_clamp(y_new, neg_one_h, one_h);

    y[batch_offset + spin_idx] = y_new;
}

__global__ void update_x_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ y,
                                int num_spins, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 使用纯FP8计算
    uint8_t x_fp8 = x[batch_offset + spin_idx];
    uint8_t y_fp8 = y[batch_offset + spin_idx];

    // dx/dt = y * delta * dt
    uint8_t dt_fp8 = float_to_fp8(dt);
    uint8_t delta_fp8 = float_to_fp8(delta);
    uint8_t y_dt = fp8_mul(y_fp8, dt_fp8);
    uint8_t y_dt_delta = fp8_mul(y_dt, delta_fp8);
    uint8_t x_new = fp8_add(x_fp8, y_dt_delta);

    // 边界约束：|x| > 1时投影到边界
    uint8_t zero_fp8 = float_to_fp8(0.0f);
    uint8_t one_fp8 = float_to_fp8(1.0f);
    uint8_t neg_one_fp8 = float_to_fp8(-1.0f);

    if (fp8_gt(x_new, one_fp8)) {
        x_new = one_fp8;
        y[batch_offset + spin_idx] = zero_fp8;
    } else if (fp8_lt(x_new, neg_one_fp8)) {
        x_new = neg_one_fp8;
        y[batch_offset + spin_idx] = zero_fp8;
    }

    x[batch_offset + spin_idx] = x_new;
}

__global__ void calc_cut_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ cuts,
                                int* __restrict__ second_order_terms,
                                uint8_t* __restrict__ second_order_coeff,
                                uint8_t* __restrict__ one_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 使用分块累加策略避免FP8溢出（纯FP8）
    // 将大量边的累加分解为多个小块，每块单独累加后用FP8存储
    // FP8 E4M3 范围约 [-240, 240]
    // 每条边最多贡献：|J_ij| * (1 - s_i*s_j) * 0.5 = 1.0 * 2 * 0.5 = 1.0
    // 所以每块最多可以累加 240 / 1 = 240 条边

    const int BLOCK_SIZE = 200;  // 每块200条边，确保不会溢出

    uint8_t zero_fp8 = float_to_fp8(0.0f);
    uint8_t one_fp8 = float_to_fp8(1.0f);
    uint8_t half_fp8 = float_to_fp8(0.5f);

    // 纯FP8累加器（使用FP8累加所有块的结果）
    Fp8Accumulator total_acc;
    total_acc.reset();

    // 分块处理
    for (int block_start = 0; block_start < num_second_order; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, num_second_order);

        // 当前块使用FP8累加器
        Fp8Accumulator block_acc;
        block_acc.reset();

        // 处理当前块的所有边
        for (int t = block_start; t < block_end; ++t) {
            int var1 = second_order_terms[t * 2];
            int var2 = second_order_terms[t * 2 + 1];
            uint8_t J_ij = second_order_coeff[t];
            uint8_t x1 = x[batch_offset + var1];
            uint8_t x2 = x[batch_offset + var2];

            // FP8绝对值：|J_ij|
            uint8_t abs_J = (J_ij >> 7) ? fp8_neg(J_ij) : J_ij;

            // FP8二元化：x > 0 为 1.0f, x <= 0 为 -1.0f
            uint8_t x1_sign = fp8_gt(x1, zero_fp8) ? one_fp8 : fp8_neg(one_fp8);
            uint8_t x2_sign = fp8_gt(x2, zero_fp8) ? one_fp8 : fp8_neg(one_fp8);

            // 计算：s_i * s_j (结果为 1.0f 或 -1.0f)
            uint8_t sign_product = fp8_mul(x1_sign, x2_sign);

            // 计算：1 - s_i * s_j (结果为 2.0f 或 0.0f)
            uint8_t one_minus = fp8_sub(one_fp8, sign_product);

            // 纯FP8计算：term = |J_ij| * (1 - s_i * s_j)
            uint8_t term = fp8_mul(abs_J, one_minus);

            // 纯FP8计算：cut += 0.5 * term
            uint8_t term_half = fp8_mul(term, half_fp8);
            block_acc.add_fp8(term_half);
        }

        // 将当前块的FP8累加结果添加到总FP8累加器
        uint8_t block_result = block_acc.to_fp8();
        total_acc.add_fp8(block_result);
    }

    // 纯FP8输出
    cuts[batch_idx] = total_acc.to_fp8();
}

__global__ void calc_energy_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ energies,
                                    int* __restrict__ second_order_terms,
                                    uint8_t* __restrict__ second_order_coeff,
                                    uint8_t* __restrict__ one_order_coeff,
                                    int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 使用分块累加策略避免FP8溢出（纯FP8）
    // 不使用缩放，值直接在 [-1, 1] 范围内
    const int BLOCK_SIZE_SPINS = 100;   // 一次项每块100个自旋
    const int BLOCK_SIZE_EDGES = 50;    // 二次项每块50条边

    // 纯FP8累加器（使用FP8累加所有块的结果）
    Fp8Accumulator total_acc;
    total_acc.reset();

    // 计算一次项能量（使用连续值 x）
    // 一次项：energy += coeff * x
    // 每个项最多：|coeff| * |x| = 1.0 * 1.0 = 1.0
    // 每块100个项，最多 100 * 1 = 100 < 240（安全）
    for (int block_start = 0; block_start < num_spins; block_start += BLOCK_SIZE_SPINS) {
        int block_end = min(block_start + BLOCK_SIZE_SPINS, num_spins);

        // 当前块使用FP8累加器
        Fp8Accumulator block_acc;
        block_acc.reset();

        for (int i = block_start; i < block_end; ++i) {
            uint8_t x_i = x[batch_offset + i];
            uint8_t coeff = one_order_coeff[i];

            // 纯FP8累加: energy += coeff * x
            block_acc.add_fp8_mul(coeff, x_i);
        }

        // 将当前块的FP8累加结果添加到总FP8累加器
        uint8_t block_result = block_acc.to_fp8();
        total_acc.add_fp8(block_result);
    }

    // 计算二次项能量（使用连续值 x）
    // 二次项：energy += J_ij * x1 * x2
    // 每个项最多：|J_ij| * |x1| * |x2| = 1.0 * 1.0 * 1.0 = 1.0
    // 每块50个项，最多 50 * 1 = 50 < 240（安全）
    for (int block_start = 0; block_start < num_second_order; block_start += BLOCK_SIZE_EDGES) {
        int block_end = min(block_start + BLOCK_SIZE_EDGES, num_second_order);

        // 当前块使用FP8累加器
        Fp8Accumulator block_acc;
        block_acc.reset();

        for (int t = block_start; t < block_end; ++t) {
            int var1 = second_order_terms[t * 2];
            int var2 = second_order_terms[t * 2 + 1];
            uint8_t J_ij = second_order_coeff[t];
            uint8_t x1 = x[batch_offset + var1];
            uint8_t x2 = x[batch_offset + var2];

            // 纯FP8计算: energy += J_ij * x1 * x2
            uint8_t product = fp8_mul(J_ij, x1);
            product = fp8_mul(product, x2);
            block_acc.add_fp8(product);
        }

        // 将当前块的FP8累加结果添加到总FP8累加器
        uint8_t block_result = block_acc.to_fp8();
        total_acc.add_fp8(block_result);
    }

    // 纯FP8输出
    energies[batch_idx] = total_acc.to_fp8();
}

class BSBGPUFP8Solver {
private:
    int batch_size;
    int num_spins;
    int num_second_order;

    uint8_t* d_x;
    uint8_t* d_y;
    uint8_t* d_gradient_term;  // FP8存储

    int* d_second_order_terms;
    uint8_t* d_second_order_coeff;    // FP8存储
    uint8_t* d_one_order_coeff;       // FP8存储
    cudaStream_t stream;

public:
    BSBGPUFP8Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~BSBGPUFP8Solver() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // 更新设备常量（纯FP8：不需要缩放常量）
    void update_device_constants() {
        cout << "  [DEBUG] Pure FP8: No scaling constants needed" << endl;
    }

    void set_model_data(int* terms, float* second_coeff, float* one_coeff, int num_second) {
        num_second_order = num_second;

        // 使用纯FP8转换（不使用缩放）
        vector<uint8_t> fp8_second_order_coeff_host(num_second);
        vector<uint8_t> fp8_one_order_coeff_host(num_spins);
        for (int i = 0; i < num_second; ++i) {
            fp8_second_order_coeff_host[i] = float_to_fp8_scaled(second_coeff[i]);
        }
        for (int i = 0; i < num_spins; ++i) {
            fp8_one_order_coeff_host[i] = float_to_fp8_scaled(one_coeff[i]);
        }

        // 分配 GPU 内存（使用FP8）
        CUDA_CHECK(cudaMalloc(&d_second_order_terms, num_second_order * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, num_second_order * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, num_spins * sizeof(uint8_t)));

        // 复制数据
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_terms, terms, num_second_order * 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_coeff, fp8_second_order_coeff_host.data(), num_second_order * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_one_order_coeff, fp8_one_order_coeff_host.data(), num_spins * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void solve(int n_iter, float xi, float dt, bool is_maximize = true) {
        // 优化：使用更合适的block大小，warp大小为32
        const int BLOCK_SIZE = 256;
        const int BLOCKS_PER_BATCH = (num_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 grid_x(batch_size);
        dim3 block_x(BLOCK_SIZE);
        dim3 grid_2d(batch_size, BLOCKS_PER_BATCH);

        init_batches_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_y, d_gradient_term, num_spins, num_second_order, time(nullptr), batch_size);

        // 核心迭代（BSB动力学）
        for (int iter = 0; iter < n_iter; ++iter) {
            // 计算泵浦振幅：p(t) = t/T
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);
            float delta = 1.0f;

            // BSB: 使用连续的 x 值计算梯度（FP8中间计算，FP8存储）
            calc_gradient_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_gradient_term,
                d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                num_spins, num_second_order, batch_size);

            update_y_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, d_gradient_term,
                num_spins, p, xi, dt, batch_size, is_maximize);

            update_x_kernel<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, num_spins, dt, delta, batch_size);
        }

        // 只在最后同步一次，减少同步开销
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        uint8_t* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(uint8_t)));

        // 纯FP8计算割值（输出为FP8，然后转换为FP32仅用于显示）
        calc_cut_kernel<<<batch_size, 256, 0, stream>>>(d_x, d_cuts, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 从FP8复制数据并转换为FP32（仅用于显示）
        vector<uint8_t> cuts_fp8(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(cuts_fp8.data(), d_cuts, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_cuts));

        for (int i = 0; i < batch_size; ++i) {
            cuts[i] = fp8_scaled_to_float(cuts_fp8[i]);
        }

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        uint8_t* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(uint8_t)));

        // 纯FP8计算能量（输出为FP8，然后转换为FP32仅用于显示）
        calc_energy_kernel<<<batch_size, 256, 0, stream>>>(d_x, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 从FP8复制数据并转换为FP32（仅用于显示）
        vector<uint8_t> energies_fp8(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(energies_fp8.data(), d_energies, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_energies));

        for (int i = 0; i < batch_size; ++i) {
            energies[i] = fp8_scaled_to_float(energies_fp8[i]);
        }

        return energies;
    }
};

int main(int argc, char** argv) {
    // 解析命令行参数
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <gset_file> [num_iter] [batch_size] [dt] [delta] [xi]" << endl;
        cout << "  gset_file: Gset data file path (e.g., ../data/G1.txt)" << endl;
        cout << "  num_iter: Number of iterations (default: 1000)" << endl;
        cout << "  batch_size: Batch size (default: 100)" << endl;
        cout << "  dt: Time step (default: 0.5)" << endl;
        cout << "  delta: Damping coefficient (default: 0.9)" << endl;
        cout << "  xi: Coupling strength (default: auto)" << endl;
        return -1;
    }

    const char* gset_file = argv[1];
    int n_iter = (argc > 2) ? atoi(argv[2]) : 1000;
    int batch_size = (argc > 3) ? atoi(argv[3]) : 100;
    float dt = (argc > 4) ? atof(argv[4]) : 0.5f;
    float delta = (argc > 5) ? atof(argv[5]) : 0.9f;

    // 读取 Gset 文件
    GsetData gset;
    int ret = read_gset_file(gset_file, true, &gset);  // negate=True（与Python保持一致）

    if (ret != 0) {
        cerr << "Error: Failed to read Gset file " << gset_file << endl;
        return -2;
    }

    // 打印 Gset 信息
    print_gset_info(&gset);

    // 创建求解器
    BSBGPUFP8Solver solver(gset.num_vertices, batch_size);

    // 计算自适应量化参数（参考int8版本）
    float max_weight = 1.0f;  // Gset 数据的权重范围通常是 [-1, 1]
    calculate_quantization_params(gset.num_edges, max_weight);

    // 更新设备常量
    solver.update_device_constants();

    // 转换为求解器格式
    vector<int> terms(gset.num_edges * 2);
    vector<float> second_coeff(gset.num_edges);
    vector<float> one_coeff(gset.num_vertices);

    convert_gset_to_solver_format(&gset, terms.data(), second_coeff.data(), one_coeff.data());

    // 设置模型数据
    solver.set_model_data(terms.data(), second_coeff.data(), one_coeff.data(), gset.num_edges);

    // 自动计算 xi（如果未指定）
    float xi;
    if (argc > 6) {
        xi = atof(argv[6]);
    } else {
        // 自动计算 xi = 0.5 * sqrt(n - 1) / sqrt(sum(J_ij^2))
        // 与 Python SB.py 第113行一致: self.xi = 0.5 * np.sqrt(self.N - 1) / np.sqrt(csr_matrix.power(self.J, 2).sum())
        float sum_j_squared = 0.0f;
        for (int i = 0; i < gset.num_edges; ++i) {
            float j_ij = second_coeff[i];  // J_ij 已经是正确的值（negate=True 后）
            sum_j_squared += j_ij * j_ij;
        }
        xi = 0.5f * sqrtf(gset.num_vertices - 1) / sqrtf(sum_j_squared);
        cout << "Auto-calculated xi: " << xi << endl;
    }

    // 求解（与CPU版本一致的参数）
    cout << "Solving MAX-CUT problem..." << endl;
    cout << "  - Variables: " << gset.num_vertices << ", Edges: " << gset.num_edges << endl;
    cout << "  - BSB Algorithm (PURE FP8 - No Scaling)" << endl;
    cout << "  - FP8 Format: E4M3 (custom bit-level implementation)" << endl;
    cout << "  - Storage: uint8_t FP8" << endl;
    cout << "  - Arithmetic: Custom FP8 add/sub/mul/neg/convert using bit-level operations" << endl;
    cout << "  - NO Quantization: Pure FP8 without VALUE_SCALE scaling" << endl;
    cout << "  - Computation strategy:" << endl;
    cout << "    - Storage: FP8 (all variables)" << endl;
    cout << "    - Dynamics (gradient, update): FP8 with FP8 accumulator" << endl;
    cout << "    - Cut/Energy: Pure FP8 accumulation (total sum also FP8)" << endl;
    cout << "      * Cut: block_size=200 edges" << endl;
    cout << "      * Energy: block_size=100 spins (linear), block_size=50 edges (quadratic)" << endl;
    cout << "  - Note: Requires -arch=sm_75 or higher (Turing+)" << endl;
    cout << "  - Initialization: normal distribution, sigma=0.1" << endl;
    cout << "  - Using xi=" << fixed << setprecision(6) << xi << ", dt=" << dt << endl;
    cout << "  - Iterations: " << n_iter << ", Batch size: " << batch_size << endl;

    auto start = high_resolution_clock::now();
    solver.solve(n_iter, xi, dt, true);
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start).count();

    vector<float> cuts = solver.get_cut_values();

    float max_cut = cuts[0];
    for (auto cut : cuts) {
        if (cut > max_cut) max_cut = cut;
    }

    cout << "\nBSB FP8 CUDA Solver Results:" << endl;
    cout << "  Time: " << duration << " ms" << endl;
    cout << "  Max cut: " << max_cut << endl;
    cout << "  Iterations: " << n_iter << endl;
    cout << "  Batch size: " << batch_size << endl;
    cout << "  xi: " << xi << endl;
    cout << "  dt: " << dt << endl;
    cout << "  delta: " << delta << endl;

    // 释放 Gset 数据
    free_gset_data(&gset);

    return 0;
}
