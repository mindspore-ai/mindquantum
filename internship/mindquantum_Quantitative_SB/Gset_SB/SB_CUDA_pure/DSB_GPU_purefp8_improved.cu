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
__device__ inline uint8_t fp8_add(uint8_t a, uint8_t b) {
    int sign_a = (a >> 7) & 0x1;
    int sign_b = (b >> 7) & 0x1;
    int exp_a = (a >> 3) & 0xF;
    int exp_b = (b >> 3) & 0xF;
    int mant_a = a & 0x7;
    int mant_b = b & 0x7;

    int value_a = sign_a ? -( (1 << 6) | (exp_a << 3) | mant_a ) : (1 << 6) | (exp_a << 3) | mant_a;
    int value_b = sign_b ? -( (1 << 6) | (exp_b << 3) | mant_b ) : (1 << 6) | (exp_b << 3) | mant_b;

    int sum = value_a + value_b;

    int sign_result = (sum < 0) ? 1 : 0;
    unsigned int abs_sum = (sum < 0) ? -sum : sum;

    if (abs_sum == 0) {
        return 0x00;
    }

    int exp_result = 6;
    int mant_result = abs_sum;

    while ((mant_result & (1 << 7)) == 0 && exp_result > 0) {
        mant_result <<= 1;
        exp_result--;
    }

    int mant_out = (mant_result >> 5) & 0x7;
    int remainder = mant_result & 0x1F;

    if (remainder >= 0x10) {
        mant_out++;
        if (mant_out >= 8) {
            mant_out = 0;
            exp_result++;
        }
    }

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
    return a ^ 0x80;
}

// 纯 FP8 减法：a - b（直接操作FP8位）
__device__ inline uint8_t fp8_sub(uint8_t a, uint8_t b) {
    return fp8_add(a, fp8_neg(b));
}

// 纯 FP8 乘法（直接操作FP8位）
__device__ inline uint8_t fp8_mul(uint8_t a, uint8_t b) {
    if (a == 0 || b == 0) {
        return 0x00;
    }

    int sign_a = (a >> 7) & 0x1;
    int sign_b = (b >> 7) & 0x1;
    int exp_a = (a >> 3) & 0xF;
    int exp_b = (b >> 3) & 0xF;
    int mant_a = a & 0x7;
    int mant_b = b & 0x7;

    int sign_result = sign_a ^ sign_b;

    const int FP8_EXP_BIAS = 7;
    int exp_result = exp_a + exp_b - FP8_EXP_BIAS;

    unsigned int full_mant_a = (1 << 3) | mant_a;
    unsigned int full_mant_b = (1 << 3) | mant_b;
    unsigned int product = full_mant_a * full_mant_b;

    int exp_offset = 0;
    while ((product & (1 << 7)) == 0 && exp_offset < 14) {
        product <<= 1;
        exp_offset++;
    }

    exp_result -= exp_offset;

    int mant_out = (product >> 5) & 0x7;
    int remainder = product & 0x1F;

    if (remainder >= 0x10) {
        mant_out++;
        if (mant_out >= 8) {
            mant_out = 0;
            exp_result++;
        }
    }

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
    int sign_a = (a >> 7) & 0x1;
    int sign_b = (b >> 7) & 0x1;

    if (sign_a != sign_b) {
        return !sign_a;
    }

    if (sign_a) {
        return a < b;
    } else {
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

    float value = mant / 8.0f;
    if (exp != 0) {
        value = (1.0f + mant / 8.0f) * exp2f(exp - 7);
    }

    return sign ? -value : value;
}

// FP32 到 FP8 转换（仅用于初始化）
__device__ inline uint8_t float_to_fp8(float f) {
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

    int true_exp = exp32 - 127;

    if (true_exp > FP8_MAX_EXP) {
        return sign_bit ? 0xFF : 0x7F;
    }

    if (true_exp < FP8_MIN_EXP) {
        return sign_bit ? 0x80 : 0x00;
    }

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

// 纯FP8量化配置
constexpr float FP8_MAX_VAL = 240.0f;
constexpr float FP8_MIN_VAL = -240.0f;

// Clamp函数
__host__ __device__ inline float clamp(float val, float min_val, float max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

// 主机端 FP8 转换函数
__host__ inline uint8_t float_to_fp8_host(float f) {
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

    int true_exp = exp32 - 127;

    if (true_exp > FP8_MAX_EXP) {
        return sign_bit ? 0xFF : 0x7F;
    }

    if (true_exp < FP8_MIN_EXP) {
        return sign_bit ? 0x80 : 0x00;
    }

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

// 主机端 FP8 到 FP32 转换
__host__ inline float fp8_to_float_host(uint8_t x) {
    int sign = (x >> 7) & 0x1;
    int exp = (x >> 3) & 0xF;
    int mant = x & 0x7;

    if (exp == 0 && mant == 0) {
        return sign ? -0.0f : 0.0f;
    }

    float value = mant / 8.0f;
    if (exp != 0) {
        value = (1.0f + mant / 8.0f) * exp2f(exp - 7);
    }

    return sign ? -value : value;
}

// 计算量化参数
inline void calculate_quantization_params(int num_edges, float max_weight) {
    cout << "\n=== Pure FP8 Parameters (No Scaling) ===" << endl;
    cout << "  Graph statistics:" << endl;
    cout << "    - Edges: " << num_edges << ", Max weight: " << max_weight << endl;
    cout << "  FP8 format: E4M3" << endl;
    cout << "    - FP8 range: [" << FP8_MIN_VAL << ", " << FP8_MAX_VAL << "]" << endl;
    cout << "  Quantization strategy:" << endl;
    cout << "    - NO VALUE_SCALE: Pure FP8 without scaling" << endl;
    cout << "    - Weights [-1, 1] stored directly as FP8" << endl;
    cout << "    - Spin x sign {+1, -1} stored as FP8" << endl;
    cout <<  "  Computation strategy:" << endl;
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

// 纯FP8累加器
struct Fp8Accumulator {
    uint8_t value;

    __device__ void reset() {
        value = float_to_fp8(0.0f);
    }

    __device__ void add_fp8(uint8_t x) {
        value = fp8_add(value, x);
    }

    __device__ void add_fp8_mul(uint8_t a, uint8_t b) {
        value = fp8_fma(a, b, value);
    }

    __device__ float to_float() {
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
    uint8_t* x_sign_fp8;    // FP8存储sign(x), 即 +1 或 -1
};

// 计算 sign(x) 并存储为 FP8
__global__ void calc_sign_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ x_sign_fp8, int num_spins, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    uint8_t zero_fp8 = float_to_fp8(0.0f);
    uint8_t one_fp8 = float_to_fp8(1.0f);
    uint8_t neg_one_fp8 = float_to_fp8(-1.0f);

    int i = 0;
    for (; i + 3 < num_spins; i += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint8_t x_val = __ldg(&x[batch_offset + i + k]);
            x_sign_fp8[batch_offset + i + k] = fp8_gt(x_val, zero_fp8) ? one_fp8 : neg_one_fp8;
        }
    }
    for (; i < num_spins; ++i) {
        uint8_t x_val = __ldg(&x[batch_offset + i]);
        x_sign_fp8[batch_offset + i] = fp8_gt(x_val, zero_fp8) ? one_fp8 : neg_one_fp8;
    }
}

// 初始化批处理数据
__global__ void init_batches_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ y,
                                   uint8_t* __restrict__ gradient_term,
                                   uint8_t* __restrict__ x_sign_fp8,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;

    uint8_t zero_fp8 = float_to_fp8(0.0f);
    uint8_t one_fp8 = float_to_fp8(1.0f);
    uint8_t neg_one_fp8 = float_to_fp8(-1.0f);

    for (int i = 0; i < num_spins; ++i) {
        float rand_val = curand_uniform(&state);
        uint8_t rand_fp8 = float_to_fp8(rand_val - 0.5f);
        uint8_t scale_fp8 = float_to_fp8(0.1f);
        x[batch_offset + i] = fp8_mul(scale_fp8, rand_fp8);

        rand_val = curand_uniform(&state);
        rand_fp8 = float_to_fp8(rand_val - 0.5f);
        y[batch_offset + i] = fp8_mul(scale_fp8, rand_fp8);

        gradient_term[batch_offset + i] = zero_fp8;
        x_sign_fp8[batch_offset + i] = fp8_gt(x[batch_offset + i], zero_fp8) ? one_fp8 : neg_one_fp8;
    }
}

// DSB梯度计算内核（使用纯FP8和sign(x)）
__global__ void calc_gradient_kernel_optimized(uint8_t* __restrict__ x, uint8_t* __restrict__ x_sign_fp8, uint8_t* gradient_term,
                                               int* __restrict__ second_order_terms, uint8_t* __restrict__ second_order_coeff, uint8_t* __restrict__ one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 使用纯FP8累加器计算梯度
    Fp8Accumulator grad;
    grad.reset();

    // 添加一次项系数（使用纯FP8）
    uint8_t one_coeff = one_order_coeff[spin_idx];
    grad.add_fp8(one_coeff);

    // 循环展开4次以提高指令级并行
    // 梯度计算：grad = coeff + sum(J_ij * sign(x_j))
    // 使用纯FP8计算
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = second_order_terms[(t + k) * 2];
            int var2 = second_order_terms[(t + k) * 2 + 1];
            uint8_t J_ij = second_order_coeff[t + k];

            if (var1 == spin_idx) {
                uint8_t sign2_fp8 = x_sign_fp8[batch_offset + var2];
                // 纯FP8计算: grad += J_ij * sign(x2)
                grad.add_fp8_mul(J_ij, sign2_fp8);
            } else if (var2 == spin_idx) {
                uint8_t sign1_fp8 = x_sign_fp8[batch_offset + var1];
                // 纯FP8计算: grad += J_ij * sign(x1)
                grad.add_fp8_mul(J_ij, sign1_fp8);
            }
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        uint8_t J_ij = second_order_coeff[t];

        if (var1 == spin_idx) {
            uint8_t sign2_fp8 = x_sign_fp8[batch_offset + var2];
            grad.add_fp8_mul(J_ij, sign2_fp8);
        } else if (var2 == spin_idx) {
            uint8_t sign1_fp8 = x_sign_fp8[batch_offset + var1];
            grad.add_fp8_mul(J_ij, sign1_fp8);
        }
    }

    // 保存纯FP8结果
    gradient_term[batch_offset + spin_idx] = grad.to_fp8();
}

// y更新内核（使用纯FP8）
__global__ void update_y_kernel_optimized(uint8_t* __restrict__ x, uint8_t* __restrict__ y, uint8_t* __restrict__ gradient_term,
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
    uint8_t one_minus_p = fp8_sub(one_fp8, p_fp8);
    uint8_t neg_one_minus_p = fp8_neg(one_minus_p);

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

// 更新自旋x（使用纯FP8）
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

// 最大割计算内核（使用纯FP8和sign(x)）
__global__ void calc_cut_kernel(uint8_t* __restrict__ x_sign_fp8, uint8_t* __restrict__ cuts,
                                int* __restrict__ second_order_terms, uint8_t* __restrict__ second_order_coeff, uint8_t* __restrict__ one_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 使用分块累加策略避免FP8溢出（纯FP8）
    const int BLOCK_SIZE = 200;

    uint8_t zero_fp8 = float_to_fp8(0.0f);
    uint8_t one_fp8 = float_to_fp8(1.0f);
    uint8_t half_fp8 = float_to_fp8(0.5f);
    uint8_t neg_half_fp8 = float_to_fp8(-0.5f);

    // 纯FP8累加器
    Fp8Accumulator total_acc;
    total_acc.reset();

    // 计算一次项（使用sign(x)）
    const int BLOCK_SIZE_SPINS = 100;
    for (int block_start = 0; block_start < num_spins; block_start += BLOCK_SIZE_SPINS) {
        int block_end = min(block_start + BLOCK_SIZE_SPINS, num_spins);

        Fp8Accumulator block_acc;
        block_acc.reset();

        for (int i = block_start; i < block_end; ++i) {
            uint8_t sign_i = x_sign_fp8[batch_offset + i];
            uint8_t coeff = one_order_coeff[i];
            // 纯FP8计算: energy += coeff * sign(x)
            block_acc.add_fp8_mul(coeff, sign_i);
        }

        uint8_t block_result = block_acc.to_fp8();
        total_acc.add_fp8(block_result);
    }

    // 计算二次项（使用sign(x1) * sign(x2)）
    for (int block_start = 0; block_start < num_second_order; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, num_second_order);

        Fp8Accumulator block_acc;
        block_acc.reset();

        for (int t = block_start; t < block_end; ++t) {
            int var1 = second_order_terms[t * 2];
            int var2 = second_order_terms[t * 2 + 1];
            uint8_t J_ij = second_order_coeff[t];
            uint8_t sign1 = x_sign_fp8[batch_offset + var1];
            uint8_t sign2 = x_sign_fp8[batch_offset + var2];

            // 纯FP8计算: sign1 * sign2
            uint8_t sign_product = fp8_mul(sign1, sign2);

            // 纯FP8计算: term = J_ij * sign1 * sign2
            uint8_t term = fp8_mul(J_ij, sign_product);

            // 纯FP8计算: cut += 0.5 * term
            uint8_t term_half = fp8_mul(term, half_fp8);
            block_acc.add_fp8(term_half);
        }

        uint8_t block_result = block_acc.to_fp8();
        total_acc.add_fp8(block_result);
    }

    // 纯FP8输出
    cuts[batch_idx] = total_acc.to_fp8();
}

// 能量计算内核（使用纯FP8和sign(x)）
__global__ void calc_energy_kernel(uint8_t* __restrict__ x_sign_fp8, uint8_t* __restrict__ energies,
                                   int* __restrict__ second_order_terms, uint8_t* __restrict__ second_order_coeff, uint8_t* __restrict__ one_order_coeff,
                                   int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 使用分块累加策略避免FP8溢出（纯FP8）
    const int BLOCK_SIZE_SPINS = 100;
    const int BLOCK_SIZE_EDGES = 50;

    // 纯FP8累加器
    Fp8Accumulator total_acc;
    total_acc.reset();

    // 计算一次项能量（使用sign(x)）
    for (int block_start = 0; block_start < num_spins; block_start += BLOCK_SIZE_SPINS) {
        int block_end = min(block_start + BLOCK_SIZE_SPINS, num_spins);

        Fp8Accumulator block_acc;
        block_acc.reset();

        for (int i = block_start; i < block_end; ++i) {
            uint8_t sign_i = x_sign_fp8[batch_offset + i];
            uint8_t coeff = one_order_coeff[i];
            // 纯FP8累加: energy += coeff * sign(x)
            block_acc.add_fp8_mul(coeff, sign_i);
        }

        uint8_t block_result = block_acc.to_fp8();
        total_acc.add_fp8(block_result);
    }

    // 计算二次项能量（使用sign(x1) * sign(x2)）
    for (int block_start = 0; block_start < num_second_order; block_start += BLOCK_SIZE_EDGES) {
        int block_end = min(block_start + BLOCK_SIZE_EDGES, num_second_order);

        Fp8Accumulator block_acc;
        block_acc.reset();

        for (int t = block_start; t < block_end; ++t) {
            int var1 = second_order_terms[t * 2];
            int var2 = second_order_terms[t * 2 + 1];
            uint8_t J_ij = second_order_coeff[t];
            uint8_t sign1 = x_sign_fp8[batch_offset + var1];
            uint8_t sign2 = x_sign_fp8[batch_offset + var2];

            // 纯FP8计算: sign1 * sign2
            uint8_t sign_product = fp8_mul(sign1, sign2);

            // 纯FP8计算: energy += J_ij * sign1 * sign2
            uint8_t term = fp8_mul(J_ij, sign_product);
            block_acc.add_fp8(term);
        }

        uint8_t block_result = block_acc.to_fp8();
        total_acc.add_fp8(block_result);
    }

    // 纯FP8输出
    energies[batch_idx] = total_acc.to_fp8();
}

class DSBGPUFP8Solver {
private:
    int batch_size;
    int num_spins;
    int num_second_order;

    uint8_t* d_x;
    uint8_t* d_y;
    uint8_t* d_gradient_term;  // FP8存储
    uint8_t* d_x_sign_fp8;    // FP8存储sign(x)

    int* d_second_order_terms;
    uint8_t* d_second_order_coeff;    // FP8存储
    uint8_t* d_one_order_coeff;       // FP8存储
    cudaStream_t stream;

public:
    DSBGPUFP8Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_x_sign_fp8, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~DSBGPUFP8Solver() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_x_sign_fp8));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void set_model_data(int* terms, float* second_coeff, float* one_coeff, int num_second) {
        num_second_order = num_second;

        vector<uint8_t> fp8_second_order_coeff_host(num_second);
        vector<uint8_t> fp8_one_order_coeff_host(num_spins);
        for (int i = 0; i < num_second; ++i) {
            fp8_second_order_coeff_host[i] = float_to_fp8_host(second_coeff[i]);
        }
        for (int i = 0; i < num_spins; ++i) {
            fp8_one_order_coeff_host[i] = float_to_fp8_host(one_coeff[i]);
        }

        CUDA_CHECK(cudaMalloc(&d_second_order_terms, num_second_order * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, num_second_order * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, num_spins * sizeof(uint8_t)));

        CUDA_CHECK(cudaMemcpyAsync(d_second_order_terms, terms, num_second_order * 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_coeff, fp8_second_order_coeff_host.data(), num_second_order * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_one_order_coeff, fp8_one_order_coeff_host.data(), num_spins * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void solve(int n_iter, float xi, float dt, bool is_maximize = true) {
        const int BLOCK_SIZE = 256;
        const int BLOCKS_PER_BATCH = (num_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 grid_x(batch_size);
        dim3 block_x(BLOCK_SIZE);
        dim3 grid_2d(batch_size, BLOCKS_PER_BATCH);

        init_batches_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_y, d_gradient_term, d_x_sign_fp8, num_spins, num_second_order, time(nullptr), batch_size);

        const float delta = 0.9f;

        for (int iter = 0; iter < n_iter; ++iter) {
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);

            // 计算sign(x)并存储为FP8
            calc_sign_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_x_sign_fp8, num_spins, batch_size);

            // 计算梯度（使用sign(x)和纯FP8）
            calc_gradient_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_x_sign_fp8, d_gradient_term,
                d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                num_spins, num_second_order, batch_size);

            // 更新y
            update_y_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, d_gradient_term,
                num_spins, p, xi, dt, batch_size, is_maximize);

            // 更新x
            update_x_kernel<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, num_spins, dt, delta, batch_size);
        }

        // 最后计算sign(x)用于最大割计算
        calc_sign_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_x_sign_fp8, num_spins, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        uint8_t* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(uint8_t)));

        calc_cut_kernel<<<batch_size, 256, 0, stream>>>(d_x_sign_fp8, d_cuts, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        vector<uint8_t> cuts_fp8(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(cuts_fp8.data(), d_cuts, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_cuts));

        for (int i = 0; i < batch_size; ++i) {
            cuts[i] = fp8_to_float_host(cuts_fp8[i]);
        }

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        uint8_t* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(uint8_t)));

        calc_energy_kernel<<<batch_size, 256, 0, stream>>>(d_x_sign_fp8, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        vector<uint8_t> energies_fp8(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(energies_fp8.data(), d_energies, batch_size * sizeof(uint8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_energies));

        for (int i = 0; i < batch_size; ++i) {
            energies[i] = fp8_to_float_host(energies_fp8[i]);
        }

        return energies;
    }
};

int main(int argc, char** argv) {
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
    int ret = read_gset_file(gset_file, true, &gset);

    if (ret != 0) {
        cerr << "Error: Failed to read Gset file " << gset_file << endl;
        return -2;
    }

    // 打印 Gset 信息
    print_gset_info(&gset);

    // 创建求解器
    DSBGPUFP8Solver solver(gset.num_vertices, batch_size);

    // 计算自适应量化参数
    float max_weight = 1.0f;
    calculate_quantization_params(gset.num_edges, max_weight);

    // 转换为求解器格式
    vector<int> terms(gset.num_edges * 2);
    vector<float> second_coeff(gset.num_edges);
    vector<float> one_coeff(gset.num_vertices);

    convert_gset_to_solver_format(&gset, terms.data(), second_coeff.data(), one_coeff.data());

    // 设置模型数据
    solver.set_model_data(terms.data(), second_coeff.data(), one_coeff.data(), gset.num_edges);

    // 自动计算 xi
    float xi;
    if (argc > 6) {
        xi = atof(argv[6]);
    } else {
        float sum_j_squared = 0.0f;
        for (int i = 0; i < gset.num_edges; ++i) {
            float j_ij = second_coeff[i];
            sum_j_squared += j_ij * j_ij;
        }
        xi = 0.5f * sqrtf(gset.num_vertices - 1) / sqrtf(sum_j_squared);
        cout << "Auto-calculated xi: " << xi << endl;
    }

    // 求解
    cout << "Solving MAX-CUT problem..." << endl;
    cout << "  - Variables: " << gset.num_vertices << ", Edges: " << gset.num_edges << endl;
    cout << "  - DSB Algorithm (PURE FP8 - No Scaling, discrete variables via sign)" << endl;
    cout << "  - FP8 Format: E4M3 (custom bit-level implementation)" << endl;
    cout << "  - Storage: uint8_t FP8 (all variables)" << endl;
    cout << "  - Arithmetic: Custom FP8 add/sub/mul/neg/convert using bit-level operations" << endl;
    cout << "  - NO Quantization: Pure FP8 without VALUE_SCALE scaling" << endl;
    cout << "  - Discretization: sign(x) -> {+1.0, -1.0} stored as FP8" << endl;
    cout << "  - Computation strategy:" << endl;
    cout << "    - Storage: FP8 (all variables including sign(x))" << endl;
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

    cout << "\nDSB FP8 CUDA Solver Results:" << endl;
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
