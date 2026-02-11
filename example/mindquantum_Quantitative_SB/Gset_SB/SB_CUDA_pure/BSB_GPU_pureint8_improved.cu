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
#include "sb_cuda_gset_reader.h"

using namespace std;
using namespace std::chrono;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << #call << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============ 纯INT8量化配置 ============
// 纯INT8：不使用任何缩放因子，所有值直接在INT8范围内
// INT8 范围：[-128, 127]
constexpr int8_t INT8_MAX_VAL = 127;
constexpr int8_t INT8_MIN_VAL = -128;

// 纯INT8：不使用缩放因子

// FP32到INT8转换（仅用于输入，host端，不使用缩放）
__host__ inline int8_t float_to_int8(float f) {
    // 直接四舍五入，不使用缩放
    int32_t temp = static_cast<int32_t>(roundf(f));
    if (temp > INT8_MAX_VAL) temp = INT8_MAX_VAL;
    if (temp < INT8_MIN_VAL) temp = INT8_MIN_VAL;
    return static_cast<int8_t>(temp);
}

// INT8到FP32转换（仅用于输出，host端，不使用缩放）
__host__ inline float int8_to_float(int8_t x) {
    // 直接转换，不除以缩放因子
    return static_cast<float>(x);
}

// 计算INT8参数（纯INT8：不使用缩放）
inline void calculate_quantization_params(int num_edges, float max_weight) {
    // 纯INT8策略：
    // - 不使用任何缩放因子
    // - 图权重 [-1, 1] 直接存储为INT8
    // - 自旋变量 x 在 [-1, 1] 范围内直接存储为INT8
    // - 所有计算使用纯INT8

    cout << "\n=== Pure INT8 Parameters (No Scaling) ===" << endl;
    cout << "  Graph statistics:" << endl;
    cout << "    - Edges: " << num_edges << ", Max weight: " << max_weight << endl;
    cout << "  INT8 format: Signed 8-bit integer" << endl;
    cout << "    - INT8 range: [" << static_cast<int>(INT8_MIN_VAL) << ", " << static_cast<int>(INT8_MAX_VAL) << "]" << endl;
    cout << "  Quantization strategy:" << endl;
    cout << "    - NO Scaling: Pure INT8 without any scaling factors" << endl;
    cout << "    - Weights [-1, 1] stored directly as INT8" << endl;
    cout << "    - Spin x [-1, 1] stored directly as INT8" << endl;
    cout << "  Computation strategy:" << endl;
    cout << "    - Storage: INT8 (all variables)" << endl;
    cout << "    - Arithmetic: INT8 (all operations)" << endl;
    cout << "    - Accumulation: INT8 (no higher precision intermediates)" << endl;
    cout << "========================================\n" << endl;
}

// ============ 纯INT8算术运算 ============

// 纯INT8加法（带饱和）
__device__ inline int8_t int8_add(int8_t a, int8_t b) {
    if (b > 0 && a > INT8_MAX_VAL - b) return INT8_MAX_VAL;
    if (b < 0 && a < INT8_MIN_VAL - b) return INT8_MIN_VAL;
    return a + b;
}

// 纯INT8减法（带饱和）
__device__ inline int8_t int8_sub(int8_t a, int8_t b) {
    if (b > 0 && a < INT8_MIN_VAL + b) return INT8_MIN_VAL;
    if (b < 0 && a > INT8_MAX_VAL + b) return INT8_MAX_VAL;
    return a - b;
}

// 纯INT8乘法（不带缩放，只做饱和）
__device__ inline int8_t int8_mul(int8_t a, int8_t b) {
    int16_t product = static_cast<int16_t>(a) * static_cast<int16_t>(b);

    // 不使用缩放，直接返回INT8范围内的结果（带饱和）
    if (product > INT8_MAX_VAL) return INT8_MAX_VAL;
    if (product < INT8_MIN_VAL) return INT8_MIN_VAL;
    return static_cast<int8_t>(product);
}

// 纯INT8乘加：a*b + c
__device__ inline int8_t int8_fma(int8_t a, int8_t b, int8_t c) {
    int8_t product = int8_mul(a, b);
    return int8_add(product, c);
}

// 纯INT8取负
__device__ inline int8_t int8_neg(int8_t a) {
    return (a == INT8_MIN_VAL) ? INT8_MIN_VAL : -a;  // -128仍然是-128
}

// 纯INT8比较函数
__device__ inline bool int8_gt(int8_t a, int8_t b) {
    return a > b;
}

__device__ inline bool int8_lt(int8_t a, int8_t b) {
    return a < b;
}

__device__ inline bool int8_ge(int8_t a, int8_t b) {
    return a >= b;
}

__device__ inline bool int8_le(int8_t a, int8_t b) {
    return a <= b;
}

// 纯INT8限制函数
__device__ inline int8_t int8_clamp(int8_t val, int8_t min_val, int8_t max_val) {
    if (int8_lt(val, min_val)) return min_val;
    if (int8_gt(val, max_val)) return max_val;
    return val;
}

// 纯INT8绝对值
__device__ inline int8_t int8_abs(int8_t a) {
    return (a < 0) ? int8_neg(a) : a;
}

// 纯INT8除法（除以2，用于割值计算）
__device__ inline int8_t int8_div2(int8_t a) {
    if (a >= 0) {
        return a / 2;
    } else {
        return (a - 1) / 2;
    }
}

// ============ INT8累加器 ============
// 用于累加多个INT8值（纯INT8，不接受其他精度）
struct Int8Accumulator {
    int8_t value;

    __device__ void reset() {
        value = 0;
    }

    __device__ void add(int8_t x) {
        value = int8_add(value, x);
    }

    __device__ void add_mul(int8_t a, int8_t b) {
        // 纯INT8计算: value += a * b
        int8_t product = int8_mul(a, b);
        value = int8_add(value, product);
    }

    __device__ int8_t to_int8() {
        return value;
    }

    __device__ float to_float() {
        // 仅用于显示，转换为FP32
        return static_cast<float>(value);
    }
};

// ============ 纯INT8常量 ============
__device__ inline int8_t int8_zero() { return 0; }
__device__ inline int8_t int8_one() { return 1; }
__device__ inline int8_t int8_neg_one() { return -1; }

// ============ 纯INT8随机数生成 ============
__device__ inline uint8_t int8_rand(uint32_t* state) {
    *state = (*state * 1103515245 + 12345) & 0x7fffffff;
    return static_cast<uint8_t>(*state % 256);
}

__global__ void init_batches_kernel(int8_t* __restrict__ x, int8_t* __restrict__ y,
                                   int8_t* __restrict__ gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    uint32_t rng_state = static_cast<uint32_t>(seed + batch_idx);
    int batch_offset = batch_idx * num_spins;

    for (int i = 0; i < num_spins; ++i) {
        uint8_t rand1 = int8_rand(&rng_state);
        uint8_t rand2 = int8_rand(&rng_state);
        
        // 映射到[-0.1, 0.1]范围
        int8_t x_val = static_cast<int8_t>(rand1) - 128;
        x_val = x_val / 10;  // 近似[-1.28, 1.27] -> [-0.13, 0.13]
        
        int8_t y_val = static_cast<int8_t>(rand2) - 128;
        y_val = y_val / 10;
        
        x[batch_offset + i] = x_val;
        y[batch_offset + i] = y_val;
        gradient_term[batch_offset + i] = int8_zero();
    }
}

__global__ void calc_gradient_kernel_optimized(int8_t* __restrict__ x, int8_t* __restrict__ gradient_term,
                                               int* __restrict__ second_order_terms,
                                               int8_t* __restrict__ second_order_coeff,
                                               int8_t* __restrict__ one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    
    // 使用累加器避免溢出
    Int8Accumulator grad;
    grad.reset();
    grad.add(one_order_coeff[spin_idx]);

    // 累加二次项
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = second_order_terms[(t + k) * 2];
            int var2 = second_order_terms[(t + k) * 2 + 1];
            int8_t J_ij = second_order_coeff[t + k];

            if (var1 == spin_idx) {
                int8_t x2_val = x[batch_offset + var2];
                grad.add_mul(J_ij, x2_val);
            } else if (var2 == spin_idx) {
                int8_t x1_val = x[batch_offset + var1];
                grad.add_mul(J_ij, x1_val);
            }
        }
    }

    for (; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        int8_t J_ij = second_order_coeff[t];

        if (var1 == spin_idx) {
            int8_t x2_val = x[batch_offset + var2];
            grad.add_mul(J_ij, x2_val);
        } else if (var2 == spin_idx) {
            int8_t x1_val = x[batch_offset + var1];
            grad.add_mul(J_ij, x1_val);
        }
    }

    gradient_term[batch_offset + spin_idx] = grad.to_int8();
}

__global__ void update_y_kernel_optimized(int8_t* __restrict__ x, int8_t* __restrict__ y,
                                          int8_t* __restrict__ gradient_term,
                                          int num_spins, int p_val, int xi_val, int dt_val,
                                          int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    int8_t x_int8 = x[batch_offset + spin_idx];
    int8_t grad_int8 = gradient_term[batch_offset + spin_idx];
    int8_t y_int8 = y[batch_offset + spin_idx];

    // dy/dt = -(1-p)*x + ξ*∇f(x)
    int8_t one_int8 = int8_one();
    int8_t p_int8 = static_cast<int8_t>(p_val);
    int8_t one_minus_p = int8_sub(one_int8, p_int8);
    int8_t neg_one_minus_p = int8_neg(one_minus_p);

    int8_t xi_int8 = static_cast<int8_t>(is_maximize ? xi_val : -xi_val);

    // term1 = -(1-p)*x
    int8_t term1 = int8_mul(neg_one_minus_p, x_int8);

    // term2 = xi*grad
    int8_t term2 = int8_mul(xi_int8, grad_int8);

    // update = (term1 + term2) * dt
    int8_t sum = int8_add(term1, term2);
    int8_t dt_int8 = static_cast<int8_t>(dt_val);
    int8_t update = int8_mul(sum, dt_int8);

    // y_new = y + update
    int8_t y_new = int8_add(y_int8, update);

    // 边界约束[-1, 1]（在缩放后的坐标系中）
    int8_t one_h = int8_one();
    int8_t neg_one_h = int8_neg_one();
    y_new = int8_clamp(y_new, neg_one_h, one_h);

    y[batch_offset + spin_idx] = y_new;
}

__global__ void update_x_kernel(int8_t* __restrict__ x, int8_t* __restrict__ y,
                                int num_spins, int dt_val, int delta_val, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    int8_t x_int8 = x[batch_offset + spin_idx];
    int8_t y_int8 = y[batch_offset + spin_idx];

    // dx/dt = y * delta * dt
    int8_t dt_int8 = static_cast<int8_t>(dt_val);
    int8_t delta_int8 = static_cast<int8_t>(delta_val);
    int8_t y_dt = int8_mul(y_int8, dt_int8);
    int8_t y_dt_delta = int8_mul(y_dt, delta_int8);
    int8_t x_new = int8_add(x_int8, y_dt_delta);

    // 边界约束：|x| > 1时投影到边界
    int8_t zero_val = int8_zero();
    int8_t one_int8 = int8_one();
    int8_t neg_one_int8 = int8_neg_one();

    if (int8_gt(x_new, one_int8)) {
        x_new = one_int8;
        y[batch_offset + spin_idx] = zero_val;
    } else if (int8_lt(x_new, neg_one_int8)) {
        x_new = neg_one_int8;
        y[batch_offset + spin_idx] = zero_val;
    }

    x[batch_offset + spin_idx] = x_new;
}

// 割值计算：纯INT8计算，输出INT8
__global__ void calc_cut_kernel(int8_t* __restrict__ x, int8_t* __restrict__ cuts,
                                int* __restrict__ second_order_terms,
                                int8_t* __restrict__ second_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 纯INT8累加器（分块累加避免溢出）
    // INT8范围：[-128, 127]，每条边最多贡献：|J_ij| * (1 - s_i*s_j) * 0.5 = 1 * 2 * 0.5 = 1
    // 每块最多可以累加 127 条边
    const int BLOCK_SIZE = 100;

    int8_t zero = int8_zero();
    int8_t one = int8_one();
    int8_t half = 1;  // INT8中0.5近似为1（实际会除以2）

    // 纯INT8累加所有块的结果
    Int8Accumulator total_acc;
    total_acc.reset();

    // 分块处理
    for (int block_start = 0; block_start < num_second_order; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, num_second_order);

        // 当前块使用INT8累加器
        Int8Accumulator block_acc;
        block_acc.reset();

        // 处理当前块的所有边
        for (int t = block_start; t < block_end; ++t) {
            int var1 = second_order_terms[t * 2];
            int var2 = second_order_terms[t * 2 + 1];
            int8_t J_ij = second_order_coeff[t];
            int8_t x1 = x[batch_offset + var1];
            int8_t x2 = x[batch_offset + var2];

            // INT8绝对值：|J_ij|
            int8_t abs_J = (J_ij >= 0) ? J_ij : int8_neg(J_ij);

            // INT8二元化：x > 0 为 1, x <= 0 为 -1
            int8_t x1_sign = int8_gt(x1, zero) ? one : int8_neg(one);
            int8_t x2_sign = int8_gt(x2, zero) ? one : int8_neg(one);

            // INT8计算：s_i * s_j
            int8_t sign_product = int8_mul(x1_sign, x2_sign);

            // INT8计算：1 - s_i * s_j
            int8_t one_minus = int8_sub(one, sign_product);

            // INT8计算：|J_ij| * (1 - s_i * s_j)
            int8_t term = int8_mul(abs_J, one_minus);

            // INT8计算：0.5 * term（右移1位近似除以2）
            int8_t term_half = term / 2;

            // 纯INT8累加
            block_acc.add(term_half);
        }

        // 将当前块的INT8累加结果添加到总INT8累加器
        int8_t block_result = block_acc.to_int8();
        total_acc.add(block_result);
    }

    // 纯INT8输出
    cuts[batch_idx] = total_acc.to_int8();
}

// 能量计算：纯INT8计算，输出INT8
__global__ void calc_energy_kernel(int8_t* __restrict__ x, int8_t* __restrict__ energies,
                                    int* __restrict__ second_order_terms,
                                    int8_t* __restrict__ second_order_coeff,
                                    int8_t* __restrict__ one_order_coeff,
                                    int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 纯INT8累加器（分块累加避免溢出）
    const int BLOCK_SIZE_SPINS = 50;
    const int BLOCK_SIZE_EDGES = 25;

    int8_t zero = int8_zero();

    // 纯INT8累加所有块的结果
    Int8Accumulator total_acc;
    total_acc.reset();

    // 计算一次项能量（分块累加）
    for (int block_start = 0; block_start < num_spins; block_start += BLOCK_SIZE_SPINS) {
        int block_end = min(block_start + BLOCK_SIZE_SPINS, num_spins);

        Int8Accumulator block_acc;
        block_acc.reset();

        for (int i = block_start; i < block_end; ++i) {
            int8_t x_i = x[batch_offset + i];
            int8_t coeff = one_order_coeff[i];

            // INT8二元化：x > 0 为 1, x <= 0 为 -1
            int8_t x_sign = int8_gt(x_i, zero) ? int8_one() : int8_neg_one();

            // 纯INT8累加: energy += coeff * x_sign
            block_acc.add_mul(coeff, x_sign);
        }

        int8_t block_result = block_acc.to_int8();
        total_acc.add(block_result);
    }

    // 计算二次项能量（分块累加）
    for (int block_start = 0; block_start < num_second_order; block_start += BLOCK_SIZE_EDGES) {
        int block_end = min(block_start + BLOCK_SIZE_EDGES, num_second_order);

        Int8Accumulator block_acc;
        block_acc.reset();

        for (int t = block_start; t < block_end; ++t) {
            int var1 = second_order_terms[t * 2];
            int var2 = second_order_terms[t * 2 + 1];
            int8_t J_ij = second_order_coeff[t];
            int8_t x1 = x[batch_offset + var1];
            int8_t x2 = x[batch_offset + var2];

            // INT8二元化
            int8_t x1_sign = int8_gt(x1, zero) ? int8_one() : int8_neg_one();
            int8_t x2_sign = int8_gt(x2, zero) ? int8_one() : int8_neg_one();

            // INT8计算: J_ij * x1_sign * x2_sign
            int8_t product1 = int8_mul(J_ij, x1_sign);
            int8_t contribution = int8_mul(product1, x2_sign);

            block_acc.add(contribution);
        }

        int8_t block_result = block_acc.to_int8();
        total_acc.add(block_result);
    }

    // 纯INT8输出
    energies[batch_idx] = total_acc.to_int8();
}

class BSBGPUINT8Solver {
private:
    int batch_size;
    int num_spins;
    int num_second_order;

    int8_t* d_x;
    int8_t* d_y;
    int8_t* d_gradient_term;

    int* d_second_order_terms;
    int8_t* d_second_order_coeff;
    int8_t* d_one_order_coeff;
    cudaStream_t stream;

public:
    BSBGPUINT8Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(int8_t)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~BSBGPUINT8Solver() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    // 更新设备常量（纯INT8：不需要缩放常量）
    void update_device_constants() {
        // 纯INT8：不需要任何缩放常量
    }

    void set_model_data(int* terms, float* second_coeff, float* one_coeff, int num_second) {
        num_second_order = num_second;

        vector<int8_t> int8_second_order_coeff(num_second);
        vector<int8_t> int8_one_order_coeff(num_spins);
        for (int i = 0; i < num_second; ++i) {
            int8_second_order_coeff[i] = float_to_int8(second_coeff[i]);
        }
        for (int i = 0; i < num_spins; ++i) {
            int8_one_order_coeff[i] = float_to_int8(one_coeff[i]);
        }

        CUDA_CHECK(cudaMalloc(&d_second_order_terms, num_second_order * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, num_second_order * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, num_spins * sizeof(int8_t)));

        CUDA_CHECK(cudaMemcpyAsync(d_second_order_terms, terms, num_second_order * 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_coeff, int8_second_order_coeff.data(), num_second_order * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_one_order_coeff, int8_one_order_coeff.data(), num_spins * sizeof(int8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void solve(int n_iter, float xi, float dt, bool is_maximize = true) {
        const int BLOCK_SIZE = 256;
        const int BLOCKS_PER_BATCH = (num_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 grid_x(batch_size);
        dim3 block_x(BLOCK_SIZE);
        dim3 grid_2d(batch_size, BLOCKS_PER_BATCH);

        init_batches_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_y, d_gradient_term, num_spins, num_second_order, time(nullptr), batch_size);

        // 纯INT8参数（不使用缩放）
        int xi_int8 = static_cast<int>(roundf(xi));
        int dt_int8 = static_cast<int>(roundf(dt));
        int delta_int8 = 1;  // BSB使用delta=1

        for (int iter = 0; iter < n_iter; ++iter) {
            // 计算p(t) = t/T
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);
            int p_int8 = static_cast<int>(roundf(p));

            calc_gradient_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_gradient_term,
                d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                num_spins, num_second_order, batch_size);

            update_y_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, d_gradient_term,
                num_spins, p_int8, xi_int8, dt_int8, batch_size, is_maximize);

            update_x_kernel<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, num_spins, dt_int8, delta_int8, batch_size);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        int8_t* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(int8_t)));

        // 纯INT8计算割值（输出为INT8，然后转换为FP32仅用于显示）
        calc_cut_kernel<<<batch_size, 256, 0, stream>>>(d_x, d_cuts, d_second_order_terms, d_second_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 从INT8复制数据并转换为FP32（仅用于显示）
        vector<int8_t> cuts_int8(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(cuts_int8.data(), d_cuts, batch_size * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_cuts));

        for (int i = 0; i < batch_size; ++i) {
            cuts[i] = static_cast<float>(cuts_int8[i]);
        }

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        int8_t* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(int8_t)));

        // 纯INT8计算能量（输出为INT8，然后转换为FP32仅用于显示）
        calc_energy_kernel<<<batch_size, 256, 0, stream>>>(d_x, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 从INT8复制数据并转换为FP32（仅用于显示）
        vector<int8_t> energies_int8(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(energies_int8.data(), d_energies, batch_size * sizeof(int8_t), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_energies));

        for (int i = 0; i < batch_size; ++i) {
            energies[i] = static_cast<float>(energies_int8[i]);
        }

        return energies;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <gset_file> [num_iter] [batch_size] [dt] [delta] [xi]" << endl;
        return -1;
    }

    const char* gset_file = argv[1];
    int n_iter = (argc > 2) ? atoi(argv[2]) : 1000;
    int batch_size = (argc > 3) ? atoi(argv[3]) : 100;
    float dt = (argc > 4) ? atof(argv[4]) : 0.5f;
    float delta = (argc > 5) ? atof(argv[5]) : 0.9f;

    GsetData gset;
    int ret = read_gset_file(gset_file, true, &gset);
    if (ret != 0) {
        cerr << "Error: Failed to read Gset file " << gset_file << endl;
        return -2;
    }

    print_gset_info(&gset);

    BSBGPUINT8Solver solver(gset.num_vertices, batch_size);

    // 计算自适应量化参数
    float max_weight = 1.0f;  // Gset 数据的权重范围通常是 [-1, 1]
    calculate_quantization_params(gset.num_edges, max_weight);

    // 更新设备常量
    solver.update_device_constants();

    vector<int> terms(gset.num_edges * 2);
    vector<float> second_coeff(gset.num_edges);
    vector<float> one_coeff(gset.num_vertices);
    convert_gset_to_solver_format(&gset, terms.data(), second_coeff.data(), one_coeff.data());
    solver.set_model_data(terms.data(), second_coeff.data(), one_coeff.data(), gset.num_edges);

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

    cout << "Solving MAX-CUT problem..." << endl;
    cout << "  - Variables: " << gset.num_vertices << ", Edges: " << gset.num_edges << endl;
    cout << "  - BSB Algorithm (PURE INT8 - No Scaling)" << endl;
    cout << "  - INT8 Format: Signed 8-bit integer" << endl;
    cout << "  - Storage: int8_t" << endl;
    cout << "  - Arithmetic: INT8 (all operations)" << endl;
    cout << "  - NO Scaling: Pure INT8 without any scaling factors" << endl;
    cout << "  - Computation strategy:" << endl;
    cout << "    - Storage: INT8 (all variables)" << endl;
    cout << "    - Dynamics (gradient, update): INT8" << endl;
    cout << "    - Cut/Energy: Pure INT8 accumulation (final sum also INT8)" << endl;
    cout << "    - NO higher precision intermediates" << endl;
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

    cout << "\nBSB PURE INT8 CUDA Solver Results:" << endl;
    cout << "  Time: " << duration << " ms" << endl;
    cout << "  Max cut: " << max_cut << endl;
    cout << "  Iterations: " << n_iter << endl;
    cout << "  Batch size: " << batch_size << endl;
    cout << "  xi: " << xi << endl;
    cout << "  dt: " << dt << endl;
    cout << "  delta: " << delta << endl;

    free_gset_data(&gset);
    return 0;
}
