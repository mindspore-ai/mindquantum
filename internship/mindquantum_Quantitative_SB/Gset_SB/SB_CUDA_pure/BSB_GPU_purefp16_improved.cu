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
#include <cuda_fp16.h>
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

// FP16配置
constexpr float FP16_MAX_VAL = 65504.0f;
constexpr float FP16_MIN_VAL = -65504.0f;

// FP16累加缩放因子：将割值缩放到合理的范围
constexpr float FP16_CUT_SCALE = 100.0f;  // 将 13345 缩放到约 133，在 FP16 精度范围内
constexpr float FP16_CUT_SCALE_INV = 1.0f / FP16_CUT_SCALE;

__host__ __device__ inline float fp16_to_float(__half x) {
    return __half2float(x);
}

__host__ __device__ inline __half float_to_fp16(float f) {
    return __float2half(f);
}

// 缩放后的FP16转换
__host__ __device__ inline __half float_to_fp16_scaled(float f) {
    return __float2half(f * FP16_CUT_SCALE);
}

__host__ __device__ inline float fp16_scaled_to_float(__half x) {
    return __half2float(x) * FP16_CUT_SCALE_INV;
}

__host__ __device__ inline int sign_fp16(__half x) {
    float x_val = fp16_to_float(x);
    return (x_val >= 0.0f) ? 1 : -1;
}

// 计算FP16参数（纯FP16：不使用缩放）
inline void calculate_fp16_quantization_params(int num_edges, float max_weight) {
    // 纯FP16策略：
    // - 不使用任何缩放因子
    // - 图权重 [-1, 1] 直接存储为FP16
    // - 自旋变量 x 在 [-1, 1] 范围内直接存储为FP16
    // - 所有计算使用纯FP16

    cout << "\n=== Pure FP16 Parameters (No Scaling) ===" << endl;
    cout << "  Graph statistics:" << endl;
    cout << "    - Edges: " << num_edges << ", Max weight: " << max_weight << endl;
    cout << "  FP16 format: Binary16" << endl;
    cout << "    - FP16 range: [" << FP16_MIN_VAL << ", " << FP16_MAX_VAL << "]" << endl;
    cout << "  Quantization strategy:" << endl;
    cout << "    - NO Scaling: Pure FP16 without any scaling factors" << endl;
    cout << "    - Weights [-1, 1] stored directly as FP16" << endl;
    cout << "    - Spin x [-1, 1] stored directly as FP16" << endl;
    cout << "  Computation strategy:" << endl;
    cout << "    - Storage: FP16 (all variables)" << endl;
    cout << "    - Arithmetic: Pure FP16 (all operations)" << endl;
    cout << "    - Accumulation: FP16 (no FP32 intermediates)" << endl;
    cout << "============================================\n" << endl;
}

// GPU批处理数据结构（全部使用FP16）
struct GPUBatchData {
    __half* x;
    __half* y;
    int* x_sign;
    __half* gradient_term;  // 梯度计算使用FP16
};

// GPU上的BSB模型数据
struct GPUBSBModel {
    int* second_order_terms;
    __half* second_order_coeff;
    __half* one_order_coeff;
    int num_spins;
    int num_second_order;
};

// 初始化批处理数据（全部使用FP16）
__global__ void init_batches_kernel(__half* x, __half* y, int* x_sign, __half* gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;
    const __half init_scale = float_to_fp16(0.02f);  // 与 Python 保持一致: 0.02 * (rand() - 0.5)

    for (int i = 0; i < num_spins; ++i) {
        // 使用均匀分布，与 Python 保持一致: 0.02 * (rand() - 0.5)
        float rand_val = curand_uniform(&state);
        __half x_val = float_to_fp16(fp16_to_float(init_scale) * (rand_val - 0.5f));
        x[batch_offset + i] = x_val;

        rand_val = curand_uniform(&state);
        __half y_val = float_to_fp16(fp16_to_float(init_scale) * (rand_val - 0.5f));
        y[batch_offset + i] = y_val;

        x_sign[batch_offset + i] = sign_fp16(x[batch_offset + i]);
        gradient_term[batch_offset + i] = float_to_fp16(0.0f);
    }
}

// BSB梯度计算内核（纯FP16计算）
__global__ void calc_gradient_kernel_optimized(__half* x, __half* gradient_term,
                                               int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    __half grad = one_order_coeff[spin_idx];

    for (int t = 0; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        __half J_ij = second_order_coeff[t];

        if (var1 == spin_idx) {
            __half x2_val = x[batch_offset + var2];
            // FP16计算: grad = grad + J_ij * x2_val
            __half prod = __hmul(J_ij, x2_val);
            grad = __hadd(grad, prod);
        } else if (var2 == spin_idx) {
            __half x1_val = x[batch_offset + var1];
            // FP16计算: grad = grad + J_ij * x1_val
            __half prod = __hmul(J_ij, x1_val);
            grad = __hadd(grad, prod);
        }
    }

    gradient_term[batch_offset + spin_idx] = grad;
}

// y更新内核（纯FP16计算）
__global__ void update_y_kernel_optimized(__half* x, __half* y, __half* gradient_term,
                                          int num_spins, float p, float xi, float dt, int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    const __half delta = __float2half(1.0f);  // BSB使用delta=1
    const __half pump_factor = __float2half(-(1.0f - p));
    const __half sign_factor = __float2half(is_maximize ? xi : -xi);

    __half x_val = x[batch_offset + spin_idx];
    __half grad_val = gradient_term[batch_offset + spin_idx];
    __half y_val = y[batch_offset + spin_idx];

    // FP16计算: term1 = pump_factor * x_val
    __half term1 = __hmul(pump_factor, x_val);

    // FP16计算: term2 = sign_factor * grad_val
    __half term2 = __hmul(sign_factor, grad_val);

    // FP16计算: sum = term1 + term2
    __half sum = __hadd(term1, term2);

    // FP16计算: update = sum * dt
    __half dt_h = __float2half(dt);
    __half update = __hmul(sum, dt_h);

    // FP16计算: y_new = y_val + update
    __half y_new = __hadd(y_val, update);

    y[batch_offset + spin_idx] = y_new;
}

// 更新自旋x（纯FP16计算）
__global__ void update_x_kernel(__half* x, __half* y, int num_spins, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    __half x_val = x[batch_offset + spin_idx];
    __half y_val = y[batch_offset + spin_idx];

    // FP16计算: y_val * dt
    __half dt_h = __float2half(dt);
    __half temp1 = __hmul(y_val, dt_h);

    // FP16计算: temp1 * delta
    __half delta_h = __float2half(delta);
    __half temp2 = __hmul(temp1, delta_h);

    // FP16计算: x_new = x_val + temp2
    __half x_new = __hadd(x_val, temp2);

    // FP16边界检查: |x| > 1时投影到边界
    __half one = __float2half(1.0f);
    __half neg_one = __float2half(-1.0f);
    __half zero = __float2half(0.0f);
    __half abs_x_new = __habs(x_new);

    // FP16比较: abs_x_new > 1.0f
    __half gt_one = __hgt(abs_x_new, one);  // 返回 1.0f 或 0.0f
    __half le_one = __hsub(zero, __hadd(gt_one, neg_one));  // 补集

    // FP16边界限制
    __half fp16_min = __float2half(FP16_MIN_VAL);
    __half fp16_max = __float2half(FP16_MAX_VAL);
    __half clamped = __hmax(__hmin(x_new, fp16_max), fp16_min);

    // FP16逻辑: 如果|x|>1则投影到±1，否则使用clamped值
    __half sign_x = __hdiv(x_new, __hadd(abs_x_new, zero));  // x_new / |x_new|，x_new!=0
    __half zero_check = __heq(x_new, zero);
    __half sign_final = __hadd(sign_x, __hmul(zero_check, zero));  // 如果x_new=0，sign=0

    __half projected = __hmul(sign_final, one);  // ±1 或 0

    // FP16选择: gt_one ? projected : clamped
    __half from_projection = __hmul(gt_one, projected);
    __half from_clamped = __hmul(le_one, clamped);
    __half x_result = __hadd(from_projection, from_clamped);

    // FP16 y重置: 如果|x|>1则y=0
    __half y_result = __hmul(le_one, y_val);  // 如果|x|<=1则保留y值，否则0

    x[batch_offset + spin_idx] = x_result;
    y[batch_offset + spin_idx] = y_result;
}

// 最大割计算内核（纯FP16计算，输出FP16）
__global__ void calc_cut_kernel(__half* x, __half* cuts,
                                int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // FP16常量
    __half zero = __float2half(0.0f);
    __half one = __float2half(1.0f);
    __half half = __float2half(0.5f);
    __half neg_one = __float2half(-1.0f);

    // 纯FP16累加器
    __half cut_h = __float2half(0.0f);

    // BSB算法：使用二元化后的值计算切割值（x_sign: x>0 为 1，x≤0 为 -1）
    // Gset 问题没有一阶项（外场），跳过

    // MAX-CUT 计算：cut = 0.5 * sum(|J_ij| * (1 - s_i * s_j))
    // 其中 s_i ∈ {-1, 1} 是二元化后的值
    // 使用纯 FP16 计算

    // 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = second_order_terms[(t + k) * 2];
            int var2 = second_order_terms[(t + k) * 2 + 1];
            __half J_ij = second_order_coeff[t + k];  // FP16 系数
            __half x1_val = x[batch_offset + var1];    // FP16 x 值
            __half x2_val = x[batch_offset + var2];    // FP16 x 值

            // FP16二元化：x > 0 为 1，x <= 0 为 -1
            __half x1_gt_zero = __hgt(x1_val, zero);
            __half x1_sign = __hadd(__hmul(x1_gt_zero, one), __hmul(__hsub(zero, x1_gt_zero), neg_one));
            __half x2_gt_zero = __hgt(x2_val, zero);
            __half x2_sign = __hadd(__hmul(x2_gt_zero, one), __hmul(__hsub(zero, x2_gt_zero), neg_one));

            // FP16计算: s_i * s_j
            __half product = __hmul(x1_sign, x2_sign);

            // FP16计算: 1 - s_i * s_j
            __half term = __hsub(one, product);

            // FP16计算: |J_ij|
            __half abs_J_ij = __habs(J_ij);

            // FP16计算: 0.5 * |J_ij| * (1 - s_i * s_j)
            __half contribution = __hmul(__hmul(half, abs_J_ij), term);

            // FP16累加
            cut_h = __hadd(cut_h, contribution);
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        __half J_ij = second_order_coeff[t];  // FP16 系数
        __half x1_val = x[batch_offset + var1];  // FP16 x 值
        __half x2_val = x[batch_offset + var2];  // FP16 x 值

        // FP16二元化：x > 0 为 1，x <= 0 为 -1
        __half x1_gt_zero = __hgt(x1_val, zero);
        __half x1_sign = __hadd(__hmul(x1_gt_zero, one), __hmul(__hsub(zero, x1_gt_zero), neg_one));
        __half x2_gt_zero = __hgt(x2_val, zero);
        __half x2_sign = __hadd(__hmul(x2_gt_zero, one), __hmul(__hsub(zero, x2_gt_zero), neg_one));

        // FP16计算: s_i * s_j
        __half product = __hmul(x1_sign, x2_sign);

        // FP16计算: 1 - s_i * s_j
        __half term = __hsub(one, product);

        // FP16计算: |J_ij|
        __half abs_J_ij = __habs(J_ij);

        // FP16计算: 0.5 * |J_ij| * (1 - s_i * s_j)
        __half contribution = __hmul(__hmul(half, abs_J_ij), term);

        // FP16累加
        cut_h = __hadd(cut_h, contribution);
    }

    // 输出FP16值
    cuts[batch_idx] = cut_h;
}

// 能量计算内核（缩放FP16计算，输出FP16）
__global__ void calc_energy_kernel(__half* x, __half* energies,
                                  int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                  int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // FP16常量
    __half zero = __float2half(0.0f);
    __half one = __float2half(1.0f);
    __half neg_one = __float2half(-1.0f);

    // 纯FP16累加器
    __half energy_h = __float2half(0.0f);

    // 计算一次项能量（使用二元化后的值，FP16 系数，纯FP16 累加）
    for (int i = 0; i < num_spins; ++i) {
        __half x_val = x[batch_offset + i];
        __half coeff = one_order_coeff[i];

        // FP16二元化：x > 0 为 1，x <= 0 为 -1
        __half x_gt_zero = __hgt(x_val, zero);
        __half x_sign = __hadd(__hmul(x_gt_zero, one), __hmul(__hsub(zero, x_gt_zero), neg_one));

        // FP16累加: coeff * x_sign
        __half contribution = __hmul(coeff, x_sign);
        energy_h = __hadd(energy_h, contribution);
    }

    // 计算二次项能量（使用二元化后的值，FP16 系数，纯FP16 累加）
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = second_order_terms[(t + k) * 2];
            int var2 = second_order_terms[(t + k) * 2 + 1];
            __half J_ij = second_order_coeff[t + k];  // FP16 系数
            __half x1_val = x[batch_offset + var1];
            __half x2_val = x[batch_offset + var2];

            // FP16二元化
            __half x1_gt_zero = __hgt(x1_val, zero);
            __half x1_sign = __hadd(__hmul(x1_gt_zero, one), __hmul(__hsub(zero, x1_gt_zero), neg_one));
            __half x2_gt_zero = __hgt(x2_val, zero);
            __half x2_sign = __hadd(__hmul(x2_gt_zero, one), __hmul(__hsub(zero, x2_gt_zero), neg_one));

            // FP16计算: J_ij * s_i * s_j
            __half contribution = __hmul(J_ij, __hmul(x1_sign, x2_sign));
            energy_h = __hadd(energy_h, contribution);
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        __half J_ij = second_order_coeff[t];  // FP16 系数
        __half x1_val = x[batch_offset + var1];
        __half x2_val = x[batch_offset + var2];

        // FP16二元化
        __half x1_gt_zero = __hgt(x1_val, zero);
        __half x1_sign = __hadd(__hmul(x1_gt_zero, one), __hmul(__hsub(zero, x1_gt_zero), neg_one));
        __half x2_gt_zero = __hgt(x2_val, zero);
        __half x2_sign = __hadd(__hmul(x2_gt_zero, one), __hmul(__hsub(zero, x2_gt_zero), neg_one));

        // FP16计算: J_ij * s_i * s_j
        __half contribution = __hmul(J_ij, __hmul(x1_sign, x2_sign));
        energy_h = __hadd(energy_h, contribution);
    }

    // 输出FP16值
    energies[batch_idx] = energy_h;
}

// BSB GPU求解器（FP16版本）
class BSBGPUFP16Solver {
private:
    int batch_size;
    int num_spins;
    int num_second_order;

    __half* d_x;
    __half* d_y;
    int* d_x_sign;
    __half* d_gradient_term;  // 全部使用FP16

    int* d_second_order_terms;
    __half* d_second_order_coeff;
    __half* d_one_order_coeff;

public:
    BSBGPUFP16Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x_sign, batch_size * num_spins * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(__half)));  // FP16存储
    }

    ~BSBGPUFP16Solver() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_x_sign));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
    }

    // 更新设备常量（纯FP16：不需要缩放常量）
    void update_device_constants() {
        // 纯FP16：不需要任何缩放常量
    }

    void set_model_data(int* terms, float* second_coeff, float* one_coeff, int num_second) {
        num_second_order = num_second;

        vector<__half> fp16_second_order_coeff(num_second);
        vector<__half> fp16_one_order_coeff(num_spins);
        for (int i = 0; i < num_second; ++i) {
            fp16_second_order_coeff[i] = float_to_fp16(second_coeff[i]);
        }
        for (int i = 0; i < num_spins; ++i) {
            fp16_one_order_coeff[i] = float_to_fp16(one_coeff[i]);
        }

        CUDA_CHECK(cudaMalloc(&d_second_order_terms, num_second_order * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, num_second_order * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, num_spins * sizeof(__half)));

        CUDA_CHECK(cudaMemcpy(d_second_order_terms, terms, num_second_order * 2 * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_second_order_coeff, fp16_second_order_coeff.data(), num_second_order * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_one_order_coeff, fp16_one_order_coeff.data(), num_spins * sizeof(__half), cudaMemcpyHostToDevice));
    }

    void solve(int n_iter, float xi, float dt, bool is_maximize = true) {
        // 优化：使用更合适的block大小，warp大小为32
        const int BLOCK_SIZE = 256;
        const int BLOCKS_PER_BATCH = (num_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 grid_x(batch_size);
        dim3 block_x(BLOCK_SIZE);
        dim3 grid_2d(batch_size, BLOCKS_PER_BATCH);

        init_batches_kernel<<<grid_x, block_x>>>(d_x, d_y, d_x_sign, d_gradient_term, num_spins, num_second_order, 42, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iter = 0; iter < n_iter; ++iter) {
            // 计算泵浦振幅：p(t) = t/T
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);
            float delta = 1.0f;

            // BSB: 使用连续的 x 值计算梯度（FP16计算，FP16存储）
            calc_gradient_kernel_optimized<<<grid_2d, block_x>>>(d_x, d_gradient_term, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            update_y_kernel_optimized<<<grid_2d, block_x>>>(d_x, d_y, d_gradient_term, num_spins, p, xi, dt, batch_size, is_maximize);
            CUDA_CHECK(cudaDeviceSynchronize());

            update_x_kernel<<<grid_2d, block_x>>>(d_x, d_y, num_spins, dt, delta, batch_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 只在最后同步一次，减少同步开销
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        __half* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(__half)));

        // 使用纯FP16计算割值
        calc_cut_kernel<<<batch_size, 256>>>(d_x, d_cuts, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 从FP16复制数据并转换为FP32
        vector<__half> cuts_fp16(batch_size);
        CUDA_CHECK(cudaMemcpy(cuts_fp16.data(), d_cuts, batch_size * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cuts));

        // 直接转换FP16到FP32
        for (int i = 0; i < batch_size; ++i) {
            cuts[i] = fp16_to_float(cuts_fp16[i]);
        }

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        __half* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(__half)));

        // 使用纯FP16计算能量
        calc_energy_kernel<<<batch_size, 256>>>(d_x, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 从FP16复制数据并转换为FP32
        vector<__half> energies_fp16(batch_size);
        CUDA_CHECK(cudaMemcpy(energies_fp16.data(), d_energies, batch_size * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_energies));

        // 直接转换FP16到FP32
        for (int i = 0; i < batch_size; ++i) {
            energies[i] = fp16_to_float(energies_fp16[i]);
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
    BSBGPUFP16Solver solver(gset.num_vertices, batch_size);

    // 计算 FP16 自适应量化参数
    float max_weight = 1.0f;  // Gset 数据的权重范围通常是 [-1, 1]
    calculate_fp16_quantization_params(gset.num_edges, max_weight);

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

    // 求解
    cout << "Solving MAX-CUT problem..." << endl;
    cout << "  - Variables: " << gset.num_vertices << ", Edges: " << gset.num_edges << endl;
    cout << "  - BSB Algorithm (PURE FP16 - No Scaling)" << endl;
    cout << "  - FP16 Format: Binary16" << endl;
    cout << "  - Storage: __half FP16" << endl;
    cout << "  - Arithmetic: FP16 (all operations)" << endl;
    cout << "  - NO Scaling: Pure FP16 without any scaling factors" << endl;
    cout << "  - Computation strategy:" << endl;
    cout << "    - Storage: FP16 (all variables)" << endl;
    cout << "    - Dynamics (gradient, update): FP16" << endl;
    cout << "    - Cut/Energy: Pure FP16 accumulation (final sum also FP16)" << endl;
    cout << "    - NO FP32 intermediates" << endl;
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

    cout << "\nBSB FP16 CUDA Solver Results:" << endl;
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
