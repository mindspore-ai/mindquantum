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
#include "../qplib_parser.h"

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

// 手动实现clamp函数
template<typename T>
__host__ __device__ T clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// PyTorch FP8_e4m3fn格式实现（GPU版本）
// 参考: torch.float8_e4m3fn
// 格式: 1位符号(S), 4位指数(E), 3位尾数(M)
// 范围: approx [-448, 448], 精度: 2^-3

__host__ __device__ inline uint8_t float_to_fp8_e4m3fn(float f) {
    // 处理特殊值
    if (isnan(f)) {
        return 0x80;  // NaN (S=1, E=15, M=0)
    }
    if (isinf(f)) {
        return signbit(f) ? 0xFF : 0x7F;  // Inf (E=15, M=7)
    }
    if (f == 0.0f) {
        return signbit(f) ? 0x80 : 0x00;  // +0 or -0
    }

    // 获取IEEE 754 FP32的位模式
    union {
        float f;
        uint32_t u;
    } converter;
    converter.f = f;
    uint32_t bits = converter.u;

    int sign_bit = (bits >> 31) & 0x1;
    int exp32 = (bits >> 23) & 0xFF;
    uint32_t mant32 = bits & 0x7FFFFF;

    // 计算真实指数 (减去FP32偏置127)
    int true_exp = exp32 - 127;

    // E4M3FN范围检查
    const int FP8_EXP_BIAS = 7;
    const int FP8_MAX_EXP = 7;
    const int FP8_MIN_EXP = -7;

    if (true_exp > FP8_MAX_EXP) {
        // 溢出，返回Inf
        return sign_bit ? 0xFF : 0x7F;
    }

    if (true_exp < FP8_MIN_EXP) {
        // 下溢到零 (E4M3FN不支持subnormal)
        return sign_bit ? 0x80 : 0x00;
    }

    // 正常值转换到E4M3FN格式
    int exp8 = true_exp + FP8_EXP_BIAS;
    exp8 = max(0, min(15, exp8));  // 限制在[0,15]

    // FP8尾数: 保留高3位
    uint8_t mant8 = (mant32 >> 20) & 0x07;

    // 组合: S(1) + E(4) + M(3)
    return (sign_bit << 7) | (exp8 << 3) | mant8;
}

__host__ __device__ inline float fp8_to_float_e4m3fn(uint8_t fp8_val) {
    // 解析E4M3FN格式
    int sign_bit = (fp8_val >> 7) & 0x01;
    int exp8 = (fp8_val >> 3) & 0x0F;
    uint8_t mant8 = fp8_val & 0x07;

    // 特殊值处理
    if (exp8 == 0 && mant8 == 0) {
        // Zero
        return sign_bit ? -0.0f : 0.0f;
    }
    if (exp8 == 15) {
        if (mant8 == 0) {
            // Inf
            return sign_bit ? -INFINITY : INFINITY;
        } else {
            // NaN
            return sign_bit ? -NAN : NAN;
        }
    }

    // 正常值: 解码为FP32
    const int FP8_EXP_BIAS = 7;
    int true_exp = exp8 - FP8_EXP_BIAS;

    // 构建FP32位模式
    uint32_t fp32_exp = true_exp + 127;
    uint32_t fp32_mant = mant8 << 20;  // 3位扩展到23位

    uint32_t fp32_bits = (sign_bit << 31) | (fp32_exp << 23) | fp32_mant;

    union {
        float f;
        uint32_t u;
    } converter;
    converter.u = fp32_bits;
    return converter.f;
}

// 优化的符号计算函数（基于PyTorch E4M3FN）
__host__ __device__ inline int sign_from_fp8_e4m3fn(uint8_t x) {
    float x_val = fp8_to_float_e4m3fn(x);
    return (x_val > 0.0f) ? 1 : 0;
}

// 保留旧函数名的兼容性包装
__host__ __device__ inline uint8_t float_to_fp8(float f) {
    return float_to_fp8_e4m3fn(f);
}

__host__ __device__ inline float fp8_to_float(uint8_t fp8_val) {
    return fp8_to_float_e4m3fn(fp8_val);
}

__host__ __device__ inline uint8_t float_to_fp8_optimized(float f) {
    return float_to_fp8_e4m3fn(f);
}

__host__ __device__ inline float fp8_to_float_optimized(uint8_t fp8_val) {
    return fp8_to_float_e4m3fn(fp8_val);
}

__host__ __device__ inline int sign_from_fp8_optimized(uint8_t x) {
    return sign_from_fp8_e4m3fn(x);
}

__host__ __device__ inline uint8_t clamp_fp8_bsb(float f) {
    return float_to_fp8_e4m3fn(f);
}

// GPU批处理数据结构（Float版本：全部使用float计算）
// 关键特征：全部使用float存储和计算，不使用double
struct GPUBatchData {
    float* x;               // 位置状态（float，存储和计算）
    float* y;              // 动量状态（float，计算）
    float* gradient_term;  // 梯度项（float，计算）
    float energy;          // 目标函数值（float）
};

// GPU上的BSB模型数据（Float版本）
struct GPUBSBModel {
    int* second_order_terms;       // 二阶项 [num_second_order * 2]
    float* second_order_coeff;    // 二阶项系数（float，计算）
    float* one_order_coeff;       // 一阶项系数（float，计算）
    int num_spins;
    int num_second_order;
};

// GPU内核函数：初始化批处理数据（Float版本：全部使用float）
// BSB特征：全部使用float存储和计算
__global__ void init_batches_kernel(float* x, float* y, float* gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size, float sigma = 0.1f) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;

    // 初始化：全部使用float
    for (int i = 0; i < num_spins; ++i) {
        // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5
        float x_normalized = sigma * curand_normal_double(&state) + 0.5f;  // [0,1]范围
        x_normalized = fmaxf(0.0f, fminf(1.0f, x_normalized));        // 限制在[0,1]

        // 转换到[-1.0,1.0]范围并存储为float
        float x_range = x_normalized * 2.0f - 1.0f;  // [0,1] -> [-1.0,1.0]
        x[batch_offset + i] = x_range;  // 存储为float

        // y初始化为小随机值：sigma * N(0,1)（float精度）
        float y_val = sigma * curand_normal_double(&state);
        y[batch_offset + i] = y_val;

        // 初始化梯度项（float精度）
        gradient_term[batch_offset + i] = 0.0f;
    }
}



// 优化后的梯度计算内核：Float版本（全部使用float）
// BSB特征：全部使用float存储和计算
__global__ void calc_gradient_kernel_optimized(float* x, float* gradient_term,
                                               int* second_order_terms, float* second_order_coeff, float* one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 初始化梯度为一阶项系数（float精度）
    float grad = one_order_coeff[spin_idx];

    // 计算当前自旋的二阶项贡献（float精度），使用float x
    for (int t = 0; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        float J_ij = second_order_coeff[t] * 0.5f;  // 系数除以2，匹配CPU版本

        if (var1 == spin_idx) {
            float x2_val = x[batch_offset + var2];
            float x2_continuous = (x2_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1]
            grad += J_ij * x2_continuous;
        } else if (var2 == spin_idx) {
            float x1_val = x[batch_offset + var1];
            float x1_continuous = (x1_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1]
            grad += J_ij * x1_continuous;
        }
    }

    gradient_term[batch_offset + spin_idx] = grad;  // 存储为float精度
}

// 优化后的y更新内核：Float版本（全部使用float）
// BSB公式：dy/dt = -(1-p(t))*(x-0.5) ± ξ*∇f(x)
// 关键特征：全部使用float存储和计算
__global__ void update_y_kernel_optimized(float* x, float* y, float* gradient_term,
                                          int num_spins, float p, float xi, float dt, int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 泵浦振幅项：-(1-p(t))，与CPU版本完全匹配
    const float pump_factor = -(1.0f - p);

    // 将x转换到[0,1]范围（float）
    float x_val = x[batch_offset + spin_idx];  // float [-1.0,1.0]
    float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1]
    float x_shifted = x_normalized - 0.5f;

    // BSB公式：ẏ_i = [-(1-p(t))*(x-0.5) ± ξ*∇f(x)] * dt
    float gradient_term_val;
    if (is_maximize) {
        gradient_term_val = xi * gradient_term[batch_offset + spin_idx];
    } else {
        gradient_term_val = -xi * gradient_term[batch_offset + spin_idx];
    }

    // 更新动量（float精度）
    float dy = (pump_factor * x_shifted + gradient_term_val) * dt;
    y[batch_offset + spin_idx] += dy;
}

// 优化后的x更新内核：Float版本（全部使用float）
__global__ void update_x_kernel_optimized(float* x, float* y,
                                          int num_spins, float dt, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 将x转换到[0,1]空间进行更新（float）
    float x_val = x[batch_offset + spin_idx];  // float [-1.0,1.0]
    float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1]
    x_normalized = fmaxf(0.0f, fminf(1.0f, x_normalized));  // 确保在[0,1]

    // 位置更新：x += y * dt（在float空间）
    x_normalized += y[batch_offset + spin_idx] * dt;

    // 更新为float：先限制在[0,1]，再转换到[-1.0,1.0]
    x_normalized = fmaxf(0.0f, fminf(1.0f, x_normalized));
    float x_new = x_normalized * 2.0f - 1.0f;  // [0,1] -> [-1.0,1.0]
    x[batch_offset + spin_idx] = x_new;  // 存储为float
}

// 边界约束内核：Float版本（全部使用float）
// 将x限制在[0,1]范围内，超出时投影到边界{0,1}并清零动量
__global__ void apply_constraint_kernel(float* x, float* y, int num_spins, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    for (int i = 0; i < num_spins; ++i) {
        // 将x转换到[0,1]范围检查边界（float）
        float x_val = x[batch_offset + i];  // float [-1.0,1.0]
        float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1]

        // 检查是否超出边界（在float精度下检查）
        if (x_normalized < 0.0f || x_normalized > 1.0f) {
            // 超出边界，投影到极值：0 -> -1.0, 1 -> 1.0
            float projected_value = (x_normalized < 0.5f) ? 0.0f : 1.0f;  // 投影到{0,1}
            float projected_range = projected_value * 2.0f - 1.0f;  // [0,1] -> [-1.0,1.0]
            x[batch_offset + i] = projected_range;

            // 清零动量（非弹性墙，float精度）
            y[batch_offset + i] = 0.0f;
        }
    }
}

// 优化的能量计算内核：Float版本（全部使用float）
__global__ void calculate_energy_kernel_optimized(float* x,
                                            int* second_order_terms, float* second_order_coeff,
                                            float* one_order_coeff, float* energies,
                                            int num_spins, int num_second_order, int batch_size,
                                            float scaling_factor, float constant_term, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 使用共享内存进行高效计算
    __shared__ float shared_energy[256];
    __shared__ float shared_quad_energy[256];

    // 初始化共享内存
    shared_energy[thread_idx] = 0.0f;
    shared_quad_energy[thread_idx] = 0.0f;
    __syncthreads();

    // 线性项贡献：使用连续变量
    for (int i = thread_idx; i < num_spins; i += blockDim.x) {
        float coeff = one_order_coeff[i];
        float x_val = x[batch_offset + i];  // float [-1.0,1.0]
        float x_continuous = (x_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1] 连续变量
        shared_energy[thread_idx] += coeff * x_continuous;
    }

    // 二次项贡献：使用连续变量
    for (int t = thread_idx; t < num_second_order; t += blockDim.x) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        float J_ij = second_order_coeff[t];

        float x1_val = x[batch_offset + var1];
        float x2_val = x[batch_offset + var2];
        float x1_continuous = (x1_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1] 连续变量
        float x2_continuous = (x2_val + 1.0f) / 2.0f;  // [-1.0,1.0] -> [0,1] 连续变量
        shared_quad_energy[thread_idx] += 0.5f * J_ij * x1_continuous * x2_continuous;
    }

    __syncthreads();

    // 高效的归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_energy[thread_idx] += shared_energy[thread_idx + s];
            shared_quad_energy[thread_idx] += shared_quad_energy[thread_idx + s];
        }
        __syncthreads();
    }

    // 最终能量计算
    if (thread_idx == 0) {
        float total_energy = shared_energy[0] + shared_quad_energy[0];

        // 缩放处理
        if (num_second_order != 0) {
            total_energy *= scaling_factor;
        }

        // 添加常数项
        total_energy += constant_term;

        energies[batch_idx] = total_energy;
    }
}

// BSB求解器类（FP8版本，Float精度）
class BSBGPUFP8Solver {
private:
    ProblemModel model;
    int batch_size;
    
    // GPU设备指针（Float版本）
    float* d_x;               // 位置状态 [batch_size * num_spins] (float，存储和计算)
    float* d_y;              // 动量状态 [batch_size * num_spins] (float，计算)
    float* d_gradient_term;   // 梯度项 [batch_size * num_spins] (float，计算)
    float* d_energies;       // 能量值 [batch_size] (float，计算)
    
    // 模型数据（Float版本）
    int* d_second_order_terms;      // 二阶项 [num_second_order * 2]
    float* d_second_order_coeff;   // 二阶项系数 [num_second_order] (float，计算)
    float* d_one_order_coeff;       // 一阶项系数 [num_spins] (float，计算)

public:
    BSBGPUFP8Solver(const ProblemModel& problem_model, int bs = 100) : model(problem_model), batch_size(bs) {
        prepare_gpu_model();
    }

    ~BSBGPUFP8Solver() {
        free_gpu_memory();
    }

    // 准备GPU模型数据（Float版本）
    void prepare_gpu_model() {
        cout << "DEBUG: Preparing Float-precision FP8_e4m3fn GPU model..." << endl;

        // 展平二阶项
        vector<int> flat_second_order_terms;
        for (int i = 0; i < model.num_quad_terms; ++i) {
            flat_second_order_terms.push_back(model.quad_terms_indices[i][0]);
            flat_second_order_terms.push_back(model.quad_terms_indices[i][1]);
        }

        cout << "DEBUG: Using Float precision: PyTorch FP8_e4m3fn for storage, float for computation" << endl;
        cout << "DEBUG: PyTorch FP8_e4m3fn: IEEE 754-like (1S+4E+3M), range [-448, 448]" << endl;
        
        // 分配GPU内存（Float精度架构）
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_second_order_terms, model.num_quad_terms * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, model.num_quad_terms * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, model.num_vars * sizeof(float)));
        
        // 复制模型数据到GPU（转换为float精度）
        vector<float> float_second_order_coeff;
        float_second_order_coeff.reserve(model.quad_terms_coeffs.size());
        for (size_t i = 0; i < model.quad_terms_coeffs.size(); ++i) {
            float_second_order_coeff.push_back(static_cast<float>(model.quad_terms_coeffs[i]));
        }
        vector<float> float_one_order_coeff;
        float_one_order_coeff.reserve(model.linear_terms.size());
        for (size_t i = 0; i < model.linear_terms.size(); ++i) {
            float_one_order_coeff.push_back(static_cast<float>(model.linear_terms[i]));
        }
        
        CUDA_CHECK(cudaMemcpy(d_second_order_terms, flat_second_order_terms.data(), 
                              flat_second_order_terms.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_second_order_coeff, float_second_order_coeff.data(), 
                              model.num_quad_terms * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_one_order_coeff, float_one_order_coeff.data(), 
                              model.num_vars * sizeof(float), cudaMemcpyHostToDevice));
        
        cout << "DEBUG: Float-precision PyTorch FP8_e4m3fn GPU model prepared successfully." << endl;
    }

    // 释放GPU内存
    void free_gpu_memory() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_energies));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
    }

    // 核心求解函数（FP8版本，Float精度）
    vector<uint8_t> solve(int max_iters = 1000, float optimized_xi = 0.365f) {
        // BSB超参数（Float精度）
        float xi = optimized_xi;
        float dt = 0.1f;

        // 确定优化方向
        bool is_maximize = (model.objective_sense == "maximize");

        // 计算线程块和网格大小
        dim3 blockSize(256);
        dim3 gridSize_batch_spins(batch_size, (model.num_vars + blockSize.x - 1) / blockSize.x);

        // 初始化批处理数据
        init_batches_kernel<<<gridSize_batch_spins, blockSize>>>(
            d_x, d_y, d_gradient_term,
            model.num_vars, model.num_quad_terms,
            time(nullptr), batch_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时开始
        auto start = high_resolution_clock::now();

        // 核心迭代（Float精度BSB动力学）
        for (int iter = 0; iter < max_iters; ++iter) {
            float p = static_cast<float>(iter) / (max_iters - 1);

            // BSB算法：直接使用连续变量x计算梯度
            calc_gradient_kernel_optimized<<<gridSize_batch_spins, blockSize>>>(d_x, d_gradient_term,
                                                                      d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                                                                      model.num_vars, model.num_quad_terms, batch_size);

            // 更新动量：dy/dt = -(1-p(t))*(x-0.5) ± ξ*∇f(x)
            update_y_kernel_optimized<<<gridSize_batch_spins, blockSize>>>(d_x, d_y, d_gradient_term,
                                                                model.num_vars, p, xi, dt, batch_size, is_maximize);

            // 更新位置：dx/dt = y
            update_x_kernel_optimized<<<gridSize_batch_spins, blockSize>>>(d_x, d_y, model.num_vars, dt, batch_size);

            // 边界条件处理：将x限制在[0,1]范围内，超出时投影到边界并清零动量
            apply_constraint_kernel<<<gridSize_batch_spins, blockSize>>>(d_x, d_y, model.num_vars, batch_size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 计时结束
        auto end = high_resolution_clock::now();
        float duration = duration_cast<microseconds>(end - start).count() / 1000.0f;

        // 在GPU端计算所有批次的能量以选择最终最优解
        float* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));

        // 调用GPU能量计算内核
        dim3 energy_block_size(256);
        dim3 energy_grid_size(batch_size);
        calculate_energy_kernel_optimized<<<energy_grid_size, energy_block_size>>>(
            d_x, d_second_order_terms, d_second_order_coeff,
            d_one_order_coeff, d_energies,
            model.num_vars, model.num_quad_terms, batch_size,
            static_cast<float>(model.scaling_factor), static_cast<float>(model.constant_term), is_maximize
        );
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 只复制能量数据到主机端，而不是所有的x数据
        vector<float> energies(batch_size);
        CUDA_CHECK(cudaMemcpy(energies.data(), d_energies, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

        // 释放GPU能量存储
        CUDA_CHECK(cudaFree(d_energies));

        // 找到最优能量对应的批次索引
        int best_idx;

        // 根据问题类型选择最优解
        if (model.objective_sense == "maximize") {
            auto max_it = max_element(energies.begin(), energies.end());
            best_idx = max_it - energies.begin();
        } else {
            auto min_it = min_element(energies.begin(), energies.end());
            best_idx = min_it - energies.begin();
        }

        // 只复制最优批次的x数据到主机端（float格式）
        vector<float> best_host_x(model.num_vars);
        CUDA_CHECK(cudaMemcpy(best_host_x.data(), d_x + best_idx * model.num_vars,
                              model.num_vars * sizeof(float), cudaMemcpyDeviceToHost));

        // 转换为FP8返回
        vector<uint8_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            solution[i] = float_to_fp8_e4m3fn(best_host_x[i]);
        }

        return solution;
    }

    int get_num_vars() const {
        return model.num_vars;
    }
};

// 主函数
int main(int argc, char* argv[]) {
    // 处理命令行参数 - 改进路径查找
    string qplib_file_path;
    if (argc > 1) {
        // 尝试从不同位置查找输入文件
        vector<string> possible_paths = {
            argv[1],
            "qplib\\" + string(argv[1]),
            "../qplib\\" + string(argv[1]),
            "../../qplib\\" + string(argv[1]),
            "../../../qplib\\" + string(argv[1]),
            "qplib/" + string(argv[1]),
            "../qplib/" + string(argv[1]),
            "../../qplib/" + string(argv[1]),
            "../../../qplib/" + string(argv[1])
        };
        
        for (const auto& path : possible_paths) {
            ifstream test_file(path);
            if (test_file.good()) {
                qplib_file_path = path;
                break;
            }
        }
        
        if (qplib_file_path.empty()) {
            qplib_file_path = argv[1];
        }
    } else {
        // 对于默认文件，也要尝试多个路径
        vector<string> default_paths = {
            "qplib/QPLIB_3565.qplib",
            "../qplib/QPLIB_3565.qplib",
            "../../qplib/QPLIB_3565.qplib",
            "../../../qplib/QPLIB_3565.qplib",
            "qplib\\QPLIB_3565.qplib",
            "../qplib\\QPLIB_3565.qplib",
            "../../qplib\\QPLIB_3565.qplib",
            "../../../qplib\\QPLIB_3565.qplib"
        };
        
        for (const auto& path : default_paths) {
            ifstream test_file(path);
            if (test_file.good()) {
                qplib_file_path = path;
                break;
            }
        }
        
        if (qplib_file_path.empty()) {
            qplib_file_path = DEFAULT_QPLIB_FILE_PATH;
        }
    }
    
    cout << "尝试解析文件: " << qplib_file_path << endl;

    cout << "========================================" << endl;
    cout << "       BSB_GPU_fp8 (Float Precision)  " << endl;
    cout << "  - File path: " << qplib_file_path << endl;
    cout << "  - Algorithm: BSB (Ballistic Simulated Bifurcation)" << endl;
    cout << "  - Key feature: Float for storage and computation (no double)" << endl;
    cout << "  - Precision: Float (storage and computation)" << endl;
    cout << "  - Energy method: Continuous variables" << endl;
    cout << "  - CUDA parallel batch processing optimized" << endl;
    cout << "========================================" << endl << endl;

    // 解析指定QPLIB文件
    ProblemModel model;
    cout << "Start parsing file: " << qplib_file_path << endl;
    bool parse_result = QPLIBParser::parseQplibFile(qplib_file_path, model);
    cout << "DEBUG: parseQplibFile returned: " << (parse_result ? "true" : "false") << endl;
    cout.flush();
    
    if (!parse_result) {
        cerr << "ERROR: File parsing failed, program exit" << endl;
        return 1;
    }
    
    cout << "DEBUG: Successfully completed QPLIB parsing!" << endl;
    cout.flush();
    
    // 添加详细的模型信息调试输出
    cout << endl << "MODEL DEBUG INFO:" << endl;
    cout << "================" << endl;
    cout << "num_vars: " << model.num_vars << endl;
    cout << "num_constraints: " << model.num_constraints << endl;
    cout << "num_quad_terms: " << model.num_quad_terms << endl;
    cout << "quad_terms_indices.size(): " << model.quad_terms_indices.size() << endl;
    cout << "quad_terms_coeffs.size(): " << model.quad_terms_coeffs.size() << endl;
    cout << "linear_terms.size(): " << model.linear_terms.size() << endl;
    cout << "num_linear_terms: " << model.num_linear_terms << endl;
    cout << "constant_term: " << model.constant_term << endl;
    cout << "objective_sense: " << model.objective_sense << endl;
    cout << "scaling_factor: " << model.scaling_factor << endl;
    cout << "xi: " << model.xi << endl;
    cout << "================" << endl << endl;
    cout.flush();
    
    // 安全检查：确保linear_terms大小与变量数量一致
    if (model.linear_terms.size() != model.num_vars) {
        cout << "WARNING: Linear terms size (" << model.linear_terms.size() 
             << ") does not match number of variables (" << model.num_vars << ")" << endl;
        cout << "INFO: Resizing linear_terms to match variable count" << endl;
        model.linear_terms.resize(model.num_vars, 0.0);
    }

    // 输出导入文件的信息
    cout << "\n========================================" << endl;
    cout << "  IMPORTED FILE INFORMATION" << endl;
    cout << "========================================" << endl;
    
    cout << "问题模型: " << model.problem_name << endl;
    cout << "  类型: " << model.problem_type << endl;
    cout << "  优化方向: " << model.objective_sense << endl;
    cout << "  变量数量: " << model.num_vars << endl;
    cout << "  约束数量: " << model.num_constraints << endl;
    cout << "  二次项数量: " << model.num_quad_terms << endl;
    cout << "  线性项数量: " << model.num_linear_terms << endl;
    cout << "  等式约束数量: " << model.num_equal_constraints << endl;
    cout << "  不等式约束数量: " << model.num_inequal_constraints << endl;
    cout << "  缩放因子: " << fixed << setprecision(6) << model.scaling_factor << endl;
    cout << "  耦合强度参数: " << fixed << setprecision(6) << model.xi << endl;
    
    cout << "========================================" << endl;
    cout.flush();  // 强制刷新输出缓冲区
    
    // 继续执行BSB求解
    cout << "\n开始执行BSB求解..." << endl;
    
    // 检查是否适合使用BSB求解器
    if (model.num_quad_terms == 0) {
        cout << "INFO: No quadratic terms found, this problem is linear." << endl;
        cout << "INFO: BSB is designed for quadratic optimization. For linear problems, consider using linear programming solvers." << endl;
        cout << "INFO: Running BSB anyway for demonstration..." << endl;
    }

    cout << endl << "DEBUG: About to initialize GPU solver..." << endl;
    cout.flush();
    
    try {
        cout << "DEBUG: Creating BSB GPU FP8 solver with 100 batches..." << endl;
        cout.flush();
        
        // 直接使用ProblemModel进行GPU计算
        BSBGPUFP8Solver solver(model, 100); // 100 batches
        cout << "DEBUG: GPU FP8 solver created successfully!" << endl;
        cout.flush();
        
        cout << "Starting GPU FP8 optimization..." << endl;
        cout.flush();

        // 使用Float精度参数：1000次迭代，优化xi=0.365
        auto start = high_resolution_clock::now();
        vector<uint8_t> fp8_solution = solver.solve(1000, 0.365f);
        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;
        cout << "TIME: Solving completed, total time: " << fixed << setprecision(6) << duration << " ms" << endl;
        cout.flush();

        // 计算目标函数值（使用连续变量，与CPU FP8一致）
        double objective_value = 0.0;
        for (size_t i = 0; i < fp8_solution.size(); ++i) {
            uint8_t x_val = fp8_solution[i];
            float x_float_val = fp8_to_float_e4m3fn(x_val);  // FP8 -> float [-1.0,1.0]
            double x_continuous = (static_cast<double>(x_float_val) + 1.0) / 2.0;  // [-1.0,1.0] -> [0,1] 连续变量
            objective_value += model.linear_terms[i] * x_continuous;
        }

        for (int t = 0; t < model.num_quad_terms; ++t) {
            int var1 = model.quad_terms_indices[t][0];
            int var2 = model.quad_terms_indices[t][1];
            double coeff = model.quad_terms_coeffs[t];

            uint8_t x1_val = fp8_solution[var1];
            uint8_t x2_val = fp8_solution[var2];
            float x1_float_val = fp8_to_float_e4m3fn(x1_val);
            float x2_float_val = fp8_to_float_e4m3fn(x2_val);
            double x1_continuous = (static_cast<double>(x1_float_val) + 1.0) / 2.0;  // [-1.0,1.0] -> [0,1] 连续变量
            double x2_continuous = (static_cast<double>(x2_float_val) + 1.0) / 2.0;  // [-1.0,1.0] -> [0,1] 连续变量

            objective_value += 0.5 * coeff * x1_continuous * x2_continuous;
        }
        
        objective_value = objective_value * model.scaling_factor + model.constant_term;

        // 输出FP8解表示（[0,255]，真正的fp8值）
        cout << "\nRESULT: Solution (" << model.num_vars << " variables, FP8_e4m3fn values [0,255]): " << endl;
        int count_ones = 0;
        for (size_t i = 0; i < fp8_solution.size(); ++i) {
            uint8_t x_val = fp8_solution[i];  // 直接输出fp8值
            int display_val = static_cast<int>(x_val);  // 转换为int以便正确显示
            cout << display_val;
            if (sign_from_fp8_e4m3fn(x_val) == 1) count_ones++;  // 使用PyTorch符号计算统计二进制1的数量
            if ((i + 1) % 16 == 0) cout << endl;
            else if (i != fp8_solution.size() - 1) cout << ", ";
        }
        cout << endl << endl;
        
        cout << "Solution statistics (continuous variables, energy calculation uses [0,1] range):" << endl;
        cout << "  - Positive values (x > 0): " << count_ones << "/" << fp8_solution.size() << endl;
        cout << "  - Non-positive values (x ≤ 0): " << (fp8_solution.size() - count_ones) << "/" << fp8_solution.size() << endl;
        cout << "  - Density of positive values: " << fixed << setprecision(2) << (count_ones * 100.0 / fp8_solution.size()) << "%" << endl;
        cout << endl;

        // 输出最终目标函数值（更醒目的格式）
        cout << "========================================" << endl;
        cout << "  ** BSB GPU Float Precision FINAL RESULTS **" << endl;
        cout << "========================================" << endl;
        cout << "  Problem: " << model.problem_name << endl;
        cout << "  Objective sense: " << model.objective_sense << endl;
        cout << "  Number of variables: " << model.num_vars << endl;
        cout << "  Number of quadratic terms: " << model.num_quad_terms << endl;
        cout << "----------------------------------------" << endl;
        cout << "  BSB OBJECTIVE VALUE: " << fixed << setprecision(6) << objective_value << endl;
        cout << "  Precision: Float (storage and computation, no double)" << endl;
        cout << "========================================" << endl;
        
        // 强制刷新输出缓冲区
        cout.flush();
        
    } catch (const exception& e) {
        cerr << "ERROR: Exception during GPU FP8 solver initialization: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "ERROR: Unknown exception during GPU FP8 solver initialization" << endl;
        return 1;
    }
    
    return 0;
}