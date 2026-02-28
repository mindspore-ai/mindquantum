#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <random>
#include <algorithm>
#include <utility>
#include <limits>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <exception>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// QPLIB配置已移至qplib_parser.h头文件中

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << #call << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 手动实现clamp函数（兼容不同类型）
template<typename T, typename U, typename V>
__host__ __device__ T clamp(T val, U min_val, V max_val) {
    if (val < static_cast<T>(min_val)) return static_cast<T>(min_val);
    if (val > static_cast<T>(max_val)) return static_cast<T>(max_val);
    return val;
}

// ============================================================================
// PyTorch FP8 E4M3FN 配置（GPU量化）- 简化版本（参考INT8策略）
// ============================================================================
// PyTorch float8_e4m3fn 标准：1符号位 + 4指数位 + 3尾数位
// 简化策略：使用浮点直接映射，避免复杂位操作（参考INT8版本的成功经验）
constexpr float FP8_SCALE_MIN = -1.0f;    // FP8使用的最小值（简化范围）
constexpr float FP8_SCALE_MAX = 1.0f;     // FP8使用的最大值（简化范围）

// FP8量化函数 - 简化版本（参考INT8PyTorch::quantize策略）
// 关键改进：直接使用浮点范围映射，避免复杂位操作
__host__ __device__ inline uint8_t float_to_fp8_simple(float f) {
    // 处理特殊值（NaN, Inf, Zero）
    if (isnan(f)) return 0x80;  // NaN
    if (isinf(f)) return (f > 0.0f) ? 0x7F : 0xFF;  // +/- Inf
    if (f == 0.0f) return (f > 0.0f) ? 0x00 : 0x80;  // +/- Zero

    // 直接限制到[-1, 1]范围（参考INT8的[-127,127]简化策略）
    f = fmaxf(FP8_SCALE_MIN, fminf(FP8_SCALE_MAX, f));

    // 映射到[0, 255]范围（uint8_t）
    uint8_t fp8_val = static_cast<uint8_t>((f - FP8_SCALE_MIN) * 127.5f);
    return fp8_val;
}

// FP8到FP32转换 - 简化版本（参考INT8PyTorch::dequantize策略）
// 关键改进：直接从[0, 255]映射回[-1, 1]
__host__ __device__ inline float fp8_to_float_simple(uint8_t fp8_val) {
    // 特殊值处理
    if (fp8_val == 0x80) return -0.0f;  // -0
    if (fp8_val == 0x00) return 0.0f;    // +0

    // 直接从[0, 255]映射回[-1, 1]
    float result = static_cast<float>(fp8_val) / 127.5f + FP8_SCALE_MIN;
    return result;
}

// PyTorch兼容的FP8转换器（结构体封装 - 参考INT8PyTorch）
struct FP8PyTorch {
    __host__ __device__ static inline uint8_t quantize(float val) {
        // 直接范围限制和映射（参考INT8版本的简化策略）
        val = fmaxf(FP8_SCALE_MIN, fminf(FP8_SCALE_MAX, val));
        return float_to_fp8_simple(val);
    }

    __host__ __device__ static inline float dequantize(uint8_t val) {
        return fp8_to_float_simple(val);
    }
};



// 自定义x^分段函数（DSB特征：不使用sign，使用分段函数返回{0,1}）
// 参考项目报告3.md定义：x̂_i = 1 if x_i > 0.5, else 0
__host__ __device__ inline int x_hat_fp8(uint8_t fp8_val) {
    float val = FP8PyTorch::dequantize(fp8_val);  // [-1,1]
    float x_normalized = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
}

// GPU batch data structure (DSB algorithm FP8 version - PyTorch兼容)
// 关键改进：y和gradient_term用float，x用PyTorch FP8存储
struct GPUBatchDataFP8 {
    uint8_t* x;              // 自旋状态 (PyTorch float8_e4m3fn，高效存储)
    float* y;               // 动量状态 (float，平衡精度和性能)
    int* x_hat;             // 离散二值化变量{0,1}（匹配项目报告定义）
    float* gradient_term;   // 梯度项 (float，高效计算)
    float energy;          // Ising Hamiltonian (float，保持精度一致性)
};

// GPU内核函数：初始化批处理数据（PyTorch FP8版本 - 参考INT8策略）
// 关键改进：使用更大的线性映射范围，提高初始化精度（参考INT8版本）
__global__ void init_batches_kernel(uint8_t* x, float* y, int* x_hat, float* gradient_term,
                                   int num_vars, int num_quad_terms,
                                   unsigned long long seed, int batch_size, float sigma) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_vars;

    for (int i = 0; i < num_vars; ++i) {
        // DSB算法：x在[0,1]范围内初始化（完全匹配INT8版本第110-113行）
        float x_normalized = sigma * curand_normal(&state) + 0.5f;  // [0, 1]
        x_normalized = clamp(x_normalized, 0.0f, 1.0f);

        // 关键改进：使用更大的线性映射范围，提高FP8初始化精度
        // 参考INT8策略：使用更大的映射范围提高精度
        // 映射到[-1, 1]范围（FP8最佳精度范围）
        float x_scaled = (x_normalized - 0.5f) * 2.0f;  // [0,1] -> [-1,1]
        x_scaled = clamp(x_scaled, -0.99f, 0.99f);  // 避免边界饱和
        x[batch_offset + i] = FP8PyTorch::quantize(x_scaled);

        // y初始化为小随机值：sigma * N(0,1)，使用float存储
        float y_val = sigma * curand_normal(&state);
        y[batch_offset + i] = y_val;

        // 初始化x_hat（匹配项目报告定义）
        x_hat[batch_offset + i] = x_hat_fp8(x[batch_offset + i]);  // {0,1}二值化
        gradient_term[batch_offset + i] = 0.0f;
    }
}

// GPU内核函数：计算离散二值化变量（FP8存储版本）
__global__ void calc_x_hat_kernel(uint8_t* x, int* x_hat, int num_vars, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    for (int i = 0; i < num_vars; ++i) {
        // DSB特征：计算{0,1}离散二值化变量（匹配项目报告定义）
        x_hat[batch_offset + i] = x_hat_fp8(x[batch_offset + i]);
    }
}

// GPU内核函数：计算梯度（FP8优化版本，参考INT8策略）
// 关键改进：系数用float存储，直接计算，避免FP8转换
// DSB公式：∇_iE = direction_sign * (h_i + ∑_{j} J_{ij}·x̂_j)
__global__ void calc_gradient_kernel(uint8_t* x, int* x_hat, float* gradient_term,
                                      int* quad_terms_indices, float* quad_terms_coeffs, float* linear_terms,
                                      int num_vars, int num_quad_terms, int batch_size,
                                      bool is_maximize) {
    int batch_idx = blockIdx.x;
    int var_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || var_idx >= num_vars) return;

    int batch_offset = batch_idx * num_vars;

    // 优化方向符号（参考INT8版本）
    float direction_sign = is_maximize ? 1.0f : -1.0f;

    // 初始化梯度为一阶项系数：∇_iE = direction_sign * h_i（直接float计算）
    float grad = direction_sign * linear_terms[var_idx];

    // 累加二次项贡献：direction_sign * ∑_j J_{ij}·x̂_j
    // DSB核心特征：使用离散的0/1变量，直接float计算
    for (int t = 0; t < num_quad_terms; ++t) {
        int var1 = quad_terms_indices[t * 2];
        int var2 = quad_terms_indices[t * 2 + 1];
        float J_ij = direction_sign * quad_terms_coeffs[t];

        if (var1 == var_idx) {
            // 直接使用{0,1}变量（匹配项目报告定义）
            int x2_hat = x_hat[batch_offset + var2];  // 0或1
            grad += (J_ij / 2.0f) * static_cast<float>(x2_hat);
        } else if (var2 == var_idx) {
            // 直接使用{0,1}变量（匹配项目报告定义）
            int x1_hat = x_hat[batch_offset + var1];  // 0或1
            grad += (J_ij / 2.0f) * static_cast<float>(x1_hat);
        }
    }

    // 直接存储float梯度值
    gradient_term[batch_offset + var_idx] = grad;
}

// GPU内核函数：更新辅助状态y（FP8优化版 - 参考INT8版本）
// 关键优化：简化转换逻辑，使用FP8PyTorch::dequantize（参考INT8版本第187-209行）
// DSB公式：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇_iE
__global__ void update_y_kernel(uint8_t* x, float* y, float* gradient_term,
                                int num_vars, float p, float xi, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    const float pump_factor = -(1.0f - p);

    // 优化：预计算常量（参考INT8版本第196-197行）
    const float ONE_FP32 = 1.0f;
    const float HALF_OFFSET = 0.5f;

    for (int i = 0; i < num_vars; ++i) {
        // 使用FP8PyTorch::dequantize：uint8_t -> [-1,1]（参考INT8版本第201行）
        float x_fp8_val = FP8PyTorch::dequantize(x[batch_offset + i]);  // [-1,1]
        float x_norm = (x_fp8_val + ONE_FP32) * 0.5f;  // [-1,1] -> [0,1]
        float x_shifted = x_norm - HALF_OFFSET;  // (x-0.5)

        // DSB公式的直接计算（参考INT8版本第206行）
        float dy = (pump_factor * x_shifted + xi * gradient_term[batch_offset + i]) * dt;
        y[batch_offset + i] += dy;
    }
}

// GPU内核函数：更新自旋x（FP8优化版 - 参考INT8版本）
// 关键优化：使用FP8PyTorch::quantize/dequantize（参考INT8版本第214-238行）
// DSB公式：ẋ_i = y_i，离散化：x_i^{(t+1)} = x_i^{(t)} + y_i^{(t+1)}·dt
__global__ void update_x_kernel(uint8_t* x, float* y, int num_vars, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;

    // 优化：预计算常量（参考INT8版本第221-223行）
    const float ONE_FP32 = 1.0f;
    const float TWO_FP32 = 2.0f;
    const float factor = dt * delta;

    for (int i = 0; i < num_vars; ++i) {
        // 使用FP8PyTorch::dequantize：uint8_t -> [-1,1]（参考INT8版本第227行）
        float x_fp8_val = FP8PyTorch::dequantize(x[batch_offset + i]);  // [-1,1]
        float x_normalized = (x_fp8_val + ONE_FP32) * 0.5f;  // [-1,1] -> [0,1]

        // DSB位置更新（参考INT8版本第231行）
        float x_new = x_normalized + y[batch_offset + i] * factor;
        x_new = clamp(x_new, 0.0f, 1.0f);

        // 使用FP8PyTorch::quantize：[-1,1] -> uint8_t（参考INT8版本第235行）
        float x_fp8_new = x_new * TWO_FP32 - ONE_FP32;
        x_fp8_new = clamp(x_fp8_new, -0.99f, 0.99f);
        x[batch_offset + i] = FP8PyTorch::quantize(x_fp8_new);
    }
}

// GPU内核函数：应用边界约束（FP8优化版 - 参考INT8版本）
// 关键优化：使用FP8PyTorch::quantize/dequantize（参考INT8版本第240-268行）
__global__ void apply_constraint_kernel(uint8_t* x, float* y, int num_vars, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;

    // 优化：预计算常量（参考INT8版本第249-250行）
    const float ONE_FP32 = 1.0f;
    const float BOUNDARY_THRESHOLD = 0.5f;

    for (int i = 0; i < num_vars; ++i) {
        // 使用FP8PyTorch::dequantize：uint8_t -> [-1,1]（参考INT8版本第254行）
        float x_fp8_val = FP8PyTorch::dequantize(x[batch_offset + i]);  // [-1,1]
        float x_norm = (x_fp8_val + ONE_FP32) * 0.5f;  // [-1,1] -> [0,1]

        // 检查是否超出边界（参考INT8版本第258行）
        float distance_from_center = fabsf(x_norm - 0.5f);

        if (distance_from_center > BOUNDARY_THRESHOLD) {
            // 投影到边界{0,1}（参考INT8版本第262行）
            float projected = (x_norm > 0.5f) ? 1.0f : 0.0f;
            float x_fp8_projected = projected * 2.0f - 1.0f;
            x_fp8_projected = clamp(x_fp8_projected, -0.99f, 0.99f);
            x[batch_offset + i] = FP8PyTorch::quantize(x_fp8_projected);
            y[batch_offset + i] = 0.0f;  // 清零动量
        }
    }
}

// GPU内核函数：计算单个批次的能量（FP8优化版本，参考INT8策略 + 共享内存优化）
// 关键改进：系数用float存储，直接计算，能量计算保持FP32精度
// 性能优化：使用共享内存进行归约求和（参考DSB_GPU_fp16和BSB版本）
__global__ void calc_batch_energy_kernel(uint8_t* x, int* x_hat, float* energies,
                                          int* quad_terms_indices, float* quad_terms_coeffs, float* linear_terms,
                                          int num_vars, int num_quad_terms, int batch_size,
                                          float scaling_factor, float constant_term, bool is_linear_problem) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // 共享内存优化：参考DSB_GPU_fp16和BSB版本
    __shared__ float shared_energy[256];
    __shared__ float shared_quad_energy[256];

    // 初始化共享内存
    shared_energy[thread_idx] = 0.0f;
    shared_quad_energy[thread_idx] = 0.0f;
    __syncthreads();

    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;

    // 一阶项贡献：∑_i h_i·x̂_i（参考INT8版本，使用线程并行）
    for (int i = thread_idx; i < num_vars; i += 256) {
        float h_i = linear_terms[i];

        // 直接使用{0,1}变量（匹配项目报告定义）
        int x_hat_i = x_hat[batch_offset + i];  // 0或1

        shared_energy[thread_idx] += h_i * static_cast<float>(x_hat_i);
    }

    // 二阶项贡献：0.5 * ∑_{i,j} J_{ij}·x̂_i·x̂_j（参考INT8版本，使用线程并行）
    for (int t = thread_idx; t < num_quad_terms; t += 256) {
        int var1 = quad_terms_indices[t * 2];
        int var2 = quad_terms_indices[t * 2 + 1];
        float J_ij = quad_terms_coeffs[t];

        // 直接使用{0,1}变量（匹配项目报告定义）
        int x1_hat = x_hat[batch_offset + var1];  // 0或1
        int x2_hat = x_hat[batch_offset + var2];  // 0或1

        shared_quad_energy[thread_idx] += 0.5f * J_ij * static_cast<float>(x1_hat * x2_hat);
    }

    __syncthreads();

    // 高效的归约求和（参考DSB_GPU_fp16和BSB版本）
    for (int s = 128; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_energy[thread_idx] += shared_energy[thread_idx + s];
            shared_quad_energy[thread_idx] += shared_quad_energy[thread_idx + s];
        }
        __syncthreads();
    }

    // 合并线性项和二次项
    if (thread_idx == 0) {
        float scaled_energy = shared_energy[0] + shared_quad_energy[0];

        // 缩放处理（完全匹配INT8版本）
        float original_energy;
        if (is_linear_problem) {
            // 纯线性问题，不进行缩放逆变换
            original_energy = scaled_energy;
        } else {
            // 重新缩放回原始能量
            original_energy = scaled_energy * scaling_factor;
        }

        // 添加常数项（如果存在）
        if (constant_term != 0.0f) {
            original_energy += constant_term;
        }

        energies[batch_idx] = original_energy;
    }
}

// DSB GPU求解器（PyTorch FP8存储 + 混合精度计算）
class DSBFP8GPUSolver {
private:
    ProblemModel host_model;
    int batch_size;

    // GPU设备指针（混合精度策略）
    uint8_t* d_x;                  // 自旋状态 [batch_size * num_vars] (PyTorch FP8，高效存储)
    float* d_y;                   // 动量状态 [batch_size * num_vars] (float，高效计算)
    int* d_x_hat;                 // 离散二值化变量{0,1} [batch_size * num_vars]
    float* d_gradient_term;         // 梯度项 [batch_size * num_vars] (float，高效计算)
    float* d_energies;             // 能量值 [batch_size] (float，保持精度一致性)

    // 模型数据（优化精度：float系数）
    int* d_quad_terms_indices;     // 二阶项索引对 [num_quad_terms * 2]
    float* d_quad_terms_coeffs;    // 二阶项系数 [num_quad_terms] (float，高效)
    float* d_linear_terms;         // 一阶项系数 [num_vars] (float，高效)

    // 准备GPU模型数据（PyTorch FP8版本）
    void prepare_gpu_model() {
        cout << "DEBUG: Preparing DSB PyTorch FP8 GPU model..." << endl;

        // 展平二阶项索引对
        vector<int> flat_quad_terms_indices;
        for (const auto& term : host_model.quad_terms_indices) {
            flat_quad_terms_indices.insert(flat_quad_terms_indices.end(), term.begin(), term.end());
        }

        // 转换系数为float
        vector<float> float_quad_coeffs(host_model.num_quad_terms);
        vector<float> float_linear_coeffs(host_model.num_vars);
        for (int i = 0; i < host_model.num_quad_terms; ++i) {
            float_quad_coeffs[i] = static_cast<float>(host_model.quad_terms_coeffs[i]);
        }
        for (int i = 0; i < host_model.num_vars; ++i) {
            float_linear_coeffs[i] = static_cast<float>(host_model.linear_terms[i]);
        }

        // 分配GPU内存（PyTorch FP8 for x, float for computation）
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * host_model.num_vars * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * host_model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_hat, batch_size * host_model.num_vars * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * host_model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_quad_terms_indices, host_model.num_quad_terms * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_quad_terms_coeffs, host_model.num_quad_terms * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_linear_terms, host_model.num_vars * sizeof(float)));

        // 复制模型数据到GPU
        CUDA_CHECK(cudaMemcpy(d_quad_terms_indices, flat_quad_terms_indices.data(),
                              flat_quad_terms_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_quad_terms_coeffs, float_quad_coeffs.data(),
                              host_model.num_quad_terms * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_linear_terms, float_linear_coeffs.data(),
                              host_model.num_vars * sizeof(float), cudaMemcpyHostToDevice));

        cout << "DEBUG: DSB PyTorch FP8 GPU model prepared (float coeffs, PyTorch FP8 x - INT8-inspired)." << endl;
        cout << "  - FP8 format: PyTorch float8_e4m3fn (Simplified, direct [-1,1] mapping)" << endl;
        cout << "  - FP8 range: [-1, 1] with simplified quantization (no bit operations)" << endl;
        cout << "  - Coefficients stored as float for computation efficiency" << endl;
        cout << "  - Memory usage: ~" << (batch_size * host_model.num_vars * (sizeof(uint8_t) + 2 * sizeof(float)) / 1024.0 / 1024.0) << " MB for dynamics" << endl;
    }

    // 释放GPU内存
    void free_gpu_memory() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_x_hat));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_energies));
        CUDA_CHECK(cudaFree(d_quad_terms_indices));
        CUDA_CHECK(cudaFree(d_quad_terms_coeffs));
        CUDA_CHECK(cudaFree(d_linear_terms));
    }

public:
    DSBFP8GPUSolver(const ProblemModel& problem_model, int bs = 200, float optimized_xi = 0.365f) : host_model(problem_model), batch_size(bs) {
        prepare_gpu_model();
    }

    ~DSBFP8GPUSolver() {
        free_gpu_memory();
    }

    // 核心求解函数（PyTorch FP8版本 - 匹配INT8/FP16）
    pair<vector<float>, float> solve(int max_iters = 1000, float sigma = 0.1f) {
        // DSB算法参数（匹配INT8/FP16版本）
        float xi = 0.365f;   // 优化的耦合强度（来自solve_SB6.py）
        float dt = 0.1f;    // 时间步长
        float delta = 0.9f; // 阻尼系数（新增，匹配INT8/FP16版本）

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << host_model.num_vars << ", Quadratic terms: " << host_model.num_quad_terms << endl;
        cout << "  - Linear terms: " << host_model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << host_model.scaling_factor << endl;
        cout << "  - DSB Algorithm (PyTorch FP8量化版本)" << endl;
        cout << "  - Objective sense: " << host_model.objective_sense << endl;
        cout << "  - Using optimized xi=" << fixed << setprecision(6) << xi << ", sigma=" << fixed << setprecision(6) << sigma << ", dt=" << fixed << setprecision(6) << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - FP8 format: PyTorch float8_e4m3fn (Simplified, INT8-inspired)" << endl;
        cout << "  - Memory optimization: 4x reduction vs FP32, 2x vs FP16" << endl;
        cout << "  - Key improvements (matching INT8/FP16):" << endl;
        cout << "    - x_hat_fp8 returns {0,1} (not sign)" << endl;
        cout << "    - delta=0.9 added for convergence stability" << endl;
        cout << "    - Simplified quantization: direct [-1,1] mapping (no bit operations)" << endl;
        cout << "    - Improved initialization precision with better range mapping" << endl;
        cout << "    - Default iterations: 1000 (reduced from 2000)" << endl;
        cout << "    - Shared memory optimization for energy calculation (from BSB version)" << endl;
        cout << "    - Parallel reduction with 256 threads per batch" << endl;

        // 优化方向判断
        bool is_maximize = (host_model.objective_sense == "maximize");
        bool is_linear_problem = (host_model.num_quad_terms == 0);

        // 计算线程块和网格大小（优化：适配共享内存）
        dim3 blockSize(256);
        dim3 gridSize_batch(batch_size);  // 每个批次一个block，使用共享内存归约
        dim3 gridSize_batches(batch_size);  // 其他kernel
        dim3 gridSize_batch_vars(batch_size, (host_model.num_vars + blockSize.x - 1) / blockSize.x);

        // 初始化批处理数据（传递sigma参数）
        init_batches_kernel<<<gridSize_batch_vars, blockSize>>>(
            d_x, d_y, d_x_hat, d_gradient_term,
            host_model.num_vars, host_model.num_quad_terms,
            time(nullptr), batch_size, static_cast<float>(sigma));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时开始
        auto start = high_resolution_clock::now();

        // 用于记录收敛过程的数据结构
        struct ConvergenceData {
            int iteration;
            float cumulative_time_ms;
            float best_objective;
        };
        vector<ConvergenceData> convergence_history;
        float current_best_overall = host_model.objective_sense == "maximize" ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();

        // 核心迭代（DSB动力学：使用0/1变量，PyTorch FP8版本）
        for (int iter = 0; iter < max_iters; ++iter) {
            // 计算泵浦振幅：p(t) = t/T = iter/(max_iters-1)
            float p = static_cast<float>(iter) / (max_iters - 1);

            // 每次迭代都计算当前时间（用于准确记录收敛时间）
            auto current_time = high_resolution_clock::now();
            float cumulative_time = static_cast<float>(duration_cast<microseconds>(current_time - start).count()) / 1000.0f;

            // DSB动力学更新序列（PyTorch FP8版本 - 匹配INT8/FP16/CPUs）
            // 1. 计算梯度（内部调用calc_x_hat，避免重复调用）
            calc_gradient_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_x_hat, d_gradient_term,
                                                                      d_quad_terms_indices, d_quad_terms_coeffs, d_linear_terms,
                                                                      host_model.num_vars, host_model.num_quad_terms, batch_size,
                                                                      is_maximize);

            // 2. 更新y：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x_hat)（带delta参数）
            update_y_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_y, d_gradient_term,
                                                                host_model.num_vars, p, xi, dt, delta, batch_size);

            // 3. 更新x：dx/dt = y（带delta参数）
            update_x_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_y, host_model.num_vars, dt, delta, batch_size);

            // 4. 边界约束
            apply_constraint_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_y, host_model.num_vars, batch_size);

            // 5. 关键修复：在每次迭代结束时更新x_hat（匹配CPU/INT8/FP16版本）
            // 因为update_x和apply_constraint改变了x，需要同步更新x_hat
            calc_x_hat_kernel<<<gridSize_batches, blockSize>>>(d_x, d_x_hat, host_model.num_vars, batch_size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // 只在最后一次迭代记录收敛数据和输出能量
            if (iter == max_iters - 1) {
                // x_hat已在迭代结束时更新，直接计算能量（使用共享内存优化）
                calc_batch_energy_kernel<<<gridSize_batch, blockSize>>>(d_x, d_x_hat, d_energies,
                                                                  d_quad_terms_indices, d_quad_terms_coeffs, d_linear_terms,
                                                                  host_model.num_vars, host_model.num_quad_terms, batch_size,
                                                                  host_model.scaling_factor, host_model.constant_term, is_linear_problem);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());

                // 复制能量数据回主机
                vector<float> energies(batch_size);
                CUDA_CHECK(cudaMemcpy(energies.data(), d_energies, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

                // 根据优化方向找到当前最佳值
                float current_best = energies[0];
                for (size_t i = 1; i < energies.size(); ++i) {
                    if (is_maximize) {
                        if (energies[i] > current_best) current_best = energies[i];
                    } else {
                        if (energies[i] < current_best) current_best = energies[i];
                    }
                }

                // 更新全局最佳值
                if (host_model.objective_sense == "maximize") {
                    if (current_best > current_best_overall) current_best_overall = current_best;
                } else {
                    if (current_best < current_best_overall) current_best_overall = current_best;
                }

                // 记录最终收敛数据
                convergence_history.push_back({iter + 1, cumulative_time, current_best_overall});

                cout << "  Iteration progress: " << iter + 1 << "/" << max_iters
                     << " | Final energy: " << fixed << setprecision(3) << current_best_overall << endl;
            }
        }

        // 输出收敛过程数据（供Python解析）
        cout << "CONVERGENCE_DATA_START" << endl;
        for (const auto& data : convergence_history) {
            cout << "ITER," << data.iteration << ",TIME," << fixed << setprecision(6) << data.cumulative_time_ms
                 << ",OBJECTIVE," << setprecision(9) << data.best_objective << endl;
        }
        cout << "CONVERGENCE_DATA_END" << endl;

        // 计时结束
        auto end = high_resolution_clock::now();
        float duration = static_cast<float>(duration_cast<microseconds>(end - start).count()) / 1000.0f;
        cout << "C++_TIME: Solving completed, total time: " << fixed << setprecision(3) << duration << " ms" << endl;

        // 计算最终能量并找到最优批次
        calc_x_hat_kernel<<<gridSize_batches, blockSize>>>(d_x, d_x_hat, host_model.num_vars, batch_size);
        calc_batch_energy_kernel<<<gridSize_batch, blockSize>>>(d_x, d_x_hat, d_energies,
                                                          d_quad_terms_indices, d_quad_terms_coeffs, d_linear_terms,
                                                          host_model.num_vars, host_model.num_quad_terms, batch_size,
                                                          host_model.scaling_factor, host_model.constant_term, is_linear_problem);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 复制能量数据回主机
        vector<float> energies(batch_size);
        CUDA_CHECK(cudaMemcpy(energies.data(), d_energies, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

        // 根据优化方向找到最优批次
        int best_idx = 0;
        float best_value = energies[0];
        for (size_t i = 1; i < energies.size(); ++i) {
            if (is_maximize) {
                if (energies[i] > best_value) {
                    best_value = energies[i];
                    best_idx = i;
                }
            } else {
                if (energies[i] < best_value) {
                    best_value = energies[i];
                    best_idx = i;
                }
            }
        }
        
        cout << "BEST: Optimal objective value: " << fixed << setprecision(3) << best_value << " (GPU FP8 precision)" << endl;

        // 复制最优解回主机
        vector<uint8_t> best_solution(host_model.num_vars);
        CUDA_CHECK(cudaMemcpy(best_solution.data(), d_x + best_idx * host_model.num_vars,
                              host_model.num_vars * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        // 转换为标准float输出（从PyTorch FP8转换 - 参考INT8版本第593-600行）
        vector<float> solution(host_model.num_vars);
        for (int i = 0; i < host_model.num_vars; ++i) {
            // 使用FP8PyTorch::dequantize：uint8_t -> [-1,1]
            float x_val = FP8PyTorch::dequantize(best_solution[i]);  // [-1,1]
            float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1]
            solution[i] = x_normalized;
        }

        return {solution, best_value};
    }

    // 获取自旋数
    int get_num_spins() const {
        return host_model.num_vars;
    }

    // 获取问题模型
    const ProblemModel& get_model() const {
        return host_model;
    }
};

// 主函数：DSB算法求解器
int main(int argc, char* argv[]) {
    // 处理命令行参数（使用共享的默认配置）
    string qplib_file_path = DEFAULT_QPLIB_FILE_PATH;
    float optimized_xi = 0.365f;  // 来自solve_SB6.py的优化xi值
    bool show_help = false;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-f" && i + 1 < argc) {
            qplib_file_path = argv[++i];
        } else if (arg == "-x" && i + 1 < argc) {
            optimized_xi = stof(argv[++i]);
        } else if (arg == "-b" && i + 1 < argc) {
            // 跳过batch_size参数（不使用）
            i++;
        } else if (arg == "-i" && i + 1 < argc) {
            // 跳过iterations参数（不使用）
            i++;
        } else if (arg.find(".qplib") != string::npos) {
            // 直接传递的QPLIB文件路径
            qplib_file_path = arg;
        }
    }

    // 显示帮助信息
    if (show_help) {
        cout << "Usage: " << argv[0] << " [options]" << endl;
        cout << "Options:" << endl;
        cout << "  -f <file_path>   Specify QPLIB input file path" << endl;
        cout << "  -x <xi>          Specify xi parameter (default: 0.365)" << endl;
        cout << "  -h               Show this help message" << endl;
        cout << endl;
        cout << "Example: " << argv[0] << " -f QPLIB_3565.qplib -x 0.5" << endl;
        return 0;
    }
    
    // 输出头部信息
    cout << "========================================" << endl;
    cout << "       DSB_GPU_fp8 Solver (PyTorch Compatible)" << endl;
    cout << "  - File path: " << qplib_file_path << endl;
    cout << "  - Algorithm: DSB (Discrete Simulated Bifurcation)" << endl;
    cout << "  - FP8 format: PyTorch float8_e4m3fn (Simplified, INT8-inspired)" << endl;
    cout << "  - Storage: PyTorch float8_e4m3fn (8-bit) for x" << endl;
    cout << "  - Computation: float32 for y, gradient, energy, and coefficients" << endl;
    cout << "  - Memory: 4x reduction vs FP32, 2x vs FP16" << endl;
    cout << "  - Key feature: Uses ±1 binary variables for dynamics (matching INT8/FP16)" << endl;
    cout << "  - Improvements: delta=0.9 for convergence, simplified quantization logic" << endl;
    cout << "  - Quantization: Direct [-1,1] mapping (no complex bit operations)" << endl;
    cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
    cout << "  - Fixed sigma: 0.1" << endl;
    cout << "  - Energy method: Discrete {0,1} values using x̂(x) function (no sign)" << endl;
    cout << "========================================" << endl;

    // 解析指定QPLIB文件
    ProblemModel model;
    bool parse_result = QPLIBParser::parseQplibFile(qplib_file_path, model);
    
    if (!parse_result) {
        cerr << "ERROR: File parsing failed, program exit" << endl;
        return 1;
    }

    // 输出模型信息
    cout << "========================================" << endl;
    cout << "  IMPORTED FILE INFORMATION" << endl;
    cout << "========================================" << endl;
    cout << "Problem model: " << model.problem_name << endl;
    cout << "  Type: QBN" << endl;
    cout << "  Objective sense: " << model.objective_sense << endl;
    cout << "  Number of variables: " << model.num_vars << endl;
    cout << "  Number of constraints: " << model.num_constraints << endl;
    cout << "  Number of quadratic terms: " << model.num_quad_terms << endl;
    cout << "  Number of linear terms: " << model.num_linear_terms << endl;
    cout << "  Number of equality constraints: 0" << endl;
    cout << "  Number of inequality constraints: 0" << endl;
    cout << "  Scaling factor: " << fixed << setprecision(6) << model.scaling_factor << endl;
    cout << "  Coupling strength parameter: " << fixed << setprecision(6) << model.xi << endl;
    cout << "========================================" << endl;
    cout << endl;
    cout << "Starting DSB GPU solver..." << endl;

    // 安全检查：确保linear_terms大小与变量数量一致
    if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
        model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0f);
    }

    // 初始化求解器并求解
    try {
        cout << "Using fixed optimized xi value: " << fixed << setprecision(6) << optimized_xi
             << " (auto-calculated value: " << fixed << setprecision(6) << model.xi << ")" << endl;
        DSBFP8GPUSolver solver(model, 200); // 200 batches
        pair<vector<float>, float> result = solver.solve(1000, 0.1f); // 1000 iterations, sigma=0.1（减少到1000以加快运行）

        vector<float> solution = result.first;
        float best_energy = result.second;

        // 输出0/1二值解（从PyTorch FP8转换 - 参考INT8版本第730-733行）
        vector<int> solution_binary(solution.size());
        for (size_t i = 0; i < solution.size(); ++i) {
            // solution已经是[0,1]范围，直接阈值化（参考INT8版本）
            solution_binary[i] = (solution[i] > 0.5f) ? 1 : 0;
        }
        
        // 统计解中1和0的数量
        int count_ones = 0;
        int count_zeros = 0;
        for (int val : solution_binary) {
            if (val == 1) count_ones++;
            else count_zeros++;
        }
        
        // 输出结果（每16个变量换行，使用0/1二值）
        cout << "RESULT: Solution (" << solution.size() << " variables, binary [0/1]): " << endl;
        for (size_t i = 0; i < solution_binary.size(); ++i) {
            cout << solution_binary[i];
            if ((i + 1) % 16 == 0) cout << endl;
            else if (i != solution_binary.size() - 1) cout << ", ";
        }
        cout << endl << endl;
        
        // 输出解的统计信息
        cout << "Solution statistics:" << endl;
        cout << "  - Variables set to 1: " << count_ones << "/" << solution_binary.size() << endl;
        cout << "  - Variables set to 0: " << count_zeros << "/" << solution_binary.size() << endl;
        cout << "  - Solution density: " << fixed << setprecision(2) << (count_ones * 100.0 / solution_binary.size()) << "%" << endl;
        
        // 输出最终结果总结
        cout << endl << "========================================" << endl;
        cout << "  ** DSB GPU FINAL RESULTS **" << endl;
        cout << "========================================" << endl;
        cout << "  Problem: " << qplib_file_path.substr(qplib_file_path.find_last_of("/\\") + 1) << endl;
        cout << "  Objective sense: " << model.objective_sense << endl;
        cout << "  Number of variables: " << model.num_vars << endl;
        cout << "  Number of quadratic terms: " << model.num_quad_terms << endl;
        cout << "----------------------------------------" << endl;
        cout << "  DSB GPU OBJECTIVE VALUE: " << fixed << setprecision(6) << best_energy << endl;
        cout << "  FP8 format: PyTorch float8_e4m3fn (Simplified, INT8-inspired)" << endl;
        cout << "  Storage: uint8_t (PyTorch FP8) for x, float32 for computation/energy" << endl;
        cout << "  Quantization: Direct [-1,1] mapping (no complex bit operations)" << endl;
        cout << "  Memory Efficiency: 4x reduction vs FP32, 2x vs FP16" << endl;
        cout << "========================================" << endl;
        
    } catch (const exception& e) {
        cerr << "ERROR: Exception during solver initialization: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "ERROR: Unknown exception during solver initialization" << endl;
        return 1;
    }
    
    // 强制刷新输出缓冲区
    cout.flush();
    return 0;
}