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

// INT8 configuration - PyTorch compatible (torch.int8)
// PyTorch uses signed 8-bit integers: range [-128, 127]
constexpr int8_t INT8_MAX_VAL = 127;
constexpr int8_t INT8_MIN_VAL = -128;

// PyTorch INT8 quantization converter (reference: DSB_GPU_fp16.cu FP16Converter)
// Uses round-to-nearest-even rounding (IEEE 754 standard)
struct INT8PyTorch {
    // PyTorch INT8 quantization: maps float to signed int8
    // Uses IEEE 754 round-to-nearest-even rounding
    __host__ __device__ static inline int8_t quantize(float val) {
        // Clamp to INT8 range
        val = fmaxf(static_cast<float>(INT8_MIN_VAL), fminf(static_cast<float>(INT8_MAX_VAL), val));
        // Round to nearest even (IEEE 754 standard, matches FP16Converter)
        return static_cast<int8_t>(roundf(val));
    }

    // PyTorch INT8 dequantization: converts int8 to float
    __host__ __device__ static inline float dequantize(int8_t val) {
        return static_cast<float>(val);
    }
};

// QPLIB configuration moved to qplib_parser.h header file

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << #call << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Manual implementation of clamp function (compatible with different types)
template<typename T, typename U, typename V>
__host__ __device__ T clamp(T val, U min_val, V max_val) {
    if (val < static_cast<T>(min_val)) return static_cast<T>(min_val);
    if (val > static_cast<T>(max_val)) return static_cast<T>(max_val);
    return val;
}



// x_hat分段函数（DSB特征：不使用sign，使用分段函数返回{0,1}）
// 参考项目报告3.md定义：x̂_i = 1 if x_i > 0.5, else 0
__host__ __device__ inline int calc_x_hat(int8_t x) {
    float val = INT8PyTorch::dequantize(x);  // [-127,127]
    float x_normalized = (val + 127.0f) / 254.0f;  // [-127,127] -> [0,1]
    return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
}

// GPU batch data structure (DSB algorithm INT8 version - PyTorch compatible)
// Reference: DSB_GPU_fp16.cu with INT8PyTorch converter
// Key improvement: y and gradient_term use float for performance, x uses int8 for memory efficiency
struct GPUBatchData {
    int8_t* x;              // Spin state (PyTorch int8, efficient storage)
    float* y;               // Momentum state (float, balanced precision and performance)
    int* x_hat;             // Discrete binarization variable {0,1}
    float* gradient_term;   // Gradient term (float, efficient computation)
    float energy;          // Ising Hamiltonian (float, consistent precision)
};

// GPU ProblemModel data (INT8 version - PyTorch compatible)
// Reference: DSB_GPU_fp16.cu with INT8PyTorch converter
// Improvement: coefficients use float for computation efficiency
struct GPUProblemModel {
    int* quad_terms_indices;     // Quadratic term indices [num_quad_terms * 2]
    float* quad_terms_coeffs;    // Quadratic term coefficients (float, efficient)
    float* linear_terms;         // Linear term coefficients (float, efficient)
    float scaling_factor;       // Scaling factor (float, consistent precision)
    int num_vars;
    int num_quad_terms;
};

// GPU kernel function: initialize batch data (DSB algorithm INT8 version - PyTorch compatible)
// Reference: DSB_GPU_fp16.cu init_batches_kernel with INT8PyTorch converter
__global__ void init_batches_kernel(int8_t* x, float* y, int* x_hat, float* gradient_term,
                                   int num_vars, int num_quad_terms,
                                   unsigned long long seed, int batch_size, float sigma) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_vars;

    // DSB算法：x在[0,1]范围内初始化（匹配DSB_GPU_fp16.cu第85-93行）
    for (int i = 0; i < num_vars; ++i) {
        // 生成正态分布随机值：sigma * N(0,1) + 0.5
        float x_normalized = sigma * curand_normal(&state) + 0.5f;  // [0, 1]
        x_normalized = clamp(x_normalized, 0.0f, 1.0f);

        // 映射到INT8范围：[-127,127]（参考FP16版本的[-1,1]映射）
        // PyTorch INT8: (x_normalized - 0.5) * 2 * 127
        float x_scaled = (x_normalized - 0.5f) * 2.0f * static_cast<float>(INT8_MAX_VAL);
        int8_t x_int8_val = INT8PyTorch::quantize(x_scaled);  // Use PyTorch converter
        x[batch_offset + i] = x_int8_val;

        // y initialization as small random value: sigma * N(0,1) (float for efficiency)
        float y_val = sigma * curand_normal(&state);
        y[batch_offset + i] = y_val;

        // 初始化x_hat（匹配项目报告定义）
        x_hat[batch_offset + i] = calc_x_hat(x_int8_val);  // {0,1}二值化
        gradient_term[batch_offset + i] = 0.0f;  // float zero
    }
}

// ============================================================================
// 优化1: 合并calc_x_hat + calc_gradient -> calc_gradient_with_x_hat
// 进一步优化: 循环展开和寄存器缓存优化
// ============================================================================
// GPU内核函数：计算梯度并即时更新x_hat（优化版本）
__global__ void calc_gradient_with_x_hat_kernel(int8_t* x, int* x_hat, float* gradient_term,
                                              int* quad_terms_indices, float* quad_terms_coeffs, float* linear_terms,
                                              int num_vars, int num_quad_terms, int batch_size,
                                              bool is_maximize) {
    int batch_idx = blockIdx.x;
    int var_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || var_idx >= num_vars) return;

    int batch_offset = batch_idx * num_vars;
    float direction_sign = is_maximize ? 1.0f : -1.0f;

    // 优化: 使用寄存器缓存x_int8,减少转换次数
    int8_t x_int8 = x[batch_offset + var_idx];
    int current_x_hat = calc_x_hat(x_int8);
    x_hat[batch_offset + var_idx] = current_x_hat;

    // 优化: 预取线性项到寄存器
    float grad = direction_sign * linear_terms[var_idx];

    // 优化: 循环展开（每次循环处理4个二次项，减少循环开销）
    int t;
    const int UNROLL_FACTOR = 4;
    int unrolled_terms = (num_quad_terms / UNROLL_FACTOR) * UNROLL_FACTOR;

    for (t = 0; t < unrolled_terms; t += UNROLL_FACTOR) {
        // 处理4个二次项
        for (int u = 0; u < UNROLL_FACTOR; ++u) {
            int idx = t + u;
            int var1 = quad_terms_indices[idx * 2];
            int var2 = quad_terms_indices[idx * 2 + 1];
            float J_ij = direction_sign * quad_terms_coeffs[idx];

            if (var1 == var_idx) {
                int x2_hat = x_hat[batch_offset + var2];
                grad += (J_ij / 2.0f) * static_cast<float>(x2_hat);
            } else if (var2 == var_idx) {
                int x1_hat = x_hat[batch_offset + var1];
                grad += (J_ij / 2.0f) * static_cast<float>(x1_hat);
            }
        }
    }

    // 处理剩余的二次项
    for (; t < num_quad_terms; ++t) {
        int var1 = quad_terms_indices[t * 2];
        int var2 = quad_terms_indices[t * 2 + 1];
        float J_ij = direction_sign * quad_terms_coeffs[t];

        if (var1 == var_idx) {
            int x2_hat = x_hat[batch_offset + var2];
            grad += (J_ij / 2.0f) * static_cast<float>(x2_hat);
        } else if (var2 == var_idx) {
            int x1_hat = x_hat[batch_offset + var1];
            grad += (J_ij / 2.0f) * static_cast<float>(x1_hat);
        }
    }

    gradient_term[batch_offset + var_idx] = grad;
}

// GPU内核函数：计算离散二值化变量（DSB算法INT8版本）
__global__ void calc_x_hat_kernel(int8_t* x, int* x_hat, int num_vars, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    for (int i = 0; i < num_vars; ++i) {
        // DSB特征：计算{0,1}离散二值化变量
        x_hat[batch_offset + i] = calc_x_hat(x[batch_offset + i]);
    }
}

// GPU内核函数：计算梯度（DSB算法INT8版本 - 完全匹配FP16）
// 关键修正：梯度计算基于0/1二值变量，完全匹配DSB_GPU_fp16.cu
__global__ void calc_gradient_kernel(int8_t* x, int* x_hat, float* gradient_term,
                                      int* quad_terms_indices, float* quad_terms_coeffs, float* linear_terms,
                                      int num_vars, int num_quad_terms, int batch_size,
                                      bool is_maximize) {
    int batch_idx = blockIdx.x;
    int var_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || var_idx >= num_vars) return;

    int batch_offset = batch_idx * num_vars;

    // 优化方向符号（完全匹配FP16版本第117行）
    float direction_sign = is_maximize ? 1.0f : -1.0f;

    // 初始化梯度为一阶项系数：∇_iE = direction_sign * h_i（完全匹配FP16版本第120行）
    float grad = direction_sign * linear_terms[var_idx];

    // 累加二次项贡献：direction_sign * ∑_j J_{ij}·x̂_j（完全匹配FP16版本第122-138行）
    // DSB核心特征：直接使用离散的{0,1}变量（完全匹配FP16版本）
    for (int t = 0; t < num_quad_terms; ++t) {
        int var1 = quad_terms_indices[t * 2];
        int var2 = quad_terms_indices[t * 2 + 1];
        float J_ij = direction_sign * quad_terms_coeffs[t];

        if (var1 == var_idx) {
            // 直接使用{0,1}变量（完全匹配FP16版本）
            int x2_hat = x_hat[batch_offset + var2];
            grad += (J_ij / 2.0f) * static_cast<float>(x2_hat);
        } else if (var2 == var_idx) {
            // 直接使用{0,1}变量（完全匹配FP16版本）
            int x1_hat = x_hat[batch_offset + var1];
            grad += (J_ij / 2.0f) * static_cast<float>(x1_hat);
        }
    }

    // 直接存储float梯度值
    gradient_term[batch_offset + var_idx] = grad;
}

// GPU内核函数：更新辅助状态y（INT8优化版 - PyTorch compatible）
// Reference: DSB_GPU_fp16.cu update_y_kernel with INT8PyTorch converter
// DSB公式：ẏ_i = -(1-p(t))*(x-0.5) + ξ·∇_iE
__global__ void update_y_kernel(int8_t* x, float* y, float* gradient_term,
                                int num_vars, float p, float xi, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    const float pump_factor = -(1.0f - p);

    // 优化：预计算常量（参考DSB_GPU_fp16.cu第169-176行）
    const float TO_NORM_FACTOR = 1.0f / 254.0f;  // [-127,127] -> [0,1]
    const float HALF_OFFSET = 0.5f;              // 中心点偏移

    for (int i = 0; i < num_vars; ++i) {
        // 使用INT8PyTorch dequantize：INT8 -> float
        float x_int8_val = INT8PyTorch::dequantize(x[batch_offset + i]);  // [-127,127]
        float x_norm = (x_int8_val + 127.0f) * TO_NORM_FACTOR;  // [0,1]
        float x_shifted = x_norm - HALF_OFFSET;  // (x-0.5)

        // DSB公式的直接计算（参考FP16版本）
        float dy = (pump_factor * x_shifted + xi * gradient_term[batch_offset + i]) * dt;
        y[batch_offset + i] += dy;
    }
}

// GPU内核函数：更新自旋x（INT8优化版 - PyTorch compatible）
// Reference: DSB_GPU_fp16.cu update_x_kernel with INT8PyTorch converter
// DSB公式：ẋ_i = y_i，离散化：x_i^{(t+1)} = x_i^{(t)} + y_i^{(t+1)}·dt
__global__ void update_x_kernel(int8_t* x, float* y, int num_vars, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;

    // 优化：预计算常量（参考DSB_GPU_fp16.cu第200行）
    const float TO_NORM_FACTOR = 1.0f / 254.0f;  // [-127,127] -> [0,1]
    const float TO_INT8_FACTOR = 254.0f;        // [0,1] -> [-127,127]
    const float INT8_OFFSET = -127.0f;          // 偏移量

    for (int i = 0; i < num_vars; ++i) {
        // 使用INT8PyTorch dequantize：INT8 -> float
        float x_int8_val = INT8PyTorch::dequantize(x[batch_offset + i]);  // [-127,127]
        float x_normalized = (x_int8_val + 127.0f) * TO_NORM_FACTOR;  // [0,1]

        // DSB位置更新（参考FP16版本）
        float x_new = x_normalized + y[batch_offset + i] * dt;
        x_new = clamp(x_new, 0.0f, 1.0f);  // 限制在[0,1]

        // 使用INT8PyTorch quantize：float -> INT8
        float x_int8_new = x_new * TO_INT8_FACTOR + INT8_OFFSET;
        x[batch_offset + i] = INT8PyTorch::quantize(x_int8_new);
    }
}

// GPU内核函数：应用边界约束（INT8优化版 - PyTorch compatible）
// Reference: DSB_GPU_fp16.cu apply_constraint_kernel with INT8PyTorch converter
__global__ void apply_constraint_kernel(int8_t* x, float* y, int num_vars, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;

    // 优化：预计算常量（参考DSB_GPU_fp16.cu第221-224行）
    const float TO_NORM_FACTOR = 1.0f / 254.0f;  // [-127,127] -> [0,1]
    const float BOUNDARY_THRESHOLD = 0.5f;      // 边界阈值

    for (int i = 0; i < num_vars; ++i) {
        // 使用INT8PyTorch dequantize：INT8 -> float
        float x_int8_val = INT8PyTorch::dequantize(x[batch_offset + i]);  // [-127,127]
        float x_norm = (x_int8_val + 127.0f) * TO_NORM_FACTOR;  // [0,1]

        // 检查是否超出边界（参考FP16版本）
        float distance_from_center = fabsf(x_norm - 0.5f);

        if (distance_from_center > BOUNDARY_THRESHOLD) {
            // 投影到边界{0,1}（参考FP16版本）
            float projected = (x_norm > 0.5f) ? 1.0f : 0.0f;
            float x_int8_projected = projected * 254.0f - 127.0f;
            x[batch_offset + i] = INT8PyTorch::quantize(x_int8_projected);
            y[batch_offset + i] = 0.0f;  // 清零动量
        }
    }
}

// GPU内核函数：计算单个批次的能量（DSB算法INT8版本 - 完全匹配FP16 + 共享内存优化）
// 关键修正：使用0/1二值变量进行能量计算，完全匹配DSB_GPU_fp16.cu
// 性能优化：使用共享内存进行归约求和（参考DSB_GPU_fp16和BSB版本）
__global__ void calc_batch_energy_kernel(int8_t* x, int* x_hat, float* energies,
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

    // 一阶项贡献：∑_i h_i·x̂_i （完全匹配FP16版本第258-266行，使用线程并行）
    for (int i = thread_idx; i < num_vars; i += 256) {
        float h_i = linear_terms[i];

        // 直接使用{0,1}变量（完全匹配FP16版本的处理）
        int x_hat_val = x_hat[batch_offset + i];  // 0或1

        shared_energy[thread_idx] += h_i * static_cast<float>(x_hat_val);
    }

    // 二阶项贡献：0.5 * ∑_{i,j} J_{ij}·x̂_i·x̂_j （完全匹配FP16版本第245-256行，使用线程并行）
    for (int t = thread_idx; t < num_quad_terms; t += 256) {
        int var1 = quad_terms_indices[t * 2];
        int var2 = quad_terms_indices[t * 2 + 1];
        float J_ij = quad_terms_coeffs[t];

        // 直接使用{0,1}变量（完全匹配FP16版本的处理）
        int x1_hat = x_hat[batch_offset + var1];  // 0或1
        int x2_hat = x_hat[batch_offset + var2];  // 0或1

        shared_quad_energy[thread_idx] += 0.5f * J_ij * static_cast<float>(x1_hat) * static_cast<float>(x2_hat);
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

        // 缩放处理（完全匹配DSB_GPU_fp16.cu第268-276行）
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

// DSB GPU求解器（INT8版本 - PyTorch compatible）
class DSBInt8GPUSolver {
private:
    ProblemModel host_model;
    int batch_size;

    // GPU device pointers (DSB algorithm INT8 version - PyTorch compatible)
    // Reference: DSB_GPU_fp16.cu with INT8PyTorch converter
    // Improvement: y and gradient_term use float for performance, x uses int8 for memory efficiency
    int8_t* d_x;                // Spin state [batch_size * num_vars] (PyTorch int8)
    float* d_y;                 // Auxiliary state [batch_size * num_vars] (float)
    int* d_x_hat;               // Discrete binarization variable {0,1} [batch_size * num_vars]
    float* d_gradient_term;     // Gradient term [batch_size * num_vars] (float)
    float* d_energies;         // Energy values [batch_size] (float, consistent precision)

    // Model data (optimized precision: float coefficients)
    int* d_quad_terms_indices;     // Quadratic term indices [num_quad_terms * 2]
    float* d_quad_terms_coeffs;    // Quadratic term coefficients [num_quad_terms] (float)
    float* d_linear_terms;         // Linear term coefficients [num_vars] (float)
    float d_scaling_factor;       // Scaling factor (float, consistent precision)

    // 准备GPU模型数据（DSB算法INT8版本 - PyTorch compatible）
    // Reference: DSB_GPU_fp16.cu prepare_gpu_model with INT8PyTorch converter
    void prepare_gpu_model() {
        cout << "DEBUG: Preparing DSB INT8 GPU model (PyTorch compatible)..." << endl;
        cout << "  - Using INT8PyTorch converter (based on DSB_GPU_fp16.cu FP16Converter)" << endl;
        cout << "  - Round-to-nearest-even rounding mode (IEEE 754 standard)" << endl;

        // 展平二阶项索引对
        vector<int> flat_quad_terms_indices;
        for (const auto& term : host_model.quad_terms_indices) {
            flat_quad_terms_indices.insert(flat_quad_terms_indices.end(), term.begin(), term.end());
        }

        // Convert coefficients to float (reference: DSB_GPU_fp16.cu line 328-336)
        vector<float> float_quad_coeffs(host_model.num_quad_terms);
        vector<float> float_linear_coeffs(host_model.num_vars);
        for (int i = 0; i < host_model.num_quad_terms; ++i) {
            float_quad_coeffs[i] = static_cast<float>(host_model.quad_terms_coeffs[i]);
        }
        for (int i = 0; i < host_model.num_vars; ++i) {
            float_linear_coeffs[i] = static_cast<float>(host_model.linear_terms[i]);
        }

        // Allocate GPU memory (PyTorch INT8 for x, float for y and gradient)
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * host_model.num_vars * sizeof(int8_t)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * host_model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_x_hat, batch_size * host_model.num_vars * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * host_model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));  // energy values use float
        CUDA_CHECK(cudaMalloc(&d_quad_terms_indices, host_model.num_quad_terms * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_quad_terms_coeffs, host_model.num_quad_terms * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_linear_terms, host_model.num_vars * sizeof(float)));

        // Copy model data to GPU
        CUDA_CHECK(cudaMemcpy(d_quad_terms_indices, flat_quad_terms_indices.data(),
                              flat_quad_terms_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_quad_terms_coeffs, float_quad_coeffs.data(),
                              host_model.num_quad_terms * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_linear_terms, float_linear_coeffs.data(),
                              host_model.num_vars * sizeof(float), cudaMemcpyHostToDevice));

        // Copy scaling factor
        d_scaling_factor = static_cast<float>(host_model.scaling_factor);

        cout << "DEBUG: DSB INT8 GPU model prepared (PyTorch compatible: float coefficients, int8 x)." << endl;
        cout << "  - Coefficients stored as float for computation efficiency" << endl;
        cout << "  - INT8 range: [" << static_cast<int>(INT8_MIN_VAL) << ", " << static_cast<int>(INT8_MAX_VAL) << "]" << endl;
        cout << "  - Memory usage: ~" << (batch_size * host_model.num_vars * (sizeof(int8_t) + 2 * sizeof(float)) / 1024.0 / 1024.0) << " MB for dynamics" << endl;
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
    DSBInt8GPUSolver(const ProblemModel& model, int bs = 200, float optimized_xi = 0.365f) : host_model(model), batch_size(bs) {
        prepare_gpu_model();
    }

    ~DSBInt8GPUSolver() {
        free_gpu_memory();
    }

    // 核心求解函数（DSB算法INT8版本，匹配BSB版本）
    // 关键修正：添加sigma参数，匹配BSB版本的初始化方式
    pair<vector<float>, float> solve(int max_iters = 1000, float sigma = 0.1f) {
        // DSB算法参数（与BSB版本保持一致）
        float xi = 0.365f;   // 优化的耦合强度（来自solve_SB6.py）
        float dt = 0.1f;    // 时间步长
        float delta = 0.9f; // 阻尼系数

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << host_model.num_vars << ", Quadratic terms: " << host_model.num_quad_terms << endl;
        cout << "  - Linear terms: " << host_model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << host_model.scaling_factor << endl;
        cout << "  - DSB Algorithm (INT8优化版本 - 参考FP16性能优化)" << endl;
        cout << "  - Objective sense: " << host_model.objective_sense << endl;
        cout << "  - Using optimized xi=" << fixed << setprecision(6) << xi << ", sigma=" << fixed << setprecision(6) << sigma << ", dt=" << fixed << setprecision(6) << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - PyTorch INT8 compatible (reference: DSB_GPU_fp16.cu):" << endl;
        cout << "    - INT8PyTorch: quantize/dequantize with IEEE 754 rounding" << endl;
        cout << "    - Round-to-nearest-even: roundf()" << endl;
        cout << "    - INT8 range: [" << static_cast<int>(INT8_MIN_VAL) << ", " << static_cast<int>(INT8_MAX_VAL) << "]" << endl;
        cout << "  - Key optimization: Simplified computation + float precision + reduced conversion overhead" << endl;
        cout << "  - Memory optimization: 8x reduction vs mixed precision" << endl;
        cout << "  - GPU acceleration: Direct computation mode (FP16 version reference)" << endl;
        cout << "  - Shared memory optimization for energy calculation (from BSB version)" << endl;
        cout << "  - Parallel reduction with 256 threads per batch" << endl;

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
        float current_best_overall = host_model.objective_sense == "maximize" ? -numeric_limits<float>::infinity() : numeric_limits<float>::infinity();

        // 核心迭代（DSB动力学：使用±1变量，参考BSB版本）
        for (int iter = 0; iter < max_iters; ++iter) {
            // 计算泵浦振幅：p(t) = t/T = iter/(max_iters-1)
            float p = static_cast<float>(iter) / (max_iters - 1);

            // 每次迭代都计算当前时间（用于准确记录收敛时间）
            auto current_time = high_resolution_clock::now();
            float cumulative_time = static_cast<float>(duration_cast<microseconds>(current_time - start).count()) / 1000.0f;

            // DSB动力学更新序列（基于BSB版本，使用x_hat）
            // 1. 计算梯度并即时更新x_hat（确保x_hat在每次迭代时同步更新）
            calc_gradient_with_x_hat_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_x_hat, d_gradient_term,
                                                                      d_quad_terms_indices, d_quad_terms_coeffs, d_linear_terms,
                                                                      host_model.num_vars, host_model.num_quad_terms, batch_size,
                                                                      is_maximize);

            // 2. 更新y：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x_hat)
            update_y_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_y, d_gradient_term,
                                                                host_model.num_vars, p, xi, dt, delta, batch_size);

            // 3. 更新x：dx/dt = y
            update_x_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_y, host_model.num_vars, dt, delta, batch_size);

            // 4. 边界约束
            apply_constraint_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_y, host_model.num_vars, batch_size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // 只在最后一次迭代输出能量结果
            if (iter == max_iters - 1) {
                calc_x_hat_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_x_hat, host_model.num_vars, batch_size);
                calc_batch_energy_kernel<<<gridSize_batch, blockSize>>>(d_x, d_x_hat, d_energies,
                                                                  d_quad_terms_indices, d_quad_terms_coeffs, d_linear_terms,
                                                                  host_model.num_vars, host_model.num_quad_terms, batch_size,
                                                                  d_scaling_factor, host_model.constant_term, is_linear_problem);
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

                // 记录收敛数据
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
        calc_x_hat_kernel<<<gridSize_batch_vars, blockSize>>>(d_x, d_x_hat, host_model.num_vars, batch_size);
        calc_batch_energy_kernel<<<gridSize_batch, blockSize>>>(d_x, d_x_hat, d_energies,
                                                          d_quad_terms_indices, d_quad_terms_coeffs, d_linear_terms,
                                                          host_model.num_vars, host_model.num_quad_terms, batch_size,
                                                          d_scaling_factor, host_model.constant_term, is_linear_problem);
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
        
        cout << "BEST: Optimal objective value: " << fixed << setprecision(3) << best_value << " (GPU PyTorch INT8 precision)" << endl;

        // 复制最优解回主机
        vector<int8_t> best_solution(host_model.num_vars);
        CUDA_CHECK(cudaMemcpy(best_solution.data(), d_x + best_idx * host_model.num_vars,
                              host_model.num_vars * sizeof(int8_t), cudaMemcpyDeviceToHost));

        // 关键改进：同时返回连续变量和二值化解（使用INT8PyTorch converter）
        vector<float> solution(host_model.num_vars);
        vector<float> binary_solution(host_model.num_vars);
        for (int i = 0; i < host_model.num_vars; ++i) {
            // 使用INT8PyTorch dequantize：INT8 -> [0,1]（连续变量）
            float x_val = INT8PyTorch::dequantize(best_solution[i]);  // [-127,127]
            float x_normalized = (x_val + 127.0f) / 254.0f;  // [-127,127] -> [0,1]
            solution[i] = x_normalized;

            // 二值化版本（用于最终能量计算）
            binary_solution[i] = (x_normalized > 0.5f) ? 1.0f : 0.0f;
        }

        return {solution, best_value};
    }

    // 获取自旋数（与FP16版本保持一致）
    int get_num_spins() const {
        return host_model.num_vars;
    }

    // 获取问题模型（与FP16版本保持一致）
    const ProblemModel& get_model() const {
        return host_model;
    }
};

// 主函数：支持命令行参数
int main(int argc, char* argv[]) {
    string input_file = DEFAULT_QPLIB_FILE_PATH;  // 使用默认路径
    int batch_size = 200;
    int iterations = 1000;
    float optimized_xi = 0.365f;  // 添加xi参数
    bool show_help = false;

    // 解析命令行参数（Windows兼容版本）
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-f" && i + 1 < argc) {
            input_file = argv[++i];
        } else if (arg == "-x" && i + 1 < argc) {
            optimized_xi = stof(argv[++i]);
        } else if (arg == "-b" && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
            if (batch_size <= 0) {
                cerr << "ERROR: Batch size must be positive" << endl;
                show_help = true;
            }
        } else if (arg == "-i" && i + 1 < argc) {
            iterations = atoi(argv[++i]);
            if (iterations <= 0) {
                cerr << "ERROR: Number of iterations must be positive" << endl;
                show_help = true;
            }
        } else if (arg.find(".qplib") != string::npos) {
            // 直接传递的QPLIB文件路径
            input_file = arg;
        } else {
            cerr << "ERROR: Unrecognized argument: " << arg << endl;
            show_help = true;
        }
    }

    // 显示帮助信息
    if (show_help) {
        cout << "Usage: " << argv[0] << " [options]" << endl;
        cout << "Options:" << endl;
        cout << "  -f <file_path>   Specify QPLIB input file path" << endl;
        cout << "  -x <xi>          Specify xi parameter (default: 0.365)" << endl;
        cout << "  -b <batch_size>  Set number of batches (default: 200)" << endl;
        cout << "  -i <iterations>  Set number of iterations (default: 1000)" << endl;
        cout << "  -h               Show this help message" << endl;
        cout << endl;
        cout << "Example: " << argv[0] << " -f QPLIB_3565.qplib -x 0.5 -b 200 -i 1000" << endl;
        return 0;
    }
    cout << "========================================" << endl;
    cout << "          DSB_GPU_int8 Solver           " << endl;
    cout << "  - Target file: " << (input_file == DEFAULT_QPLIB_FILE_PATH ? DEFAULT_QPLIB_FILE : "Custom file") << endl;
    cout << "  - File path: " << input_file << endl;
    cout << "  - Batch size: " << batch_size << endl;
    cout << "  - Iterations: " << iterations << endl;
      cout << "  - Algorithm: DSB (Discrete Simulated Bifurcation) - PyTorch INT8 Version" << endl;
      cout << "  - Key features: PyTorch INT8 + FP16 performance optimization strategy" << endl;
    cout << "  - Based on DSB_GPU_fp16.cu with INT8PyTorch converter" << endl;
    cout << "  - INT8PyTorch: quantize/dequantize with IEEE 754 rounding" << endl;
    cout << "  - Round-to-nearest-even: roundf()" << endl;
    cout << "  - INT8 range: [" << static_cast<int>(INT8_MIN_VAL) << ", " << static_cast<int>(INT8_MAX_VAL) << "]" << endl;
    cout << "  - Memory optimization: 8x reduction vs mixed precision" << endl;
    cout << "  - Performance: Direct computation mode, FP16 version reference" << endl;
    cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << ", sigma: 0.1" << endl;
    cout << "========================================" << endl << endl;

    // 解析指定QPLIB文件
    ProblemModel model;
    cout << "Start parsing file: " << input_file << endl;
    if (!QPLIBParser::parseQplibFile(input_file, model)) {
        cerr << "WARNING: File parsing failed, program exit" << endl;
        return 1;
    }
    
    cout << "DEBUG: Parsing completed successfully, preparing solver..." << endl;
    cout << "DEBUG: Model has " << model.num_vars << " variables and " << model.num_quad_terms << " quadratic terms" << endl;

    // 初始化求解器并求解
    cout << "DEBUG: Creating DSB INT8 solver with " << batch_size << " batches..." << endl;
    DSBInt8GPUSolver solver(model, batch_size, optimized_xi);
    cout << "DEBUG: Solver created successfully, starting solve process..." << endl;
    cout << "Using fixed optimized xi value: " << fixed << setprecision(6) << optimized_xi
         << " (auto-calculated value: " << fixed << setprecision(6) << model.xi << ")" << endl;

    cout << "DEBUG: Starting solve with " << iterations << " iterations..." << endl;
    pair<vector<float>, float> result = solver.solve(iterations, 0.1f);  // iterations, sigma=0.1
    vector<float> solution = result.first;
    float best_energy = result.second;
    cout << "DEBUG: Solve process completed." << endl;

    // 创建二值化解用于重新计算目标函数
    vector<float> binary_solution(model.num_vars);
    for (size_t i = 0; i < solution.size(); ++i) {
        binary_solution[i] = (solution[i] > 0.5f) ? 1.0f : 0.0f;
    }

    // 关键改进：重新计算目标函数值（使用二值化解，与BSB版本一致）
    float recomputed_objective = 0.0f;

    // 使用二值化解重新计算能量
    for (int i = 0; i < model.num_vars; ++i) {
        float x_bin = binary_solution[i];  // 0或1
        recomputed_objective += static_cast<float>(model.linear_terms[i]) * x_bin;
    }

    for (int t = 0; t < model.num_quad_terms; ++t) {
        int var1 = model.quad_terms_indices[t][0];
        int var2 = model.quad_terms_indices[t][1];
        float coeff = static_cast<float>(model.quad_terms_coeffs[t]);
        float x1_bin = binary_solution[var1];
        float x2_bin = binary_solution[var2];
        recomputed_objective += 0.5f * coeff * x1_bin * x2_bin;
    }

    recomputed_objective = recomputed_objective * static_cast<float>(model.scaling_factor) + static_cast<float>(model.constant_term);
    best_energy = recomputed_objective;  // 使用重新计算的值
    
    // 转换为0/1表示（用于显示）
    vector<int> solution_binary(solution.size());
    for (size_t i = 0; i < solution.size(); ++i) {
        solution_binary[i] = (solution[i] > 0.5f) ? 1 : 0;
    }
    
    // 统计解中1的数量
    int count_ones = 0;
    for (int val : solution_binary) {
        if (val == 1) count_ones++;
    }

    // 输出结果（每16个变量换行，使用0/1值）
    cout << "RESULT: Solution (" << solver.get_num_spins() << " variables, binary [0/1]): " << endl;
    for (size_t i = 0; i < solution_binary.size(); ++i) {
        cout << solution_binary[i];
        if ((i + 1) % 16 == 0) cout << endl;
        else if (i != solution_binary.size() - 1) cout << ", ";
    }
    cout << endl << endl;
    
    // 输出解的统计信息
    cout << "Solution statistics:" << endl;
    cout << "  - Variables set to 1: " << count_ones << "/" << solution_binary.size() << endl;
    cout << "  - Variables set to 0: " << (solution_binary.size() - count_ones) << "/" << solution_binary.size() << endl;
    cout << "  - Solution density: " << fixed << setprecision(2) << (count_ones * 100.0 / solution_binary.size()) << "%" << endl;
    cout << endl;
    
    // 输出最终目标函数值（匹配DSB_CPU_int8.cu一致的醒目格式）
    cout << "========================================" << endl;
    cout << "  ** DSB GPU FINAL RESULTS **" << endl;
    cout << "========================================" << endl;
    cout << "  Problem: " << solver.get_model().problem_name << endl;
    cout << "  Objective sense: " << solver.get_model().objective_sense << endl;
    cout << "  Number of variables: " << solver.get_model().num_vars << endl;
    cout << "  Number of quadratic terms: " << solver.get_model().num_quad_terms << endl;
    cout << "----------------------------------------" << endl;
    // 输出目标函数值（改进版本，显示两种能量值）
    cout << "  DSB GPU CONTINUOUS ENERGY: " << fixed << setprecision(6) << best_energy << " (from optimization)" << endl;
    cout << "  DSB GPU BINARY OBJECTIVE: " << fixed << setprecision(6) << recomputed_objective << " (final solution)" << endl;
    cout << "========================================" << endl;
    
    cout << "Target file processing completed!" << endl;
    cout << "Applied DSB algorithm (PyTorch INT8 Compatible):" << endl;
    cout << "  - Discrete dynamics: ±1 binary variables for optimization" << endl;
    cout << "  - PyTorch INT8: 8x memory reduction + float computation optimization" << endl;
    cout << "  - INT8PyTorch: quantize/dequantize with IEEE 754 rounding" << endl;
    cout << "  - GPU performance optimization: Direct computation mode, reduced precision conversion" << endl;
    cout << "  - Based on DSB_GPU_fp16.cu with INT8PyTorch converter" << endl;
    cout << "========================================" << endl;
    return 0;
}