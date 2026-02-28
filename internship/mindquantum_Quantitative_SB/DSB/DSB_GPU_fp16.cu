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
#include <cuda_fp16.h>
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// FP16配置与宏定义 - 基于DSB_GPU_fp8改进精度
constexpr float FP16_MAX_VAL = 65504.0f;
constexpr float FP16_MIN_VAL = -65504.0f;
constexpr float FP16_EPSILON = 0.0009765625f;  // FP16最小精度单位（2^-10）

// FP16转换器 - 统一转换接口，减少精度损失（参考DSB_GPU_fp8）
struct FP16Converter {
    __host__ __device__ static inline __half to_fp16(float val) {
        // 限制到FP16范围
        val = fmaxf(FP16_MIN_VAL, fminf(FP16_MAX_VAL, val));
        // 使用round-to-nearest-even舍入（与DSB_GPU_fp8一致）
        return __float2half_rn(val);
    }
    
    __host__ __device__ static inline float from_fp16(__half h) {
        return __half2float(h);
    }
};

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

// 自定义x^分段函数（DSB特征：不使用sign，使用分段函数返回{0,1}）
// 参考项目报告3.md定义：x̂_i = 1 if x_i > 0.5, else 0
__host__ __device__ inline int calc_x_hat(__half x) {
    float val = __half2float(x);  // [-1,1]
    float x_normalized = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
}

// GPU内核函数：初始化批处理数据（DSB算法FP16版本）
__global__ void init_batches_kernel(__half* x, __half* y, int* x_hat, __half* gradient_term,
                                   int num_vars, int num_quad_terms,
                                   unsigned long long seed, int batch_size, float sigma) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_vars;

    for (int i = 0; i < num_vars; ++i) {
        // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5
        float x_normalized = sigma * curand_normal(&state) + 0.5f;  // [0, 1]
        x_normalized = clamp(x_normalized, 0.0f, 1.0f);

        // 映射到FP16范围：[-1,1]，其中0.5映射到0（使用FP16Converter）
        float x_scaled = (x_normalized - 0.5f) * 2.0f;
        __half x_fp16 = FP16Converter::to_fp16(clamp(x_scaled, -1.0f, 1.0f));
        x[batch_offset + i] = x_fp16;

        // y初始化为小随机值：sigma * N(0,1)（使用FP16Converter）
        float y_val = sigma * curand_normal(&state);
        y[batch_offset + i] = FP16Converter::to_fp16(y_val);

        // 初始化x_hat（匹配项目报告定义）
        x_hat[batch_offset + i] = calc_x_hat(x_fp16);  // {0,1}二值化
        gradient_term[batch_offset + i] = FP16Converter::to_fp16(0.0f);
    }
}

// ============================================================================
// 优化1: 合并calc_x_hat + calc_gradient -> calc_gradient_with_x_hat
// 进一步优化: 循环展开和寄存器缓存优化
// ============================================================================
// GPU内核函数：计算梯度并即时更新x_hat（优化版本）
__global__ void calc_gradient_with_x_hat_kernel(__half* x, int* x_hat, __half* gradient_term,
                                              int* quad_terms_indices, __half* quad_terms_coeffs, __half* linear_terms,
                                              int num_vars, int num_quad_terms, int batch_size,
                                              bool is_maximize) {
    int batch_idx = blockIdx.x;
    int var_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || var_idx >= num_vars) return;

    int batch_offset = batch_idx * num_vars;
    float direction_sign = is_maximize ? 1.0f : -1.0f;

    // 优化: 使用寄存器缓存x_half,减少FP16转换次数
    __half x_half = x[batch_offset + var_idx];
    int current_x_hat = calc_x_hat(x_half);
    x_hat[batch_offset + var_idx] = current_x_hat;

    // 优化: 预取线性项到寄存器
    float grad = direction_sign * __half2float(linear_terms[var_idx]);

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
            float J_ij = direction_sign * __half2float(quad_terms_coeffs[idx]);

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
        float J_ij = direction_sign * __half2float(quad_terms_coeffs[t]);

        if (var1 == var_idx) {
            int x2_hat = x_hat[batch_offset + var2];
            grad += (J_ij / 2.0f) * static_cast<float>(x2_hat);
        } else if (var2 == var_idx) {
            int x1_hat = x_hat[batch_offset + var1];
            grad += (J_ij / 2.0f) * static_cast<float>(x1_hat);
        }
    }

    gradient_term[batch_offset + var_idx] = FP16Converter::to_fp16(grad);
}

// GPU内核函数：计算离散二值化变量（DSB算法FP16版本）
__global__ void calc_x_hat_kernel(__half* x, int* x_hat, int num_vars, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    for (int i = 0; i < num_vars; ++i) {
        // DSB特征：计算{0,1}离散二值化变量
        x_hat[batch_offset + i] = calc_x_hat(x[batch_offset + i]);
    }
}

// GPU内核函数：计算梯度（DSB算法FP16版本）
// 关键修正：使用0/1二值变量进行梯度计算，并添加优化方向支持
// DSB公式：∇_iE = direction_sign * (h_i + ∑_{j} J_{ij}·x̂_j)
// 其中x̂_j ∈ {0,1}，direction_sign根据优化方向确定（匹配DSB_CPU_fp16.cu）
__global__ void calc_gradient_kernel(__half* x, int* x_hat, __half* gradient_term,
                                      int* quad_terms_indices, __half* quad_terms_coeffs, __half* linear_terms,
                                      int num_vars, int num_quad_terms, int batch_size,
                                      bool is_maximize) {
    int batch_idx = blockIdx.x;
    int var_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || var_idx >= num_vars) return;

    int batch_offset = batch_idx * num_vars;

    // 优化方向符号（匹配CPU版本第155-156行）
    float direction_sign = is_maximize ? 1.0f : -1.0f;

    // 初始化梯度为一阶项系数：∇_iE = direction_sign * h_i
    float grad = direction_sign * __half2float(linear_terms[var_idx]);

    // 累加二次项贡献：direction_sign * ∑_j J_{ij}·x̂_j
    // DSB核心特征：直接使用离散的{0,1}变量（匹配CPU版本）
    for (int t = 0; t < num_quad_terms; ++t) {
        int var1 = quad_terms_indices[t * 2];
        int var2 = quad_terms_indices[t * 2 + 1];
        float J_ij = direction_sign * __half2float(quad_terms_coeffs[t]);

        if (var1 == var_idx) {
            // 直接使用{0,1}变量（匹配DSB算法标准）
            int x2_hat = x_hat[batch_offset + var2];  // 0或1
            grad += (J_ij / 2.0f) * static_cast<float>(x2_hat);
        } else if (var2 == var_idx) {
            // 直接使用{0,1}变量（匹配DSB算法标准）
            int x1_hat = x_hat[batch_offset + var1];  // 0或1
            grad += (J_ij / 2.0f) * static_cast<float>(x1_hat);
        }
    }

    // 使用FP16Converter统一转换（参考DSB_GPU_fp8）
    gradient_term[batch_offset + var_idx] = FP16Converter::to_fp16(grad);
}

// GPU内核函数：更新辅助状态y（DSB算法FP16版本）
// 关键修正：泵浦项使用连续的(x-0.5)，梯度项使用离散的sign(x)
// 修正的DSB公式：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x_sign)
// 匹配DSB_CPU_fp16.cu第192-210行
__global__ void update_y_kernel(__half* x, __half* y, __half* gradient_term,
                                int num_vars, float p, float xi, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    const float pump_factor = -(1.0f - p);  // 泵浦振幅项：-(1-p(t))
    
    for (int i = 0; i < num_vars; ++i) {
        // DSB修正：泵浦项使用连续的(x-0.5)，匹配CPU版本第201-203行
        __half x_half = x[batch_offset + i];  // [-1,1]
        float x_val = __half2float(x_half);
        float x_normalized = (x_val + 1.0f) / 2.0f;  // [0,1]
        float pump_term = pump_factor * (x_normalized - 0.5f);  // -(1-p(t))*(x-0.5)
        
        // 梯度项已经基于0/1变量计算（DSB特征，匹配CPU版本第206行）
        float gradient_val = __half2float(gradient_term[batch_offset + i]);
        
        // 修正的DSB公式：ẏ_i = [-(1-p(t))*(x-0.5) + ξ*∇f(x_sign)] * dt
        float dy = (pump_term + xi * gradient_val) * dt;
        float y_val = FP16Converter::from_fp16(y[batch_offset + i]) + dy;
        
        // 使用FP16Converter统一转换（参考DSB_GPU_fp8）
        y[batch_offset + i] = FP16Converter::to_fp16(y_val);
    }
}

// GPU内核函数：更新自旋x（DSB算法FP16版本）
// 对应DSB公式：ẋ_i = a(0)y_i
// 离散化：x_i^{(t+1)} = x_i^{(t)} + a(0)·y_i^{(t+1)}·dt
// 其中a(0) = 1，dt是时间步长
// FP16量化：x_i ∈ [-65504, 65504]，直接使用float16范围
__global__ void update_x_kernel(__half* x, __half* y, int num_vars, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    const float factor = dt * delta;  // 使用delta确保收敛稳定性
    
    for (int i = 0; i < num_vars; ++i) {
        // DSB位置更新：x_i^{(t+1)} = x_i^{(t)} + a(0)·y_i^{(t+1)}·dt
        float x_val = FP16Converter::from_fp16(x[batch_offset + i]);
        float y_val = FP16Converter::from_fp16(y[batch_offset + i]);
        float x_new_cont = x_val + y_val * factor;
        
        // 使用FP16Converter统一转换（参考DSB_GPU_fp8）
        x[batch_offset + i] = FP16Converter::to_fp16(x_new_cont);
    }
}

// ============================================================================
// 优化2: 合并update_x + apply_constraint -> update_x_with_constraint
// 进一步优化: 减少内存访问，使用寄存器缓存
// ============================================================================
// GPU内核函数：更新x并应用边界约束（深度优化版本）
__global__ void update_x_with_constraint_kernel(__half* x, __half* y,
                                              int num_vars, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    const float factor = dt * delta;

    // 优化: 使用寄存器缓存x和y，减少全局内存读写
    for (int i = 0; i < num_vars; ++i) {
        // 步骤1: 从全局内存加载到寄存器
        float x_val = FP16Converter::from_fp16(x[batch_offset + i]);
        float y_val = FP16Converter::from_fp16(y[batch_offset + i]);

        // 步骤2: 更新x（使用寄存器）
        float x_new_cont = x_val + y_val * factor;

        // 步骤3: 将x_new_cont转换到[0,1]范围
        float x_normalized = (x_new_cont + 1.0f) / 2.0f;

        // 步骤4: 应用边界约束
        float distance_from_center = fabsf(x_normalized - 0.5f);

        if (distance_from_center > 0.5f) {
            // 超出边界，投影到边界{0,1}并清零动量
            float projected_value = (x_normalized > 0.5f) ? 1.0f : 0.0f;
            float projected_fp16_val = (projected_value * 2.0f) - 1.0f;  // [0,1] -> [-1,1]
            x[batch_offset + i] = FP16Converter::to_fp16(projected_fp16_val);
            y[batch_offset + i] = FP16Converter::to_fp16(0.0f);
        } else {
            x[batch_offset + i] = FP16Converter::to_fp16(x_new_cont);
        }
    }
}

// GPU内核函数：应用边界约束（DSB算法FP16版本）
// 对应DSB算法的"非弹性墙"约束，匹配DSB_CPU_fp16.cu的[0,1]边界处理
// 将x限制在[0,1]范围内，超出时投影到边界{0,1}并清零动量
__global__ void apply_constraint_kernel(__half* x, __half* y, int num_vars, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;
    for (int i = 0; i < num_vars; ++i) {
        // 将FP16值转换到[0,1]范围进行边界检查（使用FP16Converter）
        float x_val = FP16Converter::from_fp16(x[batch_offset + i]);  // [-1,1]
        float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1]

        // 检查是否超出边界：|x-0.5| > 0.5（匹配CPU版本的条件）
        float distance_from_center = fabsf(x_normalized - 0.5f);

        if (distance_from_center > 0.5f) {
            // 超出边界，投影到边界{0,1}（匹配CPU版本的处理）
            float projected_value = (x_normalized > 0.5f) ? 1.0f : 0.0f;

            // 转换回[-1,1]范围并使用FP16Converter统一转换（参考DSB_GPU_fp8）
            float projected_fp16_val = (projected_value * 2.0f) - 1.0f;  // [0,1] -> [-1,1]
            x[batch_offset + i] = FP16Converter::to_fp16(projected_fp16_val);

            // 清零动量（非弹性墙，使用FP16Converter）
            y[batch_offset + i] = FP16Converter::to_fp16(0.0f);
        }
    }
}

// GPU内核函数：计算单个批次的能量（DSB算法FP16版本 + 共享内存优化）
// 关键修正：使用0/1二值变量进行能量计算，匹配DSB_CPU_fp16.cu第304-348行
// 性能优化：使用共享内存进行归约求和（参考BSB版本）
__global__ void calc_batch_energy_kernel(__half* x, int* x_hat, float* energies,
                                          int* quad_terms_indices, __half* quad_terms_coeffs, __half* linear_terms,
                                          int num_vars, int num_quad_terms, int batch_size,
                                          float scaling_factor, float constant_term, bool is_linear_problem) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    // 共享内存优化：参考BSB版本
    __shared__ float shared_energy[256];
    __shared__ float shared_quad_energy[256];

    // 初始化共享内存
    shared_energy[thread_idx] = 0.0f;
    shared_quad_energy[thread_idx] = 0.0f;
    __syncthreads();

    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_vars;

    // 一阶项贡献：∑_i h_i·x̂_i （匹配DSB_CPU_fp16.cu，使用线程并行）
    for (int i = thread_idx; i < num_vars; i += 256) {
        float h_i = __half2float(linear_terms[i]);

        // 直接使用{0,1}变量（匹配DSB算法标准）
        int x_hat_val = x_hat[batch_offset + i];  // 0或1

        shared_energy[thread_idx] += h_i * static_cast<float>(x_hat_val);
    }

    // 二阶项贡献：0.5 * ∑_{i,j} J_{ij}·x̂_i·x̂_j （匹配DSB_CPU_fp16.cu，使用线程并行）
    for (int t = thread_idx; t < num_quad_terms; t += 256) {
        int var1 = quad_terms_indices[t * 2];
        int var2 = quad_terms_indices[t * 2 + 1];
        float J_ij = __half2float(quad_terms_coeffs[t]);

        // 直接使用{0,1}变量（匹配DSB算法标准）
        int x1_hat = x_hat[batch_offset + var1];  // 0或1
        int x2_hat = x_hat[batch_offset + var2];  // 0或1

        shared_quad_energy[thread_idx] += 0.5f * J_ij * static_cast<float>(x1_hat) * static_cast<float>(x2_hat);
    }

    __syncthreads();

    // 高效的归约求和（参考BSB版本）
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

        // 缩放处理（匹配DSB_CPU_fp16.cu第332-346行）
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

// DSB GPU求解器（FP16版本）
class DSBFP16GPUSolver {
private:
    ProblemModel host_model;
    int batch_size;
    
    // GPU设备指针（DSB算法FP16版本）
    __half* d_x;                // 自旋状态 [batch_size * num_vars]
    __half* d_y;                // 辅助状态 [batch_size * num_vars]
    int* d_x_hat;               // 离散二值化变量{0,1} [batch_size * num_vars]
    __half* d_gradient_term;    // 梯度项 [batch_size * num_vars]
    float* d_energies;          // 能量值 [batch_size]
    
    // 模型数据（FP16版本）
    int* d_quad_terms_indices;  // 二阶项索引对 [num_quad_terms * 2]
    __half* d_quad_terms_coeffs;// 二阶项系数 [num_quad_terms]
    __half* d_linear_terms;     // 一阶项系数 [num_vars]

        // 准备GPU模型数据（DSB算法FP16版本 - 基于DSB_GPU_fp8改进精度）
        void prepare_gpu_model() {
        cout << "DEBUG: Preparing DSB FP16 GPU model..." << endl;
        cout << "  - Precision improvements based on DSB_GPU_fp8" << endl;
        cout << "  - Using FP16Converter for unified conversions" << endl;
        cout << "  - Round-to-nearest-even rounding mode" << endl;
        
        // 展平二阶项索引对
        vector<int> flat_quad_terms_indices;
        for (const auto& term : host_model.quad_terms_indices) {
            flat_quad_terms_indices.insert(flat_quad_terms_indices.end(), term.begin(), term.end());
        }
        
        // 转换系数为FP16（使用FP16Converter统一转换，参考DSB_GPU_fp8）
        vector<__half> fp16_quad_terms_coeffs(host_model.num_quad_terms);
        vector<__half> fp16_linear_terms(host_model.num_vars);
        for (int i = 0; i < host_model.num_quad_terms; ++i) {
            fp16_quad_terms_coeffs[i] = FP16Converter::to_fp16(static_cast<float>(host_model.quad_terms_coeffs[i]));
        }
        for (int i = 0; i < host_model.num_vars; ++i) {
            fp16_linear_terms[i] = FP16Converter::to_fp16(static_cast<float>(host_model.linear_terms[i]));
        }
        
        // 分配GPU内存（DSB算法混合类型：fp16 + float32）
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * host_model.num_vars * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * host_model.num_vars * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x_hat, batch_size * host_model.num_vars * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * host_model.num_vars * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_quad_terms_indices, host_model.num_quad_terms * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_quad_terms_coeffs, host_model.num_quad_terms * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_linear_terms, host_model.num_vars * sizeof(__half)));
        
        // 复制模型数据到GPU
        CUDA_CHECK(cudaMemcpy(d_quad_terms_indices, flat_quad_terms_indices.data(), 
                              flat_quad_terms_indices.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_quad_terms_coeffs, fp16_quad_terms_coeffs.data(), 
                              host_model.num_quad_terms * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_linear_terms, fp16_linear_terms.data(), 
                              host_model.num_vars * sizeof(__half), cudaMemcpyHostToDevice));
        
        cout << "DEBUG: DSB FP16 GPU model prepared." << endl;
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
    DSBFP16GPUSolver(const ProblemModel& problem_model, int bs = 200, float optimized_xi = 0.365f) : host_model(problem_model), batch_size(bs) {
        prepare_gpu_model();
    }

    ~DSBFP16GPUSolver() {
        free_gpu_memory();
    }

    // 核心求解函数（DSB算法FP16版本，参考DSB_CPU_fp16.cu）
    // 关键修正：添加sigma参数，匹配CPU版本的初始化方式
    pair<vector<float>, float> solve(int max_iters = 1000, float sigma = 0.1f) {
        // DSB算法参数（与DSB_CPU_fp16.cu保持一致）
        float xi = 0.365f;   // 优化的耦合强度（来自solve_SB6.py）
        float dt = 0.1f;    // 时间步长，平衡精度与速度
        float delta = 0.9f; // 阻尼系数，确保收敛稳定性

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << host_model.num_vars << ", Quadratic terms: " << host_model.num_quad_terms << endl;
        cout << "  - Linear terms: " << host_model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << host_model.scaling_factor << endl;
        cout << "  - DSB Algorithm (基于DSB_CPU_fp16.cu，使用0/1变量替代连续x)" << endl;
        cout << "  - Objective sense: " << host_model.objective_sense << endl;
        cout << "  - Using optimized xi=" << fixed << setprecision(6) << xi << ", sigma=" << fixed << setprecision(6) << sigma << ", dt=" << fixed << setprecision(6) << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - Precision improvements (based on DSB_GPU_fp8):" << endl;
        cout << "    - FP16Converter: unified conversion interface" << endl;
        cout << "    - Round-to-nearest-even rounding mode" << endl;
        cout << "    - FP16_EPSILON: " << FP16_EPSILON << " (minimum precision unit)" << endl;
        cout << "  - Key feature: Uses ±1 binary variables for dynamics and energy evaluation" << endl;
        cout << "  - GPU acceleration with precision improvements" << endl;
        cout << "  - Optimization: Merged kernels to reduce launch overhead (5->3 kernels/iteration)" << endl;

        // 优化方向判断
        bool is_maximize = (host_model.objective_sense == "maximize");
        bool is_linear_problem = (host_model.num_quad_terms == 0);

        // 计算线程块和网格大小（优化：适配共享内存）
        dim3 blockSize(256);
        dim3 gridSize_batch(batch_size);  // 每个批次一个block，使用共享内存归约
        dim3 gridSize_batches(batch_size);  // 其他kernel
        dim3 gridSize_batch_spins(batch_size, (host_model.num_vars + blockSize.x - 1) / blockSize.x);

        // 初始化批处理数据（传递sigma参数）
        init_batches_kernel<<<gridSize_batch_spins, blockSize>>>(
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

        // 核心迭代（DSB动力学：使用0/1变量，参考DSB_CPU_fp16.cu）
        for (int iter = 0; iter < max_iters; ++iter) {
            // 计算泵浦振幅：p(t) = t/T = iter/(max_iters-1)
            float p = static_cast<float>(iter) / (max_iters - 1);

            // 每次迭代都计算当前时间（用于准确记录收敛时间）
            auto current_time = high_resolution_clock::now();
            float cumulative_time = static_cast<float>(duration_cast<microseconds>(current_time - start).count()) / 1000.0f;

            // DSB动力学更新序列（优化版本：合并kernel减少调用次数）
            // 优化: 从5个kernel减少到3个kernel
            // 1. 计算梯度并即时更新x_hat（确保x_hat在每次迭代时同步更新）
            calc_gradient_with_x_hat_kernel<<<gridSize_batch_spins, blockSize>>>(d_x, d_x_hat, d_gradient_term,
                                                                              d_quad_terms_indices, d_quad_terms_coeffs, d_linear_terms,
                                                                              host_model.num_vars, host_model.num_quad_terms, batch_size,
                                                                              is_maximize);

            // 2. 更新y：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x_hat)
            update_y_kernel<<<gridSize_batches, blockSize>>>(d_x, d_y, d_gradient_term,
                                                    host_model.num_vars, p, xi, dt, delta, batch_size);

            // 3. 更新x并应用边界约束
            update_x_with_constraint_kernel<<<gridSize_batches, blockSize>>>(d_x, d_y,
                                                                          host_model.num_vars, dt, delta, batch_size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // 只在最后一次迭代输出能量结果
            if (iter == max_iters - 1) {
                // 计算x_hat并评估能量
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
        
        cout << "BEST: Optimal objective value: " << fixed << setprecision(3) << best_value << " (GPU FP16 precision)" << endl;

        // 复制最优解回主机
        vector<__half> best_solution(host_model.num_vars);
        CUDA_CHECK(cudaMemcpy(best_solution.data(), d_x + best_idx * host_model.num_vars, 
                              host_model.num_vars * sizeof(__half), cudaMemcpyDeviceToHost));

        // 转换为float输出
        vector<float> solution(host_model.num_vars);
        for (int i = 0; i < host_model.num_vars; ++i) {
            solution[i] = __half2float(best_solution[i]); // 转换为float
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

    // 主函数：DSB算法求解器（参考DSB_CPU_fp16.cu）
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
            optimized_xi = stod(argv[++i]);
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
    cout << "       DSB_GPU_fp16 Solver (Precision Improved)     " << endl;
    cout << "  - File path: " << qplib_file_path << endl;
    cout << "  - Algorithm: DSB (Discrete Simulated Bifurcation)" << endl;
    cout << "  - Key feature: Uses ±1 binary variables for dynamics and energy (matching CPU)" << endl;
    cout << "  - Optimizations: True FP16 storage (IEEE 754 half precision) + GPU parallel" << endl;
    cout << "  - Based on DSB_CPU_fp16.cu with GPU acceleration" << endl;
    cout << "  - Precision improvements (based on DSB_GPU_fp8):" << endl;
    cout << "    - FP16Converter: unified conversion interface" << endl;
    cout << "    - Round-to-nearest-even: __float2half_rn()" << endl;
    cout << "    - FP16_EPSILON: " << FP16_EPSILON << endl;
    cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
    cout << "  - Fixed sigma: 0.1" << endl;
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
        model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
    }

    // 初始化求解器并求解
    try {
        cout << "Using fixed optimized xi value: " << fixed << setprecision(6) << optimized_xi 
             << " (auto-calculated value: " << fixed << setprecision(6) << model.xi << ")" << endl;
        DSBFP16GPUSolver solver(model, 200); // 200 batches
        pair<vector<float>, float> result = solver.solve(1000, 0.1); // 1000 iterations, sigma=0.1

        vector<float> solution = result.first;
        float best_energy = result.second;

        // 匹配DSB_CPU_fp16.cu：输出0/1二值解（从±1转换）
        vector<int> solution_binary(solution.size());
        for (size_t i = 0; i < solution.size(); ++i) {
            // 将FP16值转换到[0,1]范围
            float x_val = solution[i];  // [-1,1]
            float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1]
            solution_binary[i] = (x_normalized > 0.5f) ? 1 : 0;  // 阈值化为0/1
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
        // 输出目标函数值（匹配DSB_CPU_fp16.cu的格式）
        cout << "  DSB GPU OBJECTIVE VALUE: " << fixed << setprecision(6) << best_energy << endl;
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