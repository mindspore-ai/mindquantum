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
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// ============================================================================
// PyTorch风格FP16配置（完全兼容torch.float16）
// ============================================================================
// 参考PyTorch的torch.float16实现，基于IEEE 754半精度浮点标准
constexpr float FP16_MAX_VAL = 65504.0f;   // FP16最大值（PyTorch torch.float16上限）
constexpr float FP16_MIN_VAL = -65504.0f;  // FP16最小值（PyTorch torch.float16下限）
constexpr float FP16_LOWEST_POSITIVE = 6.1035e-5f;  // FP16最小正数（subnormal）
constexpr int FP16_EXP_MAX = 30;  // FP16最大指数偏移（对应IEEE 754 exponent bias = 15，max exp = 30）
constexpr int FP16_MANTISSA_BITS = 10;  // FP16尾数位数
constexpr int FP16_EXPONENT_BITS = 5;   // FP16指数位数
constexpr int FP16_EXP_BIAS = 15;      // FP16指数偏移（IEEE 754）

// 使用这些常量进行编译时检查（避免未使用警告）
static_assert(FP16_EXP_MAX == 30, "FP16_EXP_MAX must be 30 for IEEE 754 compatibility");
static_assert(FP16_MANTISSA_BITS == 10, "FP16_MANTISSA_BITS must be 10 for IEEE 754");
static_assert(FP16_EXPONENT_BITS == 5, "FP16_EXPONENT_BITS must be 5 for IEEE 754");
static_assert(FP16_EXP_BIAS == 15, "FP16_EXP_BIAS must be 15 for IEEE 754");
static_assert(FP16_LOWEST_POSITIVE > 0.0f, "FP16_LOWEST_POSITIVE must be positive");
static_assert(FP16_MAX_VAL > 0.0f, "FP16_MAX_VAL must be positive");
static_assert(FP16_MIN_VAL < 0.0f, "FP16_MIN_VAL must be negative");



// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << #call << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 手动实现clamp函数（兼容C++11/14）
template<typename T>
__host__ __device__ T clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// FP16版本clamp函数（PyTorch风格，使用CUDA __half）
__host__ __device__ __half clamp_half(__half val, __half min_val, __half max_val) {
    if (__hlt(val, min_val)) return min_val;
    if (__hgt(val, max_val)) return max_val;
    return val;
}

// float版本clamp函数
__host__ __device__ float clamp_float(float val, float min_val, float max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// ============================================================================
// PyTorch风格FP16转换函数（完全兼容torch.float16）
// ============================================================================
// 这些函数与PyTorch的.to(torch.float16)和.to(torch.float32)行为一致
// CUDA的__half类型内部使用IEEE 754 binary16格式，与PyTorch torch.float16完全兼容
__host__ __device__ inline float fp16_to_float(__half x) {
    return __half2float(x);
}

__host__ __device__ inline __half float_to_fp16(float f) {
    return __float2half(f);
}

// 直接操作uint16_t的转换（用于更底层的操作，完全兼容PyTorch位操作）
// 这些函数与PyTorch的torch.bitwise_*操作兼容
__host__ __device__ inline uint16_t fp16_to_bits(__half h) {
    return __half_as_ushort(h);
}

__host__ __device__ inline __half bits_to_fp16(uint16_t bits) {
    return __ushort_as_half(bits);
}

// ============================================================================
// PyTorch风格符号计算函数（完全兼容torch.sign和二值化操作）
// ============================================================================

// 符号计算（PyTorch风格，基于x>0.5的二值化）
// 对应PyTorch的 (x > 0.5).float() 二值化操作
__host__ __device__ inline int sign_from_continuous(__half x) {
    // 将FP16值转换到[0,1]范围，然后根据x>0.5进行二值化
    float x_val = fp16_to_float(x);  // 直接获取浮点值（已在[-1,1]）
    float x_normalized = (x_val + 1.0f) / 2.0f;   // [-1,1] -> [0,1]
    return (x_normalized > 0.5f) ? 1 : -1;       // x > 0.5 -> 1, else -> 0 (转换为±1)
}

// FP16符号计算（PyTorch风格，直接基于浮点值）
// 对应PyTorch的 torch.sign(x)
__host__ __device__ inline int sign_fp16(__half x) {
    float x_val = fp16_to_float(x);  // 直接获取浮点值
    return (x_val >= 0.0f) ? 1 : -1;
}

// ============================================================================
// GPU批处理数据结构（PyTorch风格，使用torch.float16存储，float计算）
// ============================================================================
// 类似于PyTorch的Tensor数据布局
struct GPUBatchData {
    __half* x;              // 自旋位置状态（torch.float16, 范围[-1,1]）
    __half* y;              // 动量状态（torch.float16）
    int* x_sign;            // 自旋符号（1/-1）
    __half* gradient_term;  // 梯度项（torch.float16）
    float energy;            // 能量值（使用float避免FP16累加误差）
};

// ============================================================================
// GPU上的BSB模型数据（PyTorch风格，使用torch.float16存储，float计算）
// ============================================================================
struct GPUBSBModel {
    int* second_order_terms;  // 二阶项 [num_second_order * 2]
    __half* second_order_coeff; // 二阶项系数（torch.float16）
    __half* one_order_coeff;  // 一阶项系数（torch.float16）
    int num_spins;
    int num_second_order;
};

// parse_qplib_file函数已在qplib_parser.h中定义

// ============================================================================
// GPU内核函数（PyTorch风格，使用torch.float16）
// ============================================================================

// 初始化批处理数据（PyTorch风格，BSB算法，使用torch.float16）
// BSB特征：使用连续变量x进行梯度计算，但能量计算仍需二元解
__global__ void init_batches_kernel(__half* x, __half* y, int* x_sign, __half* gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;
    
    // BSB初始化：连续变量在[0,1]范围内（PyTorch风格）
    const float sigma = 0.1f;  // 固定sigma值
    for (int i = 0; i < num_spins; ++i) {
        // 连续初始化：sigma * N(0,1) + 0.5，转换为torch.float16
        float x_normalized = sigma * curand_normal(&state) + 0.5f;  // [0,1]范围
        x_normalized = fmaxf(0.0f, fminf(1.0f, x_normalized));      // 限制在[0,1]
        float x_fp16_val = (x_normalized * 2.0f) - 1.0f;           // 转换到[-1,1]
        x_fp16_val = fmaxf(FP16_MIN_VAL, fminf(FP16_MAX_VAL, x_fp16_val));
        x[batch_offset + i] = float_to_fp16(x_fp16_val);  // 存储为torch.float16

        // y初始化：sigma * N(0,1)
        float y_val = sigma * curand_normal(&state);
        y[batch_offset + i] = float_to_fp16(y_val);

        // 为能量计算准备二元值（基于x>0.5的二值化，PyTorch风格）
        x_sign[batch_offset + i] = sign_from_continuous(x[batch_offset + i]);
        gradient_term[batch_offset + i] = float_to_fp16(0.0f);
    }
}

// 计算自旋符号（PyTorch风格）
__global__ void calc_sign_kernel(__half* x, int* x_sign, int num_spins, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    for (int i = 0; i < num_spins; ++i) {
        // 使用sign_fp16函数避免无符号比较警告
        x_sign[batch_offset + i] = sign_fp16(x[batch_offset + i]);
    }
}

// 梯度计算内核：使用共享内存和合并访问（PyTorch风格，使用torch.float16）
// BSB特征：使用连续变量x进行梯度计算
__global__ void calc_gradient_kernel_optimized(__half* x, __half* gradient_term,
                                               int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;
    
    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    
    // 初始化梯度为一阶项系数
    __half grad = one_order_coeff[spin_idx];
    
    // 使用局部变量减少全局内存访问（移除未使用的x_val）
    for (int t = 0; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        __half J_ij_half = __hmul(second_order_coeff[t], __float2half(0.5f));
        
        if (var1 == spin_idx) {
            // 直接使用torch.float16计算，减少转换
            __half x2_val = x[batch_offset + var2];
            // 近似转换：[-1,1] -> [0,1]，使用torch.float16运算
            __half x2_norm_approx = __hadd(__hmul(x2_val, float_to_fp16(0.5f)), float_to_fp16(0.5f));
            grad = __hadd(grad, __hmul(J_ij_half, x2_norm_approx));
        } else if (var2 == spin_idx) {
            __half x1_val = x[batch_offset + var1];
            __half x1_norm_approx = __hadd(__hmul(x1_val, float_to_fp16(0.5f)), float_to_fp16(0.5f));
            grad = __hadd(grad, __hmul(J_ij_half, x1_norm_approx));
        }
    }
    
    gradient_term[batch_offset + spin_idx] = grad;
}

// y更新内核：优化版本（PyTorch风格，使用torch.float16存储，float计算）
// BSB公式：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x)
__global__ void update_y_kernel_optimized(__half* x, __half* y, __half* gradient_term,
                                          int num_spins, float p, float xi, float dt, int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 泵浦振幅项：-(1-p(t))，使用float计算
    const float pump_factor = -(1.0f - p);
    const float sign_factor = is_maximize ? xi : -xi;

    // 将x转换到[0,1]范围（float）
    float x_val = fp16_to_float(x[batch_offset + spin_idx]);  // torch.float16 -> float
    float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1]
    float x_shifted = x_normalized - 0.5f;

    // BSB更新公式的float版本：y += (pump_factor * x_shifted + sign_factor * grad) * dt
    float grad_val = fp16_to_float(gradient_term[batch_offset + spin_idx]);
    float y_val = fp16_to_float(y[batch_offset + spin_idx]);
    float update = (pump_factor * x_shifted + sign_factor * grad_val) * dt;
    y[batch_offset + spin_idx] = float_to_fp16(y_val + update);  // float -> torch.float16
}

// 更新自旋x（BSB公式，优化版本）
// BSB公式：dx/dt = y
__global__ void update_x_kernel(__half* x, __half* y, int num_spins, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 将torch.float16值转换到[0,1]范围进行更新（每个线程处理一个自旋）
    float x_val = fp16_to_float(x[batch_offset + spin_idx]);  // 直接获取浮点值（已在[-1,1]）
    float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1]

    // 位置更新：x += y * dt
    float x_new = x_normalized + fp16_to_float(y[batch_offset + spin_idx]) * dt;

    // 限制在[0,1]范围并转换为torch.float16
    float x_normalized_new = fmaxf(0.0f, fminf(1.0f, x_new));  // 限制在[0,1]
    float x_fp16_new = (x_normalized_new * 2.0f) - 1.0f;  // [0,1] -> [-1,1]
    x_fp16_new = fmaxf(FP16_MIN_VAL, fminf(FP16_MAX_VAL, x_fp16_new));
    x[batch_offset + spin_idx] = float_to_fp16(x_fp16_new);  // 存储为torch.float16
}

// 应用边界约束（边界处理，优化版本）
__global__ void apply_constraint_kernel(__half* x, __half* y, int num_spins, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 将FP16值转换到[0,1]范围
    float x_val = fp16_to_float(x[batch_offset + spin_idx]);  // 直接获取浮点值（已在[-1,1]）
    float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1]

    // 检查是否超出边界：|x-0.5| > 0.5
    if (x_normalized < 0.0f || x_normalized > 1.0f) {
        // 超出边界，投影到边界{0,1}（优化版本）
        float projected_value = (x_normalized > 0.5f) ? 1.0f : 0.0f;

        // 转换回[-1,1]范围并存储为torch.float16
        float projected_fp16_val = (projected_value * 2.0f) - 1.0f;  // [0,1] -> [-1,1]
        projected_fp16_val = fmaxf(FP16_MIN_VAL, fminf(FP16_MAX_VAL, projected_fp16_val));
        x[batch_offset + spin_idx] = float_to_fp16(projected_fp16_val);  // 存储为torch.float16

        // 清零动量（非弹性墙）
        y[batch_offset + spin_idx] = float_to_fp16(0.0f);
    }
}

// 计算单个批次的能量（BSB实现，PyTorch风格，float计算）
__global__ void calc_batch_energy_kernel(__half* x, int* x_sign, float* energies,
                                          int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                          int num_spins, int num_second_order, int batch_size, bool is_linear_problem) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    float scaled_energy = 0.0f;

    // 二次项贡献：0.5 * ∑_{i,j} J_{ij}x_i x_j（使用连续变量）
    if (num_second_order > 0) {
        for (int t = 0; t < num_second_order; ++t) {
            int var1 = second_order_terms[t * 2];
            int var2 = second_order_terms[t * 2 + 1];
            float J_ij = __half2float(second_order_coeff[t]);

            // 使用连续决策变量（[0,1]，BSB能量计算）
            float x1_val = fp16_to_float(x[batch_offset + var1]);  // 直接获取浮点值（已在[-1,1]）
            float x2_val = fp16_to_float(x[batch_offset + var2]);  // 直接获取浮点值（已在[-1,1]）
            float x1_continuous = (x1_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1] 连续变量
            float x2_continuous = (x2_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1] 连续变量

            scaled_energy += 0.5f * J_ij * x1_continuous * x2_continuous;
        }
    }

    // 线性项贡献：∑_i h_i x_i（使用连续变量）
    for (int i = 0; i < num_spins; ++i) {
        float x_val = fp16_to_float(x[batch_offset + i]);  // 直接获取浮点值（已在[-1,1]）
        float x_continuous = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1] 连续变量
        scaled_energy += __half2float(one_order_coeff[i]) * x_continuous;
    }

    // 注意：这里返回缩放后的能量，最终缩放会在host端完成
    energies[batch_idx] = scaled_energy;
}

// ============================================================================
// BSB GPU求解器（PyTorch风格，使用torch.float16进行存储和计算）
// ============================================================================
class BSBFP16CorrectedGPUSolver {
private:
    ProblemModel model;
    int batch_size;

    // GPU设备指针（扁平化存储，PyTorch风格，使用torch.float16存储，float计算）
    __half* d_x;                // 自旋位置状态 [batch_size * num_spins]
    __half* d_y;                // 动量状态 [batch_size * num_spins]
    int* d_x_sign;              // 自旋符号 [batch_size * num_spins]
    __half* d_gradient_term;    // 梯度项 [batch_size * num_spins]
    float* d_energies;         // 能量值 [batch_size]（float，不使用double）

    // 模型数据（PyTorch风格，使用torch.float16存储，float计算）
    int* d_second_order_terms;  // 二阶项 [num_second_order * 2]
    __half* d_second_order_coeff; // 二阶项系数（torch.float16）
    __half* d_one_order_coeff;  // 一阶项系数（torch.float16）

    // 准备GPU模型数据（PyTorch风格，使用torch.float16）
    void prepare_gpu_model() {
        cout << "Preparing PyTorch-style torch.float16 GPU model..." << endl;

        // 从ProblemModel转换数据格式
        vector<int> flat_second_order_terms;
        for (int i = 0; i < model.num_quad_terms; ++i) {
            flat_second_order_terms.insert(flat_second_order_terms.end(),
                                       model.quad_terms_indices[i].begin(),
                                       model.quad_terms_indices[i].end());
        }

        // 转换系数为torch.float16（完全兼容PyTorch）
        vector<__half> fp16_second_order_coeff(model.num_quad_terms);
        vector<__half> fp16_one_order_coeff(model.num_vars);
        for (int i = 0; i < model.num_quad_terms; ++i) {
            fp16_second_order_coeff[i] = float_to_fp16(static_cast<float>(model.quad_terms_coeffs[i]));
        }
        for (int i = 0; i < model.num_vars; ++i) {
            fp16_one_order_coeff[i] = float_to_fp16(static_cast<float>(model.linear_terms[i]));
        }

        // 分配GPU内存
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * model.num_vars * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * model.num_vars * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x_sign, batch_size * model.num_vars * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * model.num_vars * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_second_order_terms, model.num_quad_terms * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, model.num_quad_terms * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, model.num_vars * sizeof(__half)));

        // 复制模型数据到GPU
        CUDA_CHECK(cudaMemcpy(d_second_order_terms, flat_second_order_terms.data(),
                              flat_second_order_terms.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_second_order_coeff, fp16_second_order_coeff.data(),
                              model.num_quad_terms * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_one_order_coeff, fp16_one_order_coeff.data(),
                              model.num_vars * sizeof(__half), cudaMemcpyHostToDevice));

        cout << "PyTorch-style torch.float16 GPU model prepared." << endl;
    }

    // 释放GPU内存
    void free_gpu_memory() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_x_sign));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_energies));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
    }

public:
    BSBFP16CorrectedGPUSolver(const ProblemModel& problem_model, int bs = 200) : model(problem_model), batch_size(bs) {
        prepare_gpu_model();
    }

    ~BSBFP16CorrectedGPUSolver() {
        free_gpu_memory();
    }

    // 核心求解函数（BSB实现，float计算）
    vector<__half> solve(int max_iters = 1000, float optimized_xi = 0.365f) {
        cout << "DEBUG: Entering solve() function..." << endl;
        cout.flush();

        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << model.scaling_factor << endl;
        cout.flush();

        // BSB超参数（float精度）
        float xi = optimized_xi;  // 使用优化的xi值
        float dt = 0.1f;           // 时间步长
        float delta = 0.9f;         // 阻尼系数

        cout << "  - BSB Algorithm (PyTorch-style FP16, Float Computation)" << endl;
        cout << "  - Data type: PyTorch torch.float16 (IEEE 754 half precision) for storage" << endl;
        cout << "  - Computation type: float (no double)" << endl;
        cout << "  - Features:" << endl;
        cout << "    * IEEE 754 binary16 format (1 sign, 5 exp, 10 mantissa)" << endl;
        cout << "    * Round-to-nearest-even (RNTE) rounding mode" << endl;
        cout << "    * Fully compatible with PyTorch tensor operations" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using optimized xi=" << fixed << setprecision(6) << xi << ", sigma=0.1, dt=" << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout.flush();
        
        // 确定优化方向
        bool is_maximize = (model.objective_sense == "maximize");
        bool is_linear_problem = (model.num_quad_terms == 0);

        // 计算线程块和网格大小（与INT8/FP8版本保持一致）
        dim3 blockSize(256);
        dim3 gridSize(batch_size, (model.num_vars + blockSize.x - 1) / blockSize.x);

        // 初始化批处理数据
        init_batches_kernel<<<gridSize, blockSize>>>(
            d_x, d_y, d_x_sign, d_gradient_term,
            model.num_vars, model.num_quad_terms, 
            time(nullptr), batch_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时开始
        auto start = high_resolution_clock::now();

        // 核心迭代（BSB动力学）
        for (int iter = 0; iter < max_iters; ++iter) {
            float p = static_cast<float>(iter) / (max_iters - 1);

            // BSB算法：直接使用连续变量x计算梯度，不需要计算sign
            calc_gradient_kernel_optimized<<<gridSize, blockSize>>>(d_x, d_gradient_term,
                                                                      d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                                                                      model.num_vars, model.num_quad_terms, batch_size);

            // 更新动量：dy/dt = -(1-p(t))*(x-0.5) ± ξ*∇f(x)
            update_y_kernel_optimized<<<gridSize, blockSize>>>(d_x, d_y, d_gradient_term,
                                                                model.num_vars, p, xi, dt, batch_size, is_maximize);
            
            // 更新位置：dx/dt = y
            update_x_kernel<<<gridSize, blockSize>>>(d_x, d_y, model.num_vars, dt, delta, batch_size);

            // 边界条件处理：将x限制在[0,1]范围内，超出时投影到边界并清零动量
            apply_constraint_kernel<<<gridSize, blockSize>>>(d_x, d_y, model.num_vars, batch_size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 计时结束
        auto end = high_resolution_clock::now();
        float duration = duration_cast<microseconds>(end - start).count() / 1000.0f;

        // 计算最终能量并找到最优批次
        calc_batch_energy_kernel<<<gridSize, blockSize>>>(d_x, d_x_sign, d_energies,
                                                                      d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                                                                      model.num_vars, model.num_quad_terms, batch_size, is_linear_problem);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 复制能量数据回主机并完成最终计算
        vector<float> energies(batch_size);
        CUDA_CHECK(cudaMemcpy(energies.data(), d_energies, batch_size * sizeof(float), cudaMemcpyDeviceToHost));

        // 完成最终能量计算（缩放和常数项）
        for (auto& energy : energies) {
            if (is_linear_problem) {
                // 纯线性问题，不进行缩放逆变换
            } else {
                // 重新缩放回原始能量
                energy *= static_cast<float>(model.scaling_factor);
            }
            // 添加常数项（如果存在）
            if (model.constant_term != 0.0) {
                energy += static_cast<float>(model.constant_term);
            }
        }

        // 根据优化方向找到最优批次
        int best_idx = is_maximize ?
            max_element(energies.begin(), energies.end()) - energies.begin() :
            min_element(energies.begin(), energies.end()) - energies.begin();

        // 复制最优解回主机
        vector<__half> best_solution(model.num_vars);
        CUDA_CHECK(cudaMemcpy(best_solution.data(), d_x + best_idx * model.num_vars, 
                              model.num_vars * sizeof(__half), cudaMemcpyDeviceToHost));

        // 转换为二值化解（存储为torch.float16格式，值为0.0f或1.0f）
        vector<__half> fp16_solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            // 关键：基于x>0.5进行二值化，存储为torch.float16格式
            float x_val = fp16_to_float(best_solution[i]);  // 从torch.float16获取浮点值
            float x_normalized = (x_val + 1.0f) / 2.0f;  // [-1,1] -> [0,1]
            float binary_val = (x_normalized > 0.5f) ? 1.0f : 0.0f;  // 二值化为1.0或0.0
            fp16_solution[i] = float_to_fp16(binary_val);  // 存储为torch.float16格式
        }

        cout << "DEBUG: FP16 solution generated successfully!" << endl;
        cout << "DEBUG: Returning from solve() function..." << endl;
        cout.flush();

        return fp16_solution;
    }

    // 获取自旋数
    int get_num_spins() const {
        return model.num_vars;
    }
};

// 主函数：BSB算法求解器
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

    // 优化后的xi值
    float optimized_xi = 0.365f;

    // 可选：通过命令行参数覆盖xi值
    if (argc > 2) {
        optimized_xi = stof(argv[2]);
    }

        cout << "========================================" << endl;
        cout << "       BSB_GPU_fp16 Solver (PyTorch-style, Float Computation)     " << endl;
        cout << "  - File path: " << qplib_file_path << endl;
        cout << "  - Algorithm: BSB (Ballistic Simulated Bifurcation)" << endl;
        cout << "  - Key feature: Uses continuous x for gradient calculation" << endl;
        cout << "  - Storage: PyTorch torch.float16 (IEEE 754 half precision) + GPU parallel" << endl;
        cout << "  - Computation: Float (no double)" << endl;
        cout << "  - Compatibility: Compatible with PyTorch tensor operations" << endl;
        cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
        cout << "  - Fixed sigma: 0.1" << endl;
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
    
    // 手动输出各个字段，确保能正确显示
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
        cout << "DEBUG: Creating BSB GPU solver with 100 batches..." << endl;
        cout.flush();
        
        // 直接使用ProblemModel进行GPU计算
        BSBFP16CorrectedGPUSolver solver(model, 200); // 200 batches，提高性能
        cout << "DEBUG: GPU solver created successfully!" << endl;
        cout.flush();
        
        cout << "Starting GPU optimization..." << endl;
        cout.flush();

        // 使用800次迭代，优化的xi值
        auto start = high_resolution_clock::now();
        vector<__half> fp16_solution = solver.solve(1000, optimized_xi);
        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;
        cout << "TIME: Solving completed, total time: " << fixed << setprecision(6) << duration << " ms" << endl;
        cout.flush();
        
        // 计算目标函数值（能量计算方式，使用fp16解，float计算）
        float scaled_energy = 0.0f;
        bool is_linear_problem = (model.num_quad_terms == 0);

        // 二次项贡献：0.5 * ∑_{i,j} J_{ij}x_i x_j
        if (model.num_quad_terms > 0) {
            for (int t = 0; t < model.num_quad_terms; ++t) {
                int var1 = model.quad_terms_indices[t][0];
                int var2 = model.quad_terms_indices[t][1];
                float J_ij = static_cast<float>(model.quad_terms_coeffs[t]);

                // 使用torch.float16二值化解（0.0f/1.0f转换为0.0/1.0）
                float x1_val = (fp16_to_float(fp16_solution[var1]) > 0.5f) ? 1.0f : 0.0f;
                float x2_val = (fp16_to_float(fp16_solution[var2]) > 0.5f) ? 1.0f : 0.0f;

                scaled_energy += 0.5f * J_ij * x1_val * x2_val;
            }
        }

        // 线性项贡献：∑_i h_i x_i
        for (int i = 0; i < model.num_vars; ++i) {
            float x_val = (fp16_to_float(fp16_solution[i]) > 0.5f) ? 1.0f : 0.0f;
            scaled_energy += static_cast<float>(model.linear_terms[i]) * x_val;
        }

        // 缩放处理
        float objective_value;
        if (is_linear_problem) {
            // 纯线性问题，不进行缩放逆变换
            objective_value = scaled_energy;
        } else {
            // 重新缩放回原始能量
            objective_value = scaled_energy * static_cast<float>(model.scaling_factor);
        }

        // 添加常数项（如果存在）
        if (model.constant_term != 0.0) {
            objective_value += static_cast<float>(model.constant_term);
        }

        // 输出二元解表示（0/1，存储为torch.float16但显示为整数）
        cout << "\nRESULT: Solution (" << model.num_vars << " variables, binary values [0/1] stored as torch.float16): " << endl;
        int count_ones = 0;
        for (size_t i = 0; i < fp16_solution.size(); ++i) {
            // 从torch.float16值获取二进制表示
            float val = fp16_to_float(fp16_solution[i]);
            int binary_val = (val > 0.5f) ? 1 : 0;  // 二值化为0/1
            cout << binary_val;
            if (binary_val == 1) count_ones++;
            if ((i + 1) % 16 == 0) cout << endl;
            else if (i != fp16_solution.size() - 1) cout << ", ";
        }
        cout << endl << endl;
        
        cout << "Solution statistics:" << endl;
        cout << "  - Variables set to 1: " << count_ones << "/" << fp16_solution.size() << endl;
        cout << "  - Variables set to 0: " << (fp16_solution.size() - count_ones) << "/" << fp16_solution.size() << endl;
        cout << "  - Solution density: " << fixed << setprecision(2) << (count_ones * 100.0 / fp16_solution.size()) << "%" << endl;
        cout << endl;
        
        // 输出最终目标函数值（更醒目的格式）
        cout << "========================================" << endl;
        cout << "  ** BSB GPU FINAL RESULTS (Float Computation) **" << endl;
        cout << "========================================" << endl;
        cout << "  Problem: " << model.problem_name << endl;
        cout << "  Objective sense: " << model.objective_sense << endl;
        cout << "  Number of variables: " << model.num_vars << endl;
        cout << "  Number of quadratic terms: " << model.num_quad_terms << endl;
        cout << "----------------------------------------" << endl;
        cout << "  BSB GPU OBJECTIVE VALUE: " << fixed << setprecision(6) << objective_value << endl;
        cout << "  Storage: torch.float16, Computation: Float (no double)" << endl;
        cout << "  Optimized xi value: " << fixed << setprecision(6) << optimized_xi << endl;
        cout << "========================================" << endl;
        
        // 强制刷新输出缓冲区
        cout.flush();
        
    } catch (const exception& e) {
        cerr << "ERROR: Exception during GPU solver initialization: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "ERROR: Unknown exception during GPU solver initialization" << endl;
        return 1;
    }
    
    return 0;
}