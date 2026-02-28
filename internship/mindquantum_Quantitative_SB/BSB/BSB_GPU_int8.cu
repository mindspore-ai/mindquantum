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
#include <cuda_device_runtime_api.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <exception>
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// PyTorch风格INT8量化配置
// 参考PyTorch的int8量化：对称范围[-128, 127]，线性缩放



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

// PyTorch风格INT8量化函数
__host__ __device__ inline int8_t float_to_int8(float f) {
    // 线性量化：float -> int8，范围[-1.0, 1.0] -> [-128, 127]
    float scaled = f * 127.0f;
    scaled = fmaxf(-128.0f, fminf(127.0f, scaled));  // 限制在[-128, 127]
    return static_cast<int8_t>(roundf(scaled));
}

__host__ __device__ inline float int8_to_float(int8_t x) {
    // 反量化：int8 -> float，范围[-128, 127] -> [-1.0, 1.0]
    return static_cast<float>(x) / 127.0f;
}

// GPU批处理数据结构（混合精度：计算用float，存储前量化到INT8）
// 关键特征：x,y,gradient_term全部使用float计算，避免double开销
struct GPUBatchData {
    float* x_float;        // 位置状态（float，用于计算）
    float* y;              // 动量状态（float，避免double开销）
    float* gradient_term;   // 梯度项（float，避免double开销）
    float energy;           // 目标函数值（float）
};

// ProblemModel结构体已在qplib_parser.h中定义

// GPU上的BSB模型数据（使用float精度，避免double开销）
struct GPUBSBModel {
    int* second_order_terms;     // 二阶项 [num_second_order * 2]
    float* second_order_coeff;   // 二阶项系数（float，避免double开销）
    float* one_order_coeff;       // 一阶项系数（float，避免double开销）
    int num_spins;
    int num_second_order;
};

// parse_qplib_file函数已在qplib_parser.h中定义

// GPU内核函数：初始化批处理数据（float计算版本）
// BSB特征：x,y,gradient_term全部使用float计算，避免double开销
__global__ void init_batches_kernel(float* x_float, float* y, float* gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size, double sigma = 0.1) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;

    // 初始化：全部使用float存储和计算
    for (int i = 0; i < num_spins; ++i) {
        // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5
        float x_normalized = static_cast<float>(sigma) * curand_normal_double(&state) + 0.5f;  // [0,1]范围
        x_normalized = fmaxf(0.0f, fminf(1.0f, x_normalized));        // 限制在[0,1]

        // 转换到[-1.0,1.0]范围并存储为float
        float x_range = x_normalized * 2.0f - 1.0f;  // [0,1] -> [-1.0,1.0]
        x_float[batch_offset + i] = x_range;  // 存储为float

        // y初始化为小随机值：sigma * N(0,1)（float精度）
        float y_val = static_cast<float>(sigma) * curand_normal_double(&state);
        y[batch_offset + i] = y_val;

        // 初始化梯度项（float精度）
        gradient_term[batch_offset + i] = 0.0f;
    }
}

// GPU内核函数：计算梯度（float计算版本）
// BSB特征：x,gradient_term全部使用float计算
__global__ void calc_gradient_kernel_optimized(float* x_float, float* gradient_term,
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
        float J_ij = second_order_coeff[t] * 0.5f;  // 系数除以2，使用float

        if (var1 == spin_idx) {
            float x2_val = x_float[batch_offset + var2];
            float x2_continuous = (x2_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
            grad += J_ij * x2_continuous;
        } else if (var2 == spin_idx) {
            float x1_val = x_float[batch_offset + var1];
            float x1_continuous = (x1_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
            grad += J_ij * x1_continuous;
        }
    }

    gradient_term[batch_offset + spin_idx] = grad;
}



// GPU内核函数：更新辅助状态y（float计算版本）
// BSB公式：dy/dt = -(1-p(t))*(x-0.5) ± ξ*∇f(x)
// 关键特征：x,y,gradient_term全部使用float计算
__global__ void update_y_kernel_optimized(float* x_float, float* y, float* gradient_term,
                                int num_spins, float p, float xi, float dt, int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 泵浦振幅项：-(1-p(t))，使用float计算
    const float pump_factor = -(1.0f - p);

    // 将float x转换到[0,1]范围
    float x_fp8_val = x_float[batch_offset + spin_idx];  // float [-1.0,1.0]
    float x_normalized = (x_fp8_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
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

// GPU内核函数：更新自旋x（float计算版本）
__global__ void update_x_kernel_optimized(float* x_float, float* y, int num_spins, float dt, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 将float x转换到[0,1]空间进行更新
    float x_fp8_val = x_float[batch_offset + spin_idx];  // float [-1.0,1.0]
    float x_normalized = (x_fp8_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
    x_normalized = fmaxf(0.0f, fminf(1.0f, x_normalized));  // 确保在[0,1]

    x_normalized += y[batch_offset + spin_idx] * dt;

    // 更新为float：先限制在[0,1]，再转换到[-1.0,1.0]
    x_normalized = fmaxf(0.0f, fminf(1.0f, x_normalized));
    float x_fp8_new = x_normalized * 2.0f - 1.0f;  // [0,1] -> [-1.0,1.0]
    x_float[batch_offset + spin_idx] = x_fp8_new;  // 存储为float
}

// GPU内核函数：应用边界约束（float计算版本）
// 将x限制在[0,1]范围内，超出时投影到边界{0,1}并清零动量
__global__ void apply_constraint_kernel(float* x_float, float* y, int num_spins, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    for (int i = 0; i < num_spins; ++i) {
        // 将float x转换到[0,1]范围检查边界
        float x_fp8_val = x_float[batch_offset + i];  // float [-1.0,1.0]
        float x_normalized = (x_fp8_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]

        // 检查是否超出边界（使用float）
        if (x_normalized < 0.0f || x_normalized > 1.0f) {
            // 超出边界，投影到极值：0 -> -1.0, 1 -> 1.0
            float projected_value = (x_normalized < 0.5f) ? 0.0f : 1.0f;  // 投影到{0,1}
            float projected_fp8_range = projected_value * 2.0f - 1.0f;  // [0,1] -> [-1.0,1.0]
            x_float[batch_offset + i] = projected_fp8_range;  // 存储为float

            // 清零动量（非弹性墙，float精度）
            y[batch_offset + i] = 0.0f;
        }
    }
}

// GPU内核函数：在设备端计算能量值（float计算版本）
__global__ void calculate_energy_kernel_optimized(float* x_float,
                                            int* second_order_terms, float* second_order_coeff,
                                            float* one_order_coeff, float* energies,
                                            int num_spins, int num_second_order, int batch_size,
                                            float scaling_factor, float constant_term, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int thread_idx = threadIdx.x;

    __shared__ float shared_energy[256];
    __shared__ float shared_quad_energy[256];

    // 初始化共享内存
    shared_energy[thread_idx] = 0.0f;
    shared_quad_energy[thread_idx] = 0.0f;
    __syncthreads();

    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 线性项贡献：使用连续变量
    for (int i = thread_idx; i < num_spins; i += 256) {
        float coeff = one_order_coeff[i];
        float x_fp8_val = x_float[batch_offset + i];  // float [-1.0,1.0]
        float x_continuous = (x_fp8_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1] 连续变量
        shared_energy[thread_idx] += coeff * x_continuous;
    }

    // 二次项贡献：使用连续变量
    for (int t = thread_idx; t < num_second_order; t += 256) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        float J_ij = second_order_coeff[t];

        float x1_fp8_val = x_float[batch_offset + var1];
        float x2_fp8_val = x_float[batch_offset + var2];
        float x1_continuous = (x1_fp8_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1] 连续变量
        float x2_continuous = (x2_fp8_val + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1] 连续变量
        shared_quad_energy[thread_idx] += 0.5f * J_ij * x1_continuous * x2_continuous;
    }

    __syncthreads();

    // 高效的归约求和
    for (int s = 128; s > 0; s >>= 1) {
        if (thread_idx < s) {
            shared_energy[thread_idx] += shared_energy[thread_idx + s];
            shared_quad_energy[thread_idx] += shared_quad_energy[thread_idx + s];
        }
        __syncthreads();
    }

    // 合并线性项和二次项
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

// BSB GPU求解器（Float计算版本）
class BSBInt8CorrectedGPUSolver {
private:
    ProblemModel model;
    int batch_size;
    double dynamic_scale;      // 动态缩放因子，用于精度恢复

    // GPU设备指针（float计算版本）
    float* d_x_float;               // 位置状态 [batch_size * num_spins] (float精度)
    float* d_y;                     // 动量状态 [batch_size * num_spins] (float精度)
    float* d_gradient_term;          // 梯度项 [batch_size * num_spins] (float精度)

    // 模型数据（float精度版本）
    int* d_second_order_terms;       // 二阶项 [num_second_order * 2]
    float* d_second_order_coeff;     // 二阶项系数 [num_second_order] (float)
    float* d_one_order_coeff;        // 一阶项系数 [num_spins] (float)

    // 准备GPU模型数据（float计算版本）
    void prepare_gpu_model() {
        cout << "DEBUG: Preparing Float computation GPU model..." << endl;

        // 展平二阶项
        vector<int> flat_second_order_terms;
        for (int i = 0; i < model.num_quad_terms; ++i) {
            flat_second_order_terms.push_back(model.quad_terms_indices[i][0]);
            flat_second_order_terms.push_back(model.quad_terms_indices[i][1]);
        }

        // 使用float精度系数（避免double开销）
        vector<float> second_order_coeff(model.num_quad_terms);
        vector<float> one_order_coeff(model.num_vars);

        for (int i = 0; i < model.num_quad_terms; ++i) {
            second_order_coeff[i] = static_cast<float>(model.quad_terms_coeffs[i]);
        }
        for (int i = 0; i < model.num_vars; ++i) {
            one_order_coeff[i] = static_cast<float>(model.linear_terms[i]);
        }

        cout << "DEBUG: Using Float for x/y/gradient computation (no double)" << endl;
        cout << "DEBUG: Quantization mapping: [0,1] -> [-127,127], threshold at 0.5" << endl;

        // 分配GPU内存（float计算架构）
        CUDA_CHECK(cudaMalloc(&d_x_float, batch_size * model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * model.num_vars * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_second_order_terms, model.num_quad_terms * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, model.num_quad_terms * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, model.num_vars * sizeof(float)));

        // 复制模型数据到GPU
        CUDA_CHECK(cudaMemcpy(d_second_order_terms, flat_second_order_terms.data(),
                              flat_second_order_terms.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_second_order_coeff, second_order_coeff.data(),
                              model.num_quad_terms * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_one_order_coeff, one_order_coeff.data(),
                              model.num_vars * sizeof(float), cudaMemcpyHostToDevice));

        cout << "DEBUG: Float computation GPU model prepared." << endl;
    }

    // 释放GPU内存
    void free_gpu_memory() {
        CUDA_CHECK(cudaFree(d_x_float));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
    }

public:
    BSBInt8CorrectedGPUSolver(const ProblemModel& problem_model, int bs = 200) : model(problem_model), batch_size(bs) {
        prepare_gpu_model();
    }

    ~BSBInt8CorrectedGPUSolver() {
        free_gpu_memory();
    }

    // 核心求解函数（float计算版本）
    vector<int8_t> solve(int max_iters = 1000, double optimized_xi = 0.365) {
        cout << "DEBUG: Entering solve() function..." << endl;
        cout.flush();

        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << model.scaling_factor << endl;
        cout.flush();

        // BSB超参数（float计算版本）
        float xi = static_cast<float>(optimized_xi);  // 使用优化的xi值，转为float
        float dt = 0.1f;                             // 时间步长，使用float

        cout << "  - BSB Algorithm (Float computation version)" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using optimized xi=" << fixed << setprecision(6) << xi << ", dt=" << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - ✅ Float computation: x/y/gradient all in float (no double)" << endl;
        cout.flush();

        // 确定优化方向
        bool is_maximize = (model.objective_sense == "maximize");

        // 计算线程块和网格大小
        dim3 blockSize(256);
        dim3 gridSize_batch_spins(batch_size, (model.num_vars + blockSize.x - 1) / blockSize.x);

        // 初始化批处理数据
        init_batches_kernel<<<gridSize_batch_spins, blockSize>>>(
            d_x_float, d_y, d_gradient_term,
            model.num_vars, model.num_quad_terms,
            time(nullptr), batch_size, 0.1);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // 计时开始
        auto start = high_resolution_clock::now();

        // 核心迭代（float计算版本的BSB动力学）
        for (int iter = 0; iter < max_iters; ++iter) {
            float p = static_cast<float>(iter) / static_cast<float>(max_iters - 1);

            // BSB算法：直接使用连续变量x计算梯度，不需要计算sign
            calc_gradient_kernel_optimized<<<gridSize_batch_spins, blockSize>>>(d_x_float, d_gradient_term,
                                                                      d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                                                                      model.num_vars, model.num_quad_terms, batch_size);

            // 更新动量：dy/dt = -(1-p(t))*(x-0.5) ± ξ*∇f(x)
            update_y_kernel_optimized<<<gridSize_batch_spins, blockSize>>>(d_x_float, d_y, d_gradient_term,
                                                                model.num_vars, p, xi, dt, batch_size, is_maximize);

            // 更新位置：dx/dt = y
            update_x_kernel_optimized<<<gridSize_batch_spins, blockSize>>>(d_x_float, d_y, model.num_vars, dt, batch_size);

            // 边界条件处理：将x限制在[0,1]范围内，超出时投影到边界并清零动量
            apply_constraint_kernel<<<gridSize_batch_spins, blockSize>>>(d_x_float, d_y, model.num_vars, batch_size);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // 计时结束
        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;

        // 在GPU端计算所有批次的能量以选择最终最优解
        float* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));

        // 调用GPU能量计算内核
        dim3 energy_block_size(256);
        dim3 energy_grid_size(batch_size);
        calculate_energy_kernel_optimized<<<energy_grid_size, energy_block_size>>>(
            d_x_float, d_second_order_terms, d_second_order_coeff,
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
        CUDA_CHECK(cudaMemcpy(best_host_x.data(), d_x_float + best_idx * model.num_vars,
                              model.num_vars * sizeof(float), cudaMemcpyDeviceToHost));

        // 转换为INT8返回
        vector<int8_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            solution[i] = float_to_int8(best_host_x[i]);
        }

        return solution;
    }

    // 获取自旋数
    int get_num_vars() const {
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
    double optimized_xi = 0.365;

    // 可选：通过命令行参数覆盖xi值
    if (argc > 2) {
        optimized_xi = stod(argv[2]);
    }
    
    cout << "========================================" << endl;
    cout << "       BSB_GPU_int8 (Float Computation Only)  " << endl;
    cout << "  - File path: " << qplib_file_path << endl;
    cout << "  - Algorithm: BSB (Ballistic Simulated Bifurcation)" << endl;
    cout << "  - Key feature: Float for ALL x/y/gradient computation (no double)" << endl;
    cout << "  - Precision: Float (32-bit), INT8 only for output format" << endl;
    cout << "  - Energy method: Continuous variables (float precision)" << endl;
    cout << "  - CUDA parallel batch processing optimized" << endl;
    cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
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
        cout << "DEBUG: Creating BSB GPU Float solver with 100 batches..." << endl;
        cout.flush();

        // 直接使用ProblemModel进行GPU计算
        BSBInt8CorrectedGPUSolver solver(model, 100); // 100 batches
        cout << "DEBUG: GPU Float precision solver created successfully!" << endl;
        cout.flush();

        cout << "Starting GPU Float precision optimization..." << endl;
        cout.flush();

        // 使用与CPU版本一致的参数：1000次迭代，优化xi值
        auto start = high_resolution_clock::now();
        vector<int8_t> solution = solver.solve(1000, optimized_xi);
        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;
        cout << "TIME: Solving completed, total time: " << fixed << setprecision(6) << duration << " ms" << endl;
        cout.flush();
        
        // 计算目标函数值（使用连续变量，与CPU FP8一致）
        double objective_value = 0.0;
        for (size_t i = 0; i < solution.size(); ++i) {
            int8_t x_val = solution[i];
            float x_float_val = int8_to_float(x_val);  // INT8 -> float [-1.0,1.0]
            double x_continuous = (static_cast<double>(x_float_val) + 1.0) / 2.0;  // [-1.0,1.0] -> [0,1] 连续变量
            objective_value += model.linear_terms[i] * x_continuous;
        }

        for (int t = 0; t < model.num_quad_terms; ++t) {
            int var1 = model.quad_terms_indices[t][0];
            int var2 = model.quad_terms_indices[t][1];
            double coeff = model.quad_terms_coeffs[t];

            int8_t x1_val = solution[var1];
            int8_t x2_val = solution[var2];
            float x1_float_val = int8_to_float(x1_val);
            float x2_float_val = int8_to_float(x2_val);
            double x1_continuous = (static_cast<double>(x1_float_val) + 1.0) / 2.0;  // [-1.0,1.0] -> [0,1] 连续变量
            double x2_continuous = (static_cast<double>(x2_float_val) + 1.0) / 2.0;  // [-1.0,1.0] -> [0,1] 连续变量

            objective_value += 0.5 * coeff * x1_continuous * x2_continuous;
        }

        objective_value = objective_value * model.scaling_factor + model.constant_term;

        // 输出二元解表示（0/1，存储为INT8但显示为整数）
        cout << "\nRESULT: Solution (" << model.num_vars << " variables, binary values [0/1] stored as INT8): " << endl;
        int count_ones = 0;
        for (size_t i = 0; i < solution.size(); ++i) {
            // 从INT8值获取二进制表示
            int8_t val = solution[i];
            int binary_val = (val > 0) ? 1 : 0;  // 二值化为0/1
            cout << binary_val;
            if (binary_val == 1) count_ones++;
            if ((i + 1) % 16 == 0) cout << endl;
            else if (i != solution.size() - 1) cout << ", ";
        }
        cout << endl << endl;

        cout << "Solution statistics:" << endl;
        cout << "  - Variables set to 1: " << count_ones << "/" << solution.size() << endl;
        cout << "  - Variables set to 0: " << (solution.size() - count_ones) << "/" << solution.size() << endl;
        cout << "  - Solution density: " << fixed << setprecision(2) << (count_ones * 100.0 / solution.size()) << "%" << endl;
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
        cout << "  Storage: INT8, Computation: Float (no double)" << endl;
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