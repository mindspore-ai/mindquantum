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

// PyTorch风格FP16配置
constexpr float FP16_MAX_VAL = 65504.0f;
constexpr float FP16_MIN_VAL = -65504.0f;

__host__ __device__ inline float fp16_to_float(__half x) {
    return __half2float(x);
}

__host__ __device__ inline __half float_to_fp16(float f) {
    return __float2half(f);
}

__host__ __device__ inline int sign_fp16(__half x) {
    float x_val = fp16_to_float(x);
    return (x_val >= 0.0f) ? 1 : -1;
}

// GPU批处理数据结构
struct GPUBatchData {
    __half* x;
    __half* y;
    int* x_sign;
    float* gradient_term;  // 梯度计算使用float32
};

// GPU上的BSB模型数据
struct GPUBSBModel {
    int* second_order_terms;
    __half* second_order_coeff;
    __half* one_order_coeff;
    int num_spins;
    int num_second_order;
};

// 初始化批处理数据
__global__ void init_batches_kernel(__half* x, __half* y, int* x_sign, float* gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;
    const float init_scale = 0.02f;  // 与 Python 保持一致: 0.02 * (rand() - 0.5)

    for (int i = 0; i < num_spins; ++i) {
        // 使用均匀分布，与 Python 保持一致: 0.02 * (rand() - 0.5)
        float rand_val = curand_uniform(&state);
        float x_val = init_scale * (rand_val - 0.5f);
        x[batch_offset + i] = float_to_fp16(x_val);

        rand_val = curand_uniform(&state);
        float y_val = init_scale * (rand_val - 0.5f);
        y[batch_offset + i] = float_to_fp16(y_val);

        x_sign[batch_offset + i] = sign_fp16(x[batch_offset + i]);
        gradient_term[batch_offset + i] = 0.0f;
    }
}

// BSB梯度计算内核（梯度计算使用float32）
__global__ void calc_gradient_kernel_optimized(__half* x, float* gradient_term,
                                               int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    float grad = fp16_to_float(one_order_coeff[spin_idx]);

    for (int t = 0; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        float J_ij = fp16_to_float(second_order_coeff[t]);

        if (var1 == spin_idx) {
            float x2_val = fp16_to_float(x[batch_offset + var2]);
            grad += J_ij * x2_val;
        } else if (var2 == spin_idx) {
            float x1_val = fp16_to_float(x[batch_offset + var1]);
            grad += J_ij * x1_val;
        }
    }

    gradient_term[batch_offset + spin_idx] = grad;
}

// y更新内核（BSB公式）
__global__ void update_y_kernel_optimized(__half* x, __half* y, float* gradient_term,
                                          int num_spins, float p, float xi, float dt, int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    const float delta = 1.0f;  // BSB使用delta=1
    const float pump_factor = -(delta - p);
    const float sign_factor = is_maximize ? xi : -xi;

    float x_val = fp16_to_float(x[batch_offset + spin_idx]);
    float grad_val = gradient_term[batch_offset + spin_idx];
    float y_val = fp16_to_float(y[batch_offset + spin_idx]);

    float update = (pump_factor * x_val + sign_factor * grad_val) * dt;
    y[batch_offset + spin_idx] = float_to_fp16(y_val + update);
}

// 更新自旋x（BSB公式）
__global__ void update_x_kernel(__half* x, __half* y, int num_spins, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    float x_val = fp16_to_float(x[batch_offset + spin_idx]);
    float y_val = fp16_to_float(y[batch_offset + spin_idx]);

    float x_new = x_val + y_val * dt * delta;

    // 边界约束：|x| > 1时投影到边界
    if (fabsf(x_new) > 1.0f) {
        x_new = (x_new >= 0.0f) ? 1.0f : -1.0f;
        y[batch_offset + spin_idx] = float_to_fp16(0.0f);
    }

    x[batch_offset + spin_idx] = float_to_fp16(fmaxf(FP16_MIN_VAL, fminf(FP16_MAX_VAL, x_new)));
}

// 最大割计算内核（使用float32）
__global__ void calc_cut_kernel(__half* x, float* cuts,
                                int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    float cut = 0.0f;
    float w_sum = 0.0f;

    // BSB算法：使用连续变量 x 计算切割值
    for (int i = 0; i < num_spins; ++i) {
        float x_val = fp16_to_float(x[batch_offset + i]);
        cut += fp16_to_float(one_order_coeff[i]) * x_val;
    }

    for (int t = 0; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        float J_ij = fp16_to_float(second_order_coeff[t]);  // J_ij = -w_ij (negate=True)
        float x1_val = fp16_to_float(x[batch_offset + var1]);
        float x2_val = fp16_to_float(x[batch_offset + var2]);

        // MAX-CUT 切割值公式（BSB使用连续变量 x）:
        // Cut = 0.5 * sum(J_ij * x_i * x_j) - 0.5 * sum(J_ij)
        cut += 0.5f * J_ij * x1_val * x2_val;
        w_sum += J_ij;
    }

    // 加上常数项: -0.5 * sum(J_ij)
    cut -= 0.5f * w_sum;

    // MAX-CUT 切割值
    cuts[batch_idx] = cut;
}

// 能量计算内核（使用float32）
__global__ void calc_energy_kernel(__half* x, float* energies,
                                  int* second_order_terms, __half* second_order_coeff, __half* one_order_coeff,
                                  int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    // 计算一次项能量
    float energy = 0.0f;
    for (int i = 0; i < num_spins; ++i) {
        float x_val = fp16_to_float(x[batch_offset + i]);
        energy += fp16_to_float(one_order_coeff[i]) * x_val;
    }

    // 计算二次项能量
    for (int t = 0; t < num_second_order; ++t) {
        int var1 = second_order_terms[t * 2];
        int var2 = second_order_terms[t * 2 + 1];
        float J_ij = fp16_to_float(second_order_coeff[t]);
        float x1_val = fp16_to_float(x[batch_offset + var1]);
        float x2_val = fp16_to_float(x[batch_offset + var2]);

        energy += J_ij * x1_val * x2_val;
    }

    energies[batch_idx] = energy;
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
    float* d_gradient_term;

    int* d_second_order_terms;
    __half* d_second_order_coeff;
    __half* d_one_order_coeff;

public:
    BSBGPUFP16Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x_sign, batch_size * num_spins * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(float)));
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
        dim3 grid_x(batch_size);
        dim3 block_x(256);
        dim3 grid_y((num_spins + block_x.x - 1) / block_x.x);

        init_batches_kernel<<<grid_x, block_x>>>(d_x, d_y, d_x_sign, d_gradient_term, num_spins, num_second_order, 42, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int iter = 0; iter < n_iter; ++iter) {
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);
            float delta = 1.0f;

            dim3 grid_2d(batch_size, grid_y.x);
            // BSB: 使用连续的 x 值计算梯度
            calc_gradient_kernel_optimized<<<grid_2d, block_x>>>(d_x, d_gradient_term, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
            CUDA_CHECK(cudaDeviceSynchronize());

            update_y_kernel_optimized<<<grid_2d, block_x>>>(d_x, d_y, d_gradient_term, num_spins, p, xi, dt, batch_size, is_maximize);
            CUDA_CHECK(cudaDeviceSynchronize());

            update_x_kernel<<<grid_2d, block_x>>>(d_x, d_y, num_spins, dt, delta, batch_size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // BSB: 不需要离散化为 sign，直接使用连续的 x 值计算切割值
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        float* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(float)));

        // BSB: 只传递连续的 x 值，不使用 x_sign
        calc_cut_kernel<<<batch_size, 256>>>(d_x, d_cuts, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(cuts.data(), d_cuts, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_cuts));

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        float* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));

        calc_energy_kernel<<<batch_size, 256>>>(d_x, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(energies.data(), d_energies, batch_size * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_energies));

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
