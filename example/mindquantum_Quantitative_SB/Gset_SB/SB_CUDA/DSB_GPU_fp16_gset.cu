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
// constexpr float FP16_MAX_VAL = 65504.0f;
// constexpr float FP16_MIN_VAL = -65504.0f;
constexpr float FP16_CLAMP_MAX = 1.0f;
constexpr float FP16_CLAMP_MIN = -1.0f;

__host__ __device__ inline float clamp(float val, float min_val, float max_val) {
    return fmaxf(min_val, fminf(max_val, val));
}

__host__ __device__ inline float fp16_to_float(__half x) {
    return __half2float(x);
}

__host__ __device__ inline __half float_to_fp16(float f) {
    return __float2half(clamp(f, FP16_CLAMP_MIN, FP16_CLAMP_MAX));
}

__host__ __device__ inline int sign_fp16(__half x) {
    float x_val = fp16_to_float(x);
    return (x_val >= 0.0f) ? 1 : -1;
}

// GPU批处理数据结构（DSB FP16版本）
struct GPUBatchData {
    __half* x;                  // 自旋状态 [batch_size * num_spins] (FP16存储)
    __half* y;                 // 动量状态 [batch_size * num_spins] (FP16存储)
    int* x_sign;               // sign(x)用于DSB动力学方程
    float* gradient_term;        // 梯度项 [batch_size * num_spins] (float32计算)
};

// GPU上的DSB模型数据（FP16版本）
struct GPUBSBModel {
    int* second_order_terms;     // 二阶项索引对 [num_second_order * 2]
    __half* second_order_coeff;  // 二阶项系数 [num_second_order] (FP16存储)
    __half* one_order_coeff;     // 一阶项系数 [num_spins] (FP16存储)
    int num_spins;
    int num_second_order;
};

// 初始化批处理数据（优化版）
__global__ void init_batches_kernel_optimized(__half* __restrict__ x, __half* __restrict__ y, float* __restrict__ gradient_term,
                                              int num_spins, int num_second_order,
                                              unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;
    const float init_scale = 1.0f;  // 参考BSB使用1.0f

    // 循环展开优化
    int i = 0;
    for (; i + 3 < num_spins; i += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float rand_val = curand_uniform(&state);
            float x_val = init_scale * (rand_val - 0.5f);
            x[batch_offset + i + k] = float_to_fp16(x_val);

            rand_val = curand_uniform(&state);
            float y_val = init_scale * (rand_val - 0.5f);
            y[batch_offset + i + k] = float_to_fp16(y_val);

            gradient_term[batch_offset + i + k] = 0.0f;
        }
    }
    for (; i < num_spins; ++i) {
        float rand_val = curand_uniform(&state);
        float x_val = init_scale * (rand_val - 0.5f);
        x[batch_offset + i] = float_to_fp16(x_val);

        rand_val = curand_uniform(&state);
        float y_val = init_scale * (rand_val - 0.5f);
        y[batch_offset + i] = float_to_fp16(y_val);

        gradient_term[batch_offset + i] = 0.0f;
    }
}

// 计算sign(x)（优化版）
__global__ void calc_sign_kernel_optimized(__half* __restrict__ x, int* __restrict__ x_sign, int num_spins, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    // 循环展开优化
    int i = 0;
    for (; i + 3 < num_spins; i += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            x_sign[batch_offset + i + k] = sign_fp16(__ldg(&x[batch_offset + i + k]));
        }
    }
    for (; i < num_spins; ++i) {
        x_sign[batch_offset + i] = sign_fp16(__ldg(&x[batch_offset + i]));
    }
}

// DSB梯度计算内核（优化版）
__global__ void calc_gradient_kernel_optimized(__half* x, int* x_sign, float* __restrict__ gradient_term,
                                                int* __restrict__ second_order_terms, __half* __restrict__ second_order_coeff, __half* __restrict__ one_order_coeff,
                                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    float grad = fp16_to_float(__ldg(&one_order_coeff[spin_idx]));

    // 循环展开优化
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            float J_ij = fp16_to_float(__ldg(&second_order_coeff[t + k]));

            if (var1 == spin_idx) {
                int sign2 = __ldg(&x_sign[batch_offset + var2]);
                grad = fmaf(J_ij, static_cast<float>(sign2), grad);
            } else if (var2 == spin_idx) {
                int sign1 = __ldg(&x_sign[batch_offset + var1]);
                grad = fmaf(J_ij, static_cast<float>(sign1), grad);
            }
        }
    }
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        float J_ij = fp16_to_float(__ldg(&second_order_coeff[t]));

        if (var1 == spin_idx) {
            int sign2 = __ldg(&x_sign[batch_offset + var2]);
            grad = fmaf(J_ij, static_cast<float>(sign2), grad);
        } else if (var2 == spin_idx) {
            int sign1 = __ldg(&x_sign[batch_offset + var1]);
            grad = fmaf(J_ij, static_cast<float>(sign1), grad);
        }
    }

    gradient_term[batch_offset + spin_idx] = grad;
}

// y更新内核（优化版）
__global__ void update_y_kernel_optimized(__half* __restrict__ x, __half* __restrict__ y, float* __restrict__ gradient_term,
                                           int num_spins, float p, float xi, float dt, float delta, int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    const float pump_factor = -(delta - p);
    const float sign_factor = is_maximize ? xi : -xi;

    float x_val = fp16_to_float(__ldg(&x[batch_offset + spin_idx]));
    float grad_val = __ldg(&gradient_term[batch_offset + spin_idx]);
    float y_val = fp16_to_float(__ldg(&y[batch_offset + spin_idx]));

    // dy/dt = -(1-p)*x + xi*∇f(x)
    float update = (pump_factor * x_val + sign_factor * grad_val) * dt;
    y[batch_offset + spin_idx] = float_to_fp16(y_val + update);
}

// 更新自旋x（优化版）
__global__ void update_x_kernel(__half* __restrict__ x, __half* __restrict__ y, int num_spins, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    float x_val = fp16_to_float(__ldg(&x[batch_offset + spin_idx]));
    float y_val = fp16_to_float(__ldg(&y[batch_offset + spin_idx]));

    // dx/dt = y
    float x_new = x_val + y_val * dt * delta;

    if (fabsf(x_new) > 1.0f) {
        x_new = (x_new >= 0.0f) ? 1.0f : -1.0f;
        y[batch_offset + spin_idx] = float_to_fp16(0.0f);
    }

    x[batch_offset + spin_idx] = float_to_fp16(clamp(x_new, FP16_CLAMP_MIN, FP16_CLAMP_MAX));
}

// 最大割计算内核（优化版）
__global__ void calc_cut_kernel(int* __restrict__ x_sign, float* __restrict__ cuts,
                                int* __restrict__ second_order_terms, __half* __restrict__ second_order_coeff, __half* __restrict__ one_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    float cut = 0.0f;
    float w_sum = 0.0f;

    // DSB使用sign(x)计算切割值: 循环展开4次
    int i = 0;
    for (; i + 3 < num_spins; i += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int sign = x_sign[batch_offset + i + k];
            float coeff_val = fp16_to_float(__ldg(&one_order_coeff[i + k]));
            cut = fmaf(coeff_val, static_cast<float>(sign), cut);
        }
    }
    for (; i < num_spins; ++i) {
        int sign = x_sign[batch_offset + i];
        float coeff_val = fp16_to_float(__ldg(&one_order_coeff[i]));
        cut = fmaf(coeff_val, static_cast<float>(sign), cut);
    }

    // 二阶项: 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            float J_ij = fp16_to_float(__ldg(&second_order_coeff[t + k]));
            int sign1 = __ldg(&x_sign[batch_offset + var1]);
            int sign2 = __ldg(&x_sign[batch_offset + var2]);

            cut = fmaf(0.5f * J_ij * sign1, static_cast<float>(sign2), cut);
            w_sum += J_ij;
        }
    }
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        float J_ij = fp16_to_float(__ldg(&second_order_coeff[t]));
        int sign1 = __ldg(&x_sign[batch_offset + var1]);
        int sign2 = __ldg(&x_sign[batch_offset + var2]);

        cut = fmaf(0.5f * J_ij * sign1, static_cast<float>(sign2), cut);
        w_sum += J_ij;
    }

    cut -= 0.5f * w_sum;

    cuts[batch_idx] = cut;
}

// 能量计算内核（优化版）
__global__ void calc_energy_kernel(__half* x, int* __restrict__ x_sign, float* __restrict__ energies,
                                   int* __restrict__ second_order_terms, __half* __restrict__ second_order_coeff, __half* __restrict__ one_order_coeff,
                                   int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    float energy = 0.0f;

    // 计算一次项能量（使用sign）: 循环展开4次
    int i = 0;
    for (; i + 3 < num_spins; i += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int sign = x_sign[batch_offset + i + k];
            float coeff_val = fp16_to_float(__ldg(&one_order_coeff[i + k]));
            energy = fmaf(coeff_val, static_cast<float>(sign), energy);
        }
    }
    for (; i < num_spins; ++i) {
        int sign = x_sign[batch_offset + i];
        float coeff_val = fp16_to_float(__ldg(&one_order_coeff[i]));
        energy = fmaf(coeff_val, static_cast<float>(sign), energy);
    }

    // 计算二次项能量（使用sign）: 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            float J_ij = fp16_to_float(__ldg(&second_order_coeff[t + k]));
            int sign1 = __ldg(&x_sign[batch_offset + var1]);
            int sign2 = __ldg(&x_sign[batch_offset + var2]);

            energy = fmaf(J_ij, static_cast<float>(sign1 * sign2), energy);
        }
    }
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        float J_ij = fp16_to_float(__ldg(&second_order_coeff[t]));
        int sign1 = __ldg(&x_sign[batch_offset + var1]);
        int sign2 = __ldg(&x_sign[batch_offset + var2]);

        energy = fmaf(J_ij, static_cast<float>(sign1 * sign2), energy);
    }

    energies[batch_idx] = energy;
}

// DSB GPU求解器（FP16版本，优化版）
class DSBGPUFP16Solver {
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

    cudaStream_t stream;

public:
    DSBGPUFP16Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x_sign, batch_size * num_spins * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~DSBGPUFP16Solver() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_x_sign));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
        CUDA_CHECK(cudaStreamDestroy(stream));
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

        // 使用异步拷贝提高性能
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_terms, terms, num_second_order * 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_coeff, fp16_second_order_coeff.data(), num_second_order * sizeof(__half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_one_order_coeff, fp16_one_order_coeff.data(), num_spins * sizeof(__half), cudaMemcpyHostToDevice, stream));
    }

    void solve(int n_iter, float xi, float dt, bool is_maximize = true) {
        const int BLOCK_SIZE = 256;
        const int BLOCKS_PER_BATCH = (num_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 grid_x(batch_size);
        dim3 block_x(BLOCK_SIZE);
        dim3 grid_2d(batch_size, BLOCKS_PER_BATCH);

        init_batches_kernel_optimized<<<grid_x, block_x, 0, stream>>>(d_x, d_y, d_gradient_term, num_spins, num_second_order, time(nullptr), batch_size);

        const float delta = 0.9f;

        for (int iter = 0; iter < n_iter; ++iter) {
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);

            // 计算sign(x)
            calc_sign_kernel_optimized<<<grid_x, block_x, 0, stream>>>(d_x, d_x_sign, num_spins, batch_size);

            // 计算梯度（使用sign(x)）
            calc_gradient_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_x_sign, d_gradient_term, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);

            // 更新y
            update_y_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, d_gradient_term, num_spins, p, xi, dt, delta, batch_size, is_maximize);

            // 更新x
            update_x_kernel<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, num_spins, dt, delta, batch_size);
        }

        // 最后计算sign(x)用于最大割计算
        calc_sign_kernel_optimized<<<grid_x, block_x, 0, stream>>>(d_x, d_x_sign, num_spins, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        float* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(float)));

        calc_cut_kernel<<<batch_size, 256, 0, stream>>>(d_x_sign, d_cuts, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);

        CUDA_CHECK(cudaMemcpyAsync(cuts.data(), d_cuts, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_cuts));

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        float* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));

        calc_energy_kernel<<<batch_size, 256, 0, stream>>>(d_x, d_x_sign, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);

        CUDA_CHECK(cudaMemcpyAsync(energies.data(), d_energies, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
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
    DSBGPUFP16Solver solver(gset.num_vertices, batch_size);

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

    cout << "\nDSB FP16 CUDA Solver Results:" << endl;
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
