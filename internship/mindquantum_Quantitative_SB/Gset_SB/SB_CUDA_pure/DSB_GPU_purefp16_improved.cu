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

// 纯FP16配置（无缩放）
constexpr float FP16_MAX_VAL = 65504.0f;
constexpr float FP16_MIN_VAL = -65504.0f;

// 纯FP16转换函数
__host__ __device__ inline float fp16_to_float(__half x) {
    return __half2float(x);
}

__host__ __device__ inline __half float_to_fp16(float f) {
    return __float2half(f);
}

// sign函数：判断FP16值的符号
__host__ __device__ inline __half sign_fp16(__half x) {
    float x_val = fp16_to_float(x);
    return __float2half((x_val >= 0.0f) ? 1.0f : -1.0f);
}

// 计算FP16参数（纯FP16：不使用缩放）
inline void calculate_fp16_quantization_params(int num_edges, float max_weight) {
    cout << "\n=== Pure FP16 Parameters (No Scaling) ===" << endl;
    cout << "  Graph statistics:" << endl;
    cout << "    - Edges: " << num_edges << ", Max weight: " << max_weight << endl;
    cout << "  FP16 format: Binary16" << endl;
    cout << "    - FP16 range: [" << FP16_MIN_VAL << ", " << FP16_MAX_VAL << "]" << endl;
    cout << "  Quantization strategy:" << endl;
    cout << "    - NO Scaling: Pure FP16 without any scaling factors" << endl;
    cout << "    - Weights [-1, 1] stored directly as FP16" << endl;
    cout << "    - Spin sign {+1, -1} stored as FP16" << endl;
    cout << "  Computation strategy:" << endl;
    cout << "    - Storage: FP16 (all variables)" << endl;
    cout << "    - Arithmetic: Pure FP16 (all operations)" << endl;
    cout << "    - NO FP32 intermediates" << endl;
    cout << "============================================\n" << endl;
}

// GPU批处理数据结构（全部使用FP16）
struct GPUBatchData {
    __half* x;
    __half* y;
    __half* gradient_term;  // FP16存储
    __half* x_sign_fp16;    // FP16存储sign(x), 即 +1 或 -1
};

// 计算sign(x)并存储为FP16
__global__ void calc_sign_kernel(__half* __restrict__ x, __half* __restrict__ x_sign_fp16, int num_spins, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    __half zero = __float2half(0.0f);

    int i = 0;
    for (; i + 3 < num_spins; i += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            __half x_val = __ldg(&x[batch_offset + i + k]);
            x_sign_fp16[batch_offset + i + k] = (fp16_to_float(x_val) >= 0.0f) ? __float2half(1.0f) : __float2half(-1.0f);
        }
    }
    for (; i < num_spins; ++i) {
        __half x_val = __ldg(&x[batch_offset + i]);
        x_sign_fp16[batch_offset + i] = (fp16_to_float(x_val) >= 0.0f) ? __float2half(1.0f) : __float2half(-1.0f);
    }
}

// 初始化批处理数据（全部使用FP16）
__global__ void init_batches_kernel(__half* __restrict__ x, __half* __restrict__ y,
                                   __half* __restrict__ gradient_term,
                                   __half* __restrict__ x_sign_fp16,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;
    const __half init_scale = __float2half(0.02f);
    const __half neg_half_scale = __float2half(-0.01f);

    __half zero = __float2half(0.0f);
    __half one = __float2half(1.0f);
    __half neg_one = __float2half(-1.0f);

    for (int i = 0; i < num_spins; ++i) {
        float rand_val = curand_uniform(&state);
        __half x_val = __hmul(init_scale, __float2half(rand_val - 0.5f));
        x[batch_offset + i] = x_val;

        rand_val = curand_uniform(&state);
        __half y_val = __hmul(init_scale, __float2half(rand_val - 0.5f));
        y[batch_offset + i] = y_val;

        gradient_term[batch_offset + i] = zero;
        x_sign_fp16[batch_offset + i] = (fp16_to_float(x_val) >= 0.0f) ? one : neg_one;
    }
}

// DSB梯度计算内核（使用纯FP16和sign(x)）
__global__ void calc_gradient_kernel_optimized(__half* __restrict__ x, __half* __restrict__ x_sign_fp16, __half* __restrict__ gradient_term,
                                               int* __restrict__ second_order_terms, __half* __restrict__ second_order_coeff, __half* __restrict__ one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    // 使用纯FP16计算梯度
    // DSB: grad = coeff + sum(J_ij * sign(x_j))
    __half grad = one_order_coeff[spin_idx];

    // 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            __half J_ij = __ldg(&second_order_coeff[t + k]);

            if (var1 == spin_idx) {
                __half sign2 = __ldg(&x_sign_fp16[batch_offset + var2]);
                // 纯FP16计算: grad += J_ij * sign(x2)
                __half term = __hmul(J_ij, sign2);
                grad = __hadd(grad, term);
            } else if (var2 == spin_idx) {
                __half sign1 = __ldg(&x_sign_fp16[batch_offset + var1]);
                // 纯FP16计算: grad += J_ij * sign(x1)
                __half term = __hmul(J_ij, sign1);
                grad = __hadd(grad, term);
            }
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        __half J_ij = __ldg(&second_order_coeff[t]);

        if (var1 == spin_idx) {
            __half sign2 = __ldg(&x_sign_fp16[batch_offset + var2]);
            __half term = __hmul(J_ij, sign2);
            grad = __hadd(grad, term);
        } else if (var2 == spin_idx) {
            __half sign1 = __ldg(&x_sign_fp16[batch_offset + var1]);
            __half term = __hmul(J_ij, sign1);
            grad = __hadd(grad, term);
        }
    }

    gradient_term[batch_offset + spin_idx] = grad;
}

// y更新内核（纯FP16计算）
__global__ void update_y_kernel_optimized(__half* __restrict__ x, __half* __restrict__ y, __half* __restrict__ gradient_term,
                                          int num_spins, float p, float xi, float dt, int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    __half x_val = x[batch_offset + spin_idx];
    __half grad_val = gradient_term[batch_offset + spin_idx];
    __half y_val = y[batch_offset + spin_idx];

    // dy/dt = -(1-p)*x + xi*∇f(x)
    // 纯FP16计算
    __half one_minus_p = __hsub(__float2half(1.0f), __float2half(p));
    __half neg_one_minus_p = __hneg(one_minus_p);
    __half xi_h = __float2half(is_maximize ? xi : -xi);

    // term1 = -(1-p)*x
    __half term1 = __hmul(neg_one_minus_p, x_val);

    // term2 = xi*grad
    __half term2 = __hmul(xi_h, grad_val);

    // update = (term1 + term2) * dt
    __half sum = __hadd(term1, term2);
    __half dt_h = __float2half(dt);
    __half update = __hmul(sum, dt_h);

    // y_new = y + update
    __half y_new = __hadd(y_val, update);

    y[batch_offset + spin_idx] = y_new;
}

// 更新自旋x（纯FP16计算）
__global__ void update_x_kernel(__half* __restrict__ x, __half* __restrict__ y,
                                int num_spins, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;

    __half x_val = x[batch_offset + spin_idx];
    __half y_val = y[batch_offset + spin_idx];

    // dx/dt = y * delta * dt
    __half dt_h = __float2half(dt);
    __half delta_h = __float2half(delta);
    __half y_dt = __hmul(y_val, dt_h);
    __half y_dt_delta = __hmul(y_dt, delta_h);
    __half x_new = __hadd(x_val, y_dt_delta);

    // 边界约束：|x| > 1时投影到边界
    __half zero = __float2half(0.0f);
    __half one = __float2half(1.0f);
    __half neg_one = __float2half(-1.0f);

    __half abs_x_new = __habs(x_new);

    // FP16比较: abs_x_new > 1.0f
    if (fp16_to_float(abs_x_new) > 1.0f) {
        x_new = (fp16_to_float(x_new) >= 0.0f) ? one : neg_one;
        y[batch_offset + spin_idx] = zero;
    }

    x[batch_offset + spin_idx] = x_new;
}

// 最大割计算内核（使用纯FP16和sign(x)）
__global__ void calc_cut_kernel(__half* __restrict__ x_sign_fp16, __half* __restrict__ cuts,
                                int* __restrict__ second_order_terms, __half* __restrict__ second_order_coeff, __half* __restrict__ one_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    __half cut_h = __float2half(0.0f);
    __half zero = __float2half(0.0f);
    __half one = __float2half(1.0f);
    __half half = __float2half(0.5f);

    // 计算一次项（使用sign(x)）
    for (int i = 0; i < num_spins; ++i) {
        __half sign_i = __ldg(&x_sign_fp16[batch_offset + i]);
        __half coeff = __ldg(&one_order_coeff[i]);
        // 纯FP16计算: cut += coeff * sign(x)
        __half term = __hmul(coeff, sign_i);
        cut_h = __hadd(cut_h, term);
    }

    // 计算二次项（使用sign(x1) * sign(x2)）
    // 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            __half J_ij = __ldg(&second_order_coeff[t + k]);
            __half sign1 = __ldg(&x_sign_fp16[batch_offset + var1]);
            __half sign2 = __ldg(&x_sign_fp16[batch_offset + var2]);

            // 纯FP16计算: sign1 * sign2
            __half product = __hmul(sign1, sign2);

            // 纯FP16计算: 0.5 * J_ij * sign1 * sign2
            __half term = __hmul(__hmul(half, J_ij), product);
            cut_h = __hadd(cut_h, term);
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        __half J_ij = __ldg(&second_order_coeff[t]);
        __half sign1 = __ldg(&x_sign_fp16[batch_offset + var1]);
        __half sign2 = __ldg(&x_sign_fp16[batch_offset + var2]);

        // 纯FP16计算: sign1 * sign2
        __half product = __hmul(sign1, sign2);

        // 纯FP16计算: 0.5 * J_ij * sign1 * sign2
        __half term = __hmul(__hmul(half, J_ij), product);
        cut_h = __hadd(cut_h, term);
    }

    cuts[batch_idx] = cut_h;
}

// 能量计算内核（使用纯FP16和sign(x)）
__global__ void calc_energy_kernel(__half* __restrict__ x_sign_fp16, __half* __restrict__ energies,
                                   int* __restrict__ second_order_terms, __half* __restrict__ second_order_coeff, __half* __restrict__ one_order_coeff,
                                   int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;

    __half energy_h = __float2half(0.0f);

    // 计算一次项能量（使用sign(x)）
    for (int i = 0; i < num_spins; ++i) {
        __half sign_i = __ldg(&x_sign_fp16[batch_offset + i]);
        __half coeff = __ldg(&one_order_coeff[i]);
        // 纯FP16计算: energy += coeff * sign(x)
        __half term = __hmul(coeff, sign_i);
        energy_h = __hadd(energy_h, term);
    }

    // 计算二次项能量（使用sign(x1) * sign(x2)）
    // 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            __half J_ij = __ldg(&second_order_coeff[t + k]);
            __half sign1 = __ldg(&x_sign_fp16[batch_offset + var1]);
            __half sign2 = __ldg(&x_sign_fp16[batch_offset + var2]);

            // 纯FP16计算: J_ij * sign1 * sign2
            __half term = __hmul(J_ij, __hmul(sign1, sign2));
            energy_h = __hadd(energy_h, term);
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        __half J_ij = __ldg(&second_order_coeff[t]);
        __half sign1 = __ldg(&x_sign_fp16[batch_offset + var1]);
        __half sign2 = __ldg(&x_sign_fp16[batch_offset + var2]);

        // 纯FP16计算: J_ij * sign1 * sign2
        __half term = __hmul(J_ij, __hmul(sign1, sign2));
        energy_h = __hadd(energy_h, term);
    }

    energies[batch_idx] = energy_h;
}

class DSBGPUFP16Solver {
private:
    int batch_size;
    int num_spins;
    int num_second_order;

    __half* d_x;
    __half* d_y;
    __half* d_gradient_term;  // FP16存储
    __half* d_x_sign_fp16;    // FP16存储sign(x)

    int* d_second_order_terms;
    __half* d_second_order_coeff;
    __half* d_one_order_coeff;
    cudaStream_t stream;

public:
    DSBGPUFP16Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaMalloc(&d_x_sign_fp16, batch_size * num_spins * sizeof(__half)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~DSBGPUFP16Solver() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_x_sign_fp16));
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

        CUDA_CHECK(cudaMemcpyAsync(d_second_order_terms, terms, num_second_order * 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_coeff, fp16_second_order_coeff.data(), num_second_order * sizeof(__half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_one_order_coeff, fp16_one_order_coeff.data(), num_spins * sizeof(__half), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void solve(int n_iter, float xi, float dt, bool is_maximize = true) {
        const int BLOCK_SIZE = 256;
        const int BLOCKS_PER_BATCH = (num_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 grid_x(batch_size);
        dim3 block_x(BLOCK_SIZE);
        dim3 grid_2d(batch_size, BLOCKS_PER_BATCH);

        init_batches_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_y, d_gradient_term, d_x_sign_fp16, num_spins, num_second_order, time(nullptr), batch_size);

        const float delta = 0.9f;

        for (int iter = 0; iter < n_iter; ++iter) {
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);

            // 计算sign(x)并存储为FP16
            calc_sign_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_x_sign_fp16, num_spins, batch_size);

            // 计算梯度（使用sign(x)和纯FP16）
            calc_gradient_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_x_sign_fp16, d_gradient_term,
                d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                num_spins, num_second_order, batch_size);

            // 更新y
            update_y_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, d_gradient_term,
                num_spins, p, xi, dt, batch_size, is_maximize);

            // 更新x
            update_x_kernel<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, num_spins, dt, delta, batch_size);
        }

        // 最后计算sign(x)用于最大割计算
        calc_sign_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_x_sign_fp16, num_spins, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        __half* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(__half)));

        calc_cut_kernel<<<batch_size, 256, 0, stream>>>(d_x_sign_fp16, d_cuts, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        vector<__half> cuts_fp16(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(cuts_fp16.data(), d_cuts, batch_size * sizeof(__half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_cuts));

        for (int i = 0; i < batch_size; ++i) {
            cuts[i] = __half2float(cuts_fp16[i]);
        }

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        __half* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(__half)));

        calc_energy_kernel<<<batch_size, 256, 0, stream>>>(d_x_sign_fp16, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        vector<__half> energies_fp16(batch_size);
        CUDA_CHECK(cudaMemcpyAsync(energies_fp16.data(), d_energies, batch_size * sizeof(__half), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_energies));

        for (int i = 0; i < batch_size; ++i) {
            energies[i] = __half2float(energies_fp16[i]);
        }

        return energies;
    }
};

int main(int argc, char** argv) {
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
    int ret = read_gset_file(gset_file, true, &gset);

    if (ret != 0) {
        cerr << "Error: Failed to read Gset file " << gset_file << endl;
        return -2;
    }

    // 打印 Gset 信息
    print_gset_info(&gset);

    // 创建求解器
    DSBGPUFP16Solver solver(gset.num_vertices, batch_size);

    // 计算FP16自适应量化参数
    float max_weight = 1.0f;
    calculate_fp16_quantization_params(gset.num_edges, max_weight);

    // 转换为求解器格式
    vector<int> terms(gset.num_edges * 2);
    vector<float> second_coeff(gset.num_edges);
    vector<float> one_coeff(gset.num_vertices);

    convert_gset_to_solver_format(&gset, terms.data(), second_coeff.data(), one_coeff.data());

    // 设置模型数据
    solver.set_model_data(terms.data(), second_coeff.data(), one_coeff.data(), gset.num_edges);

    // 自动计算 xi
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

    // 求解
    cout << "Solving MAX-CUT problem..." << endl;
    cout << "  - Variables: " << gset.num_vertices << ", Edges: " << gset.num_edges << endl;
    cout << "  - DSB Algorithm (PURE FP16 - No Scaling, discrete variables via sign)" << endl;
    cout << "  - FP16 Format: Binary16" << endl;
    cout << "  - Storage: __half FP16 (all variables)" << endl;
    cout << "  - Arithmetic: FP16 (all operations)" << endl;
    cout << "  - NO Scaling: Pure FP16 without any scaling factors" << endl;
    cout << "  - Discretization: sign(x) -> {+1.0, -1.0} stored as FP16" << endl;
    cout << "  - Computation strategy:" << endl;
    cout << "    - Storage: FP16 (all variables including sign(x))" << endl;
    cout << "    - Dynamics (gradient, update): FP16" << endl;
    cout << "    - Cut/Energy: Pure FP16 accumulation (final sum also FP16)" << endl;
    cout << "    - NO FP32 intermediates" << endl;
    cout << "  - Initialization: normal distribution, sigma=0.02" << endl;
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
