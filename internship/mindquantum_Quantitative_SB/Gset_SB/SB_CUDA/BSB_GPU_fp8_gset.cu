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
#include <ctime>
#include "sb_cuda_gset_reader.h"

using namespace std;
using namespace std::chrono;

// Clamp函数（与CPU版本一致）
__host__ __device__ inline float clamp(float val, float min_val, float max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            cerr << "CUDA error in " << #call << " at line " << __LINE__ << ": " << cudaGetErrorString(err) << endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// PyTorch FP8_e4m3fn格式实现（精度优化版，参考CPU版）
// 格式: 1位符号(S), 4位指数(E), 3位尾数(M)
__host__ __device__ inline uint8_t float_to_fp8_e4m3fn(float f) {
    // 处理特殊值
    if (isnan(f)) return 0x80;
    if (isinf(f)) return signbit(f) ? 0xFF : 0x7F;
    if (f == 0.0f) return signbit(f) ? 0x80 : 0x00;

    // 获取IEEE 754 FP32的符号、指数和尾数
    uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
    int sign_bit = (bits >> 31) & 0x1;
    int exp32 = (bits >> 23) & 0xFF;
    uint32_t mant32 = bits & 0x7FFFFF;

    const int FP8_EXP_BIAS = 7;
    const int FP8_MAX_EXP = 7;
    const int FP8_MIN_EXP = -7;

    // 计算实际指数
    int true_exp = exp32 - 127;

    // E4M3FN范围检查: 指数范围[-7, 7], 偏置7
    if (true_exp > FP8_MAX_EXP) {
        return sign_bit ? 0xFF : 0x7F;
    }

    if (true_exp < FP8_MIN_EXP) {
        // 次正规数处理：当指数过小时，返回零（简化处理）
        return sign_bit ? 0x80 : 0x00;
    }

    // 正常值转换（带更精确的舍入）
    int exp8 = true_exp + FP8_EXP_BIAS;
    exp8 = max(0, min(15, exp8));

    // 保留高3位尾数，并检查是否需要进位（round-to-nearest-even）
    uint8_t mant8 = (mant32 >> 20) & 0x07;
    uint32_t remainder = mant32 & 0xFFFFF;

    // 更精确的舍入：检查第20位以及后续是否全部为零
    if (remainder >= 0x80000) {
        mant8 += 1;
        if (mant8 >= 8) {
            mant8 = 0;
            exp8 += 1;
            if (exp8 > 15) {
                return sign_bit ? 0xFF : 0x7F;
            }
        }
    }

    return (sign_bit << 7) | (exp8 << 3) | mant8;
}

__host__ __device__ inline float fp8_to_float_e4m3fn(uint8_t fp8) {
    // 解析E4M3FN格式
    int sign_bit = (fp8 >> 7) & 0x01;
    int exp8 = (fp8 >> 3) & 0x0F;
    uint8_t mant8 = fp8 & 0x07;

    const int FP8_EXP_BIAS = 7;

    // 特殊值处理
    if (exp8 == 0 && mant8 == 0) {
        return sign_bit ? -0.0f : 0.0f;
    }
    if (exp8 == 15) {
        if (mant8 == 0) {
            return sign_bit ? -INFINITY : INFINITY;
        } else {
            float nan_val = NAN;
            return sign_bit ? -nan_val : nan_val;
        }
    }

    // 正常值解码
    int true_exp = exp8 - FP8_EXP_BIAS;

    uint32_t fp32_exp = true_exp + 127;
    uint32_t fp32_mant = mant8 << 20;

    uint32_t fp32_bits = (sign_bit << 31) | (fp32_exp << 23) | fp32_mant;
    float result = *reinterpret_cast<float*>(&fp32_bits);

    return result;
}

__host__ __device__ inline int sign_fp8(uint8_t x) {
    float x_val = fp8_to_float_e4m3fn(x);
    return (x_val >= 0.0f) ? 1 : -1;
}

// GPU批处理数据结构
struct GPUBatchData {
    uint8_t* x;
    uint8_t* y;
    float* gradient_term;
};

__global__ void init_batches_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ y, float* __restrict__ gradient_term,
                                   int num_spins, int num_second_order,
                                   unsigned long long seed, int batch_size) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_idx >= batch_size) return;

    curandState_t state;
    curand_init(seed + batch_idx, 0, 0, &state);

    int batch_offset = batch_idx * num_spins;
    const float init_scale = 0.1f;

    for (int i = 0; i < num_spins; ++i) {
        // x在[-1,1]范围内初始化
        float rand_val = curand_uniform(&state);
        float x_val = init_scale * (rand_val - 0.5f);
        x[batch_offset + i] = float_to_fp8_e4m3fn(x_val);

        rand_val = curand_uniform(&state);
        float y_val = init_scale * (rand_val - 0.5f);
        y[batch_offset + i] = float_to_fp8_e4m3fn(y_val);

        gradient_term[batch_offset + i] = 0.0f;
    }
}

__global__ void calc_gradient_kernel_optimized(uint8_t* x, float* gradient_term,
                                               int* __restrict__ second_order_terms, uint8_t* __restrict__ second_order_coeff,
                                               uint8_t* __restrict__ one_order_coeff,
                                               int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    float grad = fp8_to_float_e4m3fn(__ldg(&one_order_coeff[spin_idx]));

    // 循环展开4次以提高指令级并行
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            float J_ij = fp8_to_float_e4m3fn(__ldg(&second_order_coeff[t + k]));

            if (var1 == spin_idx) {
                float x2_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var2]));
                grad += J_ij * x2_val;
            } else if (var2 == spin_idx) {
                float x1_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var1]));
                grad += J_ij * x1_val;
            }
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        float J_ij = fp8_to_float_e4m3fn(__ldg(&second_order_coeff[t]));

        if (var1 == spin_idx) {
            float x2_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var2]));
            grad += J_ij * x2_val;
        } else if (var2 == spin_idx) {
            float x1_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var1]));
            grad += J_ij * x1_val;
        }
    }

    gradient_term[batch_offset + spin_idx] = grad;
}

__global__ void update_y_kernel_optimized(uint8_t* __restrict__ x, uint8_t* __restrict__ y,
                                          float* __restrict__ gradient_term,
                                          int num_spins, float p, float xi, float dt,
                                          int batch_size, bool is_maximize) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    const float delta = 1.0f;
    const float pump_factor = -(delta - p);  // dy/dt = -(1-p)*x
    const float sign_factor = is_maximize ? xi : -xi;

    float x_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + spin_idx]));
    float grad_val = __ldg(&gradient_term[batch_offset + spin_idx]);
    float y_val = fp8_to_float_e4m3fn(__ldg(&y[batch_offset + spin_idx]));

    // dy/dt = -(1-p)*x + ξ*∇f(x)，对称值为0
    float update = (pump_factor * x_val + sign_factor * grad_val) * dt;
    y[batch_offset + spin_idx] = float_to_fp8_e4m3fn(clamp(y_val + update, -1.0f, 1.0f));
}

__global__ void update_x_kernel(uint8_t* __restrict__ x, uint8_t* __restrict__ y,
                                int num_spins, float dt, float delta, int batch_size) {
    int batch_idx = blockIdx.x;
    int spin_idx = threadIdx.x + blockIdx.y * blockDim.x;

    if (batch_idx >= batch_size || spin_idx >= num_spins) return;

    int batch_offset = batch_idx * num_spins;
    float x_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + spin_idx]));
    float y_val = fp8_to_float_e4m3fn(__ldg(&y[batch_offset + spin_idx]));

    // dx/dt = y，在[-1,1]范围内更新，对称值为0
    float x_new = x_val + y_val * dt * delta;

    // 边界约束：|x| > 1时投影到边界
    if (fabsf(x_new) > 1.0f) {
        x_new = (x_new >= 0.0f) ? 1.0f : -1.0f;
        y[batch_offset + spin_idx] = float_to_fp8_e4m3fn(0.0f);
    }

    x[batch_offset + spin_idx] = float_to_fp8_e4m3fn(x_new);
}

__global__ void calc_cut_kernel(uint8_t* __restrict__ x, float* __restrict__ cuts,
                                int* __restrict__ second_order_terms,
                                uint8_t* __restrict__ second_order_coeff,
                                uint8_t* __restrict__ one_order_coeff,
                                int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    float cut = 0.0f;
    float w_sum = 0.0f;

    // Gset 问题没有一阶项（外场），但为了通用性仍然计算
    for (int i = 0; i < num_spins; ++i) {
        float x_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + i]));
        cut += fp8_to_float_e4m3fn(__ldg(&one_order_coeff[i])) * x_val;
    }

    // BSB算法：使用连续变量 x 计算切割值（x在[-1,1]范围）
    // 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            float J_ij = fp8_to_float_e4m3fn(__ldg(&second_order_coeff[t + k]));
            float x1_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var1]));
            float x2_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var2]));

            cut += 0.5f * J_ij * x1_val * x2_val;
            w_sum += J_ij;
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        float J_ij = fp8_to_float_e4m3fn(__ldg(&second_order_coeff[t]));
        float x1_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var1]));
        float x2_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var2]));

        cut += 0.5f * J_ij * x1_val * x2_val;
        w_sum += J_ij;
    }

    // 加上常数项: -0.5 * sum(J_ij)
    cut -= 0.5f * w_sum;

    cuts[batch_idx] = cut;
}

__global__ void calc_energy_kernel(uint8_t* __restrict__ x, float* __restrict__ energies,
                                    int* __restrict__ second_order_terms,
                                    uint8_t* __restrict__ second_order_coeff,
                                    uint8_t* __restrict__ one_order_coeff,
                                    int num_spins, int num_second_order, int batch_size) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size) return;

    int batch_offset = batch_idx * num_spins;
    float energy = 0.0f;

    // 计算一次项能量（x在[-1,1]范围）
    for (int i = 0; i < num_spins; ++i) {
        float x_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + i]));
        energy += fp8_to_float_e4m3fn(__ldg(&one_order_coeff[i])) * x_val;
    }

    // 计算二次项能量（x在[-1,1]范围）
    // 循环展开4次
    int t = 0;
    for (; t + 3 < num_second_order; t += 4) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int var1 = __ldg(&second_order_terms[(t + k) * 2]);
            int var2 = __ldg(&second_order_terms[(t + k) * 2 + 1]);
            float J_ij = fp8_to_float_e4m3fn(__ldg(&second_order_coeff[t + k]));
            float x1_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var1]));
            float x2_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var2]));

            energy += J_ij * x1_val * x2_val;
        }
    }

    // 处理剩余的项
    for (; t < num_second_order; ++t) {
        int var1 = __ldg(&second_order_terms[t * 2]);
        int var2 = __ldg(&second_order_terms[t * 2 + 1]);
        float J_ij = fp8_to_float_e4m3fn(__ldg(&second_order_coeff[t]));
        float x1_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var1]));
        float x2_val = fp8_to_float_e4m3fn(__ldg(&x[batch_offset + var2]));

        energy += J_ij * x1_val * x2_val;
    }

    energies[batch_idx] = energy;
}

class BSBGPUFP8Solver {
private:
    int batch_size;
    int num_spins;
    int num_second_order;

    uint8_t* d_x;
    uint8_t* d_y;
    float* d_gradient_term;

    int* d_second_order_terms;
    uint8_t* d_second_order_coeff;
    uint8_t* d_one_order_coeff;
    cudaStream_t stream;

public:
    BSBGPUFP8Solver(int N, int bs = 200) : batch_size(bs), num_spins(N), num_second_order(0) {
        CUDA_CHECK(cudaMalloc(&d_x, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_y, batch_size * num_spins * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_gradient_term, batch_size * num_spins * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&stream));
    }

    ~BSBGPUFP8Solver() {
        CUDA_CHECK(cudaFree(d_x));
        CUDA_CHECK(cudaFree(d_y));
        CUDA_CHECK(cudaFree(d_gradient_term));
        CUDA_CHECK(cudaFree(d_second_order_terms));
        CUDA_CHECK(cudaFree(d_second_order_coeff));
        CUDA_CHECK(cudaFree(d_one_order_coeff));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void set_model_data(int* terms, float* second_coeff, float* one_coeff, int num_second) {
        num_second_order = num_second;

        vector<uint8_t> fp8_second_order_coeff(num_second);
        vector<uint8_t> fp8_one_order_coeff(num_spins);
        for (int i = 0; i < num_second; ++i) {
            fp8_second_order_coeff[i] = float_to_fp8_e4m3fn(second_coeff[i]);
        }
        for (int i = 0; i < num_spins; ++i) {
            fp8_one_order_coeff[i] = float_to_fp8_e4m3fn(one_coeff[i]);
        }

        CUDA_CHECK(cudaMalloc(&d_second_order_terms, num_second_order * 2 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_second_order_coeff, num_second_order * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_one_order_coeff, num_spins * sizeof(uint8_t)));

        // 使用异步内存传输
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_terms, terms, num_second_order * 2 * sizeof(int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_second_order_coeff, fp8_second_order_coeff.data(), num_second_order * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_one_order_coeff, fp8_one_order_coeff.data(), num_spins * sizeof(uint8_t), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void solve(int n_iter, float xi, float dt, bool is_maximize = true) {
        // 优化：使用更合适的block大小，warp大小为32
        const int BLOCK_SIZE = 256;
        const int BLOCKS_PER_BATCH = (num_spins + BLOCK_SIZE - 1) / BLOCK_SIZE;

        dim3 grid_x(batch_size);
        dim3 block_x(BLOCK_SIZE);
        dim3 grid_2d(batch_size, BLOCKS_PER_BATCH);

        init_batches_kernel<<<grid_x, block_x, 0, stream>>>(d_x, d_y, d_gradient_term, num_spins, num_second_order, time(nullptr), batch_size);

        // 核心迭代（BSB动力学）
        for (int iter = 0; iter < n_iter; ++iter) {
            // 计算泵浦振幅：p(t) = t/T
            float p = static_cast<float>(iter) / static_cast<float>(n_iter);
            float delta = 1.0f;

            // BSB: 使用连续的 x 值计算梯度（FP32计算，FP8存储）
            calc_gradient_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_gradient_term,
                d_second_order_terms, d_second_order_coeff, d_one_order_coeff,
                num_spins, num_second_order, batch_size);

            update_y_kernel_optimized<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, d_gradient_term,
                num_spins, p, xi, dt, batch_size, is_maximize);

            update_x_kernel<<<grid_2d, block_x, 0, stream>>>(d_x, d_y, num_spins, dt, delta, batch_size);
        }

        // 只在最后同步一次，减少同步开销
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    vector<float> get_cut_values() {
        vector<float> cuts(batch_size);
        float* d_cuts;
        CUDA_CHECK(cudaMalloc(&d_cuts, batch_size * sizeof(float)));

        // BSB: 只传递连续的 x 值，不使用 x_sign
        calc_cut_kernel<<<batch_size, 256, 0, stream>>>(d_x, d_cuts, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        CUDA_CHECK(cudaMemcpyAsync(cuts.data(), d_cuts, batch_size * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree(d_cuts));

        return cuts;
    }

    vector<float> get_energies() {
        vector<float> energies(batch_size);
        float* d_energies;
        CUDA_CHECK(cudaMalloc(&d_energies, batch_size * sizeof(float)));

        calc_energy_kernel<<<batch_size, 256, 0, stream>>>(d_x, d_energies, d_second_order_terms, d_second_order_coeff, d_one_order_coeff, num_spins, num_second_order, batch_size);
        CUDA_CHECK(cudaStreamSynchronize(stream));

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
    BSBGPUFP8Solver solver(gset.num_vertices, batch_size);

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

    // 求解（与CPU版本一致的参数）
    cout << "Solving MAX-CUT problem..." << endl;
    cout << "  - Variables: " << gset.num_vertices << ", Edges: " << gset.num_edges << endl;
    cout << "  - BSB Algorithm (FP8, continuous variables)" << endl;
    cout << "  - FP8 Format: PyTorch float8_e4m3fn (improved rounding)" << endl;
    cout << "  - Storage: FP8, Computation: FP32 (high precision)" << endl;
    cout << "  - Initialization: normal distribution, sigma=0.1" << endl;
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

    cout << "\nBSB FP8 CUDA Solver Results:" << endl;
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
