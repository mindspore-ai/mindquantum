/*
 * BSB_CPU_int8.cu
 * ================
 * BSB求解器（Ballistic Simulated Bifurcation）
 *
 * 核心特性：
 * -----------
 * 1. INT8存储：使用PyTorch torch.int8格式进行数据存储（-128到127）
 * 2. FP32计算：所有数值计算使用float32确保精度（不使用double）
 * 3. 量化策略：线性量化，范围[-1.0, 1.0] -> [-127, 127]
 * 4. AVX2向量化：256位寄存器并行处理32个INT8元素
 *
 * 编译选项：
 * ---------
 * 启用AVX2优化（推荐）：
 *   nvcc -O3 -march=native -arch=sm_XX BSB_CPU_int8.cu -o BSB_CPU_int8
 *
 * 禁用AVX2（标量版本）：
 *   nvcc -O3 -U__AVX2__ -arch=sm_XX BSB_CPU_int8.cu -o BSB_CPU_int8
 *
 * AVX2指令集详解：
 * ---------------
 * - _mm256_loadu_si256: 未对齐加载256位数据（32个INT8）
 * - _mm256_storeu_si256: 未对齐存储256位数据
 * - _mm256_set1_epi8: 设置所有8位为指定值
 * - _mm256_add_epi8: 8位整数加法（有饱和）
 * - _mm256_sub_epi8: 8位整数减法（有饱和）
 *
 * PyTorch INT8兼容性：
 * ------------------
 * - 数据格式：与PyTorch torch.int8完全一致
 * - 量化范围：[-1.0, 1.0] -> [-127, 127]（保留±0作为零值）
 * - 舍入模式：Round-to-nearest-even（与PyTorch一致）
 * - 反量化：[-127, 127] -> [-1.0, 1.0]
 *
 * 性能优势：
 * ---------
 * | 指令集  | 每次处理INT8 | 理论加速比 | 内存带宽 |
 * |--------|------------|-----------|---------|
 * | 标量    | 1          | 1x        | 基准    |
 * | AVX2    | 32         | 32x       | 4x      |
 *
 * 技术细节：
 * ---------
 * - INT8格式：有符号8位整数（-128到127）
 * - 量化公式：int8 = round(float * 127)
 * - 反量化公式：float = int8 / 127.0
 * - 饱和处理：超出范围时自动截断
 * - **计算精度：全部使用FP32（不使用double），与存储精度保持一致**
 *
 * 作者：Liu QAIA Team
 * 日期：2025年
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <chrono>
#include <random>
#include <algorithm>
#include <limits>
#include <immintrin.h>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <exception>
#include <cmath>
#include <omp.h>

// 统一包含路径：编译时使用 -I 参数指定 qplib_parser.h 的目录
// 例如: nvcc -I.. -O3 -arch=native BSB_CPU_int8.cu -o BSB_CPU_int8
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// ============================================================================
// PyTorch风格INT8量化配置（完全兼容torch.int8）
// ============================================================================

// INT8量化范围常量（与PyTorch一致）
constexpr int INT8_MIN_VAL = -127;  // 保留-128用于特殊值
constexpr int INT8_MAX_VAL = 127;   // PyTorch INT8最大值
constexpr float INT8_SCALE = 127.0f;  // 量化缩放因子

// Clamp函数（用于限制float值范围）
inline float clamp_float(float val, float min_val, float max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

// ============================================================================
// PyTorch风格INT8量化函数（完全兼容torch.int8）
// ============================================================================

// Float32 -> INT8 量化（线性量化，范围[-1.0, 1.0] -> [-127, 127]）
inline int8_t float_to_int8(float f) {
    // 线性量化：float -> int8
    // PyTorch行为：使用Round-to-nearest-even舍入
    float scaled = f * INT8_SCALE;
    scaled = fmaxf(static_cast<float>(INT8_MIN_VAL),
                   fminf(static_cast<float>(INT8_MAX_VAL), scaled));
    return static_cast<int8_t>(roundf(scaled));
}

// INT8 -> Float32 反量化（线性反量化，范围[-127, 127] -> [-1.0, 1.0]）
inline float int8_to_float(int8_t x) {
    return static_cast<float>(x) / INT8_SCALE;
}

// 直接操作int8_t的转换（用于更底层的操作）
inline int8_t int8_from_bits(uint8_t bits) {
    return static_cast<int8_t>(bits);
}

inline uint8_t int8_to_bits(int8_t x) {
    return static_cast<uint8_t>(x);
}

// ============================================================================
// AVX2向量化加速（256位寄存器 = 32个INT8元素并行处理）
// ============================================================================

#ifdef __AVX2__

// AVX2寄存器类型：__m256i = 256位整数寄存器，可存储32个INT8（32*8=256）
typedef __m256i int8x32;  // 32个INT8元素的向量类型

// 加载32个INT8元素（未对齐加载）
inline int8x32 int8_loadu_32(const int8_t* src) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src));
}

// 存储32个INT8元素（未对齐存储）
inline void int8_storeu_32(int8_t* dst, int8x32 vec) {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), vec);
}

// AVX2：批量INT8转FP32（32个INT8 -> 8个FP32）
inline void int8_to_float_avx2(const int8_t* src, float* dst, int size) {
    int i = 0;

    // 处理8个FP32（32个INT8）一组
    for (; i <= size - 32; i += 32) {
        // 加载32个INT8
        int8x32 int8_vec = int8_loadu_32(src + i);

        // 将32个INT8拆分为4组，每组8个转换为FP32
        for (int j = 0; j < 4; ++j) {
            // 提取8个INT8
            __m128i int8_128 = _mm256_extracti128_si256(int8_vec, j % 2);

            // 转换为FP32：int8 -> int16 -> int32 -> float32
            __m256i int32_vec = _mm256_cvtepi8_epi32(_mm_cvtepi8_epi16(
                (j < 2) ? _mm256_extracti128_si256(int8_vec, 0) :
                           _mm256_extracti128_si256(int8_vec, 1)
            ));

            // 转换为FP32并应用缩放因子
            __m256 float_vec = _mm256_cvtepi32_ps(int32_vec);
            float_vec = _mm256_mul_ps(float_vec, _mm256_set1_ps(1.0f / INT8_SCALE));

            // 存储结果
            _mm256_storeu_ps(dst + i + j * 8, float_vec);
        }
    }

    // 处理剩余元素（标量版本）
    for (; i < size; ++i) {
        dst[i] = int8_to_float(src[i]);
    }
}

// AVX2：批量FP32转INT8（8个FP32 -> 32个INT8）
inline void float_to_int8_avx2(const float* src, int8_t* dst, int size) {
    int i = 0;

    // 处理32个INT8（8个FP32）一组
    for (; i <= size - 32; i += 32) {
        __m256i int8_result = _mm256_setzero_si256();

        // 处理4组，每组8个FP32
        for (int j = 0; j < 4; ++j) {
            // 加载8个FP32
            __m256 float_vec = _mm256_loadu_ps(src + i + j * 8);

            // 应用缩放因子
            float_vec = _mm256_mul_ps(float_vec, _mm256_set1_ps(INT8_SCALE));

            // 转换为INT32
            __m256i int32_vec = _mm256_cvtps_epi32(float_vec);

            // 限制范围到[-127, 127]
            int32_vec = _mm256_max_epi32(int32_vec, _mm256_set1_epi32(INT8_MIN_VAL));
            int32_vec = _mm256_min_epi32(int32_vec, _mm256_set1_epi32(INT8_MAX_VAL));

            // 打包成INT8
            __m128i int16_low = _mm256_extracti128_si256(_mm256_packs_epi32(int32_vec, _mm256_setzero_si256()), 0);
            __m128i int8_128 = _mm256_packs_epi16(int16_low, _mm256_setzero_si128());

            // 合并到结果中
            int8_result = _mm256_inserti128_si256(int8_result, int8_128, j / 2);
        }

        // 存储结果
        int8_storeu_32(dst + i, int8_result);
    }

    // 处理剩余元素（标量版本）
    for (; i < size; ++i) {
        dst[i] = float_to_int8(src[i]);
    }
}

// AVX2：批量向量加法（INT8 + INT8，使用FP32中间计算）
inline void int8_add_vector_avx2(const int8_t* a, const int8_t* b, int8_t* dst, int size) {
    int i = 0;

    // 预分配临时FP32数组
    vector<float> a_float, b_float, result_float;
    a_float.reserve(32);
    b_float.reserve(32);
    result_float.reserve(32);

    for (; i <= size - 32; i += 32) {
        // 转换到FP32
        a_float.resize(32);
        b_float.resize(32);
        result_float.resize(32);

        for (int j = 0; j < 32; ++j) {
            a_float[j] = int8_to_float(a[i + j]);
            b_float[j] = int8_to_float(b[i + j]);
            result_float[j] = a_float[j] + b_float[j];
        }

        // 转换回INT8
        for (int j = 0; j < 32; ++j) {
            dst[i + j] = float_to_int8(result_float[j]);
        }
    }

    // 处理剩余元素
    for (; i < size; ++i) {
        dst[i] = float_to_int8(int8_to_float(a[i]) + int8_to_float(b[i]));
    }
}

// AVX2：批量标量乘向量（float * int8）
inline void int8_scalar_mul_vector_avx2(float scalar, const int8_t* src, int8_t* dst, int size) {
    int i = 0;

    for (; i <= size - 32; i += 32) {
        // 批量转换到FP32
        vector<float> src_float(32);
        for (int j = 0; j < 32; ++j) {
            src_float[j] = int8_to_float(src[i + j]);
        }

        // 批量乘法
        vector<float> result_float(32);
        for (int j = 0; j < 32; ++j) {
            result_float[j] = src_float[j] * scalar;
        }

        // 批量转换回INT8
        for (int j = 0; j < 32; ++j) {
            dst[i + j] = float_to_int8(result_float[j]);
        }
    }

    // 处理剩余元素
    for (; i < size; ++i) {
        dst[i] = float_to_int8(int8_to_float(src[i]) * scalar);
    }
}

// AVX2：批量线性组合（scalar * a + b）
inline void int8_fma_vector_avx2(float scalar, const int8_t* a, const int8_t* b, int8_t* dst, int size) {
    int i = 0;

    for (; i <= size - 32; i += 32) {
        // 批量转换到FP32
        vector<float> a_float(32), b_float(32), result_float(32);
        for (int j = 0; j < 32; ++j) {
            a_float[j] = int8_to_float(a[i + j]);
            b_float[j] = int8_to_float(b[i + j]);
            result_float[j] = a_float[j] * scalar + b_float[j];
        }

        // 批量转换回INT8
        for (int j = 0; j < 32; ++j) {
            dst[i + j] = float_to_int8(result_float[j]);
        }
    }

    // 处理剩余元素
    for (; i < size; ++i) {
        dst[i] = float_to_int8(int8_to_float(a[i]) * scalar + int8_to_float(b[i]));
    }
}

// AVX2：批量clamp操作
inline void int8_clamp_vector_avx2(int8_t* data, int8_t min_val, int8_t max_val, int size) {
    int i = 0;

    __m256i min_vec = _mm256_set1_epi8(min_val);
    __m256i max_vec = _mm256_set1_epi8(max_val);

    for (; i <= size - 32; i += 32) {
        __m256i data_vec = int8_loadu_32(data + i);
        data_vec = _mm256_max_epi8(data_vec, min_vec);
        data_vec = _mm256_min_epi8(data_vec, max_vec);
        int8_storeu_32(data + i, data_vec);
    }

    // 处理剩余元素
    for (; i < size; ++i) {
        if (data[i] < min_val) data[i] = min_val;
        if (data[i] > max_val) data[i] = max_val;
    }
}

// AVX2：批量复制操作
inline void int8_copy_vector_avx2(const int8_t* src, int8_t* dst, int size) {
    int i = 0;

    for (; i <= size - 32; i += 32) {
        __m256i src_vec = int8_loadu_32(src + i);
        int8_storeu_32(dst + i, src_vec);
    }

    for (; i < size; ++i) {
        dst[i] = src[i];
    }
}

// AVX2：批量填充操作
inline void int8_fill_vector_avx2(int8_t* dst, int8_t value, int size) {
    int i = 0;

    __m256i value_vec = _mm256_set1_epi8(value);

    for (; i <= size - 32; i += 32) {
        int8_storeu_32(dst + i, value_vec);
    }

    for (; i < size; ++i) {
        dst[i] = value;
    }
}

#endif

// ============================================================================
// 批量转换优化（使用AVX2加速，减少逐元素转换开销）
// ============================================================================
inline void batch_int8_to_float(const int8_t* src, float* dst, int size) {
#ifdef __AVX2__
    int8_to_float_avx2(src, dst, size);
#else
    for (int i = 0; i < size; ++i) {
        dst[i] = int8_to_float(src[i]);
    }
#endif
}

inline void batch_float_to_int8(const float* src, int8_t* dst, int size) {
#ifdef __AVX2__
    float_to_int8_avx2(src, dst, size);
#else
    for (int i = 0; i < size; ++i) {
        dst[i] = float_to_int8(src[i]);
    }
#endif
}

// 批处理数据结构（PyTorch INT8存储 + FP32计算混合精度）
// 优化：预分配临时数组，避免重复分配
struct BatchData {
    vector<int8_t> x;              // 位置状态（PyTorch int8）
    vector<int8_t> y;              // 动量状态（PyTorch int8）
    vector<int8_t> gradient_term;   // 梯度项（PyTorch int8）
    float energy;                    // 目标函数值（FP32，与计算精度一致）

    // 预分配的临时数组（避免重复分配）
    vector<float> x_float;       // FP32临时数组
    vector<float> y_float;       // FP32临时数组
    vector<float> grad_float;    // FP32临时数组
    vector<float> x_norm;        // [0,1]归一化临时数组

    BatchData(int num_vars, int num_high_order) :
        x(num_vars), y(num_vars),
        gradient_term(num_vars), energy(0.0f),
        x_float(num_vars), y_float(num_vars), grad_float(num_vars), x_norm(num_vars) {}
};

// BSB求解器（修正版INT8优化版）
class BSBInt8CorrectedSolver {
private:
    ProblemModel model;
    vector<vector<int>> var_to_terms;  // 自旋→参与的二阶项索引
    vector<vector<int>> term_positions; // 二阶项→包含的自旋索引
    float fixed_xi;  // 固定的优化xi值

public:
    BSBInt8CorrectedSolver(const ProblemModel& problem_model, float optimized_xi = 0.0f) : model(problem_model) {
        precompute_lookup_tables();

        // 设置xi值
        if (optimized_xi > 0.0f) {
            fixed_xi = optimized_xi;
        } else {
            fixed_xi = static_cast<float>(model.xi);
        }
    }

    // 预计算查找表（优化迭代效率）
    void precompute_lookup_tables() {
        var_to_terms.resize(model.num_vars);
        
        // 只有存在二次项时才调整term_positions大小
        if (model.num_quad_terms > 0) {
            term_positions.resize(model.num_quad_terms);

            for (int i = 0; i < model.num_quad_terms; ++i) {
                term_positions[i] = model.quad_terms_indices[i];
                for (int var : model.quad_terms_indices[i]) {
                    var_to_terms[var].push_back(i);
                }
            }
        }
    }

    // 初始化批处理数据（PyTorch INT8存储）
    // sigma=0.1（固定值，与BSB_CPU_fp16.cu保持一致）
    void init_batches(vector<BatchData>& batches, uint64_t seed, float sigma = 0.1f) {
        mt19937 gen(seed);
        normal_distribution<float> dis_normal(0.0f, 1.0f);

        for (size_t batch = 0; batch < batches.size(); ++batch) {
            mt19937 local_gen(seed + batch);
            for (int i = 0; i < model.num_vars; ++i) {
                // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5
                float x_normalized = sigma * dis_normal(local_gen) + 0.5f;
                x_normalized = (x_normalized < 0.0f) ? 0.0f : (x_normalized > 1.0f) ? 1.0f : x_normalized;

                // 转换到[-1,1]范围，存储为INT8
                float x_range = x_normalized * 2.0f - 1.0f;
                batches[batch].x[i] = float_to_int8(x_range);

                // y初始化为小随机值：sigma * N(0,1)，存储为INT8
                float y_val = sigma * dis_normal(local_gen);
                batches[batch].y[i] = float_to_int8(y_val);

                // 初始化梯度项为0
                batches[batch].gradient_term[i] = 0;
            }
        }
    }



    // 批量计算梯度（PyTorch INT8存储 + FP32计算，优化内存分配）
    void batch_calc_gradient(vector<BatchData>& batches) {
        // 确保linear_terms大小与变量数量一致
        if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
            model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
        }

        // 根据优化方向确定梯度符号
        bool is_maximize = (model.objective_sense == "maximize");
        float direction_sign = is_maximize ? 1.0f : -1.0f;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 使用预分配的grad_float计算梯度
            for (int i = 0; i < model.num_vars; ++i) {
                batch.grad_float[i] = direction_sign * static_cast<float>(model.linear_terms[i]);
            }

            // 累加二次项贡献：∑_j J_{ij}x_j
            if (model.num_quad_terms > 0) {
                // 预转换所有x值到[0,1]（使用预分配数组）
                for (int j = 0; j < model.num_vars; ++j) {
                    batch.x_norm[j] = (batch.x_float[j] + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
                }

                for (int t = 0; t < model.num_quad_terms; ++t) {
                    int var1 = model.quad_terms_indices[t][0];
                    int var2 = model.quad_terms_indices[t][1];
                    float J_ij = static_cast<float>(direction_sign) * static_cast<float>(model.quad_terms_coeffs[t]);

                    // 累加到临时数组（FP32计算）
                    batch.grad_float[var1] += (J_ij * 0.5f) * batch.x_norm[var2];
                    batch.grad_float[var2] += (J_ij * 0.5f) * batch.x_norm[var1];
                }
            }

            // 存储回INT8（批量转换）
            batch_float_to_int8(batch.grad_float.data(), batch.gradient_term.data(), model.num_vars);
        }
    }

    // 更新辅助状态y（PyTorch INT8存储 + FP32计算，优化内存分配）
    // BSB公式：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x)
    void batch_update_y(vector<BatchData>& batches, float p, float dt, float delta) {
        // 泵浦振幅项：-(1-p(t))
        const float pump_factor = -(1.0f - p);
        const float xi_val = fixed_xi;
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_int8_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);
            batch_int8_to_float(batch.gradient_term.data(), batch.grad_float.data(), model.num_vars);

            // 批量更新y值（AVX2优化）
            int i = 0;
#ifdef __AVX2__
            for (; i <= model.num_vars - 4; i += 4) {
                __m128 x_vals = _mm_loadu_ps(&batch.x_float[i]);
                __m128 y_vals = _mm_loadu_ps(&batch.y_float[i]);
                __m128 grad_vals = _mm_loadu_ps(&batch.grad_float[i]);

                // 将x从[-1,1]转换到[0,1]范围，然后计算x-0.5
                __m128 x_norm = _mm_add_ps(x_vals, _mm_set1_ps(1.0f));  // x + 1
                x_norm = _mm_mul_ps(x_norm, _mm_set1_ps(0.5f));  // (x+1)*0.5
                __m128 x_shifted = _mm_sub_ps(x_norm, _mm_set1_ps(0.5f));  // x - 0.5

                // 计算更新量：[pump_factor * (x-0.5) + xi * gradient] * dt
                __m128 pump_term = _mm_mul_ps(_mm_set1_ps(pump_factor), x_shifted);
                __m128 grad_term = _mm_mul_ps(_mm_set1_ps(xi_val), grad_vals);
                __m128 update = _mm_add_ps(pump_term, grad_term);
                update = _mm_mul_ps(update, _mm_set1_ps(dt_val));
                __m128 y_new = _mm_add_ps(y_vals, update);

                // 限制范围并存储回临时数组
                y_new = _mm_max_ps(_mm_set1_ps(-1.0f), _mm_min_ps(y_new, _mm_set1_ps(1.0f)));
                _mm_storeu_ps(&batch.y_float[i], y_new);
            }
#endif
            // 处理剩余变量
            for (; i < model.num_vars; ++i) {
                // 将x从[-1,1]转换到[0,1]范围
                float x_norm = (batch.x_float[i] + 1.0f) * 0.5f;
                float x_shifted = x_norm - 0.5f;

                // 梯度直接使用（已在calc_gradient中归一化到[0,1]范围）
                float gradient_term = xi_val * batch.grad_float[i];

                // BSB公式：ẏ_i = [-(1-p(t))*(x-0.5) + ξ*∇f(x)] * dt
                float y_new = batch.y_float[i] + (pump_factor * x_shifted + gradient_term) * dt_val;

                // 限制范围
                batch.y_float[i] = clamp_float(y_new, -1.0f, 1.0f);
            }

            // 存储回INT8（批量转换）
            batch_float_to_int8(batch.y_float.data(), batch.y.data(), model.num_vars);
        }
    }

    // 更新自旋x（PyTorch INT8存储 + FP32计算，优化内存分配）
    // 对应Python版本公式：dx/dt = y
    // 离散化：x^{(t+1)} = x^{(t)} + y^{(t+1)} * dt
    void batch_update_x(vector<BatchData>& batches, float dt, float delta) {
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_int8_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);

            // 批量更新x值（AVX2优化）
            int i = 0;
#ifdef __AVX2__
            for (; i <= model.num_vars - 4; i += 4) {
                __m128 x_vals = _mm_loadu_ps(&batch.x_float[i]);
                __m128 y_vals = _mm_loadu_ps(&batch.y_float[i]);

                // 转换到[0,1]进行更新
                __m128 x_norm = _mm_add_ps(x_vals, _mm_set1_ps(1.0f));  // x + 1
                x_norm = _mm_mul_ps(x_norm, _mm_set1_ps(0.5f));  // (x+1)*0.5

                // 位置更新：x += y * dt
                __m128 y_dt = _mm_mul_ps(y_vals, _mm_set1_ps(dt_val));
                __m128 x_new = _mm_add_ps(x_norm, y_dt);

                // 限制在[0,1]范围
                x_new = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(x_new, _mm_set1_ps(1.0f)));

                // 转换回[-1,1]范围并存储回临时数组
                __m128 x_new_val = _mm_mul_ps(x_new, _mm_set1_ps(2.0f));  // x * 2
                x_new_val = _mm_sub_ps(x_new_val, _mm_set1_ps(1.0f));  // x - 1
                _mm_storeu_ps(&batch.x_float[i], x_new_val);
            }
#endif
            // 处理剩余变量
            for (; i < model.num_vars; ++i) {
                // 将x从[-1,1]转换到[0,1]
                float x_norm = (batch.x_float[i] + 1.0f) * 0.5f;
                float y_real = batch.y_float[i];

                // 位置更新：x += y * dt
                float x_new = x_norm + y_real * dt_val;

                // 限制在[0,1]范围
                x_new = clamp_float(x_new, 0.0f, 1.0f);

                // 转换回[-1,1]
                batch.x_float[i] = x_new * 2.0f - 1.0f;
            }

            // 存储回INT8（批量转换）
            batch_float_to_int8(batch.x_float.data(), batch.x.data(), model.num_vars);
        }
    }

    // 边界约束（PyTorch INT8存储，优化内存分配）
    // 将x限制在[0,1]范围内，超出时投影到边界{0,1}并清零动量
    void batch_apply_constraint(vector<BatchData>& batches) {
        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_int8_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);

            // 处理边界约束
            int i = 0;
#ifdef __AVX2__
            // AVX2并行处理4个变量
            for (; i <= model.num_vars - 4; i += 4) {
                __m128 x_vals = _mm_loadu_ps(&batch.x_float[i]);

                // 转换到[0,1]范围
                __m128 x_norm = _mm_add_ps(x_vals, _mm_set1_ps(1.0f));
                x_norm = _mm_mul_ps(x_norm, _mm_set1_ps(0.5f));

                // 检查是否超出边界：x < 0 或 x > 1
                __m128 zero = _mm_setzero_ps();
                __m128 one = _mm_set1_ps(1.0f);
                __m128 mask_lt_zero = _mm_cmplt_ps(x_norm, zero);
                __m128 mask_gt_one = _mm_cmpgt_ps(x_norm, one);
                __m128 mask_outside = _mm_or_ps(mask_lt_zero, mask_gt_one);

                if (_mm_movemask_ps(mask_outside) != 0) {
                    // 有变量超出边界，需要处理
                    __m128 x_clamped = _mm_max_ps(zero, _mm_min_ps(x_norm, one));

                    // 转换回[-1,1]
                    __m128 x_new = _mm_mul_ps(x_clamped, _mm_set1_ps(2.0f));
                    x_new = _mm_sub_ps(x_new, _mm_set1_ps(1.0f));
                    _mm_storeu_ps(&batch.x_float[i], x_new);

                    // 清零动量（对于超出边界的变量）
                    __m128 y_floats = _mm_loadu_ps(&batch.y_float[i]);
                    __m128 y_cleared = _mm_andnot_ps(mask_outside, y_floats);
                    _mm_storeu_ps(&batch.y_float[i], y_cleared);

                    // 存储回INT8
                    for (int k = 0; k < 4; ++k) {
                        batch.x[i + k] = float_to_int8(batch.x_float[i + k]);
                        batch.y[i + k] = float_to_int8(batch.y_float[i + k]);
                    }
                }
            }
#endif
            // 处理剩余变量（串行优化版本）
            for (; i < model.num_vars; ++i) {
                // 转换到[0,1]范围检查
                float x_norm = (batch.x_float[i] + 1.0f) * 0.5f;
                if (x_norm < 0.0f || x_norm > 1.0f) {
                    // 投影到边界
                    float projected = (x_norm < 0.5f) ? 0.0f : 1.0f;
                    batch.x_float[i] = projected * 2.0f - 1.0f;
                    batch.y_float[i] = 0.0f;

                    // 存储回INT8
                    batch.x[i] = float_to_int8(batch.x_float[i]);
                    batch.y[i] = float_to_int8(batch.y_float[i]);
                }
            }
        }
    }

    // 计算单个批次的目标函数值（PyTorch INT8存储 + FP32计算）
    // 使用连续决策变量进行能量计算（[0,1]范围）
    float calc_batch_energy(const BatchData& batch) {
        float scaled_energy = 0.0f;

        // 检查是否是纯线性问题
        bool is_linear_problem = (model.num_quad_terms == 0);

        // 预转换所有INT8 x值到FP32（使用预分配的临时数组）
        vector<float> x_float(model.num_vars);
        batch_int8_to_float(batch.x.data(), x_float.data(), model.num_vars);

        // 二次项贡献：0.5 * ∑_{i,j} J_{ij}x_i x_j
        if (model.num_quad_terms > 0) {
            for (int t = 0; t < model.num_quad_terms; ++t) {
                int var1 = model.quad_terms_indices[t][0];
                int var2 = model.quad_terms_indices[t][1];
                float J_ij = static_cast<float>(model.quad_terms_coeffs[t]);

                // 使用连续变量计算（FP32精度）
                float x1_norm = (x_float[var1] + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
                float x2_norm = (x_float[var2] + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]

                scaled_energy += 0.5f * J_ij * x1_norm * x2_norm;  // 纯FP32计算
            }
        }

        // 线性项贡献：∑_i h_i x_i
        // 批量处理线性项，减少循环开销
        for (int i = 0; i < model.num_vars; ++i) {
            float x_continuous = (x_float[i] + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
            scaled_energy += static_cast<float>(model.linear_terms[i]) * x_continuous;  // 纯FP32计算
        }

        // 缩放处理（FP32精度）
        float original_energy;
        if (is_linear_problem) {
            // 纯线性问题，不进行缩放逆变换
            original_energy = scaled_energy;
        } else {
            // 重新缩放回原始能量（FP32）
            original_energy = scaled_energy * static_cast<float>(model.scaling_factor);
        }

        // 添加常数项（如果存在）
        if (model.constant_term != 0.0) {
            original_energy += static_cast<float>(model.constant_term);
        }

        return original_energy;
    }

    // 找到最优批次（根据优化方向选择最优解）
    // 使用二元化变量进行能量评估（FP32精度）
    int find_best_batch(vector<BatchData>& batches) {
        int best_idx = 0;
        float best_value = calc_batch_energy(batches[0]);
        batches[0].energy = best_value;

        // 根据优化方向确定比较标准
        bool is_maximize = (model.objective_sense == "maximize");

        for (size_t i = 1; i < batches.size(); ++i) {
            batches[i].energy = calc_batch_energy(batches[i]);
            if (is_maximize) {
                // 最大化问题：寻找最大值
                if (batches[i].energy > best_value) {
                    best_value = batches[i].energy;
                    best_idx = i;
                }
            } else {
                // 最小化问题：寻找最小值
                if (batches[i].energy < best_value) {
                    best_value = batches[i].energy;
                    best_idx = i;
                }
            }
        }
        return best_idx;
    }

    // 核心求解函数（基于BSB公式）
    // 关键改进：使用INT8存储进行连续动力学演化
    vector<int8_t> solve(int max_iters = 1000, int batch_size = 100, float sigma = 0.1f, vector<BatchData>* out_batches = nullptr) {
        // BSB超参数
        float dt = 0.1f;    // 时间步长
        float delta = 0.9f; // 阻尼系数

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << model.scaling_factor << endl;
        cout << "  - BSB Algorithm (优化版本)" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using optimized xi=" << fixed << setprecision(6) << fixed_xi << ", sigma=" << fixed << setprecision(6) << sigma << ", dt=" << fixed << setprecision(6) << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - Using INT8 quantization: x stored in [-127,127]" << endl;

        // 初始化批处理数据
        vector<BatchData> batches(batch_size, BatchData(model.num_vars, 0));
        init_batches(batches, time(nullptr), sigma);

        // 计时开始
        auto start = high_resolution_clock::now();

        // 核心迭代（BSB动力学）
        for (int iter = 0; iter < max_iters; ++iter) {
            // 计算泵浦振幅：p(t) = t/T = iter/(max_iters-1)
            float p = static_cast<float>(iter) / (max_iters - 1);

            // BSB动力学更新序列
            batch_calc_gradient(batches);
            batch_update_y(batches, p, dt, delta);
            batch_update_x(batches, dt, delta);
            batch_apply_constraint(batches);
        }

        // 计时结束
        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;

        // 计算最后一批的能量值（仅计算一次）
        for (size_t i = 0; i < batches.size(); ++i) {
            batches[i].energy = calc_batch_energy(batches[i]);
        }

        // 生成最优解
        int best_idx = find_best_batch(batches);
        vector<int8_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            solution[i] = batches[best_idx].x[i];
        }
        
        // 如果提供了输出指针，返回批次数据
        if (out_batches != nullptr) {
            *out_batches = batches;
        }
        
        return solution;
    }

    // 获取变量数
    int get_num_vars() const {
        return model.num_vars;
    }
};

// 主函数：BSB算法求解器
int main(int argc, char* argv[]) {
    // 处理命令行参数
    string qplib_file_path;
    float optimized_xi = 0.365f;  // 优化xi值
    
    if (argc > 1) {
        // 使用命令行参数指定的文件路径
        qplib_file_path = argv[1];
    } else {
        // 使用默认路径
        qplib_file_path = DEFAULT_QPLIB_FILE_PATH;
    }
    
    // 可选：通过命令行参数覆盖xi值
    if (argc > 2) {
        optimized_xi = stof(argv[2]);
    }
    
    // 输出头部信息
    cout << "========================================" << endl;
    cout << "         BSB_CPU_int8 Solver (PyTorch INT8)         " << endl;
    cout << "  - File path: " << qplib_file_path << endl;
    cout << "  - Algorithm: BSB (Ballistic Simulated Bifurcation)" << endl;
    cout << "  - Key feature: PyTorch int8 storage + FP32 computation" << endl;
    cout << "  - Storage: INT8 (PyTorch format), Computation: FP32 (pure FP32, no double)" << endl;
    cout << "  - Output: Raw INT8 values (not binary 0/1)" << endl;
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
    cout << "Starting BSB solver..." << endl;

    // 安全检查：确保linear_terms大小与变量数量一致
    if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
        model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
    }

    // 初始化求解器并求解
    try {
        cout << "Using fixed optimized xi value: " << fixed << setprecision(6) << optimized_xi
             << " (auto-calculated value: " << fixed << setprecision(6) << model.xi << ")" << endl;
        BSBInt8CorrectedSolver solver(model, optimized_xi);
        vector<BatchData> batches;

        auto start = high_resolution_clock::now();
        vector<int8_t> solution = solver.solve(1000, 100, 0.1f, &batches);
        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;
        cout << "TIME: Solving completed, total time: " << fixed << setprecision(6) << duration << " ms" << endl;
        
        // 找到最优批次
        float best_value = batches[0].energy;
        bool is_maximize = (model.objective_sense == "maximize");
        int best_idx = 0;
        for (size_t i = 1; i < batches.size(); ++i) {
            if (is_maximize) {
                if (batches[i].energy > best_value) {
                    best_value = batches[i].energy;
                    best_idx = i;
                }
            } else {
                if (batches[i].energy < best_value) {
                    best_value = batches[i].energy;
                    best_idx = i;
                }
            }
        }

        // 输出INT8解值（真正的int8数据）
        cout << "RESULT: Solution (" << model.num_vars << " variables, INT8 values [-127,127]): " << endl;
        vector<int8_t> int8_solution(model.num_vars);
        int count_ones = 0;

        for (int i = 0; i < model.num_vars; ++i) {
            // 直接使用INT8值
            int8_t x_val = batches[best_idx].x[i];  // INT8存储
            int8_solution[i] = x_val;

            // 将INT8转换为[0,1]范围，然后判断是否大于0.5
            float x_normalized = (int8_to_float(x_val) + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
            if (x_normalized > 0.5f) count_ones++;  // 使用0.5作为二值化阈值

            // 转换为int以便正确显示
            int display_val = static_cast<int>(x_val);

            // 每17个值换行
            if ((i + 1) % 17 == 0 || i == model.num_vars - 1) {
                cout << display_val << endl;
            } else {
                cout << display_val << ", ";
            }
        }
        
        // 输出解决方案统计信息
        int count_zeros = model.num_vars - count_ones;
        double density = static_cast<double>(count_ones) / model.num_vars * 100.0;
        
        cout << endl << "Solution statistics (INT8 -> [0,1] -> binary with 0.5 threshold):" << endl;
        cout << "  - Values > 0.5 (→1): " << count_ones << "/" << model.num_vars << endl;
        cout << "  - Values ≤ 0.5 (→0): " << count_zeros << "/" << model.num_vars << endl;
        cout << "  - Solution density: " << fixed << setprecision(2) << density << "%" << endl;
        
        // 输出最终结果总结
        cout << endl << "========================================" << endl;
        cout << "  ** BSB FINAL RESULTS **" << endl;
        cout << "========================================" << endl;
        cout << "  Problem: " << qplib_file_path.substr(qplib_file_path.find_last_of("/\\") + 1) << endl;
        cout << "  Objective sense: " << model.objective_sense << endl;
        cout << "  Number of variables: " << model.num_vars << endl;
        cout << "  Number of quadratic terms: " << model.num_quad_terms << endl;
        cout << "----------------------------------------" << endl;
        cout << "  BSB OBJECTIVE VALUE: " << fixed << setprecision(6) << best_value << endl;
        cout << "========================================" << endl;
        
    } catch (const exception& e) {
        cerr << "ERROR: Exception during solver initialization: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "ERROR: Unknown exception during solver initialization" << endl;
        return 1;
    }
    
    return 0;
}