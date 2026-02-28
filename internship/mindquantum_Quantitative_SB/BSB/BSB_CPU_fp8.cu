// clang-format off
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
#include <omp.h>  // 添加OpenMP支持

// 统一包含路径：编译时使用 -I 参数指定 qplib_parser.h 的目录
// 例如: nvcc -I.. -O3 -arch=native BSB_CPU_fp8.cu -o BSB_CPU_fp16
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;
// Clamp函数（需要先定义）
template<typename T>
inline T clamp(T val, T min_val, T max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

inline float clamp_float(float val, float min_val, float max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

inline double clamp_double(double val, double min_val, double max_val) {
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}

// ============================================================================
// FP8 E4M3FN 配置（PyTorch float8_e4m3fn 标准）
// ============================================================================
// 参考PyTorch的float8_e4m3fn实现，不使用额外缩放
constexpr int FP8_EXP_BIAS = 7;
constexpr int FP8_MAX_EXP = 7;
constexpr int FP8_MIN_EXP = -7;

// 直接使用[-1,1]范围，避免缩放带来的精度问题

// PyTorch FP8_e4m3fn量化实现（精度优化版）
// 改进：使用缩放因子限制量化范围，提升[-1,1]内的精度
struct fp8_t {
    uint8_t data;  // 8位存储

    fp8_t() : data(0x00) {}
    fp8_t(float f) : data(float_to_fp8_scaled(f)) {}

    // PyTorch风格的FP8转换（直接使用E4M3FN格式）
    static uint8_t float_to_fp8_scaled(float f) {
        // 处理特殊值
        if (isnan(f)) return 0x80;
        if (isinf(f)) return (f > 0.0f) ? 0x7F : 0xFF;
        if (f == 0.0f) return (f > 0.0f) ? 0x00 : 0x80;

        // 获取IEEE 754 FP32的符号、指数和尾数
        uint32_t bits = *reinterpret_cast<uint32_t*>(&f);
        int sign_bit = (bits >> 31) & 0x1;
        int exp32 = (bits >> 23) & 0xFF;
        uint32_t mant32 = bits & 0x7FFFFF;

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

    // FP8到FP32转换（不使用缩放）
    static float fp8_to_float_scaled(uint8_t fp8_val) {
        // 解析E4M3FN格式
        int sign_bit = (fp8_val >> 7) & 0x01;
        int exp8 = (fp8_val >> 3) & 0x0F;
        uint8_t mant8 = fp8_val & 0x07;

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

    // 转换为浮点值
    float to_float() const { return fp8_to_float_scaled(data); }

    // 兼容性转换函数
    operator float() const { return fp8_to_float_scaled(data); }

    // 符号判断（用于二值化）
    int get_sign() const {
        float val = to_float();
        return (val > 0.0f) ? 1 : 0;
    }
};

// FP8转换函数
inline float fp8_to_float(fp8_t h) {
    return h.to_float();
}

inline fp8_t float_to_fp8(float f) {
    return fp8_t(f);
}

// ============================================================================
// SIMD批量转换优化（减少逐元素转换开销）
// ============================================================================

// 批量转换：FP8 -> FP32 (32个元素，优化内存访问)
inline void batch_fp8_to_float(const fp8_t* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = fp8_to_float(src[i]);
    }
}

// 批量转换：FP32 -> FP8 (32个元素，优化内存访问)
inline void batch_float_to_fp8(const float* src, fp8_t* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = float_to_fp8(src[i]);
    }
}

// 批量转换：FP8 -> FP32 + 归一化到[0,1] (一次转换完成两步)
inline void batch_fp8_to_norm(const fp8_t* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = (fp8_to_float(src[i]) + 1.0f) * 0.5f;
    }
}

// 批处理数据结构（优化内存布局）
struct BatchData {
    vector<fp8_t> x;             // 位置状态（PyTorch float8_e4m3fn）
    vector<fp8_t> y;             // 动量状态（PyTorch float8_e4m3fn）
    vector<fp8_t> gradient_term; // 梯度项（PyTorch float8_e4m3fn）
    float energy;                // 能量值（float存储）

    // 预分配的临时数组（避免重复分配）
    vector<float> x_float;       // FP32临时数组
    vector<float> y_float;       // FP32临时数组
    vector<float> grad_float;    // FP32临时数组
    vector<float> x_norm;        // [0,1]归一化临时数组

    BatchData(int num_vars) :
        x(num_vars), y(num_vars), gradient_term(num_vars), energy(0.0f),
        x_float(num_vars), y_float(num_vars), grad_float(num_vars), x_norm(num_vars) {}
};

// BSB求解器（基于INT8成功策略优化版）
class BSBFP8Solver {
private:
    ProblemModel model;
    vector<vector<int>> var_to_terms;
    vector<vector<int>> term_positions;
    float fixed_xi;  // 使用float存储xi

public:
    BSBFP8Solver(const ProblemModel& problem_model, float optimized_xi = 0.0f) : model(problem_model) {
        precompute_lookup_tables();
        
        // 使用float存储xi值
        if (optimized_xi > 0.0f) {
            fixed_xi = optimized_xi;
        } else {
            fixed_xi = static_cast<float>(model.xi);
        }
    }

    void precompute_lookup_tables() {
        var_to_terms.resize(model.num_vars);
        
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

    // 初始化（PyTorch FP8存储，使用float精度）
    void init_batches(vector<BatchData>& batches, uint64_t seed, float sigma = 0.1f) {
        mt19937 gen(seed);
        normal_distribution<float> dis_normal(0.0f, 1.0f);

        for (size_t batch = 0; batch < batches.size(); ++batch) {
            mt19937 local_gen(seed + batch);
            for (int i = 0; i < model.num_vars; ++i) {
                // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5
                float x_normalized = sigma * dis_normal(local_gen) + 0.5f;
                x_normalized = (x_normalized < 0.0f) ? 0.0f : (x_normalized > 1.0f) ? 1.0f : x_normalized;

                // 转换到[-1,1]范围，存储为FP8
                float x_range = x_normalized * 2.0f - 1.0f;
                batches[batch].x[i] = float_to_fp8(x_range);

                // y初始化为小随机值：sigma * N(0,1)，存储为FP8
                float y_val = sigma * dis_normal(local_gen);
                batches[batch].y[i] = float_to_fp8(y_val);

                // 初始化梯度项为0
                batches[batch].gradient_term[i] = float_to_fp8(0.0f);
            }
        }
    }

    // 批量计算梯度（PyTorch FP8存储 + FP32计算，参考INT8成功实现）
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
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

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
                    double J_ij = static_cast<double>(direction_sign) * model.quad_terms_coeffs[t];

                    // 累加到临时数组
                    batch.grad_float[var1] += static_cast<float>(J_ij / 2.0) * batch.x_norm[var2];
                    batch.grad_float[var2] += static_cast<float>(J_ij / 2.0) * batch.x_norm[var1];
                }
            }

            // 存储回FP8（批量转换）
            batch_float_to_fp8(batch.grad_float.data(), batch.gradient_term.data(), model.num_vars);
        }
    }

    // 更新辅助状态y（PyTorch FP8存储 + FP32计算，参考INT8）
    // BSB公式：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x)
    void batch_update_y(vector<BatchData>& batches, float p, float dt, float delta) {
        // 泵浦振幅项：-(1-p(t))
        const float pump_factor = -(1.0f - p);
        const float xi_val = fixed_xi;
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_fp8_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);
            batch_fp8_to_float(batch.gradient_term.data(), batch.grad_float.data(), model.num_vars);

            // 批量更新y值（AVX2优化）
            int i = 0;
#ifdef __AVX2__
            for (; i <= model.num_vars - 4; i += 4) {
                __m128 x_vals = _mm_loadu_ps(&batch.x_float[i]);
                __m128 y_vals = _mm_loadu_ps(&batch.y_float[i]);
                __m128 grad_vals = _mm_loadu_ps(&batch.grad_float[i]);

                // 计算x-0.5
                __m128 x_norm = _mm_add_ps(x_vals, _mm_set1_ps(1.0f));  // x + 1
                x_norm = _mm_mul_ps(x_norm, _mm_set1_ps(0.5f));  // (x+1)*0.5
                __m128 x_shifted = _mm_sub_ps(x_norm, _mm_set1_ps(0.5f));

                // 计算更新
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
                float x_shifted = (batch.x_float[i] + 1.0f) * 0.5f - 0.5f;
                float dy = (pump_factor * x_shifted + xi_val * batch.grad_float[i]) * dt_val;
                batch.y_float[i] = clamp_float(batch.y_float[i] + dy, -1.0f, 1.0f);
            }

            // 存储回FP8（批量转换）
            batch_float_to_fp8(batch.y_float.data(), batch.y.data(), model.num_vars);
        }
    }

    // 更新主状态x（PyTorch FP8存储 + FP32计算，参考INT8）
    // BSB公式：dx/dt = y
    void batch_update_x(vector<BatchData>& batches, float dt, float delta) {
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_fp8_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);

            // 批量更新x值（AVX2优化）
            int i = 0;
#ifdef __AVX2__
            for (; i <= model.num_vars - 4; i += 4) {
                __m128 x_vals = _mm_loadu_ps(&batch.x_float[i]);
                __m128 y_vals = _mm_loadu_ps(&batch.y_float[i]);

                // 转换到[0,1]进行更新
                __m128 x_norm = _mm_add_ps(x_vals, _mm_set1_ps(1.0f));  // x + 1
                x_norm = _mm_mul_ps(x_norm, _mm_set1_ps(0.5f));  // (x+1)*0.5

                // 更新：x += y * dt
                __m128 y_dt = _mm_mul_ps(y_vals, _mm_set1_ps(dt_val));
                __m128 x_new = _mm_add_ps(x_norm, y_dt);

                // 限制在[0,1]
                x_new = _mm_max_ps(_mm_setzero_ps(), _mm_min_ps(x_new, _mm_set1_ps(1.0f)));

                // 转换回[-1,1]
                __m128 x_fp8_val = _mm_sub_ps(x_new, _mm_set1_ps(0.5f));  // x - 0.5
                x_fp8_val = _mm_mul_ps(x_fp8_val, _mm_set1_ps(2.0f));  // (x-0.5)*2
                x_fp8_val = _mm_max_ps(_mm_set1_ps(-1.0f), _mm_min_ps(x_fp8_val, _mm_set1_ps(1.0f)));

                // 存储回临时数组
                _mm_storeu_ps(&batch.x_float[i], x_fp8_val);
            }
#endif
            // 处理剩余变量
            for (; i < model.num_vars; ++i) {
                float x_norm = (batch.x_float[i] + 1.0f) * 0.5f;  // [0,1]
                float x_new = clamp_float(x_norm + batch.y_float[i] * dt_val, 0.0f, 1.0f);
                batch.x_float[i] = clamp_float(x_new * 2.0f - 1.0f, -1.0f, 1.0f);
            }

            // 存储回FP8（批量转换）
            batch_float_to_fp8(batch.x_float.data(), batch.x.data(), model.num_vars);
        }
    }

    // 边界约束（PyTorch FP8存储 + FP32计算，参考INT8）
    // 将x限制在[0,1]范围内，超出时投影到边界{0,1}并清零动量
    void batch_apply_constraint(vector<BatchData>& batches) {
        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_fp8_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);

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

                    // 存储回FP8
                    for (int k = 0; k < 4; ++k) {
                        batch.x[i + k] = float_to_fp8(batch.x_float[i + k]);
                        batch.y[i + k] = float_to_fp8(batch.y_float[i + k]);
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

                    // 存储回FP8
                    batch.x[i] = float_to_fp8(batch.x_float[i]);
                    batch.y[i] = float_to_fp8(batch.y_float[i]);
                }
            }
        }
    }

    // 参考BSB_INT8策略：直接使用连续变量计算能量（PyTorch FP8存储）
    float calc_batch_energy(const BatchData& batch) {
        float scaled_energy = 0.0f;

        // 检查是否是纯线性问题
        bool is_linear_problem = (model.num_quad_terms == 0);

        // 预转换所有FP8 x值到FP32（使用预分配的临时数组）
        vector<float> x_float(model.num_vars);
        batch_fp8_to_float(batch.x.data(), x_float.data(), model.num_vars);

        // 二次项贡献：0.5 * ∑_{i,j} J_{ij}x_i x_j（先计算二次项，与INT8一致）
        if (model.num_quad_terms > 0) {
            for (int t = 0; t < model.num_quad_terms; ++t) {
                int var1 = model.quad_terms_indices[t][0];
                int var2 = model.quad_terms_indices[t][1];
                float J_ij = static_cast<float>(model.quad_terms_coeffs[t]);

                // 使用连续变量计算
                float x1_norm = (x_float[var1] + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
                float x2_norm = (x_float[var2] + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]

                scaled_energy += 0.5f * J_ij * x1_norm * x2_norm;
            }
        }

        // 线性项贡献：∑_i h_i x_i（后计算线性项，与INT8一致）
        for (int i = 0; i < model.num_vars; ++i) {
            float x_continuous = (x_float[i] + 1.0f) * 0.5f;  // [-1.0,1.0] -> [0,1]
            scaled_energy += static_cast<float>(model.linear_terms[i]) * x_continuous;
        }

        // 缩放处理
        float original_energy;
        if (is_linear_problem) {
            // 纯线性问题，不进行缩放逆变换
            original_energy = scaled_energy;
        } else {
            // 重新缩放回原始能量
            original_energy = scaled_energy * static_cast<float>(model.scaling_factor);
        }

        // 添加常数项（如果存在）
        if (model.constant_term != 0.0) {
            original_energy += static_cast<float>(model.constant_term);
        }

        return original_energy;
    }

    // 找到最优批次（参考INT8成功实现）
    int find_best_batch(vector<BatchData>& batches) {
        int best_idx = 0;
        float best_value = calc_batch_energy(batches[0]);
        batches[0].energy = best_value;

        bool is_maximize = (model.objective_sense == "maximize");

        for (size_t i = 1; i < batches.size(); ++i) {
            batches[i].energy = calc_batch_energy(batches[i]);
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
        return best_idx;
    }

    // 核心求解函数（参考BSB_INT8，使用float精度）
    vector<fp8_t> solve(int max_iters = 1000, int batch_size = 100, float sigma = 0.1f, vector<BatchData>* out_batches = nullptr) {
        float dt = 0.1f;     // 参考BSB_FP16，使用dt=0.1
        float delta = 0.9f;   // 参考BSB_FP16的delta值

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << model.scaling_factor << endl;
        cout << "  - BSB Algorithm (FP8, continuous variables like INT8)" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using xi=" << fixed << setprecision(6) << fixed_xi << ", sigma=" << sigma << ", dt=" << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - Energy calculation: Continuous variables (simple accumulation like INT8)" << endl;
        cout << "  - FP8 Format: PyTorch float8_e4m3fn" << endl;
        cout << "  - Initialization: double precision (matching INT8 for better accuracy)" << endl;

        vector<BatchData> batches;
        batches.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batches.emplace_back(model.num_vars);
        }
        init_batches(batches, time(nullptr), sigma);  // 使用更小的sigma=0.1

        auto start = high_resolution_clock::now();

        // 用于记录收敛过程的数据结构
        struct ConvergenceData {
            int iteration;
            double cumulative_time_ms;
            float best_objective;
        };
        vector<ConvergenceData> convergence_history;
        float current_best_overall = model.objective_sense == "maximize" ? -numeric_limits<float>::infinity() : numeric_limits<float>::infinity();

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

        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;

        // 计算最后一批的能量值（仅计算一次）
        for (size_t i = 0; i < batches.size(); ++i) {
            batches[i].energy = calc_batch_energy(batches[i]);
        }

        int best_idx = find_best_batch(batches);
        vector<fp8_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            // 直接返回FP8存储的值
            solution[i] = batches[best_idx].x[i];
        }

        if (out_batches != nullptr) {
            *out_batches = batches;
        }

        return solution;
    }

    /**
     * @brief 获取模型中的变量数量
     */
    int get_num_vars() const {
        return model.num_vars;
    }
};

// 主函数
int main(int argc, char* argv[]) {
    string qplib_file_path;
    float optimized_xi = 0.365f;
    
    if (argc > 1) {
        qplib_file_path = argv[1];
    } else {
        qplib_file_path = DEFAULT_QPLIB_FILE_PATH;
    }
    
    if (argc > 2) {
        optimized_xi = stod(argv[2]);
    }

    cout << "========================================" << endl;
    cout << "    BSB_CPU_fp8 Solver (PyTorch FP8) " << endl;
    cout << "  - File path: " << qplib_file_path << endl;
    cout << "  - Algorithm: BSB (Ballistic Simulated Bifurcation)" << endl;
    cout << "  - Key feature: PyTorch float8_e4m3fn storage + FP32 computation" << endl;
    cout << "  - Storage: FP8 (PyTorch format), Computation: FP32 (high precision)" << endl;
    cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
    cout << "  - Energy method: Continuous variables (like FP16), dt=0.1" << endl;
    cout << "========================================" << endl;

    ProblemModel model;
    bool parse_result = QPLIBParser::parseQplibFile(qplib_file_path, model);
    
    if (!parse_result) {
        cerr << "ERROR: File parsing failed, program exit" << endl;
        return 1;
    }

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
    cout << "  Scaling factor: " << fixed << setprecision(6) << model.scaling_factor << endl;
    cout << "  Coupling strength parameter: " << fixed << setprecision(6) << model.xi << endl;
    cout << "========================================" << endl;

    if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
        model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
    }

    try {
        cout << "Using fixed optimized xi value: " << fixed << setprecision(6) << optimized_xi << endl;
        BSBFP8Solver solver(model, optimized_xi);
        vector<BatchData> batches;

        auto start = high_resolution_clock::now();
        vector<fp8_t> solution = solver.solve(1000, 100, 0.1f, &batches);
        auto end = high_resolution_clock::now();
        double duration = duration_cast<microseconds>(end - start).count() / 1000.0;
        cout << "TIME: Solving completed, total time: " << fixed << setprecision(6) << duration << " ms" << endl;
        
        double best_value = batches[0].energy;
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

        cout << "RESULT: Solution (" << model.num_vars << " variables, binary [0/1]): " << endl;
        int count_ones = 0;

        for (int i = 0; i < model.num_vars; ++i) {
            float x_val = fp8_to_float(batches[best_idx].x[i]);
            float x_norm = (x_val + 1.0f) / 2.0f;
            int binary_val = (x_norm > 0.5f) ? 1 : 0;
            count_ones += binary_val;

            if ((i + 1) % 17 == 0 || i == model.num_vars - 1) {
                cout << binary_val << endl;
            } else {
                cout << binary_val << ", ";
            }
        }
        
        int count_zeros = model.num_vars - count_ones;
        double density = static_cast<double>(count_ones) / model.num_vars * 100.0;
        
        cout << endl << "Solution statistics:" << endl;
        cout << "  - Variables set to 1: " << count_ones << "/" << model.num_vars << endl;
        cout << "  - Variables set to 0: " << count_zeros << "/" << model.num_vars << endl;
        cout << "  - Solution density: " << fixed << setprecision(2) << density << "%" << endl;
        
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