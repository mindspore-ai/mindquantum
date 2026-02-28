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

// ============================================================================
// PyTorch FP8 E4M3FN 配置（真正IEEE 754浮点数）
// ============================================================================
// PyTorch float8_e4m3fn 标准：1符号位 + 4指数位 + 3尾数位
// constexpr float FP8_MAX_VAL = 240.0f;   // 最大正值（未使用，注释以消除警告）
// constexpr float FP8_MIN_VAL = -240.0f;  // 最小负值（未使用，注释以消除警告）
// constexpr float FP8_EPSILON = 0.125f;   // 最小精度（2^-3）（未使用，注释以消除警告）

// PyTorch FP8 E4M3FN 转换器（完整IEEE 754浮点实现）
struct FP8PyTorch {
    // float -> FP8 E4M3FN bits
    static inline uint8_t quantize(float val) {
        union { float f; uint32_t u; } f2u;
        f2u.f = val;

        uint32_t sign = (f2u.u >> 31) & 0x1;
        uint32_t exp = (f2u.u >> 23) & 0xFF;
        uint32_t mantissa = f2u.u & 0x7FFFFF;

        // 处理特殊值
        if (exp == 0xFF) {
            // NaN 或 Inf
            if (mantissa == 0) {
                // Inf
                return (sign << 7) | 0x7E;  // FP8 Inf: 1,11110,00
            } else {
                // NaN
                return (sign << 7) | 0x7F;  // FP8 NaN: 1,11111,00
            }
        }

        if (exp == 0 && mantissa == 0) {
            // Zero
            return sign << 7;
        }

        // 调整指数：float bias 127 -> fp8 bias 7
        int32_t fp8_exp = exp - 127 + 7;

        // 处理下溢和上溢
        if (fp8_exp >= 0x0F) {
            // 上溢 -> Inf
            return (sign << 7) | 0x7E;
        }

        if (fp8_exp <= 0) {
            // 下溢或非规格化
            // E4M3FN 没有 subnormal，直接截断为0
            if (val > 0.0f) return 0x80;  // +0
            if (val < 0.0f) return 0x00;  // -0
            return 0x80;  // +0
        }

        // 截断尾数到3位
        uint8_t fp8_mantissa = (mantissa >> 20) & 0x07;

        return (sign << 7) | (fp8_exp << 3) | fp8_mantissa;
    }

    // FP8 E4M3FN bits -> float
    static inline float dequantize(uint8_t fp8_val) {
        uint8_t sign = (fp8_val >> 7) & 0x1;
        uint8_t exp = (fp8_val >> 3) & 0x0F;
        uint8_t mantissa = fp8_val & 0x07;

        // 特殊值
        if (exp == 0x0F) {
            // Inf 或 NaN
            if (mantissa == 0) {
                // Inf
                union { float f; uint32_t u; } u2f;
                u2f.u = (sign << 31) | 0x7F800000;  // float Inf
                return u2f.f;
            } else {
                // NaN
                union { float f; uint32_t u; } u2f;
                u2f.u = (sign << 31) | 0x7FC00000;  // float NaN
                return u2f.f;
            }
        }

        // 调整指数：fp8 bias 7 -> float bias 127
        uint32_t f_exp = (exp == 0) ? 0 : exp + 120;

        // 扩展尾数到23位
        uint32_t f_mantissa = mantissa << 20;

        union { float f; uint32_t u; } u2f;
        u2f.u = (sign << 31) | (f_exp << 23) | f_mantissa;
        return u2f.f;
    }
};

// FP8数据结构（简化版本，兼容原有接口）
struct fp8_t {
    uint8_t data;  // 8位存储

    fp8_t() : data(0x00) {}
    fp8_t(float f) : data(FP8PyTorch::quantize(f)) {}

    // 转换为浮点值（使用FP8PyTorch::dequantize）
    float to_float() const { return FP8PyTorch::dequantize(data); }

    // 兼容性转换函数
    operator float() const { return FP8PyTorch::dequantize(data); }

    // x_hat判断（DSB特征：不使用sign，使用分段函数返回{0,1}）
    int get_x_hat() const {
        float val = to_float();  // [-1,1]
        float x_normalized = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
        return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
    }
};

// FP8转换函数（参考GPU版本）
inline float fp8_to_float(fp8_t h) {
    return h.to_float();
}

inline fp8_t float_to_fp8(float f) {
    return fp8_t(f);
}

// 自定义x^分段函数（DSB特征：不使用sign，使用分段函数）
// 参考项目报告3.md第232行定义：x̂_i = 1 if x_i > 0.5, else 0
inline int x_hat_fp8(fp8_t x) {
    float val = fp8_to_float(x);  // [-1,1]
    float x_normalized = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
}

// FP8范围限制函数（参考GPU FP8版本）
inline float fp8_clamp(float val) {
    // 限制在[-1, 1]范围（避免饱和）
    return (val > 0.0f) ? ((val > 1.0f) ? 1.0f : val) : ((val < -1.0f) ? -1.0f : val);
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
// 关键改进：参考GPU版本的[-1,1] -> [0,1]映射
inline void batch_fp8_to_norm(const fp8_t* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        float val = fp8_to_float(src[i]);  // [-1,1]
        dst[i] = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    }
}

// 批处理数据结构（完全匹配GPU版本第101-109行）
struct BatchData {
    vector<fp8_t> x;             // 位置状态（FP8，完全匹配GPU第104行）
    vector<float> y;             // 动量状态（float，完全匹配GPU第105行）
    vector<float> gradient_term;   // 梯度项（float，完全匹配GPU第107行）
    vector<int> x_hat;           // 离散二值化变量{0,1}（匹配项目报告定义）
    float energy;                // 能量值（float保持精度一致性）

    // 预分配的临时数组（避免重复分配）
    vector<float> x_float;       // FP32临时数组
    vector<float> grad_float;    // FP32临时数组
    vector<float> x_norm;        // [0,1]归一化临时数组

    BatchData(int num_vars) :
        x(num_vars), y(num_vars), x_hat(num_vars),
        gradient_term(num_vars), energy(0.0f),
        x_float(num_vars), grad_float(num_vars), x_norm(num_vars) {}
};

// DSB求解器（基于BSB_FP8成功策略，使用sign(x)）
class DSBFP8Solver {
private:
    ProblemModel model;
    vector<vector<int>> var_to_terms;
    vector<vector<int>> term_positions;
    float fixed_xi;  // 使用float存储xi，与GPU版本一致

public:
    DSBFP8Solver(const ProblemModel& problem_model, float optimized_xi = 0.0f) : model(problem_model) {
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

    // 初始化（参考GPU FP8版本，使用真正的PyTorch FP8）
    void init_batches(vector<BatchData>& batches, uint64_t seed, float sigma = 0.1f) {
        mt19937 gen(seed);
        normal_distribution<float> dis_normal(0.0f, 1.0f);  // 使用float精度（匹配GPU）

        for (size_t batch = 0; batch < batches.size(); ++batch) {
            mt19937 local_gen(seed + batch);
            for (int i = 0; i < model.num_vars; ++i) {
                // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5（匹配GPU FP8第126-127行）
                float x_normalized = static_cast<float>(sigma) * dis_normal(local_gen) + 0.5f;
                x_normalized = clamp_float(x_normalized, 0.0f, 1.0f);

                // 映射到[-1, 1]范围（匹配GPU FP8第132-133行）
                float x_scaled = (x_normalized - 0.5f) * 2.0f;
                x_scaled = clamp_float(x_scaled, -0.99f, 0.99f);  // 避免边界饱和
                batches[batch].x[i] = float_to_fp8(x_scaled);  // 存储为FP8

                // y初始化为小随机值：sigma * N(0,1)，使用float存储（完全匹配GPU第137-138行）
                float y_val = static_cast<float>(sigma) * dis_normal(local_gen);
                batches[batch].y[i] = y_val;  // float存储，完全匹配GPU第105行

                // 初始化x_hat（匹配项目报告定义）
                batches[batch].x_hat[i] = x_hat_fp8(batches[batch].x[i]);  // {0,1}二值化

                // 初始化gradient_term为0（匹配GPU FP8第141行）
                batches[batch].gradient_term[i] = 0.0f;  // float存储
            }
        }
    }

    // 批量计算离散二值化变量（匹配项目报告定义）
    // x_hat是{0,1}的二值变量，不使用sign函数
    void batch_calc_x_hat(vector<BatchData>& batches) {
        for (auto& batch : batches) {
            for (int i = 0; i < model.num_vars; ++i) {
                batch.x_hat[i] = x_hat_fp8(batch.x[i]);  // 返回{0,1}
            }
        }
    }

    // 批量计算梯度（匹配GPU版本第157-197行）
    // 关键改进：gradient_term使用float存储（匹配GPU第196行）
    void batch_calc_gradient(vector<BatchData>& batches) {
        // 确保linear_terms大小与变量数量一致
        if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
            model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
        }

        // 根据优化方向确定梯度符号（匹配GPU第172行）
        bool is_maximize = (model.objective_sense == "maximize");
        float direction_sign = is_maximize ? 1.0f : -1.0f;

        // 先计算所有批次的离散二值化变量（DSB特征：{0,1}二值化）
        batch_calc_x_hat(batches);

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 初始化梯度为线性项系数（匹配GPU第175行）
            for (int i = 0; i < model.num_vars; ++i) {
                batch.grad_float[i] = direction_sign * static_cast<float>(model.linear_terms[i]);
            }

            // 累加二次项贡献（匹配GPU第179-193行）
            // DSB特征：使用{0,1}二值化变量进行计算（不使用sign）
            if (model.num_quad_terms > 0) {
                for (int t = 0; t < model.num_quad_terms; ++t) {
                    int var1 = model.quad_terms_indices[t][0];
                    int var2 = model.quad_terms_indices[t][1];
                    float J_ij = direction_sign * static_cast<float>(model.quad_terms_coeffs[t]);

                    // 直接使用{0,1}二值化变量（匹配项目报告定义）
                    int x2_hat = batch.x_hat[var2];  // 0或1
                    int x1_hat = batch.x_hat[var1];  // 0或1

                    batch.grad_float[var1] += (J_ij / 2.0f) * static_cast<float>(x2_hat);
                    batch.grad_float[var2] += (J_ij / 2.0f) * static_cast<float>(x1_hat);
                }
            }

            // 直接存储为float（匹配GPU第196行，不转换为FP8）
            for (int i = 0; i < model.num_vars; ++i) {
                batch.gradient_term[i] = batch.grad_float[i];
            }
        }
    }

    // 更新辅助状态y（完全匹配GPU版本第199-229行）
    void batch_update_y(vector<BatchData>& batches, float p, float dt, float delta) {
        // 泵浦振幅项：-(1-p(t))
        const float pump_factor = -(1.0f - p);
        const float xi_val = fixed_xi;
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 直接使用gradient_term（已是float，完全匹配GPU第107行）
            for (int i = 0; i < model.num_vars; ++i) {
                float x_val = batch.x_float[i];  // [-1,1]
                float x_normalized = (x_val + 1.0f) * 0.5f;  // [0,1]
                float pump_term = pump_factor * (x_normalized - 0.5f);
                float gradient_term = xi_val * batch.gradient_term[i];

                // DSB公式：dy = [-(1-p)*(x-0.5) + ξ*∇f] * dt
                float dy = (pump_term + gradient_term) * dt_val;
                batch.y[i] += dy;  // float存储，完全匹配GPU第105行
            }
        }
    }

    // 更新主状态x（完全匹配GPU版本第231-257行）
    // BSB公式：dx/dt = y
    void batch_update_x(vector<BatchData>& batches, float dt, float delta) {
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 批量更新x值（AVX2优化）
            int i = 0;
#ifdef __AVX2__
            for (; i <= model.num_vars - 4; i += 4) {
                __m128 x_vals = _mm_loadu_ps(&batch.x_float[i]);
                __m128 y_vals = _mm_loadu_ps(&batch.y[i]);  // 直接使用float y

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
                float x_new = clamp_float(x_norm + batch.y[i] * dt_val, 0.0f, 1.0f);  // 直接使用float y
                batch.x_float[i] = clamp_float(x_new * 2.0f - 1.0f, -1.0f, 1.0f);
            }

            // 存储回FP8（批量转换）
            batch_float_to_fp8(batch.x_float.data(), batch.x.data(), model.num_vars);
        }
    }

    // 边界约束（完全匹配GPU版本第259-289行）
    // 将x限制在[0,1]范围内，超出时投影到边界{0,1}并清零动量
    void batch_apply_constraint(vector<BatchData>& batches) {
        for (auto& batch : batches) {
            // 使用预分配的临时数组（避免重复分配）
            batch_fp8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 处理边界约束（完全匹配GPU版本第259-289行）
            for (int i = 0; i < model.num_vars; ++i) {
                // 将float值转换到[0,1]范围
                float x_val = batch.x_float[i];  // [-1,1]
                float x_normalized = (x_val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]

                // 检查是否超出边界
                float distance_from_center = fabsf(x_normalized - 0.5f);

                if (distance_from_center > 0.5f) {
                    // 超出边界，投影到边界{0,1}
                    float projected_value = (x_normalized > 0.5f) ? 1.0f : 0.0f;
                    batch.x_float[i] = (projected_value * 2.0f) - 1.0f;  // [0,1] -> [-1,1]
                    batch.y[i] = 0.0f;  // 清零动量（float，完全匹配GPU第105行）
                }
            }

            // 存储回FP8（批量转换）
            batch_float_to_fp8(batch.x_float.data(), batch.x.data(), model.num_vars);
        }
    }

    // 计算单个批次的目标函数值（匹配项目报告定义）
    // DSB特征：使用{0,1}二值化变量进行能量计算（不使用sign）
    float calc_batch_energy(const BatchData& batch) {
        float scaled_energy = 0.0f;
        bool is_linear_problem = (model.num_quad_terms == 0);

        // 二次项贡献：0.5 * ∑_{i,j} J_{ij}x̂_i x̂_j
        if (model.num_quad_terms > 0) {
            for (int t = 0; t < model.num_quad_terms; ++t) {
                int var1 = model.quad_terms_indices[t][0];
                int var2 = model.quad_terms_indices[t][1];
                float J_ij = static_cast<float>(model.quad_terms_coeffs[t]);

                // 直接使用{0,1}二值化变量（匹配项目报告定义）
                int x1_hat = batch.x_hat[var1];  // 0或1
                int x2_hat = batch.x_hat[var2];  // 0或1

                scaled_energy += 0.5f * J_ij * static_cast<float>(x1_hat * x2_hat);
            }
        }

        // 线性项贡献：∑_i h_i x̂_i
        for (int i = 0; i < model.num_vars; ++i) {
            // 直接使用{0,1}二值化变量（匹配项目报告定义）
            int x_hat = batch.x_hat[i];  // 0或1
            scaled_energy += static_cast<float>(model.linear_terms[i]) * static_cast<float>(x_hat);
        }

        // 缩放处理（匹配GPU第420-428行）
        float original_energy;
        if (is_linear_problem) {
            original_energy = scaled_energy;
        } else {
            original_energy = scaled_energy * static_cast<float>(model.scaling_factor);
        }

        // 添加常数项（如果存在）
        if (model.constant_term != 0.0) {
            original_energy += static_cast<float>(model.constant_term);
        }

        return original_energy;
    }

    // 找到最优批次
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

    // 核心求解函数（基于BSB_FP8成功策略）
    vector<fp8_t> solve(int max_iters = 1000, int batch_size = 100, float sigma = 0.1f, vector<BatchData>* out_batches = nullptr) {
        float dt = 0.1f;
        float delta = 0.9f;

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << model.scaling_factor << endl;
        cout << "  - DSB Algorithm (FP8, PyTorch float8_e4m3fn - Simplified, GPU-inspired)" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using xi=" << fixed << setprecision(6) << fixed_xi << ", sigma=" << sigma << ", dt=" << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - Key feature: x̂(x) -> {0,1} discretization (no sign function), PyTorch FP8 format" << endl;
        cout << "  - FP8 Format: PyTorch float8_e4m3fn (Simplified, [-1,1] range)" << endl;
        cout << "  - Energy calculation: Discrete {0,1} values (DSB characteristic, matches project report)" << endl;
        cout << "  - Storage: FP8 for x,y; Float for gradient_term (matches GPU)" << endl;
        cout << "  - Performance optimized: Reference GPU implementation" << endl;

        vector<BatchData> batches;
        batches.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batches.emplace_back(model.num_vars);
        }
        init_batches(batches, time(nullptr), sigma);

        auto start = high_resolution_clock::now();
        
        // 用于记录收敛过程的数据结构
        struct ConvergenceData {
            int iteration;
            float cumulative_time_ms;
            float best_objective;
        };
        vector<ConvergenceData> convergence_history;
        float current_best_overall = model.objective_sense == "maximize" ? -numeric_limits<float>::infinity() : numeric_limits<float>::infinity();

        for (int iter = 0; iter < max_iters; ++iter) {
            float p = static_cast<float>(iter) / (max_iters - 1);

            // 每次迭代都计算当前时间（用于准确记录收敛时间）
            auto current_time = high_resolution_clock::now();
            float cumulative_time = static_cast<float>(duration_cast<microseconds>(current_time - start).count()) / 1000.0f;

            // DSB动力学更新序列（参考FP16版本第418-429行）
            // 1. 计算梯度（使用sign(x)而不是连续变量x）
            batch_calc_gradient(batches);

            // 2. 更新动量：dy/dt = -(1-p(t))*sgn(x) + ξ*∇f(x)
            batch_update_y(batches, p, dt, delta);

            // 3. 更新位置：dx/dt = y
            batch_update_x(batches, dt, delta);

            // 4. 边界条件处理：将x限制在[0,1]范围内，超出时投影到边界并清零动量
            batch_apply_constraint(batches);

            // 5. 关键修复：在每次迭代结束时更新x_hat（匹配项目报告定义）
            // 因为batch_update_x和batch_apply_constraint改变了x，需要同步更新x_hat
            batch_calc_x_hat(batches);

            // 只在最后一次迭代记录收敛数据和输出能量
            if (iter == max_iters - 1) {
                // 计算当前最佳目标函数值（x_hat已在迭代结束时更新）
                float current_best = calc_batch_energy(batches[0]);
                bool is_maximize = (model.objective_sense == "maximize");

                for (size_t i = 1; i < batches.size(); ++i) {
                    float energy = calc_batch_energy(batches[i]);
                    if (is_maximize) {
                        if (energy > current_best) {
                            current_best = energy;
                        }
                    } else {
                        if (energy < current_best) {
                            current_best = energy;
                        }
                    }
                }

                // 更新全局最佳值
                if (is_maximize) {
                    if (current_best > current_best_overall) current_best_overall = current_best;
                } else {
                    if (current_best < current_best_overall) current_best_overall = current_best;
                }

                // 记录最终收敛数据
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

        auto end = high_resolution_clock::now();
        float duration = static_cast<float>(duration_cast<microseconds>(end - start).count()) / 1000.0f;
        cout << "C++_TIME: Total solving time: " << fixed << setprecision(3) << duration << " ms" << endl;

        // 生成最优解（x_sign已在最后一次迭代结束时更新）
        int best_idx = find_best_batch(batches);
        vector<fp8_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            solution[i] = batches[best_idx].x[i];
        }

        cout << "BEST: Optimal objective value: " << fixed << setprecision(3) << batches[best_idx].energy << endl;

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
        optimized_xi = stof(argv[2]);
    }

        cout << "========================================" << endl;
        cout << "    DSB_CPU_fp8 Solver (PyTorch FP8 - Simplified) " << endl;
        cout << "  - File path: " << qplib_file_path << endl;
        cout << "  - Algorithm: DSB (Discrete Simulated Bifurcation)" << endl;
        cout << "  - Key feature: PyTorch float8_e4m3fn storage + FP32 computation (Simplified)" << endl;
        cout << "  - Storage: FP8 (PyTorch format, [-1,1] range), Computation: FP32" << endl;
        cout << "  - Quantization: Direct [-1,1] mapping (matches GPU implementation)" << endl;
        cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
        cout << "  - Energy method: Discrete {0,1} values using x̂(x) function (no sign)" << endl;
        cout << "  - Based on project report 3.md with x̂_i = 1 if x_i > 0.5 else 0" << endl;
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
        cout << "Using fixed optimized xi value: " << fixed << setprecision(6) << optimized_xi
             << " (auto-calculated value: " << fixed << setprecision(6) << model.xi << ")" << endl;
        DSBFP8Solver solver(model, optimized_xi);
        vector<BatchData> batches;
        vector<fp8_t> solution = solver.solve(1000, 100, 0.1, &batches);
        
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

        cout << "RESULT: Solution (" << model.num_vars << " variables, binary [0/1]): " << endl;
        int count_ones = 0;
        
        for (int i = 0; i < model.num_vars; ++i) {
            // 直接使用{0,1}二值化变量（匹配项目报告定义）
            int binary_val = batches[best_idx].x_hat[i];  // 0或1
            count_ones += binary_val;
            
            if ((i + 1) % 17 == 0 || i == model.num_vars - 1) {
                cout << binary_val << endl;
            } else {
                cout << binary_val << ", ";
            }
        }
        
        int count_zeros = model.num_vars - count_ones;
        float density = static_cast<float>(count_ones) / model.num_vars * 100.0f;
        
        cout << endl << "Solution statistics:" << endl;
        cout << "  - Variables set to 1: " << count_ones << "/" << model.num_vars << endl;
        cout << "  - Variables set to 0: " << count_zeros << "/" << model.num_vars << endl;
        cout << "  - Solution density: " << fixed << setprecision(2) << density << "%" << endl;
        
        cout << endl << "========================================" << endl;
        cout << "  ** DSB FINAL RESULTS **" << endl;
        cout << "========================================" << endl;
        cout << "  Problem: " << qplib_file_path.substr(qplib_file_path.find_last_of("/\\") + 1) << endl;
        cout << "  Objective sense: " << model.objective_sense << endl;
        cout << "  Number of variables: " << model.num_vars << endl;
        cout << "  Number of quadratic terms: " << model.num_quad_terms << endl;
        cout << "----------------------------------------" << endl;
        cout << "  DSB OBJECTIVE VALUE: " << fixed << setprecision(4) << best_value << endl;
        cout << "  Storage: FP8 for x,y; Float for gradient_term; x̂ function for {0,1}" << endl;
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