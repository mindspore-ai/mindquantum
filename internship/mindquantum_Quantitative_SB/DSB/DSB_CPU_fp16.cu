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
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// ============================================================================
// PyTorch FP16 配置 (IEEE 754 half precision)
// ============================================================================
// PyTorch float16 使用 IEEE 754 half precision 格式：1符号位 + 5指数位 + 10尾数位
// FP16数据结构（模拟PyTorch float16 - IEEE 754 half precision）
struct fp16_t {
    uint16_t bits;  // 16位存储（IEEE 754 half precision格式）

    fp16_t() : bits(0) {}

    // 从 float 构造 fp16_t（模拟 PyTorch float16 转换）
    fp16_t(float f) { bits = float_to_fp16_bits(f); }

    // 转换为 float
    inline float to_float() const {
        return fp16_bits_to_float(bits);
    }

    // 兼容性转换函数
    operator float() const { return to_float(); }

    // x_hat判断（DSB特征：不使用sign，使用分段函数返回{0,1}）
    int get_x_hat() const {
        float val = to_float();  // [-1,1]
        float x_normalized = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
        return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
    }

private:
    // float -> FP16 bits（模拟 IEEE 754 half precision）
    static inline uint16_t float_to_fp16_bits(float f) {
        union { float f; uint32_t u; } f2u;
        f2u.f = f;

        uint32_t f_exp = (f2u.u >> 23) & 0xFF;
        uint32_t f_sig = f2u.u & 0x7FFFFF;
        uint32_t f_sign = (f2u.u >> 31) & 0x1;

        // 处理特殊值
        if (f_exp == 0xFF) {
            // NaN 或 Inf
            uint16_t h_exp = 0x1F;
            uint16_t h_sig = (f_sig >> 13) | (f_sig ? 0x200 : 0);
            return (f_sign << 15) | (h_exp << 10) | h_sig;
        }

        if (f_exp == 0 && f_sig == 0) {
            // Zero
            return f_sign << 15;
        }

        // 调整指数：float exponent bias 127 -> fp16 exponent bias 15
        int32_t exp = f_exp - 127 + 15;

        if (exp >= 31) {
            // Overflow -> Inf
            return (f_sign << 15) | 0x7C00;
        }

        if (exp <= 0) {
            // Denormalized or underflow
            if (exp < -10) {
                // Underflow -> Zero
                return f_sign << 15;
            }

            // Denormalized
            exp = 0;
            f_sig = (f_sig | 0x800000) >> (1 - exp);
            if (f_sig > 0x3FF) f_sig = 0x3FF;
            return (f_sign << 15) | (uint16_t)f_sig;
        }

        // Normalized number
        uint16_t h_sig = f_sig >> 13;
        return (f_sign << 15) | ((uint16_t)exp << 10) | h_sig;
    }

    // FP16 bits -> float
    static inline float fp16_bits_to_float(uint16_t h) {
        uint32_t h_exp = (h >> 10) & 0x1F;
        uint32_t h_sig = h & 0x3FF;
        uint32_t h_sign = (h >> 15) & 0x1;

        uint32_t f_exp, f_sig;

        if (h_exp == 0x1F) {
            // NaN 或 Inf
            f_exp = 0xFF;
            f_sig = (h_sig << 13) | (h_sig ? 0x400000 : 0);
        } else if (h_exp == 0) {
            // Denormalized or zero
            if (h_sig == 0) {
                f_exp = 0;
                f_sig = 0;
            } else {
                // Denormalized -> Normalized
                uint32_t sig = h_sig;
                while ((sig & 0x400) == 0) {
                    sig <<= 1;
                    h_exp--;
                }
                f_exp = h_exp + 112;  // 127 - 15
                f_sig = (sig & 0x3FF) << 13;
            }
        } else {
            // Normalized
            f_exp = h_exp + 112;  // 127 - 15
            f_sig = h_sig << 13;
        }

        union { float f; uint32_t u; } u2f;
        u2f.u = (h_sign << 31) | (f_exp << 23) | f_sig;
        return u2f.f;
    }
};

// FP16转换函数
inline float fp16_to_float(fp16_t h) {
    return h.to_float();
}

inline fp16_t float_to_fp16(float f) {
    return fp16_t(f);
}

// 自定义x_hat分段函数（DSB特征：不使用sign，使用分段函数）
// 参考项目报告3.md第232行定义：x̂_i = 1 if x_i > 0.5, else 0
inline int x_hat_fp16(fp16_t x) {
    float val = fp16_to_float(x);  // [-1,1]
    float x_normalized = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
}

// ============================================================================
// SIMD批量转换优化（参考FP8成功策略）
// ============================================================================

// 批量转换：FP16 -> FP32
inline void batch_fp16_to_float(const fp16_t* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = fp16_to_float(src[i]);
    }
}

// 批量转换：FP32 -> FP16
inline void batch_float_to_fp16(const float* src, fp16_t* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = float_to_fp16(src[i]);
    }
}

// 批量转换：FP16 -> FP32 + 归一化到[0,1]
inline void batch_fp16_to_norm(const fp16_t* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        float val = fp16_to_float(src[i]);  // [-1,1]
        dst[i] = (val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]
    }
}

// 手动实现clamp函数（兼容C++11/14）
template<typename T>
T clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// 浮点数版本clamp
inline float clamp_float(float val, float min_val, float max_val) {
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}

// FP16范围限制函数（参考GPU FP16版本）
inline float fp16_clamp(float val) {
    return (val > 0.0f) ? ((val > 1.0f) ? 1.0f : val) : ((val < -1.0f) ? -1.0f : val);
}

// 批处理数据结构（参考FP8成功策略）
struct BatchData {
    vector<fp16_t> x;             // 位置状态（FP16，模拟PyTorch float16）
    vector<fp16_t> y;             // 动量状态（FP16，模拟PyTorch float16）
    vector<float> gradient_term;   // 梯度项（float，参考FP8成功策略）
    vector<int> x_hat;            // 离散二值化变量{0,1}（DSB关键特征）
    float energy;                 // 能量值（float，避免累加误差）

    // 预分配的临时数组（参考FP8优化策略）
    vector<float> x_float;       // FP32临时数组
    vector<float> y_float;       // FP32临时数组
    vector<float> grad_float;    // FP32临时数组
    vector<float> x_norm;        // [0,1]归一化临时数组

    BatchData(int num_vars) :
        x(num_vars), y(num_vars), x_hat(num_vars),
        gradient_term(num_vars), energy(0.0f),
        x_float(num_vars), y_float(num_vars), grad_float(num_vars), x_norm(num_vars) {}
};

// DSB求解器（参考FP8成功策略，使用sign(x)）
// 关键改进：FP16量化存储 + float梯度项（避免量化误差累积）
class DSBFP16Solver {
private:
    ProblemModel model;
    vector<vector<int>> var_to_terms;  // 自旋→参与的二阶项索引
    vector<vector<int>> term_positions; // 二阶项→包含的自旋索引
    float fixed_xi;  // 固定的优化xi值（来自solve_SB6.py）

public:
    DSBFP16Solver(const ProblemModel& problem_model, float optimized_xi = 0.0f) : model(problem_model) {
        precompute_lookup_tables();

        // 设置xi值（与solve_SB6.py保持一致）
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

    // 初始化（参考GPU FP16版本，使用真正的PyTorch float16）
    void init_batches(vector<BatchData>& batches, uint64_t seed, float sigma = 0.1f) {
        mt19937 gen(seed);
        normal_distribution<float> dis_normal(0.0f, 1.0f);  // 使用float精度（匹配GPU）

        for (size_t batch = 0; batch < batches.size(); ++batch) {
            mt19937 local_gen(seed + batch);
            for (int i = 0; i < model.num_vars; ++i) {
                // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5（匹配GPU FP16第87-88行）
                float x_normalized = sigma * dis_normal(local_gen) + 0.5f;
                x_normalized = clamp_float(x_normalized, 0.0f, 1.0f);

                // 映射到FP16范围：[-1,1]，其中0.5映射到0（匹配GPU FP16第91-92行）
                float x_scaled = (x_normalized - 0.5f) * 2.0f;
                batches[batch].x[i] = float_to_fp16(clamp_float(x_scaled, -1.0f, 1.0f));  // 存储为FP16

                // y初始化为小随机值：sigma * N(0,1)（匹配GPU FP16第96-97行）
                float y_val = sigma * dis_normal(local_gen);
                batches[batch].y[i] = float_to_fp16(y_val);  // 存储为FP16

                // 初始化x_hat（匹配项目报告定义）
                batches[batch].x_hat[i] = x_hat_fp16(batches[batch].x[i]);  // {0,1}二值化
            }
        }
    }

    // 批量计算离散二值化变量（参考项目报告定义）
    // x_hat是{0,1}的二值变量（DSB关键特征）
    void batch_calc_x_hat(vector<BatchData>& batches) {
        for (auto& batch : batches) {
            for (int i = 0; i < model.num_vars; ++i) {
                batch.x_hat[i] = x_hat_fp16(batch.x[i]);  // 返回{0,1}
            }
        }
    }

    // 批量计算梯度（参考FP8成功策略）
    // 关键改进：gradient_term使用float存储（避免量化误差累积）
    void batch_calc_gradient(vector<BatchData>& batches) {
        // 确保linear_terms大小与变量数量一致
        if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
            model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
        }

        // 根据优化方向确定梯度符号
        bool is_maximize = (model.objective_sense == "maximize");
        float direction_sign = is_maximize ? 1.0f : -1.0f;

        // 先计算所有批次的离散二值化变量（DSB特征：{0,1}二值化）
        batch_calc_x_hat(batches);

        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP8优化策略）
            batch_fp16_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 初始化梯度为线性项系数
            for (int i = 0; i < model.num_vars; ++i) {
                batch.grad_float[i] = direction_sign * static_cast<float>(model.linear_terms[i]);
            }

            // 累加二次项贡献
            // DSB特征：直接使用{0,1}变量进行计算
            if (model.num_quad_terms > 0) {
                for (int t = 0; t < model.num_quad_terms; ++t) {
                    int var1 = model.quad_terms_indices[t][0];
                    int var2 = model.quad_terms_indices[t][1];
                    float J_ij = direction_sign * static_cast<float>(model.quad_terms_coeffs[t]);

                    // 直接使用{0,1}二值化变量
                    int x2_hat = batch.x_hat[var2];  // 0或1
                    int x1_hat = batch.x_hat[var1];  // 0或1

                    batch.grad_float[var1] += (J_ij / 2.0f) * static_cast<float>(x2_hat);
                    batch.grad_float[var2] += (J_ij / 2.0f) * static_cast<float>(x1_hat);
                }
            }

            // 直接存储为float（参考FP8成功策略）
            for (int i = 0; i < model.num_vars; ++i) {
                batch.gradient_term[i] = batch.grad_float[i];
            }
        }
    }

    // 更新辅助状态y（参考FP8成功策略）
    void batch_update_y(vector<BatchData>& batches, float p, float dt, float delta) {
        // 泵浦振幅项：-(1-p(t))
        const float pump_factor = -(1.0f - p);
        const float xi_val = fixed_xi;
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP8优化策略）
            batch_fp16_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_fp16_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);

            // 直接使用gradient_term（已是float，参考FP8成功策略）
            for (int i = 0; i < model.num_vars; ++i) {
                float x_val = batch.x_float[i];  // [-1,1]
                float x_normalized = (x_val + 1.0f) * 0.5f;  // [0,1]
                float pump_term = pump_factor * (x_normalized - 0.5f);
                float gradient_term = xi_val * batch.gradient_term[i];

                // DSB公式：dy = [-(1-p)*(x-0.5) + ξ*∇f] * dt
                float y_current = batch.y_float[i];
                float dy = (pump_term + gradient_term) * dt_val;
                batch.y_float[i] = y_current + dy;
            }

            // 存储回FP16（批量转换）
            batch_float_to_fp16(batch.y_float.data(), batch.y.data(), model.num_vars);
        }
    }

    // 更新主状态x（参考FP8成功策略，FP16量化存储）
    void batch_update_x(vector<BatchData>& batches, float dt, float delta) {
        const float dt_val = dt;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP8优化策略）
            batch_fp16_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_fp16_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);

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
                __m128 x_fp16_val = _mm_sub_ps(x_new, _mm_set1_ps(0.5f));  // x - 0.5
                x_fp16_val = _mm_mul_ps(x_fp16_val, _mm_set1_ps(2.0f));  // (x-0.5)*2
                x_fp16_val = _mm_max_ps(_mm_set1_ps(-1.0f), _mm_min_ps(x_fp16_val, _mm_set1_ps(1.0f)));

                // 存储回临时数组
                _mm_storeu_ps(&batch.x_float[i], x_fp16_val);
            }
#endif
            // 处理剩余变量
            for (; i < model.num_vars; ++i) {
                float x_norm = (batch.x_float[i] + 1.0f) * 0.5f;  // [0,1]
                float x_new = clamp_float(x_norm + batch.y_float[i] * dt_val, 0.0f, 1.0f);
                batch.x_float[i] = clamp_float(x_new * 2.0f - 1.0f, -1.0f, 1.0f);
            }

            // 存储回FP16（批量转换）
            batch_float_to_fp16(batch.x_float.data(), batch.x.data(), model.num_vars);
        }
    }

    // 边界约束（参考FP16版本逻辑）
    void batch_apply_constraint(vector<BatchData>& batches) {
        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP8优化策略）
            batch_fp16_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);
            batch_fp16_to_float(batch.y.data(), batch.y_float.data(), model.num_vars);

            // 处理边界约束（参考FP16版本第276-294行的简洁逻辑）
            for (int i = 0; i < model.num_vars; ++i) {
                // 将float值转换到[0,1]范围（参考FP16版本第277-279行）
                float x_val = batch.x_float[i];  // [-1,1]
                float x_normalized = (x_val + 1.0f) * 0.5f;  // [-1,1] -> [0,1]

                // 检查是否超出边界（参考FP16版本第282-294行）
                float distance_from_center = fabsf(x_normalized - 0.5f);

                if (distance_from_center > 0.5f) {
                    // 超出边界，投影到边界{0,1}
                    float projected_value = (x_normalized > 0.5f) ? 1.0f : 0.0f;
                    batch.x_float[i] = (projected_value * 2.0f) - 1.0f;  // [0,1] -> [-1,1]
                    batch.y_float[i] = 0.0f;  // 清零动量
                }
            }

            // 存储回FP16（批量转换）
            batch_float_to_fp16(batch.x_float.data(), batch.x.data(), model.num_vars);
            batch_float_to_fp16(batch.y_float.data(), batch.y.data(), model.num_vars);
        }
    }

    // 计算单个批次的目标函数值（参考FP8成功策略）
    // DSB特征：直接使用{0,1}变量进行能量计算
    float calc_batch_energy(const BatchData& batch) {
        float scaled_energy = 0.0f;
        bool is_linear_problem = (model.num_quad_terms == 0);

        // 二次项贡献：0.5 * ∑_{i,j} J_{ij}x̂_i x̂_j
        if (model.num_quad_terms > 0) {
            for (int t = 0; t < model.num_quad_terms; ++t) {
                int var1 = model.quad_terms_indices[t][0];
                int var2 = model.quad_terms_indices[t][1];
                float J_ij = static_cast<float>(model.quad_terms_coeffs[t]);

                // 直接使用{0,1}二值化变量
                int x1_hat = batch.x_hat[var1];  // 0或1
                int x2_hat = batch.x_hat[var2];  // 0或1

                scaled_energy += 0.5f * J_ij * static_cast<float>(x1_hat) * static_cast<float>(x2_hat);
            }
        }

        // 线性项贡献：∑_i h_i x̂_i
        for (int i = 0; i < model.num_vars; ++i) {
            // 直接使用{0,1}二值化变量
            int x_hat = batch.x_hat[i];  // 0或1
            scaled_energy += static_cast<float>(model.linear_terms[i]) * static_cast<float>(x_hat);
        }

        // 缩放处理
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

    // 找到最优批次（根据优化方向选择最优解）
    // 使用sign(x)进行能量评估
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

    // 核心求解函数（基于FP8成功策略）
    vector<fp16_t> solve(int max_iters = 1000, int batch_size = 100, float sigma = 0.1f, vector<BatchData>* out_batches = nullptr) {
        float dt = 0.1f;
        float delta = 0.9f;

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << model.scaling_factor << endl;
        cout << "  - DSB Algorithm (FP16, PyTorch float16 - CPU, IEEE 754)" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using xi=" << fixed << setprecision(6) << fixed_xi << ", sigma=" << sigma << ", dt=" << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - Key feature: x̂(x) -> {0,1} discretization (no sign), PyTorch FP16 format" << endl;
        cout << "  - FP16 Format: PyTorch float16 (IEEE 754 half precision, 1 sign + 5 exp + 10 mantissa)" << endl;
        cout << "  - Storage: FP16 for x,y; Float for gradient_term (matches GPU FP16)" << endl;

        // 初始化批处理数据
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
            float p = static_cast<float>(iter) / static_cast<float>(max_iters - 1);

            // 每次迭代都计算当前时间（用于准确记录收敛时间）
            auto current_time = high_resolution_clock::now();
            float cumulative_time = static_cast<float>(duration_cast<microseconds>(current_time - start).count()) / 1000.0f;

            // DSB动力学更新序列
            // 1. 计算梯度（使用sign(x)而不是连续变量x）
            batch_calc_gradient(batches);

            // 2. 更新动量：dy/dt = -(1-p(t))*sgn(x) + ξ*∇f(x)
            batch_update_y(batches, p, dt, delta);

            // 3. 更新位置：dx/dt = y
            batch_update_x(batches, dt, delta);

            // 4. 边界条件处理
            batch_apply_constraint(batches);

            // 5. 关键修复：在每次迭代结束时更新x_hat（匹配项目报告定义）
            // 因为batch_update_x和batch_apply_constraint改变了x，需要同步更新x_hat
            batch_calc_x_hat(batches);

            // 只在最后一次迭代记录收敛数据和输出能量
            if (iter == max_iters - 1) {
                // 计算当前最佳目标函数值（x_hat已在迭代结束时更新）
                bool is_maximize = (model.objective_sense == "maximize");
                float current_best = calc_batch_energy(batches[0]);

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
        cout << "C++_TIME: Solving completed, total time: " << fixed << setprecision(3) << duration << " ms" << endl;

        // 生成最优解（x_hat已在最后一次迭代结束时更新）
        int best_idx = find_best_batch(batches);
        vector<fp16_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            solution[i] = batches[best_idx].x[i];
        }

        cout << "BEST: Optimal objective value: " << fixed << setprecision(4) << batches[best_idx].energy << " (FP16 precision)" << endl;

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

// 主函数：DSB算法求解器
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
    cout << "    DSB_CPU_fp16 Solver (PyTorch FP16, IEEE 754) " << endl;
    cout << "  - File path: " << qplib_file_path << endl;
    cout << "  - Algorithm: DSB (Discrete Simulated Bifurcation)" << endl;
    cout << "  - Key feature: True PyTorch float16 (IEEE 754 half precision)" << endl;
    cout << "  - FP16 Format: 1 sign bit + 5 exponent bits + 10 mantissa bits" << endl;
    cout << "  - Range: [-65504, 65504], Epsilon: 0.0009765625" << endl;
    cout << "  - Storage: FP16 for x,y; Float for gradient_term (matches GPU FP16)" << endl;
    cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
    cout << "  - Energy method: Discrete {0,1} values using x̂(x) function (no sign)" << endl;
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
        DSBFP16Solver solver(model, optimized_xi);
        vector<BatchData> batches;
        vector<fp16_t> solution = solver.solve(1000, 100, 0.1f, &batches);

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
            // 直接使用{0,1}二值化变量
            int binary_val = batches[best_idx].x_hat[i];  // 0或1
            count_ones += binary_val;

            if ((i + 1) % 17 == 0 || i == model.num_vars - 1) {
                cout << binary_val << endl;
            } else {
                cout << binary_val << ", ";
            }
        }

        int count_zeros = model.num_vars - count_ones;
        float density = static_cast<float>(count_ones) / static_cast<float>(model.num_vars) * 100.0f;

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
        cout << "  Storage: True PyTorch FP16 (IEEE 754) for x,y; Float for gradient_term; x̂ function for {0,1}" << endl;
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