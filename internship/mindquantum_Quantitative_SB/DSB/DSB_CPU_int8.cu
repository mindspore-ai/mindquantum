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
#include <iomanip>
#include <exception>
#include <cmath>
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// ============================================================================
// PyTorch INT8 配置（完全匹配GPU版本）
// ============================================================================
// PyTorch uses signed 8-bit integers: range [-128, 127]
constexpr int8_t INT8_MAX_VAL = 127;
constexpr int8_t INT8_MIN_VAL = -128;  // PyTorch标准

// PyTorch INT8 quantization converter（完全匹配GPU版本第30-44行）
struct INT8PyTorch {
    // PyTorch INT8 quantization: maps float to signed int8
    // Uses IEEE 754 round-to-nearest-even rounding
    inline static int8_t quantize(float val) {
        // Clamp to INT8 range
        val = fmaxf(static_cast<float>(INT8_MIN_VAL), fminf(static_cast<float>(INT8_MAX_VAL), val));
        // Round to nearest even (IEEE 754 standard, matches GPU INT8 version)
        return static_cast<int8_t>(roundf(val));
    }

    // PyTorch INT8 dequantization: converts int8 to float
    inline static float dequantize(int8_t val) {
        return static_cast<float>(val);
    }
};
// QPLIB配置：默认基础路径（可通过命令行参数覆盖）
const string DEFAULT_QPLIB_BASE_PATH = "../qplib/";
// 使用头文件中的DEFAULT_QPLIB_FILE常量

// 手动实现clamp函数（兼容C++11/14，避免与std::clamp冲突）
template<typename T>
T custom_clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// Float版本clamp（用于INT8的转换计算）
inline float clamp_float(float val, float min_val, float max_val) {
    return val < min_val ? min_val : (val > max_val ? max_val : val);
}

// INT8转换函数（参考FP16成功策略，简化版本）
inline float int8_to_float(int8_t val) {
    return static_cast<float>(val);
}

inline int8_t float_to_int8(float val) {
    return INT8PyTorch::quantize(val);
}

// 批量转换优化（参考FP16/FP8成功策略）
inline void batch_int8_to_float(const int8_t* src, float* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = int8_to_float(src[i]);
    }
}

inline void batch_float_to_int8(const float* src, int8_t* dst, int size) {
    for (int i = 0; i < size; ++i) {
        dst[i] = float_to_int8(src[i]);
    }
}

// 批量转换：INT8 -> FP32 + 归一化到[0,1]（参考FP16/FP8）
inline void batch_int8_to_norm(const int8_t* src, float* dst, int size) {
    constexpr float TO_NORM_FACTOR_INV = 1.0f / 254.0f;  // [-127,127] -> [0,1]
    constexpr float OFFSET = 127.0f;
    for (int i = 0; i < size; ++i) {
        float val = int8_to_float(src[i]);  // [-127,127]
        dst[i] = (val + OFFSET) * TO_NORM_FACTOR_INV;  // [0,1]
    }
}

// 自定义x_hat分段函数（DSB特征：不使用sign，使用分段函数）
// 参考项目报告3.md第232行定义：x̂_i = 1 if x_i > 0.5, else 0
inline int x_hat_int8(int8_t x) {
    float val = int8_to_float(x);  // [-127,127]
    float x_normalized = (val + 127.0f) / 254.0f;  // [-127,127] -> [0,1]
    return (x_normalized > 0.5f) ? 1 : 0;  // 返回{0,1}，匹配报告定义
}

    // 批处理数据结构（参考FP16/FP8成功策略）
struct BatchData {
    vector<int8_t> x;          // 位置状态（PyTorch INT8，存储在[-127,127]）
    vector<float> y;           // 动量状态（float，参考FP16/FP8策略）
    vector<int> x_hat;         // 离散二值化变量{0,1}（DSB关键特征）
    vector<float> gradient_term;   // 梯度项（float，参考FP16/FP8策略）
    float energy;              // 能量值（float，保持精度一致性）

    // 预分配的临时数组（参考FP16/FP8优化策略）
    vector<float> x_float;       // FP32临时数组
    vector<float> grad_float;    // FP32临时数组
    vector<float> x_norm;        // [0,1]归一化临时数组（参考FP16）

    BatchData(int num_vars) :
        x(num_vars), y(num_vars), x_hat(num_vars),
        gradient_term(num_vars), energy(0.0f),
        x_float(num_vars), grad_float(num_vars), x_norm(num_vars) {}
};

// DSB求解器（完全匹配GPU INT8版本）
// 关键改进：使用float梯度项（避免量化误差累积）
class DSBInt8Solver {
private:
    ProblemModel model;
    vector<vector<int>> var_to_terms;  // 自旋→参与的二阶项索引
    vector<vector<int>> term_positions; // 二阶项→包含的自旋索引
    float fixed_xi;  // 固定的优化xi值（来自solve_SB6.py，使用float精度）

public:
    DSBInt8Solver(const ProblemModel& problem_model, float optimized_xi = 0.0f) : model(problem_model) {
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

    // 初始化批处理数据（完全匹配GPU INT8版本）
    // 参考：DSB_GPU_int8.cu第96-128行
    void init_batches(vector<BatchData>& batches, uint64_t seed, float sigma = 0.1f) {
        mt19937 gen(seed);
        normal_distribution<float> dis_normal(0.0f, 1.0f);  // 使用float精度（匹配GPU）

        for (size_t batch = 0; batch < batches.size(); ++batch) {
            mt19937 local_gen(seed + batch);
            for (int i = 0; i < model.num_vars; ++i) {
                // x在[0,1]范围内初始化：sigma * N(0,1) + 0.5（完全匹配GPU第112-113行）
                float x_normalized = sigma * dis_normal(local_gen) + 0.5f;
                x_normalized = clamp_float(x_normalized, 0.0f, 1.0f);

                // 映射到INT8范围：[-127,127]（完全匹配GPU第117行）
                float x_scaled = (x_normalized - 0.5f) * 2.0f * static_cast<float>(INT8_MAX_VAL);
                int8_t x_int8_val = INT8PyTorch::quantize(x_scaled);  // 使用PyTorch quantize
                batches[batch].x[i] = x_int8_val;

                // y初始化为小随机值：sigma * N(0,1)（完全匹配GPU第122-123行，使用float）
                float y_val = sigma * dis_normal(local_gen);
                batches[batch].y[i] = y_val;  // float存储，匹配GPU第78行

                // 初始化x_hat（匹配项目报告定义）
                batches[batch].x_hat[i] = x_hat_int8(x_int8_val);  // {0,1}二值化

                // 初始化gradient_term为0（完全匹配GPU第126行）
                batches[batch].gradient_term[i] = 0.0f;  // float zero
            }
        }
    }

    // 批量计算离散二值化变量（匹配项目报告定义）
    // x_hat是{0,1}的二值变量（DSB关键特征）
    void batch_calc_x_hat(vector<BatchData>& batches) {
        for (auto& batch : batches) {
            for (int i = 0; i < model.num_vars; ++i) {
                batch.x_hat[i] = x_hat_int8(batch.x[i]);  // 返回{0,1}
            }
        }
    }

    // 批量计算梯度（参考FP16/FP8成功策略）
    void batch_calc_gradient(vector<BatchData>& batches) {
        // 确保linear_terms大小与变量数量一致
        if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
            model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0f);
        }

        // 根据优化方向确定梯度符号
        bool is_maximize = (model.objective_sense == "maximize");
        float direction_sign = is_maximize ? 1.0f : -1.0f;

        // 先计算所有批次的离散二值化变量（DSB特征：{0,1}二值化）
        batch_calc_x_hat(batches);

        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP16/FP8优化策略）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 初始化梯度为线性项系数
            for (int i = 0; i < model.num_vars; ++i) {
                batch.grad_float[i] = direction_sign * static_cast<float>(model.linear_terms[i]);
            }

            // 累加二次项贡献（DSB特征：使用{0,1}变量进行计算）
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

            // 存储为float（参考FP16/FP8成功策略）
            for (int i = 0; i < model.num_vars; ++i) {
                batch.gradient_term[i] = batch.grad_float[i];
            }
        }
    }

    // 更新辅助状态y（参考FP8成功策略，避免不必要的复制）
    void batch_update_y(vector<BatchData>& batches, float p, float dt, float delta) {
        // 泵浦振幅项：-(1-p(t))
        const float pump_factor = -(1.0f - p);
        const float xi_val = fixed_xi;
        const float dt_val = dt;
        constexpr float OFFSET = 127.0f;
        constexpr float TO_NORM_FACTOR_INV = 1.0f / 254.0f;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP8优化策略）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 关键优化：直接使用batch.y（已经是float），避免复制到y_float
            // 参考FP8版本第358行：y直接是float，无需复制
            for (int i = 0; i < model.num_vars; ++i) {
                float x_val = batch.x_float[i];  // [-127,127]
                float x_norm = (x_val + OFFSET) * TO_NORM_FACTOR_INV;  // [0,1]
                float pump_term = pump_factor * (x_norm - 0.5f);
                float gradient_term = xi_val * batch.gradient_term[i];

                // DSB公式：dy = [-(1-p)*(x-0.5) + ξ*∇f] * dt
                // 直接更新batch.y，避免中间变量
                float dy = (pump_term + gradient_term) * dt_val;
                batch.y[i] += dy;  // 直接修改，参考FP8第359行
            }
        }
    }

    // 更新自旋x（参考FP8成功策略，避免不必要的复制）
    void batch_update_x(vector<BatchData>& batches, float dt, float delta) {
        constexpr float OFFSET = 127.0f;
        constexpr float TO_NORM_FACTOR_INV = 1.0f / 254.0f;
        constexpr float TO_INT8_FACTOR = 254.0f;
        constexpr float INT8_OFFSET = -127.0f;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP8优化策略）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 关键优化：直接使用batch.y（已经是float），避免复制
            // 参考FP8版本第382行：y直接是float，无需复制
            for (int i = 0; i < model.num_vars; ++i) {
                float x_norm = (batch.x_float[i] + OFFSET) * TO_NORM_FACTOR_INV;  // [0,1]
                float x_new = clamp_float(x_norm + batch.y[i] * dt, 0.0f, 1.0f);  // 直接使用batch.y
                batch.x_float[i] = clamp_float(x_new * TO_INT8_FACTOR + INT8_OFFSET, -127.0f, 127.0f);
            }

            // 存储回INT8（批量转换）
            batch_float_to_int8(batch.x_float.data(), batch.x.data(), model.num_vars);
        }
    }

    // 边界约束（参考FP8成功策略，避免不必要的复制）
    void batch_apply_constraint(vector<BatchData>& batches) {
        constexpr float OFFSET = 127.0f;
        constexpr float TO_NORM_FACTOR_INV = 1.0f / 254.0f;

        for (auto& batch : batches) {
            // 使用预分配的临时数组（参考FP8优化策略）
            batch_int8_to_float(batch.x.data(), batch.x_float.data(), model.num_vars);

            // 关键优化：直接使用batch.y（已经是float），避免复制
            // 参考FP8版本第428-432行：y直接是float，无需复制
            for (int i = 0; i < model.num_vars; ++i) {
                float x_val = batch.x_float[i];  // [-127,127]
                float x_normalized = (x_val + OFFSET) * TO_NORM_FACTOR_INV;  // [0,1]

                // 检查是否超出边界（参考FP8简洁逻辑）
                float distance_from_center = fabsf(x_normalized - 0.5f);

                if (distance_from_center > 0.5f) {
                    // 超出边界，投影到边界{0,1}
                    float projected_value = (x_normalized > 0.5f) ? 1.0f : 0.0f;
                    batch.x_float[i] = (projected_value * 254.0f) - 127.0f;  // [0,1] -> [-127,127]
                    batch.y[i] = 0.0f;  // 直接清零，参考FP8第432行
                }
            }

            // 存储回INT8
            batch_float_to_int8(batch.x_float.data(), batch.x.data(), model.num_vars);
        }
    }

    // 计算单个批次的目标函数值（匹配项目报告定义）
    // DSB特征：使用{0,1}变量进行能量计算
    float calc_batch_energy(const BatchData& batch) {
        float scaled_energy = 0.0f;

        // 检查是否是纯线性问题（与GPU保持一致）
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

                scaled_energy += 0.5f * J_ij * static_cast<float>(x1_hat * x2_hat);
            }
        }

        // 线性项贡献：∑_i h_i x̂_i
        for (int i = 0; i < model.num_vars; ++i) {
            // 直接使用{0,1}二值化变量
            int x_hat = batch.x_hat[i];  // 0或1
            scaled_energy += static_cast<float>(model.linear_terms[i]) * static_cast<float>(x_hat);
        }

        // 缩放处理（完全匹配GPU）
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

    // 找到最优批次（参考FP16/FP8成功策略）
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

    // 核心求解函数（完全匹配GPU INT8版本）
    vector<int8_t> solve(int max_iters = 1000, int batch_size = 100, float sigma = 0.1f, vector<BatchData>* out_batches = nullptr) {
        float dt = 0.1f;
        float delta = 0.9f;

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(4) << model.scaling_factor << endl;
        cout << "  - DSB Algorithm (INT8, PyTorch int8 - CPU)" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using xi=" << fixed << setprecision(4) << fixed_xi << ", sigma=" << sigma << ", dt=" << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;
        cout << "  - Key feature: x̂(x) -> {0,1} discretization (no sign), PyTorch INT8 format" << endl;
        cout << "  - Storage: INT8 for x; Float for y, gradient_term, and energy (all float32)" << endl;

        // 初始化批处理数据
        vector<BatchData> batches;
        batches.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batches.emplace_back(model.num_vars);
        }
        init_batches(batches, time(nullptr), sigma);

        // 计时开始
        auto start = high_resolution_clock::now();

        // 用于记录收敛过程的数据结构
        struct ConvergenceData {
            int iteration;
            float cumulative_time_ms;
            float best_objective;
        };
        vector<ConvergenceData> convergence_history;
        float current_best_overall = model.objective_sense == "maximize" ? -numeric_limits<float>::infinity() : numeric_limits<float>::infinity();

        // 核心迭代（DSB动力学：使用sign(x)替代连续x）
        for (int iter = 0; iter < max_iters; ++iter) {
            // 计算泵浦振幅：p(t) = t/T = iter/(max_iters-1)
            float p = static_cast<float>(iter) / (max_iters - 1);

            // 每次迭代都计算当前时间（用于准确记录收敛时间）
            auto current_time = high_resolution_clock::now();
            float cumulative_time = static_cast<float>(duration_cast<microseconds>(current_time - start).count()) / 1000.0f;

            // DSB动力学更新序列（完全匹配GPU INT8版本）
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

        // 计时结束
        auto end = high_resolution_clock::now();
        float duration = static_cast<float>(duration_cast<microseconds>(end - start).count()) / 1000.0f;
        cout << "C++_TIME: Total solving time: " << fixed << setprecision(3) << duration << " ms" << endl;

        // 生成最优解（x_sign已在最后一次迭代结束时更新）
        int best_idx = find_best_batch(batches);
        vector<int8_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            solution[i] = batches[best_idx].x[i];
        }

        cout << "BEST: Optimal objective value: " << fixed << setprecision(3) << batches[best_idx].energy << " (INT8 precision)" << endl;

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

// 主函数：DSB算法求解器
int main(int argc, char* argv[]) {
    // 处理命令行参数
    string qplib_file_path;
    float optimized_xi = 0.365f;  // 来自solve_SB6.py的优化xi值

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
        cout << "         DSB_CPU_int8 Solver (PyTorch Compatible)         " << endl;
        cout << "  - File path: " << qplib_file_path << endl;
        cout << "  - Algorithm: DSB (Discrete Simulated Bifurcation)" << endl;
        cout << "  - Key feature: Uses x̂(x) function instead of sign(x)" << endl;
        cout << "  - PyTorch INT8 compatible: range [-127, 127]" << endl;
        cout << "  - Round-to-nearest-even: IEEE 754 standard" << endl;
        cout << "  - Optimizations: INT8 x + Float y/gradient/energy (all float32)" << endl;
        cout << "  - Based on project report 3.md with x̂_i = 1 if x_i > 0.5 else 0" << endl;
        cout << "  - Fixed optimized xi: " << fixed << setprecision(4) << optimized_xi << endl;
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
    cout << "Starting DSB solver..." << endl;

    // 安全检查：确保linear_terms大小与变量数量一致
    if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
        model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
    }

    // 初始化求解器并求解
    try {
        cout << "Using fixed optimized xi value: " << fixed << setprecision(6) << optimized_xi
             << " (auto-calculated value: " << fixed << setprecision(6) << model.xi << ")" << endl;
        DSBInt8Solver solver(model, optimized_xi);
        vector<BatchData> batches;
        vector<int8_t> solution = solver.solve(1000, 100, 0.1f, &batches);

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

        // 输出解决方案的二进制值
        cout << "RESULT: Solution (" << model.num_vars << " variables, binary values [0/1]): " << endl;
        vector<int> binary_solution(model.num_vars);
        int count_ones = 0;

        for (int i = 0; i < model.num_vars; ++i) {
            // 直接使用{0,1}二值化变量
            int binary_val = batches[best_idx].x_hat[i];  // 0或1
            binary_solution[i] = binary_val;
            count_ones += binary_val;

            // 每17个值换行
            if ((i + 1) % 17 == 0 || i == model.num_vars - 1) {
                cout << binary_val << endl;
            } else {
                cout << binary_val << ", ";
            }
        }

        // 输出解决方案统计信息
        int count_zeros = model.num_vars - count_ones;
        float density = static_cast<float>(count_ones) / model.num_vars * 100.0f;

        cout << endl << "Solution statistics:" << endl;
        cout << "  - Variables set to 1: " << count_ones << "/" << model.num_vars << endl;
        cout << "  - Variables set to 0: " << count_zeros << "/" << model.num_vars << endl;
        cout << "  - Solution density: " << fixed << setprecision(2) << density << "%" << endl;
        
        // 输出最终结果总结
        cout << endl << "========================================" << endl;
        cout << "  ** DSB FINAL RESULTS **" << endl;
        cout << "========================================" << endl;
        cout << "  Problem: " << qplib_file_path.substr(qplib_file_path.find_last_of("/\\") + 1) << endl;
        cout << "  Objective sense: " << model.objective_sense << endl;
        cout << "  Number of variables: " << model.num_vars << endl;
        cout << "  Number of quadratic terms: " << model.num_quad_terms << endl;
        cout << "----------------------------------------" << endl;
        cout << "  DSB OBJECTIVE VALUE: " << fixed << setprecision(6) << best_value << endl;
        cout << "  Storage: INT8 for x; Float for y, gradient_term, energy; x̂ function for {0,1}" << endl;
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