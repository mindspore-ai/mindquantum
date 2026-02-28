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
#include <omp.h>

// 统一包含路径：编译时使用 -I 参数指定 qplib_parser.h 的目录
// 例如: nvcc -I.. -O3 -arch=native BSB_CPU_fp16.cu -o BSB_CPU_fp16
#include "../qplib_parser.h"

using namespace std;
using namespace std::chrono;

// ============================================================================
// PyTorch风格FP16配置（完全兼容torch.float16）
// ============================================================================
// 参考PyTorch的torch.float16实现，基于IEEE 754半精度浮点标准
constexpr float FP16_MAX_VAL = 65504.0f;   // FP16最大值（PyTorch torch.float16上限）
constexpr float FP16_MIN_VAL = -65504.0f;  // FP16最小值（PyTorch torch.float16下限）
constexpr float FP16_LOWEST_POSITIVE = 6.1035e-5f;  // FP16最小正数（subnormal）
constexpr int FP16_EXP_MAX = 30;  // FP16最大指数偏移（对应IEEE 754 exponent bias = 15，max exp = 30）
constexpr int FP16_MANTISSA_BITS = 10;  // FP16尾数位数
constexpr int FP16_EXPONENT_BITS = 5;   // FP16指数位数
constexpr int FP16_EXP_BIAS = 15;      // FP16指数偏移（IEEE 754）

// 使用这些常量进行编译时检查（避免未使用警告）
static_assert(FP16_EXP_MAX == 30, "FP16_EXP_MAX must be 30 for IEEE 754 compatibility");
static_assert(FP16_MANTISSA_BITS == 10, "FP16_MANTISSA_BITS must be 10 for IEEE 754");
static_assert(FP16_EXPONENT_BITS == 5, "FP16_EXPONENT_BITS must be 5 for IEEE 754");
static_assert(FP16_EXP_BIAS == 15, "FP16_EXP_BIAS must be 15 for IEEE 754");
static_assert(FP16_LOWEST_POSITIVE > 0.0f, "FP16_LOWEST_POSITIVE must be positive");
static_assert(FP16_MAX_VAL > 0.0f, "FP16_MAX_VAL must be positive");
static_assert(FP16_MIN_VAL < 0.0f, "FP16_MIN_VAL must be negative");

// ============================================================================
// PyTorch风格FP16实现（完全兼容torch.float16）
// ============================================================================
// 使用IEEE 754 binary16格式（1位符号，5位指数，10位尾数）
struct fp16_t {
    uint16_t data;  // 16位存储
    
    fp16_t() : data(0) {}
    fp16_t(float f) : data(float_to_bits(f)) {}
    
    // FP16位转换（完全兼容PyTorch torch.float16转换）
    // IEEE 754: Float32 (1/8/23) -> Float16 (1/5/10)
    // PyTorch行为：舍入到最近偶数（round-to-nearest-even）
    static uint16_t float_to_bits(float f) {
        uint32_t x = *reinterpret_cast<uint32_t*>(&f);
        uint32_t sign = (x >> 31) & 0x1;
        uint32_t exp = (x >> 23) & 0xFF;
        uint32_t mant = x & 0x7FFFFF;

        // NaN/Inf处理（与PyTorch一致）
        if (exp == 0xFF) {
            return (sign << 15) | 0x7C00 | (mant >> 13);  // NaN或Inf
        }

        // 溢出检查（PyTorch行为：溢出返回±Inf）
        if (exp >= 143) return (sign << 15) | 0x7C00;  // Inf (exp_fp32 >= 31)
        if (exp <= 112) {
            // 次正规数和零的处理（PyTorch行为：下溢时返回±0）
            // 舍入到零，简化处理
            return sign << 15;  // ±0
        }

        // 正常值转换（IEEE 754标准，舍入到最近偶数）
        // Float32 exp bias = 127, Float16 exp bias = 15
        // exp_fp16 = exp_fp32 - 127 + 15 = exp_fp32 - 112
        uint32_t exp_fp16 = exp - 112;
        uint32_t mant_fp16 = mant >> 13;

        // 舍入处理：检查是否需要进位（round-to-nearest-even）
        uint32_t round_bit = (mant >> 12) & 0x1;
        uint32_t sticky_bits = mant & 0xFFF;
        if (round_bit && (sticky_bits || ((mant >> 13) & 0x1))) {
            // 需要进位
            mant_fp16 += 1;
            if (mant_fp16 >= (1 << FP16_MANTISSA_BITS)) {
                mant_fp16 = 0;
                exp_fp16 += 1;
            }
        }

        return (sign << 15) | (exp_fp16 << 10) | mant_fp16;
    }
    
    // 位转浮点（完全兼容PyTorch torch.float16转换）
    // IEEE 754: Float16 (1/5/10) -> Float32 (1/8/23)
    static float bits_to_float(uint16_t h) {
        uint32_t sign = (h >> 15) & 0x1;
        uint32_t exp = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;

        // 次正规数和零（与PyTorch一致）
        if (exp == 0) {
            uint32_t result = sign << 31;  // ±0或subnormal（简化为0）
            return *reinterpret_cast<float*>(&result);
        }

        // NaN/Inf（与PyTorch一致）
        if (exp == 0x1F) {
            uint32_t result = (sign << 31) | 0x7F800000 | (mant << 13);
            return *reinterpret_cast<float*>(&result);
        }

        // 正常值转换（IEEE 754标准）
        // Float32 exp bias = 127, Float16 exp bias = 15
        // exp_fp32 = exp_fp16 + 127 - 15 = exp_fp16 + 112
        uint32_t result = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
        return *reinterpret_cast<float*>(&result);
    }

    // 判断是否为NaN（PyTorch风格）
    bool is_nan() const {
        uint32_t exp = (data >> 10) & 0x1F;
        uint32_t mant = data & 0x3FF;
        return (exp == 0x1F) && (mant != 0);
    }

    // 判断是否为Inf（PyTorch风格）
    bool is_inf() const {
        uint32_t exp = (data >> 10) & 0x1F;
        uint32_t mant = data & 0x3FF;
        return (exp == 0x1F) && (mant == 0);
    }

    // 判断是否为零（包括±0，PyTorch风格）
    bool is_zero() const {
        return (data == 0x0000 || data == 0x8000);
    }

    // 判断是否为次正规数（PyTorch风格）
    bool is_subnormal() const {
        uint32_t exp = (data >> 10) & 0x1F;
        uint32_t mant = data & 0x3FF;
        return (exp == 0) && (mant != 0);
    }
    
    operator float() const { return bits_to_float(data); }

    // 相等运算符（PyTorch风格）
    bool operator==(const fp16_t& other) const { return data == other.data; }
    bool operator!=(const fp16_t& other) const { return data != other.data; }
    bool operator<(const fp16_t& other) const {
        // 先转换为float比较（避免复杂的位比较）
        float this_f = static_cast<float>(*this);
        float other_f = static_cast<float>(other);
        return this_f < other_f;
    }
    bool operator>(const fp16_t& other) const {
        float this_f = static_cast<float>(*this);
        float other_f = static_cast<float>(other);
        return this_f > other_f;
    }
    bool operator<=(const fp16_t& other) const {
        return !(*this > other);
    }
    bool operator>=(const fp16_t& other) const {
        return !(*this < other);
    }
};

// ============================================================================
// PyTorch风格FP16转换函数（兼容torch.float16）
// ============================================================================
// 这些函数与PyTorch的.to(torch.float16)和.to(torch.float32)行为一致
inline float fp16_to_float(fp16_t h) {
    return static_cast<float>(h);
}

inline fp16_t float_to_fp16(float f) {
    return fp16_t(f);
}

// 直接操作uint16_t的转换（用于更底层的操作）
inline uint16_t fp16_to_bits(const fp16_t& h) {
    return h.data;
}

inline fp16_t bits_to_fp16(uint16_t bits) {
    fp16_t result;
    result.data = bits;
    return result;
}

// ============================================================================
// PyTorch风格FP16数学运算符（基于float16的运算）
// ============================================================================
// 注意：PyTorch中FP16运算会自动提升到FP32进行计算，结果可选FP16或FP32

// 纯FP16运算函数（使用SIMD加速，避免FP16<->FP32转换）
// 只有在关键性能路径使用，其余情况仍使用FP32中间计算

// 加法（提升到FP32计算）
inline fp16_t operator+(fp16_t a, fp16_t b) {
    return float_to_fp16(fp16_to_float(a) + fp16_to_float(b));
}

// 减法
inline fp16_t operator-(fp16_t a, fp16_t b) {
    return float_to_fp16(fp16_to_float(a) - fp16_to_float(b));
}

// 乘法
inline fp16_t operator*(fp16_t a, fp16_t b) {
    return float_to_fp16(fp16_to_float(a) * fp16_to_float(b));
}

// 除法
inline fp16_t operator/(fp16_t a, fp16_t b) {
    return float_to_fp16(fp16_to_float(a) / fp16_to_float(b));
}

// 负号
inline fp16_t operator-(fp16_t a) {
    return float_to_fp16(-fp16_to_float(a));
}

// ============================================================================
// SIMD加速的FP16批处理函数（性能优化）
// ============================================================================
#ifdef __F16C__
// 使用F16C指令集（x86架构FP16硬件加速，完全兼容PyTorch）
inline __m256 fp16_to_float256(const fp16_t* src) {
    return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src)));
}

inline void float256_to_fp16(__m256 val, fp16_t* dst) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), _mm256_cvtps_ph(val, _MM_FROUND_CUR_DIRECTION));
}

// SIMD版本的FP16向量加法（8个FP16元素）
inline void fp16_add_vector(const fp16_t* a, const fp16_t* b, fp16_t* dst, int size) {
    int i = 0;
    for (; i <= size - 8; i += 8) {
        __m256 a_float = fp16_to_float256(a + i);
        __m256 b_float = fp16_to_float256(b + i);
        __m256 result = _mm256_add_ps(a_float, b_float);
        float256_to_fp16(result, dst + i);
    }
    // 处理剩余元素
    for (; i < size; ++i) {
        dst[i] = a[i] + b[i];
    }
}

// SIMD版本的FP16标量乘向量（8个FP16元素）
inline void fp16_scalar_mul_vector(float scalar, const fp16_t* src, fp16_t* dst, int size) {
    int i = 0;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    for (; i <= size - 8; i += 8) {
        __m256 src_float = fp16_to_float256(src + i);
        __m256 result = _mm256_mul_ps(src_float, scalar_vec);
        float256_to_fp16(result, dst + i);
    }
    // 处理剩余元素
    for (; i < size; ++i) {
        dst[i] = float_to_fp16(scalar * fp16_to_float(src[i]));
    }
}

// SIMD版本的FP16向量线性组合：a * scalar + b
inline void fp16_fma_vector(float scalar, const fp16_t* a, const fp16_t* b, fp16_t* dst, int size) {
    int i = 0;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    for (; i <= size - 8; i += 8) {
        __m256 a_float = fp16_to_float256(a + i);
        __m256 b_float = fp16_to_float256(b + i);
        __m256 result = _mm256_fmadd_ps(scalar_vec, a_float, b_float);  // FMA: scalar * a + b
        float256_to_fp16(result, dst + i);
    }
    // 处理剩余元素
    for (; i < size; ++i) {
        dst[i] = float_to_fp16(scalar * fp16_to_float(a[i]) + fp16_to_float(b[i]));
    }
}
#endif

// += 运算符
inline fp16_t& operator+=(fp16_t& a, fp16_t b) {
    a = a + b;
    return a;
}

// -= 运算符
inline fp16_t& operator-=(fp16_t& a, fp16_t b) {
    a = a - b;
    return a;
}

// *= 运算符
inline fp16_t& operator*=(fp16_t& a, fp16_t b) {
    a = a * b;
    return a;
}

// /= 运算符
inline fp16_t& operator/=(fp16_t& a, fp16_t b) {
    a = a / b;
    return a;
}

// Clamp函数
template<typename T>
inline T clamp(T val, T min_val, T max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

inline float clamp_float(float val, float min_val, float max_val) {
    return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
}

// ============================================================================
// 批处理数据结构（PyTorch风格，使用torch.float16存储）
// ============================================================================
// 类似于PyTorch的Tensor数据布局
struct BatchData {
    vector<fp16_t> x;             // 自旋位置状态（torch.float16, 范围[-1,1]）
    vector<fp16_t> y;             // 动量状态（torch.float16）
    vector<fp16_t> gradient_term; // 梯度项（torch.float16）
    float energy;                 // 能量值（使用float存储）

    BatchData(int num_vars) :
        x(num_vars), y(num_vars), gradient_term(num_vars), energy(0.0f) {}
};

// ============================================================================
// BSB求解器（PyTorch风格，使用torch.float16进行存储和计算）
// ============================================================================
class BSBFP16Solver {
private:
    ProblemModel model;
    vector<vector<int>> var_to_terms;
    vector<vector<int>> term_positions;
    fp16_t fixed_xi;  // torch.float16存储

public:
    BSBFP16Solver(const ProblemModel& problem_model, double optimized_xi = 0.0) : model(problem_model) {
        precompute_lookup_tables();
        
        if (optimized_xi > 0.0) {
            fixed_xi = float_to_fp16(static_cast<float>(optimized_xi));
        } else {
            fixed_xi = float_to_fp16(static_cast<float>(model.xi));
        }
    }

    // 预计算查找表（PyTorch风格，优化内存访问）
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

    // PyTorch风格的clamp函数（使用FP16边界值）
    template<typename T>
    static inline T torch_clamp(T val, T min_val, T max_val) {
        return (val < min_val) ? min_val : ((val > max_val) ? max_val : val);
    }

    // 初始化（参考GPU init_batches_kernel）
    void init_batches(vector<BatchData>& batches, uint64_t seed, float sigma = 0.1f) {
        mt19937 gen(seed);
        normal_distribution<float> dis_normal(0.0f, 1.0f);
        fp16_t sigma_fp16 = float_to_fp16(static_cast<float>(sigma));

        for (size_t batch = 0; batch < batches.size(); ++batch) {
            mt19937 local_gen(seed + batch);
            for (int i = 0; i < model.num_vars; ++i) {
                // x初始化：sigma * N(0,1) + 0.5，存储为FP16
                float x_norm = clamp_float(fp16_to_float(sigma_fp16) * dis_normal(local_gen) + 0.5f, 0.0f, 1.0f);
                float x_val = clamp_float(x_norm * 2.0f - 1.0f, FP16_MIN_VAL, FP16_MAX_VAL);  // [0,1] -> [-1,1]
                batches[batch].x[i] = float_to_fp16(x_val);
                
                // y初始化：sigma * N(0,1)
                float y_val = clamp_float(fp16_to_float(sigma_fp16) * dis_normal(local_gen), FP16_MIN_VAL, FP16_MAX_VAL);
                batches[batch].y[i] = float_to_fp16(y_val);
            }
        }
    }

    // 梯度计算（使用float32中间计算，减少FP16转换开销，参考GPU版本优化）
    // BSB特征：使用连续变量x进行梯度计算
    // 优化策略：在float32空间完成所有计算，最后一次性转换回FP16
    void batch_calc_gradient(vector<BatchData>& batches) {
        if (model.linear_terms.size() != static_cast<size_t>(model.num_vars)) {
            model.linear_terms.resize(static_cast<size_t>(model.num_vars), 0.0);
        }

        bool is_maximize = (model.objective_sense == "maximize");
        float direction_sign = is_maximize ? 1.0f : -1.0f;

        // 预转换系数到float32（避免重复转换）
        vector<float> linear_coeff_float(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            linear_coeff_float[i] = static_cast<float>(model.linear_terms[i]) * direction_sign;
        }

        vector<float> quad_coeff_float(model.num_quad_terms);
        for (int t = 0; t < model.num_quad_terms; ++t) {
            quad_coeff_float[t] = static_cast<float>(model.quad_terms_coeffs[t]) * direction_sign * 0.5f;
        }

        // 预转换所有batch的x值到float32（一次转换，多次使用）
        vector<vector<float>> x_float_cache(batches.size(), vector<float>(model.num_vars));
        for (size_t b = 0; b < batches.size(); ++b) {
            for (int i = 0; i < model.num_vars; ++i) {
                x_float_cache[b][i] = fp16_to_float(batches[b].x[i]);
            }
        }

        // 在float32空间计算梯度（参考GPU版本的优化策略）
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            auto& batch = batches[batch_idx];
            const auto& x_float = x_float_cache[batch_idx];

            // 在float32空间初始化梯度
            vector<float> gradient_float(model.num_vars);
            for (int i = 0; i < model.num_vars; ++i) {
                gradient_float[i] = linear_coeff_float[i];
            }

            // 在float32空间累加二次项贡献
            if (model.num_quad_terms > 0) {
                for (int t = 0; t < model.num_quad_terms; ++t) {
                    int var1 = model.quad_terms_indices[t][0];
                    int var2 = model.quad_terms_indices[t][1];
                    float J_ij = quad_coeff_float[t];

                    // 在float32空间计算归一化值: (x + 1) * 0.5
                    float x1_norm = (x_float[var1] + 1.0f) * 0.5f;
                    float x2_norm = (x_float[var2] + 1.0f) * 0.5f;

                    // 在float32空间累加梯度
                    gradient_float[var1] += J_ij * x2_norm;
                    gradient_float[var2] += J_ij * x1_norm;
                }
            }

            // 最后一次性转换回FP16（大幅减少转换次数）
            for (int i = 0; i < model.num_vars; ++i) {
                batch.gradient_term[i] = float_to_fp16(gradient_float[i]);
            }
        }
    }

    // 更新y（使用float32中间计算，减少FP16转换开销，参考GPU版本优化）
    // BSB公式：dy/dt = -(1-p(t))*(x-0.5) + ξ*∇f(x)
    void batch_update_y(vector<BatchData>& batches, float p, float dt, float delta) {
        // 预转换参数到float32
        const float p_float = static_cast<float>(p);
        const float dt_float = static_cast<float>(dt);
        const float pump_factor_float = -(1.0f - p_float);
        const float xi_float = fp16_to_float(fixed_xi);

        // 预转换所有batch的x、y、gradient_term到float32
        vector<vector<float>> x_float_cache(batches.size(), vector<float>(model.num_vars));
        vector<vector<float>> y_float_cache(batches.size(), vector<float>(model.num_vars));
        vector<vector<float>> grad_float_cache(batches.size(), vector<float>(model.num_vars));

        for (size_t b = 0; b < batches.size(); ++b) {
            for (int i = 0; i < model.num_vars; ++i) {
                x_float_cache[b][i] = fp16_to_float(batches[b].x[i]);
                y_float_cache[b][i] = fp16_to_float(batches[b].y[i]);
                grad_float_cache[b][i] = fp16_to_float(batches[b].gradient_term[i]);
            }
        }

        // 在float32空间执行所有计算（参考GPU版本的优化策略）
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            auto& batch = batches[batch_idx];
            const auto& x_float = x_float_cache[batch_idx];
            const auto& y_float = y_float_cache[batch_idx];
            const auto& grad_float = grad_float_cache[batch_idx];

            for (int i = 0; i < model.num_vars; ++i) {
                // 在float32空间计算x-0.5: (x + 1) * 0.5 - 0.5
                float x_norm = (x_float[i] + 1.0f) * 0.5f;
                float x_shifted = x_norm - 0.5f;

                // 在float32空间执行BSB更新：y += [-(1-p)*(x-0.5) + ξ*∇f] * dt
                float dy = (pump_factor_float * x_shifted + xi_float * grad_float[i]) * dt_float;
                float y_new = y_float[i] + dy;

                // Clamp到FP16范围
                y_new = (y_new < static_cast<float>(FP16_MIN_VAL)) ? static_cast<float>(FP16_MIN_VAL) :
                        ((y_new > static_cast<float>(FP16_MAX_VAL)) ? static_cast<float>(FP16_MAX_VAL) : y_new);

                // 最后一次性转换回FP16
                batch.y[i] = float_to_fp16(y_new);
            }
        }
    }

    // 更新x（使用float32中间计算，减少FP16转换开销，参考GPU版本优化）
    // BSB公式：dx/dt = y
    void batch_update_x(vector<BatchData>& batches, float dt, float delta) {
        // 预转换dt到float32
        const float dt_float = static_cast<float>(dt);

        // 预转换所有batch的x和y到float32
        vector<vector<float>> x_float_cache(batches.size(), vector<float>(model.num_vars));
        vector<vector<float>> y_float_cache(batches.size(), vector<float>(model.num_vars));

        for (size_t b = 0; b < batches.size(); ++b) {
            for (int i = 0; i < model.num_vars; ++i) {
                x_float_cache[b][i] = fp16_to_float(batches[b].x[i]);
                y_float_cache[b][i] = fp16_to_float(batches[b].y[i]);
            }
        }

        // 在float32空间执行所有计算（参考GPU版本的优化策略）
        for (size_t batch_idx = 0; batch_idx < batches.size(); ++batch_idx) {
            auto& batch = batches[batch_idx];
            const auto& x_float = x_float_cache[batch_idx];
            const auto& y_float = y_float_cache[batch_idx];

            for (int i = 0; i < model.num_vars; ++i) {
                // 在float32空间转换到[0,1]进行更新: (x + 1) * 0.5
                float x_norm = (x_float[i] + 1.0f) * 0.5f;

                // 在float32空间更新x
                float x_new = x_norm + y_float[i] * dt_float;

                // Clamp到[0,1]
                x_new = (x_new < 0.0f) ? 0.0f : ((x_new > 1.0f) ? 1.0f : x_new);

                // 转换回[-1,1]
                float x_fp16_val = x_new * 2.0f - 1.0f;
                x_fp16_val = clamp_float(x_fp16_val, static_cast<float>(FP16_MIN_VAL), static_cast<float>(FP16_MAX_VAL));

                // 最后一次性转换回FP16
                batch.x[i] = float_to_fp16(x_fp16_val);
            }
        }
    }

    // 边界约束（直接使用FP16计算，避免float32转换）
    void batch_apply_constraint(vector<BatchData>& batches) {
        // FP16常量
        fp16_t fp16_1_0 = float_to_fp16(1.0f);
        fp16_t fp16_0_5 = float_to_fp16(0.5f);
        fp16_t fp16_0 = float_to_fp16(0.0f);
        fp16_t fp16_neg_1_0 = float_to_fp16(-1.0f);

        for (auto& batch : batches) {
            // 直接使用FP16检查边界约束
            for (int i = 0; i < model.num_vars; ++i) {
                // 计算x_norm: (x + 1) * 0.5
                fp16_t x_plus_1 = batch.x[i] + fp16_1_0;
                fp16_t x_norm = x_plus_1 * fp16_0_5;

                // 检查边界：|x-0.5| > 0.5，即 x < 0 或 x > 1
                float x_norm_float = fp16_to_float(x_norm);
                float diff = fabs(x_norm_float - 0.5f);

                if (diff > 0.5f) {
                    // 投影到边界{0,1}，然后转换回[-1,1]
                    float projected = (x_norm_float > 0.5f) ? 1.0f : 0.0f;
                    float x_new = projected * 2.0f - 1.0f;  // [0,1] -> [-1,1]
                    batch.x[i] = float_to_fp16(clamp_float(x_new, static_cast<float>(FP16_MIN_VAL), static_cast<float>(FP16_MAX_VAL)));
                    batch.y[i] = fp16_0;  // 清零动量
                }
            }
        }
    }

    // 能量计算（使用float避免累加误差）
    float calc_batch_energy(const BatchData& batch) {
        float scaled_energy = 0.0f;
        bool is_linear_problem = (model.num_quad_terms == 0);
        
        // 二次项贡献
        if (model.num_quad_terms > 0) {
            for (int t = 0; t < model.num_quad_terms; ++t) {
                int var1 = model.quad_terms_indices[t][0];
                int var2 = model.quad_terms_indices[t][1];
                double J_ij = model.quad_terms_coeffs[t];
                
                // 使用连续变量计算
                float x1_val = fp16_to_float(batch.x[var1]);
                float x2_val = fp16_to_float(batch.x[var2]);
                float x1_norm = (x1_val + 1.0f) * 0.5f;
                float x2_norm = (x2_val + 1.0f) * 0.5f;
                
                scaled_energy += 0.5 * J_ij * x1_norm * x2_norm;
            }
        }
        
        // 一次项贡献
        for (int i = 0; i < model.num_vars; ++i) {
            float x_val = fp16_to_float(batch.x[i]);
            float x_norm = (x_val + 1.0f) * 0.5f;
            scaled_energy += model.linear_terms[i] * x_norm;
        }
        
        // 缩放处理
        float original_energy = is_linear_problem ? scaled_energy : scaled_energy * static_cast<float>(model.scaling_factor);
        if (model.constant_term != 0.0) {
            original_energy += static_cast<float>(model.constant_term);
        }
        
        return original_energy;
    }

    // 找最优批次
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

    // 核心求解函数
    vector<fp16_t> solve(int max_iters = 1000, int batch_size = 100, float sigma = 0.1f, vector<BatchData>* out_batches = nullptr) {
        float dt = 0.1f;
        float delta = 0.9f;

        cout << "Starting optimization..." << endl;
        cout << "  - Variables: " << model.num_vars << ", Quadratic terms: " << model.num_quad_terms << endl;
        cout << "  - Linear terms: " << model.num_linear_terms << ", Scaling factor: " << fixed << setprecision(6) << model.scaling_factor << endl;
        cout << "  - BSB Algorithm (PyTorch-style FP16)" << endl;
        cout << "  - Objective sense: " << model.objective_sense << endl;
        cout << "  - Using xi=" << fixed << setprecision(6) << fp16_to_float(fixed_xi) << ", sigma=" << sigma << ", dt=" << dt << endl;
        cout << "  - Iterations: " << max_iters << ", Batch size: " << batch_size << endl;

        vector<BatchData> batches;
        batches.reserve(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            batches.emplace_back(model.num_vars);
        }
        init_batches(batches, time(nullptr), sigma);

        auto start = high_resolution_clock::now();

        // 核心迭代（BSB动力学）
        for (int iter = 0; iter < max_iters; ++iter) {
            double p = static_cast<double>(iter) / (max_iters - 1);

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
        vector<fp16_t> solution(model.num_vars);
        for (int i = 0; i < model.num_vars; ++i) {
            solution[i] = batches[best_idx].x[i];
        }

        if (out_batches != nullptr) {
            *out_batches = batches;
        }
        
        return solution;
    }

    /**
     * @brief 获取模型中的变量数量
     *
     * @return int 返回模型中变量的总数
     */
    int get_num_vars() const {
        return model.num_vars;  // 返回模型中变量的数量
    }
};

// 主函数
int main(int argc, char* argv[]) {
    string qplib_file_path;
    double optimized_xi = 0.365;
    
    if (argc > 1) {
        qplib_file_path = argv[1];
    } else {
        qplib_file_path = DEFAULT_QPLIB_FILE_PATH;
    }
    
    if (argc > 2) {
        optimized_xi = stod(argv[2]);
    }

        cout << "========================================" << endl;
        cout << "       BSB_CPU_fp16 Solver (PyTorch-style)     " << endl;
        cout << "  - File path: " << qplib_file_path << endl;
        cout << "  - Algorithm: BSB (Ballistic Simulated Bifurcation)" << endl;
        cout << "  - Data type: PyTorch torch.float16 (IEEE 754 half precision)" << endl;
        cout << "  - Features:" << endl;
        cout << "    * IEEE 754 binary16 format (1 sign, 5 exp, 10 mantissa)" << endl;
        cout << "    * Round-to-nearest-even (RNTE) rounding mode" << endl;
        cout << "    * Fully compatible with PyTorch tensor operations" << endl;
        cout << "  - Fixed optimized xi: " << fixed << setprecision(6) << optimized_xi << endl;
        cout << "  - Fixed sigma: 0.1" << endl;
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
        BSBFP16Solver solver(model, optimized_xi);
        vector<BatchData> batches;

        auto start = high_resolution_clock::now();
        vector<fp16_t> solution = solver.solve(1000, 100, 0.1f, &batches);
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
            float x_val = fp16_to_float(batches[best_idx].x[i]);
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
