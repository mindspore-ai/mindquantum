//   Copyright 2022 <Huawei Technologies Co., Ltd>
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.

#include "config/openmp.hpp"

#include "math/pr/parameter_resolver.hpp"
#include "simulator/utils.hpp"
#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.hpp"
#endif
#include "simulator/vector/detail/cpu_vector_policy.hpp"
namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffNQubitsMatrix(qs_data_p_t bra, qs_data_p_t ket,
                                                                        const qbits_t& objs, const qbits_t& ctrls,
                                                                        const VVT<py_qs_data_t>& gate, index_t dim)
    -> qs_data_t {
    size_t n_qubit = objs.size();
    size_t m_dim = (1UL << n_qubit);
    size_t ctrl_mask = 0;
    for (auto& i : ctrls) {
        ctrl_mask |= 1UL << i;
    }
    std::vector<size_t> obj_masks{};
    for (size_t i = 0; i < m_dim; i++) {
        size_t n = 0;
        size_t mask_j = 0;
        for (size_t j = i; j != 0; j >>= 1) {
            if (j & 1) {
                mask_j += 1UL << objs[n];
            }
            n += 1;
        }
        obj_masks.push_back(mask_j);
    }
    auto obj_mask = obj_masks.back();
    calc_type res_real = 0, res_imag = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                for (omp::idx_t l = 0; l < dim; l++) {
                                                    if (((l & ctrl_mask) == ctrl_mask) && ((l & obj_mask) == 0)) {
                                                        for (size_t i = 0; i < m_dim; i++) {
                                                            qs_data_t tmp = 0;
                                                            for (size_t j = 0; j < m_dim; j++) {
                                                                tmp += gate[i][j] * ket[obj_masks[j] | l];
                                                            }
                                                            tmp = std::conj(bra[obj_masks[i] | l]) * tmp;
                                                            res_real += tmp.real();
                                                            res_imag += tmp.imag();
                                                        }
                                                    }
                                                })
    return {res_real, res_imag};
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffTwoQubitsMatrix(qs_data_p_t bra, qs_data_p_t ket,
                                                                          const qbits_t& objs, const qbits_t& ctrls,
                                                                          const VVT<py_qs_data_t>& gate, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
        for (omp::idx_t l = 0; l < (dim / 4); l++) {
            index_t i;
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask,
                            l, i);
            auto j = i + mask.obj_min_mask;
            auto k = i + mask.obj_max_mask;
            auto m = i + mask.obj_mask;
            auto v00 = gate[0][0] * ket[i] + gate[0][1] * ket[j] + gate[0][2] * ket[k] + gate[0][3] * ket[m];
            auto v01 = gate[1][0] * ket[i] + gate[1][1] * ket[j] + gate[1][2] * ket[k] + gate[1][3] * ket[m];
            auto v10 = gate[2][0] * ket[i] + gate[2][1] * ket[j] + gate[2][2] * ket[k] + gate[2][3] * ket[m];
            auto v11 = gate[3][0] * ket[i] + gate[3][1] * ket[j] + gate[3][2] * ket[k] + gate[3][3] * ket[m];
            auto this_res = std::conj(bra[i]) * v00;
            this_res += std::conj(bra[j]) * v01;
            this_res += std::conj(bra[k]) * v10;
            this_res += std::conj(bra[m]) * v11;
            res_real += this_res.real();
            res_imag += this_res.imag();
        })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
        for (omp::idx_t l = 0; l < (dim / 4); l++) {
            index_t i;
            SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                            mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
            if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = gate[0][0] * ket[i] + gate[0][1] * ket[j] + gate[0][2] * ket[k] + gate[0][3] * ket[m];
                auto v01 = gate[1][0] * ket[i] + gate[1][1] * ket[j] + gate[1][2] * ket[k] + gate[1][3] * ket[m];
                auto v10 = gate[2][0] * ket[i] + gate[2][1] * ket[j] + gate[2][2] * ket[k] + gate[2][3] * ket[m];
                auto v11 = gate[3][0] * ket[i] + gate[3][1] * ket[j] + gate[3][2] * ket[k] + gate[3][3] * ket[m];
                auto this_res = std::conj(bra[i]) * v00;
                this_res += std::conj(bra[j]) * v01;
                this_res += std::conj(bra[k]) * v10;
                this_res += std::conj(bra[m]) * v11;
                res_real += this_res.real();
                res_imag += this_res.imag();
            }
        })
    }
    // clang-format on
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffSingleQubitMatrix(qs_data_p_t bra, qs_data_p_t ket,
                                                                            const qbits_t& objs, const qbits_t& ctrls,
                                                                            const VVT<py_qs_data_t>& m, index_t dim)
    -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    auto t1 = m[0][0] * ket[i] + m[0][1] * ket[j];
                    auto t2 = m[1][0] * ket[i] + m[1][1] * ket[j];
                    auto this_res = std::conj(bra[i]) * t1 + std::conj(bra[j]) * t2;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                });
        // clang-format on
    } else {
        if (mask.ctrl_qubits.size() == 1) {
            index_t ctrl_low = 0UL;
            for (qbit_t i = 0; i < mask.ctrl_qubits[0]; i++) {
                ctrl_low = (ctrl_low << 1) + 1;
            }
            index_t first_low_mask = mask.obj_low_mask;
            index_t second_low_mask = ctrl_low;
            if (mask.obj_low_mask > ctrl_low) {
                first_low_mask = ctrl_low;
                second_low_mask = mask.obj_low_mask;
            }
            auto first_high_mask = ~first_low_mask;
            auto second_high_mask = ~second_low_mask;
            // clang-format off
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                        auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                        i = ((i & second_high_mask) << 1) + (i & second_low_mask) + mask.ctrl_mask;
                        auto j = i + mask.obj_mask;
                        auto t1 = m[0][0] * ket[i] + m[0][1] * ket[j];
                        auto t2 = m[1][0] * ket[i] + m[1][1] * ket[j];
                        auto this_res = std::conj(bra[i]) * t1 + std::conj(bra[j]) * t2;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    });
            // clang-format on
        } else {
            // clang-format off
            THRESHOLD_OMP(
                MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                    for (omp::idx_t l = 0; l < (dim / 2); l++) {
                        auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                            auto j = i + mask.obj_mask;
                            auto t1 = m[0][0] * ket[i] + m[0][1] * ket[j];
                            auto t2 = m[1][0] * ket[i] + m[1][1] * ket[j];
                            auto this_res = std::conj(bra[i]) * t1 + std::conj(bra[j]) * t2;
                            res_real += this_res.real();
                            res_imag += this_res.imag();
                        }
                    });
            // clang-format on
        }
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffMatrixGate(qs_data_p_t bra, qs_data_p_t ket,
                                                                     const qbits_t& objs, const qbits_t& ctrls,
                                                                     const VVT<py_qs_data_t>& m, index_t dim)
    -> qs_data_t {
    if (objs.size() == 1) {
        return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
    }
    if (objs.size() == 2) {
        return derived::ExpectDiffTwoQubitsMatrix(bra, ket, objs, ctrls, m, dim);
    }
    return derived::ExpectDiffNQubitsMatrix(bra, ket, objs, ctrls, m, dim);
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                             const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto is = static_cast<calc_type>(0.5 * std::cos(val / 2)) * IMAGE_MI;
    VVT<py_qs_data_t> gate = {{c, is}, {is, c}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                             const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    calc_type c = -0.5 * std::sin(val / 2);
    calc_type s = 0.5 * std::cos(val / 2);
    VVT<py_qs_data_t> gate = {{c, -s}, {s, c}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                             const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    calc_type c = -0.5 * std::sin(val / 2);
    calc_type s = 0.5 * std::cos(val / 2);
    auto e0 = c + IMAGE_MI * s;
    auto e1 = c + IMAGE_I * s;
    VVT<py_qs_data_t> gate = {{e0, 0}, {0, e1}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffGP(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                             const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    auto e = std::complex<calc_type>(0, -1);
    e *= std::exp(std::complex<calc_type>(0, -val));
    VVT<py_qs_data_t> gate = {{e, 0}, {0, e}};
    return derived::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                             const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    calc_type res_real = 0, res_imag = 0;
    auto e = -std::sin(val) + IMAGE_I * std::cos(val);

    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    auto this_res = std::conj(bra[j]) * ket[j] * e;
                    res_real += this_res.real();
                    res_imag += this_res.imag();
                })
        // clang-format on
    } else {
        // clang-format off
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        auto this_res = std::conj(bra[j]) * ket[j] * e;
                        res_real += this_res.real();
                        res_imag += this_res.imag();
                    }
                })
        // clang-format on
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRxx(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                              const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type_>(std::cos(val / 2) / 2) * IMAGE_MI;
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto v00 = c * ket[i] + s * ket[m];
                                                        auto v01 = c * ket[j] + s * ket[k];
                                                        auto v10 = c * ket[k] + s * ket[j];
                                                        auto v11 = c * ket[m] + s * ket[i];
                                                        auto this_res = std::conj(bra[i]) * v00;
                                                        this_res += std::conj(bra[j]) * v01;
                                                        this_res += std::conj(bra[k]) * v10;
                                                        this_res += std::conj(bra[m]) * v11;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto v00 = c * ket[i] + s * ket[m];
                                                            auto v01 = c * ket[j] + s * ket[k];
                                                            auto v10 = c * ket[k] + s * ket[j];
                                                            auto v11 = c * ket[m] + s * ket[i];
                                                            auto this_res = std::conj(bra[i]) * v00;
                                                            this_res += std::conj(bra[j]) * v01;
                                                            this_res += std::conj(bra[k]) * v10;
                                                            this_res += std::conj(bra[m]) * v11;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRxy(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                              const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type_>(std::cos(val / 2) / 2);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto v00 = c * ket[i] - s * ket[m];
                                                        auto v01 = c * ket[j] - s * ket[k];
                                                        auto v10 = c * ket[k] + s * ket[j];
                                                        auto v11 = c * ket[m] + s * ket[i];
                                                        auto this_res = std::conj(bra[i]) * v00;
                                                        this_res += std::conj(bra[j]) * v01;
                                                        this_res += std::conj(bra[k]) * v10;
                                                        this_res += std::conj(bra[m]) * v11;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto v00 = c * ket[i] - s * ket[m];
                                                            auto v01 = c * ket[j] - s * ket[k];
                                                            auto v10 = c * ket[k] + s * ket[j];
                                                            auto v11 = c * ket[m] + s * ket[i];
                                                            auto this_res = std::conj(bra[i]) * v00;
                                                            this_res += std::conj(bra[j]) * v01;
                                                            this_res += std::conj(bra[k]) * v10;
                                                            this_res += std::conj(bra[m]) * v11;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRxz(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                              const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type_>(std::cos(val / 2) / 2) * IMAGE_MI;
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto v00 = c * ket[i] + s * ket[j];
                                                        auto v01 = c * ket[j] + s * ket[i];
                                                        auto v10 = c * ket[k] - s * ket[m];
                                                        auto v11 = c * ket[m] - s * ket[k];
                                                        auto this_res = std::conj(bra[i]) * v00;
                                                        this_res += std::conj(bra[j]) * v01;
                                                        this_res += std::conj(bra[k]) * v10;
                                                        this_res += std::conj(bra[m]) * v11;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto v00 = c * ket[i] + s * ket[j];
                                                            auto v01 = c * ket[j] + s * ket[i];
                                                            auto v10 = c * ket[k] - s * ket[m];
                                                            auto v11 = c * ket[m] - s * ket[k];
                                                            auto this_res = std::conj(bra[i]) * v00;
                                                            this_res += std::conj(bra[j]) * v01;
                                                            this_res += std::conj(bra[k]) * v10;
                                                            this_res += std::conj(bra[m]) * v11;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRyz(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                              const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type_>(std::cos(val / 2) / 2);
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto v00 = c * ket[i] - s * ket[j];
                                                        auto v01 = c * ket[j] + s * ket[i];
                                                        auto v10 = c * ket[k] + s * ket[m];
                                                        auto v11 = c * ket[m] - s * ket[k];
                                                        auto this_res = std::conj(bra[i]) * v00;
                                                        this_res += std::conj(bra[j]) * v01;
                                                        this_res += std::conj(bra[k]) * v10;
                                                        this_res += std::conj(bra[m]) * v11;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto v00 = c * ket[i] - s * ket[j];
                                                            auto v01 = c * ket[j] + s * ket[i];
                                                            auto v10 = c * ket[k] + s * ket[m];
                                                            auto v11 = c * ket[m] - s * ket[k];
                                                            auto this_res = std::conj(bra[i]) * v00;
                                                            this_res += std::conj(bra[j]) * v01;
                                                            this_res += std::conj(bra[k]) * v10;
                                                            this_res += std::conj(bra[m]) * v11;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRyy(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                              const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type_>(std::cos(val / 2) / 2) * IMAGE_I;
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto v00 = c * ket[i] + s * ket[m];
                                                        auto v01 = c * ket[j] - s * ket[k];
                                                        auto v10 = c * ket[k] - s * ket[j];
                                                        auto v11 = c * ket[m] + s * ket[i];
                                                        auto this_res = std::conj(bra[i]) * v00;
                                                        this_res += std::conj(bra[j]) * v01;
                                                        this_res += std::conj(bra[k]) * v10;
                                                        this_res += std::conj(bra[m]) * v11;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto v00 = c * ket[i] + s * ket[m];
                                                            auto v01 = c * ket[j] - s * ket[k];
                                                            auto v10 = c * ket[k] - s * ket[j];
                                                            auto v11 = c * ket[m] + s * ket[i];
                                                            auto this_res = std::conj(bra[i]) * v00;
                                                            this_res += std::conj(bra[j]) * v01;
                                                            this_res += std::conj(bra[k]) * v10;
                                                            this_res += std::conj(bra[m]) * v11;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};

template <typename derived_, typename calc_type_>
auto CPUVectorPolicyBase<derived_, calc_type_>::ExpectDiffRzz(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                              const qbits_t& ctrls, calc_type val, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);

    auto c = static_cast<calc_type_>(-std::sin(val / 2) / 2);
    auto s = static_cast<calc_type_>(std::cos(val / 2) / 2);
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    calc_type res_real = 0, res_imag = 0;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        auto m = i + mask.obj_mask;
                                                        auto j = i + mask.obj_min_mask;
                                                        auto k = i + mask.obj_max_mask;
                                                        auto this_res = std::conj(bra[i]) * ket[i] * me;
                                                        this_res += std::conj(bra[j]) * ket[j] * e;
                                                        this_res += std::conj(bra[k]) * ket[k] * e;
                                                        this_res += std::conj(bra[m]) * ket[m] * me;
                                                        res_real += this_res.real();
                                                        res_imag += this_res.imag();
                                                    })
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
                                                    for (omp::idx_t l = 0; l < (dim / 4); l++) {
                                                        index_t i;
                                                        SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask,
                                                                      mask.obj_high_mask, mask.obj_rev_high_mask, l, i);
                                                        if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                                                            auto m = i + mask.obj_mask;
                                                            auto j = i + mask.obj_min_mask;
                                                            auto k = i + mask.obj_max_mask;
                                                            auto this_res = std::conj(bra[i]) * ket[i] * me;
                                                            this_res += std::conj(bra[j]) * ket[j] * e;
                                                            this_res += std::conj(bra[k]) * ket[k] * e;
                                                            this_res += std::conj(bra[m]) * ket[m] * me;
                                                            res_real += this_res.real();
                                                            res_imag += this_res.imag();
                                                        }
                                                    })
    }
    return {res_real, res_imag};
};
#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
