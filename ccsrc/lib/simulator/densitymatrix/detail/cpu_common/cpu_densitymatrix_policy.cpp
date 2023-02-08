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

#include <cmath>

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <ratio>
#include <stdexcept>
#include <vector>

#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#ifdef __x86_64__
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_double_policy.hpp"
#    include "simulator/densitymatrix/detail/cpu_densitymatrix_arm_float_policy.hpp"
#endif
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"
namespace mindquantum::sim::densitymatrix::detail {
// Warning: only correct when x >= y
template <typename derived_, typename calc_type_>
index_t CPUDensityMatrixPolicyBase<derived_, calc_type_>::IdxMap(index_t x, index_t y) {
    return (x * (x + 1)) / 2 + y;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::GetValue(qs_data_p_t qs, index_t x, index_t y) -> qs_data_t {
    if (x >= y) {
        return qs[IdxMap(x, y)];
    } else {
        return std::conj(qs[IdxMap(y, x)]);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SetValue(qs_data_p_t qs, index_t x, index_t y, qs_data_t data) {
    if (x >= y) {
        qs[IdxMap(x, y)] = data;
    } else {
        qs[IdxMap(y, x)] = std::conj(data);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SelfMultiply(qs_data_p_t qs, index_t x, index_t y, qs_data_t data) {
    if (x >= y) {
        qs[IdxMap(x, y)] *= data;
    } else {
        qs[IdxMap(y, x)] *= std::conj(data);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SwapValue(qs_data_p_t qs, index_t x0, index_t y0, index_t x1, index_t y1,
                                           qs_data_t coeff) {
    if (x0 >= y0) {
        qs_data_t tmp = qs[IdxMap(x0, y0)];
        qs[IdxMap(x0, y0)] = coeff * GetValue(qs, x1, y1);
        SetValue(qs, x1, y1, coeff * tmp);
    } else {
        qs_data_t tmp = std::conj(qs[IdxMap(y0, x0)]);
        qs[IdxMap(y0, x0)] = std::conj(coeff * GetValue(qs, x1, y1));
        SetValue(qs, x1, y1, coeff * tmp);
    }
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    index_t n_elements = (dim * dim + dim) / 2;
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::Reset(qs_data_p_t qs, index_t dim, bool zero_state) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { qs[i] = 0; })
    if (!zero_state) {
        qs[0] = 1;
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        free(qs);
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    index_t dim = 1UL << n_qubits;
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < dim; i++) {
        for (index_t j = 0; j <= i; j++) {
            std::cout << "(" << qs[IdxMap(i, j)].real() << ", " << qs[IdxMap(i, j)].imag() << ")"
                      << ",";
        }
        for (index_t j = i + 1; j < dim; j++) {
            std::cout << "(" << qs[IdxMap(j, i)].real() << ", " << -qs[IdxMap(j, i)].imag() << ")"
                      << ",";
        }
        std::cout << std::endl;
    }
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                for (index_t j = 0; j < (dim / 2); j++) {
                    if ((j & ctrl_mask) != ctrl_mask) {
                        qs[IdxMap(i, j)] = 0;
                    }
                }
            }
        })
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = derived::InitState(dim, false);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { out[i] = qs[i]; })
    return out;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::GetQS(qs_data_p_t qs, index_t dim) -> matrix_t {
    matrix_t out(dim, py_qs_datas_t(dim));
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j < i; j++) {
                out[i][j] = qs[IdxMap(i, j)];
            }
            for (index_t j = i; j < dim; j++) {
                out[i][j] = std::conj(qs[IdxMap(j, i)]);
            }
        })
    return out;
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::SetQS(qs_data_p_t qs, const py_qs_datas_t& vec_out, index_t dim) {
    if (vec_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            for (index_t j = 0; j <= i; j++) {
                qs[IdxMap(i, j)] = vec_out[i] * vec_out[j];
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::CopyQS(qs_data_p_t qs, const qs_data_p_t qs_out, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { qs[i] = qs_out[i]; })
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::Purity(qs_data_p_t qs, index_t dim) -> calc_type {
    calc_type p = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { p += 2 * std::norm(qs[i]); })
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < dim; i++) { p += -std::norm(qs[IdxMap(i, i)]); })
    return p;
}

template <typename derived_, typename calc_type_>
bool CPUDensityMatrixPolicyBase<derived_, calc_type_>::IsPure(qs_data_p_t qs, index_t dim) {
    auto p = Purity(qs, dim);
    if (std::abs(p - 1) < 1e-8) {
        return true;
    } else {
        return false;
    }
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::PureStateVector(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    if (!IsPure(qs, dim)) {
        throw(std::runtime_error("PureStateVector(): Cannot transform mixed density matrix to vector."));
    }
    py_qs_datas_t qs_vector(dim);
    index_t base;
    calc_type base_value;
    for (index_t i = 0; i < dim; i++) {
        if (qs[IdxMap(i, i)].real() > 1e-8) {
            base = i;
            base_value = std::sqrt(qs[IdxMap(i, i)].real());
            qs_vector[i] = base_value;
            break;
        }
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = base + 1; i < dim; i++) { qs_vector[i] = qs[IdxMap(i, base)] / base_value; })
    return qs_vector;
}

template <typename derived_, typename calc_type_>
auto CPUDensityMatrixPolicyBase<derived_, calc_type_>::DiagonalConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, index_t dim)
    -> calc_type {
    // collect diagonal amplitude with index mask satisfied condition.
    calc_type res_real = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real)), dim, DimTh,
                     for (omp::idx_t i = 0; i < dim; i++) {
                         if ((i & mask) == condi) {
                             res_real += qs[IdxMap(i, i)].real();
                         }
                     });
    return res_real;
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim) {
    matrix_t tmp(dim, VT<qs_data_t>(dim));
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(static_cast<int64_t>(i & mask.mask_z));  // -1
                    auto axis3power = CountOne(static_cast<int64_t>(i & mask.mask_y));  // -1j
                    auto c = ComplexCast<double, calc_type>::apply(POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                    for (index_t col = 0; col < dim; col++) {
                        tmp[j][col] += GetValue(qs, i, col) * coeff * c;
                    }
                    if (i != j) {
                        for (index_t col = 0; col < dim; col++) {
                            tmp[i][col] += GetValue(qs, j, col) * coeff / c;
                        }
                    }
                }
            })
    }
    Reset(qs, dim, true);
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = CountOne(static_cast<int64_t>(i & mask.mask_z));  // -1
                    auto axis3power = CountOne(static_cast<int64_t>(i & mask.mask_y));  // -1j
                    auto c = ComplexCast<double, calc_type>::apply(POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                    for (index_t row = 0; row <= i; row++) {
                        qs[IdxMap(i, row)] += std::conj(tmp[row][j] * coeff * c);
                    }
                    if (i != j) {
                        for (index_t row = 0; row <= j; row++) {
                            qs[IdxMap(j, row)] += std::conj(tmp[row][i] * coeff / c);
                        }
                    }
                }
            })
    }
}

template <typename derived_, typename calc_type_>
template <class binary_op>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                   qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim,
                                                   const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            auto _i_0 = IdxMap(i, 0);
            if ((i & mask) == condi) {
                for (index_t j = 0; j <= i; j++) {
                    if ((j & mask) == condi) {
                        des[_i_0 + j] = op(src[_i_0 + j], succ_coeff);
                    } else {
                        des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                    }
                }
            } else {
                for (index_t j = 0; j <= i; j++) {
                    des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                }
            }
        })
}

template <typename derived_, typename calc_type_>
template <index_t mask, index_t condi, class binary_op>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff,
                                                   qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            auto _i_0 = IdxMap(i, 0);
            if ((i & mask) == condi) {
                for (index_t j = 0; j <= i; j++) {
                    if ((j & mask) == condi) {
                        des[_i_0 + j] = op(src[_i_0 + j], succ_coeff);
                    } else {
                        des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                    }
                }
            } else {
                for (index_t j = 0; j <= i; j++) {
                    des[_i_0 + j] = op(src[_i_0 + j], fail_coeff);
                }
            }
        })
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim) {
    derived::template ConditionalBinary<0, 0>(src, des, value, 0, dim, std::multiplies<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::plus<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                  qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::minus<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::multiplies<qs_data_t>());
}
template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::divides<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void CPUDensityMatrixPolicyBase<derived_, calc_type_>::ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                                 const qbits_t& ctrls, const matrix_t& m, index_t dim) {
    if (objs.size() == 1) {
        derived::ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        derived::ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        throw std::runtime_error("Can not custom " + std::to_string(objs.size()) + " qubits gate for cpu backend.");
    }
}

#ifdef __x86_64__
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmFloat, float>;
template struct CPUDensityMatrixPolicyBase<CPUDensityMatrixPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::densitymatrix::detail
