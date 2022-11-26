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

#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"

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

namespace mindquantum::sim::densitymatrix::detail {

// Warning: only correct when x >= y
index_t CPUDensityMatrixPolicyBase::IdxMap(index_t x, index_t y) {
    return (x * x + x) / 2 + y;
}

auto CPUDensityMatrixPolicyBase::GetValue(qs_data_p_t qs, index_t x, index_t y) -> qs_data_t {
    if (x >= y) {
        return qs[IdxMap(x, y)];
    } else {
        return std::conj(qs[IdxMap(y, x)]);
    }
}

void CPUDensityMatrixPolicyBase::SetValue(qs_data_p_t qs, index_t x, index_t y, qs_data_t data) {
    if (x >= y) {
        qs[IdxMap(x, y)] = data;
    } else {
        qs[IdxMap(y, x)] = std::conj(data);
    }
}

void CPUDensityMatrixPolicyBase::SelfMultiply(qs_data_p_t qs, index_t x, index_t y, qs_data_t data) {
    if (x >= y) {
        qs[IdxMap(x, y)] *= data;
    } else {
        qs[IdxMap(y, x)] *= std::conj(data);
    }
}

auto CPUDensityMatrixPolicyBase::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    index_t n_elements = (dim * dim + dim) / 2;
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(n_elements, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

void CPUDensityMatrixPolicyBase::Reset(qs_data_p_t qs, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { qs[i] = 0; })
    qs[0] = 1;
}

void CPUDensityMatrixPolicyBase::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        free(qs);
    }
}

void CPUDensityMatrixPolicyBase::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
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

void CPUDensityMatrixPolicyBase::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (index_t i = 0; i < dim; i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                for (index_t j = 0; j < (dim / 2); j++) {
                    if ((j & ctrl_mask) != ctrl_mask) {
                        qs[IdxMap(i, j)] = 0;
                    }
                }
            }
        })
}

auto CPUDensityMatrixPolicyBase::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = CPUDensityMatrixPolicyBase::InitState(dim, false);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { out[i] = qs[i]; })
    return out;
}

auto CPUDensityMatrixPolicyBase::GetQS(qs_data_p_t qs, index_t dim) -> matrix_t {
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

void CPUDensityMatrixPolicyBase::SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = qs_out[i]; })
}

void CPUDensityMatrixPolicyBase::CopyQS(qs_data_p_t qs, const qs_data_p_t qs_out, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { qs[i] = qs_out[i]; })
}

auto CPUDensityMatrixPolicyBase::Purity(qs_data_p_t qs, index_t dim) -> calc_type {
    calc_type p = 0;
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < (dim * dim + dim) / 2; i++) { p += 2 * std::norm(qs[i]); })
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: p)), dim, DimTh,
                     for (omp::idx_t i = 0; i < dim; i++) { p += -std::norm(qs[IdxMap(i, i)]); })
    return p;
}

bool CPUDensityMatrixPolicyBase::IsPure(qs_data_p_t qs, index_t dim) {
    auto p = Purity(qs, dim);
    if (std::abs(p - 1) < 1e-8) {
        return true;
    } else {
        return false;
    }
}

auto CPUDensityMatrixPolicyBase::PureStateVector(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    if (!IsPure(qs, dim)) {
        throw(std::runtime_error("PureStateVector(): Cannot transform mixd density matrix to vector."));
    }
    py_qs_datas_t qs_vector(dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs_vector[i] = std::sqrt(qs[IdxMap(i, i)]); })
    return qs_vector;
}

auto CPUDensityMatrixPolicyBase::DiagonalConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, bool abs,
                                                            index_t dim) -> calc_type {
    // collect diagonal amplitude with index mask satisfied condition.
    calc_type res_real = 0;
    if (abs) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real)), dim, DimTh,
                         for (omp::idx_t i = 0; i < dim; i++) {
                             if ((i & mask) == condi) {
                                 auto _ii = IdxMap(i, i);
                                 res_real += qs[_ii].real();
                             }
                         });
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real)), dim, DimTh,
                         for (omp::idx_t i = 0; i < dim; i++) {
                             if ((i & mask) == condi) {
                                 res_real += qs[IdxMap(i, i)].real();
                             }
                         });
    }
    return res_real;
}

auto CPUDensityMatrixPolicyBase::ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim)
    -> qs_data_p_t {
    qs_data_p_t out = CPUDensityMatrixPolicyBase::InitState(dim, false);
    for (const auto& [pauli_string, coeff_] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        auto coeff = coeff_;
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t i = 0; i < dim; i++) {
                auto j = (i ^ mask_f);
                for (index_t a = 0; a < dim; a++) {
                    auto b = (a ^ mask_f);
                    if (i <= j) {
                        if (a <= b) {
                            auto axis2power_l = CountOne(static_cast<int64_t>(i & mask.mask_z));  // -1
                            auto axis3power_l = CountOne(static_cast<int64_t>(i & mask.mask_y));  // -1j
                            auto c_l = POLAR[static_cast<char>((mask.num_y + 2 * axis3power_l + 2 * axis2power_l) & 3)];

                            auto axis2power_r = CountOne(static_cast<int64_t>(a & mask.mask_z));  // -1
                            auto axis3power_r = CountOne(static_cast<int64_t>(a & mask.mask_y));  // -1j
                            auto c_r = POLAR[static_cast<char>((mask.num_y + 2 * axis3power_r + 2 * axis2power_r) & 3)];
                            SetValue(out, b, j, qs[IdxMap(i, a)] * coeff * coeff * c_l * std::conj(c_r));
                            // out[IdxMap(b, j)] += qs[IdxMap(i, a)] * coeff * coeff * c_l * std::conj(c_r);
                            // if (i != j) {
                            //     if (a != b) {
                            //         out[IdxMap(i, a)] += qs[IdxMap(b, j)] / c_l / std::conj(c_r);
                            //     }
                            // }
                        }
                    }
                }
            })
    }
    return out;
};

template <class binary_op>
void CPUDensityMatrixPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                   qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim,
                                                   const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (index_t i = 0; i < dim; i++) {
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

template <index_t mask, index_t condi, class binary_op>
void CPUDensityMatrixPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff,
                                                   qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (index_t i = 0; i < dim; i++) {
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

void CPUDensityMatrixPolicyBase::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim) {
    ConditionalBinary<0, 0>(src, des, value, 0, dim, std::multiplies<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::plus<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                  qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::minus<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::multiplies<qs_data_t>());
}
void CPUDensityMatrixPolicyBase::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::divides<qs_data_t>());
}

}  // namespace mindquantum::sim::densitymatrix::detail