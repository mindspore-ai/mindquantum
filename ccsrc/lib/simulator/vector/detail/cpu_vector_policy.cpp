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

#include "simulator/vector/detail/cpu_vector_policy.hpp"

#include <cmath>

#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <ratio>
#include <stdexcept>
#include <vector>

#include "config/openmp.hpp"

#include "core/sparse/algo.hpp"
#include "core/utils.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::vector::detail {
auto CPUVectorPolicyBase::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(dim, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

void CPUVectorPolicyBase::Reset(qs_data_p_t qs, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = 0; })
    qs[0] = 1;
}

void CPUVectorPolicyBase::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        free(qs);
    }
}

void CPUVectorPolicyBase::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < (1UL << n_qubits); i++) {
        std::cout << "(" << qs[i].real() << ", " << qs[i].imag() << ")" << std::endl;
    }
}

void CPUVectorPolicyBase::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                qs[i] = 0;
            }
        })
}

template <class binary_op>
void CPUVectorPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                            qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim,
                                            const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & mask) == condi) {
                des[i] = op(src[i], succ_coeff);
            } else {
                des[i] = op(src[i], fail_coeff);
            }
        })
}

template <index_t mask, index_t condi, class binary_op>
void CPUVectorPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff,
                                            qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    // if index mask satisfied condition, multiply by succe_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & mask) == condi) {
                des[i] = op(src[i], succ_coeff);
            } else {
                des[i] = op(src[i], fail_coeff);
            }
        })
}

void CPUVectorPolicyBase::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim) {
    ConditionalBinary<0, 0>(src, des, value, 0, dim, std::multiplies<qs_data_t>());
}
void CPUVectorPolicyBase::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                         qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::plus<qs_data_t>());
}
void CPUVectorPolicyBase::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                           qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::minus<qs_data_t>());
}
void CPUVectorPolicyBase::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                         qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::multiplies<qs_data_t>());
}
void CPUVectorPolicyBase::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                         qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::divides<qs_data_t>());
}

auto CPUVectorPolicyBase::ConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, bool abs, index_t dim)
    -> qs_data_t {
    // collect amplitude with index mask satisfied condition.
    calc_type res_real = 0, res_imag = 0;
    if (abs) {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real)), dim, DimTh,
                         for (omp::idx_t i = 0; i < dim; i++) {
                             if ((i & mask) == condi) {
                                 res_real += qs[i].real() * qs[i].real() + qs[i].imag() * qs[i].imag();
                             }
                         });
    } else {
        THRESHOLD_OMP(
            MQ_DO_PRAGMA(omp parallel for schedule(static) reduction(+: res_real, res_imag)), dim, DimTh,
                         for (omp::idx_t i = 0; i < dim; i++) {
                             if ((i & mask) == condi) {
                                 res_real += qs[i].real();
                                 res_imag += qs[i].imag();
                             }
                         });
    }
    return qs_data_t(res_real, res_imag);
}

auto CPUVectorPolicyBase::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = CPUVectorPolicyBase::InitState(dim, false);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { out[i] = qs[i]; })
    return out;
};

auto CPUVectorPolicyBase::Vdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t i = 0; i < dim; i++) {
                res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
            })
    // clang-format on
    return {res_real, res_imag};
}

template <index_t mask, index_t condi>
auto CPUVectorPolicyBase::ConditionVdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t i = 0; i < dim; i++) {
                if ((i & mask) == condi) {
                    res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                    res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
                }
            })
    // clang-format on
    return {res_real, res_imag};
}

auto CPUVectorPolicyBase::OneStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
    -> py_qs_data_t {
    SingleQubitGateMask mask({obj_qubit}, {});
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask) + mask.obj_mask;
                res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
            })
    // clang-format on
    return {res_real, res_imag};
}

auto CPUVectorPolicyBase::ZeroStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
    -> py_qs_data_t {
    SingleQubitGateMask mask({obj_qubit}, {});
    calc_type res_real = 0, res_imag = 0;
    // clang-format off
    THRESHOLD_OMP(
        MQ_DO_PRAGMA(omp parallel for reduction(+:res_real, res_imag) schedule(static)), dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                res_real += bra[i].real() * ket[i].real() + bra[i].imag() * ket[i].imag();
                res_imag += bra[i].real() * ket[i].imag() - bra[i].imag() * ket[i].real();
            })
    // clang-format on
    return {res_real, res_imag};
}

auto CPUVectorPolicyBase::GetQS(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    py_qs_datas_t out(dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { out[i] = qs[i]; })
    return out;
}

void CPUVectorPolicyBase::SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = qs_out[i]; })
}

auto CPUVectorPolicyBase::ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim)
    -> qs_data_p_t {
    qs_data_p_t out = CPUVectorPolicyBase::InitState(dim, false);
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
                    auto c = POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)];
                    out[j] += qs[i] * coeff * c;
                    if (i != j) {
                        out[i] += qs[j] * coeff / c;
                    }
                }
            })
    }
    return out;
};

void CPUVectorPolicyBase::ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto tmp = qs[j];
                qs[j] = qs[k];
                qs[k] = tmp;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto tmp = qs[j];
                    qs[j] = qs[k];
                    qs[k] = tmp;
                }
            })
    }
}

void CPUVectorPolicyBase::ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, bool daggered,
                                     index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    calc_type frac = 1.0;
    if (daggered) {
        frac = -1.0;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto tmp = qs[i + mask.obj_min_mask];
                qs[i + mask.obj_min_mask] = frac * qs[i + mask.obj_max_mask] * IMAGE_I;
                qs[i + mask.obj_max_mask] = frac * tmp * IMAGE_I;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto tmp = qs[i + mask.obj_min_mask];
                    qs[i + mask.obj_min_mask] = frac * qs[i + mask.obj_max_mask] * IMAGE_I;
                    qs[i + mask.obj_max_mask] = frac * tmp * IMAGE_I;
                }
            })
    }
}

void CPUVectorPolicyBase::ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val) * IMAGE_MI;
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val) * IMAGE_MI;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = c * qs[i] + s * qs[m];
                auto v01 = c * qs[j] + s * qs[k];
                auto v10 = c * qs[k] + s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v00 = c * qs[i] + s * qs[m];
                    auto v01 = c * qs[j] + s * qs[k];
                    auto v10 = c * qs[k] + s * qs[j];
                    auto v11 = c * qs[m] + s * qs[i];
                    qs[i] = v00;
                    qs[j] = v01;
                    qs[k] = v10;
                    qs[m] = v11;
                }
            })
        if (diff) {
            CPUVectorPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

void CPUVectorPolicyBase::ApplyYY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val) * IMAGE_I;
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val) * IMAGE_I;
    }
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto v00 = c * qs[i] + s * qs[m];
                auto v01 = c * qs[j] - s * qs[k];
                auto v10 = c * qs[k] - s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto v00 = c * qs[i] + s * qs[m];
                    auto v01 = c * qs[j] - s * qs[k];
                    auto v10 = c * qs[k] - s * qs[j];
                    auto v11 = c * qs[m] + s * qs[i];
                    qs[i] = v00;
                    qs[j] = v01;
                    qs[k] = v10;
                    qs[m] = v11;
                }
            })
        if (diff) {
            CPUVectorPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

void CPUVectorPolicyBase::ApplyZZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = std::cos(val);
    auto s = std::sin(val);
    if (diff) {
        c = -std::sin(val);
        s = std::cos(val);
    }
    auto e = c + IMAGE_I * s;
    auto me = c + IMAGE_MI * s;
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                index_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto m = i + mask.obj_mask;
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                qs[i] *= me;
                qs[j] *= e;
                qs[k] *= e;
                qs[m] *= me;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                index_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto m = i + mask.obj_mask;
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    qs[i] *= me;
                    qs[j] *= e;
                    qs[k] *= e;
                    qs[m] *= me;
                }
            })
        if (diff) {
            CPUVectorPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

auto CPUVectorPolicyBase::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, qs_data_p_t vec,
                                    index_t dim) -> qs_data_p_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto out = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, reinterpret_cast<calc_type*>(vec));
    return reinterpret_cast<qs_data_p_t>(out);
}
auto CPUVectorPolicyBase::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                    const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b, qs_data_p_t vec,
                                    index_t dim) -> qs_data_p_t {
    if ((dim != a->dim_) || (dim != b->dim_)) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto out = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, b, reinterpret_cast<calc_type*>(vec));
    return reinterpret_cast<qs_data_p_t>(out);
}
}  // namespace mindquantum::sim::vector::detail
