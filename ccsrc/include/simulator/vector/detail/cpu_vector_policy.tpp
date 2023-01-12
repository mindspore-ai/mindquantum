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
#ifndef INCLUDE_VECTOR_DETAIL_CPU_VECTOR_POLICY_TPP
#define INCLUDE_VECTOR_DETAIL_CPU_VECTOR_POLICY_TPP
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
#include "config/type_promotion.hpp"

#include "core/sparse/algo.hpp"
#include "core/utils.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/cpu_vector_policy.hpp"

namespace mindquantum::sim::vector::detail {
template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    auto qs = reinterpret_cast<qs_data_p_t>(calloc(dim, sizeof(qs_data_t)));
    if (zero_state) {
        qs[0] = 1;
    }
    return qs;
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::Reset(qs_data_p_t qs, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = 0; })
    qs[0] = 1;
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        free(qs);
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < (1UL << n_qubits); i++) {
        std::cout << "(" << qs[i].real() << ", " << qs[i].imag() << ")" << std::endl;
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & ctrl_mask) != ctrl_mask) {
                qs[i] = 0;
            }
        })
}

template <typename calc_type_>
template <class binary_op>
void CPUVectorPolicyBase<calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                        qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim,
                                                        const binary_op& op) {
    // if index mask satisfied condition, multiply by success_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & mask) == condi) {
                des[i] = op(src[i], succ_coeff);
            } else {
                des[i] = op(src[i], fail_coeff);
            }
        })
}

template <typename calc_type_>
template <index_t mask, index_t condi, class binary_op>
void CPUVectorPolicyBase<calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff,
                                                        qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    // if index mask satisfied condition, multiply by success_coeff, otherwise multiply fail_coeff
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) {
            if ((i & mask) == condi) {
                des[i] = op(src[i], succ_coeff);
            } else {
                des[i] = op(src[i], fail_coeff);
            }
        })
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim) {
    ConditionalBinary<0, 0>(src, des, value, 0, dim, std::multiplies<qs_data_t>());
}
template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                     qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::plus<qs_data_t>());
}
template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                       qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::minus<qs_data_t>());
}
template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                     qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::multiplies<qs_data_t>());
}
template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                     qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, std::divides<qs_data_t>());
}

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, bool abs,
                                                         index_t dim) -> qs_data_t {
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out = CPUVectorPolicyBase::InitState(dim, false);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { out[i] = qs[i]; })
    return out;
};

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::Vdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
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

template <typename calc_type_>
template <index_t mask, index_t condi>
auto CPUVectorPolicyBase<calc_type_>::ConditionVdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::OneStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ZeroStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::GetQS(qs_data_p_t qs, index_t dim) -> VT<py_qs_data_t> {
    VT<py_qs_data_t> out(dim);
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { out[i] = qs[i]; })
    return out;
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::SetQS(qs_data_p_t qs, const VT<py_qs_data_t>& qs_out, index_t dim) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    THRESHOLD_OMP_FOR(
        dim, DimTh, for (omp::idx_t i = 0; i < dim; i++) { qs[i] = qs_out[i]; })
}

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham,
                                                 index_t dim) -> qs_data_p_t {
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
                    auto c = ComplexCast<double, calc_type>::apply(
                        POLAR[static_cast<char>((mask.num_y + 2 * axis3power + 2 * axis2power) & 3)]);
                    out[j] += qs[i] * coeff * c;
                    if (i != j) {
                        out[i] += qs[j] * coeff / c;
                    }
                }
            })
    }
    return out;
};

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
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

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 bool daggered, index_t dim) {
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

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
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

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyYY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
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

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyZZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto out = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, reinterpret_cast<calc_type*>(vec));
    return reinterpret_cast<qs_data_p_t>(out);
}

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b,
                                                qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if ((dim != a->dim_) || (dim != b->dim_)) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto out = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, b, reinterpret_cast<calc_type*>(vec));
    return reinterpret_cast<qs_data_p_t>(out);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyZLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 qs_data_t val, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask) + mask.obj_mask;
                qs[i] *= val;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask) + mask.obj_mask;
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    qs[i] *= val;
                }
            })
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyZLike(qs, objs, ctrls, -1, dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplySGate(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(0, 1), dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplySdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(0, -1), dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyT(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(1, 1) / static_cast<calc_type>(std::sqrt(2.0)), dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyTdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(1, -1) / static_cast<calc_type>(std::sqrt(2.0)), dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyPS(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    if (!diff) {
        ApplyZLike(qs, objs, ctrls, qs_data_t(std::cos(val), std::sin(val)), dim);
    } else {
        SingleQubitGateMask mask(objs, ctrls);
        auto e = -std::sin(val) + IMAGE_I * std::cos(val);
        if (!mask.ctrl_mask) {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    auto j = i + mask.obj_mask;
                    qs[i] = 0;
                    qs[j] *= e;
                })
        } else {
            THRESHOLD_OMP_FOR(
                dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                    auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                    if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                        auto j = i + mask.obj_mask;
                        qs[i] = 0;
                        qs[j] *= e;
                    }
                })
            CPUVectorPolicyBase::SetToZeroExcept(qs, mask.ctrl_mask, dim);
        }
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyXLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 qs_data_t v1, qs_data_t v2, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i | mask.obj_mask;
                auto tmp = qs[i];
                qs[i] = qs[j] * v1;
                qs[j] = tmp * v2;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i | mask.obj_mask;
                    auto tmp = qs[i];
                    qs[i] = qs[j] * v1;
                    qs[j] = tmp * v2;
                }
            })
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, 1, 1, dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, IMAGE_MI, IMAGE_I, dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                                           const qbits_t& ctrls,
                                                           const std::vector<std::vector<py_qs_data_t>>& gate,
                                                           index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        // clang-format off
        THRESHOLD_OMP_FOR(dim, DimTh,
            for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                auto j = i + mask.obj_min_mask;
                auto k = i + mask.obj_max_mask;
                auto m = i + mask.obj_mask;
                auto v00 = gate[0][0] * src[i] + gate[0][1] * src[j] + gate[0][2] * src[k] + gate[0][3] * src[m];
                auto v01 = gate[1][0] * src[i] + gate[1][1] * src[j] + gate[1][2] * src[k] + gate[1][3] * src[m];
                auto v10 = gate[2][0] * src[i] + gate[2][1] * src[j] + gate[2][2] * src[k] + gate[2][3] * src[m];
                auto v11 = gate[3][0] * src[i] + gate[3][1] * src[j] + gate[3][2] * src[k] + gate[3][3] * src[m];
                des[i] = v00;
                des[j] = v01;
                des[k] = v10;
                des[m] = v11;
            })
        // clang-format on
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 4); l++) {
                omp::idx_t i;
                SHIFT_BIT_TWO(mask.obj_low_mask, mask.obj_rev_low_mask, mask.obj_high_mask, mask.obj_rev_high_mask, l,
                              i);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_min_mask;
                    auto k = i + mask.obj_max_mask;
                    auto m = i + mask.obj_mask;
                    auto v00 = gate[0][0] * src[i] + gate[0][1] * src[j] + gate[0][2] * src[k] + gate[0][3] * src[m];
                    auto v01 = gate[1][0] * src[i] + gate[1][1] * src[j] + gate[1][2] * src[k] + gate[1][3] * src[m];
                    auto v10 = gate[2][0] * src[i] + gate[2][1] * src[j] + gate[2][2] * src[k] + gate[2][3] * src[m];
                    auto v11 = gate[3][0] * src[i] + gate[3][1] * src[j] + gate[3][2] * src[k] + gate[3][3] * src[m];
                    des[i] = v00;
                    des[j] = v01;
                    des[k] = v10;
                    des[m] = v11;
                }
            })
    }
}
template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                             const qbits_t& ctrls,
                                                             const std::vector<std::vector<py_qs_data_t>>& m,
                                                             index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i + mask.obj_mask;
                auto t1 = m[0][0] * src[i] + m[0][1] * src[j];
                auto t2 = m[1][0] * src[i] + m[1][1] * src[j];
                des[i] = t1;
                des[j] = t2;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (omp::idx_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i + mask.obj_mask;
                    auto t1 = m[0][0] * src[i] + m[0][1] * src[j];
                    auto t2 = m[1][0] * src[i] + m[1][1] * src[j];
                    des[i] = t1;
                    des[j] = t2;
                }
            });
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                                      const qbits_t& ctrls,
                                                      const std::vector<std::vector<py_qs_data_t>>& m, index_t dim) {
    if (objs.size() == 1) {
        ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        throw std::runtime_error("Can not custom " + std::to_string(objs.size()) + " qubits gate for cpu backend.");
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    std::vector<std::vector<py_qs_data_t>> m{{M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyGP(qs_data_p_t qs, qbit_t obj_qubit, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    auto c = std::exp(std::complex<calc_type>(0, -val));
    std::vector<std::vector<py_qs_data_t>> m = {{c, 0}, {0, c}};
    ApplySingleQubitMatrix(qs, qs, obj_qubit, ctrls, m, dim);
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = -std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = -0.5 * std::cos(val / 2);
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename calc_type_>
void CPUVectorPolicyBase<calc_type_>::ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = std::cos(val / 2);
    auto b = std::sin(val / 2);
    if (diff) {
        a = -0.5 * std::sin(val / 2);
        b = 0.5 * std::cos(val / 2);
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffTwoQubitsMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                                const qbits_t& ctrls, const VVT<py_qs_data_t>& gate,
                                                                index_t dim) -> qs_data_t {
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffSingleQubitMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                                  const qbits_t& ctrls, const VVT<py_qs_data_t>& m,
                                                                  index_t dim) -> qs_data_t {
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffMatrixGate(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                           const qbits_t& ctrls, const VVT<py_qs_data_t>& m,
                                                           index_t dim) -> qs_data_t {
    if (objs.size() == 1) {
        return ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
    }
    if (objs.size() == 2) {
        return ExpectDiffTwoQubitsMatrix(bra, ket, objs, ctrls, m, dim);
    }
    throw std::runtime_error("Expectation of " + std::to_string(objs.size()) + " not implement for cpu backend.");
}

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto is = static_cast<calc_type>(0.5 * std::cos(val / 2)) * IMAGE_MI;
    VVT<py_qs_data_t> gate = {{c, is}, {is, c}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    calc_type c = -0.5 * std::sin(val / 2);
    calc_type s = 0.5 * std::cos(val / 2);
    VVT<py_qs_data_t> gate = {{c, -s}, {s, c}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    calc_type c = -0.5 * std::sin(val / 2);
    calc_type s = 0.5 * std::cos(val / 2);
    auto e0 = c + IMAGE_MI * s;
    auto e1 = c + IMAGE_I * s;
    VVT<py_qs_data_t> gate = {{e0, 0}, {0, e1}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
};

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffGP(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto e = std::complex<calc_type>(0, -1);
    e *= std::exp(std::complex<calc_type>(0, -val));
    VVT<py_qs_data_t> gate = {{e, 0}, {0, e}};
    return CPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffXX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val) * IMAGE_MI;
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffYY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val) * IMAGE_I;
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

template <typename calc_type_>
auto CPUVectorPolicyBase<calc_type_>::ExpectDiffZZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);

    auto c = -std::sin(val);
    auto s = std::cos(val);
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
}  // namespace mindquantum::sim::vector::detail
#endif
