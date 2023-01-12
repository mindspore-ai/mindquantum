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
#ifndef INCLUDE_VECTOR_DETAIL_GPU_VECTOR_POLICY_TPP
#define INCLUDE_VECTOR_DETAIL_GPU_VECTOR_POLICY_TPP
#include <cmath>

#include <cassert>
#include <complex>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include <thrust/transform_reduce.h>

#include "core/sparse/algo.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {
template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    qs_data_p_t qs;
    cudaMalloc((void**) &qs, sizeof(qs_data_t) * dim);  // NOLINT
    cudaMemset(qs, 0, sizeof(qs_data_t) * dim);
    if (zero_state) {
        qs_data_t one = qs_data_t(1.0, 0.0);
        cudaMemcpy(qs, &one, sizeof(qs_data_t), cudaMemcpyHostToDevice);
    }
    return qs;
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        cudaFree(qs);
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::Reset(qs_data_p_t qs, index_t dim) {
    cudaMemset(qs, 0, sizeof(qs_data_t) * dim);
    qs_data_t one(1, 0);
    cudaMemcpy(qs, &one, sizeof(qs_data_t), cudaMemcpyHostToDevice);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
    if (n_qubits > q_limit) {
        n_qubits = q_limit;
    }
    std::complex<calc_type>* h_qs = reinterpret_cast<std::complex<calc_type>*>(
        malloc((1UL << n_qubits) * sizeof(std::complex<calc_type>)));
    cudaMemcpy(h_qs, qs, sizeof(qs_data_t) * (1UL << n_qubits), cudaMemcpyDeviceToHost);
    std::cout << n_qubits << " qubits cpu simulator (little endian)." << std::endl;
    for (index_t i = 0; i < (1UL << n_qubits); i++) {
        std::cout << "(" << h_qs[i].real() << ", " << h_qs[i].imag() << ")" << std::endl;
    }
    free(h_qs);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    thrust::counting_iterator<index_t> i(0);
    thrust::for_each(i, i + dim, [=] __device__(index_t i) {
        if ((i & ctrl_mask) != ctrl_mask) {
            qs[i] = 0;
        }
    });
}

template <typename calc_type_>
template <index_t mask, index_t condi, class binary_op>
void GPUVectorPolicyBase<calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff,
                                                        qs_data_t fail_coeff, index_t dim, const binary_op& op) {
    thrust::counting_iterator<size_t> i(0);
    thrust::for_each(i, i + dim, [=] __device__(size_t i) {
        if ((i & mask) == condi) {
            des[i] = op(src[i], succ_coeff);
        } else {
            des[i] = op(src[i], fail_coeff);
        }
    });
}

template <typename calc_type_>
template <class binary_op>
void GPUVectorPolicyBase<calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                        qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim,
                                                        const binary_op& op) {
    thrust::counting_iterator<size_t> i(0);
    thrust::for_each(i, i + dim, [=] __device__(size_t i) {
        if ((i & mask) == condi) {
            des[i] = op(src[i], succ_coeff);
        } else {
            des[i] = op(src[i], fail_coeff);
        }
    });
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                     qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::plus<qs_data_t>());
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                       qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::minus<qs_data_t>());
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                     qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::multiplies<qs_data_t>());
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                                     qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::divides<qs_data_t>());
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim) {
    ConditionalBinary<0, 0>(src, des, value, 0, dim, thrust::multiplies<qs_data_t>());
}
template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, bool abs,
                                                         index_t dim) -> qs_data_t {
    qs_data_t res = 0;
    thrust::counting_iterator<size_t> l(0);
    if (abs) {
        res = thrust::transform_reduce(
            l, l + dim,
            [=] __device__(size_t l) {
                if ((l & mask) == condi) {
                    return thrust::conj(qs[l]) * qs[l];
                }
                return qs_data_t(0.0, 0.0);
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else {
        res = thrust::transform_reduce(
            l, l + dim,
            [=] __device__(size_t l) {
                if ((l & mask) == condi) {
                    return thrust::conj(qs[l]) * qs[l];
                }
                return qs_data_t(0.0, 0.0);
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return res;
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::GetQS(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    py_qs_datas_t out(dim);
    cudaMemcpy(out.data(), qs, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    return out;
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    cudaMemcpy(qs, qs_out.data(), sizeof(qs_data_t) * dim, cudaMemcpyHostToDevice);
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham,
                                                 index_t dim) -> qs_data_p_t {
    qs_data_p_t out = GPUVectorPolicyBase::InitState(dim, false);
    for (const auto& [pauli_string, coeff] : ham) {
        auto mask = GenPauliMask(pauli_string);
        auto mask_f = mask.mask_x | mask.mask_y;
        if (dim >= (1UL << 18)) {
            auto mask_z = mask.mask_z;
            auto mask_y = mask.mask_y;
            auto num_y = mask.num_y;
            auto this_coeff = qs_data_t(coeff);
            thrust::counting_iterator<index_t> i(0);
            thrust::for_each(i, i + dim, [=] __device__(index_t i) {
                auto j = (i ^ mask_f);
                if (i <= j) {
                    auto axis2power = __popcll(i & mask_z);
                    auto axis3power = __popcll(i & mask_y);
                    auto idx = (num_y + 2 * axis3power + 2 * axis2power) & 3;
                    auto c = GPUVectorPolicyBase::qs_data_t(1, 0);
                    if (idx == 1) {
                        c = GPUVectorPolicyBase::qs_data_t(0, 1);
                    } else if (idx == 2) {
                        c = GPUVectorPolicyBase::qs_data_t(-1, 0);
                    } else if (idx == 3) {
                        c = GPUVectorPolicyBase::qs_data_t(0, -1);
                    }
                    out[j] += qs[i] * this_coeff * c;
                    if (i != j) {
                        out[i] += qs[j] * this_coeff / c;
                    }
                }
            });
        } else {
            ApplyTerm<qs_data_t, qs_data_p_t, calc_type>
                <<<128, 128>>>(out, qs, coeff, mask.num_y, mask.mask_y, mask.mask_z, mask_f, dim);
        }
    }
    return out;
};
template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out;
    cudaMalloc((void**) &out, sizeof(qs_data_t) * dim);  // NOLINT
    cudaMemcpy(out, qs, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToDevice);
    return out;
};

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::Vdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
    thrust::device_ptr<qs_data_t> dev_bra(bra);
    thrust::device_ptr<qs_data_t> dev_ket(ket);
    qs_data_t res = thrust::inner_product(dev_bra, dev_bra + dim, dev_ket, qs_data_t(0, 0), thrust::plus<qs_data_t>(),
                                          conj_a_dot_b<qs_data_t>());
    return res;
}

template <typename calc_type_>
template <index_t mask, index_t condi>
auto GPUVectorPolicyBase<calc_type_>::ConditionVdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
    thrust::counting_iterator<size_t> i(0);
    return thrust::transform_reduce(
        i, i + dim,
        [=] __device__(size_t i) {
            if ((i & mask) == condi) {
                return thrust::conj(bra[i]) * ket[i];
            } else {
                return qs_data_t(0, 0);
            }
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::OneStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
    -> py_qs_data_t {
    SingleQubitGateMask mask({obj_qubit}, {});
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    thrust::counting_iterator<size_t> l(0);
    return thrust::transform_reduce(
        l, l + dim / 2,
        [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask) + obj_mask;
            return thrust::conj(bra[i]) * ket[i];
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ZeroStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
    -> py_qs_data_t {
    SingleQubitGateMask mask({obj_qubit}, {});
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    thrust::counting_iterator<size_t> l(0);
    return thrust::transform_reduce(
        l, l + dim / 2,
        [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            return thrust::conj(bra[i]) * ket[i];
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto tmp = qs[i + obj_min_mask];
            qs[i + obj_min_mask] = qs[i + obj_max_mask];
            qs[i + obj_max_mask] = tmp;
        });
    } else {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto tmp = qs[i + obj_min_mask];
                qs[i + obj_min_mask] = qs[i + obj_max_mask];
                qs[i + obj_max_mask] = tmp;
            }
        });
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 bool daggered, index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    calc_type frac = 1.0;
    if (daggered) {
        frac = -1.0;
    }
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto tmp = qs[i + obj_min_mask];
            qs[i + obj_min_mask] = frac * qs[i + obj_max_mask] * qs_data_t(0, 1);
            qs[i + obj_max_mask] = frac * tmp * qs_data_t(0, 1);
        });
    } else {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto tmp = qs[i + obj_min_mask];
                qs[i + obj_min_mask] = frac * qs[i + obj_max_mask] * qs_data_t(0, 1);
                qs[i + obj_max_mask] = frac * tmp * qs_data_t(0, 1);
            }
        });
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    auto obj_mask = mask.obj_mask;
    auto c = static_cast<qs_data_t>(std::cos(val));
    auto s = static_cast<calc_type>(std::sin(val)) * qs_data_t(0, -1);
    if (diff) {
        c = static_cast<qs_data_t>(-std::sin(val));
        s = static_cast<calc_type>(std::cos(val)) * qs_data_t(0, -1);
    }
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto m = i + obj_mask;
            auto v00 = c * qs[i] + s * qs[m];
            auto v01 = c * qs[j] + s * qs[k];
            auto v10 = c * qs[k] + s * qs[j];
            auto v11 = c * qs[m] + s * qs[i];
            qs[i] = v00;
            qs[j] = v01;
            qs[k] = v10;
            qs[m] = v11;
        });
    } else {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto m = i + obj_mask;
                auto v00 = c * qs[i] + s * qs[m];
                auto v01 = c * qs[j] + s * qs[k];
                auto v10 = c * qs[k] + s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            }
        });
        if (diff) {
            GPUVectorPolicyBase::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyYY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    auto obj_mask = mask.obj_mask;
    auto c = static_cast<qs_data_t>(std::cos(val));
    auto s = static_cast<calc_type>(std::sin(val)) * qs_data_t(0, 1);
    if (diff) {
        c = static_cast<qs_data_t>(-std::sin(val));
        s = static_cast<qs_data_t>(std::cos(val)) * qs_data_t(0, 1);
    }
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto m = i + obj_mask;
            auto v00 = c * qs[i] + s * qs[m];
            auto v01 = c * qs[j] - s * qs[k];
            auto v10 = c * qs[k] - s * qs[j];
            auto v11 = c * qs[m] + s * qs[i];
            qs[i] = v00;
            qs[j] = v01;
            qs[k] = v10;
            qs[m] = v11;
        });
    } else {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) & ctrl_mask) {
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto m = i + obj_mask;
                auto v00 = c * qs[i] + s * qs[m];
                auto v01 = c * qs[j] - s * qs[k];
                auto v10 = c * qs[k] - s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            }
        });
        if (diff) {
            GPUVectorPolicyBase::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyZZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    DoubleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    auto obj_mask = mask.obj_mask;
    auto c = static_cast<calc_type>(std::cos(val));
    auto s = static_cast<calc_type>(std::sin(val));
    if (diff) {
        c = static_cast<calc_type>(-std::sin(val));
        s = static_cast<calc_type>(std::cos(val));
    }
    auto e = c + qs_data_t(0, 1) * s;
    auto me = c + qs_data_t(0, -1) * s;
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto m = i + obj_mask;
            qs[i] *= me;
            qs[j] *= e;
            qs[k] *= e;
            qs[m] *= me;
        });
    } else {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto m = i + obj_mask;
                qs[i] *= me;
                qs[j] *= e;
                qs[k] *= e;
                qs[m] *= me;
            }
        });
        if (diff) {
            GPUVectorPolicyBase::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}
template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto host = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host, vec, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto host_res = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, reinterpret_cast<calc_type*>(host));
    auto out = InitState(dim);
    cudaMemcpy(out, reinterpret_cast<std::complex<calc_type>*>(host_res), sizeof(qs_data_t) * dim,
               cudaMemcpyHostToDevice);
    if (host != nullptr) {
        free(host);
    }
    if (host_res != nullptr) {
        free(host_res);
    }
    return out;
}
template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b,
                                                qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if ((dim != a->dim_) || (dim != b->dim_)) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto host = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host, vec, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto host_res = sparse::Csr_Dot_Vec<calc_type, calc_type>(a, b, reinterpret_cast<calc_type*>(host));
    auto out = InitState(dim);
    cudaMemcpy(out, reinterpret_cast<std::complex<calc_type>*>(host_res), sizeof(qs_data_t) * dim,
               cudaMemcpyHostToDevice);
    if (host != nullptr) {
        free(host);
    }
    if (host_res != nullptr) {
        free(host_res);
    }
    return out;
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyZLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 qs_data_t val, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask) + obj_mask;
            qs[i] *= val;
        });
    } else {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask) + obj_mask;
            if ((i & ctrl_mask) == ctrl_mask) {
                qs[i] *= val;
            }
        });
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyZLike(qs, objs, ctrls, -1, dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplySGate(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(0, 1), dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplySdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(0, -1), dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyT(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(1, 1) / std::sqrt(2.0), dim);
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyTdag(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                index_t dim) {
    ApplyZLike(qs, objs, ctrls, qs_data_t(1, -1) / std::sqrt(2.0), dim);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyPS(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    if (!diff) {
        ApplyZLike(qs, objs, ctrls, qs_data_t(std::cos(val), std::sin(val)), dim);
    } else {
        SingleQubitGateMask mask(objs, ctrls);
        thrust::counting_iterator<index_t> l(0);
        auto obj_high_mask = mask.obj_high_mask;
        auto obj_low_mask = mask.obj_low_mask;
        auto obj_mask = mask.obj_mask;
        auto ctrl_mask = mask.ctrl_mask;
        qs_data_t e = qs_data_t(-std::sin(val), std::cos(val));
        if (!mask.ctrl_mask) {
            if (!mask.ctrl_mask) {
                thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
                    auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                    auto j = i + obj_mask;
                    qs[i] = 0;
                    qs[j] *= e;
                });
            } else {
                thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
                    auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                    if ((i & ctrl_mask) == ctrl_mask) {
                        auto j = i + obj_mask;
                        qs[i] = 0;
                        qs[j] *= e;
                    }
                });
            }
            GPUVectorPolicyBase::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyXLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                 qs_data_t v1, qs_data_t v2, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    thrust::counting_iterator<index_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            auto j = i + obj_mask;
            auto tmp = qs[i];
            qs[i] = qs[j] * v1;
            qs[j] = tmp * v2;
        });
    } else {
        thrust::for_each(l, l + dim / 2, [=] __device__(index_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + obj_mask;
                auto tmp = qs[i];
                qs[i] = qs[j] * v1;
                qs[j] = tmp * v2;
            }
        });
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, 1, 1, dim);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, qs_data_t(0, -1), qs_data_t(0, 1), dim);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyTwoQubitsMatrix(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                                           const qbits_t& ctrls,
                                                           const std::vector<std::vector<py_qs_data_t>>& m,
                                                           index_t dim) {
    DoubleQubitGateMask mask(objs, ctrls);
    qs_data_t m00 = m[0][0];
    qs_data_t m01 = m[0][1];
    qs_data_t m02 = m[0][2];
    qs_data_t m03 = m[0][3];
    qs_data_t m10 = m[1][0];
    qs_data_t m11 = m[1][1];
    qs_data_t m12 = m[1][2];
    qs_data_t m13 = m[1][3];
    qs_data_t m20 = m[2][0];
    qs_data_t m21 = m[2][1];
    qs_data_t m22 = m[2][2];
    qs_data_t m23 = m[2][3];
    qs_data_t m30 = m[3][0];
    qs_data_t m31 = m[3][1];
    qs_data_t m32 = m[3][2];
    qs_data_t m33 = m[3][3];
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    auto obj_mask = mask.obj_mask;
    thrust::counting_iterator<size_t> l(0);
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + (dim / 4), [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto m = i + obj_mask;
            auto v00 = m00 * src[i] + m01 * src[j] + m02 * src[k] + m03 * src[m];
            auto v01 = m10 * src[i] + m11 * src[j] + m12 * src[k] + m13 * src[m];
            auto v10 = m20 * src[i] + m21 * src[j] + m22 * src[k] + m23 * src[m];
            auto v11 = m30 * src[i] + m31 * src[j] + m32 * src[k] + m33 * src[m];
            src[i] = v00;
            src[j] = v01;
            src[k] = v10;
            src[m] = v11;
        });
    } else {
        thrust::for_each(l, l + (dim / 4), [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto m = i + obj_mask;
                auto v00 = m00 * src[i] + m01 * src[j] + m02 * src[k] + m03 * src[m];
                auto v01 = m10 * src[i] + m11 * src[j] + m12 * src[k] + m13 * src[m];
                auto v10 = m20 * src[i] + m21 * src[j] + m22 * src[k] + m23 * src[m];
                auto v11 = m30 * src[i] + m31 * src[j] + m32 * src[k] + m33 * src[m];
                src[i] = v00;
                src[j] = v01;
                src[k] = v10;
                src[m] = v11;
            }
        });
    }
}
template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplySingleQubitMatrix(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                             const qbits_t& ctrls,
                                                             const std::vector<std::vector<py_qs_data_t>>& m,
                                                             index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, ctrls);
    qs_data_t m00 = m[0][0];
    qs_data_t m01 = m[0][1];
    qs_data_t m10 = m[1][0];
    qs_data_t m11 = m[1][1];
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    thrust::counting_iterator<size_t> l(0);
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + (dim / 2), [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            auto j = i + obj_mask;
            auto a = m00 * src[i] + m01 * src[j];
            auto b = m10 * src[i] + m11 * src[j];
            des[i] = a;
            des[j] = b;
        });
    } else {
        thrust::for_each(l, l + (dim / 2), [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            if ((i & ctrl_mask) == ctrl_mask) {
                auto j = i + obj_mask;
                auto a = m00 * src[i] + m01 * src[j];
                auto b = m10 * src[i] + m11 * src[j];
                des[i] = a;
                des[j] = b;
            }
        });
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = static_cast<calc_type>(std::cos(val / 2));
    auto b = static_cast<calc_type>(-std::sin(val / 2));
    if (diff) {
        a = static_cast<calc_type>(-0.5 * std::sin(val / 2));
        b = static_cast<calc_type>(-0.5 * std::cos(val / 2));
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyMatrixGate(qs_data_p_t src, qs_data_p_t des, const qbits_t& objs,
                                                      const qbits_t& ctrls,
                                                      const std::vector<std::vector<py_qs_data_t>>& m, index_t dim) {
    if (objs.size() == 1) {
        ApplySingleQubitMatrix(src, des, objs[0], ctrls, m, dim);
    } else if (objs.size() == 2) {
        ApplyTwoQubitsMatrix(src, des, objs, ctrls, m, dim);
    } else {
        throw std::runtime_error("Can not custom " + std::to_string(objs.size()) + " qubits gate for gpu backend.");
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = static_cast<calc_type>(std::cos(val / 2));
    auto b = static_cast<calc_type>(std::sin(val / 2));
    if (diff) {
        a = static_cast<calc_type>(-0.5 * std::sin(val / 2));
        b = static_cast<calc_type>(0.5 * std::cos(val / 2));
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    std::vector<std::vector<py_qs_data_t>> m{{M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyGP(qs_data_p_t qs, qbit_t obj_qubit, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    auto c = std::exp(std::complex<calc_type>(0, -val));
    std::vector<std::vector<py_qs_data_t>> m = {{c, 0}, {0, c}};
    ApplySingleQubitMatrix(qs, qs, obj_qubit, ctrls, m, dim);
}

template <typename calc_type_>
void GPUVectorPolicyBase<calc_type_>::ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val,
                                              index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = static_cast<calc_type>(std::cos(val / 2));
    auto b = static_cast<calc_type>(std::sin(val / 2));
    if (diff) {
        a = static_cast<calc_type>(-0.5 * std::sin(val / 2));
        b = static_cast<calc_type>(0.5 * std::cos(val / 2));
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffTwoQubitsMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                                const qbits_t& ctrls,
                                                                const std::vector<py_qs_datas_t>& m, index_t dim)
    -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    qs_data_t m00 = m[0][0];
    qs_data_t m01 = m[0][1];
    qs_data_t m02 = m[0][2];
    qs_data_t m03 = m[0][3];
    qs_data_t m10 = m[1][0];
    qs_data_t m11 = m[1][1];
    qs_data_t m12 = m[1][2];
    qs_data_t m13 = m[1][3];
    qs_data_t m20 = m[2][0];
    qs_data_t m21 = m[2][1];
    qs_data_t m22 = m[2][2];
    qs_data_t m23 = m[2][3];
    qs_data_t m30 = m[3][0];
    qs_data_t m31 = m[3][1];
    qs_data_t m32 = m[3][2];
    qs_data_t m33 = m[3][3];
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        return thrust::transform_reduce(
            l, l + dim / 4,
            [=] __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = m00 * ket[i] + m01 * ket[j] + m02 * ket[k] + m03 * ket[m];
                auto v01 = m10 * ket[i] + m11 * ket[j] + m12 * ket[k] + m13 * ket[m];
                auto v10 = m20 * ket[i] + m21 * ket[j] + m22 * ket[k] + m23 * ket[m];
                auto v11 = m30 * ket[i] + m31 * ket[j] + m32 * ket[k] + m33 * ket[m];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return thrust::transform_reduce(
        l, l + dim / 4,
        [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) != ctrl_mask) {
                return qs_data_t(0, 0);
            }
            auto m = i + obj_mask;
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto v00 = m00 * ket[i] + m01 * ket[j] + m02 * ket[k] + m03 * ket[m];
            auto v01 = m10 * ket[i] + m11 * ket[j] + m12 * ket[k] + m13 * ket[m];
            auto v10 = m20 * ket[i] + m21 * ket[j] + m22 * ket[k] + m23 * ket[m];
            auto v11 = m30 * ket[i] + m31 * ket[j] + m32 * ket[k] + m33 * ket[m];
            auto this_res = thrust::conj(bra[i]) * v00;
            this_res += thrust::conj(bra[j]) * v01;
            this_res += thrust::conj(bra[k]) * v10;
            this_res += thrust::conj(bra[m]) * v11;
            return this_res;
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffSingleQubitMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                                  const qbits_t& ctrls,
                                                                  const std::vector<py_qs_datas_t>& m, index_t dim)
    -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    qs_data_t m00 = m[0][0];
    qs_data_t m01 = m[0][1];
    qs_data_t m10 = m[1][0];
    qs_data_t m11 = m[1][1];
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        return thrust::transform_reduce(
            l, l + dim / 2,
            [=] __device__(size_t l) {
                auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                auto j = i + obj_mask;
                auto t1 = m00 * ket[i] + m01 * ket[j];
                auto t2 = m10 * ket[i] + m11 * ket[j];
                auto this_res = thrust::conj(bra[i]) * t1 + thrust::conj(bra[j]) * t2;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else if (mask.ctrl_qubits.size() == 1) {
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
        return thrust::transform_reduce(
            l, l + dim / 4,
            [=] __device__(size_t l) {
                auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                i = ((i & second_high_mask) << 1) + (i & second_low_mask) + ctrl_mask;
                auto j = i + obj_mask;
                auto t1 = m00 * ket[i] + m01 * ket[j];
                auto t2 = m10 * ket[i] + m11 * ket[j];
                auto this_res = thrust::conj(bra[i]) * t1 + thrust::conj(bra[j]) * t2;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return thrust::transform_reduce(
        l, l + dim / 2,
        [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            if ((i & ctrl_mask) != ctrl_mask) {
                return qs_data_t(0, 0);
            }
            auto j = i + obj_mask;
            auto t1 = m00 * ket[i] + m01 * ket[j];
            auto t2 = m10 * ket[i] + m11 * ket[j];
            auto this_res = thrust::conj(bra[i]) * t1 + thrust::conj(bra[j]) * t2;
            return this_res;
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffMatrixGate(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                           const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m,
                                                           index_t dim) -> qs_data_t {
    if (objs.size() == 1) {
        return ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
    }
    if (objs.size() == 2) {
        return ExpectDiffTwoQubitsMatrix(bra, ket, objs, ctrls, m, dim);
    }
    throw std::runtime_error("Expectation of " + std::to_string(objs.size()) + " not implement for gpu backend.");
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto is = static_cast<calc_type>(0.5 * std::cos(val / 2)) * qs_data_t(0, -1);
    std::vector<py_qs_datas_t> gate = {{c, is}, {is, c}};
    return GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto s = static_cast<calc_type>(0.5 * std::cos(val / 2));
    std::vector<py_qs_datas_t> gate = {{c, -s}, {s, c}};
    return GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-0.5 * std::sin(val / 2));
    auto s = static_cast<calc_type>(0.5 * std::cos(val / 2));
    auto e0 = c + qs_data_t(0, -1) * s;
    auto e1 = c + qs_data_t(0, 1) * s;
    std::vector<py_qs_datas_t> gate = {{e0, 0}, {0, e1}};
    return GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffXX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val));
    auto s = static_cast<calc_type>(std::cos(val)) * qs_data_t(0, -1);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        return thrust::transform_reduce(
            l, l + dim / 4,
            [=] __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[m];
                auto v01 = c * ket[j] + s * ket[k];
                auto v10 = c * ket[k] + s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return thrust::transform_reduce(
        l, l + dim / 4,
        [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) != ctrl_mask) {
                return qs_data_t(0, 0);
            }
            auto m = i + obj_mask;
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto v00 = c * ket[i] + s * ket[m];
            auto v01 = c * ket[j] + s * ket[k];
            auto v10 = c * ket[k] + s * ket[j];
            auto v11 = c * ket[m] + s * ket[i];
            auto this_res = thrust::conj(bra[i]) * v00;
            this_res += thrust::conj(bra[j]) * v01;
            this_res += thrust::conj(bra[k]) * v10;
            this_res += thrust::conj(bra[m]) * v11;
            return this_res;
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffYY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val));
    auto s = static_cast<calc_type>(std::cos(val)) * qs_data_t(0, 1);
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        return thrust::transform_reduce(
            l, l + dim / 4,
            [=] __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto v00 = c * ket[i] + s * ket[m];
                auto v01 = c * ket[j] + s * ket[k];
                auto v10 = c * ket[k] + s * ket[j];
                auto v11 = c * ket[m] + s * ket[i];
                auto this_res = thrust::conj(bra[i]) * v00;
                this_res += thrust::conj(bra[j]) * v01;
                this_res += thrust::conj(bra[k]) * v10;
                this_res += thrust::conj(bra[m]) * v11;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return thrust::transform_reduce(
        l, l + dim / 4,
        [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) != ctrl_mask) {
                return qs_data_t(0, 0);
            }
            auto m = i + obj_mask;
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto v00 = c * ket[i] + s * ket[m];
            auto v01 = c * ket[j] + s * ket[k];
            auto v10 = c * ket[k] + s * ket[j];
            auto v11 = c * ket[m] + s * ket[i];
            auto this_res = thrust::conj(bra[i]) * v00;
            this_res += thrust::conj(bra[j]) * v01;
            this_res += thrust::conj(bra[k]) * v10;
            this_res += thrust::conj(bra[m]) * v11;
            return this_res;
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffZZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = static_cast<calc_type>(-std::sin(val));
    auto s = static_cast<calc_type>(std::cos(val));
    auto e = c + qs_data_t(0, 1) * s;
    auto me = c + qs_data_t(0, -1) * s;
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_rev_high_mask = mask.obj_rev_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_rev_low_mask = mask.obj_rev_low_mask;
    auto obj_mask = mask.obj_mask;
    auto obj_min_mask = mask.obj_min_mask;
    auto obj_max_mask = mask.obj_max_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!mask.ctrl_mask) {
        return thrust::transform_reduce(
            l, l + dim / 4,
            [=] __device__(size_t l) {
                index_t i;
                SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
                auto m = i + obj_mask;
                auto j = i + obj_min_mask;
                auto k = i + obj_max_mask;
                auto this_res = thrust::conj(bra[i]) * ket[i] * me;
                this_res += thrust::conj(bra[j]) * ket[j] * e;
                this_res += thrust::conj(bra[k]) * ket[k] * e;
                this_res += thrust::conj(bra[m]) * ket[m] * me;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return thrust::transform_reduce(
        l, l + dim / 4,
        [=] __device__(size_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            if ((i & ctrl_mask) != ctrl_mask) {
                return qs_data_t(0, 0);
            }
            auto m = i + obj_mask;
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto this_res = thrust::conj(bra[i]) * ket[i] * me;
            this_res += thrust::conj(bra[j]) * ket[j] * e;
            this_res += thrust::conj(bra[k]) * ket[k] * e;
            this_res += thrust::conj(bra[m]) * ket[m] * me;
            return this_res;
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffGP(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto e = std::complex<calc_type>(0, -1);
    e *= std::exp(std::complex<calc_type>(0, -val));
    std::vector<py_qs_datas_t> gate = {{e, 0}, {0, e}};
    return GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, gate, dim);
}

template <typename calc_type_>
auto GPUVectorPolicyBase<calc_type_>::ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                   const qbits_t& ctrls, calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto e = static_cast<calc_type>(std::cos(val)) + qs_data_t(0, 1) * static_cast<calc_type>(std::sin(val));
    thrust::counting_iterator<size_t> l(0);
    auto obj_high_mask = mask.obj_high_mask;
    auto obj_low_mask = mask.obj_low_mask;
    auto obj_mask = mask.obj_mask;
    auto ctrl_mask = mask.ctrl_mask;
    if (!(mask.ctrl_mask)) {
        return thrust::transform_reduce(
            l, l + dim / 2,
            [=] __device__(size_t l) {
                auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
                auto j = i + obj_mask;
                auto this_res = thrust::conj(bra[j]) * ket[j] * e;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    } else if (mask.ctrl_qubits.size() == 1) {
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
        return thrust::transform_reduce(
            l, l + dim / 4,
            [=] __device__(size_t l) {
                auto i = ((l & first_high_mask) << 1) + (l & first_low_mask);
                i = ((i & second_high_mask) << 1) + (i & second_low_mask) + ctrl_mask;
                auto j = i + obj_mask;
                auto this_res = thrust::conj(bra[j]) * ket[j] * e;
                return this_res;
            },
            qs_data_t(0, 0), thrust::plus<qs_data_t>());
    }
    return thrust::transform_reduce(
        l, l + dim / 2,
        [=] __device__(size_t l) {
            auto i = ((l & obj_high_mask) << 1) + (l & obj_low_mask);
            if ((i & ctrl_mask) != ctrl_mask) {
                return qs_data_t(0, 0);
            }
            auto j = i + obj_mask;
            auto this_res = thrust::conj(bra[j]) * ket[j] * e;
            return this_res;
        },
        qs_data_t(0, 0), thrust::plus<qs_data_t>());
}

}  // namespace mindquantum::sim::vector::detail

#endif
