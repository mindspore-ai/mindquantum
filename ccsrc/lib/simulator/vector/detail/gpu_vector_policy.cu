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
#include <cstdlib>
#include <stdexcept>

#include <thrust/transform_reduce.h>

#include "core/sparse/algo.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {
auto GPUVectorPolicyBase::InitState(index_t dim, bool zero_state) -> qs_data_p_t {
    qs_data_p_t qs;
    cudaMalloc((void**) &qs, sizeof(qs_data_t) * dim);  // NOLINT
    cudaMemset(qs, 0, sizeof(qs_data_t) * dim);
    if (zero_state) {
        qs_data_t one = qs_data_t(1.0, 0.0);
        cudaMemcpy(qs, &one, sizeof(qs_data_t), cudaMemcpyHostToDevice);
    }
    return qs;
}

void GPUVectorPolicyBase::FreeState(qs_data_p_t qs) {
    if (qs != nullptr) {
        cudaFree(qs);
    }
}

void GPUVectorPolicyBase::Reset(qs_data_p_t qs, index_t dim) {
    cudaMemset(qs, 0, sizeof(qs_data_t) * dim);
    qs_data_t one(1, 0);
    cudaMemcpy(qs, &one, sizeof(qs_data_t), cudaMemcpyHostToDevice);
}

void GPUVectorPolicyBase::Display(qs_data_p_t qs, qbit_t n_qubits, qbit_t q_limit) {
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

void GPUVectorPolicyBase::SetToZeroExcept(qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    thrust::counting_iterator<index_t> i(0);
    thrust::for_each(i, i + dim, [=] __device__(index_t i) {
        if ((i & ctrl_mask) != ctrl_mask) {
            qs[i] = 0;
        }
    });
}

template <index_t mask, index_t condi, class binary_op>
void GPUVectorPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, qs_data_t succ_coeff,
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

template <class binary_op>
void GPUVectorPolicyBase::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
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

void GPUVectorPolicyBase::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                         qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::plus<qs_data_t>());
}
void GPUVectorPolicyBase::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                           qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::minus<qs_data_t>());
}
void GPUVectorPolicyBase::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                         qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::multiplies<qs_data_t>());
}
void GPUVectorPolicyBase::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask, index_t condi,
                                         qs_data_t succ_coeff, qs_data_t fail_coeff, index_t dim) {
    ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::divides<qs_data_t>());
}
void GPUVectorPolicyBase::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value, index_t dim) {
    ConditionalBinary<0, 0>(src, des, value, 0, dim, thrust::multiplies<qs_data_t>());
}
auto GPUVectorPolicyBase::ConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi, bool abs, index_t dim)
    -> qs_data_t {
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

auto GPUVectorPolicyBase::GetQS(qs_data_p_t qs, index_t dim) -> py_qs_datas_t {
    py_qs_datas_t out(dim);
    cudaMemcpy(out.data(), qs, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    return out;
}

void GPUVectorPolicyBase::SetQS(qs_data_p_t qs, const py_qs_datas_t& qs_out, index_t dim) {
    if (qs_out.size() != dim) {
        throw std::invalid_argument("state size not match");
    }
    cudaMemcpy(qs, qs_out.data(), sizeof(qs_data_t) * dim, cudaMemcpyHostToDevice);
}

auto GPUVectorPolicyBase::ApplyTerms(qs_data_p_t qs, const std::vector<PauliTerm<calc_type>>& ham, index_t dim)
    -> qs_data_p_t {
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
            ApplyTerm<<<128, 128>>>(out, qs, coeff, mask.num_y, mask.mask_y, mask.mask_z, mask_f, dim);
        }
    }
    return out;
};
auto GPUVectorPolicyBase::Copy(qs_data_p_t qs, index_t dim) -> qs_data_p_t {
    qs_data_p_t out;
    cudaMalloc((void**) &out, sizeof(qs_data_t) * dim);  // NOLINT
    cudaMemcpy(out, qs, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToDevice);
    return out;
};

auto GPUVectorPolicyBase::Vdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
    thrust::device_ptr<qs_data_t> dev_bra(bra);
    thrust::device_ptr<qs_data_t> dev_ket(ket);
    qs_data_t res = thrust::inner_product(dev_bra, dev_bra + dim, dev_ket, qs_data_t(0, 0), thrust::plus<qs_data_t>(),
                                          conj_a_dot_b());
    return res;
}

template <index_t mask, index_t condi>
auto GPUVectorPolicyBase::ConditionVdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
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

auto GPUVectorPolicyBase::OneStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
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

auto GPUVectorPolicyBase::ZeroStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit, index_t dim)
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

void GPUVectorPolicyBase::ApplySWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
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

void GPUVectorPolicyBase::ApplyISWAP(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, bool daggered,
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

void GPUVectorPolicyBase::ApplyXX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
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

void GPUVectorPolicyBase::ApplyYY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
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

void GPUVectorPolicyBase::ApplyZZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, calc_type val, index_t dim,
                                  bool diff) {
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
auto GPUVectorPolicyBase::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, qs_data_p_t vec,
                                    index_t dim) -> qs_data_p_t {
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
auto GPUVectorPolicyBase::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                    const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b, qs_data_p_t vec,
                                    index_t dim) -> qs_data_p_t {
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
}  // namespace mindquantum::sim::vector::detail
