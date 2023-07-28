/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <thrust/transform_reduce.h>

#include "config/openmp.hpp"

#include "core/sparse/algo.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::Vdot(const qs_data_p_t& bra, const qs_data_p_t& ket, index_t dim)
    -> py_qs_data_t {
    if (bra == nullptr && ket == nullptr) {
        return 1.0;
    } else if (bra == nullptr) {
        py_qs_data_t out;
        cudaMemcpy(&out, ket, sizeof(qs_data_t), cudaMemcpyDeviceToHost);
        return out;
    } else if (ket == nullptr) {
        py_qs_data_t out;
        cudaMemcpy(&out, bra, sizeof(qs_data_t), cudaMemcpyDeviceToHost);
        return std::conj(out);
    }
    thrust::device_ptr<qs_data_t> dev_bra(bra);
    thrust::device_ptr<qs_data_t> dev_ket(ket);
    qs_data_t res = thrust::inner_product(dev_bra, dev_bra + dim, dev_ket, qs_data_t(0, 0), thrust::plus<qs_data_t>(),
                                          conj_a_dot_b<qs_data_t>());
    return res;
}

template <typename derived_, typename calc_type_>
template <index_t mask, index_t condi>
auto GPUVectorPolicyBase<derived_, calc_type_>::ConditionVdot(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                              index_t dim) -> py_qs_data_t {
    if (bra == nullptr && ket == nullptr) {
        if ((0 & mask) == condi) {
            return 1.0;
        }
        return 0.0;
    } else if (bra == nullptr) {
        if ((0 & mask) == condi) {
            py_qs_data_t out;
            cudaMemcpy(&out, ket, sizeof(qs_data_t), cudaMemcpyDeviceToHost);
            return out;
        }
        return 0.0;
    } else if (ket == nullptr) {
        if ((0 & mask) == condi) {
            py_qs_data_t out;
            cudaMemcpy(&out, bra, sizeof(qs_data_t), cudaMemcpyDeviceToHost);
            return std::conj(out);
        }
        return 0.0;
    }
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

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::OneStateVdot(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                             qbit_t obj_qubit, index_t dim) -> py_qs_data_t {
    if (bra == nullptr || ket == nullptr) {
        return 0.0;
    }
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

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ZeroStateVdot(const qs_data_p_t& bra, const qs_data_p_t& ket,
                                                              qbit_t obj_qubit, index_t dim) -> py_qs_data_t {
    if (bra == nullptr && ket == nullptr) {
        return 1.0;
    } else if (bra == nullptr) {
        py_qs_data_t out;
        cudaMemcpy(&out, ket, sizeof(qs_data_t), cudaMemcpyDeviceToHost);
        return out;
    } else if (ket == nullptr) {
        py_qs_data_t out;
        cudaMemcpy(&out, bra, sizeof(qs_data_t), cudaMemcpyDeviceToHost);
        return std::conj(out);
    }
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

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                          const qs_data_p_t& vec_out, index_t dim) -> qs_data_p_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto vec = vec_out;
    bool will_free = false;
    if (vec == nullptr) {
        vec = derived::InitState(dim);
        will_free = true;
    }
    auto host = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host, vec, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto host_res = sparse::Csr_Dot_Vec<calc_type_, calc_type_>(a, reinterpret_cast<calc_type*>(host));
    auto out = InitState(dim);
    cudaMemcpy(out, reinterpret_cast<std::complex<calc_type>*>(host_res), sizeof(qs_data_t) * dim,
               cudaMemcpyHostToDevice);
    if (host != nullptr) {
        free(host);
    }
    if (host_res != nullptr) {
        free(host_res);
    }
    if (will_free) {
        derived::FreeState(&vec);
    }
    return out;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                          const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b,
                                                          const qs_data_p_t& vec_out, index_t dim) -> qs_data_p_t {
    if ((dim != a->dim_) || (dim != b->dim_)) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto vec = vec_out;
    bool will_free = false;
    if (vec == nullptr) {
        vec = derived::InitState(dim);
        will_free = true;
    }
    auto host = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host, vec, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto host_res = sparse::Csr_Dot_Vec<calc_type_, calc_type_>(a, b, reinterpret_cast<calc_type*>(host));
    auto out = InitState(dim);
    cudaMemcpy(out, reinterpret_cast<std::complex<calc_type>*>(host_res), sizeof(qs_data_t) * dim,
               cudaMemcpyHostToDevice);
    if (host != nullptr) {
        free(host);
    }
    if (host_res != nullptr) {
        free(host_res);
    }
    if (will_free) {
        derived::FreeState(&vec);
    }
    return out;
}
template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectationOfCsr(
    const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, const qs_data_p_t& bra_out, const qs_data_p_t& ket_out,
    index_t dim) -> py_qs_data_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    auto host_bra = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host_bra, bra, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto host_ket = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host_bra, ket, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto out = sparse::ExpectationOfCsr<calc_type, calc_type>(a, reinterpret_cast<calc_type*>(host_bra),
                                                              reinterpret_cast<calc_type*>(host_ket));
    if (host_bra != nullptr) {
        free(host_bra);
    }
    if (host_ket != nullptr) {
        free(host_ket);
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return out;
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ExpectationOfCsr(
    const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a, const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b,
    const qs_data_p_t& bra_out, const qs_data_p_t& ket_out, index_t dim) -> py_qs_data_t {
    if ((dim != a->dim_) || (dim != b->dim_)) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
    }
    auto bra = bra_out;
    auto ket = ket_out;
    bool will_free_bra = false, will_free_ket = false;
    if (bra == nullptr) {
        bra = derived::InitState(dim);
        will_free_bra = true;
    }
    if (ket == nullptr) {
        ket = derived::InitState(dim);
        will_free_ket = true;
    }
    auto host_bra = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host_bra, bra, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto host_ket = reinterpret_cast<std::complex<calc_type>*>(malloc(dim * sizeof(std::complex<calc_type>)));
    cudaMemcpy(host_bra, ket, sizeof(qs_data_t) * dim, cudaMemcpyDeviceToHost);
    auto out = sparse::ExpectationOfCsr<calc_type, calc_type>(a, b, reinterpret_cast<calc_type*>(host_bra),
                                                              reinterpret_cast<calc_type*>(host_ket));
    if (host_bra != nullptr) {
        free(host_bra);
    }
    if (host_ket != nullptr) {
        free(host_ket);
    }
    if (will_free_bra) {
        derived::FreeState(&bra);
    }
    if (will_free_ket) {
        derived::FreeState(&ket);
    }
    return out;
}
template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
