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
auto GPUVectorPolicyBase<derived_, calc_type_>::Vdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim) -> py_qs_data_t {
    thrust::device_ptr<qs_data_t> dev_bra(bra);
    thrust::device_ptr<qs_data_t> dev_ket(ket);
    qs_data_t res = thrust::inner_product(dev_bra, dev_bra + dim, dev_ket, qs_data_t(0, 0), thrust::plus<qs_data_t>(),
                                          conj_a_dot_b<qs_data_t>());
    return res;
}

template <typename derived_, typename calc_type_>
template <index_t mask, index_t condi>
auto GPUVectorPolicyBase<derived_, calc_type_>::ConditionVdot(qs_data_p_t bra, qs_data_p_t ket, index_t dim)
    -> py_qs_data_t {
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
auto GPUVectorPolicyBase<derived_, calc_type_>::OneStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit,
                                                             index_t dim) -> py_qs_data_t {
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
auto GPUVectorPolicyBase<derived_, calc_type_>::ZeroStateVdot(qs_data_p_t bra, qs_data_p_t ket, qbit_t obj_qubit,
                                                              index_t dim) -> py_qs_data_t {
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
                                                          qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if (dim != a->dim_) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
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
    return out;
}
template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::CsrDotVec(const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& a,
                                                          const std::shared_ptr<sparse::CsrHdMatrix<calc_type>>& b,
                                                          qs_data_p_t vec, index_t dim) -> qs_data_p_t {
    if ((dim != a->dim_) || (dim != b->dim_)) {
        throw std::runtime_error("Sparse hamiltonian size not match with quantum state size.");
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
    return out;
}

template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
