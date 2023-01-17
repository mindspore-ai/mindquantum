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

#include "simulator/utils.hpp"
#include "simulator/vector/detail/gpu_vector_double_policy.cuh"
#include "simulator/vector/detail/gpu_vector_float_policy.cuh"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"

namespace mindquantum::sim::vector::detail {

template <typename derived_, typename calc_type_>
template <index_t mask, index_t condi, class binary_op>
void GPUVectorPolicyBase<derived_, calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des,
                                                                  qs_data_t succ_coeff, qs_data_t fail_coeff,
                                                                  index_t dim, const binary_op& op) {
    thrust::counting_iterator<size_t> i(0);
    thrust::for_each(i, i + dim, [=] __device__(size_t i) {
        if ((i & mask) == condi) {
            des[i] = op(src[i], succ_coeff);
        } else {
            des[i] = op(src[i], fail_coeff);
        }
    });
}

template <typename derived_, typename calc_type_>
template <class binary_op>
void GPUVectorPolicyBase<derived_, calc_type_>::ConditionalBinary(qs_data_p_t src, qs_data_p_t des, index_t mask,
                                                                  index_t condi, qs_data_t succ_coeff,
                                                                  qs_data_t fail_coeff, index_t dim,
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

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ConditionalAdd(qs_data_p_t src, qs_data_p_t des, index_t mask,
                                                               index_t condi, qs_data_t succ_coeff,
                                                               qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::plus<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ConditionalMinus(qs_data_p_t src, qs_data_p_t des, index_t mask,
                                                                 index_t condi, qs_data_t succ_coeff,
                                                                 qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim, thrust::minus<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ConditionalMul(qs_data_p_t src, qs_data_p_t des, index_t mask,
                                                               index_t condi, qs_data_t succ_coeff,
                                                               qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim,
                                        thrust::multiplies<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ConditionalDiv(qs_data_p_t src, qs_data_p_t des, index_t mask,
                                                               index_t condi, qs_data_t succ_coeff,
                                                               qs_data_t fail_coeff, index_t dim) {
    derived::template ConditionalBinary(src, des, mask, condi, succ_coeff, fail_coeff, dim,
                                        thrust::divides<qs_data_t>());
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::QSMulValue(qs_data_p_t src, qs_data_p_t des, qs_data_t value,
                                                           index_t dim) {
    derived::template ConditionalBinary<0, 0>(src, des, value, 0, dim, thrust::multiplies<qs_data_t>());
}

template <typename derived_, typename calc_type_>
auto GPUVectorPolicyBase<derived_, calc_type_>::ConditionalCollect(qs_data_p_t qs, index_t mask, index_t condi,
                                                                   bool abs, index_t dim) -> qs_data_t {
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

template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
