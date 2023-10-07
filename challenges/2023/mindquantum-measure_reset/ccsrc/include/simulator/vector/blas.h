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

#ifndef INCLUDE_QUANTUMSTATE_BLAS_HPP_
#define INCLUDE_QUANTUMSTATE_BLAS_HPP_
#include <cassert>

#include "simulator/vector/vector_state.h"
namespace mindquantum::sim::vector::detail {
template <typename qs_policy_t_>
struct BLAS {
    using qs_policy_t = qs_policy_t_;
    using qs_data_t = typename qs_policy_t::qs_data_t;
    using sim_t = VectorState<qs_policy_t>;
    static auto InnerProduct(const sim_t& bra, const sim_t& ket) {
        assert(bra.dim == ket.dim);
        return qs_policy_t::Vdot(bra.qs, ket.qs, bra.dim);
    }
};
}  // namespace mindquantum::sim::vector::detail
#endif
