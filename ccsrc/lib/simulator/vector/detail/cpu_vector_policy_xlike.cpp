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

#include "core/utils.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"
#include "simulator/vector/detail/cpu_vector_policy.hpp"

namespace mindquantum::sim::vector::detail {
// X like operator
// ========================================================================================================

void CPUVectorPolicyBase::ApplyXLike(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, qs_data_t v1,
                                     qs_data_t v2, index_t dim) {
    SingleQubitGateMask mask(objs, ctrls);
    if (!mask.ctrl_mask) {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                auto j = i | mask.obj_mask;
                auto tmp = qs[i];
                qs[i] = qs[j] * v1;
                qs[j] = tmp * v2;
            })
    } else {
        THRESHOLD_OMP_FOR(
            dim, DimTh, for (index_t l = 0; l < (dim / 2); l++) {
                auto i = ((l & mask.obj_high_mask) << 1) + (l & mask.obj_low_mask);
                if ((i & mask.ctrl_mask) == mask.ctrl_mask) {
                    auto j = i | mask.obj_mask;
                    auto tmp = qs[i];
                    qs[i] = qs[j] * v1;
                    qs[j] = tmp * v1;
                }
            })
    }
}

void CPUVectorPolicyBase::ApplyX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, 1, 1, dim);
}

void CPUVectorPolicyBase::ApplyY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls, index_t dim) {
    ApplyXLike(qs, objs, ctrls, IMAGE_MI, IMAGE_I, dim);
}
}  // namespace mindquantum::sim::vector::detail
