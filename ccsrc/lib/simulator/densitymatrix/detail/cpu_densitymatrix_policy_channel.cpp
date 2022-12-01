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

#include "config/openmp.hpp"

#include "core/utils.hpp"
#include "simulator/densitymatrix/detail/cpu_densitymatrix_policy.hpp"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::densitymatrix::detail {
// Single qubit operator
// ========================================================================================================

// method is based on 'mq_vector' simulator, extended to densitymatrix
void CPUDensityMatrixPolicyBase::ApplySingleQubitChannel(qs_data_p_t src, qs_data_p_t des, qbit_t obj_qubit,
                                                         const VT<matrix_t>& kraus_set, index_t dim) {
    SingleQubitGateMask mask({obj_qubit}, {});

    THRESHOLD_OMP_FOR(
        dim, DimTh, for (index_t a = 0; a < (dim / 2); a++) {  // loop on the row
            auto r0 = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
            auto r1 = r0 + mask.obj_mask;
            for (index_t b = 0; b <= a; b++) {  // loop on the column
                auto c0 = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                auto c1 = c0 + mask.obj_mask;
                qs_data_t src_00 = src[IdxMap(r0, c0)];
                qs_data_t src_11 = src[IdxMap(r1, c1)];
                qs_data_t src_10 = src[IdxMap(r1, c0)];
                qs_data_t src_01 = GetValue(src, r0, c1);
                qs_data_t des_00 = 0;
                qs_data_t des_11 = 0;
                qs_data_t des_01 = 0;
                qs_data_t des_10 = 0;
                for (const auto& m : kraus_set) {
                    des_00 += m[0][0] * std::conj(m[0][0]) * src_00 + m[0][0] * std::conj(m[0][1]) * src_01
                              + m[0][1] * std::conj(m[0][0]) * src_10 + m[0][1] * std::conj(m[0][1]) * src_11;
                    des_11 += m[1][0] * std::conj(m[1][0]) * src_00 + m[1][0] * std::conj(m[1][1]) * src_01
                              + m[1][1] * std::conj(m[1][0]) * src_10 + m[1][1] * std::conj(m[1][1]) * src_11;
                    des_01 += m[0][0] * std::conj(m[1][0]) * src_00 + m[0][0] * std::conj(m[1][1]) * src_01
                              + m[0][1] * std::conj(m[1][0]) * src_10 + m[0][1] * std::conj(m[1][1]) * src_11;
                    des_10 += m[1][0] * std::conj(m[0][0]) * src_00 + m[1][0] * std::conj(m[0][1]) * src_01
                              + m[1][1] * std::conj(m[0][0]) * src_10 + m[1][1] * std::conj(m[0][1]) * src_11;
                }

                des[IdxMap(r0, c0)] = des_00;
                des[IdxMap(r1, c1)] = des_11;
                des[IdxMap(r1, c0)] = des_10;
                SetValue(des, r0, c1, des_01);
            }
        })
}

void CPUDensityMatrixPolicyBase::ApplyAmplitudeDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma,
                                                       index_t dim) {
    VT<matrix_t> kraus_set{{{1, 0}, {0, std::sqrt(1 - gamma)}}, {{0, std::sqrt(gamma)}, {0, 0}}};
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyPhaseDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma, index_t dim) {
    VT<matrix_t> kraus_set{{{1, 0}, {0, std::sqrt(1 - gamma)}}, {{0, 0}, {0, std::sqrt(gamma)}}};
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyPauli(qs_data_p_t qs, const qbits_t& objs, const VT<calc_type>& probs, index_t dim) {
    VT<matrix_t> kraus_set;
    if (std::abs(probs[0]) > 1e-8) {
        kraus_set.push_back({{0, std::sqrt(probs[0])}, {std::sqrt(probs[0]), 0}});
    }
    if (std::abs(probs[1]) > 1e-8) {
        kraus_set.push_back({{0, std::sqrt(probs[1]) * IMAGE_MI}, {std::sqrt(probs[1]) * IMAGE_I, 0}});
    }
    if (std::abs(probs[2]) > 1e-8) {
        kraus_set.push_back({{std::sqrt(probs[2]), 0}, {0, -std::sqrt(probs[2])}});
    }
    if (std::abs(probs[3]) > 1e-8) {
        kraus_set.push_back({{std::sqrt(probs[3]), 0}, {0, std::sqrt(probs[3])}});
    }
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyKraus(qs_data_p_t qs, const qbits_t& objs, VT<matrix_t>& kraus_set, index_t dim) {
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyHermitianAmplitudeDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma,
                                                                index_t dim) {
    VT<matrix_t> kraus_set{{{1, 0}, {0, std::sqrt(1 - gamma)}}, {{0, 0}, {std::sqrt(gamma), 0}}};
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

}  // namespace mindquantum::sim::densitymatrix::detail
