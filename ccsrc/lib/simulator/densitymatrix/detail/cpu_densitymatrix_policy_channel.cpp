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
            auto i = ((a & mask.obj_high_mask) << 1) + (a & mask.obj_low_mask);
            auto j = i + mask.obj_mask;
            for (index_t b = 0; b <= a; b++) {  // loop on the column
                auto p = ((b & mask.obj_high_mask) << 1) + (b & mask.obj_low_mask);
                auto q = p + mask.obj_mask;
                qs_data_t src_ip = src[IdxMap(i, p)];
                qs_data_t src_jq = src[IdxMap(j, q)];
                qs_data_t src_jp = src[IdxMap(j, p)];
                qs_data_t src_iq;
                if (i > q) {  // for matrix[row, col], only in this case (row < col) is possible
                    src_iq = src[IdxMap(i, q)];
                } else {
                    src_iq = std::conj(src[IdxMap(q, i)]);
                }
                qs_data_t des_ip = 0;
                qs_data_t des_jq = 0;
                qs_data_t des_iq = 0;
                qs_data_t des_jp = 0;
                for (const auto& m : kraus_set) {
                    des_ip += m[0][0] * std::conj(m[0][0]) * src_ip + m[0][0] * std::conj(m[0][1]) * src_iq
                              + m[0][1] * std::conj(m[0][0]) * src_jp + m[0][1] * std::conj(m[0][1]) * src_jq;
                    des_jq += m[1][0] * std::conj(m[1][0]) * src_ip + m[1][0] * std::conj(m[1][1]) * src_iq
                              + m[1][1] * std::conj(m[1][0]) * src_jp + m[1][1] * std::conj(m[1][1]) * src_jq;
                    des_iq += m[0][0] * std::conj(m[1][0]) * src_ip + m[0][0] * std::conj(m[1][1]) * src_iq
                              + m[0][1] * std::conj(m[1][0]) * src_jp + m[0][1] * std::conj(m[1][1]) * src_jq;
                    des_jp += m[1][0] * std::conj(m[0][0]) * src_ip + m[1][0] * std::conj(m[0][1]) * src_iq
                              + m[1][1] * std::conj(m[0][0]) * src_jp + m[1][1] * std::conj(m[0][1]) * src_jq;
                }

                des[IdxMap(i, p)] = des_ip;
                des[IdxMap(j, q)] = des_jq;
                des[IdxMap(j, p)] = des_jp;
                if (i > q) {
                    des[IdxMap(i, q)] = des_iq;
                } else {
                    des[IdxMap(q, i)] = std::conj(des_iq);
                }
            }
        })
}

void CPUDensityMatrixPolicyBase::ApplyAmplitudeDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma,
                                                       index_t dim) {
    VT<matrix_t> kraus_set{{{1, 0}, {0, sqrt(1 - gamma)}}, {{0, sqrt(gamma)}, {0, 0}}};
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyPhaseDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma, index_t dim) {
    VT<matrix_t> kraus_set{{{1, 0}, {0, sqrt(1 - gamma)}}, {{0, 0}, {0, sqrt(gamma)}}};
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyPauli(qs_data_p_t qs, const qbits_t& objs, VT<calc_type>& probs, index_t dim) {
    VT<matrix_t> kraus_set;
    if (probs[0] != 0) {
        kraus_set.push_back({{0, probs[0]}, {probs[0], 0}});
    }
    if (probs[1] != 0) {
        kraus_set.push_back({{0, probs[1] * IMAGE_MI}, {probs[1] * IMAGE_I, 0}});
    }
    if (probs[2] != 0) {
        kraus_set.push_back({{probs[2], 0}, {0, -probs[2]}});
    }
    if (probs[3] != 0) {
        kraus_set.push_back({{probs[3], 0}, {0, probs[3]}});
    }
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyKraus(qs_data_p_t qs, const qbits_t& objs, VT<matrix_t>& kraus_set, index_t dim) {
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

void CPUDensityMatrixPolicyBase::ApplyHermitianAmplitudeDamping(qs_data_p_t qs, const qbits_t& objs, calc_type gamma,
                                                                index_t dim) {
    VT<matrix_t> kraus_set{{{1, 0}, {0, sqrt(1 - gamma)}}, {{0, 0}, {sqrt(gamma), 0}}};
    ApplySingleQubitChannel(qs, qs, objs[0], kraus_set, dim);
}

}  // namespace mindquantum::sim::densitymatrix::detail
