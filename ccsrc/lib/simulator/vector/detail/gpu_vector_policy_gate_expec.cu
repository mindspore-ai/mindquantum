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
#include <complex>

#include <cassert>
#include <stdexcept>

#include <thrust/transform_reduce.h>

#include "thrust/device_ptr.h"
#include "thrust/functional.h"
#include "thrust/inner_product.h"
#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "simulator/types.hpp"
#include "simulator/utils.hpp"

namespace mindquantum::sim::vector::detail {
auto GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs,
                                                      const qbits_t& ctrls, const std::vector<py_qs_datas_t>& m,
                                                      index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto m00 = m[0][0];
    auto m01 = m[0][1];
    auto m10 = m[1][0];
    auto m11 = m[1][1];
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

auto GPUVectorPolicyBase::ExpectDiffRX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    auto c = -0.5 * std::sin(val / 2);
    auto is = 0.5 * std::cos(val / 2) * qs_data_t(0, -1);
    std::vector<py_qs_datas_t> gate = {{c, is}, {is, c}};
    return GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
}

auto GPUVectorPolicyBase::ExpectDiffRY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = -0.5 * std::sin(val / 2);
    auto s = 0.5 * std::cos(val / 2);
    std::vector<py_qs_datas_t> gate = {{c, -s}, {s, c}};
    return GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
}

auto GPUVectorPolicyBase::ExpectDiffRZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto c = -0.5 * std::sin(val / 2);
    auto s = 0.5 * std::cos(val / 2);
    auto e0 = c + qs_data_t(0, -1) * s;
    auto e1 = c + qs_data_t(0, 1) * s;
    std::vector<py_qs_datas_t> gate = {{e0, 0}, {0, e1}};
    return GPUVectorPolicyBase::ExpectDiffSingleQubitMatrix(bra, ket, objs, ctrls, m, dim);
}

auto GPUVectorPolicyBase::ExpectDiffXX(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val) * qs_data_t(0, -1);
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

auto GPUVectorPolicyBase::ExpectDiffYY(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val) * qs_data_t(0, 1);
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

auto GPUVectorPolicyBase::ExpectDiffZZ(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    DoubleQubitGateMask mask(objs, ctrls);
    auto c = -std::sin(val);
    auto s = std::cos(val);
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

auto GPUVectorPolicyBase::ExpectDiffPS(qs_data_p_t bra, qs_data_p_t ket, const qbits_t& objs, const qbits_t& ctrls,
                                       calc_type val, index_t dim) -> qs_data_t {
    SingleQubitGateMask mask(objs, ctrls);
    auto e = std::cos(val) + qs_data_t(0, 1) * std::sin(val);
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
