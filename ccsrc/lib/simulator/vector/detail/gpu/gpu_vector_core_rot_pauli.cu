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
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRX(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = static_cast<calc_type>(std::cos(val / 2));
    auto b = static_cast<calc_type>(-std::sin(val / 2));
    if (diff) {
        a = static_cast<calc_type>(-0.5 * std::sin(val / 2));
        b = static_cast<calc_type>(-0.5 * std::cos(val / 2));
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {0, b}}, {{0, b}, {a, 0}}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRY(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = static_cast<calc_type>(std::cos(val / 2));
    auto b = static_cast<calc_type>(std::sin(val / 2));
    if (diff) {
        a = static_cast<calc_type>(-0.5 * std::sin(val / 2));
        b = static_cast<calc_type>(0.5 * std::cos(val / 2));
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, 0}, {-b, 0}}, {{b, 0}, {a, 0}}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRZ(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    SingleQubitGateMask mask(objs, ctrls);
    auto a = static_cast<calc_type>(std::cos(val / 2));
    auto b = static_cast<calc_type>(std::sin(val / 2));
    if (diff) {
        a = static_cast<calc_type>(-0.5 * std::sin(val / 2));
        b = static_cast<calc_type>(0.5 * std::cos(val / 2));
    }
    std::vector<std::vector<py_qs_data_t>> m{{{a, -b}, {0, 0}}, {{0, 0}, {a, b}}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
    if (diff && mask.ctrl_mask) {
        derived::SetToZeroExcept(qs, mask.ctrl_mask, dim);
    }
}
template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRzz(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
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
    auto c = static_cast<calc_type>(std::cos(val / 2));
    auto s = static_cast<calc_type>(std::sin(val / 2));
    if (diff) {
        c = static_cast<calc_type>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type>(std::cos(val / 2) / 2);
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
            derived::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRxx(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
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
    auto c = static_cast<qs_data_t>(std::cos(val / 2));
    auto s = static_cast<calc_type>(std::sin(val / 2)) * qs_data_t(0, -1);
    if (diff) {
        c = static_cast<qs_data_t>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type>(std::cos(val / 2) / 2) * qs_data_t(0, -1);
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
            derived::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRxy(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
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
    auto c = static_cast<qs_data_t>(std::cos(val / 2));
    auto s = static_cast<calc_type>(std::sin(val / 2));
    if (diff) {
        c = static_cast<qs_data_t>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type>(std::cos(val / 2) / 2);
    }
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto m = i + obj_mask;
            auto v00 = c * qs[i] - s * qs[m];
            auto v01 = c * qs[j] - s * qs[k];
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
                auto v00 = c * qs[i] - s * qs[m];
                auto v01 = c * qs[j] - s * qs[k];
                auto v10 = c * qs[k] + s * qs[j];
                auto v11 = c * qs[m] + s * qs[i];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            }
        });
        if (diff) {
            derived::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRxz(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
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
    auto c = static_cast<qs_data_t>(std::cos(val / 2));
    auto s = static_cast<calc_type>(std::sin(val / 2)) * qs_data_t(0, -1);
    if (diff) {
        c = static_cast<qs_data_t>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type>(std::cos(val / 2) / 2) * qs_data_t(0, -1);
    }
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto m = i + obj_mask;
            auto v00 = c * qs[i] + s * qs[j];
            auto v01 = c * qs[j] + s * qs[i];
            auto v10 = c * qs[k] - s * qs[m];
            auto v11 = c * qs[m] - s * qs[k];
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
                auto v00 = c * qs[i] + s * qs[j];
                auto v01 = c * qs[j] + s * qs[i];
                auto v10 = c * qs[k] - s * qs[m];
                auto v11 = c * qs[m] - s * qs[k];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            }
        });
        if (diff) {
            derived::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRyz(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
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
    auto c = static_cast<qs_data_t>(std::cos(val / 2));
    auto s = static_cast<calc_type>(std::sin(val / 2));
    if (diff) {
        c = static_cast<qs_data_t>(-std::sin(val / 2) / 2);
        s = static_cast<calc_type>(std::cos(val / 2) / 2);
    }
    if (!mask.ctrl_mask) {
        thrust::for_each(l, l + dim / 4, [=] __device__(index_t l) {
            index_t i;
            SHIFT_BIT_TWO(obj_low_mask, obj_rev_low_mask, obj_high_mask, obj_rev_high_mask, l, i);
            auto j = i + obj_min_mask;
            auto k = i + obj_max_mask;
            auto m = i + obj_mask;
            auto v00 = c * qs[i] - s * qs[j];
            auto v01 = c * qs[j] + s * qs[i];
            auto v10 = c * qs[k] + s * qs[m];
            auto v11 = c * qs[m] - s * qs[k];
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
                auto v00 = c * qs[i] - s * qs[j];
                auto v01 = c * qs[j] + s * qs[i];
                auto v10 = c * qs[k] + s * qs[m];
                auto v11 = c * qs[m] - s * qs[k];
                qs[i] = v00;
                qs[j] = v01;
                qs[k] = v10;
                qs[m] = v11;
            }
        });
        if (diff) {
            derived::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template <typename derived_, typename calc_type_>
void GPUVectorPolicyBase<derived_, calc_type_>::ApplyRyy(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                         calc_type val, index_t dim, bool diff) {
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
    auto c = static_cast<qs_data_t>(std::cos(val / 2));
    auto s = static_cast<calc_type>(std::sin(val / 2)) * qs_data_t(0, 1);
    if (diff) {
        c = static_cast<qs_data_t>(-std::sin(val / 2) / 2);
        s = static_cast<qs_data_t>(std::cos(val / 2) / 2) * qs_data_t(0, 1);
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
            derived::SetToZeroExcept(qs, ctrl_mask, dim);
        }
    }
}

template struct GPUVectorPolicyBase<GPUVectorPolicyFloat, float>;
template struct GPUVectorPolicyBase<GPUVectorPolicyDouble, double>;

}  // namespace mindquantum::sim::vector::detail
