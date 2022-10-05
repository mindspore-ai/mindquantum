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

#include "simulator/vector/detail/gpu_vector_policy.cuh"
#include "simulator/types.hpp"

namespace mindquantum::sim::vector::detail {
__global__ void SetToZeroExceptCore(GPUVectorPolicyBase::qs_data_p_t qs, index_t ctrl_mask, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((i & ctrl_mask) != ctrl_mask) {
            qs[i] = 0;
        }
    }
}

__global__ void ApplyXCoreNoControl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto tmp = qs[i];
            qs[i] = qs[i | obj_mask];
            qs[i | obj_mask] = tmp;
        }
    }
}

__global__ void ApplyXCoreWithControl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                      index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask) && ((i & ctrl_mask) == ctrl_mask)) {
            auto tmp = qs[i];
            qs[i] = qs[i | obj_mask];
            qs[i | obj_mask] = tmp;
        }
    }
}

__global__ void ApplyYCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto tmp = qs[i];
            qs[i] = qs[i | obj_mask] * GPUVectorPolicyBase::qs_data_t(0, -1);
            qs[i | obj_mask] = tmp * GPUVectorPolicyBase::qs_data_t(0, 1);
        }
    }
}

__global__ void ApplyYCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                   index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto tmp = qs[i];
            qs[i] = qs[i | obj_mask] * GPUVectorPolicyBase::qs_data_t(0, -1);
            qs[i | obj_mask] = tmp * GPUVectorPolicyBase::qs_data_t(0, 1);
        }
    }
}

__global__ void ApplyZCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((i & obj_mask)) {
            qs[i] *= -1;
        }
    }
}

__global__ void ApplyZCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                   index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((i & obj_mask) && ((i & ctrl_mask) == ctrl_mask)) {
            qs[i] *= -1;
        }
    }
}

__global__ void ApplyHCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto a = qs[i];
            auto b = qs[i | obj_mask];
            qs[i] = (a + b) * M_SQRT1_2;
            qs[i | obj_mask] = (a - b) * M_SQRT1_2;
        }
    }
}

__global__ void ApplyHCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                   index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto a = qs[i];
            auto b = qs[i | obj_mask];
            qs[i] = (a + b) * M_SQRT1_2;
            qs[i | obj_mask] = (a - b) * M_SQRT1_2;
        }
    }
}

__global__ void ApplySWAPCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, qbit_t q0, qbit_t q1,
                                    index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto base_one = (1UL << q0);
    auto base_two = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto tmp = qs[i + base_one];
            qs[i + base_one] = qs[i + base_two];
            qs[i + base_two] = tmp;
        }
    }
}

__global__ void ApplySWAPCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                      qbit_t q0, qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto base_one = (1UL << q0);
    auto base_two = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto tmp = qs[i + base_one];
            qs[i + base_one] = qs[i + base_two];
            qs[i + base_two] = tmp;
        }
    }
}

__global__ void ApplyISWAPCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, qbit_t q0, qbit_t q1,
                                     index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto base_one = (1UL << q0);
    auto base_two = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto tmp = qs[i + base_one];
            qs[i + base_one] = qs[i + base_two] * GPUVectorPolicyBase::qs_data_t(0, 1);
            qs[i + base_two] = tmp * GPUVectorPolicyBase::qs_data_t(0, 1);
        }
    }
}

__global__ void ApplyISWAPCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                       qbit_t q0, qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto base_one = (1UL << q0);
    auto base_two = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto tmp = qs[i + base_one];
            qs[i + base_one] = qs[i + base_two];
            qs[i + base_two] = tmp;
        }
    }
}

__global__ void ApplyRXCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask,
                                  GPUVectorPolicyBase::qs_data_t c, GPUVectorPolicyBase::qs_data_t is, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto t1 = c * qs[i] + is * qs[i | obj_mask];
            auto t2 = is * qs[i] + c * qs[i | obj_mask];
            qs[i] = t1;
            qs[i | obj_mask] = t2;
        }
    }
}

__global__ void ApplyRXCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                    GPUVectorPolicyBase::qs_data_t c, GPUVectorPolicyBase::qs_data_t is, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto t1 = c * qs[i] + is * qs[i | obj_mask];
            auto t2 = is * qs[i] + c * qs[i | obj_mask];
            qs[i] = t1;
            qs[i | obj_mask] = t2;
        }
    }
}

__global__ void ApplyRYCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, calc_type c, calc_type s,
                                  index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto t1 = c * qs[i] - s * qs[i | obj_mask];
            auto t2 = s * qs[i] + c * qs[i | obj_mask];
            qs[i] = t1;
            qs[i | obj_mask] = t2;
        }
    }
}

__global__ void ApplyRYCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                    calc_type c, calc_type s, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto t1 = c * qs[i] - s * qs[i | obj_mask];
            auto t2 = s * qs[i] + c * qs[i | obj_mask];
            qs[i] = t1;
            qs[i | obj_mask] = t2;
        }
    }
}

__global__ void ApplyRZCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask,
                                  GPUVectorPolicyBase::qs_data_t e0, GPUVectorPolicyBase::qs_data_t e1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            qs[i] *= e0;
            qs[i | obj_mask] *= e1;
        }
    }
}

__global__ void ApplyRZCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                    GPUVectorPolicyBase::qs_data_t e0, GPUVectorPolicyBase::qs_data_t e1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            qs[i] *= e0;
            qs[i | obj_mask] *= e1;
        }
    }
}

__global__ void ApplyXXCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask,
                                  GPUVectorPolicyBase::qs_data_t c, GPUVectorPolicyBase::qs_data_t s, qbit_t q0,
                                  qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto obj1_mask = (1UL << q0);
    auto obj2_mask = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto v00 = c * qs[i] + s * qs[i + obj_mask];
            auto v01 = c * qs[i + obj1_mask] + s * qs[i + obj2_mask];
            auto v10 = c * qs[i + obj2_mask] + s * qs[i + obj1_mask];
            auto v11 = c * qs[i + obj_mask] + s * qs[i];
            qs[i] = v00;
            qs[i + obj1_mask] = v01;
            qs[i + obj2_mask] = v10;
            qs[i + obj_mask] = v11;
        }
    }
}

__global__ void ApplyXXCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                    GPUVectorPolicyBase::qs_data_t c, GPUVectorPolicyBase::qs_data_t s, qbit_t q0,
                                    qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto obj1_mask = (1UL << q0);
    auto obj2_mask = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto v00 = c * qs[i] + s * qs[i + obj_mask];
            auto v01 = c * qs[i + obj1_mask] + s * qs[i + obj2_mask];
            auto v10 = c * qs[i + obj2_mask] + s * qs[i + obj1_mask];
            auto v11 = c * qs[i + obj_mask] + s * qs[i];
            qs[i] = v00;
            qs[i + obj1_mask] = v01;
            qs[i + obj2_mask] = v10;
            qs[i + obj_mask] = v11;
        }
    }
}

__global__ void ApplyYYCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask,
                                  GPUVectorPolicyBase::qs_data_t c, GPUVectorPolicyBase::qs_data_t s, qbit_t q0,
                                  qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto obj1_mask = (1UL << q0);
    auto obj2_mask = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            auto v00 = c * qs[i] + s * qs[i + obj_mask];
            auto v01 = c * qs[i + obj1_mask] - s * qs[i + obj2_mask];
            auto v10 = c * qs[i + obj2_mask] - s * qs[i + obj1_mask];
            auto v11 = c * qs[i + obj_mask] + s * qs[i];
            qs[i] = v00;
            qs[i + obj1_mask] = v01;
            qs[i + obj2_mask] = v10;
            qs[i + obj_mask] = v11;
        }
    }
}

__global__ void ApplyYYCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                    GPUVectorPolicyBase::qs_data_t c, GPUVectorPolicyBase::qs_data_t s, qbit_t q0,
                                    qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto obj1_mask = (1UL << q0);
    auto obj2_mask = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            auto v00 = c * qs[i] + s * qs[i + obj_mask];
            auto v01 = c * qs[i + obj1_mask] - s * qs[i + obj2_mask];
            auto v10 = c * qs[i + obj2_mask] - s * qs[i + obj1_mask];
            auto v11 = c * qs[i + obj_mask] + s * qs[i];
            qs[i] = v00;
            qs[i + obj1_mask] = v01;
            qs[i + obj2_mask] = v10;
            qs[i + obj_mask] = v11;
        }
    }
}

__global__ void ApplyZZCoreNoCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask,
                                  GPUVectorPolicyBase::qs_data_t e, GPUVectorPolicyBase::qs_data_t me, qbit_t q0,
                                  qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto obj1_mask = (1UL << q0);
    auto obj2_mask = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            qs[i] *= me;
            qs[i + obj1_mask] *= e;
            qs[i + obj2_mask] *= e;
            qs[i + obj_mask] *= me;
        }
    }
}

__global__ void ApplyZZCoreWithCtrl(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                    GPUVectorPolicyBase::qs_data_t e, GPUVectorPolicyBase::qs_data_t me, qbit_t q0,
                                    qbit_t q1, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    auto obj1_mask = (1UL << q0);
    auto obj2_mask = (1UL << q1);
    for (index_t i = index; i < dim; i += stride) {
        if ((!(i & obj_mask)) && ((i & ctrl_mask) == ctrl_mask)) {
            qs[i] *= me;
            qs[i + obj1_mask] *= e;
            qs[i + obj2_mask] *= e;
            qs[i + obj_mask] *= me;
        }
    }
}

__global__ void ApplyPSCoreNoCtrlNoDiff(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask,
                                        GPUVectorPolicyBase::qs_data_t e, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (i & obj_mask) {
            qs[i] *= e;
        }
    }
}

__global__ void ApplyPSCoreNoCtrlDiff(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask,
                                      GPUVectorPolicyBase::qs_data_t e, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask)) {
            qs[i] = 0;
            qs[i | obj_mask] *= e;
        }
    }
}

__global__ void ApplyPSCoreWithCtrlNoDiff(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                          GPUVectorPolicyBase::qs_data_t e, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if ((i & obj_mask) && ((i & ctrl_mask) == ctrl_mask)) {
            qs[i] *= e;
        }
    }
}

__global__ void ApplyPSCoreWithCtrlDiff(GPUVectorPolicyBase::qs_data_p_t qs, index_t obj_mask, index_t ctrl_mask,
                                        GPUVectorPolicyBase::qs_data_t e, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
        if (!(i & obj_mask) && ((i & ctrl_mask) == ctrl_mask)) {
            qs[i] = 0;
            qs[i | obj_mask] *= e;
        }
    }
}

__global__ void ApplyTerm(GPUVectorPolicyBase::qs_data_p_t des, GPUVectorPolicyBase::qs_data_p_t src, calc_type coeff,
                          index_t num_y, index_t mask_y, index_t mask_z, index_t mask_f, index_t dim) {
    index_t index = threadIdx.x + blockIdx.x * blockDim.x;
    index_t stride = blockDim.x * gridDim.x;
    for (index_t i = index; i < dim; i += stride) {
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
            des[j] += src[i] * coeff * c;
            if (i != j) {
                des[i] += src[j] * coeff / c;
            }
        }
    }
}

}  // namespace mindquantum::sim::vector::detail
