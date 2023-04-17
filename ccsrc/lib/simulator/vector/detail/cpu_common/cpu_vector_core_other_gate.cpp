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

#include "math/pr/parameter_resolver.hpp"
#ifdef __x86_64__
#    include "simulator/vector/detail/cpu_vector_avx_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_avx_float_policy.hpp"
#elif defined(__amd64)
#    include "simulator/vector/detail/cpu_vector_arm_double_policy.hpp"
#    include "simulator/vector/detail/cpu_vector_arm_float_policy.hpp"
#endif
#include "simulator/vector/detail/cpu_vector_policy.hpp"
namespace mindquantum::sim::vector::detail {
template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyH(qs_data_p_t qs, const qbits_t& objs, const qbits_t& ctrls,
                                                       index_t dim) {
    std::vector<std::vector<py_qs_data_t>> m{{M_SQRT1_2, M_SQRT1_2}, {M_SQRT1_2, -M_SQRT1_2}};
    derived::ApplySingleQubitMatrix(qs, qs, objs[0], ctrls, m, dim);
}

template <typename derived_, typename calc_type_>
void CPUVectorPolicyBase<derived_, calc_type_>::ApplyGP(qs_data_p_t qs, qbit_t obj_qubit, const qbits_t& ctrls,
                                                        calc_type val, index_t dim, bool diff) {
    auto c = std::exp(std::complex<calc_type>(0, -val));
    std::vector<std::vector<py_qs_data_t>> m = {{c, 0}, {0, c}};
    derived::ApplySingleQubitMatrix(qs, qs, obj_qubit, ctrls, m, dim);
}

#ifdef __x86_64__
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyAvxDouble, double>;
#elif defined(__amd64)
template struct CPUVectorPolicyBase<CPUVectorPolicyArmFloat, float>;
template struct CPUVectorPolicyBase<CPUVectorPolicyArmDouble, double>;
#endif
}  // namespace mindquantum::sim::vector::detail
