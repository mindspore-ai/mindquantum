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

#include "math/operators/transform.hpp"

namespace operators::transform {
fermion_op_t fermion_number_operator(int n_modes, int mode, const parameter::ParameterResolver& coeff) {
    fermion_op_t out{};
    if (mode < 0) {
        for (auto m = 0; m < n_modes; m++) {
            out += fermion_number_operator(n_modes, m, coeff);
        }
    } else {
        out += fermion_op_t({{mode, fermion::TermValue::Ad}, {mode, fermion::TermValue::A}}, coeff);
    }
    return out;
}
}  // namespace operators::transform
