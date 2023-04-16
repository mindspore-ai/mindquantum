/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDQUANTUM_GATE_GATES_HPP_
#define MINDQUANTUM_GATE_GATES_HPP_

#include <cmath>

#include <functional>
#include <string>
#include <utility>

#include "core/mq_base_types.hpp"
#include "core/utils.hpp"
#include "math/pr/parameter_resolver.hpp"
#include "math/tensor/matrix.hpp"
#include "math/tensor/ops/advance_math.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/ops_cpu/advance_math.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"
#include "ops/basic_gate.hpp"
#include "ops/gate_id.hpp"

#ifndef M_SQRT1_2
#    define M_SQRT1_2 0.707106781186547524400844362104849039
#endif  // !M_SQRT1_2

#ifndef M_PI
#    define M_PI 3.14159265358979323846264338327950288
#endif  // !M_PI

#ifndef M_PI_2
#    define M_PI_2 1.57079632679489661923132169163975144
#endif  // !M_PI_2

namespace mindquantum {
tensor::Matrix U3Matrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda);

tensor::Matrix FSimMatrix(tensor::Tensor theta, tensor::Tensor phi);

tensor::Matrix U3DiffThetaMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda);

tensor::Matrix FSimDiffThetaMatrix(tensor::Tensor theta);

tensor::Matrix U3DiffPhiMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda);

tensor::Matrix FSimDiffPhiMatrix(tensor::Tensor phi);

tensor::Matrix U3DiffLambdaMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda);

struct U3 : public Parameterizable {
    parameter::ParameterResolver theta;
    parameter::ParameterResolver phi;
    parameter::ParameterResolver lambda;
    tensor::Matrix base_matrix_;
    U3(const parameter::ParameterResolver& theta, const parameter::ParameterResolver& phi,
       const parameter::ParameterResolver& lambda, const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits);
};

struct FSim : public Parameterizable {
    parameter::ParameterResolver theta;
    parameter::ParameterResolver phi;
    tensor::Matrix base_matrix_;
    FSim(const parameter::ParameterResolver& theta, const parameter::ParameterResolver& phi,
         const VT<Index>& obj_qubits, const VT<Index>& ctrl_qubits);
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_GATES_HPP_
