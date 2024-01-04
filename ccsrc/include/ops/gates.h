/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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

#include "core/mq_base_types.h"
#include "core/utils.h"
#include "math/pr/parameter_resolver.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops/advance_math.h"
#include "math/tensor/ops/memory_operator.h"
#include "math/tensor/ops_cpu/advance_math.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"
#include "ops/basic_gate.h"
#include "ops/gate_id.h"

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

tensor::Matrix RnMatrix(tensor::Tensor alpha, tensor::Tensor beta, tensor::Tensor gamma);

tensor::Matrix U3DiffThetaMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda);

tensor::Matrix FSimDiffThetaMatrix(tensor::Tensor theta);

tensor::Matrix U3DiffPhiMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda);

tensor::Matrix FSimDiffPhiMatrix(tensor::Tensor phi);

tensor::Matrix U3DiffLambdaMatrix(tensor::Tensor theta, tensor::Tensor phi, tensor::Tensor lambda);

tensor::Matrix RnDiffAlphaMatrix(tensor::Tensor alpha, tensor::Tensor beta, tensor::Tensor gamma);

tensor::Matrix RnDiffBetaMatrix(tensor::Tensor alpha, tensor::Tensor beta, tensor::Tensor gamma);

tensor::Matrix RnDiffGammaMatrix(tensor::Tensor alpha, tensor::Tensor beta, tensor::Tensor gamma);

class U3 : public Parameterizable {
 public:
    U3(const parameter::ParameterResolver& theta, const parameter::ParameterResolver& phi,
       const parameter::ParameterResolver& lambda, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits);
    const tensor::Matrix& GetBaseMatrix() {
        return base_matrix_;
    }
    parameter::ParameterResolver GetTheta() const {
        return theta;
    }
    parameter::ParameterResolver GetPhi() const {
        return phi;
    }
    parameter::ParameterResolver GetLambda() const {
        return lambda;
    }

 private:
    parameter::ParameterResolver theta;
    parameter::ParameterResolver phi;
    parameter::ParameterResolver lambda;
    tensor::Matrix base_matrix_;
};

class FSim : public Parameterizable {
 public:
    FSim(const parameter::ParameterResolver& theta, const parameter::ParameterResolver& phi, const qbits_t& obj_qubits,
         const qbits_t& ctrl_qubits);
    const tensor::Matrix& GetBaseMatrix() {
        return base_matrix_;
    }
    parameter::ParameterResolver GetTheta() const {
        return theta;
    }
    parameter::ParameterResolver GetPhi() const {
        return phi;
    }

 private:
    parameter::ParameterResolver theta;
    parameter::ParameterResolver phi;
    tensor::Matrix base_matrix_;
};

class Rn : public Parameterizable {
 public:
    Rn(const parameter::ParameterResolver& alpha, const parameter::ParameterResolver& beta,
       const parameter::ParameterResolver& gamma, const qbits_t& obj_qubits, const qbits_t& ctrl_qubits);
    const tensor::Matrix& GetBaseMatrix() {
        return base_matrix_;
    }
    parameter::ParameterResolver GetAlpha() const {
        return alpha;
    }
    parameter::ParameterResolver GetBeta() const {
        return beta;
    }
    parameter::ParameterResolver GetGamma() const {
        return gamma;
    }

 private:
    parameter::ParameterResolver alpha;
    parameter::ParameterResolver beta;
    parameter::ParameterResolver gamma;
    tensor::Matrix base_matrix_;
};
}  // namespace mindquantum
#endif  // MINDQUANTUM_GATE_GATES_HPP_
