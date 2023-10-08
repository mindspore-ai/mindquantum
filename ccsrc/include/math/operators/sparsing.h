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
#ifndef MATH_OPERATORS_SPARSING_H_
#define MATH_OPERATORS_SPARSING_H_
#include "math/operators/qubit_operator_view.h"
#include "math/tensor/csr_matrix.h"
namespace operators {
namespace tn = tensor;
tn::CsrMatrix GetMatrix(const qubit::QubitOperator& ops, int n_qubits = -1);
}  // namespace operators
#endif
