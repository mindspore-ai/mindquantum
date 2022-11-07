//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#ifndef GATES_HPP
#define GATES_HPP

#include <tweedledum/Operators/Ising.h>
#include <tweedledum/Operators/Standard.h>

#include "ops/gates/qubit_operator.hpp"

#include "experimental/ops/gates/allocate.hpp"
#include "experimental/ops/gates/deallocate.hpp"
#include "experimental/ops/gates/entangle.hpp"
#include "experimental/ops/gates/invalid.hpp"
#include "experimental/ops/gates/measure.hpp"
#include "experimental/ops/gates/ph.hpp"
#include "experimental/ops/gates/qft.hpp"
#include "experimental/ops/gates/sqrtswap.hpp"
#include "experimental/ops/gates/time_evolution.hpp"

namespace mindquantum::ops {
using namespace tweedledum::Op;  // NOLINT(build/namespaces)
}  // namespace mindquantum::ops

namespace tweedledum {
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::P> = 1;
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::Rx> = 1;
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::Ry> = 1;
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::Rz> = 1;
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::Rxx> = 1;
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::Ryy> = 1;
template <>
inline constexpr uint8_t num_param_v<mindquantum::ops::Rzz> = 1;
}  // namespace tweedledum

#endif /* GATES_HPP */
