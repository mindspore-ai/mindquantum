//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#ifndef MATH_TENSOR_TRAITS_HPP_
#define MATH_TENSOR_TRAITS_HPP_

#include <complex>
namespace tensor {
enum class TDevice : int {
    CPU,
    GPU,
};

enum class TDtype : int {
    Float32,
    Float64,
    Complex64,
    Complex128,
};

// -----------------------------------------------------------------------------

template <TDtype dtype>
struct to_device;

template <>
struct to_device<TDtype::Float32> {
    using type = float;
};

template <>
struct to_device<TDtype::Float64> {
    using type = double;
};

template <>
struct to_device<TDtype::Complex64> {
    using type = std::complex<float>;
};

template <>
struct to_device<TDtype::Complex128> {
    using type = std::complex<double>;
};
template <TDtype dtype>
using to_device_t = to_device<dtype>;
}  // namespace tensor
#endif
