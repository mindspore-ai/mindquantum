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

#include "math/tensor/traits.hpp"

#include <stdexcept>

namespace tensor {
std::string to_string(TDtype dtype) {
    switch (dtype) {
        case (TDtype::Float32):
            return to_string<TDtype::Float32>();
        case (TDtype::Float64):
            return to_string<TDtype::Float64>();
        case (TDtype::Complex64):
            return to_string<TDtype::Complex64>();
        case (TDtype::Complex128):
            return to_string<TDtype::Complex128>();
        default:
            throw std::runtime_error("Unknown dtype.");
    }
    return "";
}

// -----------------------------------------------------------------------------

std::string to_string(TDevice device) {
    switch (device) {
        case TDevice::CPU:
            return to_string<TDevice::CPU>();
        case TDevice::GPU:
            return to_string<TDevice::GPU>();
        default:
            throw std::runtime_error("Unknown device.");
    }
    return "";
}
}  // namespace tensor
