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

TDtype ToRealType(TDtype dtype) {
    switch (dtype) {
        case TDtype::Float32:
        case TDtype::Float64:
            return dtype;
        case TDtype::Complex64:
            return to_real_dtype_t<TDtype::Complex64>;
        case TDtype::Complex128:
            return to_real_dtype_t<TDtype::Complex128>;
    }
}

TDtype ToComplexType(TDtype dtype) {
    switch (dtype) {
        case TDtype::Complex128:
        case TDtype::Complex64:
            return dtype;
        case TDtype::Float32:
            return TDtype::Complex64;
        case TDtype::Float64:
            return TDtype::Complex128;
    }
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

// -----------------------------------------------------------------------------

int bit_size(TDtype dtype) {
    switch (dtype) {
        case (TDtype::Float32): {
            return sizeof(to_device_t<TDtype::Float32>);
        }
        case (TDtype::Float64): {
            return sizeof(to_device_t<TDtype::Float64>);
        }
        case (TDtype::Complex64): {
            return sizeof(to_device_t<TDtype::Complex64>);
        }
        case (TDtype::Complex128): {
            return sizeof(to_device_t<TDtype::Complex128>);
        }
    }
    return 0;
}

// -----------------------------------------------------------------------------
#define UPPER_TYPE(dtype)                                                                                              \
    case dtype: {                                                                                                      \
        switch (t1) {                                                                                                  \
            case TDtype::Float32:                                                                                      \
                return upper_type<TDtype::Float32, dtype>::get();                                                      \
            case TDtype::Float64:                                                                                      \
                return upper_type<TDtype::Float64, dtype>::get();                                                      \
            case TDtype::Complex128:                                                                                   \
                return upper_type<TDtype::Complex128, dtype>::get();                                                   \
            case TDtype::Complex64:                                                                                    \
                return upper_type<TDtype::Complex64, dtype>::get();                                                    \
        }                                                                                                              \
        break;                                                                                                         \
    }

TDtype upper_type_v(TDtype t1, TDtype t2) {
    switch (t2) {
        UPPER_TYPE(TDtype::Float32);
        UPPER_TYPE(TDtype::Float64);
        UPPER_TYPE(TDtype::Complex128);
        UPPER_TYPE(TDtype::Complex64);
    }
}
}  // namespace tensor
