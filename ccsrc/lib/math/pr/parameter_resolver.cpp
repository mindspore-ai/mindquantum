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

#include "math/pr/parameter_resolver.hpp"

#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/traits.hpp"

namespace parameter {
tn::TDtype ParameterResolver::GetDtype() const {
    return this->const_value.dtype;
}

size_t ParameterResolver::Size() const {
    return this->data_.size();
}

void ParameterResolver::CastTo(tn::TDtype dtype) {
    this->const_value = tn::ops::cast_to(this->const_value, dtype);
    for (auto& [k, v] : this->data_) {
        v = tn::ops::cast_to(v, dtype);
    }
}

std::string ParameterResolver::ToString() const {
    std::string out = "ParameterResolver (dtype: " + tensor::to_string(this->const_value.dtype) + ",\n";
    if (this->data_.size() == 0) {
        out += "  data: []\n";
    } else {
        out += "  data: [\n";
        int i = 0;
        for (auto& [k, v] : this->data_) {
            out += "         " + k + ": " + tn::ops::to_string(v, true);
            i += 1;
            if (i != this->data_.size()) {
                out += ",";
            }
            out += "\n  ]\n";
        }
    }
    out += "  const value: " + tn::ops::to_string(this->const_value, true) + "\n";
    out += ")";
    return out;
}
}  // namespace parameter
std::ostream& operator<<(std::ostream& os, const parameter::ParameterResolver& pr) {
    os << pr.ToString();
    return os;
}
