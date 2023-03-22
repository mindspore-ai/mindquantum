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

#include <stdexcept>

#include "math/tensor/ops/advance_math.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
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

bool ParameterResolver::Contains(const std::string& key) const {
    return this->data_.find(key) != this->data_.end();
}

bool ParameterResolver::NoGradContains(const std::string& key) const {
    return this->no_grad_parameters_.find(key) != this->no_grad_parameters_.end();
}

bool ParameterResolver::EncoderContains(const std::string& key) const {
    return this->encoder_parameters_.find(key) != this->encoder_parameters_.end();
}

std::set<std::string> ParameterResolver::GetAllParameters() const {
    std::set<std::string> out{};
    for (auto& [k, v] : this->data_) {
        out.insert(k);
    }
    return out;
}

std::set<std::string> ParameterResolver::GetRequiresGradParameters() const {
    return this->GetAllParameters() - this->no_grad_parameters_;
}

std::set<std::string> ParameterResolver::GetAnsatzParameters() const {
    return this->GetAllParameters() - this->encoder_parameters_;
}

bool ParameterResolver::IsConst() const {
    for (auto& [k, v] : this->data_) {
        if (!tn::ops::is_all_zero(v)) {
            return false;
        }
    }
    return true;
}

bool ParameterResolver::IsNotZero() const {
    if (!tn::ops::is_all_zero(this->const_value)) {
        return true;
    }
    for (auto& [k, v] : this->data_) {
        if (!tn::ops::is_all_zero(v)) {
            return true;
        }
    }
    return false;
}

void ParameterResolver::SetItem(const std::string& key, const tn::Tensor& t) {
    if (t.dim != 1) {
        throw std::runtime_error("For SetItem of tensor, the given tensor should only has one value.");
    }
    this->data_[key] = tn::ops::cast_to(t, this->const_value.dtype);
}

// -----------------------------------------------------------------------------

ParameterResolver& ParameterResolver::operator+=(const ParameterResolver& rhs) {
    if ((this->encoder_parameters_.size() == 0) && (this->no_grad_parameters_.size() == 0)
        && (rhs.encoder_parameters_.size() == 0) && (rhs.no_grad_parameters_.size())) {
    } else {
        if (((this->encoder_parameters_ & rhs.GetAnsatzParameters()).size() != 0)
            || ((this->GetAnsatzParameters() & rhs.encoder_parameters_).size() != 0)) {
            throw std::runtime_error("encoder or ansatz property of parameter conflict.");
        }
        if (((this->no_grad_parameters_ & rhs.GetRequiresGradParameters()).size() != 0)
            || ((this->GetRequiresGradParameters() & rhs.no_grad_parameters_).size() != 0)) {
            throw std::runtime_error("gradient property of parameter conflict.");
        }
        for (auto& [key, value] : rhs.data_) {
            if (this->Contains(key)) {
                this->data_[key] += value;
            } else {
                this->SetItem(key, value);
                if (rhs.EncoderContains(key)) {
                    this->encoder_parameters_.insert(key);
                }
                if (rhs.NoGradContains(key)) {
                    this->no_grad_parameters_.insert(key);
                }
            }
        }
    }
    this->const_value += rhs.const_value;
}

ParameterResolver& ParameterResolver::operator-=(const ParameterResolver& rhs) {
    if ((this->encoder_parameters_.size() == 0) && (this->no_grad_parameters_.size() == 0)
        && (rhs.encoder_parameters_.size() == 0) && (rhs.no_grad_parameters_.size())) {
    } else {
        if (((this->encoder_parameters_ & rhs.GetAnsatzParameters()).size() != 0)
            || ((this->GetAnsatzParameters() & rhs.encoder_parameters_).size() != 0)) {
            throw std::runtime_error("encoder or ansatz property of parameter conflict.");
        }
        if (((this->no_grad_parameters_ & rhs.GetRequiresGradParameters()).size() != 0)
            || ((this->GetRequiresGradParameters() & rhs.no_grad_parameters_).size() != 0)) {
            throw std::runtime_error("gradient property of parameter conflict.");
        }
        for (auto& [key, value] : rhs.data_) {
            if (this->Contains(key)) {
                this->data_[key] -= value;
            } else {
                this->SetItem(key, 0.0 - value);
                if (rhs.EncoderContains(key)) {
                    this->encoder_parameters_.insert(key);
                }
                if (rhs.NoGradContains(key)) {
                    this->no_grad_parameters_.insert(key);
                }
            }
        }
    }
    this->const_value -= rhs.const_value;
}

ParameterResolver& ParameterResolver::operator*=(const ParameterResolver& rhs) {
    if (this->IsConst()) {
        for (auto& [k, v] : rhs.data_) {
            this->data_[k] = this->const_value * v;
            if (!this->Contains(k)) {
                if (rhs.EncoderContains(k)) {
                    this->encoder_parameters_.insert(k);
                }
                if (rhs.NoGradContains(k)) {
                    this->no_grad_parameters_.insert(k);
                }
            }
        }
    } else if (rhs.IsConst()) {
        for (auto& [k, v] : this->data_) {
            this->data_[k] *= rhs.const_value;
        }
    } else {
        throw std::runtime_error("Parameter resolver only support first order variable.");
    }

    this->const_value *= rhs.const_value;
    return *this;
}

ParameterResolver& ParameterResolver::operator/=(const ParameterResolver& rhs) {
    if (!rhs.IsConst()) {
        throw std::runtime_error("Cannot div a non constant ParameterResolver.");
    }
    for (auto& [k, v] : this->data_) {
        this->data_[k] /= rhs.const_value;
    }
    this->const_value /= rhs.const_value;
    return *this;
}

}  // namespace parameter
std::ostream& operator<<(std::ostream& os, const parameter::ParameterResolver& pr) {
    os << pr.ToString();
    return os;
}
