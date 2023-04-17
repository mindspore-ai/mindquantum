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

#include <iostream>
#include <stdexcept>

#include "math/tensor/ops/advance_math.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace parameter {
ParameterResolver::ParameterResolver(const tn::Tensor& const_value) : const_value(const_value) {
}

ParameterResolver::ParameterResolver(const std::string& key, const tn::Tensor& const_value, tn::TDtype dtype) {
    this->const_value = const_value.astype(dtype);
    this->data_[key] = tn::ops::ones(1, dtype);
}

ParameterResolver::ParameterResolver(const std::map<std::string, tn::Tensor>& data, const tn::Tensor& const_value,
                                     tn::TDtype dtype) {
    this->const_value = const_value.astype(dtype);
    for (auto& [k, v] : data) {
        this->data_[k] = v.astype(dtype);
    }
}

tn::TDtype ParameterResolver::GetDtype() const {
    return this->const_value.dtype;
}

size_t ParameterResolver::Size() const {
    return this->data_.size();
}

void ParameterResolver::CastTo(tn::TDtype dtype) {
    if (dtype == this->GetDtype()) {
        return;
    }
    this->const_value = tn::ops::cast_to(this->const_value, dtype);
    for (auto& [k, v] : this->data_) {
        v = tn::ops::cast_to(v, dtype);
    }
}
void ParameterResolver::SetConstValue(const tn::Tensor& a) {
    this->const_value = a.astype(this->const_value.dtype);
}

tn::Tensor ParameterResolver::GetConstValue() const {
    return this->const_value;
}

std::string ParameterResolver::ToString() const {
    std::string out = "ParameterResolver(dtype: " + tensor::to_string(this->const_value.dtype) + ",";
    if (this->data_.size() == 0) {
        out += " const: " + tn::ops::to_string(this->const_value, true);
        out += ")";
        return out;
    }
    out += "\n";
    if (this->data_.size() == 0) {
        out += "  data: [],\n";
    } else {
        out += "  data: [\n";
        int i = 0;
        for (auto& [k, v] : this->data_) {
            out += "         " + k + ": " + tn::ops::to_string(v, true);
            i += 1;
            if (i != this->data_.size()) {
                out += ",";
            }
            out += "\n";
        }
        out += "  ],\n";
    }
    out += "  const: " + tn::ops::to_string(this->const_value, true);
    if (this->no_grad_parameters_.size() != 0) {
        out += ",\n  no grad parameters: {";
        int i = 0;
        for (auto& v : this->no_grad_parameters_) {
            out += v;
            i += 1;
            if (i != this->data_.size()) {
                out += ", ";
            }
        }
        out += "}";
    }
    if (this->encoder_parameters_.size() != 0) {
        out += ",\n  encoder parameters: {";
        int i = 0;
        for (auto& v : this->encoder_parameters_) {
            out += v;
            i += 1;
            if (i != this->data_.size()) {
                out += ", ";
            }
        }
        out += "}";
    }
    out += "\n)";
    return out;
}

bool ParameterResolver::Contains(const std::string& key) const {
    auto res = this->data_.find(key) != this->data_.end();
    return res;
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
std::vector<std::string> ParameterResolver::subs(const ParameterResolver& other) {
    std::vector<std::string> will_pop;
    auto origin_dtype = this->GetDtype();
    for (auto& [k, v] : this->data_) {
        if (other.Contains(k)) {
            this->const_value = this->const_value + v * other.data_.at(k);
            will_pop.push_back(k);
        }
    }
    if (will_pop.size() != 0) {
        for (auto& k : will_pop) {
            this->Pop(k);
        }
        auto new_dtype = this->GetDtype();
        if (origin_dtype != new_dtype) {
            for (auto& [k, v] : this->data_) {
                v = v.astype(new_dtype);
            }
        }
    }
    return will_pop;
}

tn::Tensor ParameterResolver::GetItem(const std::string& key) const {
    if (!this->Contains(key)) {
        throw std::runtime_error("parameter " + key + " not in this parameter resolver.");
    }
    return this->data_.at(key);
}

// -----------------------------------------------------------------------------

ParameterResolver& ParameterResolver::operator+=(const ParameterResolver& rhs) {
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
    this->const_value += rhs.const_value;
    return *this;
}

ParameterResolver& ParameterResolver::operator-=(const ParameterResolver& rhs) {
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
            this->SetItem(key, static_cast<float>(0.0) - value);
            if (rhs.EncoderContains(key)) {
                this->encoder_parameters_.insert(key);
            }
            if (rhs.NoGradContains(key)) {
                this->no_grad_parameters_.insert(key);
            }
        }
    }
    this->const_value -= rhs.const_value;
    return *this;
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

bool ParameterResolver::operator==(const ParameterResolver& value) {
    if (this->data_.size() != value.data_.size()) {
        return false;
    }
    if (!tn::ops::all_equal_to(this->const_value, value.const_value)) {
        return false;
    }
    for (auto& [k, v] : this->data_) {
        if (!value.Contains(k)) {
            return false;
        }
        if (!tn::ops::all_equal_to(v, this->data_[k])) {
            return false;
        }
    }
    return true;
}

bool ParameterResolver::operator!=(const ParameterResolver& value) {
    return !(*this == value);
}

ParameterResolver ParameterResolver::operator-() const {
    auto out = *this;
    out.const_value = 0.0 - out.const_value;
    for (auto& [k, v] : out.data_) {
        v = 0.0 - v;
    }
    return out;
}

ParameterResolver operator+(const ParameterResolver& lhs, const ParameterResolver& rhs) {
    auto origin_type = lhs.const_value.dtype;
    auto out = lhs;
    auto new_const = lhs.const_value + rhs.const_value;
    auto new_type = new_const.dtype;
    if (new_type != origin_type) {
        out.CastTo(new_type);
    }
    out += rhs;
    return out;
}

ParameterResolver operator-(const ParameterResolver& lhs, const ParameterResolver& rhs) {
    auto origin_type = lhs.const_value.dtype;
    auto out = lhs;
    auto new_const = lhs.const_value + rhs.const_value;
    auto new_type = new_const.dtype;
    if (new_type != origin_type) {
        out.CastTo(new_type);
    }
    out -= rhs;
    return out;
}

ParameterResolver operator*(const ParameterResolver& lhs, const ParameterResolver& rhs) {
    auto origin_type = lhs.const_value.dtype;
    auto out = lhs;
    auto new_const = lhs.const_value + rhs.const_value;
    auto new_type = new_const.dtype;
    if (new_type != origin_type) {
        out.CastTo(new_type);
    }
    out *= rhs;
    return out;
}

ParameterResolver operator/(const ParameterResolver& lhs, const ParameterResolver& rhs) {
    auto origin_type = lhs.const_value.dtype;
    auto out = lhs;
    auto new_const = lhs.const_value + rhs.const_value;
    auto new_type = new_const.dtype;
    if (new_type != origin_type) {
        out.CastTo(new_type);
    }
    out /= rhs;
    return out;
}

// -----------------------------------------------------------------------------

std::vector<std::string> ParameterResolver::ParamsName() const {
    std::vector<std::string> out = {};
    for (auto& [k, v] : this->data_) {
        out.push_back(k);
    }
    return out;
}

std::vector<tn::Tensor> ParameterResolver::ParaValue() const {
    std::vector<tn::Tensor> out = {};
    for (auto& [k, v] : this->data_) {
        out.push_back(v);
    }
    return out;
}
auto ParameterResolver::ParaData() const -> data_t {
    return this->data_;
}

void ParameterResolver::RequiresGrad() {
    this->no_grad_parameters_ = {};
}

void ParameterResolver::NoGrad() {
    this->no_grad_parameters_ = {};
    for (auto& [k, v] : this->data_) {
        this->no_grad_parameters_.insert(k);
    }
}

void ParameterResolver::RequiresGradPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->NoGradContains(name)) {
            this->no_grad_parameters_.erase(name);
        }
    }
}

void ParameterResolver::NoGradPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->Contains(name)) {
            this->no_grad_parameters_.insert(name);
        }
    }
}

void ParameterResolver::AnsatzPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->EncoderContains(name)) {
            this->encoder_parameters_.erase(name);
        }
    }
}

void ParameterResolver::EncoderPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->Contains(name)) {
            this->encoder_parameters_.insert(name);
        }
    }
}

void ParameterResolver::AsEncoder() {
    for (auto& [k, v] : this->data_) {
        this->encoder_parameters_.insert(k);
    }
}

void ParameterResolver::AsAnsatz() {
    this->encoder_parameters_ = {};
}

void ParameterResolver::Update(const ParameterResolver& other) {
    if (((this->encoder_parameters_ & other.GetAnsatzParameters()).size() != 0)
        | ((this->GetAnsatzParameters() & other.encoder_parameters_).size() != 0)) {
        throw std::runtime_error("encoder or ansatz property of parameter conflict.");
    }
    if (((this->no_grad_parameters_ & other.GetRequiresGradParameters()).size() != 0)
        | ((this->GetRequiresGradParameters() & other.no_grad_parameters_).size() != 0)) {
        throw std::runtime_error("gradient property of parameter conflict.");
    }

    for (auto& [key, value] : other.data_) {
        if (this->Contains(key)) {
            this->SetItem(key, value);
        } else {
            this->SetItem(key, value);
            if (other.EncoderContains(key)) {
                this->encoder_parameters_.insert(key);
            }
            if (other.NoGradContains(key)) {
                this->no_grad_parameters_.insert(key);
            }
        }
    }
    this->const_value = other.const_value;
}

ParameterResolver ParameterResolver::Conjugate() const {
    auto out = *this;
    out.const_value = out.const_value.conj();
    for (auto& [k, v] : out.data_) {
        v = v.conj();
    }
    return out;
}

ParameterResolver ParameterResolver::Combination(const ParameterResolver& pr) const {
    auto c = this->const_value;
    for (auto& [k, v] : this->data_) {
        c += v * pr.GetItem(k);
    }
    auto out = ParameterResolver();
    out.const_value = c;
    return out;
}

ParameterResolver ParameterResolver::Real() const {
    auto out = *this;
    out.const_value = out.const_value.real();
    for (auto& [k, v] : out.data_) {
        v = v.real();
    }
    return out;
}

void ParameterResolver::KeepReal() {
    this->const_value = this->const_value.real();
    for (auto& [k, v] : this->data_) {
        v = v.real();
    }
}

void ParameterResolver::KeepImag() {
    this->const_value = this->const_value.imag();
    for (auto& [k, v] : this->data_) {
        v = v.imag();
    }
}

ParameterResolver ParameterResolver::Imag() const {
    auto out = *this;
    out.const_value = out.const_value.imag();
    for (auto& [k, v] : out.data_) {
        v = v.imag();
    }
    return out;
}

tn::Tensor ParameterResolver::Pop(const std::string& key) {
    auto out = this->GetItem(key);
    this->data_.erase(key);
    if (this->EncoderContains(key)) {
        this->encoder_parameters_.erase(key);
    }
    if (this->NoGradContains(key)) {
        this->no_grad_parameters_.erase(key);
    }
    return out;
}

bool ParameterResolver::IsHermitian() const {
    return !(*this - this->Conjugate()).IsNotZero();
}

bool ParameterResolver::IsAntiHermitian() const {
    return !(*this + this->Conjugate()).IsNotZero();
}

bool ParameterResolver::HasRequireGradParams() {
    return this->data_.size() > this->no_grad_parameters_.size();
}
std::map<std::string, size_t> GetRequiresGradParameters(const std::vector<ParameterResolver>& prs) {
    std::map<std::string, size_t> title = {};
    size_t idx = 0;
    for (auto& pr : prs) {
        for (auto& name : pr.GetRequiresGradParameters()) {
            if (!title.count(name)) {
                title[name] = idx;
                ++idx;
            }
        }
    }
    return title;
}

std::pair<std::map<std::string, size_t>, tensor::Matrix> Jacobi(const std::vector<ParameterResolver>& prs) {
    auto title = GetRequiresGradParameters(prs);
    auto upper_t = tensor::TDtype::Float32;
    for (auto& pr : prs) {
        upper_t = tensor::upper_type_v(upper_t, pr.GetDtype());
    }
    auto jacobi = tensor::ops::zeros(prs.size() * title.size());
    for (size_t i = 0; i < prs.size(); i++) {
        for (auto& name : prs[i].GetRequiresGradParameters()) {
            tensor::ops::cpu::set(&jacobi, prs[i].data_.at(name), i * title.size() + title.at(name));
        }
    }
    return {title, tensor::Matrix(std::move(jacobi), prs.size(), title.size())};
}
}  // namespace parameter
std::ostream& operator<<(std::ostream& os, const parameter::ParameterResolver& pr) {
    os << pr.ToString();
    return os;
}
