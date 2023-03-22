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

#ifndef MATH_PR_PARAMETER_RESOLVER_HPP_
#define MATH_PR_PARAMETER_RESOLVER_HPP_

#include <map>
#include <set>
#include <string>
#include <type_traits>

#include "math/tensor/ops.hpp"
#include "math/tensor/ops/memory_operator.hpp"
#include "math/tensor/tensor.hpp"
#include "math/tensor/traits.hpp"

namespace parameter {
namespace tn = tensor;
template <typename T>
std::set<T> operator-(const std::set<T>& s1, const std::set<T>& s2) {
    std::set<T> out;
    std::set_difference(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(out, out.begin()));
    return out;
}

template <typename T>
std::set<T> operator&(const std::set<T>& s1, const std::set<T>& s2) {
    std::set<T> out;
    std::set_intersection(s1.begin(), s1.end(), s2.begin(), s2.end(), std::inserter(out, out.begin()));
    return out;
}

struct ParameterResolver {
    using data_t = std::map<std::string, tn::Tensor>;
    data_t data_{};
    tn::Tensor const_value = tn::ops::init_with_value(static_cast<double>(0.0));
    std::set<std::string> no_grad_parameters_{};
    std::set<std::string> encoder_parameters_{};
    ParameterResolver() = default;

    template <typename T, typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>, tn::Tensor>>>
    explicit ParameterResolver(T const_value, const std::map<std::string, T>& data = {},
                               const std::set<std::string>& no_grad_parameters = {},
                               const std::set<std::string>& encoder_parameter = {}) {
        this->const_value = tn::ops::init_with_value(const_value);
        for (auto& [k, v] : data) {
            data_[k] = tn::ops::init_with_value(v);
        }
        this->no_grad_parameters_ = no_grad_parameters;
        this->encoder_parameters_ = encoder_parameter;
    }

    // -----------------------------------------------------------------------------
    tn::TDtype GetDtype() const;
    size_t Size() const;
    void CastTo(tn::TDtype dtype);

    template <typename T>
    void SetConst(const T& a) {
        tn::ops::set(&(this->const_value), a, 0);
    }
    std::string ToString() const;
    bool Contains(const std::string& key) const;
    bool NoGradContains(const std::string& key) const;
    bool EncoderContains(const std::string& key) const;
    std::set<std::string> GetAllParameters() const;
    std::set<std::string> GetRequiresGradParameters() const;
    std::set<std::string> GetAnsatzParameters() const;
    bool IsConst() const;
    bool IsNotZero() const;
    void SetItem(const std::string& key, const tn::Tensor& t);

    template <typename T>
    void SetItem(const std::string& key, const T& a) {
        this->SetItem(key, tn::ops::init_with_value(a, this->const_value.device));
    }

    template <typename T>
    ParameterResolver& operator+=(const T& value) {
        this->const_value += value;
        return *this;
    }

    ParameterResolver& operator+=(const ParameterResolver& rhs);

    template <typename T>
    ParameterResolver& operator-=(const T& value) {
        this->const_value -= value;
        return *this;
    }

    ParameterResolver& operator-=(const ParameterResolver& value);

    template <typename T>
    ParameterResolver& operator*=(const T& value) {
        this->const_value *= value;
        for (auto& [k, v] : this->data_) {
            v *= value;
        }
        return *this;
    }

    ParameterResolver& operator*=(const ParameterResolver& value);

    template <typename T>
    ParameterResolver& operator/=(const T& value) {
        this->const_value /= value;
        return *this;
    }

    ParameterResolver& operator/=(const ParameterResolver& value);
};

// -----------------------------------------------------------------------------
// TODO(xuxs):need cast
template <typename T>
ParameterResolver operator+(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    out += rhs;
    return out;
}

template <typename T>
ParameterResolver operator-(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    out -= rhs;
    return out;
}
template <typename T>
ParameterResolver operator*(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    out *= rhs;
    return out;
}
template <typename T>
ParameterResolver operator/(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    out /= rhs;
    return out;
}
}  // namespace parameter
std::ostream& operator<<(std::ostream& os, const parameter::ParameterResolver& pr);
#endif /* MATH_PR_PARAMETER_RESOLVER_HPP_ */
