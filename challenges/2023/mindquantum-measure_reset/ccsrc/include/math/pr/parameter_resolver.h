/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#ifndef MATH_PR_PARAMETER_RESOLVER_HPP_
#define MATH_PR_PARAMETER_RESOLVER_HPP_

#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "config/config.h"
#include "math/tensor/matrix.h"
#include "math/tensor/ops.h"
#include "math/tensor/ops/memory_operator.h"
#include "math/tensor/ops_cpu/memory_operator.h"
#include "math/tensor/tensor.h"
#include "math/tensor/traits.h"

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

    template <typename T,
              typename
              = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<T>,
                                                 tn::Tensor> && !std::is_same_v<std::remove_cvref_t<T>, std::string>>>
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

    explicit ParameterResolver(const std::string& key, const tn::Tensor& const_value = tn::ops::zeros(1),
                               tn::TDtype dtype = tn::TDtype::Float64);
    explicit ParameterResolver(const std::map<std::string, tn::Tensor>& data,
                               const tn::Tensor& const_value = tn::ops::zeros(1),
                               tn::TDtype dtype = tn::TDtype::Float64);
    explicit ParameterResolver(const tn::Tensor& const_value);
    ParameterResolver(const ParameterResolver& other) = default;
    ParameterResolver& operator=(const ParameterResolver& t) = default;
    // -----------------------------------------------------------------------------
    tn::TDtype GetDtype() const;
    size_t Size() const;
    void CastTo(tn::TDtype dtype);

    template <typename T>
    void SetConst(const T& a) {
        tn::ops::set(&(this->const_value), a, 0);
    }
    void SetConstValue(const tn::Tensor& a);
    tn::Tensor GetConstValue() const;
    std::string ToString() const;
    bool Contains(const std::string& key) const;
    bool NoGradContains(const std::string& key) const;
    bool EncoderContains(const std::string& key) const;
    std::set<std::string> GetAllParameters() const;
    std::set<std::string> GetRequiresGradParameters() const;
    std::set<std::string> GetAnsatzParameters() const;
    bool IsConst() const;
    bool IsNotZero() const;
    std::vector<std::string> subs(const ParameterResolver& other);
    template <typename T>
    void SetItem(const std::string& key, const T& a) {
        if constexpr (std::is_same_v<T, tn::Tensor>) {
            if (a.dim != 1) {
                throw std::runtime_error("For SetItem of tensor, the given tensor should only has one value.");
            }
            this->data_[key] = a.astype(this->const_value.dtype);
        } else {
            this->SetItem(key, tn::ops::init_with_value(a, this->const_value.device).astype(this->GetDtype()));
        }
    }

    template <typename T>
    void SetItems(const std::vector<std::string>& keys, const std::vector<T>& values) {
        if (keys.size() != values.size()) {
            throw std::runtime_error("SetItems args dimension mismatch.");
        }
        for (size_t i = 0; i < keys.size(); i++) {
            this->SetItem(keys[i], values[i]);
        }
    }

    tn::Tensor GetItem(const std::string& key) const;

    std::vector<std::string> ParamsName() const;

    std::vector<tn::Tensor> ParaValue() const;
    data_t ParaData() const;

    void RequiresGrad();

    void NoGrad();

    void RequiresGradPart(const std::vector<std::string>& names);

    void NoGradPart(const std::vector<std::string>& names);

    void AnsatzPart(const std::vector<std::string>& names);

    void EncoderPart(const std::vector<std::string>& names);

    void AsEncoder();
    void AsAnsatz();

    void Update(const ParameterResolver& other);

    ParameterResolver Conjugate() const;

    ParameterResolver Combination(const ParameterResolver& pr) const;

    ParameterResolver Real() const;
    void KeepReal();
    void KeepImag();
    ParameterResolver Imag() const;

    tn::Tensor Pop(const std::string& key);

    bool IsHermitian() const;

    bool IsAntiHermitian() const;

    bool HasRequireGradParams();

    // -----------------------------------------------------------------------------

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

    bool operator==(const ParameterResolver& value);
    bool operator!=(const ParameterResolver& value);
    ParameterResolver operator-() const;
};

// -----------------------------------------------------------------------------

template <typename T>
ParameterResolver operator+(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    auto ori_dtype = out.const_value.dtype;
    auto new_const = out.const_value + rhs;
    auto new_dtype = new_const.dtype;
    if (ori_dtype != new_dtype) {
        out.CastTo(new_dtype);
    }
    out += rhs;
    return out;
}

template <typename T>
ParameterResolver operator-(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    auto ori_dtype = out.const_value.dtype;
    auto new_const = out.const_value + rhs;
    auto new_dtype = new_const.dtype;
    if (ori_dtype != new_dtype) {
        out.CastTo(new_dtype);
    }
    out -= rhs;
    return out;
}

template <typename T>
ParameterResolver operator*(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    auto ori_dtype = out.const_value.dtype;
    auto new_const = out.const_value + rhs;
    auto new_dtype = new_const.dtype;
    if (ori_dtype != new_dtype) {
        out.CastTo(new_dtype);
    }
    out *= rhs;
    return out;
}

template <typename T>
ParameterResolver operator/(const ParameterResolver& lhs, T rhs) {
    auto out = lhs;
    auto ori_dtype = out.const_value.dtype;
    auto new_const = out.const_value + rhs;
    auto new_dtype = new_const.dtype;
    if (ori_dtype != new_dtype) {
        out.CastTo(new_dtype);
    }
    out /= rhs;
    return out;
}

ParameterResolver operator+(const ParameterResolver& lhs, const ParameterResolver& rhs);
ParameterResolver operator-(const ParameterResolver& lhs, const ParameterResolver& rhs);
ParameterResolver operator*(const ParameterResolver& lhs, const ParameterResolver& rhs);
ParameterResolver operator/(const ParameterResolver& lhs, const ParameterResolver& rhs);

std::map<std::string, size_t> GetRequiresGradParameters(const std::vector<ParameterResolver>& prs);
std::pair<std::map<std::string, size_t>, tensor::Matrix> Jacobi(const std::vector<ParameterResolver>& prs);
}  // namespace parameter
std::ostream& operator<<(std::ostream& os, const parameter::ParameterResolver& pr);
#endif /* MATH_PR_PARAMETER_RESOLVER_HPP_ */
