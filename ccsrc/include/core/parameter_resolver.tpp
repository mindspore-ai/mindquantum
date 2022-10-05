//   Copyright 2022 <Huawei Technologies Co., Ltd>
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

#ifndef PARAMETER_RESOLVER_TPP
#define PARAMETER_RESOLVER_TPP

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "config/logging.hpp"

#include "core/parameter_resolver.hpp"

//

// Needs to be *after* core/parameter_resolver.hpp
#include "config/format/parameter_resolver.hpp"

namespace mindquantum {
template <typename T>
template <typename U, typename>
ParameterResolver<T>::ParameterResolver(U&& value) : const_value(static_cast<T>(std::forward<U>(value))) {
}

template <typename T>
ParameterResolver<T>::ParameterResolver(const MST<T>& data, T value) : data_(data), const_value(value) {
    for (ITER(param, this->data_)) {
        if (param->first == "") {
            throw std::runtime_error("Parameter name cannot be empty.");
        }
    }
}

template <typename T>
ParameterResolver<T>::ParameterResolver(const MST<T>& data, T value, SS no_grad_parameters, SS encoder_parameters)
    : data_(data)
    , const_value(value)
    , no_grad_parameters_(std::move(no_grad_parameters))
    , encoder_parameters_(std::move(encoder_parameters)) {
}

template <typename T>
template <typename U, typename>
ParameterResolver<T>::ParameterResolver(const MST<U>& data) : ParameterResolver(data, static_cast<T>(0.)) {
}
template <typename T>
template <typename U, typename V, typename>
ParameterResolver<T>::ParameterResolver(const MST<U>& data, V value_const) : const_value(static_cast<T>(value_const)) {
    for (const auto& [key, value] : data) {
        data_.emplace(key, static_cast<T>(value));
    }
    for (ITER(param, this->data_)) {
        if (param->first == "") {
            throw std::runtime_error("Parameter name cannot be empty.");
        }
    }
}
template <typename T>
template <typename U, typename V, typename>
ParameterResolver<T>::ParameterResolver(const MST<U>& data, V&& value_const, SS no_grad_parameters,
                                        SS encoder_parameters)
    : const_value(static_cast<T>(std::forward<V>(value_const)))
    , no_grad_parameters_(std::move(no_grad_parameters))
    , encoder_parameters_(std::move(encoder_parameters)) {
    for (const auto& [key, value] : data) {
        data_.emplace(key, static_cast<T>(value));
    }
}
template <typename T>
ParameterResolver<T>::ParameterResolver(const std::string& param) {
    this->data_[param] = static_cast<T>(1);
}
template <typename T>
template <typename U, typename>
ParameterResolver<T>::ParameterResolver(const ParameterResolver<U>& other)
    : const_value{static_cast<T>(other.const_value)}
    , no_grad_parameters_{other.no_grad_parameters_}
    , encoder_parameters_{other.encoder_parameters_} {
    std::transform(begin(other.data_), end(other.data_), std::inserter(data_, end(data_)),
                   [](const auto& param) -> typename MST<T>::value_type {
                       return {param.first, static_cast<T>(param.second)};
                   });
}

// =============================================================================

template <typename T>
ParameterResolver<T>::operator T() const {
    if (!IsConst()) {
        throw std::runtime_error("ParameterResolver: cannot convert to const value since not const!");
    }
    return this->const_value;
}
template <typename T>
template <typename U, typename>
ParameterResolver<T>::operator traits::to_cmplx_type_t<U>() const {
    if (!IsConst()) {
        throw std::runtime_error("ParameterResolver: cannot convert to const value since not const!");
    }
    return static_cast<traits::to_cmplx_type_t<T>>(this->const_value);
}

template <typename T>
size_t ParameterResolver<T>::Size() const {
    return this->data_.size();
}

template <typename T>
void ParameterResolver<T>::SetConst(T const_value) {
    this->const_value = const_value;
}

template <typename T>
auto ParameterResolver<T>::NIndex(size_t n) const {
    if (n >= this->Size()) {
        throw std::runtime_error("ParameterResolver: Index out of range.");
    }
    auto index_p = this->data_.begin();
    std::advance(index_p, n);
    return index_p;
}

template <typename T>
std::string ParameterResolver<T>::GetKey(size_t index) const {
    return this->NIndex(index)->first;
}

template <typename T>
T ParameterResolver<T>::GetItem(const std::string& key) const {
    if (!this->Contains(key)) {
        throw std::runtime_error("parameter " + key + " not in this parameter resolver.");
    }
    return this->data_.at(key);
}

template <typename T>
T ParameterResolver<T>::GetItem(size_t index) const {
    return this->NIndex(index)->second;
}

template <typename T>
void ParameterResolver<T>::SetItem(const std::string& key, T value) {
    data_[key] = value;
}

template <typename T>
void ParameterResolver<T>::SetItems(const VS& name, const VT<T>& data) {
    if (name.size() != data.size()) {
        throw std::runtime_error("size of name and data mismatch.");
    }
    for (size_t i = 0; i < name.size(); i++) {
        this->SetItem(name[i], data[i]);
    }
}

template <typename T>
bool ParameterResolver<T>::IsConst() const {
    if (this->data_.size() == 0) {
        return true;
    }
    for (ITER(p, this->data_)) {
        if (!IsTwoNumberClose(p->second, static_cast<T>(0.0))) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool ParameterResolver<T>::IsNotZero() const {
    if (!IsTwoNumberClose(this->const_value, 0.0)) {
        return true;
    }
    for (ITER(p, this->data_)) {
        if (!IsTwoNumberClose(p->second, 0.0)) {
            return true;
        }
    }
    return false;
}

template <typename T>
bool ParameterResolver<T>::Contains(const std::string& key) const {
    return this->data_.find(key) != this->data_.end();
}

template <typename T>
bool ParameterResolver<T>::NoGradContains(const std::string& key) const {
    return this->no_grad_parameters_.find(key) != this->no_grad_parameters_.end();
}

template <typename T>
bool ParameterResolver<T>::EncoderContains(const std::string& key) const {
    return this->encoder_parameters_.find(key) != this->encoder_parameters_.end();
}

template <typename T>
SS ParameterResolver<T>::GetAllParameters() const {
    SS all_params = {};
    for (ITER(p, this->data_)) {
        all_params.insert(p->first);
    }
    return all_params;
}

template <typename T>
SS ParameterResolver<T>::GetRequiresGradParameters() const {
    return this->GetAllParameters() - this->no_grad_parameters_;
}

template <typename T>
SS ParameterResolver<T>::GetAnsatzParameters() const {
    return this->GetAllParameters() - this->encoder_parameters_;
}

template <typename T>
void ParameterResolver<T>::PrintInfo() const {
    std::cout << *this << std::endl;
}

template <typename T>
std::string ParameterResolver<T>::ToString() const {
    auto& pr = *this;
    std::ostringstream os;
    size_t i = 0;
    os << "{";
    for (ITER(p, pr.data_)) {
        os << "'" << p->first << "': " << p->second;
        if (i < pr.Size() - 1) {
            os << ", ";
        }
        i++;
    }
    os << "}, const: " << pr.const_value;
    return os.str();
}

template <typename T>
ParameterResolver<T> ParameterResolver<T>::Copy() {
    auto out = *this;
    return out;
}

template <typename T>
template <typename other_t>
bool ParameterResolver<T>::IsEqual(other_t value) const {
    if (this->Size() != 0) {
        return false;
    }
    return IsTwoNumberClose(this->const_value, value);
}

template <typename T>
template <typename other_t>
bool ParameterResolver<T>::IsEqual(const ParameterResolver<other_t> pr) const {
    if (!IsTwoNumberClose(this->const_value, pr.const_value)) {
        return false;
    }
    if (this->data_.size() != pr.data_.size()) {
        return false;
    }
    if (this->no_grad_parameters_.size() != pr.no_grad_parameters_.size()) {
        return false;
    }
    if (this->encoder_parameters_.size() != pr.encoder_parameters_.size()) {
        return false;
    }
    for (ITER(p, this->data_)) {
        if (!pr.Contains(p->first)) {
            return false;
        }
        if (!IsTwoNumberClose(p->second, pr.GetItem(p->first))) {
            return false;
        }
    }
    for (ITER(p, this->no_grad_parameters_)) {
        if (!pr.NoGradContains(*p)) {
            return false;
        }
    }
    for (ITER(p, this->encoder_parameters_)) {
        if (!pr.EncoderContains(*p)) {
            return false;
        }
    }
    return true;
}

template <typename T>
std::vector<std::string> ParameterResolver<T>::ParamsName() const {
    std::vector<std::string> pn;
    for (ITER(p, this->data_)) {
        pn.push_back(p->first);
    }
    return pn;
}

template <typename T>
std::vector<T> ParameterResolver<T>::ParaValue() const {
    std::vector<T> pv;
    for (ITER(p, this->data_)) {
        pv.push_back(p->second);
    }
    return pv;
}

template <typename T>
void ParameterResolver<T>::RequiresGrad() {
    this->no_grad_parameters_ = {};
}

template <typename T>
void ParameterResolver<T>::NoGrad() {
    this->no_grad_parameters_ = {};
    for (ITER(p, this->data_)) {
        this->no_grad_parameters_.insert(p->first);
    }
}

template <typename T>
void ParameterResolver<T>::RequiresGradPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->NoGradContains(name)) {
            this->no_grad_parameters_.erase(name);
        }
    }
}

template <typename T>
void ParameterResolver<T>::NoGradPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->Contains(name)) {
            this->no_grad_parameters_.insert(name);
        }
    }
}

template <typename T>
void ParameterResolver<T>::AnsatzPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->EncoderContains(name)) {
            this->encoder_parameters_.erase(name);
        }
    }
}

template <typename T>
void ParameterResolver<T>::EncoderPart(const std::vector<std::string>& names) {
    for (auto& name : names) {
        if (this->Contains(name)) {
            this->encoder_parameters_.insert(name);
        }
    }
}

template <typename T>
void ParameterResolver<T>::AsEncoder() {
    for (ITER(p, this->data_)) {
        this->encoder_parameters_.insert(p->first);
    }
}

template <typename T>
void ParameterResolver<T>::AsAnsatz() {
    this->encoder_parameters_ = {};
}

template <typename T>
template <typename other_t>
void ParameterResolver<T>::Update(const ParameterResolver<other_t>& other) {
    if ((this->encoder_parameters_.size() == 0) & (this->no_grad_parameters_.size() == 0)
        & (other.encoder_parameters_.size() == 0) & (other.no_grad_parameters_.size() == 0)) {
        for (ITER(p, other.data_)) {
            this->data_[p->first] = p->second;
        }
    } else {
        if (((this->encoder_parameters_ & other.GetAnsatzParameters()).size() != 0)
            | ((this->GetAnsatzParameters() & other.encoder_parameters_).size() != 0)) {
            throw std::runtime_error("encoder or ansatz property of parameter conflict.");
        }
        if (((this->no_grad_parameters_ & other.GetRequiresGradParameters()).size() != 0)
            | ((this->GetRequiresGradParameters() & other.no_grad_parameters_).size() != 0)) {
            throw std::runtime_error("gradient property of parameter conflict.");
        }

        for (ITER(p, other.data_)) {
            auto& key = p->first;
            const auto& value = p->second;
            if (this->Contains(key)) {
                this->data_[key] = value;
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
    }
    this->const_value = other.const_value;
}

template <typename T>
ParameterResolver<T> ParameterResolver<T>::Conjugate() const {
    auto out = *this;
    for (ITER(p, out.data_)) {
        out.data_[p->first] = Conj(p->second);
    }
    out.const_value = Conj(out.const_value);
    return out;
}

template <typename T>
ParameterResolver<T> ParameterResolver<T>::Combination(const ParameterResolver<T>& pr) const {
    auto c = this->const_value;
    for (ITER(p, this->data_)) {
        c += p->second * pr.GetItem(p->first);
    }
    return ParameterResolver<T>(c);
}

template <typename T>
auto ParameterResolver<T>::Real() const {
    using real_t = typename traits::to_real_type_t<T>;
    ParameterResolver<real_t> pr = {};
    pr.const_value = std::real(this->const_value);
    for (ITER(p, this->data_)) {
        auto& key = p->first;
        pr.data_[p->first] = std::real(p->second);
        if (this->EncoderContains(key)) {
            pr.encoder_parameters_.insert(key);
        }
        if (this->NoGradContains(key)) {
            pr.no_grad_parameters_.insert(key);
        }
    }
    return pr;
}
template <typename T>
void ParameterResolver<T>::KeepReal() {
    this->const_value = std::real(this->const_value);
    for (auto& [name, value] : this->data_) {
        value = std::real(value);
    }
}
template <typename T>
void ParameterResolver<T>::KeepImag() {
    this->const_value = std::imag(this->const_value);
    for (auto& [name, value] : this->data_) {
        value = std::imag(value);
    }
}
template <typename T>
auto ParameterResolver<T>::Imag() const {
    using real_t = typename traits::to_real_type_t<T>;
    ParameterResolver<real_t> pr = {};
    pr.const_value = std::imag(this->const_value);
    for (ITER(p, this->data_)) {
        auto& key = p->first;
        pr.data_[p->first] = std::imag(p->second);
        if (this->EncoderContains(key)) {
            pr.encoder_parameters_.insert(key);
        }
        if (this->NoGradContains(key)) {
            pr.no_grad_parameters_.insert(key);
        }
    }
    return pr;
}

template <typename T>
T ParameterResolver<T>::Pop(const std::string& key) {
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

template <typename T>
bool ParameterResolver<T>::IsHermitian() const {
    if constexpr (traits::is_complex_v<T>) {
        return *this == this->Conjugate();
    } else {
        return true;
    }
}

template <typename T>
bool ParameterResolver<T>::IsAntiHermitian() const {
    if constexpr (traits::is_complex_v<T>) {
        return *this == -this->Conjugate();
    } else {
        return false;
    }
}

template <typename T>
template <typename number_t>
auto ParameterResolver<T>::Cast() const {
    if constexpr (std::is_same_v<T, number_t>) {
        return *this;
    } else {
        ParameterResolver<number_t> out;
        for (ITER(param, this->data_)) {
            const auto& key = param->first;
            const auto& value = param->second;
            out.data_[param->first] = static_cast<number_t>(value);
            if (this->EncoderContains(key)) {
                out.encoder_parameters_.insert(key);
            }
            if (this->NoGradContains(key)) {
                out.no_grad_parameters_.insert(key);
            }
        }
        out.const_value = static_cast<number_t>(const_value);
        return out;
    }
}
// =============================================================================
// Arithmetic operators

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator+=(other_t value) {
    this->const_value += value;
    return *this;
}

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator+=(const ParameterResolver<other_t>& other) {
    MQ_TRACE("{} ({}) += {} ({})", *this, this->IsConst(), other, other.IsConst());

    if ((this->encoder_parameters_.size() == 0) && (this->no_grad_parameters_.size() == 0)
        && (other.encoder_parameters_.size() == 0) && (other.no_grad_parameters_.size() == 0)) {
        MQ_TRACE("other.data_ = {}", other.data_);
        for (ITER(p, other.data_)) {
            this->data_[p->first] += p->second;
        }
    } else {
        if (((this->encoder_parameters_ & other.GetAnsatzParameters()).size() != 0)
            || ((this->GetAnsatzParameters() & other.encoder_parameters_).size() != 0)) {
            throw std::runtime_error("encoder or ansatz property of parameter conflict.");
        }
        if (((this->no_grad_parameters_ & other.GetRequiresGradParameters()).size() != 0)
            || ((this->GetRequiresGradParameters() & other.no_grad_parameters_).size() != 0)) {
            throw std::runtime_error("gradient property of parameter conflict.");
        }

        MQ_TRACE("other.data_ = {}", other.data_);
        for (ITER(p, other.data_)) {
            auto& key = p->first;
            const auto& value = p->second;
            if (this->Contains(key)) {
                MQ_TRACE("data_[{}] ({}) += {}", key, this->data_[key], value);
                this->data_[key] += value;
            } else {
                MQ_TRACE("SetItem({}, {})", key, value);
                this->SetItem(key, value);
                if (other.EncoderContains(key)) {
                    this->encoder_parameters_.insert(key);
                }
                if (other.NoGradContains(key)) {
                    this->no_grad_parameters_.insert(key);
                }
            }
        }
    }
    MQ_TRACE("{} += {}", this->const_value, other.const_value);
    this->const_value += other.const_value;
    return *this;
}

template <typename T>
ParameterResolver<T> ParameterResolver<T>::operator-() const {
    auto out = *this;
    out.const_value = -out.const_value;
    for (ITER(p, out.data_)) {
        out.data_[p->first] = -out.data_[p->first];
    }
    return out;
}

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator-=(other_t value) {
    *this += (-value);
    return *this;
}

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator-=(const ParameterResolver<other_t>& other) {
    *this += (-other);
    return *this;
}

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator*=(other_t value) {
    this->const_value *= value;
    for (ITER(p, this->data_)) {
        this->data_[p->first] *= value;
    }
    return *this;
}

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator*=(const ParameterResolver<other_t>& other) {
    MQ_TRACE("{} ({}) *= {} ({})", *this, this->IsConst(), other, other.IsConst());

    /* WARNING: This algorithm is *not* symmetric!
     * e.g. if: X = {'b': (1e-09,0)}, const: (0,0)
     *          Y = {}, const: (-0,-1)
     *      then:
     *          X *= Y -> {'b': (1e-09,0)}, const: (0,-0)
     *          Y *= X -> {'b': (0,-1e-09)}, const: (0,0)
     *      with
     *          X == Y (since PRECISION is 1.e-8 by default)
     *
     * This is because in both cases, the first branch below is taken (since IsConst() evaluates to true in both cases).
     * However, in the first case, other.data_ is empty and in the second case it is not.
     */
    if (this->IsConst()) {
        MQ_TRACE("other.data_ = {}", other.data_);
        for (ITER(p, other.data_)) {
            MQ_TRACE("data_[{}] = {} * {}", p->first, this->const_value, p->second);
            this->data_[p->first] = this->const_value * p->second;
            if (!this->Contains(p->first)) {
                if (other.EncoderContains(p->first)) {
                    this->encoder_parameters_.insert(p->first);
                }
                if (other.NoGradContains(p->first)) {
                    this->no_grad_parameters_.insert(p->first);
                }
            }
        }
    } else if (other.IsConst()) {
        MQ_TRACE("this->data_ = {}", this->data_);
        for (ITER(p, this->data_)) {
            MQ_TRACE("data_[{}] ({}) *= {}", p->first, this->data_[p->first], other.const_value);
            this->data_[p->first] *= other.const_value;
        }
    } else {
        throw std::runtime_error("Parameter resolver only support first order variable.");
    }

    MQ_TRACE("const_value: {} *= {}", this->const_value, other.const_value);
    this->const_value *= other.const_value;
    return *this;
}

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator/=(other_t value) {
    this->const_value /= value;
    for (ITER(p, this->data_)) {
        this->data_[p->first] /= value;
    }
    return *this;
}

template <typename T>
template <typename other_t>
ParameterResolver<T>& ParameterResolver<T>::operator/=(const ParameterResolver<other_t>& other) {
    if (!other.IsConst()) {
        throw std::runtime_error("Cannot div a non constant ParameterResolver.");
    }
    for (ITER(p, this->data_)) {
        this->data_[p->first] /= other.const_value;
    }
    this->const_value /= other.const_value;
    return *this;
}

// =============================================================================
}  // namespace mindquantum

#endif /* PARAMETER_RESOLVER_TPP */
