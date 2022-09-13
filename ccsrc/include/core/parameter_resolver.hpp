/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDQUANTUM_PR_PARAMETER_RESOLVER_H_
#define MINDQUANTUM_PR_PARAMETER_RESOLVER_H_

#include <algorithm>
#include <complex>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include <nlohmann/json.hpp>

#include "config/common_type.hpp"
#include "config/real_cast.hpp"
#include "config/type_promotion.hpp"
#include "config/type_traits.hpp"

#include "core/utils.hpp"

// =============================================================================

namespace mindquantum {
namespace details {
template <RealCastType cast_type, typename float_t>
struct real_cast_impl<cast_type, ParameterResolver<float_t>> {
    using type = ParameterResolver<float_t>;
    static auto apply(const type& coeff) {
        return coeff;
    }
};
template <RealCastType cast_type, typename float_t>
struct real_cast_impl<cast_type, ParameterResolver<std::complex<float_t>>> {
    using type = ParameterResolver<std::complex<float_t>>;
    static auto apply(const type& coeff) {
        constexpr auto is_complex_valued = traits::is_complex_v<float_t>;
        ParameterResolver<traits::to_real_type_t<float_t>> new_coeff;
        if constexpr (cast_type == RealCastType::REAL) {
            new_coeff.const_value = coeff.const_value.real();
            for (auto param = coeff.data_.begin(); param != coeff.data_.end(); ++param) {
                new_coeff.data_[param->first] = param->second.real();
            }
        } else {
            new_coeff.const_value = coeff.const_value.imag();
            for (auto param = coeff.data_.begin(); param != coeff.data_.end(); ++param) {
                new_coeff.data_[param->first] = param->second.imag();
            }
        }
        return new_coeff;
    }
};
}  // namespace details

// =============================================================================

namespace traits {
template <typename float_t>
struct to_real_type<ParameterResolver<float_t>> {
    using type = ParameterResolver<to_real_type_t<float_t>>;
};
template <typename float_t>
struct to_cmplx_type<ParameterResolver<float_t>> {
    using type = ParameterResolver<to_cmplx_type_t<float_t>>;
};

template <typename T>
struct type_promotion<ParameterResolver<T>> : details::type_promotion_encapsulated_fp<T, ParameterResolver> {};

template <typename float_t, typename U>
struct common_type<ParameterResolver<float_t>, U> {
    using type = ParameterResolver<std::common_type_t<float_t, U>>;
};
template <typename T, typename float_t>
struct common_type<T, ParameterResolver<float_t>> {
    using type = ParameterResolver<std::common_type_t<T, float_t>>;
};
template <typename float_t, typename float2_t>
struct common_type<ParameterResolver<float_t>, ParameterResolver<float2_t>> {
    using type = ParameterResolver<std::common_type_t<float_t, float2_t>>;
};
}  // namespace traits

// =============================================================================

template <typename T1, typename T2>
bool IsTwoNumberClose(T1 v1, T2 v2) {
    if (std::abs(v1 - v2) < PRECISION) {
        return true;
    }
    return false;
}

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

// template <typename T>
// std::ostream& operator<<(std::ostream& os, const std::set<T>& s) {
//     os << "(";
//     for (ITER(p, s)) {
//         os << *p << ", ";
//     }
//     os << ")";
//     return os;
// }

template <typename T>
void Print(std::ostream& os, const std::set<T>& s) {
    os << "(";
    for (ITER(p, s)) {
        os << *p << ", ";
    }
    os << ")";
}

template <typename T>
T Conj(const T& value) {
    return value;
}

template <typename T>
std::complex<T> Conj(const std::complex<T>& value) {
    return std::conj(value);
}

template <typename T>
struct RemoveComplex {
    using type = T;
};

template <typename T>
struct RemoveComplex<std::complex<T>> {
    using type = T;
};

// -----------------------------------------------------------------------------

template <typename T>
struct ParameterResolver {
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ParameterResolver<T>, data_, const_value, no_grad_parameters_, encoder_parameters_);

    MST<T> data_{};
    T const_value = 0;
    SS no_grad_parameters_{};
    SS encoder_parameters_{};
    explicit ParameterResolver(T const_value) : const_value(const_value) {
    }
    ParameterResolver(const MST<T>& data, T const_value) : data_(data), const_value(const_value) {
        for (ITER(p, this->data_)) {
            if (p->first == "") {
                throw std::runtime_error("Parameter name cannot be empty.");
            }
        }
    }
    ParameterResolver(const MST<T>& data, T const_value, const SS& ngp, const SS& ep)
        : data_(data), const_value(const_value), no_grad_parameters_(ngp), encoder_parameters_(ep) {
    }
    explicit ParameterResolver(const std::string& p) {
        this->data_[p] = 1;
    }
    ParameterResolver() = default;
    ParameterResolver(const ParameterResolver&) = default;
    ParameterResolver(ParameterResolver&&) noexcept = default;
    ParameterResolver& operator=(const ParameterResolver&) = default;
    ParameterResolver& operator=(ParameterResolver&&) noexcept = default;
    ~ParameterResolver() noexcept = default;
    size_t Size() const {
        return this->data_.size();
    }

    inline void SetConst(T const_value) {
        this->const_value = const_value;
    }

    inline auto NIndex(size_t n) const {
        if (n >= this->Size()) {
            throw std::runtime_error("ParameterResolver: Index out of range.");
        }
        auto index_p = this->data_.begin();
        std::advance(index_p, n);
        return index_p;
    }

    std::string GetKey(size_t index) const {
        return this->NIndex(index)->first;
    }

    T GetItem(const std::string& key) const {
        if (!this->Contains(key)) {
            throw std::runtime_error("parameter " + key + " not in this parameter resolver.");
        }
        return this->data_.at(key);
    }

    T GetItem(size_t index) const {
        return this->NIndex(index)->second;
    }

    inline void SetItem(const std::string& key, T value) {
        data_[key] = value;
    }

    inline void SetItems(const VS& name, const VT<T>& data) {
        if (name.size() != data.size()) {
            throw std::runtime_error("size of name and data mismatch.");
        }
        for (size_t i = 0; i < name.size(); i++) {
            this->SetItem(name[i], data[i]);
        }
    }

    bool IsConst() const {
        if (this->data_.size() == 0) {
            return true;
        }
        for (ITER(p, this->data_)) {
            if (!IsTwoNumberClose(p->second, 0.0)) {
                return false;
            }
        }
        return true;
    }

    bool IsNotZero() const {
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

    inline bool Contains(const std::string& key) const {
        return this->data_.find(key) != this->data_.end();
    }

    inline bool NoGradContains(const std::string& key) const {
        return this->no_grad_parameters_.find(key) != this->no_grad_parameters_.end();
    }

    inline bool EncoderContains(const std::string& key) const {
        return this->encoder_parameters_.find(key) != this->encoder_parameters_.end();
    }

    inline SS GetAllParameters() const {
        SS all_params = {};
        for (ITER(p, this->data_)) {
            all_params.insert(p->first);
        }
        return all_params;
    }

    inline SS GetRequiresGradParameters() const {
        return this->GetAllParameters() - this->no_grad_parameters_;
    }

    inline SS GetAnsatzParameters() const {
        return this->GetAllParameters() - this->encoder_parameters_;
    }

    ParameterResolver<T>& operator+=(T value) {
        this->const_value += value;
        return *this;
    }

    ParameterResolver<T>& operator+=(const ParameterResolver<T>& other) {
        if ((this->encoder_parameters_.size() == 0) & (this->no_grad_parameters_.size() == 0)
            & (other.encoder_parameters_.size() == 0) & (other.no_grad_parameters_.size() == 0)) {
            for (ITER(p, other.data_)) {
                this->data_[p->first] += p->second;
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
                    this->data_[key] += value;
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
        this->const_value += other.const_value;
        return *this;
    }

    const ParameterResolver<T> operator+(T value) const {
        auto pr = *this;
        pr += value;
        return pr;
    }

    const ParameterResolver<T> operator+(const ParameterResolver<T>& pr) const {
        auto out = *this;
        out += pr;
        return out;
    }

    const ParameterResolver<T> operator-() const {
        auto out = *this;
        out.const_value = -out.const_value;
        for (ITER(p, out.data_)) {
            out.data_[p->first] = -out.data_[p->first];
        }
        return out;
    }

    ParameterResolver<T>& operator-=(T value) {
        *this += (-value);
        return *this;
    }

    ParameterResolver<T>& operator-=(const ParameterResolver<T>& other) {
        auto tmp = other;
        *this += (-tmp);
        return *this;
    }

    const ParameterResolver<T> operator-(const ParameterResolver<T>& pr) const {
        auto out = pr;
        return *this + (-out);
    }

    const ParameterResolver<T> operator-(T value) const {
        auto out = *this;
        out -= value;
        return out;
    }

    ParameterResolver<T>& operator*=(T value) {
        this->const_value *= value;
        for (ITER(p, this->data_)) {
            this->data_[p->first] *= value;
        }
        return *this;
    }

    ParameterResolver<T>& operator*=(const ParameterResolver<T> other) {
        if (this->IsConst()) {
            for (ITER(p, other.data_)) {
                this->data_[p->first] = this->const_value * p->second;
                if (!this->Contains(p->first)) {
                    if (other.EncoderContains(p->first)) {
                        this->encoder_parameters_.insert(p->first);
                    }
                    if (other.NoGradContains(p->first)) {
                        this->no_grad_parameters_.insert(p->first);
                    }
                }
                this->const_value = 0;
            }
        } else {
            if (other.IsConst()) {
                for (ITER(p, this->data_)) {
                    this->data_[p->first] *= other.const_value;
                }
                this->const_value *= other.const_value;
            } else {
                throw std::runtime_error("Parameter resolver only support first order variable.");
            }
        }
        this->const_value *= other.const_value;
        return *this;
    }

    const ParameterResolver<T> operator*(T value) const {
        auto pr = *this;
        pr *= value;
        return pr;
    }

    const ParameterResolver<T> operator*(const ParameterResolver<T> other) const {
        auto pr = *this;
        pr *= other;
        return pr;
    }

    ParameterResolver<T>& operator/=(T value) {
        for (ITER(p, this->data_)) {
            this->data_[p->first] /= value;
        }
        this->const_value /= value;
        return *this;
    }

    ParameterResolver<T>& operator/=(const ParameterResolver<T> other) {
        if (!other.IsConst()) {
            throw std::runtime_error("Cannot div a non constant ParameterResolver.");
        }
        for (ITER(p, this->data_)) {
            this->data_[p->first] /= other.const_value;
        }
        this->const_value /= other.const_value;
        return *this;
    }

    const ParameterResolver<T> operator/(T value) const {
        auto pr = *this;
        pr /= value;
        return pr;
    }

    const ParameterResolver<T> operator/(const ParameterResolver<T> other) const {
        auto out = *this;
        out /= other;
        return out;
    }

    void PrintInfo() const {
        std::cout << *this << std::endl;
    }

    std::string ToString() const {
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

    ParameterResolver<T> Copy() {
        auto out = *this;
        return out;
    }

    bool operator==(T value) const {
        if (this->Size() != 0) {
            return false;
        }
        return IsTwoNumberClose(this->const_value, value);
    }

    bool operator==(const ParameterResolver<T> pr) const {
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

    std::vector<std::string> ParamsName() const {
        std::vector<std::string> pn;
        for (ITER(p, this->data_)) {
            pn.push_back(p->first);
        }
        return pn;
    }

    std::vector<T> ParaValue() const {
        std::vector<T> pv;
        for (ITER(p, this->data_)) {
            pv.push_back(p->second);
        }
        return pv;
    }

    void RequiresGrad() {
        this->no_grad_parameters_ = {};
    }

    void NoGrad() {
        this->no_grad_parameters_ = {};
        for (ITER(p, this->data_)) {
            this->no_grad_parameters_.insert(p->first);
        }
    }

    void RequiresGradPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->NoGradContains(name)) {
                this->no_grad_parameters_.erase(name);
            }
        }
    }

    void NoGradPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->Contains(name)) {
                this->no_grad_parameters_.insert(name);
            }
        }
    }

    void AnsatzPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->EncoderContains(name)) {
                this->encoder_parameters_.erase(name);
            }
        }
    }

    void EncoderPart(const std::vector<std::string>& names) {
        for (auto& name : names) {
            if (this->Contains(name)) {
                this->encoder_parameters_.insert(name);
            }
        }
    }

    void AsEncoder() {
        for (ITER(p, this->data_)) {
            this->encoder_parameters_.insert(p->first);
        }
    }

    void AsAnsatz() {
        this->encoder_parameters_ = {};
    }

    void Update(const ParameterResolver<T>& other) {
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

    ParameterResolver<T> Conjugate() const {
        auto out = *this;
        for (ITER(p, out.data_)) {
            out.data_[p->first] = Conj(p->second);
        }
        out.const_value = Conj(out.const_value);
        return out;
    }

    ParameterResolver<T> Combination(const ParameterResolver<T>& pr) const {
        auto c = this->const_value;
        for (ITER(p, this->data_)) {
            c += p->second * pr.GetItem(p->first);
        }
        return ParameterResolver<T>(c);
    }

    auto Real() const {
        using type = typename RemoveComplex<T>::type;
        ParameterResolver<type> pr = {};
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
    void KeepReal() {
        this->const_value = std::real(this->const_value);
        for (auto& [name, value] : this->data_) {
            value = std::real(value);
        }
    }
    void KeepImag() {
        this->const_value = std::imag(this->const_value);
        for (auto& [name, value] : this->data_) {
            value = std::imag(value);
        }
    }
    auto Imag() const {
        using type = typename RemoveComplex<T>::type;
        ParameterResolver<type> pr = {};
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

    T Pop(const std::string& key) {
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

    bool IsHermitian() const {
        return *this == this->Conjugate();
    }

    bool IsAntiHermitian() const {
        return *this == -this->Conjugate();
    }

    auto ToComplexPR() const {
        using type = typename RemoveComplex<T>::type;
        ParameterResolver<std::complex<type>> out;
        for (ITER(p, this->data_)) {
            auto& key = p->first;
            auto& t = p->second;
            out.data_[p->first] = std::complex<type>(t);
            if (this->EncoderContains(key)) {
                out.encoder_parameters_.insert(key);
            }
            if (this->NoGradContains(key)) {
                out.no_grad_parameters_.insert(key);
            }
        }
        out.const_value = std::complex<type>(this->const_value);
        return out;
    }

    bool IsComplexPR() const {
        using type = typename RemoveComplex<T>::type;
        return !std::is_same<type, T>::value;
    }
};

template <typename T>
ParameterResolver<T> operator+(T value, const ParameterResolver<T>& pr) {
    auto out = pr;
    out += value;
    return out;
}

template <typename T>
ParameterResolver<T> operator-(T value, const ParameterResolver<T>& pr) {
    auto out = pr;
    return (-out) + value;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const ParameterResolver<T>& pr) {
    os << pr.ToString();
    return os;
}

template <typename T>
ParameterResolver<T> operator*(T value, const ParameterResolver<T>& pr) {
    auto out = pr;
    out *= value;
    return out;
}

template <typename T>
ParameterResolver<T> operator/(T value, const ParameterResolver<T>& pr) {
    return ParameterResolver<T>(value) / pr;
}

}  // namespace mindquantum

#endif  // MINDQUANTUM_PR_PARAMETER_RESOLVER_H_
