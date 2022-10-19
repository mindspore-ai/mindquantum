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
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <fmt/ranges.h>
#include <nlohmann/json.hpp>

#include "config/common_type.hpp"
#include "config/real_cast.hpp"
#include "config/type_promotion.hpp"
#include "config/type_traits.hpp"

#include "core/mq_base_types.hpp"
#include "core/two_dim_matrix.hpp"
#include "core/utils.hpp"

#if MQ_HAS_CONCEPTS
#    include "config/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

namespace mindquantum {
template <typename float_t>
class ParameterResolver;

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
        constexpr auto is_complex_valued = traits::is_std_complex_v<float_t>;
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
// Template specialisations to support real-to-complex (and inverse) for ParameterResolver
template <typename float_t>
struct to_real_type<ParameterResolver<float_t>> {
    using type = ParameterResolver<to_real_type_t<float_t>>;
};
template <typename float_t>
struct to_cmplx_type<ParameterResolver<float_t>> {
    using type = ParameterResolver<to_cmplx_type_t<float_t>>;
};

// -----------------------------------------------------------------------------
// Template specialitation for is_complex<T>

template <typename float_t>
struct is_complex<ParameterResolver<float_t>> : is_complex<float_t> {};

// -----------------------------------------------------------------------------
// Template specialitation to support up_cast<...> and down_cast<...> for ParameterResolver<T>

template <typename T>
struct type_promotion<ParameterResolver<T>> : details::type_promotion_encapsulated_fp<T, ParameterResolver> {};

// -----------------------------------------------------------------------------
// Template specialitations for mindquantum::common_type<...>

template <typename float_t, typename U>
struct common_type<ParameterResolver<float_t>, U> {
    using type = ParameterResolver<common_type_t<float_t, U>>;
};
template <typename T, typename float_t>
struct common_type<T, ParameterResolver<float_t>> {
    using type = ParameterResolver<common_type_t<T, float_t>>;
};
template <typename float_t, typename float2_t>
struct common_type<ParameterResolver<float_t>, ParameterResolver<float2_t>> {
    using type = ParameterResolver<common_type_t<float_t, float2_t>>;
};

/* NB: these two are required to avoid ambiguous cases like:
 *       - ParameterResolver<T> - std::complex<U>
 *       - std::complex<T> - ParameterResolver<U>
 */
template <typename float_t, typename float2_t>
struct common_type<ParameterResolver<float_t>, std::complex<float2_t>> {
    using type = ParameterResolver<common_type_t<float_t, std::complex<float2_t>>>;
};
template <typename float_t, typename float2_t>
struct common_type<std::complex<float_t>, ParameterResolver<float2_t>> {
    using type = ParameterResolver<common_type_t<std::complex<float_t>, float2_t>>;
};

// -----------------------------------------------------------------------------
// Type traits to identity a ParameterResolver<T>

template <typename T>
struct is_parameter_resolver : std::false_type {};

template <typename T>
struct is_parameter_resolver<ParameterResolver<T>> : std::true_type {};

template <typename T>
inline constexpr auto is_parameter_resolver_v = is_parameter_resolver<T>::value;

template <typename T>
inline constexpr auto is_parameter_resolver_decay_v = is_parameter_resolver_v<std::remove_cvref_t<T>>;

// -----------------------------------------------------------------------------
// Template specialitations for mindquantum::traits::is_scalar<...>

template <typename T>
struct is_scalar<ParameterResolver<T>> : std::true_type {};
}  // namespace traits

// =============================================================================

#if MQ_HAS_CONCEPTS
namespace concepts {
template <typename type_t>
concept parameter_resolver = traits::is_parameter_resolver_v<std::remove_cvref_t<type_t>>;

template <typename type_t>
concept parameter_resolver_scalar = concepts::scalar<type_t> && !parameter_resolver<type_t>;
}  // namespace concepts
#endif  // MQ_HAS_CONCEPTS

// =============================================================================

template <typename T1, typename T2>
bool IsTwoNumberClose(T1 v1, T2 v2) {
    return static_cast<bool>(std::abs(v1 - v2) < PRECISION);
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

// -----------------------------------------------------------------------------

template <typename T>
struct ParameterResolver {
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(ParameterResolver<T>, data_, const_value, no_grad_parameters_, encoder_parameters_);

    MST<T> data_{};
    T const_value = static_cast<T>(0);
    SS no_grad_parameters_{};
    SS encoder_parameters_{};
    using value_type = T;

    template <typename U,
              typename = std::enable_if_t<
                  std::is_arithmetic_v<std::remove_cvref_t<U>> || traits::is_std_complex_v<std::remove_cvref_t<U>>>>
    explicit ParameterResolver(U&& value);

    //! Constructor from a dictionary [string -> T] and a constant of type T
    explicit ParameterResolver(const MST<T>& data, T value = static_cast<T>(0));

    //! Constructor from a dictionary [string -> T] and a constant of type T and gradient parameters
    ParameterResolver(const MST<T>& data, T value, SS no_grad_parameters, SS encoder_parameters);

    //! Constructor from a dictionary [string -> some_type] without a constant
    /*!
     * \note \c some_type does not need to be the same as \c T
     */
    template <typename U,
              typename = std::enable_if_t<
                  std::is_arithmetic_v<std::remove_cvref_t<U>> || traits::is_std_complex_v<std::remove_cvref_t<U>>>>
    explicit ParameterResolver(const MST<U>& data);

    //! Constructor from a dictionary [string -> some_type] and a constant of some other_type
    /*!
     * \note \c some_type  and \c other_type do not need to be the same as \c T
     */
    template <
        typename U, typename V,
        typename = std::enable_if_t<
            (std::is_arithmetic_v<std::remove_cvref_t<U>> || traits::is_std_complex_v<std::remove_cvref_t<U>>) &&(
                std::is_arithmetic_v<std::remove_cvref_t<V>> || traits::is_std_complex_v<std::remove_cvref_t<V>>)>>
    ParameterResolver(const MST<U>& data, V value_const);

    template <
        typename U, typename V,
        typename = std::enable_if_t<
            (std::is_arithmetic_v<std::remove_cvref_t<U>> || traits::is_std_complex_v<std::remove_cvref_t<U>>) &&(
                std::is_arithmetic_v<std::remove_cvref_t<V>> || traits::is_std_complex_v<std::remove_cvref_t<V>>)>>
    ParameterResolver(const MST<U>& data, V&& value_const, SS no_grad_parameters, SS encoder_parameters);

    explicit ParameterResolver(const std::string& param);

    template <typename U,
              typename = std::enable_if_t<
                  std::is_arithmetic_v<std::remove_cvref_t<U>> || traits::is_std_complex_v<std::remove_cvref_t<U>>>>
    explicit ParameterResolver(const ParameterResolver<U>& other);

    ParameterResolver() = default;
    ParameterResolver(const ParameterResolver&) = default;
    ParameterResolver(ParameterResolver&&) noexcept = default;
    ParameterResolver& operator=(const ParameterResolver&) = default;
    ParameterResolver& operator=(ParameterResolver&&) noexcept = default;
    ~ParameterResolver() noexcept = default;

    explicit operator T() const;
    template <typename U = T,
              typename = std::enable_if_t<
                  std::is_same_v<T, std::remove_cvref_t<U>> && !traits::is_complex_v<std::remove_cvref_t<U>>>>
    explicit operator traits::to_cmplx_type_t<U>() const;

    size_t Size() const;

    inline void SetConst(T const_value);

    inline auto NIndex(size_t n) const;

    std::string GetKey(size_t index) const;

    T GetItem(const std::string& key) const;

    T GetItem(size_t index) const;

    inline void SetItem(const std::string& key, T value);

    inline void SetItems(const VS& name, const VT<T>& data);

    bool IsConst() const;

    bool IsNotZero() const;

    inline bool Contains(const std::string& key) const;

    inline bool NoGradContains(const std::string& key) const;

    inline bool EncoderContains(const std::string& key) const;

    inline SS GetAllParameters() const;

    inline SS GetRequiresGradParameters() const;

    inline SS GetAnsatzParameters() const;

    template <typename other_t>
    ParameterResolver<T>& operator+=(other_t value);

    template <typename other_t>
    ParameterResolver<T>& operator+=(const ParameterResolver<other_t>& other);

    ParameterResolver<T> operator-() const;

    template <typename other_t>
    ParameterResolver<T>& operator-=(other_t value);

    template <typename other_t>
    ParameterResolver<T>& operator-=(const ParameterResolver<other_t>& other);

    template <typename other_t>
    ParameterResolver<T>& operator*=(other_t value);

    template <typename other_t>
    ParameterResolver<T>& operator*=(const ParameterResolver<other_t>& other);

    template <typename other_t>
    ParameterResolver<T>& operator/=(other_t value);

    template <typename other_t>
    ParameterResolver<T>& operator/=(const ParameterResolver<other_t>& other);

    void PrintInfo() const;

    std::string ToString() const;

    ParameterResolver<T> Copy();

    template <typename other_t>
    bool IsEqual(other_t value) const;

    template <typename other_t>
    bool IsEqual(const ParameterResolver<other_t> pr) const;

    std::vector<std::string> ParamsName() const;

    std::vector<T> ParaValue() const;

    void RequiresGrad();

    void NoGrad();

    void RequiresGradPart(const std::vector<std::string>& names);

    void NoGradPart(const std::vector<std::string>& names);

    void AnsatzPart(const std::vector<std::string>& names);

    void EncoderPart(const std::vector<std::string>& names);

    void AsEncoder();
    void AsAnsatz();

    template <typename other_t>
    void Update(const ParameterResolver<other_t>& other);

    ParameterResolver<T> Conjugate() const;

    ParameterResolver<T> Combination(const ParameterResolver<T>& pr) const;

    auto Real() const;
    void KeepReal();
    void KeepImag();
    auto Imag() const;

    T Pop(const std::string& key);

    bool IsHermitian() const;

    bool IsAntiHermitian() const;

    template <typename number_t>
    auto Cast() const;

    bool HasRequireGradParams() {
        return this->data_.size() > this->no_grad_parameters_.size();
    }
};

template <typename T>
std::ostream& operator<<(std::ostream& os, const ParameterResolver<T>& pr) {
    os << pr.ToString();
    return os;
}

template <typename T>
MST<size_t> GetRequiresGradParameters(const VT<ParameterResolver<T>>& prs) {
    MST<size_t> title = {};
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

template <typename T>
std::pair<MST<size_t>, Dim2Matrix<T>> Jacobi(const VT<ParameterResolver<T>>& prs) {
    auto title = GetRequiresGradParameters(prs);
    VVT<CT<T>> jacobi(prs.size(), VT<CT<T>>(title.size(), CT<T>(0.0, 0.0)));
    for (size_t i = 0; i < prs.size(); i++) {
        for (auto& name : prs[i].GetRequiresGradParameters()) {
            jacobi[i][title.at(name)] = prs[i].data_.at(name);
        }
    }
    return {title, Dim2Matrix<T>(jacobi)};
}
}  // namespace mindquantum

#include "parameter_resolver.tpp"
//
#include "core/parameter_resolver_external_ops.hpp"

#endif  // MINDQUANTUM_PR_PARAMETER_RESOLVER_H_
