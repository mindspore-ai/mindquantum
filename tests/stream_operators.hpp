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

#ifndef TESTS_STREAM_OPERATORS
#define TESTS_STREAM_OPERATORS

#include <complex>
#include <map>
#include <optional>
#include <ostream>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "ops/gates/terms_operator.hpp"  // For mindquantum::ops::TermValue

// =============================================================================

template <typename T, typename alloc_t>
std::ostream& operator<<(std::ostream& out, const std::vector<T, alloc_t>& v);
template <typename... Ts>
std::ostream& operator<<(std::ostream& out, const std::tuple<Ts...>& t);
template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, const std::pair<T, U>& p);
template <typename key_t, typename value_t, typename comp_t, typename alloc_t>
std::ostream& operator<<(std::ostream& out, const std::map<key_t, value_t, comp_t, alloc_t>& m);

// -----------------------------------------------------------------------------

inline std::ostream& operator<<(std::ostream& out, mindquantum::ops::TermValue val) {
    using TermValue = mindquantum::ops::TermValue;
    if (val == TermValue::I) {
        out << 'I';
    } else if (val == TermValue::X) {
        out << 'X';
    } else if (val == TermValue::Y) {
        out << 'Y';
    } else if (val == TermValue::Z) {
        out << 'Z';
    } else if (val == TermValue::a) {
        out << 'a';
    } else if (val == TermValue::adg) {
        out << "adg";
    } else {
        out << "UNKNOWN";
    }
    return out;
}

// -----------------------------------------------------------------------------

template <typename T>
std::ostream& operator<<(std::ostream& out, const std::complex<T>& c) {
    out << c.real() << " + " << c.imag() << 'i';
    return out;
}

template <typename T, typename U>
std::ostream& operator<<(std::ostream& out, const std::pair<T, U>& p) {
    out << '(' << p.first << ',' << p.second << ')';
    return out;
}

template <class tuple_t, size_t... idx>
void print(std::ostream& out, const tuple_t& _tup, std::index_sequence<idx...>) {
    out << "(";
    (..., (out << (idx == 0 ? "" : ", ") << std::get<idx>(_tup)));
    out << ")\n";
}

template <class... args_t>
void print(std::ostream& out, const std::tuple<args_t...>& _tup) {
    print(out, _tup, std::make_index_sequence<sizeof...(args_t)>());
}

template <class... args_t>
std::ostream& operator<<(std::ostream& out, std::tuple<args_t...> const& t) {
    out << "(";
    print(out, t);
    return out << ")";
}

template <typename T, typename alloc_t>
std::ostream& operator<<(std::ostream& out, const std::vector<T, alloc_t>& v) {
    out << '[';
    for (const auto& el : v) {
        out << el << ",";
    }
    return out << ']';
}

template <typename key_t, typename value_t, typename comp_t, typename alloc_t>
std::ostream& operator<<(std::ostream& out, const std::map<key_t, value_t, comp_t, alloc_t>& m) {
    out << "{\n";
    for (const auto& [k, v] : m) {
        out << "  " << k << ": " << v << '\n';
    }
    return out << '}';
}

// -----------------------------------------------------------------------------

namespace internal::traits {
template <typename T, typename = void>
struct to_string : std::false_type {};
template <typename T>
struct to_string<T, std::void_t<decltype(std::declval<const T&>().to_string())>> : std::true_type {};

}  // namespace internal::traits

// -------------------------------------

template <typename T>
std::enable_if_t<internal::traits::to_string<T>::value, std::ostream&> operator<<(std::ostream& out, const T& object) {
    return out << object.to_string();
}

// -----------------------------------------------------------------------------

namespace internal::traits {
template <typename T, typename = void>
struct has_stream_operator : std::false_type {};
template <typename T>
struct has_stream_operator<std::optional<T>,
                           std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>
    : std::true_type {};
}  // namespace internal::traits

// -------------------------------------

template <typename T>
std::enable_if_t<internal::traits::has_stream_operator<std::optional<T>>::value, std::ostream&> operator<<(
    std::ostream& out, const std::optional<T>& object) {
    if (object) {
        return out << "<std::optional>: " << object.value();
    }
    return out << "<invalid std::optional>";
}

#endif /* TESTS_STREAM_OPERATORS */
