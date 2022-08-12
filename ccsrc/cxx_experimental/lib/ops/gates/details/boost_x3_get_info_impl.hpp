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

#ifndef BOOST_X3_GET_INFO_IMPL_HPP
#define BOOST_X3_GET_INFO_IMPL_HPP

#include <string>

#include <boost/spirit/home/x3.hpp>

namespace boost::spirit::x3 {
template <>
struct get_info<uint_type> {
    using result_type = std::string;
    result_type operator()(const uint_type& /* type */) const noexcept {
        using std::literals::string_literals::operator""s;
        return "unsigned int"s;
    }
};
template <typename rule_t>
struct get_info<plus<rule_t>> {
    using result_type = std::string;
    result_type operator()(const plus<rule_t>& type) const noexcept {
        return "one or more of: " + get_info<rule_t>{}(type.subject);
    }
};
template <typename rule_t>
struct get_info<omit_directive<rule_t>> {
    using result_type = std::string;
    result_type operator()(const omit_directive<rule_t>& type) const noexcept {
        return get_info<rule_t>{}(type.subject);
    }
};
template <typename rule_t>
struct get_info<and_predicate<rule_t>> {
    using result_type = std::string;
    result_type operator()(const and_predicate<rule_t>& type) const noexcept {
        using std::literals::string_literals::operator""s;
        return "&("s + get_info<rule_t>{}(type.subject) + ")";
    }
};
template <typename rule_t>
struct get_info<not_predicate<rule_t>> {
    using result_type = std::string;
    result_type operator()(const not_predicate<rule_t>& type) const noexcept {
        using std::literals::string_literals::operator""s;
        return "!("s + get_info<rule_t>{}(type.subject) + ")";
    }
};
template <typename left_t, typename right_t>
struct get_info<list<left_t, right_t>> {
    using result_type = std::string;
    result_type operator()(const list<left_t, right_t>& type) const noexcept {
        using std::literals::string_literals::operator""s;
        return "list of ["s + get_info<left_t>{}(type.left) + "], delimited by " + get_info<right_t>{}(type.right);
    }
};
}  // namespace boost::spirit::x3

#endif /* BOOST_X3_GET_INFO_IMPL_HPP */
