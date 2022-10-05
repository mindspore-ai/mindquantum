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

#ifndef BOOST_X3_PARSE_OBJECT_HPP
#define BOOST_X3_PARSE_OBJECT_HPP

#include <type_traits>
#ifndef ENABLE_LOGGING
#    include <iostream>
#endif  // !ENABLE_LOGGING

#include <boost/spirit/home/x3/core/parse.hpp>
#include <boost/spirit/home/x3/directive/with.hpp>
#include <boost/spirit/home/x3/support/utility/error_reporting.hpp>

#include "config/logging.hpp"

#include "experimental/core/parser/boost_x3_error_handler.hpp"

namespace mindquantum::parser {
namespace details::traits {
template <typename T, typename = void>
struct call_clear {
    static void apply(T& /* value */) {
    }
};
template <typename T>
struct call_clear<T, std::void_t<decltype(std::declval<T&>().clear())>> {
    static void apply(T& value) {
        value.clear();
    }
};
}  // namespace details::traits

template <typename iterator_t, typename object_t, typename rule_t>
bool parse_object(iterator_t iter, iterator_t end, object_t& object, rule_t&& start_rule) {
#ifdef ENABLE_LOGGING
    mindquantum::parser::error_handler<iterator_t> handler(iter, end);
#else
    boost::spirit::x3::error_handler<iterator_t> handler(iter, end, std::cerr);
#endif  // ENABLE_LOGGING

    if (iter == end) {
        MQ_ERROR("Cannot parse empty string!");
        details::traits::call_clear<object_t>::apply(object);
        return false;
    }

    const auto parser = boost::spirit::x3::with<boost::spirit::x3::error_handler_tag>(std::ref(handler))[start_rule];
    if (boost::spirit::x3::parse(iter, end, parser, object) && iter == end) {
        return true;
    }

    // Parsing failed
    details::traits::call_clear<object_t>::apply(object);
    return false;
}

template <typename iterator_t, typename object_t, typename rule_t, typename skipper_t>
bool parse_object_skipper(iterator_t iter, iterator_t end, object_t& object, rule_t&& start_rule, skipper_t&& skipper) {
#ifdef ENABLE_LOGGING
    mindquantum::parser::error_handler<iterator_t> handler(iter, end);
#else
    boost::spirit::x3::error_handler<iterator_t> handler(iter, end, std::cerr);
#endif  // ENABLE_LOGGING

    if (iter == end) {
        MQ_ERROR("Cannot parse empty string!");
        details::traits::call_clear<object_t>::apply(object);
        return false;
    }

    const auto parser = boost::spirit::x3::with<boost::spirit::x3::error_handler_tag>(std::ref(handler))[start_rule];
    if (boost::spirit::x3::phrase_parse(iter, end, parser, skipper, object) && iter == end) {
        return true;
    }

    // Parsing failed
    details::traits::call_clear<object_t>::apply(object);
    return false;
}
}  // namespace mindquantum::parser

#endif /* BOOST_X3_PARSE_OBJECT_HPP */
