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

#ifndef BOOST_X3_PARSE_TERM_HPP
#define BOOST_X3_PARSE_TERM_HPP

#ifndef ENABLE_LOGGING
#    include <iostream>
#endif  // !ENABLE_LOGGING

#include <boost/spirit/home/x3/core/parse.hpp>
#include <boost/spirit/home/x3/directive/with.hpp>
#include <boost/spirit/home/x3/support/utility/error_reporting.hpp>

#include "core/logging.hpp"
#include "core/parser/boost_x3_error_handler.hpp"

namespace mindquantum::parser {
template <typename iterator_t, typename terms_t, typename rule_t>
bool parse_term(iterator_t iter, iterator_t end, terms_t& terms, rule_t&& start_rule) {
#ifdef ENABLE_LOGGING
    mindquantum::parser::error_handler<iterator_t> handler(iter, end);
#else
    boost::spirit::x3::error_handler<iterator_t> handler(iter, end, std::cerr);
#endif  // ENABLE_LOGGING

    if (iter == end) {
        MQ_ERROR("Cannot parse empty string!");
        terms.clear();
        return false;
    }

    const auto parser = boost::spirit::x3::with<boost::spirit::x3::error_handler_tag>(std::ref(handler))[start_rule];
    if (boost::spirit::x3::parse(iter, end, parser, terms) && iter == end) {
        return true;
    }

    // Parsing failed
    terms.clear();
    return false;
}
}  // namespace mindquantum::parser

#endif /* BOOST_X3_PARSE_TERM_HPP */
