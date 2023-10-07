/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef BOOST_X3_ERROR_HANDLER_HPP
#define BOOST_X3_ERROR_HANDLER_HPP

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/spirit/home/x3/auxiliary/guard.h>
#include <boost/spirit/home/x3/support/ast/position_tagged.h>
#include <boost/spirit/home/x3/support/context.h>
#include <boost/spirit/home/x3/support/utility/error_reporting.h>
#include <boost/spirit/home/x3/support/utility/utf8.h>

#include <nlohmann/detail/input/position_t.h>

#include "config/logging.h"

namespace mindquantum::parser {
template <typename iterator_t>
class error_handler;

namespace x3::rule {
struct error_handler {
    template <typename iterator_t, typename error_t, typename context_t>
    auto on_error(iterator_t& /* iter */, const iterator_t& /* end */, const error_t& error, const context_t& context) {
        namespace x3 = boost::spirit::x3;
        const auto& handler = x3::get<x3::error_handler_tag>(context).get();
        handler(error.where(), "error: expecting: " + error.which());
        return x3::error_handler_result::fail;
    }
};
}  // namespace x3::rule

#ifdef ENABLE_LOGGING

//! Error handler class for parsing with Boost.Spirit.X3
/*!
 * \note Similar to \c boost::spirit::x3::error_handler but without \c std::ostream support
 */
template <typename iterator_t>
class error_handler {
 public:
    using iterator_type = iterator_t;
    using result_type = void;
    using position_tagged = boost::spirit::x3::position_tagged;
    template <typename T>
    using position_cache = boost::spirit::x3::position_cache<T>;

    error_handler(iterator_t first, iterator_t last, std::string file = "", int tabs = 4)
        : logger(spdlog::default_logger()), file(std::move(file)), tabs(tabs), pos_cache(first, last) {
    }

    void operator()(iterator_t err_pos, std::string const& error_message) const;
    void operator()(iterator_t err_first, iterator_t err_last, std::string const& error_message) const;
    void operator()(position_tagged pos, std::string const& message) const {
        auto where = pos_cache.position_of(pos);
        (*this)(where.begin(), where.end(), message);
    }

    template <typename AST>
    void tag(AST& ast, iterator_t first, iterator_t last) {  // NOLINT
        return pos_cache.annotate(ast, first, last);
    }

    boost::iterator_range<iterator_t> position_of(position_tagged pos) const {
        return pos_cache.position_of(pos);
    }

    position_cache<std::vector<iterator_t>> const& get_position_cache() const {
        return pos_cache;
    }

 private:
    void print_file_line(std::size_t line) const;
    void print_line(iterator_t line_start, iterator_t last) const;
    // NOLINTNEXTLINE
    void print_indicator(iterator_t& line_start, iterator_t last, char ind, std::string_view indicator) const;
    iterator_t get_line_start(iterator_t first, iterator_t pos) const;
    std::size_t position(iterator_t iter) const;

    std::shared_ptr<spdlog::logger> logger;
    std::string file;
    int tabs;
    position_cache<std::vector<iterator_t>> pos_cache;
};

template <typename iterator_t>
void error_handler<iterator_t>::print_file_line(std::size_t line) const {
    if (!std::empty(file)) {
        MQ_LOGGER_ERROR(logger, "In file {}, line: {}", file, line);
    } else {
        MQ_LOGGER_ERROR(logger, "On line: {}", line);
    }
}

template <typename iterator_t>
void error_handler<iterator_t>::print_line(iterator_t start, iterator_t last) const {
    auto end = start;
    while (end != last) {
        auto ch = *end;
        if (ch == '\r' || ch == '\n') {
            break;
        }
        ++end;
    }
    using char_type = typename std::iterator_traits<iterator_t>::value_type;
    std::basic_string<char_type> line{start, end};
    MQ_LOGGER_ERROR(logger, boost::spirit::x3::to_utf8(line));
}

template <typename iterator_t>
void error_handler<iterator_t>::print_indicator(iterator_t& start, iterator_t last, char ind,  // NOLINT
                                                std::string_view indicator) const {
    std::size_t length(0);
    for (; start != last; ++start) {
        auto ch = *start;
        if (ch == '\r' || ch == '\n') {
            break;
        }
        if (ch == '\t') {
            length += tabs;
        } else {
            ++length;
        }
    }
    switch (ind) {
        case ' ':
            MQ_LOGGER_ERROR(logger, "{: >{}}{}", "", length, indicator);
            break;
        case '~':
            MQ_LOGGER_ERROR(logger, "{:~>{}}{}", "", length, indicator);
            break;
        default:
            MQ_LOGGER_ERROR(logger, "{:->{}}{}", "", length, indicator);
            break;
    }
}

template <class iterator_t>
inline iterator_t error_handler<iterator_t>::get_line_start(iterator_t first, iterator_t pos) const {
    iterator_t latest = first;
    for (iterator_t iter = first; iter != pos;) {
        if (*iter == '\r' || *iter == '\n') {
            latest = ++iter;
        } else {
            ++iter;
        }
    }
    return latest;
}

template <typename iterator_t>
std::size_t error_handler<iterator_t>::position(iterator_t iter) const {
    std::size_t line{1};
    typename std::iterator_traits<iterator_t>::value_type prev{0};

    for (iterator_t pos = pos_cache.first(); pos != iter; ++pos) {
        auto ch = *pos;
        switch (ch) {
            case '\n':
                if (prev != '\r')
                    ++line;
                break;
            case '\r':
                ++line;
                break;
            default:
                break;
        }
        prev = ch;
    }

    return line;
}

template <typename iterator_t>
void error_handler<iterator_t>::operator()(iterator_t err_pos, std::string const& error_message) const {
    iterator_t first = pos_cache.first();
    iterator_t last = pos_cache.last();

    print_file_line(position(err_pos));
    MQ_LOGGER_ERROR(logger, error_message);

    iterator_t start = get_line_start(first, err_pos);
    print_line(start, last);
    print_indicator(start, err_pos, '_', "^_");
}

template <typename iterator_t>
void error_handler<iterator_t>::operator()(iterator_t err_first, iterator_t err_last,
                                           std::string const& error_message) const {
    iterator_t first = pos_cache.first();
    iterator_t last = pos_cache.last();

    print_file_line(position(err_first));
    MQ_LOGGER_ERROR(logger, error_message);

    iterator_t start = get_line_start(first, err_first);
    print_line(start, last);
    print_indicator(start, err_first, ' ', "");
    print_indicator(start, err_last, '~', " <<-- Here");
}
#endif  // ENABLE_LOGGING
}  // namespace mindquantum::parser

#endif /* BOOST_X3_ERROR_HANDLER_HPP */
