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

#include "experimental/ops/gates/details/fermion_operator_term_policy.hpp"

#include <algorithm>
#include <iterator>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3.hpp>

#include "boost_x3_complex_number.hpp"
#include "boost_x3_get_info_impl.hpp"
#include "boost_x3_parse_object.hpp"

#include "experimental/core/logging.hpp"
#include "experimental/core/parser/boost_x3_error_handler.hpp"
#include "experimental/ops/gates/fermion_operator.hpp"

// -----------------------------------------------------------------------------

using namespace std::literals::string_literals;  // NOLINT(build/namespaces_literals)

// =============================================================================

namespace x3 = boost::spirit::x3;

namespace ast::fm_op {
using mindquantum::ops::TermValue;
using term_t = mindquantum::ops::FermionOperator::term_t;
using terms_t = mindquantum::ops::FermionOperator::terms_t;
using coeff_term_dict_t = mindquantum::ops::FermionOperator::coeff_term_dict_t;
using term_coeff_t = std::pair<mindquantum::ops::FermionOperator::coeff_term_dict_t::key_type,
                               mindquantum::ops::FermionOperator::coeff_term_dict_t::mapped_type>;

struct TermOp : x3::symbols<TermValue> {
    TermOp() {
        add("^", TermValue::adg);
        add("v", TermValue::a);
    }
} const term_op;
}  // namespace ast::fm_op

// -----------------------------------------------------------------------------

namespace parser::fm_op {
namespace ast = ::ast::fm_op;
using mindquantum::parser::complex;

struct term_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<term_class, ast::term_t> term = "FermionOperator term (ie. [0-9]+^?)";
static const auto term_def = x3::lexeme[x3::uint_ > (ast::term_op | x3::attr(ast::TermValue::a))];

struct terms_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<terms_class, ast::terms_t> terms = "FermionOperator terms list";
static const auto terms_def = +term;

// -------------------------------------

struct json_value_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<json_value_class, ast::term_coeff_t> json_value_term
    = R"s(FermionOperator JSON key-value pair ("<term-list>": "<complex-num>"))s";
static const auto json_value_term_def = x3::expect[x3::lit('"')]
                                        >> (('"' >> x3::attr(ast::terms_t{})) | (x3::expect[+term] >> '"')) > ':'
                                        > x3::lit('"') > complex > '"';
struct json_dict_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<json_dict_class, ast::coeff_term_dict_t> json_dict = "JSON representation of a FermionOperator";
static const auto json_dict_def = x3::expect['{'] > (json_value_term % ',') > '}';

// -------------------------------------

BOOST_SPIRIT_DEFINE(term, terms, json_value_term, json_dict);
}  // namespace parser::fm_op

// -------------------------------------

namespace boost::spirit::x3 {
template <>
struct get_info<ast::fm_op::TermOp> {
    using result_type = std::string;
    result_type operator()(const ast::fm_op::TermOp& /* type */) const noexcept {
        using std::literals::string_literals::operator""s;
        return "ladder operator (a: '' or 'v', adg: '^')"s;
    }
};
}  // namespace boost::spirit::x3

// =============================================================================

namespace mindquantum::ops::details {
auto FermionOpteratorTermPolicyBase::hermitian(term_t term) -> term_t {
    term.second = term.second == TermValue::adg ? TermValue::a : TermValue::adg;
    return term;
}

auto FermionOpteratorTermPolicyBase::hermitian(const terms_t& terms) -> terms_t {
    terms_t new_terms;
    new_terms.reserve(terms.size());

    std::transform(rbegin(terms), rend(terms), std::inserter(new_terms, end(new_terms)),
                   [](const auto& term) { return hermitian(term); });
    return new_terms;
}

auto FermionOpteratorTermPolicyBase::to_string(const term_t& term) -> std::string {
    return fmt::format("{}{}", std::get<0>(term), to_string(std::get<1>(term)));
}

auto FermionOpteratorTermPolicyBase::parse_terms_string(std::string_view terms_string) -> terms_t {
    MQ_INFO("Attempting to parse: '{}'", terms_string);
    if (terms_t terms; parser::parse_object_skipper(begin(terms_string), end(terms_string), terms,
                                                    ::parser::fm_op::terms, x3::space)) {
        return terms;
    }
    MQ_ERROR("FermionOperator terms string parsing failed for '{}'", terms_string);
    return {};
}

// =============================================================================

}  // namespace mindquantum::ops::details
