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

#include "ops/gates/qubit_operator.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <boost/fusion/include/adapt_struct.hpp>
#include <boost/fusion/include/at_c.hpp>
#include <boost/fusion/include/std_pair.hpp>
#include <boost/spirit/home/x3.hpp>

#include <unsupported/Eigen/KroneckerProduct>

#include <tweedledum/Operators/Standard/X.h>
#include <tweedledum/Operators/Standard/Y.h>
#include <tweedledum/Operators/Standard/Z.h>

#include <fmt/format.h>

#include "core/format/format_complex.hpp"
#include "core/logging.hpp"
#include "core/parser/boost_x3_error_handler.hpp"
#include "details/boost_x3_complex_number.hpp"
#include "details/boost_x3_get_info_impl.hpp"
#include "details/boost_x3_parse_object.hpp"
#include "ops/gates.hpp"
#include "ops/gates/terms_operator.hpp"

// -----------------------------------------------------------------------------

using namespace std::literals::string_literals;  // NOLINT(build/namespaces_literals)

// =============================================================================

namespace x3 = boost::spirit::x3;

namespace ast::qb_op {
using mindquantum::ops::TermValue;
using term_t = mindquantum::ops::QubitOperator::term_t;
using terms_t = mindquantum::ops::QubitOperator::terms_t;
using complex_term_dict_t = mindquantum::ops::QubitOperator::complex_term_dict_t;
using term_coeff_t = std::pair<mindquantum::ops::QubitOperator::complex_term_dict_t::key_type,
                               mindquantum::ops::QubitOperator::complex_term_dict_t::mapped_type>;

struct TermOp : x3::symbols<TermValue> {
    TermOp() {
        add("X", TermValue::X)("Y", TermValue::Y)("Z", TermValue::Z);
    }
} const term_op;
}  // namespace ast::qb_op

// -----------------------------------------------------------------------------

namespace parser::qb_op {
namespace ast = ::ast::qb_op;
using mindquantum::parser::complex;

struct term_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<term_class, ast::term_t> term = "QubitOperator term (ie. [XYZ][0-9]+)";
static const auto term_def = x3::lexeme[((ast::term_op > x3::uint_)[([](auto& ctx) {
    x3::_val(ctx) = std::make_pair(boost::fusion::at_c<1>(x3::_attr(ctx)), boost::fusion::at_c<0>(x3::_attr(ctx)));
})])];

struct terms_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<terms_class, ast::terms_t> terms = "QubitOperator terms list";
/* NB: Simply using '+term' will not work here since we have to reject cases like: 'X1 YY'
 *     So we make sure that everything will match using a look-ahead on each term before actually matching the term.
 */
static const auto terms_def = +(&term >> term);

// -------------------------------------

struct json_value_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<json_value_class, ast::term_coeff_t> json_value_term
    = R"s(QubitOperator JSON key-value pair ("<term-list>": "<complex-num>"))s";
static const auto json_value_term_def = x3::expect[x3::lit('"')]
                                        >> (('"' >> x3::attr(ast::terms_t{})) | (x3::expect[+term] >> '"')) > ':'
                                        > x3::lit('"') > complex > '"';
struct json_dict_class : mindquantum::parser::x3::rule::error_handler {};
const x3::rule<json_dict_class, ast::complex_term_dict_t> json_dict = "JSON representation of a QubitOperator";
static const auto json_dict_def = x3::expect['{'] > (json_value_term % ',') > '}';

// -------------------------------------

BOOST_SPIRIT_DEFINE(term, terms, json_value_term, json_dict);
}  // namespace parser::qb_op

// -------------------------------------

namespace boost::spirit::x3 {
template <>
struct get_info<ast::qb_op::TermOp> {
    using result_type = std::string;
    result_type operator()(const ast::qb_op::TermOp& /* type */) const noexcept {
        using std::literals::string_literals::operator""s;
        return "local operator (X, Y, Z)"s;
    }
};
}  // namespace boost::spirit::x3

// =============================================================================

namespace mindquantum::ops {
constexpr std::tuple<std::complex<double>, TermValue> pauli_products(const TermValue& left_op,
                                                                     const TermValue& right_op) {
    if (left_op == TermValue::I && right_op == TermValue::X) {
        return {1., TermValue::X};
    }
    if (left_op == TermValue::X && right_op == TermValue::I) {
        return {1., TermValue::X};
    }
    if (left_op == TermValue::I && right_op == TermValue::Y) {
        return {1., TermValue::Y};
    }
    if (left_op == TermValue::Y && right_op == TermValue::I) {
        return {1., TermValue::Y};
    }
    if (left_op == TermValue::I && right_op == TermValue::Z) {
        return {1., TermValue::Z};
    }
    if (left_op == TermValue::Z && right_op == TermValue::I) {
        return {1., TermValue::Z};
    }
    if (left_op == TermValue::X && right_op == TermValue::X) {
        return {1., TermValue::I};
    }
    if (left_op == TermValue::Y && right_op == TermValue::Y) {
        return {1., TermValue::I};
    }
    if (left_op == TermValue::Z && right_op == TermValue::Z) {
        return {1., TermValue::I};
    }
    if (left_op == TermValue::X && right_op == TermValue::Y) {
        return {{0, 1.}, TermValue::Z};
    }
    if (left_op == TermValue::X && right_op == TermValue::Z) {
        return {{0, -1.}, TermValue::Y};
    }
    if (left_op == TermValue::Y && right_op == TermValue::X) {
        return {{0, -1.}, TermValue::Z};
    }
    if (left_op == TermValue::Y && right_op == TermValue::Z) {
        return {{0, 1.}, TermValue::X};
    }
    if (left_op == TermValue::Z && right_op == TermValue::X) {
        return {{0, 1.}, TermValue::Y};
    }
    if (left_op == TermValue::Z && right_op == TermValue::Y) {
        return {{0, -1.}, TermValue::X};
    }

    return {1., TermValue::I};
}

// =============================================================================

QubitOperator::QubitOperator(std::string_view terms_string, coefficient_t coeff)
    : QubitOperator(parse_string_(terms_string), coeff) {
}

// =============================================================================

uint32_t QubitOperator::count_gates() const noexcept {
    return std::accumulate(begin(terms_), end(terms_), 0UL, [](const auto& val, const auto& term) {
        const auto& [ops, coeff] = term;
        return val + std::size(ops);
    });
}
// =============================================================================

auto QubitOperator::matrix(std::optional<uint32_t> n_qubits) const -> std::optional<csr_matrix_t> {
    namespace Op = tweedledum::Op;
    using dense_2x2_t = Eigen::Matrix2cd;

    // NB: required since we are indexing into the array below
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::I) == 0);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::X) == 1);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::Y) == 2);
    static_assert(static_cast<std::underlying_type_t<TermValue>>(TermValue::Z) == 3);

    static const std::array<csr_matrix_t, 4> pauli_matrices = {
        dense_2x2_t::Identity().sparseView(),
        Op::X::matrix().sparseView(),
        Op::Y::matrix().sparseView(),
        Op::Z::matrix().sparseView(),
    };

    if (std::empty(terms_)) {
        return std::nullopt;
    }

    // NB: this is assuming that num_targets_ is up-to-date
    const auto& n_qubits_local = num_targets_;

    if (n_qubits_local == 0UL && !n_qubits) {
        std::cerr << "You should specify n_qubits for converting an identity qubit operator.";
        return std::nullopt;
    }

    if (n_qubits && n_qubits.value() < n_qubits_local) {
        std::cerr << fmt::format(
            "Given n_qubits {} is smaller than the number of qubits of this qubit operator, which is {}.\n",
            n_qubits.value(), n_qubits_local);
        return std::nullopt;
    }

    const auto n_qubits_value = n_qubits.value_or(n_qubits_local);

    const auto process_term = [n_qubits_value](const auto& local_ops, const auto& coeff) -> csr_matrix_t {
        if (std::empty(local_ops)) {
            return (Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Identity(
                        1U << n_qubits_value, 1U << n_qubits_value))
                       .sparseView()
                   * coeff;
        }

        // TODO(dnguyen): The `total` variable is probably not required and could be removed altogether...
        std::vector<csr_matrix_t> total(n_qubits_value,
                                        pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(TermValue::I)]);
        for (const auto& [qubit_id, local_op] : local_ops) {
            total[qubit_id] = pauli_matrices[static_cast<std::underlying_type_t<TermValue>>(local_op)];
        }

        csr_matrix_t init(1, 1);
        init.insert(0, 0) = coeff;
        return std::accumulate(begin(total), end(total), init, [](const csr_matrix_t& init, const auto& matrix) {
            return Eigen::kroneckerProduct(matrix, init).eval();
        });
    };

    auto it = begin(terms_);
    auto result = process_term(it->first, it->second);
    ++it;
    for (; it != end(terms_); ++it) {
        result += process_term(it->first, it->second);
    }

    return result;
}

// =============================================================================

auto QubitOperator::split() const noexcept -> std::vector<QubitOperator> {
    std::vector<QubitOperator> result;
    for (const auto& [local_ops, coeff] : terms_) {
        result.emplace_back(local_ops, coeff);
    }
    return result;
}

// =============================================================================

auto QubitOperator::loads(std::string_view string_data) -> std::optional<QubitOperator> {
    if (complex_term_dict_t terms_dict; parser::parse_object_skipper(begin(string_data), end(string_data), terms_dict,
                                                                     ::parser::qb_op::json_dict, x3::space)) {
        return QubitOperator(terms_dict);
    }
    MQ_ERROR("QubitOperator JSON string parsing failed for '{}'", string_data);
    return {};
}

// =============================================================================

auto QubitOperator::parse_string_(std::string_view terms_string) -> terms_t {
    MQ_INFO("Attempting to parse: '{}'", terms_string);
    if (terms_t terms; parser::parse_object_skipper(begin(terms_string), end(terms_string), terms,
                                                    ::parser::qb_op::terms, x3::space)) {
        return terms;
    }
    MQ_ERROR("QubitOperator terms string parsing failed for '{}'", terms_string);
    return {};
}

// =============================================================================

auto QubitOperator::simplify_(terms_t terms, coefficient_t coeff) -> std::tuple<terms_t, coefficient_t> {
    if (std::empty(terms)) {
        return {terms_t{}, coeff};
    }
    std::sort(
        begin(terms), end(terms), [](const auto& lhs, const auto& rhs) constexpr { return lhs.first < rhs.first; });

    terms_t reduced_terms;
    auto left_term = terms.front();
    for (auto it(begin(terms) + 1); it != end(terms); ++it) {
        const auto& [left_qubit_id, left_operator] = left_term;
        const auto& [right_qubit_id, right_operator] = *it;

        if (left_qubit_id == right_qubit_id) {
            const auto [new_coeff, new_op] = pauli_products(left_operator, right_operator);
            left_term = term_t{left_qubit_id, new_op};
            coeff *= new_coeff;
        } else {
            if (left_term.second != TermValue::I) {
                reduced_terms.emplace_back(left_term);
            }
            left_term = *it;
        }
    }

    return {std::move(terms), coeff};
}

// =============================================================================

auto QubitOperator::sort_terms_(terms_t local_ops, coefficient_t coeff) -> std::pair<terms_t, coefficient_t> {
    std::sort(begin(local_ops), end(local_ops));
    return {std::move(local_ops), coeff};  // TODO(dnguyen): Should we move? or can (N)RVO take care of that?
}

// =============================================================================

}  // namespace mindquantum::ops
