//   Copyright 2021 <Huawei Technologies Co., Ltd>
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

#ifndef DECOMPOSITIONS_CONCEPTS_HPP
#define DECOMPOSITIONS_CONCEPTS_HPP

#include <type_traits>
#include <vector>

#include "experimental/ops/parametric/config.hpp"

namespace tweedledum {
class Circuit;
class Qubit;
}  // namespace tweedledum

namespace mindquantum::concepts {
using circuit_t = tweedledum::Circuit;
using qubits_t = std::vector<tweedledum::Qubit>;

using ops::parametric::double_list_t;
using ops::parametric::gate_param_t;
using ops::parametric::param_list_t;
}  // namespace mindquantum::concepts

#if MQ_HAS_CONCEPTS
#    include <string_view>

#    include "experimental/core/concepts.hpp"
#    include "experimental/core/traits.hpp"
#    include "experimental/decompositions/config.hpp"

namespace mindquantum::decompositions {
class AtomStorage;
}  // namespace mindquantum::decompositions

namespace mindquantum::concepts {
template <typename atom_t>
concept BaseDecomposition = requires(atom_t atom, decompositions::AtomStorage& storage, const instruction_t& inst) {
    { std::remove_cvref_t<atom_t>::name() } -> std::same_as<std::string_view>;
    { std::remove_cvref_t<atom_t>::num_targets() } -> std::same_as<decompositions::num_target_t>;
    { std::remove_cvref_t<atom_t>::num_controls() } -> std::same_as<decompositions::num_control_t>;
    {std::remove_cvref_t<atom_t>::create(storage)};
    requires std::constructible_from<std::remove_cvref_t<atom_t>, decompositions::AtomStorage&>;
    { atom.is_applicable(inst) } -> std::same_as<bool>;
};

template <typename atom_t>
concept GateDecomposition = requires(atom_t) {
    requires BaseDecomposition<atom_t>;
    typename std::remove_cvref_t<atom_t>::kinds_t;
    requires traits::is_tuple_v<typename std::remove_cvref_t<atom_t>::kinds_t>;
    { std::remove_cvref_t<atom_t>::num_params() } -> std::same_as<decompositions::num_param_t>;
};

template <typename atom_t>
concept GeneralDecomposition = requires(atom_t) {
    requires BaseDecomposition<atom_t>;
    requires !GateDecomposition<atom_t>;
    typename std::remove_cvref_t<atom_t>::non_gate_decomposition;
};

template <typename T>
concept has_non_param_apply = requires(T t, circuit_t circuit, qubits_t qubits) {
    {t.apply(circuit, qubits)};
};

template <typename T>
concept has_param_apply = requires(T t, circuit_t circuit, qubits_t qubits, const gate_param_t params) {
    {t.apply(circuit, qubits, params)};
};

template <typename T>
concept has_double_apply = requires(T t, circuit_t circuit, qubits_t qubits, double d) {
    {t.apply(circuit, qubits, d)};
};
template <typename T>
concept has_double_list_apply = requires(T t, circuit_t circuit, qubits_t qubits, double_list_t v) {
    {t.apply(circuit, qubits, v)};
};
template <typename T>
concept has_param_list_apply = requires(T t, circuit_t circuit, qubits_t qubits, param_list_t v) {
    {t.apply(circuit, qubits, v)};
};
}  // namespace mindquantum::concepts
#else
namespace mindquantum::concepts {
template <typename atom_t, class = void>
struct GateDecomposition_ : std::false_type {};

template <typename atom_t>
struct GateDecomposition_<atom_t, std::void_t<typename atom_t::kinds_t>> : std::true_type {};

template <typename atom_t>
static constexpr auto GateDecomposition = GateDecomposition_<atom_t>::value;

template <typename atom_t, class = void>
struct GeneralDecomposition_ : std::false_type {};

template <typename atom_t>
struct GeneralDecomposition_<atom_t, std::void_t<typename atom_t::non_gate_decomposition>> : std::true_type {};

template <typename atom_t>
static constexpr auto GeneralDecomposition = !GateDecomposition<atom_t> && GeneralDecomposition_<atom_t>::value;

// ---------------------------------

template <class Op, class = void>
struct has_non_param_apply_ : std::false_type {};

template <class Op>
struct has_non_param_apply_<
    Op, std::void_t<decltype(std::declval<Op>().apply(std::declval<circuit_t&>(), std::declval<qubits_t&>()))>>
    : std::true_type {};

template <class Op>
static constexpr auto has_non_param_apply = has_non_param_apply_<Op>::value;

// ---------------------------------

template <class Op, class = void>
struct has_param_apply_ : std::false_type {};

template <class Op>
struct has_param_apply_<Op, std::void_t<decltype(std::declval<Op>().apply(
                                std::declval<circuit_t&>(), std::declval<qubits_t&>(), std::declval<gate_param_t&>()))>>
    : std::true_type {};

template <class Op>
static constexpr auto has_param_apply = has_param_apply_<Op>::value;

// ---------------------------------

template <class Op, class = void>
struct has_double_apply_ : std::false_type {};

template <class Op>
struct has_double_apply_<Op, std::void_t<decltype(std::declval<Op>().apply(
                                 std::declval<circuit_t&>(), std::declval<qubits_t&>(), std::declval<double>()))>>
    : std::true_type {};
template <class Op>
static constexpr auto has_double_apply = has_double_apply_<Op>::value;

// ---------------------------------

template <class Op, class = void>
struct has_double_list_apply_ : std::false_type {};

template <class Op>
struct has_double_list_apply_<
    Op, std::void_t<decltype(std::declval<Op>().apply(std::declval<circuit_t&>(), std::declval<qubits_t&>(),
                                                      std::declval<double_list_t>()))>> : std::true_type {};
template <class Op>
static constexpr auto has_double_list_apply = has_double_list_apply_<Op>::value;

// ---------------------------------

template <class Op, class = void>
struct has_param_list_apply_ : std::false_type {};

template <class Op>
struct has_param_list_apply_<
    Op, std::void_t<decltype(std::declval<Op>().apply(std::declval<circuit_t&>(), std::declval<qubits_t&>(),
                                                      std::declval<param_list_t&>()))>> : std::true_type {};
template <class Op>
static constexpr auto has_param_list_apply = has_param_list_apply_<Op>::value;
}  // namespace mindquantum::concepts
#endif  // MQ_HAS_CONCEPTS

#endif /* DECOMPOSITIONS_CONCEPTS_HPP */
