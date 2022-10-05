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

#ifndef GATE_BASE_HPP
#define GATE_BASE_HPP

#include <algorithm>
#include <cassert>
#include <complex>
#include <cstdint>
#include <ctime>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

#include <tweedledum/IR/Operator.h>

#include <symengine/basic.h>
#include <symengine/complex_double.h>
#include <symengine/eval_double.h>
#include <symengine/visitor.h>

// #include <frozen/unordered_map.h>

#include "config/detected.hpp"

#include "experimental/core/config.hpp"
#include "experimental/core/operator_traits.hpp"
#include "experimental/ops/parametric/config.hpp"
#include "experimental/ops/parametric/param_names.hpp"
#include "experimental/ops/parametric/to_symengine.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/gate_concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

// namespace frozen
// {
//      template <>
//      struct elsa<std::string_view>
//      {
//           constexpr std::size_t operator()(std::string_view value) const noexcept
//           {
//                std::size_t d = 5381;
//                for (std::size_t i = 0; i < value.size(); ++i)
//                     d = d * 33 + static_cast<size_t>(value[i]);
//                return d;
//           }
//           // https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
//           // With the lowest bits removed, based on experimental setup.
//           constexpr std::size_t operator()(std::string_view value, std::size_t seed) const noexcept
//           {
//                std::size_t d = (0x811c9dc5 ^ seed) * static_cast<size_t>(0x01000193);
//                for (std::size_t i = 0; i < value.size(); ++i)
//                     d = (d ^ static_cast<size_t>(value[i])) * static_cast<size_t>(0x01000193);
//                return d >> 8;
//           }
//      };
// }  // namespace frozen

namespace mindquantum::ops::parametric {
#if MQ_HAS_CONCEPTS
template <typename derived_t, typename non_param_t, concepts::parameter... params_t>
#else
template <typename derived_t, typename non_param_t, typename... params_t>
#endif  // MQ_HAS_CONCEPTS
class ParametricBase;
}  // namespace mindquantum::ops::parametric

namespace mindquantum::traits {
//! Detect if a template argument pack is a ParametricBase
/*!
 * Only needed as a workaround for GCC 7+ when MQ_HAS_CONCEPTS == false
 */
template <typename... Ts>
struct is_param_base : std::false_type {};

#if MQ_HAS_CONCEPTS
template <typename derived_t, typename non_param_t, concepts::parameter... params_t>
#else
template <typename derived_t, typename non_param_t, typename... params_t>
#endif  // MQ_HAS_CONCEPTS
struct is_param_base<ops::parametric::ParametricBase<derived_t, non_param_t, params_t...>> : std::true_type {
};
}  // namespace mindquantum::traits

namespace mindquantum::ops::parametric {
#if MQ_HAS_CONCEPTS
template <typename derived_t, typename non_param_t, concepts::parameter... params_t>
#else
template <typename derived_t, typename non_param_t, typename... params_t>
#endif  // MQ_HAS_CONCEPTS
class ParametricBase {
 public:
    static_assert(sizeof...(params_t) > 0, "Need to define at least 1 parameter");

    using base_t = ParametricBase;
    using self_t = ParametricBase;

    using operator_t = tweedledum::Operator;
    using is_parametric = void;
    using non_param_type = non_param_t;
    static constexpr auto num_params = sizeof...(params_t);
    using params_type = std::tuple<params_t...>;
    using param_array_t = std::array<basic_t, num_params>;

    // using map_t = details::umap_t<std::string_view, basic_t, num_params>;
    using subs_map_t = SymEngine::map_basic_basic;

    static constexpr auto has_const_num_targets = traits::has_const_num_targets_v<non_param_type>;

    // =====================================================================
    // :: create_op([num_targets])

    //! Create a default instance of \c derived_t type
    /*!
     * Overload only available if \c non_param_type has compile-time constant number of qubits
     */
    MQ_NODISCARD static constexpr derived_t create_op()
#if MQ_HAS_CONCEPTS
        requires(has_const_num_targets)
#endif  //  MQ_HAS_CONCEPTS
    {
        return derived_t{param_array_t{SymEngine::symbol(std::string(params_t::name))...}};
    }

    //! Create a default instance of \c derived_t type
    /*!
     * Overload only available if \c non_param_type does not have compile-time constant number of qubits
     */
    MQ_NODISCARD static constexpr derived_t create_op(uint32_t num_targets)
#if MQ_HAS_CONCEPTS
        requires(!has_const_num_targets)
#else
#endif  //  MQ_HAS_CONCEPTS
    {
        return derived_t{num_targets, param_array_t{SymEngine::symbol(std::string(params_t::name))...}};
    }

    // ---------------------------------------------------------------------
    // :: create_op([num_targets], transforms...)

    //! Create an instance of \c derived_t type with some transformation on the parameters
    /*!
     * Overload only available if \c non_param_type has compile-time constant number of qubits
     */
    template <typename... funcs_t>
    MQ_NODISCARD static constexpr derived_t create_op(funcs_t&&... transforms)
#if MQ_HAS_CONCEPTS
        requires(traits::has_const_num_targets_v<non_param_type>)
#endif  //  MQ_HAS_CONCEPTS
    {
        static_assert(sizeof...(funcs_t) == sizeof...(params_t),
                      "You need to specify as many transformation functions as there are parameters");
        return derived_t{param_array_t{expand(transforms(SymEngine::symbol(std::string(params_t::name))))...}};
    }

    //! Create an instance of \c derived_t type with some transformation on the parameters
    /*!
     * Overload only available if \c non_param_type does not have compile-time constant number of qubits
     */
    template <typename... funcs_t>
    MQ_NODISCARD static constexpr derived_t create_op(uint32_t num_targets, funcs_t&&... transforms)
#if MQ_HAS_CONCEPTS
        requires(!has_const_num_targets)
#endif  //  MQ_HAS_CONCEPTS
    {
        static_assert(sizeof...(funcs_t) == sizeof...(params_t),
                      "You need to specify as many transformation functions as there are parameters");
        return derived_t{num_targets,
                         param_array_t{expand(transforms(SymEngine::symbol(std::string(params_t::name))))...}};
    }

    // ---------------------------------------------------------------------
    // :: ParametricBase([num_targets], params...)

    //! Constructor from a list of either C++ numeric types or symbolic expressions
    /*!
     * \param expr List of SymEngine expressions
     * \note This constructor expects exactly \c num_params arguments
     */
#if MQ_HAS_CONCEPTS
    template <typename... Ts>
        requires(has_const_num_targets && (concepts::expr_or_number<Ts> || ...))
    constexpr ParametricBase(Ts&&... args)  // NOLINT(runtime/explicit)
#else
    template <typename... Ts, typename T = self_t,
              typename = std::enable_if_t<T::has_const_num_targets && !traits::is_param_base<Ts...>::value>>
    constexpr ParametricBase(Ts&&... args)  // NOLINT(runtime/explicit)
#endif  // MQ_HAS_CONCEPTS
        : num_targets_(traits::num_targets<non_param_type>), params_{to_symengine(std::forward<Ts>(args))...} {
        static_assert(sizeof...(Ts) == num_params, "You need to specify a value for all the parameters");
    }

    //! Constructor from a list of either C++ numeric types or symbolic expressions
    /*!
     * \param num_targets Number of target qubits
     * \param expr List of SymEngine expressions
     * \note This constructor expects exactly \c num_params arguments
     */
#if MQ_HAS_CONCEPTS
    template <typename... Ts>
        requires(!has_const_num_targets && (concepts::expr_or_number<Ts> || ...))
    constexpr ParametricBase(uint32_t num_targets, Ts&&... args)  // NOLINT(runtime/explicit)
#else
    template <typename... Ts, typename T = self_t, typename = std::enable_if_t<!T::has_const_num_targets>>
    constexpr ParametricBase(uint32_t num_targets, Ts&&... args)  // NOLINT(runtime/explicit)
#endif  // MQ_HAS_CONCEPTS
        : num_targets_(num_targets), params_{to_symengine(std::forward<Ts>(args))...} {
        static_assert(sizeof...(Ts) == num_params, "You need to specify a value for all the parameters");
    }

    // ---------------------------------------------------------------------
    // :: ParametricBase([num_targets], params)

    //! Constructor from an array of parameters
#if MQ_HAS_CONCEPTS
    constexpr ParametricBase(param_array_t&& params) requires(has_const_num_targets)  // NOLINT(runtime/explicit)
#else
    template <typename T = self_t, typename = std::enable_if_t<T::has_const_num_targets>>
    constexpr ParametricBase(param_array_t&& params)  // NOLINT(runtime/explicit)
#endif  //  MQ_HAS_CONCEPTS
        : num_targets_(traits::num_targets<non_param_type>), params_{expand_all(std::move(params))} {
    }

    //! Constructor from an array of parameters
#if MQ_HAS_CONCEPTS
    constexpr ParametricBase(uint32_t num_targets, param_array_t&& params) requires(!has_const_num_targets)
#else
    template <typename T = self_t, typename = std::enable_if_t<!T::has_const_num_targets>>
    constexpr ParametricBase(uint32_t num_targets, param_array_t&& params)
#endif  //  MQ_HAS_CONCEPTS
        : num_targets_(num_targets), params_{expand_all(std::move(params))} {
    }

    // ---------------------------------------------------------------------
    // Other defaulted/deleted constructors

    ParametricBase() = delete;
    ParametricBase(const ParametricBase&) = default;

#if defined(MQ_CLANG_MAJOR) && MQ_CLANG_MAJOR < 9
    ParametricBase(ParametricBase&&) = default;
#else
    ParametricBase(ParametricBase&&) noexcept = default;
#endif  // MQ_CLANG_MAJOR

    ParametricBase& operator=(const ParametricBase&) = default;
    ParametricBase& operator=(ParametricBase&&) noexcept = default;

    // ---------------------------

#if MQ_HAS_CONCEPTS
    MQ_NODISCARD static constexpr auto num_targets() noexcept requires(has_const_num_targets)
#else
    /* This is totally hacky...
     *
     * We are essentially relying on the fact that people will be using the type traits
     * traits::num_targets<operator_t> in order to get the number of targets at compile time (if available)
     */
    template <typename T = non_param_type, typename = std::enable_if_t<traits::has_const_num_targets_v<T>, uint32_t>>
    MQ_NODISCARD static constexpr auto num_targets_static() noexcept
#endif  //  MQ_HAS_CONCEPTS
    {
        return traits::num_targets<non_param_type>;
    }

#if MQ_HAS_CONCEPTS
    MQ_NODISCARD constexpr auto num_targets() const noexcept requires(!has_const_num_targets)
#else
    template <typename T = non_param_type, typename = std::enable_if_t<!traits::has_const_num_targets_v<T>, uint32_t>>
    MQ_NODISCARD constexpr auto num_targets() const noexcept
#endif  //  MQ_HAS_CONCEPTS
    {
        return num_targets_;
    }

    //! Test whether another operation is the same as this instance
    MQ_NODISCARD bool operator==(const ParametricBase& other) const noexcept {
        return std::equal(begin(params_), end(params_), begin(other.params_),
                          [](const auto& a, const auto& b) { return eq(*a, *b); });
    }

#if !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS
    //! Test whether another operation is the same as this instance
    MQ_NODISCARD bool operator!=(const ParametricBase& other) const noexcept {
        return !(*this == other);
    }
#endif  // !MQ_HAS_OPERATOR_NOT_EQUAL_SYNTHESIS

    //! True if this operation has no particular ordering of qubits
    MQ_NODISCARD bool is_symmetric() const noexcept {
        return true;
    }

    //! Get the name of the parameter at some index
    /*!
     * \param idx Index of parameter
     */
    MQ_NODISCARD static constexpr const auto& param_name(std::size_t idx) {
        assert(idx < num_params);
        return pos_[idx];
    }

    //! Parameter getter method
    /*!
     * \param idx Index of parameter
     * \return Parameter at index \c idx
     */
    MQ_NODISCARD constexpr const auto& param(std::size_t idx) const {
        assert(idx < num_params);
        return params_[idx];
    }

    //! Get all the parameters in an array
    /*!
     * \return SymEngine::vec_basic (\c std::vector<...>) containing all parameters
     */
    MQ_NODISCARD auto params() const noexcept {
        SymEngine::vec_basic params;
        std::copy(begin(params_), end(params_), std::inserter(params, end(params)));
        return params;
    }

    //! Evaluate the parameters of this parametric gate using some substitutions
    /*!
     * This function does not attempt to fully evaluate this parametric gate (ie. evaluate all parameter
     * numerically)
     *
     * \param subs_map Dictionary containing all the substitution to perform
     * \return An new instance of the parametric gate with evaluated parameters
     * \sa non_param_type eval_full(const SymEngine::map_basic_basic& subs_map) const
     */
    MQ_NODISCARD derived_t eval(const subs_map_t& subs_map) const {
        auto new_params = base_t::params_;
        for (auto& new_param : new_params) {
            // NB: expand required to normalize the expressions (e.g. required for testing for equality)
            new_param = expand(new_param->subs(subs_map));
        }
        if constexpr (has_const_num_targets) {
            return {std::move(new_params)};
        } else {
            return {num_targets_, std::move(new_params)};
        }
    }

    //! Fully evaluate this parametric gate
    /*!
     * Attempt to fully evaluate this parametric gate (ie. evaluate all parameter numerically). The constructor
     * of the non-parametric gate type is called by passing each of the numerically evaluated parameter in the
     * order that is defined when passing the type as the template parameters to this base class.
     *
     * \return An instance of a non-parametric gate (\c non_param_type)
     * \throw SymEngine::SymEngineException if the expression cannot be fully evaluated numerically
     */
    MQ_NODISCARD auto eval_full() const {
        return eval_full_impl_(std::index_sequence_for<params_t...>{});
    }

    //! Fully evaluate this parametric gate
    /*!
     * Attempt to fully evaluate this parametric gate (ie. evaluate all parameter numerically). The constructor
     * of the non-parametric gate type is called by passing each of the numerically evaluated parameter in the
     * order that is defined when passing the type as the template parameters to this base class.
     *
     * This overload accepts a dictionary of substitutions to perform on the parameters.
     *
     * \param subs_map Dictionary containing all the substitution to perform
     * \return An instance of a non-parametric gate (\c non_param_type)
     * \throw SymEngine::SymEngineException if the expression cannot be fully evaluated numerically
     */
    MQ_NODISCARD auto eval_full(const subs_map_t& subs_map) const {
        return eval_full_impl_(std::index_sequence_for<params_t...>{}, subs_map);
    }

    //! Evaluate the parameters of this parametric gate using some substitutions
    /*!
     * This function will attempt to fully evaluate this parametric gate if possible. This will happen only if,
     * after substitutions, all the parameters have a numeric representation.
     *
     * \param subs_map Dictionary containing all the substitution to perform
     * \return An new instance of the parametric gate with evaluated parameters
     * \sa non_param_type eval_full(const SymEngine::map_basic_basic& subs_map) const
     */
    MQ_NODISCARD operator_t eval_smart() const {
        auto new_params = base_t::params_;
        for (auto& new_param : new_params) {
            // NB: expand required to normalize the expressions (e.g. required for testing for equality)
            new_param = expand(new_param);
        }

        if (std::all_of(begin(new_params), end(new_params), [](const auto& p) { return is_a_Number(*p); })) {
            return eval_smart_impl_(std::index_sequence_for<params_t...>{}, std::move(new_params));
        }

        // TODO(dnguyen): Add support for default implementation of `to_param_type`
        return derived_t::to_param_type(*static_cast<const derived_t*>(this), std::move(new_params));
    }

    //! Evaluate the parameters of this parametric gate using some substitutions
    /*!
     * This function will attempt to fully evaluate this parametric gate if possible. This will happen only if,
     * after substitutions, all the parameters have a numeric representation.
     *
     * \param subs_map Dictionary containing all the substitution to perform
     * \return An new instance of the parametric gate with evaluated parameters
     * \sa non_param_type eval_full(const SymEngine::map_basic_basic& subs_map) const
     */
    MQ_NODISCARD operator_t eval_smart(const subs_map_t& subs_map) const {
        auto new_params = base_t::params_;
        for (auto& new_param : new_params) {
            // NB: expand required to normalize the expressions (e.g. required for testing for equality)
            new_param = expand(new_param->subs(subs_map));
        }

        if (std::all_of(begin(new_params), end(new_params), [](const auto& p) { return is_a_Number(*p); })) {
            return eval_smart_impl_(std::index_sequence_for<params_t...>{}, std::move(new_params));
        }

        // TODO(dnguyen): Add support for default implementation of `to_param_type`
        return derived_t::to_param_type(*static_cast<const derived_t*>(this), std::move(new_params));
    }

 protected:
    static constexpr std::array<std::string_view, sizeof...(params_t)> pos_ = {params_t::name...};
    // TODO(dnguyen): remove this attribute for operator that have compile-time constant number of targets
    uint32_t num_targets_;
    const param_array_t params_;

 private:
    static constexpr auto expand_all(param_array_t&& params) {
        return expand_all_impl(std::make_index_sequence<std::tuple_size_v<param_array_t>>{}, std::move(params));
    }

    template <std::size_t... indices>
    static constexpr auto expand_all_impl(std::index_sequence<indices...> /*unused*/, param_array_t&& params) {
        return param_array_t{expand(params[indices])...};
    }

    //! Helper function for \c eval_full
    template <std::size_t... indices>
    constexpr auto eval_full_impl_(std::index_sequence<indices...> /*unused*/) const {
        // NB: expand required to normalize the expressions (e.g. required for testing for equality)
        // TODO(dnguyen): Add support for default implementation of `to_non_param_type`
        return derived_t::to_non_param_type(*static_cast<const derived_t*>(this),
                                            params_t::param_type::eval(expand(params_[indices]))...);
    }

    //! Helper function for \c eval_full
    template <std::size_t... indices>
    constexpr auto eval_full_impl_(std::index_sequence<indices...> /*unused*/, const subs_map_t& subs_map) const {
        // NB: expand required to normalize the expressions (e.g. required for testing for equality)
        // TODO(dnguyen): Add support for default implementation of `to_non_param_type`
        return derived_t::to_non_param_type(*static_cast<const derived_t*>(this),
                                            params_t::param_type::eval(expand(params_[indices]->subs(subs_map)))...);
    }

    //! Helper function for \c eval_smart
    template <std::size_t... indices>
    constexpr auto eval_smart_impl_(std::index_sequence<indices...> /*unused*/,
                                    std::array<basic_t, num_params>&& params) const {
        // TODO(dnguyen): Add support for default implementation of `to_non_param_type`
        return derived_t::to_non_param_type(*static_cast<const derived_t*>(this),
                                            params_t::param_type::eval(params[indices])...);
    }
};
}  // namespace mindquantum::ops::parametric

#endif /* GATE_BASE_HPP */
