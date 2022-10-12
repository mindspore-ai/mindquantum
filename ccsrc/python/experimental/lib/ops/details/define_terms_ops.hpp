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

#ifndef PYTHON_DETAILS_DEFINE_TERMS_OPS_HPP
#define PYTHON_DETAILS_DEFINE_TERMS_OPS_HPP

#include <cstdint>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "config/constexpr_type_name.hpp"
#include "config/type_traits.hpp"

#include "experimental/ops/gates/fermion_operator.hpp"
#include "experimental/ops/gates/qubit_operator.hpp"
#include "experimental/ops/gates/terms_operator_base.hpp"

#include "python/details/create_from_container_class.hpp"
#include "python/details/define_binary_operator_helpers.hpp"
#include "python/details/get_fully_qualified_tp_name.hpp"

namespace bindops {
namespace ops = mindquantum::ops;
namespace python = mindquantum::python;
using namespace pybind11::literals;  // NOLINT(build/namespaces_literals)

// -----------------------------------------------------------------------------

template <class op_t, class base_t>
auto bind_ops(pybind11::module& module, const std::string_view& name) {  // NOLINT
    using coeff_t = typename op_t::coefficient_t;
    // NB: this below is required because of GCC < 9
    using factory_func_t = decltype(&python::create_from_python_container_class<op_t>);

    auto klass
        = pybind11::class_<op_t, base_t, std::shared_ptr<op_t>>(module, name.data())
              // ------------------------------
              // Constructors
              .def(pybind11::init<>())
              .def(pybind11::init<const ops::term_t&, coeff_t>(), "term"_a, "coeff"_a = op_t::coeff_policy_t::one)
              .def(pybind11::init<const ops::terms_t&, coeff_t>(), "terms"_a, "coeff"_a = op_t::coeff_policy_t::one)
              .def(pybind11::init<const ops::py_terms_t&, coeff_t>(), "terms"_a, "coeff"_a = op_t::coeff_policy_t::one)
              .def(pybind11::init<const typename op_t::coeff_term_dict_t&>(), "coeff_terms"_a)
              .def(pybind11::init<std::string_view, coeff_t>(), "terms_string"_a, "coeff"_a = op_t::coeff_policy_t::one)
              //! *VERY* important: this overload below needs to be the LAST
              .def(pybind11::init(static_cast<factory_func_t>(&python::create_from_python_container_class<op_t>)),
                   "py_class"_a, "Constructor from the encapsulating Python class (using a _cpp_obj attribute)");

    // ------------------------------
    // Properties
#if (defined __GNUC__) && __GNUC__ == 7 && __GNUC_MINOR__ == 5
    const auto read_func = static_cast<coeff_t (op_t::*)() const>(&op_t::constant);
    const auto write_func = static_cast<coeff_t (op_t::*)() const>(&op_t::constant);
    klass.def_property("constant", read_func, write_func);
#else
    klass.def_property("constant", static_cast<coeff_t (op_t::*)() const>(&op_t::constant),
                       static_cast<coeff_t (op_t::*)() const>(&op_t::constant));
#endif /* GCC 7.5.X */
    klass.def_property_readonly("imag", &op_t::imag)
        .def_property_readonly(
            "is_complex", [](const op_t&) constexpr { return !op_t::is_real_valued; })
        .def_property_readonly("is_singlet", &op_t::is_singlet)
        .def_property_readonly("real", &op_t::real)
        .def_property_readonly("size", &op_t::size)
        // ------------------------------
        // Member functions
        .def("cast_complex", &op_t::template cast<mindquantum::traits::to_cmplx_type_t<coeff_t>>)
        .def("compress", &op_t::compress, "abs_tol"_a = op_t::EQ_TOLERANCE)
        .def("count_qubits", &op_t::count_qubits)
        .def("dumps", &op_t::dumps, "indent"_a = 4)
        .def("get_coeff", &op_t::get_coeff)
        .def("hermitian", &op_t::hermitian)
        .def("is_identity", &op_t::is_identity, "abs_tol"_a = op_t::EQ_TOLERANCE)
        .def("matrix", &op_t::sparse_matrix, "n_qubits"_a)
        .def("num_targets", &op_t::num_targets)
        .def("singlet", &op_t::singlet)
        .def("singlet_coeff", &op_t::singlet_coeff)
        .def("split", &op_t::split)
        .def("subs", &op_t::subs, "subs_proxy"_a)
        .def("terms", &op_t::get_terms_pair)
        .def_static("identity", &op_t::identity)
        .def_static("loads", op_t::loads, "string_data"_a)
        // ------------------------------
        // Python magic methods
        .def("__len__", &op_t::size, pybind11::is_operator())
        .def(
            "__copy__", [](const op_t& base) -> op_t { return base; }, pybind11::is_operator())
        .def(
            "__str__", [](const op_t& base) { return base.to_string(); }, pybind11::is_operator())
        .def(
            "__repr__",
            [](const op_t& base) {
                return fmt::format("{}({})", mindquantum::get_type_name<op_t>(), base.to_string());
            },
            pybind11::is_operator())
        // ------------------------------
        // Python arithmetic operators
        .PYBIND11_DEFINE_BINOP_INPLACE(add, op_t&, const op_t&, +)
        .PYBIND11_DEFINE_BINOP_EXT(add, const op_t&, const op_t&, +)
        .PYBIND11_DEFINE_BINOP_INPLACE(sub, op_t&, const op_t&, -)
        .PYBIND11_DEFINE_BINOP_EXT(sub, const op_t&, const op_t&, -)
        .PYBIND11_DEFINE_BINOP_INPLACE(mul, op_t&, const op_t&, *)
        .PYBIND11_DEFINE_BINOP_EXT(mul, const op_t&, const op_t&, *)
        .PYBIND11_DEFINE_UNOP(__neg__, const op_t&, -)
        .PYBIND11_DEFINE_BINOP(__eq__, const op_t&, const op_t&, ==)
        .def(
            "__pow__", [](const op_t& base, unsigned int exponent) { return base.pow(exponent); },
            pybind11::is_operator());
    pybind11::implicitly_convertible<pybind11::object, op_t>();
    return klass;
}

template <typename T>
using fop_t = ops::FermionOperator<T>;

template <typename T>
using qop_t = ops::QubitOperator<T>;

// -----------------------------------------------------------------------------

template <typename... args_t>
struct define_fermion_ops {
    template <typename... strings_t>
    static auto apply(pybind11::module& module, strings_t&&... names) {  // NOLINT
        static_assert(sizeof...(args_t) == sizeof...(strings_t));
        return std::tuple_cat(
            std::make_tuple(bind_ops<fop_t<args_t>, ops::FermionOperatorBase>(module, std::forward<strings_t>(names))
                                .def("normal_ordered", &fop_t<args_t>::normal_ordered))...);
    }
};

template <typename... args_t>
struct define_qubit_ops {
    template <typename... strings_t>
    static auto apply(pybind11::module& module, strings_t&&... names) {  // NOLINT
        static_assert(sizeof...(args_t) == sizeof...(strings_t));
        return std::make_tuple(bind_ops<qop_t<args_t>, ops::QubitOperatorBase>(module, std::forward<strings_t>(names))
                                   .def("count_gates", &qop_t<args_t>::count_gates)...);
    }
};

// -----------------------------------------------------------------------------

template <typename self_t, typename... types_t>
struct cast_helper_impl;

template <typename self_t>
struct cast_helper_impl<self_t> {
    static pybind11::object try_cast(const self_t& /* self */, const pybind11::object& type) {
        MQ_DEBUG(fmt::format(
            "Invalid type passed to cast() member function: {}",
            pybind11::detail::get_fully_qualified_tp_name(std::launder(reinterpret_cast<PyTypeObject*>(type.ptr())))));
        throw std::runtime_error(fmt::format(
            "Invalid type passed to cast() member function: {}",
            pybind11::detail::get_fully_qualified_tp_name(std::launder(reinterpret_cast<PyTypeObject*>(type.ptr())))));
        return pybind11::none();
    }
};

template <typename self_t, typename T, typename... types_t>
struct cast_helper_impl<self_t, T, types_t...> {
    static pybind11::object try_cast(const self_t& self, const pybind11::object& type) {
        MQ_DEBUG(
            "Trying to cast {} to {} using C++ type {}", mindquantum::get_type_name<self_t>(),
            pybind11::detail::get_fully_qualified_tp_name(std::launder(reinterpret_cast<PyTypeObject*>(type.ptr()))),
            mindquantum::get_type_name<T>());
        if constexpr (!std::is_base_of<pybind11::detail::type_caster_generic,
                                       pybind11::detail::make_caster<std::remove_cvref_t<T>>>::value) {
            if constexpr (self_t::is_real_valued) {
                if (PyFloat_Check(type.ptr())) {
                    const auto value = self.template cast<double>();
                    return pybind11::cast(value);
                }
            }
            if (PyComplex_Check(type.ptr())) {
                const auto value = self.template cast<std::complex<double>>();
                return pybind11::cast(value);
            }
        } else {
            if (type.is(pybind11::type::of<T>())) {
                const auto value = self.template cast<T>();
                return pybind11::cast(value);
            }
        }

        // If casting was not possible, keep trying other types
        return cast_helper_impl<self_t, types_t...>::try_cast(self, type);
    }
};

template <typename self_t, typename... types_t>
pybind11::object cast(const self_t& self, const pybind11::object& type) {
    if (!PyType_Check(type.ptr())) {
        throw pybind11::type_error("Expect type as argument!");
    }
    return cast_helper_impl<self_t, types_t...>::try_cast(self, type);
}
}  // namespace bindops

#endif /* PYTHON_DETAILS_DEFINE_TERMS_OPS_HPP */
