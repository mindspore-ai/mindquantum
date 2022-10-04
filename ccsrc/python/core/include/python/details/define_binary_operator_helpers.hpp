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

#ifndef DEFINE_BINARY_OPERATOR_HELPERS_HPP
#define DEFINE_BINARY_OPERATOR_HELPERS_HPP

#include <tuple>

#include <pybind11/pybind11.h>

namespace bindops {
#define TO_STRING1(X)    #X
#define TO_STRING(X)     TO_STRING1(X)
#define CONCAT2(A, B)    A##B
#define CONCAT3(A, B, C) A##B##C

#define PYBIND11_DEFINE_BINOP_IMPL(py_name, lhs_t, rhs_t, op)                                                          \
    def(                                                                                                               \
        py_name, [](lhs_t lhs, rhs_t rhs) { return lhs op rhs; }, pybind11::is_operator())
#define PYBIND11_DEFINE_BINOP_REV_IMPL(py_name, lhs_t, rhs_t, op)                                                      \
    def(                                                                                                               \
        py_name, [](lhs_t lhs, rhs_t rhs) { return rhs op lhs; }, pybind11::is_operator())
#define PYBIND11_DEFINE_BINOP(py_name, lhs_t, rhs_t, op) PYBIND11_DEFINE_BINOP_IMPL(#py_name, lhs_t, rhs_t, op)
#define PYBIND11_DEFINE_BINOP_INPLACE(py_name, lhs_t, rhs_t, op)                                                       \
    PYBIND11_DEFINE_BINOP_IMPL(TO_STRING(CONCAT3(__i, py_name, __)), lhs_t, rhs_t, CONCAT2(op, =))
#define PYBIND11_DEFINE_BINOP_EXT(py_name, lhs_t, rhs_t, op)                                                           \
    PYBIND11_DEFINE_BINOP_IMPL(TO_STRING(CONCAT3(__, py_name, __)), lhs_t, rhs_t, op)
#define PYBIND11_DEFINE_BINOP_REV(py_name, lhs_t, rhs_t, op)                                                           \
    PYBIND11_DEFINE_BINOP_REV_IMPL(TO_STRING(CONCAT3(__r, py_name, __)), lhs_t, rhs_t, op)
#define PYBIND11_DEFINE_UNOP(py_name, lhs_t, op)                                                                       \
    def(                                                                                                               \
        #py_name, [](lhs_t base) { return op base; }, pybind11::is_operator())

namespace details {
#define BINOP_DEF_CLASS_HELPER_TYPE(name, py_name, op)                                                                 \
    template <typename T>                                                                                              \
    struct name {                                                                                                      \
        template <typename py_klass_t>                                                                                 \
        static constexpr void inplace(py_klass_t& klass) {                                                             \
            using op_t = typename py_klass_t::type;                                                                    \
            klass.PYBIND11_DEFINE_BINOP_INPLACE(py_name, op_t&, T, op);                                                \
        }                                                                                                              \
        template <typename py_klass_t>                                                                                 \
        static constexpr void external(py_klass_t& klass) {                                                            \
            using op_t = typename py_klass_t::type;                                                                    \
            klass.PYBIND11_DEFINE_BINOP_EXT(py_name, const op_t&, T, op);                                              \
        }                                                                                                              \
        template <typename py_klass_t>                                                                                 \
        static constexpr void reverse(py_klass_t& klass) {                                                             \
            using op_t = typename py_klass_t::type;                                                                    \
            klass.PYBIND11_DEFINE_BINOP_REV(py_name, const op_t&, T, op);                                              \
        }                                                                                                              \
    }

BINOP_DEF_CLASS_HELPER_TYPE(plus, add, +);
BINOP_DEF_CLASS_HELPER_TYPE(minus, sub, -);
BINOP_DEF_CLASS_HELPER_TYPE(times, mul, *);
BINOP_DEF_CLASS_HELPER_TYPE(divides, truediv, /);

#undef BINOP_DEF_CLASS_HELPER_TYPE
}  // namespace details

// -----------------------------------------------------------------------------

template <template <typename... others_t> class func_t, typename py_klass_t>
struct binop_definition {
    template <typename... args_t>
    struct helper_t {
        static constexpr void inplace(py_klass_t& klass) {
        }
        static constexpr void external(py_klass_t& klass) {
        }
        static constexpr void reverse(py_klass_t& klass) {
        }
    };

    template <typename... args_t>
    static constexpr void inplace(py_klass_t& klass) {
        helper_t<args_t...>::inplace(klass);
    }
    template <typename... args_t>
    static constexpr void external(py_klass_t& klass) {
        helper_t<args_t...>::external(klass);
    }
    template <typename... args_t>
    static constexpr void reverse(py_klass_t& klass) {
        helper_t<args_t...>::reverse(klass);
    }
};

template <template <typename... others_t> class func_t, typename py_klass_t>
template <typename T, typename... args_t>
struct binop_definition<func_t, py_klass_t>::helper_t<T, args_t...> {
    static constexpr void inplace(py_klass_t& klass) {
        func_t<T>::inplace(klass);
        helper_t<args_t...>::inplace(klass);
    }
    static constexpr void external(py_klass_t& klass) {
        func_t<T>::external(klass);
        helper_t<args_t...>::external(klass);
    }
    static constexpr void reverse(py_klass_t& klass) {
        func_t<T>::reverse(klass);
        helper_t<args_t...>::reverse(klass);
    }
};
template <template <typename... others_t> class func_t, typename py_klass_t>
template <typename... args_t>
struct binop_definition<func_t, py_klass_t>::helper_t<std::tuple<args_t...>> : helper_t<args_t...> {};

template <template <typename... others_t> class func_t, typename py_klass_t>
template <typename... args_t, typename... other_args_t>
struct binop_definition<func_t, py_klass_t>::helper_t<std::tuple<args_t...>, other_args_t...>
    : helper_t<args_t..., other_args_t...> {};
}  // namespace bindops

#endif /* DEFINE_BINARY_OPERATOR_HELPERS_HPP */
