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

#ifndef MQ_FORMAT_TWEEDLEDUM_HPP
#define MQ_FORMAT_TWEEDLEDUM_HPP

#include <vector>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/IR/Instruction.h>
#include <tweedledum/IR/Operator.h>
#include <tweedledum/IR/Qubit.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "experimental/ops/gates.hpp"
#include "experimental/ops/meta/dagger.hpp"

// =============================================================================

namespace mindquantum::fmt_details {
struct formatter_base {
    FMT_CONSTEXPR auto parse(::fmt::format_parse_context& ctx) -> decltype(ctx.begin()) {
        return ctx.begin();
    }
};
}  // namespace mindquantum::fmt_details

//! Custom formatter for a tweedledum::Qubit:::Polarity
template <typename char_type>
struct fmt::formatter<tweedledum::Qubit::Polarity, char_type> : mindquantum::fmt_details::formatter_base {
    using type_t = tweedledum::Qubit::Polarity;

    template <typename format_context_t>
    auto format(const type_t& polarity, format_context_t& ctx) const -> decltype(ctx.out()) {
        if (polarity == tweedledum::Qubit::positive) {
            return fmt::format_to(ctx.out(), "pos");
        }
        return fmt::format_to(ctx.out(), "neg");
    }
};

//! Custom formatter for a tweedledum::Qubit
template <typename char_type>
struct fmt::formatter<tweedledum::Qubit, char_type> : mindquantum::fmt_details::formatter_base {
    using type_t = tweedledum::Qubit;

    template <typename format_context_t>
    auto format(const type_t& qubit, format_context_t& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "Q[{}]({})", qubit.uid(), qubit.polarity());
    }
};

// -----------------------------------------------------------------------------

//! Custom formatter for a tweedledum::Cbit:::Polarity
template <typename char_type>
struct fmt::formatter<tweedledum::Cbit::Polarity, char_type> : mindquantum::fmt_details::formatter_base {
    using type_t = tweedledum::Cbit::Polarity;

    template <typename format_context_t>
    auto format(const type_t& polarity, format_context_t& ctx) const -> decltype(ctx.out()) {
        if (polarity == tweedledum::Cbit::positive) {
            return fmt::format_to(ctx.out(), "pos");
        }
        return fmt::format_to(ctx.out(), "neg");
    }
};

//! Custom formatter for a tweedledum::Cbit
template <typename char_type>
struct fmt::formatter<tweedledum::Cbit, char_type> : mindquantum::fmt_details::formatter_base {
    using type_t = tweedledum::Cbit;

    template <typename format_context_t>
    auto format(const type_t& cbit, format_context_t& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "C[{}]({})", cbit, cbit.polarity());
    }
};

// -----------------------------------------------------------------------------

//! Custom formatter for a tweedledum::Operator
template <typename char_type>
struct fmt::formatter<tweedledum::Operator, char_type> : mindquantum::fmt_details::formatter_base {
    using type_t = tweedledum::Operator;

    template <typename format_context_t>
    auto format(const type_t& op, format_context_t& ctx) const -> decltype(ctx.out()) {
        namespace ops = mindquantum::ops;
        if (op.is_a<mindquantum::ops::DaggerOperation>()) {
            fmt::format_to(ctx.out(), "Dagger({})", op.adjoint());
        } else {
            fmt::format_to(ctx.out(), "{}", op);
        }
        if (op.is_a<ops::P>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::P>().angle());
        } else if (op.is_a<ops::Ph>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::Ph>().angle());
        } else if (op.is_a<ops::Rx>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::Rx>().angle());
        } else if (op.is_a<ops::Rxx>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::Rxx>().angle());
        } else if (op.is_a<ops::Ry>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::Ry>().angle());
        } else if (op.is_a<ops::Ryy>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::Ryy>().angle());
        } else if (op.is_a<ops::Rz>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::Rz>().angle());
        } else if (op.is_a<ops::Rzz>()) {
            fmt::format_to(ctx.out(), "({})", op.cast<ops::Rzz>().angle());
        }
        return ctx.out();
    }
};

// -----------------------------------------------------------------------------

//! Custom formatter for a tweedledum::Instruction
template <typename char_type>
struct fmt::formatter<tweedledum::Instruction, char_type> : mindquantum::fmt_details::formatter_base {
    using type_t = tweedledum::Instruction;

    template <typename format_context_t>
    auto format(const type_t& inst, format_context_t& ctx) const -> decltype(ctx.out()) {
        return fmt::format_to(ctx.out(), "{} [{}]", static_cast<const tweedledum::Operator&>(inst), inst.qubits());
    }
};

// -----------------------------------------------------------------------------

//! Custom formatter for a tweedledum::Circuit
template <typename char_type>
struct fmt::formatter<tweedledum::Circuit, char_type> : mindquantum::fmt_details::formatter_base {
    using type_t = tweedledum::Circuit;

    template <typename format_context_t>
    auto format(const type_t& circuit, format_context_t& ctx) const -> decltype(ctx.out()) {
        fmt::format_to(ctx.out(), "Circuit ({} | {})", circuit.qubits(), circuit.cbits());

        circuit.foreach_instruction(
            [&ctx](const tweedledum::Instruction& inst) { fmt::format_to(ctx.out(), "{}\n", inst); });
        return ctx.out();
    }
};

// =============================================================================

#endif /* MQ_FORMAT_TWEEDLEDUM_HPP */
