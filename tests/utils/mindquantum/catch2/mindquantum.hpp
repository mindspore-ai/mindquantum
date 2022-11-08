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

#ifndef MQ_CATCH2_MINDQUANTUM_HPP
#define MQ_CATCH2_MINDQUANTUM_HPP

#include <string>

#include "mindquantum/catch2/catch2_fmt_formatter.hpp"
#include "ops/gates/term_value.hpp"

#ifdef MINDQUANTUM_CXX_EXPERIMENTAL
#    include "experimental/core/circuit_block.hpp"
#endif  // MINDQUANTUM_CXX_EXPERIMENTAL

#include <catch2/catch_tostring.hpp>

// =============================================================================

#ifdef MINDQUANTUM_CXX_EXPERIMENTAL
namespace Catch {
template <>
struct StringMaker<mindquantum::QubitID> : mindquantum::details::FmtStringMakerBase<mindquantum::QubitID> {};

template <>
struct StringMaker<mindquantum::ops::TermValue> {
    static std::string convert(mindquantum::ops::TermValue value) {
        using mindquantum::ops::TermValue;
        static const auto& enumInfo = ::Catch::getMutableRegistryHub().getMutableEnumValuesRegistry().registerEnum(
            "mindquantum::ops::TermValue", "I, X, Y, Z, a, adg",
            {TermValue::I, TermValue::X, TermValue::Y, TermValue::Z, TermValue::a, TermValue::adg});
        return static_cast<std::string>(enumInfo.lookup(static_cast<int>(value)));
    }
};
}  // namespace Catch
#endif  // MINDQUANTUM_CXX_EXPERIMENTAL

// =============================================================================

#endif /* MQ_CATCH2_MINDQUANTUM_HPP */
