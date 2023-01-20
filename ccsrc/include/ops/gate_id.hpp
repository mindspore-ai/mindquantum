//   Copyright 2023 <Huawei Technologies Co., Ltd>
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

#ifndef MQ_GATE_ID
#define MQ_GATE_ID

#include <fmt/format.h>
#include <nlohmann/json.hpp>

namespace mindquantum {
enum class GateID : uint8_t {
    I,      //
    X,      //
    Y,      //
    Z,      //
    RX,     //
    RY,     //
    RZ,     //
    Rxx,    //
    Ryy,    //
    Rzz,    //
    H,      //
    SWAP,   //
    ISWAP,  //
    T,      //
    S,      //
    Tdag,   //
    Sdag,   //
    CNOT,   //
    CZ,     //
    GP,     //
    PS,     //
    U3,     //
    FSim,   //
    M,      //
    PL,     // Pauli channel
    AD,     // amplitude damping channel
    PD,     // phase damping channel
    KRAUS,
    CUSTOM,
};

// NOLINTNEXTLINE(*avoid-c-arrays,readability-identifier-length)
NLOHMANN_JSON_SERIALIZE_ENUM(GateID, {{GateID::I, "I"},         {GateID::X, "X"},          {GateID::Y, "Y"},
                                      {GateID::Z, "Z"},         {GateID::RX, "RX"},        {GateID::RY, "RY"},
                                      {GateID::RZ, "RZ"},       {GateID::Rxx, "Rxx"},      {GateID::Ryy, "Ryy"},
                                      {GateID::Rzz, "Rzz"},     {GateID::H, "H"},          {GateID::SWAP, "SWAP"},
                                      {GateID::ISWAP, "ISWAP"}, {GateID::T, "T"},          {GateID::S, "S"},
                                      {GateID::Tdag, "Tdag"},   {GateID::Sdag, "Sdag"},    {GateID::CNOT, "CNOT"},
                                      {GateID::CZ, "CZ"},       {GateID::GP, "GP"},        {GateID::PS, "PS"},
                                      {GateID::U3, "U3"},       {GateID::FSim, "FSim"},    {GateID::M, "M"},
                                      {GateID::PL, "PL"},       {GateID::AD, "AD"},        {GateID::PD, "PD"},
                                      {GateID::KRAUS, "KRAUS"}, {GateID::CUSTOM, "CUSTOM"}});
}  // namespace mindquantum
template <typename char_t>
struct fmt::formatter<mindquantum::GateID, char_t> {
    FMT_CONSTEXPR auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {  // NOLINT(runtime/references)
        return ctx.begin();
    }

    template <typename format_context_t>
    auto format(const mindquantum::GateID& value, format_context_t& ctx) const  // NOLINT(runtime/references)
        -> decltype(ctx.out()) {
        if (value == mindquantum::GateID::I) {
            return fmt::format_to(ctx.out(), "I");
        }
        if (value == mindquantum::GateID::X) {
            return fmt::format_to(ctx.out(), "X");
        }
        if (value == mindquantum::GateID::Y) {
            return fmt::format_to(ctx.out(), "Y");
        }
        if (value == mindquantum::GateID::Z) {
            return fmt::format_to(ctx.out(), "Z");
        }
        if (value == mindquantum::GateID::RX) {
            return fmt::format_to(ctx.out(), "RX");
        }
        if (value == mindquantum::GateID::RY) {
            return fmt::format_to(ctx.out(), "RY");
        }
        if (value == mindquantum::GateID::RZ) {
            return fmt::format_to(ctx.out(), "RZ");
        }
        if (value == mindquantum::GateID::Rxx) {
            return fmt::format_to(ctx.out(), "Rxx");
        }
        if (value == mindquantum::GateID::Ryy) {
            return fmt::format_to(ctx.out(), "Ryy");
        }
        if (value == mindquantum::GateID::Rzz) {
            return fmt::format_to(ctx.out(), "Rzz");
        }
        if (value == mindquantum::GateID::H) {
            return fmt::format_to(ctx.out(), "H");
        }
        if (value == mindquantum::GateID::SWAP) {
            return fmt::format_to(ctx.out(), "SWAP");
        }
        if (value == mindquantum::GateID::ISWAP) {
            return fmt::format_to(ctx.out(), "ISWAP");
        }
        if (value == mindquantum::GateID::T) {
            return fmt::format_to(ctx.out(), "T");
        }
        if (value == mindquantum::GateID::S) {
            return fmt::format_to(ctx.out(), "S");
        }
        if (value == mindquantum::GateID::Tdag) {
            return fmt::format_to(ctx.out(), "Tdag");
        }
        if (value == mindquantum::GateID::Sdag) {
            return fmt::format_to(ctx.out(), "Sdag");
        }
        if (value == mindquantum::GateID::CNOT) {
            return fmt::format_to(ctx.out(), "CNOT");
        }
        if (value == mindquantum::GateID::CZ) {
            return fmt::format_to(ctx.out(), "CZ");
        }
        if (value == mindquantum::GateID::GP) {
            return fmt::format_to(ctx.out(), "GP");
        }
        if (value == mindquantum::GateID::PS) {
            return fmt::format_to(ctx.out(), "PS");
        }
        if (value == mindquantum::GateID::U3) {
            return fmt::format_to(ctx.out(), "U3");
        }
        if (value == mindquantum::GateID::FSim) {
            return fmt::format_to(ctx.out(), "FSim");
        }
        if (value == mindquantum::GateID::M) {
            return fmt::format_to(ctx.out(), "M");
        }
        if (value == mindquantum::GateID::PL) {
            return fmt::format_to(ctx.out(), "PL");
        }
        if (value == mindquantum::GateID::AD) {
            return fmt::format_to(ctx.out(), "AD");
        }
        if (value == mindquantum::GateID::PD) {
            return fmt::format_to(ctx.out(), "PD");
        }
        if (value == mindquantum::GateID::KRAUS) {
            return fmt::format_to(ctx.out(), "KRAUS");
        }
        if (value == mindquantum::GateID::CUSTOM) {
            return fmt::format_to(ctx.out(), "CUSTOM");
        }
        return fmt::format_to(ctx.out(), "Invalid <mindquantum::GateID>");
    }
};
#endif
