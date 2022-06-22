//   Copyright 2020 <Huawei Technologies Co., Ltd>
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

#include "cengines/cpp_graph_mapper.hpp"

#include <vector>

#include <tweedledum/Passes/Mapping/jit_map.h>
#include <tweedledum/Passes/Mapping/sabre_map.h>

#include "core/circuit_block.hpp"

namespace td = tweedledum;

// =============================================================================

namespace mindquantum::details {
using device_t = cengines::CppGraphMapper::device_t;
using circuit_t = cengines::CppGraphMapper::circuit_t;
using placement_t = cengines::CppGraphMapper::placement_t;

auto sabre_hot_start(const device_t& device, const circuit_t& circuit, placement_t& placement) {
    td::SabreRouter router(device, circuit, placement);
    return router.run();
}

auto sabre_cold_start(const device_t& device, const circuit_t& circuit) {
    return sabre_map(device, circuit);
}

auto jit_hot_start(const device_t& device, const circuit_t& circuit, placement_t& placement) {
    td::JitRouter router(device, circuit, placement);
    return router.run();
}

auto jit_cold_start(const device_t& device, const circuit_t& circuit) {
    return jit_map(device, circuit);
}
}  // namespace mindquantum::details

// =============================================================================

mindquantum::cengines::CppGraphMapper::CppGraphMapper(const mapping_param_t& params) : device_(0), params_(params) {
}

mindquantum::cengines::CppGraphMapper::CppGraphMapper(uint32_t num_qubits, const edge_list_t& edge_list,
                                                      const mapping_param_t& params)
    : device_(num_qubits), params_(params) {
    for (auto edge : edge_list) {
        assert(std::get<0>(edge) < num_qubits);
        assert(std::get<1>(edge) < num_qubits);
        device_.add_edge(std::get<0>(edge), std::get<1>(edge));
    }
}

void mindquantum::cengines::CppGraphMapper::path_device(uint32_t num_qubits, bool cyclic) {
    if (cyclic) {
        device_ = td::Device::ring(num_qubits);
    } else {
        device_ = td::Device::path(num_qubits);
    }
}

void mindquantum::cengines::CppGraphMapper::grid_device(uint32_t num_columns, uint32_t num_rows) {
    device_ = td::Device::grid(num_columns, num_rows);
}

// =============================================================================

auto mindquantum::cengines::CppGraphMapper::cold_start(const device_t& device, const circuit_t& circuit) const
    -> mapping_ret_t {
    return std::visit(
        overload{
            [&device, &circuit](const mapping::sabre_config&) { return details::sabre_cold_start(device, circuit); },
            [&device, &circuit](const mapping::jit_config&) { return details::jit_cold_start(device, circuit); }},
        params_);
}

auto mindquantum::cengines::CppGraphMapper::hot_start(const device_t& device, const circuit_t& circuit,
                                                      placement_t& placement) const -> mapping_ret_t {
    return std::visit(overload{[&device, &circuit, &placement](const mapping::sabre_config& params) {
                                   return details::sabre_hot_start(device, circuit, placement);
                               },
                               [&device, &circuit, &placement](const mapping::jit_config& params) {
                                   return details::jit_hot_start(device, circuit, placement);
                               }},
                      params_);
}

// =============================================================================
