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

#ifndef CPP_GRAPH_MAPPER_HPP
#define CPP_GRAPH_MAPPER_HPP

#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Target/Device.h>
#include <tweedledum/Target/Mapping.h>
#include <tweedledum/Target/Placement.h>

#include "core/config.hpp"

#include "core/details/macros.hpp"
#include "core/details/visitor.hpp"
#include "mapping/types.hpp"

namespace mindquantum::cengines {
namespace td = tweedledum;

//! C++ class to represent an arbitrary graph mapper
/*!
 * This is intended to be instantiated in Python by users in order to
 * define the hardware architecture they want to use for mapping
 */
class CppGraphMapper {
 public:
    using device_t = tweedledum::Device;
    using circuit_t = tweedledum::Circuit;
    using placement_t = tweedledum::Placement;
    using mapping_t = tweedledum::Mapping;

    using mapping_ret_t = std::pair<circuit_t, mapping_t>;

    using mapping_param_t = std::variant<mapping::sabre_config, mapping::jit_config>;

    using edge_list_t = std::vector<std::tuple<uint32_t, uint32_t>>;

    //! Constructor with empty graph
    /*!
     * The mapping algorithm used by the mapper is defined by the type of
     * the parameters that is passed in argument. The mapper currently
     * supports three mapping algorithms:
     *   - SABRE (SWAP-based Bidirectional heuristic search algorithm)
     *   - JIT (just-in-time algorithm)
     *   - SAT (boolean satisfiability problem solver)
     *
     * \param params Parameters for the mapping algorithm
     */
    explicit CppGraphMapper(const mapping_param_t& params);

    //! Constructor with graph defined by number of qubits and edges
    /*!
     * The mapping algorithm used by the mapper is defined by the type of
     * the parameters that is passed in argument. The mapper currently
     * supports three mapping algorithms:
     *   - SABRE (SWAP-based Bidirectional heuristic search algorithm)
     *   - ZDD (zero-suppressed decision diagram)
     *   - SAT (boolean satisfiability problem solver)
     *
     * \param num_qubits Number of qubits
     * \param edge_list List of edges in the graph
     * \param params Parameters for the mapping algorithm
     */
    CppGraphMapper(uint32_t num_qubits, const edge_list_t& edge_list, const mapping_param_t& params);

    //! Simple getter for the underlying device used by this mapper
    const auto& device() const {
        return device_;
    }

    //! Apply the mapping algorithm to a network given an initial mapping
    /*!
     * This method will use the architecture defined within the instance
     * of the CppGraphMapper, as well as the mapping parameters in order
     * to choose which mapping algorithm to call.
     *
     * \param state Current mapping state
     */
    mapping_ret_t hot_start(const device_t& device, const circuit_t& circuit, placement_t& placement) const;

    //! Apply the mapping algorithm to a network without an initial mapping
    /*!
     * This method will use the architecture defined within the instance
     * of the CppGraphMapper, as well as the mapping parameters in order
     * to choose which mapping algorithm to call.
     *
     * \param state Current mapping state
     */
    mapping_ret_t cold_start(const device_t& device, const circuit_t& circuit) const;

    //! Simple getter for mapping parameters
    const auto& get_mapping_parameters() const {
        return params_;
    }

    //! Set device graph to be a 1D arrangement of qubits
    void path_device(uint32_t num_qubits, bool cyclic = false);

    //! Set device graph to be a 2D grid of qubits
    void grid_device(uint32_t num_columns, uint32_t num_rows);

 private:
    device_t device_;
    mapping_param_t params_;
};
}  // namespace mindquantum::cengines

#endif /* CPP_GRAPH_MAPPER_HPP */
