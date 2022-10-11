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

#ifndef CIRCUIT_BLOCK_HPP
#define CIRCUIT_BLOCK_HPP

#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <tweedledum/IR/Circuit.h>
#include <tweedledum/Target/Device.h>
#include <tweedledum/Target/Mapping.h>
#include <tweedledum/Target/Placement.h>

#include <fmt/format.h>

#include "experimental/core/config.hpp"
#include "experimental/core/types.hpp"

#if MQ_HAS_CONCEPTS
#    include "experimental/core/concepts.hpp"
#endif  // MQ_HAS_CONCEPTS

#ifdef UNIT_TESTS
class UnitTestAccessor;
#endif  // UNIT_TESTS

namespace mindquantum {
#if MQ_HAS_CONCEPTS
namespace concepts {
using device_t = tweedledum::Device;
using circuit_t = tweedledum::Circuit;
using placement_t = tweedledum::Placement;
using mapping_t = tweedledum::Mapping;

template <typename func_t>
concept cold_start_t = requires(func_t func, device_t device, circuit_t circuit) {
    { func(device, circuit) } -> std::same_as<std::pair<circuit_t, mapping_t>>;
};  // NOLINT(readability/braces)
template <typename func_t>
concept hot_start_t = requires(func_t func, device_t device, circuit_t circuit, placement_t placement) {
    { func(device, circuit, placement) } -> std::same_as<std::pair<circuit_t, mapping_t>>;
};  // NOLINT(readability/braces)
}  // namespace concepts
#endif  // MQ_HAS_CONCEPTS

//! External qubit ID
class QubitID {
 public:
    using cbit_t = tweedledum::Cbit;
    using qubit_t = tweedledum::Qubit;

    //! Constructor
    /*!
     * \param id External qubit ID
     */
    constexpr explicit QubitID(qubit_id_t id) : id_(id) {
    }

    //! Simple getter
    /*!
     * \return Numeric value of ID
     */
    MQ_NODISCARD constexpr qubit_id_t get() const {
        return qubit_id_t(*this);
    }

    //! Conversion operator
    /*!
     * \return Numeric value of ID
     */
    explicit constexpr operator qubit_id_t() const {
        return id_;
    }

    //! Conversion operator
    /*!
     * \return Numeric value of ID as a Tweedledum Qubit
     */
    explicit constexpr operator qubit_t() const {
        return qubit_t(id_);
    }

    //! Conversion operator
    /*!
     * \return Numeric value of ID as a Tweedledum Qubit
     */
    explicit constexpr operator cbit_t() const {
        return cbit_t(id_);
    }

    //! Less operator
    /*!
     * \return True if numerical values is less than \c other
     */
    MQ_NODISCARD constexpr bool operator<(const QubitID& other) const {
        return id_ < other.id_;
    }

    //! Less operator
    /*!
     * \return True if numerical values is less than \c other
     */
    MQ_NODISCARD constexpr bool operator<(qubit_id_t id) const {
        return id_ < id;
    }

    //! Equality operator
    /*!
     * \return True if this is equal to \c other
     */
    MQ_NODISCARD constexpr bool operator==(const QubitID& other) const {
        return id_ == other.id_;
    }

    //! Equality operator
    /*!
     * \return True if this is equal to \c other
     */
    MQ_NODISCARD constexpr bool operator==(qubit_id_t id) const {
        return id_ == id;
    }

 private:
    qubit_id_t id_;
};

bool operator<(qubit_id_t id, const QubitID& qubit);

//! A quantum circuit block
/*!
 * A circuit block is a quantum circuit (ie. a list of quantum/classical gates applied to some qubits) that may or may
 * not be applied to some physical qubits.
 *
 * In addition to being a quantum circuit, a circuit block also keep tracks of any qubit mapping that has been performed
 * and makes sure that the mapping stays consistent when executing a list of quantum circuits.
 */
class CircuitBlock {
 public:
    using cbit_t = tweedledum::Cbit;
    using qubit_t = tweedledum::Qubit;
    using inst_ref_t = tweedledum::InstRef;
    using instruction_t = tweedledum::Instruction;
    using device_t = tweedledum::Device;
    using placement_t = tweedledum::Placement;
    using mapping_t = tweedledum::Mapping;
    using circuit_t = tweedledum::Circuit;
    using ext_id_t = QubitID;
    using td_qid_t = qubit_t;
    using td_cid_t = cbit_t;
    using id_map_ext_to_int_t = std::map<ext_id_t, std::tuple<td_qid_t, std::optional<td_cid_t>>, std::less<>>;
    using id_map_qint_to_ext_t = std::map<td_qid_t, ext_id_t, std::less<>>;
    using id_map_cint_to_ext_t = std::map<td_cid_t, ext_id_t, std::less<>>;

    static constexpr struct chained_t {
    } chain_ctor{};

    //! Constructor
    CircuitBlock();

    //! Chaining constructor
    /*!
     * Create a new block that has \c parent as parent.
     *
     * \param parent Parent circuit block
     */
    CircuitBlock(const CircuitBlock& parent, chained_t);

    //! Chaining constructor
    /*!
     * Create a new block that has \c parent as parent but excludes some wires
     *
     * \param parent Parent circuit block
     * \param exclude_ids Qubit IDs from parent block to ignore when constructing the new block
     */
    CircuitBlock(const CircuitBlock& parent, const std::vector<ext_id_t>& exclude_ids, chained_t);

    // -----------------------------------------------

    //! Simple accessor for the size of the underlying circuit
    auto size() const {
        return std::size(circuit_);
    }

    //! Check whether a mapping was performed
    bool has_mapping() const;

    //! Check whether a qubit with corresponding ID is known
    /*!
     * \param qubit_id External qubit ID
     * \return True if an internal ID is associated with \c qubit_id
     */
    bool has_qubit(ext_id_t qubit_id) const;

    //! Check whether a cbit with corresponding ID is known
    /*!
     * \param qubit_id External qubit ID
     * \return True if an internal ID is associated with \c qubit_id
     */
    bool has_cbit(ext_id_t qubit_id) const;

    //! Simple accessor for the underlying External IDs
    std::vector<ext_id_t> ext_ids() const;

    //! Simple accessor for the underlying internal IDs (qubits only)
    std::vector<td_qid_t> td_ids() const;

    //! Simple accessor for the underlying internal IDs (qubits only)
    std::vector<td_qid_t> qubits() const {
        return td_ids();
    }

    // -----------------------------------------------

    //! Add a qubit to the circuit
    /*!
     * \param qubit_id (External) qubit ID to add
     * \return True if qubit could be added, false otherwise
     */
    bool add_qubit(ext_id_t qubit_id);

    // -----------------------------------------------

    //! Convert internal qubit IDs to External IDs
    /*!
     * \param ref Internal ID (Tweedledum qubit)
     * \return Corresponding External ID
     * \throw std::out_of_range if no External ID is known \c ref
     */
    qubit_id_t translate_id(const td_qid_t& ref) const;

    //! Convert internal cbit IDs to External IDs
    /*!
     * \param ref Internal ID (Tweedledum cbit)
     * \return Corresponding External ID
     * \throw std::out_of_range if no External ID is known \c ref
     */
    qubit_id_t translate_id(const td_cid_t& ref) const;

    //! Convert internal qubit IDs to External IDs (Tweedledum type as return value)
    /*!
     * \param ref Internal ID (Tweedledum wire reference)
     * \return Corresponding External ID (as Tweedledum wire reference)
     * \throw std::out_of_range if no External ID is known \c ref
     */
    td_qid_t translate_id_td(const td_qid_t& ref) const;

    //! Convert internal cbit IDs to External IDs (Tweedledum type as return value)
    /*!
     * \param ref Internal ID (Tweedledum wire reference)
     * \return Corresponding External ID (as Tweedledum wire reference)
     * \throw std::out_of_range if no External ID is known \c ref
     */
    td_cid_t translate_id_td(const td_cid_t& ref) const;

    //! Convert internal External IDs to qubit IDs
    /*!
     * \param qubit_id External qubit ID
     * \return Corresponding internal ID
     * \throw std::out_of_range if no internal ID is known for \c qubit_id
     */
    td_qid_t translate_id(const ext_id_t& qubit_id) const;

    // -----------------------------------------------

    //! Add an operation to the underlying circuit
    /*!
     * \param optor Operator to apply
     * \param control_qubits List of External qubit IDs representing controls
     * \param target_qubits List of External qubit IDs representing targets
     */
    template <typename OpT>
    inst_ref_t apply_operator(OpT&& optor, const qubit_ids_t& control_qubits, const qubit_ids_t& target_qubits);

    //! Add an operation to the underlying circuit
    /*!
     * \param optor Operator to apply
     * \param control_qubits List of External qubit IDs representing controls
     * \param target_qubits List of External qubit IDs representing targets
     */
    inst_ref_t apply_operator(const instruction_t& optor, const qubit_ids_t& control_qubits,
                              const qubit_ids_t& target_qubits);

    //! Apply a measurement on a qubit
    /*!
     * \param id External qubit ID
     */
    inst_ref_t apply_measurement(qubit_id_t id);

    // -----------------------------------------------

    //! Apply a mapping to the underlying circuit
    /*!
     * \param device Hardware device the mapping is applied to
     * \param cold_start Callable to do a "cold start" mapping (ie. first mapping)
     * \param hot_start Callable to do a "hot start" mapping (ie. subsqeuent mapping)
     *
     * \note The signature of the callables are:
     *         - void (*) (td::MapState&);
     */
#if MQ_HAS_CONCEPTS
    template <concepts::cold_start_t Fn, concepts::hot_start_t Gn>
#else
    template <typename Fn, typename Gn>
#endif  // MQ_HAS_CONCEPTS
    void apply_mapping(const device_t& device, const Fn& cold_start, const Gn& hot_start);

    // -----------------------------------------------

    //! Iterate over all operations
    template <typename Fn>
    void foreach_instruction(Fn&& fn) const;

    //! Iterate over all operations in reverse order
    template <typename Fn>
    void foreach_r_instruction(Fn&& fn) const;

    //! Apply a transform to the circuit
    /*!
     * \tparam Fn Needs to be a callable type with one of the following
     *            signatures:
     *              - void (*) (const td::Circuit&);
     *              - td::Circuit (*) (const td::Circuit&);<br>
     *
     * \note The first signature does not modify the underlying network,
     *       while the second assigns the result of the transformation
     *       to the underlying network.
     */
    template <typename Fn>
    void transform(const Fn& fn);

 private:
#ifdef UNIT_TESTS
    friend class ::UnitTestAccessor;
#endif  // UNIT_TESTS

    std::vector<qubit_t> translate_ext_ids_(const qubit_ids_t& control_qubits, const qubit_ids_t& target_qubits);
    void update_mappings_(const std::vector<qubit_t>& old_to_new);

    const device_t* device_;
    circuit_t circuit_;

    id_map_ext_to_int_t ext_to_td_;
    id_map_qint_to_ext_t qtd_to_ext_;
    id_map_cint_to_ext_t ctd_to_ext_;

    std::optional<mapping_t> mapping_;
};
}  // namespace mindquantum

#include "circuit_block.tpp"  // NOLINT(build/include)

#endif /* CIRCUIT_BLOCK_HPP */
